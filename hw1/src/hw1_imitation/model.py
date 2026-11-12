"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


def build_mlp(input_dim: int, action_dim: int, chunk_size: int, hidden_dims: tuple[int, ...]) -> nn.Module:
    linear_relu = lambda in_dim, out_dim: nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
    return nn.Sequential(
        linear_relu(input_dim, hidden_dims[0]),
        *[linear_relu(in_dim, out_dim) for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])],
        nn.Linear(hidden_dims[-1], chunk_size * action_dim),
        Rearrange("b (c a) -> b c a", c=chunk_size, a=action_dim),
    )


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.mlp = build_mlp(state_dim, action_dim, chunk_size, hidden_dims)

    def forward(self, state: torch.Tensor):
        return self.mlp(state)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self(state)
        return F.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # mlp does (state, interpolated action chunk, t) -> predicted velocity
        input_dim = state_dim + (action_dim * chunk_size) + 1
        self.mlp = build_mlp(input_dim, action_dim, chunk_size, hidden_dims)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        bs, state_dim = state.shape
        bs, chunk_size, action_dim = action_chunk.shape

        noise = torch.randn_like(action_chunk)
        t = torch.rand((bs, 1, 1), device=state.device)

        interp = (1 - t) * noise + t * action_chunk

        net_input = torch.cat(
            [
                rearrange(state, "b s -> b s"),
                rearrange(interp, "b c a -> b (c a)", c=chunk_size, a=action_dim),
                rearrange(t, "b 1 1 -> b 1"),
            ],
            dim=-1,
        )

        pred_vel = self.mlp(net_input)
        target_vel = action_chunk - noise
        return F.mse_loss(pred_vel, target_vel)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        bs, state_dim = state.shape
        device = state.device

        net_input = lambda x_t, t: torch.cat(
            [
                rearrange(state, "b s -> b s"),
                rearrange(x_t, "b c a -> b (c a)"),
                torch.ones((bs, 1), device=device) * t,
            ],
            dim=-1,
        )
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        dt = ts[1] - ts[0]
        x = torch.randn((bs, self.chunk_size, self.action_dim), device=device)
        for t in ts[:-1]:  # 0...1-dt
            pred_action_chunk_vel = self.mlp(net_input(x, t=t))
            x = x + pred_action_chunk_vel * dt

        return x


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
