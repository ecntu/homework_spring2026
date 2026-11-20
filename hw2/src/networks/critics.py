import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu

from einops import rearrange


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return rearrange(self.network(obs), "b 1 -> b")

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:

        obs, q_values = ptu.from_numpy(obs), ptu.from_numpy(q_values)
        q_values = rearrange(q_values, "b -> b 1")

        # compute the loss using the observations and q_values
        loss = F.mse_loss(self(obs), q_values)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"Baseline Loss": loss.item()}
