"""Neural displacement field for surface-based multi-view DIC."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DeformationFieldConfig:
    hidden_dim: int = 32
    hidden_layers: int = 5
    use_positional_encoding: bool = True
    positional_encoding_frequencies: int = 6
    include_input_in_encoding: bool = True
    output_scale: float = 1.0


class PositionalEncoding(nn.Module):
    """Sinusoidal 3-D coordinate encoding."""

    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 6,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        bands = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("frequency_bands", bands)
        self.output_dim = 2 * self.input_dim * self.num_frequencies
        if self.include_input:
            self.output_dim += self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = x[..., :, None] * self.frequency_bands
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)
        encoded = encoded.flatten(start_dim=-2)
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        return encoded


class NeuralDisplacementField(nn.Module):
    """Tanh MLP mapping normalized reference coordinates to world displacement."""

    def __init__(
        self,
        config: DeformationFieldConfig | None = None,
        coord_center: torch.Tensor | None = None,
        coord_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DeformationFieldConfig()
        self.use_positional_encoding = bool(self.config.use_positional_encoding)
        if coord_center is None:
            coord_center = torch.zeros(3, dtype=torch.float32)
        if coord_scale is None:
            coord_scale = torch.ones(3, dtype=torch.float32)
        self.register_buffer("coord_center", coord_center.detach().float().reshape(1, 3))
        self.register_buffer("coord_scale", coord_scale.detach().float().reshape(1, 3).clamp_min(1e-8))

        if self.use_positional_encoding:
            self.encoder = PositionalEncoding(
                input_dim=3,
                num_frequencies=self.config.positional_encoding_frequencies,
                include_input=self.config.include_input_in_encoding,
            )
            input_dim = self.encoder.output_dim
        else:
            self.encoder = nn.Identity()
            input_dim = 3

        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(self.config.hidden_layers):
            layers.append(nn.Linear(last_dim, self.config.hidden_dim))
            layers.append(nn.Tanh())
            last_dim = self.config.hidden_dim
        layers.append(nn.Linear(last_dim, 3))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(linear_layers[-1].bias)

    def normalize(self, points_world: torch.Tensor) -> torch.Tensor:
        return (points_world - self.coord_center) / self.coord_scale

    def forward_normalized(self, points_normalized: torch.Tensor) -> torch.Tensor:
        features = self.encoder(points_normalized)
        return self.mlp(features) * float(self.config.output_scale)

    def forward(self, points_world: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize(points_world))
