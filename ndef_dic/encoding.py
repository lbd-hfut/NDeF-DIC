"""
Positional encoding for neural fields.

Implements the NeRF-style sinusoidal encoding:
  γ(x) = [sin(2^0·π·x), cos(2^0·π·x), ..., sin(2^(L-1)·π·x), cos(2^(L-1)·π·x)]

This allows MLPs to represent high-frequency functions despite their spectral bias.
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Maps input coordinates to higher-dimensional frequency features.

    Args:
        n_freqs: Number of frequency bands (L).
        include_input: Whether to include the original input in the output.
        log_sampling: If True, frequencies grow geometrically (2^k).
                      If False, linearly spaced frequencies.
        input_dim: Dimensionality of input coordinates.
    """

    def __init__(
        self,
        n_freqs: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
        input_dim: int = 3,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        self.input_dim = input_dim

        if log_sampling:
            # Geometric progression: 2^0, 2^1, ..., 2^(L-1)
            freq_bands = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        else:
            # Linear spacing: 1, 2, ..., L
            freq_bands = torch.arange(1, n_freqs + 1, dtype=torch.float32)

        # Register as buffer so they move to GPU with the module
        self.register_buffer("freq_bands", freq_bands * torch.pi)

        # Output dimensionality
        self.output_dim = (input_dim if include_input else 0) + 2 * input_dim * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) input coordinates.

        Returns:
            (..., output_dim) encoded features.
        """
        # x: (..., D), freq_bands: (L,)
        # x_proj: (..., D, L)
        x_proj = x.unsqueeze(-1) * self.freq_bands  # (..., D, L)

        # Encode: concatenate sin and cos
        sin_feat = torch.sin(x_proj)  # (..., D, L)
        cos_feat = torch.cos(x_proj)  # (..., D, L)

        # Interleave sin/cos per frequency band and flatten:
        # [sin_1, cos_1, sin_2, cos_2, ...] for each input dim
        encoded = torch.cat([sin_feat, cos_feat], dim=-1)  # (..., D, 2L)
        encoded = encoded.flatten(-2, -1)  # (..., 2*D*L)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, n_freqs={self.n_freqs}, "
            f"include_input={self.include_input}, output_dim={self.output_dim}"
        )
