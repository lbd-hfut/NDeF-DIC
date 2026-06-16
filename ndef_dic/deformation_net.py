"""
Step 3: Neural Deformation Field Network Φ(x, t): ℝ⁴ → ℝ³.

Components:
  - HashGridEncoder:   Multi-resolution hash grid (Instant-NGP style), pure PyTorch.
  - PositionalEncoding: NeRF-style sinusoidal encoding (for temporal PE strategy).
  - TemporalEncoder:   Adaptive time encoding (binary / PE / PE+smooth).
  - DeformationNetwork: Full Φ network: encode → MLP → tanh gate → displacement.

Reference: docs/step3-neural-deformation-field.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional


# =========================================================================
# HashGridEncoder
# =========================================================================

class HashGridEncoder(nn.Module):
    """Multi-resolution hash grid encoding (Instant-NGP / Muller et al.).

    Encodes 3D coordinates via L levels of progressively finer grids,
    each with its own learned hash table and trilinear interpolation.

    Args:
        n_levels:             Number of resolution levels (L=16).
        n_features_per_level: Features per level (F=2).
        base_resolution:      Coarsest grid resolution (16).
        finest_resolution:    Finest grid resolution (512).
        hash_table_size:      Entries per hash table (2^19).
        n_input_dims:         Input dimensionality (3).

    Output: (N, n_levels * n_features_per_level) = (N, 32).

    Memory: 16 × 2^19 × 2 × 4 bytes ≈ 67 MB for float32.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hash_table_size: int = 2**19,
        n_input_dims: int = 3,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.hash_table_size = hash_table_size
        self.n_input_dims = n_input_dims

        # Resolution growth factor: b^15 = 512/16 = 32 → b = 32^(1/15)
        growth = (finest_resolution / base_resolution) ** (1.0 / max(n_levels - 1, 1))

        # Hash tables: (n_levels, hash_table_size, n_features_per_level)
        # Wrapped in nn.Parameter for optimizer tracking
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.empty(hash_table_size, n_features_per_level)
            )
            for _ in range(n_levels)
        ])

        # Resolution per level
        self.register_buffer(
            "resolutions",
            torch.tensor([
                int(base_resolution * (growth ** i)) for i in range(n_levels)
            ], dtype=torch.float32),
        )

        # Prime numbers for spatial hashing (per dimension)
        self.register_buffer(
            "hash_primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.int64),
        )

        self.output_dim = n_levels * n_features_per_level

        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize hash table entries with uniform small random values."""
        for table in self.hash_tables:
            nn.init.uniform_(table, -1e-4, 1e-4)

    @staticmethod
    def _hash_integer(
        coords: torch.Tensor,    # (..., 3) int
        primes: torch.Tensor,    # (3,)
        table_size: int,
    ) -> torch.Tensor:
        """Spatial hash of integer corner coordinates via XOR of primes.

        Args:
            coords: (..., 3) integer grid coordinates.
            primes: (3,) per-dimension prime multipliers.
            table_size: Hash table capacity.

        Returns:
            (...,) hash indices in [0, table_size-1].
        """
        # XOR accumulation: (x*P0) ^ (y*P1) ^ (z*P2) mod T
        h = (coords[..., 0].to(torch.int64) * primes[0]) ^ \
            (coords[..., 1].to(torch.int64) * primes[1]) ^ \
            (coords[..., 2].to(torch.int64) * primes[2])
        return (h % table_size).to(torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 3D coordinates into hash-grid features.

        Args:
            x: (N, 3) coordinates in [-1, 1]^3.

        Returns:
            features: (N, n_levels * n_features_per_level) = (N, 32).
        """
        N = x.shape[0]
        device = x.device
        all_features = []

        for level in range(self.n_levels):
            resolution = self.resolutions[level].item()
            table = self.hash_tables[level]

            # Scale: [-1, 1] → [0, resolution]
            x_scaled = (x + 1.0) / 2.0 * resolution  # (N, 3)

            # Floor corners
            x0 = torch.floor(x_scaled).to(torch.int64)         # (N, 3)
            x1 = x0 + 1                                          # (N, 3)
            x0 = torch.clamp(x0, 0, int(resolution) - 1)
            x1 = torch.clamp(x1, 0, int(resolution) - 1)

            # Trilinear interpolation weights
            frac = x_scaled - x0.float()  # (N, 3) in [0, 1)
            w0 = (1.0 - frac).unsqueeze(-1)  # (N, 3, 1)
            w1 = frac.unsqueeze(-1)          # (N, 3, 1)

            # 8 corners of voxel: (N, 8, 3)
            corners_xyz = [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            ]
            corner_coords = []
            for cx, cy, cz in corners_xyz:
                xi = x1 if cx else x0
                yi = x1 if cy else x0
                zi = x1 if cz else x0
                corner_coords.append(torch.stack([
                    xi[:, 0], yi[:, 1], zi[:, 2]
                ], dim=-1))  # (N, 3)

            corner_coords = torch.stack(corner_coords, dim=1)  # (N, 8, 3)

            # Hash each corner → hash table lookup
            hash_idx = self._hash_integer(
                corner_coords, self.hash_primes, self.hash_table_size
            )  # (N, 8)

            corner_features = table[hash_idx]  # (N, 8, F)

            # Trilinear interpolation weights for each corner
            corner_weights = []
            for cx, cy, cz in corners_xyz:
                wx = w1[:, 0] if cx else w0[:, 0]  # (N, 1)
                wy = w1[:, 1] if cy else w0[:, 1]  # (N, 1)
                wz = w1[:, 2] if cz else w0[:, 2]  # (N, 1)
                corner_weights.append(wx * wy * wz)  # (N, 1)

            corner_weights = torch.stack(corner_weights, dim=1)  # (N, 8, 1)

            # Weighted sum: Σ corner_weight * corner_feature
            level_features = (corner_weights * corner_features).sum(dim=1)  # (N, F)
            all_features.append(level_features)

        return torch.cat(all_features, dim=-1)  # (N, L*F)


# =========================================================================
# PositionalEncoding (reused from temp/ndef_dic/encoding.py)
# =========================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (NeRF-style).

    γ(x) = [sin(π·f_0·x), cos(π·f_0·x), ..., sin(π·f_{L-1}·x), cos(π·f_{L-1}·x)]

    Args:
        n_freqs:        Number of frequency bands L.
        include_input:  If True, prepend original input to output.
        log_sampling:   If True, frequencies grow geometrically (2^k).
        input_dim:      Dimensionality of input.
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
            freq_bands = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        else:
            freq_bands = torch.arange(1, n_freqs + 1, dtype=torch.float32)

        self.register_buffer("freq_bands", freq_bands * math.pi)

        self.output_dim = (input_dim if include_input else 0) + 2 * input_dim * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x.unsqueeze(-1) * self.freq_bands  # (..., D, L)
        sin_feat = torch.sin(x_proj)
        cos_feat = torch.cos(x_proj)
        encoded = torch.cat([sin_feat, cos_feat], dim=-1).flatten(-2, -1)
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        return encoded


# =========================================================================
# TemporalEncoder
# =========================================================================

class TemporalEncoder(nn.Module):
    """Adaptive temporal encoding for Φ(x,t).

    Strategies (auto-selected or explicit):
      - "binary":     1 dim, indicator 1_{t>0}.    For <=2 load steps.
      - "pe":         PE(L=6), no input → 12 dims. For 3-50 steps.
      - "pe_smooth":  PE(L=8), no input → 16 dims. For >50 steps.

    Args:
        n_freqs:      Frequency bands for PE strategies.
        strategy:     "binary" | "pe" | "pe_smooth".
        include_input: Include raw t in PE output (default False → 12 dims).
    """

    def __init__(
        self,
        n_freqs: int = 6,
        strategy: str = "binary",
        include_input: bool = False,
    ):
        super().__init__()
        self.strategy = strategy

        if strategy == "binary":
            self.pe = None
            self.output_dim = 1
        elif strategy in ("pe", "pe_smooth"):
            nf = n_freqs if strategy == "pe" else (n_freqs + 2)  # pe_smooth: L=8
            self.pe = PositionalEncoding(
                n_freqs=nf, include_input=include_input,
                log_sampling=True, input_dim=1,
            )
            self.output_dim = self.pe.output_dim
        else:
            raise ValueError(f"Unknown temporal encoding strategy: {strategy}")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode normalized time t ∈ [0, 1].

        Args:
            t: (N, 1) or (N,) or scalar time.

        Returns:
            features: (N, output_dim).
        """
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        if self.strategy == "binary":
            return (t > 0.0).float()  # (N, 1)

        return self.pe(t)  # (N, output_dim)


# =========================================================================
# DeformationNetwork
# =========================================================================

class DeformationNetwork(nn.Module):
    """Neural Deformation Field Φ(x, t): ℝ⁴ → ℝ³.

    Architecture:
      Spatial:  HashGridEncoder(L=16, F=2) → 32 dims  OR
                PositionalEncoding(L=10) → 63 dims (dense, GPU-friendly)
      Temporal: TemporalEncoder(strategy)   → 1 or 12 dims
      Concat:   spatial_dim + time_dims = input_dim
      MLP:      input_dim → 256 → 256 → (skip+input_dim) → 256 → 256 → 256 → 3
      Gate:     Φ = tanh(α·t) · Φ_raw

    **Hard constraint**: Φ(x, 0) = (0,0,0) regardless of MLP weights,
    since tanh(α·0) = 0.

    Args:
        hash_grid_config: kwargs for HashGridEncoder (ignored if spatial_encoding='frequency').
        temporal_config:  kwargs for TemporalEncoder.
        spatial_encoding: "hash_grid" | "frequency" — dense PE avoids GPU hash lookup overhead.
        pe_n_freqs:       Frequency bands for positional encoding (default 10).
        hidden_dim:       MLP hidden dimension (256).
        alpha:            Steepness of tanh gate (5.0).
        learnable_alpha:  If True, alpha is a learnable parameter.
    """

    def __init__(
        self,
        hash_grid_config: Optional[dict] = None,
        temporal_config: Optional[dict] = None,
        spatial_encoding: str = "hash_grid",
        pe_n_freqs: int = 10,
        hidden_dim: int = 256,
        alpha: float = 5.0,
        learnable_alpha: bool = False,
    ):
        super().__init__()

        self.spatial_encoding = spatial_encoding

        # Encoders
        if spatial_encoding == "frequency":
            self.hash_encoder = PositionalEncoding(
                n_freqs=pe_n_freqs, include_input=True,
                log_sampling=True, input_dim=3,
            )  # 3 + 2*3*10 = 63 dims, all dense ops
        else:
            self.hash_encoder = HashGridEncoder(**(hash_grid_config or {}))

        self.temporal_encoder = TemporalEncoder(**(temporal_config or {}))

        spatial_dim = self.hash_encoder.output_dim   # 32 (hash) or 63 (PE)
        temporal_dim = self.temporal_encoder.output_dim  # 1 or 12
        input_dim = spatial_dim + temporal_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # MLP layers (explicit for clarity)
        self.lin0 = nn.Linear(input_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim + input_dim, hidden_dim)  # skip
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, 3)  # output: (u, v, w)

        # Tanh gate
        if learnable_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha_param", torch.tensor(alpha, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights — Xavier for tanh, output layer near-zero."""
        for layer in [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Output layer: near-zero → Φ ≈ 0 at initialization
        nn.init.normal_(self.lin5.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.lin5.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, 3) surface points in world coordinates.
            t: (N, 1) or scalar normalized time in [0, 1].

        Returns:
            phi: (N, 3) displacement vectors (u, v, w) in world units.
                 Guaranteed zero at t=0 by tanh gate.
        """
        # Normalize t
        if t.dim() == 0:
            t = t.view(1, 1).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1).expand(x.shape[0], 1)
        elif t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0], 1)

        # Encode
        feat_space = self.hash_encoder(x)         # (N, 32)
        feat_time = self.temporal_encoder(t)      # (N, T)
        feat = torch.cat([feat_space, feat_time], dim=-1)  # (N, input_dim)

        # MLP with skip at layer 2 (tanh activation for smooth deformation)
        h = torch.tanh(self.lin0(feat))
        h = torch.tanh(self.lin1(h))
        h = torch.tanh(self.lin2(torch.cat([h, feat], dim=-1)))
        h = torch.tanh(self.lin3(h))
        h = torch.tanh(self.lin4(h))
        phi_raw = self.lin5(h)  # (N, 3)

        # Tanh gate: hard constraint Φ=0 at t=0
        gate = torch.tanh(self.alpha_param * t)  # (N, 1)
        phi = gate * phi_raw                      # (N, 3)

        return phi
