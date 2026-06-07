"""
Core neural network modules for NDeF-DIC.

- SDFNetwork: Implicit surface (signed distance function).
- IntensityField: Speckle pattern intensity on the surface.
- DeformationField: 3D displacement from reference to deformed.
- AppearanceEmbedding: Per-camera exposure correction.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from .encoding import PositionalEncoding
from .config import SDFConfig, IntensityConfig, DeformationConfig, AppearanceConfig


# ---------------------------------------------------------------------------
# Helper: build MLP with optional skip connection
# ---------------------------------------------------------------------------

def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: int,
    hidden_dim: int,
    skip_layer: int,
    activation: str,
    output_activation: str | None = None,
    softplus_beta: float = 100.0,
) -> nn.ModuleList:
    """
    Build a list of layers for an MLP with a skip connection.

    At skip_layer, the original input is concatenated to the hidden features.
    """
    layers = nn.ModuleList()

    for i in range(hidden_layers):
        if i == 0:
            in_dim = input_dim
        elif i == skip_layer:
            in_dim = hidden_dim + input_dim  # skip connection adds input
        else:
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        if activation == "softplus":
            layers.append(nn.Softplus(beta=softplus_beta))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "sine":
            layers.append(Sine())
        else:
            raise ValueError(f"Unknown activation: {activation}")

    # Final output layer
    layers.append(nn.Linear(hidden_dim, output_dim))
    if output_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif output_activation == "tanh":
        layers.append(nn.Tanh())

    return layers, skip_layer, input_dim


class Sine(nn.Module):
    """Sinusoidal activation for SIREN-style networks."""
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# ---------------------------------------------------------------------------
# SDF Network
# ---------------------------------------------------------------------------

class SDFNetwork(nn.Module):
    """
    Neural Signed Distance Function.

    Maps 3D point x -> signed distance s.
    Surface is the zero level-set: {x | S(x) = 0}.

    Uses geometric initialization so the network initially approximates
    a sphere of radius `init_radius`.
    """

    def __init__(self, config: SDFConfig):
        super().__init__()
        self.config = config

        # Positional encoding
        self.encoder = PositionalEncoding(
            n_freqs=config.n_freqs,
            include_input=True,
            log_sampling=True,
            input_dim=3,
        )
        input_dim = self.encoder.output_dim  # 3 + 2*3*L = 39 for L=6

        # MLP body
        self.layers, self.skip_layer, self._input_dim = _make_mlp(
            input_dim=input_dim,
            output_dim=1,  # scalar SDF
            hidden_layers=config.hidden_layers,
            hidden_dim=config.hidden_dim,
            skip_layer=config.skip_layer,
            activation=config.activation,
            output_activation=None,  # SDF is unbounded
            softplus_beta=config.softplus_beta,
        )

        if config.geo_init:
            self._geometric_init(config.init_radius)

    def _geometric_init(self, radius: float):
        """
        Initialize weights so that the network approximates S(x) = ||x|| - radius.

        This gives a sphere as the initial surface, which is a much better
        starting point than a random SDF for COLMAP point supervision.
        """
        # The last layer bias controls the "radius" of the zero level-set.
        # We set it so that S(0) ≈ -radius (inside) and S grows outward.
        with torch.no_grad():
            # Last layer: bias to -radius so that S(0) ≈ -radius
            last_linear = None
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    last_linear = layer
                    break
            if last_linear is not None:
                init.constant_(last_linear.bias, -radius)
                # Small weights for smooth initial SDF
                init.normal_(last_linear.weight, mean=0.0, std=1e-4)

            # Other layers: near-identity initialization for Softplus to work well
            for layer in self.layers:
                if isinstance(layer, nn.Linear) and layer is not last_linear:
                    # Xavier-like but with small scale for geometric init
                    fan_in = layer.weight.shape[1]
                    std = np.sqrt(2.0 / fan_in) * 0.5
                    init.trunc_normal_(layer.weight, mean=0.0, std=std, a=-2*std, b=2*std)
                    init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) 3D points in world coordinates.

        Returns:
            s: (N, 1) signed distance values.
        """
        h = self.encoder(x)  # (N, input_dim)

        for i, layer in enumerate(self.layers):
            if i > 0 and isinstance(self.layers[i - 1], nn.ReLU) and i // 2 == self.skip_layer:
                # After activation at skip layer, concat original encoding
                h = layer(torch.cat([h, self.encoder(x)], dim=-1))
            elif i > 0 and isinstance(self.layers[i - 1], (nn.Softplus, nn.ReLU, Sine)) and i // 2 == self.skip_layer:
                h = layer(torch.cat([h, self.encoder(x)], dim=-1))
            else:
                h = layer(h)

        return h  # (N, 1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradient of the SDF at x.

        Args:
            x: (N, 3) points. Requires grad or will be set.

        Returns:
            grad: (N, 3) gradient vectors ∇S(x).
        """
        x.requires_grad_(True)
        s = self.forward(x)
        grad = torch.autograd.grad(
            outputs=s,
            inputs=x,
            grad_outputs=torch.ones_like(s),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad


# ---------------------------------------------------------------------------
# Intensity Field
# ---------------------------------------------------------------------------

class IntensityField(nn.Module):
    """
    Grayscale intensity (speckle pattern) defined on the 3D surface.

    Maps 3D surface point x -> grayscale value I(x) ∈ [0, 1].

    Under the Lambertian + brightness constancy assumptions, I(x) is
    the intrinsic reflectance of the material point — invariant to
    deformation and viewing direction.
    """

    def __init__(self, config: IntensityConfig):
        super().__init__()
        self.config = config

        # Positional encoding (higher L for speckle texture detail)
        self.encoder = PositionalEncoding(
            n_freqs=config.n_freqs,
            include_input=True,
            log_sampling=True,
            input_dim=3,
        )
        input_dim = self.encoder.output_dim  # 63 for L=10

        # MLP body
        layers, skip_layer, _input_dim = _make_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_layers=config.hidden_layers,
            hidden_dim=config.hidden_dim,
            skip_layer=config.skip_layer,
            activation=config.activation,
            output_activation="sigmoid",  # grayscale ∈ [0, 1]
        )
        self.layers = layers
        self.skip_layer = skip_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) surface points in world coordinates.

        Returns:
            gray: (N, 1) grayscale intensity ∈ [0, 1].
        """
        h = self.encoder(x)

        for i, layer in enumerate(self.layers):
            if i > 0 and isinstance(self.layers[i - 1], (nn.ReLU, nn.Sigmoid)) and i // 2 == self.skip_layer:
                h = layer(torch.cat([h, self.encoder(x)], dim=-1))
            else:
                h = layer(h)

        return h  # (N, 1) in [0, 1]


# ---------------------------------------------------------------------------
# Deformation Field
# ---------------------------------------------------------------------------

class DeformationField(nn.Module):
    """
    3D deformation field from reference to deformed configuration.

    Maps surface point x and load step t -> displacement (u, v, w) in
    Cartesian world coordinates.

    Φ(x, t=0) is driven to (0,0,0) by the photometric loss on reference frames.
    """

    def __init__(self, config: DeformationConfig):
        super().__init__()
        self.config = config

        # Spatial encoding
        self.encoder_space = PositionalEncoding(
            n_freqs=config.n_freqs_space,
            include_input=True,
            log_sampling=True,
            input_dim=3,
        )  # 3 + 2*3*8 = 51 dims

        # Temporal encoding
        self.encoder_time = PositionalEncoding(
            n_freqs=config.n_freqs_time,
            include_input=True,
            log_sampling=True,
            input_dim=1,
        )  # 1 + 2*1*4 = 9 dims

        input_dim = self.encoder_space.output_dim + self.encoder_time.output_dim  # 60

        # MLP body
        layers, skip_layer, _input_dim = _make_mlp(
            input_dim=input_dim,
            output_dim=3,  # (u, v, w)
            hidden_layers=config.hidden_layers,
            hidden_dim=config.hidden_dim,
            skip_layer=config.skip_layer,
            activation=config.activation,
            output_activation=None,  # displacements are unbounded
        )
        self.layers = layers
        self.skip_layer = skip_layer

        # Initialize last layer with small weights (deformations start near zero)
        self._init_last_layer()

    def _init_last_layer(self):
        """Initialize the output layer to produce near-zero displacements."""
        with torch.no_grad():
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    init.normal_(layer.weight, mean=0.0, std=1e-4)
                    init.constant_(layer.bias, 0.0)
                    break

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) reference surface points in world coordinates.
            t: (N, 1) or scalar load step ∈ [0, 1].

        Returns:
            uvw: (N, 3) displacement vectors in world units.
        """
        # Encode space and time separately
        feat_space = self.encoder_space(x)      # (N, 51)
        feat_time = self.encoder_time(t)        # (N, 9)
        h = torch.cat([feat_space, feat_time], dim=-1)  # (N, 60)

        for i, layer in enumerate(self.layers):
            if i > 0 and isinstance(self.layers[i - 1], (nn.ReLU,)) and i // 2 == self.skip_layer:
                h = layer(torch.cat([h, torch.cat([feat_space, feat_time], dim=-1)], dim=-1))
            else:
                h = layer(h)

        return h  # (N, 3)


# ---------------------------------------------------------------------------
# Appearance Embedding
# ---------------------------------------------------------------------------

class AppearanceEmbedding(nn.Module):
    """
    Per-camera appearance embedding for exposure correction.

    Each camera is assigned a learnable low-dimensional vector ℓ_c.
    This is mapped to an affine transform (scale, bias) applied to
    the rendered intensity.

    All frames from the same camera share the same embedding,
    ensuring ℓ_c captures only static camera properties.

    Following NeRF-W but simplified to affine correction only.
    """

    def __init__(self, config: AppearanceConfig, n_cameras: int):
        super().__init__()
        self.config = config
        self.n_cameras = n_cameras

        # Embedding lookup table: one vector per camera
        self.embedding = nn.Embedding(n_cameras, config.embedding_dim)
        init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        # Small MLP to map embedding -> (scale, bias)
        self.head = nn.Sequential(
            nn.Linear(config.embedding_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),  # scale, bias
        )

        # Initialize head to produce near-identity mapping
        with torch.no_grad():
            init.constant_(self.head[-1].bias, 0.0)  # bias ≈ 0
            init.constant_(self.head[-1].weight, 0.0)  # scale ≈ 0 → a ≈ 1 after tanh
            # The scale is 1 + a after tanh, so a≈0 means scale≈1 (identity)

    def forward(self, camera_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            camera_ids: (N,) int tensor, camera indices [0, n_cameras-1].

        Returns:
            scale: (N, 1) multiplicative correction.
            bias:  (N, 1) additive correction.
        """
        ℓ = self.embedding(camera_ids)       # (N, embedding_dim)
        params = self.head(ℓ)                # (N, 2)

        # Constrain scale to a reasonable range
        s_min, s_max = self.config.scale_range
        b_min, b_max = self.config.bias_range

        scale = s_min + (s_max - s_min) * torch.sigmoid(params[:, 0:1])
        bias = b_min + (b_max - b_min) * torch.tanh(params[:, 1:2])

        return scale, bias

    def correct(self, intensity: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply appearance correction to rendered intensity.

        Args:
            intensity: (N, 1) base rendered intensity.
            camera_ids: (N,) camera indices.

        Returns:
            corrected: (N, 1) appearance-corrected intensity.
        """
        scale, bias = self.forward(camera_ids)
        return scale * intensity + bias

    def regularization(self) -> torch.Tensor:
        """L2 regularization on embeddings to prevent drift."""
        return self.embedding.weight.pow(2).mean()
