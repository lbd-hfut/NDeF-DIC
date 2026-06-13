"""
PINN-Stereo: Neural Implicit Multi-View Stereo via ZNSSD Photometric Consistency.

Per-camera depth networks D_c(u,v) → d, optimized by cross-camera ZNSSD
photometric consistency with embedded epipolar constraints.

Path B of Step 1 dense reconstruction.
Reference: docs/step1-geometric-reconstruction.md, Section 3.2
"""

import os
import time
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# =========================================================================
# Positional Encoding (self-contained, matches temp/ndef_dic/encoding.py)
# =========================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for 2D pixel coordinates.

    γ(u,v) = [sin/cos(π·u), sin/cos(2π·u), ..., sin/cos(2^(L-1)π·u),
              sin/cos(π·v), sin/cos(2π·v), ..., sin/cos(2^(L-1)π·v)]
    + original (u,v) → output_dim = 2 + 4*L
    """

    def __init__(self, n_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input

        # Geometric progression: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands * torch.pi)

        # output_dim = 2 (input) + 2 * 2 * L (sin+cos per freq per dim)
        self.output_dim = (2 if include_input else 0) + 4 * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 2) normalized pixel coords in [-1, 1].
        Returns:
            (..., output_dim) encoded features.
        """
        x_proj = x.unsqueeze(-1) * self.freq_bands     # (..., 2, L)
        sin_feat = torch.sin(x_proj)                     # (..., 2, L)
        cos_feat = torch.cos(x_proj)                     # (..., 2, L)
        encoded = torch.cat([sin_feat, cos_feat], dim=-1).flatten(-2, -1)  # (..., 4L)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)    # (..., 2+4L)
        return encoded


# =========================================================================
# Depth Network
# =========================================================================

class DepthNetwork(nn.Module):
    """
    Per-camera implicit depth function D_c(u, v) → d.

    Architecture (per design doc):
      PE(L=10) → 42 dims → MLP(256×4, skip at layer 2) → softplus → depth
    """

    def __init__(self, n_freqs: int = 10, hidden_dim: int = 256,
                 n_layers: int = 4, skip_layer: int = 2,
                 softplus_beta: float = 100.0, init_bias: float = 5.7):
        super().__init__()

        self.encoder = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        input_dim = self.encoder.output_dim  # 42
        self.skip_layer = skip_layer

        # Build MLP layers
        layers = []
        for i in range(n_layers):
            if i == 0:
                in_dim = input_dim
            elif i == skip_layer:
                in_dim = hidden_dim + input_dim  # skip connection
            else:
                in_dim = hidden_dim

            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Softplus(beta=softplus_beta))

        # Final output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.ModuleList(layers)
        self.softplus_beta = softplus_beta

        # Initialize
        self._initialize(init_bias)

    def _initialize(self, init_bias: float):
        """Xavier-normal init with tuned last-layer bias for initial depth guess."""
        # Find the last Linear layer
        last_linear_idx = None
        for i in range(len(self.layers) - 1, -1, -1):
            if isinstance(self.layers[i], nn.Linear):
                last_linear_idx = i
                break

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                if i == last_linear_idx:
                    nn.init.constant_(layer.bias, init_bias)
                else:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, uv_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_norm: (B, 2) pixel coords normalized to [-1, 1].
        Returns:
            depth: (B, 1) positive depth in camera frame.
        """
        h = self.encoder(uv_norm)  # (B, 42)

        for i, layer in enumerate(self.layers):
            if i > 0 and i // 2 == self.skip_layer and isinstance(layer, nn.Linear):
                # Re-encode for skip connection
                enc = self.encoder(uv_norm)
                h = layer(torch.cat([h, enc], dim=-1))
            else:
                h = layer(h)

        # Softplus ensures positive depth
        return F.softplus(h, beta=self.softplus_beta)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class PINNStereoConfig:
    """PINN-Stereo training and architecture parameters."""

    # Network
    n_frequencies: int = 10
    hidden_dim: int = 256
    n_layers: int = 4
    skip_layer: int = 2
    softplus_beta: float = 100.0

    # Stage 1: sparse initialization
    stage1_epochs: int = 500
    stage1_lr: float = 1e-3
    stage1_batch_size: int = 2048

    # Stage 2: multi-view ZNSSD refinement
    stage2_epochs_max: int = 5000
    stage2_lr: float = 1e-4
    stage2_patience: int = 50
    stage2_batch_size: int = 128
    stage2_batches_per_epoch: int = 200  # random batches per epoch

    # ZNSSD
    patch_radius: int = 5              # → 11×11 window
    znssd_eps: float = 1e-6

    # ROI
    roi_dilation: int = 15

    # Fusion
    depth_consistency_threshold: float = 0.05
    min_consistent_views: int = 2

    # Device
    device: str = "cuda"


# =========================================================================
# PINN-Stereo Main Class
# =========================================================================

class PINNStereo:
    """
    Multi-view neural stereo via per-camera implicit depth networks.

    Usage:
        stereo = PINNStereo(config, K_list, R_list, t_list,
                           images, sparse_points, (W, H))
        stereo.train_stage1()
        stereo.train_stage2()
        points, normals = stereo.fuse_point_cloud()
        stereo.save_depth_maps(output_dir)
    """

    def __init__(
        self,
        config: PINNStereoConfig,
        K_list: List[np.ndarray],
        R_list: List[np.ndarray],
        t_list: List[np.ndarray],
        images: List[torch.Tensor],       # [cam] → (H, W) float32 [0,1]
        sparse_points: np.ndarray,         # (N, 3) world coords
        image_dims: Tuple[int, int],       # (W, H)
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.n_cam = len(K_list)
        self.W, self.H = image_dims

        # Convert camera params to tensors (on device)
        self.K_tensors = [torch.from_numpy(K).float().to(self.device) for K in K_list]
        self.R_tensors = [torch.from_numpy(R).float().to(self.device) for R in R_list]
        self.t_tensors = [
            torch.from_numpy(t.reshape(3)).float().to(self.device) for t in t_list
        ]

        # Images stay on CPU until needed per batch
        self.images = images  # List of (H, W) float32 tensors

        # Create per-camera depth networks
        self.networks = nn.ModuleList([
            DepthNetwork(
                n_freqs=config.n_frequencies,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                skip_layer=config.skip_layer,
                softplus_beta=config.softplus_beta,
                init_bias=5.7,  # log(300) ≈ 5.7, reasonable start for WD=300
            ).to(self.device)
            for _ in range(self.n_cam)
        ])

        # Build ROI masks and sparse supervision from sparse points
        self.sparse_pts = torch.from_numpy(sparse_points).float().to(self.device)
        self.roi_masks = self._build_roi_masks()
        self.sparse_data = self._prepare_sparse_supervision()

        # Compute per-image mean/std for ZNSSD normalization
        self.img_stats = self._compute_image_stats()

        print(f"[PINN-Stereo] {self.n_cam} cameras, {len(sparse_points)} sparse pts, "
              f"device={self.device}")
        for c in range(self.n_cam):
            n_roi = self.roi_masks[c].sum().item()
            n_sp = len(self.sparse_data[c]["uv"]) if self.sparse_data[c] is not None else 0
            print(f"  Cam {c}: ROI={n_roi} px, sparse={n_sp} pts")

    # ------------------------------------------------------------------
    # ROI mask construction
    # ------------------------------------------------------------------

    def _build_roi_masks(self) -> List[torch.Tensor]:
        """Project sparse points to each camera to create binary ROI masks, then dilate."""
        masks = []

        with torch.no_grad():
            for c in range(self.n_cam):
                K = self.K_tensors[c]
                R = self.R_tensors[c]
                t = self.t_tensors[c]

                # Project sparse points
                uv, depth = self._project_points(self.sparse_pts, K, R, t)

                # Filter to image bounds
                valid = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < self.W) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < self.H) &
                    (depth > 1e-6)
                )

                # Create binary mask
                mask = torch.zeros(self.H, self.W, dtype=torch.bool, device=self.device)
                if valid.any():
                    u_int = uv[valid, 0].long().clamp(0, self.W - 1)
                    v_int = uv[valid, 1].long().clamp(0, self.H - 1)
                    mask[v_int, u_int] = True

                # Dilate mask (simple box dilation on device)
                if self.config.roi_dilation > 0:
                    mask = self._dilate_mask(mask, self.config.roi_dilation)

                masks.append(mask)

        return masks

    def _dilate_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Box-dilate a binary mask by radius pixels (on-device)."""
        if radius <= 0:
            return mask

        # Use max_pool2d for fast dilation
        kernel_size = 2 * radius + 1
        padding = radius
        # max_pool treats True > False
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        dilated = F.max_pool2d(mask_float, kernel_size=kernel_size,
                               stride=1, padding=padding)
        return dilated.squeeze() > 0.5

    # ------------------------------------------------------------------
    # Sparse supervision preparation
    # ------------------------------------------------------------------

    def _prepare_sparse_supervision(self) -> List[Optional[Dict]]:
        """
        Per camera: project sparse world points → (u_pixel, v_pixel, depth_cam).
        Returns list of dicts with 'uv' (M, 2) and 'depth' (M,) tensors, or None.
        """
        data = []

        with torch.no_grad():
            for c in range(self.n_cam):
                K = self.K_tensors[c]
                R = self.R_tensors[c]
                t = self.t_tensors[c]

                uv, depth = self._project_points(self.sparse_pts, K, R, t)

                valid = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < self.W) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < self.H) &
                    (depth > 1e-6)
                )

                if valid.sum() < 10:
                    print(f"  [WARN] Cam {c}: only {valid.sum().item()} sparse pts in FOV")
                    data.append(None)
                    continue

                data.append({
                    "uv": uv[valid].clone(),
                    "depth": depth[valid].clone(),
                })

        return data

    # ------------------------------------------------------------------
    # Projection utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _project_points(x_world: torch.Tensor, K: torch.Tensor,
                        R: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D world points to pixel coordinates.
        Convention matches renderer.py:project_points.

        Returns:
            uv: (N, 2) pixel coords (col, row)
            depth: (N,) camera-frame depth (Z)
        """
        x_cam = x_world @ R.T + t.unsqueeze(0)   # (N, 3)
        uv_h = x_cam @ K.T                        # (N, 3)
        depth = uv_h[:, 2]                        # (N,)
        uv = uv_h[:, :2] / depth.unsqueeze(-1).clamp(min=1e-8)
        return uv, depth

    @staticmethod
    def _pixel_to_world(uv: torch.Tensor, depth: torch.Tensor,
                        K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Backproject pixel + depth → 3D world point.
        uv: (B, 2) pixel coords
        depth: (B,) camera-frame depth
        Returns: (B, 3) world coords
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (uv[:, 0] - cx) / fx * depth
        y_cam = (uv[:, 1] - cy) / fy * depth
        z_cam = depth

        P_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B, 3)

        # World = R^T @ (cam - t)
        t_v = t.squeeze(-1) if t.dim() == 2 else t  # (3,)
        P_world = (R.T @ (P_cam - t_v).T).T          # (B, 3)
        return P_world

    # ------------------------------------------------------------------
    # Cross-camera warp (the embedded epipolar constraint)
    # ------------------------------------------------------------------

    def _warp_pixels(
        self,
        uv_src: torch.Tensor,       # (B, 2) source pixel coords
        depth: torch.Tensor,         # (B,) predicted depth
        cam_src: int,
        cam_dst: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warp pixels from source camera to target camera via 3D world.
        (u_dst, v_dst) automatically satisfies the epipolar constraint.

        Returns:
            uv_dst_pixel: (B, 2) destination pixel coords
            uv_dst_norm:  (B, 2) normalized [-1,1] for grid_sample
        """
        K_src = self.K_tensors[cam_src]
        R_src = self.R_tensors[cam_src]
        t_src = self.t_tensors[cam_src]
        K_dst = self.K_tensors[cam_dst]
        R_dst = self.R_tensors[cam_dst]
        t_dst = self.t_tensors[cam_dst]

        # Backproject: pixel → 3D world
        P_world = self._pixel_to_world(uv_src, depth, K_src, R_src, t_src)  # (B, 3)

        # Project: world → target pixel
        uv_dst, depth_dst = self._project_points(P_world, K_dst, R_dst, t_dst)  # (B, 2), (B,)

        # Normalize for grid_sample
        u_norm = 2.0 * uv_dst[:, 0] / (self.W - 1) - 1.0
        v_norm = 2.0 * uv_dst[:, 1] / (self.H - 1) - 1.0
        uv_dst_norm = torch.stack([u_norm, v_norm], dim=-1)

        return uv_dst, uv_dst_norm, depth_dst

    # ------------------------------------------------------------------
    # Patch-based ZNSSD
    # ------------------------------------------------------------------

    def _extract_patches_grid(
        self,
        image: torch.Tensor,           # (H, W)
        uv_centers_norm: torch.Tensor,  # (B, 2) normalized coords
    ) -> torch.Tensor:
        """
        Extract patches around uv_centers via grid_sample.

        Processes in sub-batches to limit GPU memory usage.

        Args:
            image: (H, W) grayscale float tensor.
            uv_centers_norm: (B, 2) normalized coords in [-1, 1].

        Returns:
            patches: (B, 1, P, P) where P = 2*patch_radius + 1.
        """
        r = self.config.patch_radius
        P = 2 * r + 1
        B = uv_centers_norm.shape[0]
        device = uv_centers_norm.device

        # Build offset grid in normalized coordinates (shared across batch)
        dy, dx = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=torch.float32),
            torch.arange(-r, r + 1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        dx_norm = 2.0 * dx / (self.W - 1)
        dy_norm = 2.0 * dy / (self.H - 1)
        offsets_norm = torch.stack([dx_norm, dy_norm], dim=-1)  # (P, P, 2)

        # Process in sub-batches to avoid expanding image to (B, 1, H, W)
        # which would use B * H * W * 4 bytes = ~6GB for B=1024
        max_sub_batch = 128
        all_patches = []

        for start in range(0, B, max_sub_batch):
            end = min(start + max_sub_batch, B)
            sb_uv = uv_centers_norm[start:end]  # (sb, 2)
            sb_size = end - start

            grid = sb_uv.unsqueeze(1).unsqueeze(1) + offsets_norm.unsqueeze(0)  # (sb, P, P, 2)
            img_batch = image.unsqueeze(0).unsqueeze(0).expand(sb_size, 1, self.H, self.W)
            patches = F.grid_sample(img_batch, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
            all_patches.append(patches)

        return torch.cat(all_patches, dim=0)  # (B, 1, P, P)

    def _znssd(self, patch_a: torch.Tensor, patch_b: torch.Tensor) -> torch.Tensor:
        """
        ZNSSD between two batches of patches.

        Args:
            patch_a, patch_b: (B, 1, P, P)

        Returns:
            scalar mean ZNSSD loss.
        """
        B = patch_a.shape[0]
        patch_a = patch_a.reshape(B, -1)  # (B, P²)
        patch_b = patch_b.reshape(B, -1)  # (B, P²)

        eps = self.config.znssd_eps

        # Zero-mean normalization (per-patch)
        mu_a = patch_a.mean(dim=-1, keepdim=True)
        sigma_a = patch_a.std(dim=-1, keepdim=True) + eps
        mu_b = patch_b.mean(dim=-1, keepdim=True)
        sigma_b = patch_b.std(dim=-1, keepdim=True) + eps

        norm_a = (patch_a - mu_a) / sigma_a
        norm_b = (patch_b - mu_b) / sigma_b

        znssd = ((norm_a - norm_b) ** 2).sum(dim=-1)  # (B,)
        return znssd.mean()

    def _compute_image_stats(self) -> List[Dict[str, float]]:
        """Compute per-image statistics for diagnostics."""
        stats = []
        for c in range(self.n_cam):
            img = self.images[c]
            stats.append({
                "mean": img.mean().item(),
                "std": img.std().item(),
                "min": img.min().item(),
                "max": img.max().item(),
            })
        return stats

    # ------------------------------------------------------------------
    # Stage 1: Sparse Initialization
    # ------------------------------------------------------------------

    def train_stage1(self) -> Dict:
        """
        Train each depth network to fit sparse COLMAP/GT points via MSE.

        Returns dict with 'final_loss' and 'per_camera_losses'.
        """
        print(f"\n{'='*60}")
        print(f"[PINN-Stereo] Stage 1: Sparse Initialization")
        print(f"{'='*60}")

        cfg = self.config
        per_cam_final = np.zeros(self.n_cam)

        for c in range(self.n_cam):
            if self.sparse_data[c] is None:
                print(f"  Cam {c}: skipped (no sparse data)")
                continue

            net = self.networks[c]
            uv_gt = self.sparse_data[c]["uv"]      # (M, 2)
            depth_gt = self.sparse_data[c]["depth"]  # (M,)

            # Normalize UV to [-1, 1]
            uv_norm = torch.stack([
                2.0 * uv_gt[:, 0] / (self.W - 1) - 1.0,
                2.0 * uv_gt[:, 1] / (self.H - 1) - 1.0,
            ], dim=-1)

            optimizer = torch.optim.Adam(net.parameters(), lr=cfg.stage1_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.stage1_epochs, eta_min=cfg.stage1_lr * 0.01
            )

            net.train()
            n_pts = len(uv_norm)
            batch_size = min(cfg.stage1_batch_size, n_pts)

            for epoch in range(cfg.stage1_epochs):
                # Shuffle
                perm = torch.randperm(n_pts, device=self.device)
                total_loss = 0.0
                n_batches = 0

                for i in range(0, n_pts, batch_size):
                    idx = perm[i:i + batch_size]
                    uv_batch = uv_norm[idx]
                    d_gt = depth_gt[idx].unsqueeze(-1)  # (B, 1)

                    d_pred = net(uv_batch)
                    loss = F.mse_loss(d_pred, d_gt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                if (epoch + 1) % 100 == 0:
                    avg_loss = total_loss / max(n_batches, 1)
                    print(f"  Cam {c} Epoch {epoch+1}/{cfg.stage1_epochs}: "
                          f"MSE={avg_loss:.6f}")

            per_cam_final[c] = total_loss / max(n_batches, 1)

        avg_loss = per_cam_final[per_cam_final > 0].mean()
        print(f"\n  Stage 1 complete. Mean final MSE: {avg_loss:.6f}")

        return {"final_loss": avg_loss, "per_camera_losses": per_cam_final}

    # ------------------------------------------------------------------
    # Stage 2: Multi-View ZNSSD Refinement
    # ------------------------------------------------------------------

    def train_stage2(self) -> Dict:
        """
        Refine depth networks via cross-camera ZNSSD photometric consistency.

        For each epoch, samples random batches from ROI, predicts depth,
        warps patch to adjacent cameras (c-1, c+1), and minimizes ZNSSD.

        Uses plateau-based early stopping.

        Returns dict with 'epochs_run', 'best_loss', 'converged'.
        """
        print(f"\n{'='*60}")
        print(f"[PINN-Stereo] Stage 2: Multi-View ZNSSD Refinement")
        print(f"{'='*60}")

        cfg = self.config
        device = self.device

        # Move images to GPU once for faster access
        gpu_images = [img.to(device) for img in self.images]

        # Prepare per-camera train/val split of ROI pixels
        roi_pixels = {}  # cam → {'train': (N_train, 2), 'val': (N_val, 2)}
        for c in range(self.n_cam):
            mask = self.roi_masks[c]
            rows, cols = torch.where(mask)
            if len(rows) < 100:
                roi_pixels[c] = None
                continue

            uv = torch.stack([cols.float(), rows.float()], dim=-1)  # (N, 2)
            n_total = len(uv)
            n_val = max(100, int(n_total * 0.2))
            n_train = n_total - n_val

            perm = torch.randperm(n_total, device=device)
            roi_pixels[c] = {
                "train": uv[perm[:n_train]],
                "val": uv[perm[n_train:]],
            }

        # Set up optimizers (one per network)
        optimizers = [
            torch.optim.Adam(net.parameters(), lr=cfg.stage2_lr)
            if roi_pixels[c] is not None else None
            for c, net in enumerate(self.networks)
        ]

        best_loss = float("inf")
        patience_counter = 0
        batches_per_epoch = cfg.stage2_batches_per_epoch

        for epoch in range(cfg.stage2_epochs_max):
            # ---- Training pass ----
            total_train_loss = 0.0
            n_active = 0

            for c in range(self.n_cam):
                if roi_pixels[c] is None:
                    continue

                net = self.networks[c]
                net.train()
                opt = optimizers[c]

                uv_train = roi_pixels[c]["train"]
                n_pts = len(uv_train)
                batch_size = min(cfg.stage2_batch_size, n_pts)

                cam_loss = 0.0

                for _ in range(batches_per_epoch):
                    # Random sample
                    idx = torch.randint(0, n_pts, (batch_size,), device=device)
                    uv_pixel = uv_train[idx]  # (B, 2)

                    # Normalize
                    uv_norm = torch.stack([
                        2.0 * uv_pixel[:, 0] / (self.W - 1) - 1.0,
                        2.0 * uv_pixel[:, 1] / (self.H - 1) - 1.0,
                    ], dim=-1)

                    # Predict depth
                    depth = net(uv_norm).squeeze(-1)  # (B,)

                    # ZNSSD against adjacent cameras
                    adj_cams = [(c - 1) % self.n_cam, (c + 1) % self.n_cam]
                    pair_losses = []

                    for c_dst in adj_cams:
                        if roi_pixels[c_dst] is None:
                            continue

                        # Scheme A: warp center pixel, extract patches at source and target
                        uv_center_warped, uv_center_warped_norm, depth_warped = self._warp_pixels(
                            uv_pixel, depth, c, c_dst
                        )

                        # Filter invalid warps
                        valid_warp = (
                            (uv_center_warped[:, 0] >= 0) & (uv_center_warped[:, 0] < self.W) &
                            (uv_center_warped[:, 1] >= 0) & (uv_center_warped[:, 1] < self.H) &
                            (depth_warped > 1e-6)
                        )
                        if valid_warp.sum() < 10:
                            continue

                        # Only use valid samples
                        patches_src = self._extract_patches_grid(
                            gpu_images[c], uv_norm[valid_warp]
                        )
                        patches_tgt = self._extract_patches_grid(
                            gpu_images[c_dst], uv_center_warped_norm[valid_warp]
                        )
                        pair_losses.append(self._znssd(patches_src, patches_tgt))

                    if not pair_losses:
                        continue

                    loss = sum(pair_losses) / len(pair_losses)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()

                    cam_loss += loss.item()

                avg_cam_loss = cam_loss / batches_per_epoch
                total_train_loss += avg_cam_loss
                n_active += 1

            if n_active == 0:
                print(f"  Epoch {epoch+1}: no active cameras")
                break

            avg_train = total_train_loss / n_active

            # ---- Validation pass ----
            val_loss = self._validate_stage2(roi_pixels, gpu_images)
            is_best = val_loss < best_loss

            if (epoch + 1) % 10 == 0 or is_best:
                status = "*" if is_best else " "
                print(f"  Epoch {epoch+1:4d}: train ZNSSD={avg_train:.6f}, "
                      f"val ZNSSD={val_loss:.6f}{status}")

            if is_best:
                best_loss = val_loss
                patience_counter = 0
                # Save best model state
                self._best_state = {
                    c: {k: v.clone() for k, v in net.state_dict().items()}
                    for c, net in enumerate(self.networks)
                }
            else:
                patience_counter += 1

            if patience_counter >= cfg.stage2_patience:
                print(f"\n  Early stop at epoch {epoch+1} "
                      f"(patience={cfg.stage2_patience}, best={best_loss:.6f})")
                break

        # Restore best model
        if hasattr(self, "_best_state"):
            for c, state in self._best_state.items():
                self.networks[c].load_state_dict(state)

        return {
            "epochs_run": epoch + 1,
            "best_loss": best_loss,
            "converged": patience_counter >= cfg.stage2_patience,
        }

    def _validate_stage2(self, roi_pixels: Dict, gpu_images: List[torch.Tensor] = None) -> float:
        """Compute ZNSSD on held-out validation pixels."""
        cfg = self.config
        device = self.device
        total_loss = 0.0
        n_cams = 0
        batch_size = 64  # smaller for validation

        imgs = gpu_images or [img.to(device) for img in self.images]

        with torch.no_grad():
            for c in range(self.n_cam):
                if roi_pixels[c] is None:
                    continue

                net = self.networks[c]
                net.eval()

                uv_val = roi_pixels[c]["val"]
                n_val = len(uv_val)
                n_sample = min(batch_size, n_val)
                idx = torch.randperm(n_val, device=device)[:n_sample]
                uv_batch = uv_val[idx]

                uv_norm = torch.stack([
                    2.0 * uv_batch[:, 0] / (self.W - 1) - 1.0,
                    2.0 * uv_batch[:, 1] / (self.H - 1) - 1.0,
                ], dim=-1)

                depth = net(uv_norm).squeeze(-1)

                adj_cams = [(c - 1) % self.n_cam, (c + 1) % self.n_cam]
                pair_losses = []

                for c_dst in adj_cams:
                    if roi_pixels[c_dst] is None:
                        continue

                    uv_center_w, uv_center_wn, depth_w = self._warp_pixels(uv_batch, depth, c, c_dst)
                    valid_w = (
                        (uv_center_w[:, 0] >= 0) & (uv_center_w[:, 0] < self.W) &
                        (uv_center_w[:, 1] >= 0) & (uv_center_w[:, 1] < self.H) &
                        (depth_w > 1e-6)
                    )
                    if valid_w.sum() < 5:
                        continue

                    patches_src = self._extract_patches_grid(imgs[c], uv_norm[valid_w])
                    patches_tgt = self._extract_patches_grid(imgs[c_dst], uv_center_wn[valid_w])
                    pair_losses.append(self._znssd(patches_src, patches_tgt).item())

                if pair_losses:
                    total_loss += sum(pair_losses) / len(pair_losses)
                    n_cams += 1

        return total_loss / max(n_cams, 1)

    # ------------------------------------------------------------------
    # Fusion: depth maps → 3D point cloud
    # ------------------------------------------------------------------

    def fuse_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert per-camera depth predictions to a unified 3D point cloud.

        For each camera's ROI pixels:
          1. Predict depth → backproject → 3D point
          2. Check FOV in adjacent cameras (geometric only, no network cross-check
             since per-camera networks are independently trained)
          3. Merge all valid points + let postprocess handle SOR filtering

        Returns:
            points: (N, 3) world coordinates
            normals: (N, 3) placeholder zeros (normals computed by postprocess)
        """
        print(f"\n[PINN-Stereo] Fusing depth maps → point cloud...")

        all_points = []
        for c in range(self.n_cam):
            mask = self.roi_masks[c]
            rows, cols = torch.where(mask)
            if len(rows) < 10:
                continue

            net = self.networks[c]
            net.eval()

            uv_pixel = torch.stack([cols.float(), rows.float()], dim=-1).to(self.device)

            # Process in chunks to avoid OOM
            chunk_size = 50000
            cam_points = []

            with torch.no_grad():
                for i in range(0, len(uv_pixel), chunk_size):
                    uv_chunk = uv_pixel[i:i + chunk_size]

                    uv_norm = torch.stack([
                        2.0 * uv_chunk[:, 0] / (self.W - 1) - 1.0,
                        2.0 * uv_chunk[:, 1] / (self.H - 1) - 1.0,
                    ], dim=-1)

                    depth = net(uv_norm).squeeze(-1)  # (K,)

                    # Backproject to world
                    P_world = self._pixel_to_world(
                        uv_chunk, depth, self.K_tensors[c],
                        self.R_tensors[c], self.t_tensors[c]
                    )  # (K, 3)

                    # Geometric validation: check this 3D point projects to
                    # at least one adjacent camera (FOV check only, not depth consistency)
                    check_cams = [(c + 1) % self.n_cam, (c + 2) % self.n_cam]
                    seen_by_other = torch.zeros(len(depth), dtype=torch.bool, device=self.device)

                    for c_check in check_cams:
                        uv_check, depth_check = self._project_points(
                            P_world, self.K_tensors[c_check],
                            self.R_tensors[c_check], self.t_tensors[c_check]
                        )
                        in_fov = (
                            (depth_check > 1e-6) &
                            (uv_check[:, 0] >= 0) & (uv_check[:, 0] < self.W) &
                            (uv_check[:, 1] >= 0) & (uv_check[:, 1] < self.H)
                        )
                        seen_by_other = seen_by_other | in_fov

                    cam_points.append(P_world[seen_by_other].cpu().numpy())

            if cam_points:
                all_points.append(np.concatenate(cam_points, axis=0))

            print(f"  Cam {c}: {len(uv_pixel)} ROI px → "
                  f"{sum(len(cp) for cp in (cam_points or [np.empty((0,3))]))} valid pts")

        if not all_points:
            print("  [WARN] No valid points found!")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        points = np.concatenate(all_points, axis=0).astype(np.float32)
        # Subsample if too many points (for manageable postprocess)
        if len(points) > 500000:
            idx = np.random.RandomState(42).choice(len(points), 500000, replace=False)
            points = points[idx]

        normals = np.zeros_like(points)  # Placeholder — postprocess computes these

        print(f"  Total fused: {len(points)} points (subsampled from candidates)")
        return points, normals

    # ------------------------------------------------------------------
    # Save depth maps
    # ------------------------------------------------------------------

    def save_depth_maps(self, output_dir: str):
        """Save per-camera full-resolution depth maps as .npy files."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"[PINN-Stereo] Saving depth maps to {output_dir}...")
        for c in range(self.n_cam):
            net = self.networks[c]
            net.eval()

            # Full image grid
            v_grid, u_grid = torch.meshgrid(
                torch.arange(self.H, device=self.device, dtype=torch.float32),
                torch.arange(self.W, device=self.device, dtype=torch.float32),
                indexing="ij",
            )

            # Predict in chunks
            depth_map = torch.zeros(self.H, self.W, device=self.device)
            mask = self.roi_masks[c]

            rows, cols = torch.where(mask)
            if len(rows) == 0:
                np.save(os.path.join(output_dir, f"cam_{c}_depth.npy"),
                        depth_map.cpu().numpy())
                continue

            uv_pixel = torch.stack([cols.float(), rows.float()], dim=-1)
            chunk_size = 10000

            with torch.no_grad():
                for i in range(0, len(uv_pixel), chunk_size):
                    uv_chunk = uv_pixel[i:i + chunk_size]
                    uv_norm = torch.stack([
                        2.0 * uv_chunk[:, 0] / (self.W - 1) - 1.0,
                        2.0 * uv_chunk[:, 1] / (self.H - 1) - 1.0,
                    ], dim=-1)
                    d = net(uv_norm).squeeze(-1)
                    r = uv_chunk[:, 1].long()
                    co = uv_chunk[:, 0].long()
                    depth_map[r, co] = d

            np.save(os.path.join(output_dir, f"cam_{c}_depth.npy"),
                    depth_map.cpu().numpy())

        # Save depth visualization images
        for c in range(self.n_cam):
            dmap = np.load(os.path.join(output_dir, f"cam_{c}_depth.npy"))
            if dmap.max() > 0:
                d_vis = np.clip(dmap / dmap[dmap > 0].max(), 0, 1)
                d_vis = (d_vis * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"cam_{c}_depth.png"), d_vis)

        print(f"  Saved {self.n_cam} depth maps (.npy + .png)")


# =========================================================================
# Entry point (interface-compatible with dense_mvs.run_dense_mvs)
# =========================================================================

def run_pinn_stereo(
    image_dir: str,
    sfm_path: str,
    workspace_dir: str,
    calib_dir: str,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    image_width: int,
    image_height: int,
    ref_paths: Optional[List[str]] = None,
    config: Optional[PINNStereoConfig] = None,
    clean: bool = False,
) -> Dict:
    """
    Run PINN-Stereo dense reconstruction.

    Interface-compatible with dense_mvs.run_dense_mvs().

    Args:
        image_dir:   Directory containing camera image folders (cam_0, cam_1, ...).
        sfm_path:    [unused] Sparse SfM path (for interface compatibility).
        workspace_dir: Output directory.
        calib_dir:   Calibration directory with points3D.mat.
        K_list:      Per-camera intrinsics (N_cam × 3 × 3).
        R_list:      Per-camera rotation (world→cam).
        t_list:      Per-camera translation.
        image_width:  Image width.
        image_height: Image height.
        ref_paths:   List of absolute paths to reference images.
        config:      PINNStereoConfig (uses defaults if None).
        clean:       Remove previous results.

    Returns:
        Dict with keys: status, dense_points, dense_normals, depth_map_dir, message.
    """
    import shutil
    from scipy.io import loadmat

    print(f"\n{'='*60}")
    print(f"  PINN-Stereo Dense Reconstruction")
    print(f"{'='*60}")

    if clean and os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    cfg = config or PINNStereoConfig()

    # ---- 1. Load reference images ----
    print("[PINN-Stereo] Loading reference images...")

    if ref_paths is None:
        # Auto-discover from image_dir
        cam_dirs = sorted([
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ])
        ref_paths = []
        for cd in cam_dirs:
            files = sorted([
                f for f in os.listdir(os.path.join(image_dir, cd))
                if f.lower().endswith((".bmp", ".png", ".jpg", ".tif"))
            ])
            if files:
                ref_paths.append(os.path.join(image_dir, cd, files[0]))

    images = []
    for rp in ref_paths:
        img = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {rp}")
        img_f = img.astype(np.float32) / 255.0
        images.append(torch.from_numpy(img_f))

    print(f"  Loaded {len(images)} images ({image_width}×{image_height})")

    # ---- 2. Load sparse points ----
    pts_path = os.path.join(calib_dir, "points3D.mat")
    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"Sparse points not found: {pts_path}")

    pts_data = loadmat(pts_path)
    if "points3D" in pts_data:
        sparse_pts = pts_data["points3D"]
        if sparse_pts.dtype == object:
            flat = np.array([float(sparse_pts.flat[i].item())
                           for i in range(sparse_pts.size)])
            sparse_pts = flat.reshape(sparse_pts.shape[0], 3)
        sparse_pts = sparse_pts.astype(np.float64)
    else:
        raise KeyError(f"points3D.mat missing 'points3D' field")

    print(f"  Sparse points: {sparse_pts.shape}")

    # ---- 3. Run PINN-Stereo ----
    stereo = PINNStereo(
        config=cfg,
        K_list=K_list,
        R_list=R_list,
        t_list=t_list,
        images=images,
        sparse_points=sparse_pts,
        image_dims=(image_width, image_height),
    )

    t0 = time.time()

    # Stage 1
    s1_result = stereo.train_stage1()

    # Stage 2
    s2_result = stereo.train_stage2()
    s2_result["stage1"] = s1_result

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    # ---- 4. Fuse and save ----
    dense_points, dense_normals = stereo.fuse_point_cloud()

    depth_map_dir = os.path.join(workspace_dir, "depth_maps")
    stereo.save_depth_maps(depth_map_dir)

    # Save fused point cloud
    ply_path = os.path.join(workspace_dir, "dense_points.ply")
    _save_ply(ply_path, dense_points, dense_normals)
    np.save(os.path.join(workspace_dir, "dense_normals.npy"), dense_normals)

    print(f"  Fused points saved to {ply_path}")

    return {
        "status": "ok",
        "dense_points": dense_points,
        "dense_normals": dense_normals,
        "depth_map_dir": depth_map_dir,
        "message": (f"PINN-Stereo complete: {len(dense_points)} points, "
                    f"Stage1 loss={s1_result['final_loss']:.4f}, "
                    f"Stage2 best={s2_result['best_loss']:.4f}"),
    }


# =========================================================================
# PLY I/O (matches dense_mvs.py)
# =========================================================================

def _save_ply(path: str, points: np.ndarray, normals: Optional[np.ndarray] = None):
    """Save point cloud as ASCII PLY."""
    n = len(points)
    has_n = normals is not None and len(normals) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_n:
                nx, ny, nz = normals[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
