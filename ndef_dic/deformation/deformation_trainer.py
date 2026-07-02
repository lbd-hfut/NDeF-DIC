"""
Step 3: DeformationFieldTrainer — Φ 网络训练编排器。

Orchestrates the full training loop:
  SurfaceProvider → sample surface points
  DeformationNetwork → compute Φ(x,t)
  MultiCamDataset → extract image patches
  dic_losses → ZNSSD + smoothness regularization

Reference: docs/step2-step3-interface.md (Section 3), docs/step3-neural-deformation-field.md
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..step2_surface_provider import SurfaceProvider, normalize_points, unnormalize_points
from .deformation_net import DeformationNetwork
from .dataset import MultiCamDataset
from .dic_losses import znssd, deformation_smoothness_loss


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class PhaseConfig:
    """Single training phase hyperparameters."""
    patch_size: int = 32
    lambda_smooth: float = 1e-2
    iterations: int = 2000
    lr: float = 1e-3
    log_interval: int = 100
    validate_interval: int = 500


@dataclass
class DeformationTrainerConfig:
    """Full training configuration.

    Phases follow coarse-to-fine schedule:
      Phase 1: patch=32, λ=1e-2 (coarse matching)
      Phase 2: patch=16, λ=1e-3 (refinement)
      Phase 3: patch=8,  λ=1e-4 (fine detail)
    """
    # Sampling
    batch_size: int = 1024
    cameras_per_point: int = 3

    # Phases (coarse-to-fine)
    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(patch_size=32, lambda_smooth=1e-2, iterations=2000, lr=1e-3),
        PhaseConfig(patch_size=16, lambda_smooth=1e-3, iterations=5000, lr=5e-4),
        PhaseConfig(patch_size=8,  lambda_smooth=1e-4, iterations=3000, lr=1e-4),
    ])

    # Load step strategy
    load_step_strategy: str = "joint"  # "joint" | "sequential" | "curriculum"

    # Optimization
    grad_clip_norm: float = 1.0

    # Device
    device: str = "cuda"


# =========================================================================
# Trainer
# =========================================================================

class DeformationFieldTrainer:
    """Step 3 training orchestrator.

    Wires together SurfaceProvider, DeformationNetwork, and MultiCamDataset
    and runs coarse-to-fine multi-view ZNSSD training.

    Usage:
        surface = create_surface_provider(data_dir, calib_dir, method="point_cloud")
        dataset = MultiCamDataset(data_dir, ...)
        deform_net = DeformationNetwork()
        trainer = DeformationFieldTrainer(surface, deform_net, dataset, config)
        trainer.train()
    """

    def __init__(
        self,
        surface: SurfaceProvider,
        deformation_net: DeformationNetwork,
        dataset: MultiCamDataset,
        config: Optional[DeformationTrainerConfig] = None,
    ):
        self.surface = surface
        self.deformation_net = deformation_net
        self.dataset = dataset
        self.config = config or DeformationTrainerConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        # Move network to device
        self.deformation_net.to(self.device)

        # Optimizer (created per-phase)
        self.optimizer: Optional[torch.optim.Adam] = None

        # State tracking
        self.current_phase = 0
        self.iteration = 0
        self.loss_history: List[Dict] = []

        # Cached images (moved to GPU in _cache_images)
        self._ref_images_gpu: List[torch.Tensor] = []
        self._def_images_gpu: Dict[int, List[torch.Tensor]] = {}

    # =================================================================
    # Image caching
    # =================================================================

    def _cache_images(self):
        """Move all reference and deformed images to GPU once."""
        print("[Trainer] Caching images to GPU...")
        t0 = time.time()

        self._ref_images_gpu = [
            img.to(self.device) for img in self.dataset.ref_images
        ]

        for step in range(1, self.dataset.n_steps + 1):
            self._def_images_gpu[step] = [
                img.to(self.device) if img is not None else None
                for img in self.dataset.def_images[step]
            ]

        print(f"  Cached {len(self._ref_images_gpu)} ref + "
              f"{self.dataset.n_steps} def steps in {time.time() - t0:.1f}s")

    # =================================================================
    # Main training loop
    # =================================================================

    def train(self):
        """Run all training phases."""
        self._cache_images()

        total_start = time.time()

        for phase_idx, phase in enumerate(self.config.phases):
            self.current_phase = phase_idx
            print(f"\n{'='*60}")
            print(f"[Trainer] Phase {phase_idx + 1}/{len(self.config.phases)}: "
                  f"patch={phase.patch_size}, λ={phase.lambda_smooth}, "
                  f"lr={phase.lr}, iters={phase.iterations}")
            print(f"{'='*60}")

            self._train_phase(phase)

        elapsed = time.time() - total_start
        print(f"\n[Trainer] Training complete in {elapsed:.0f}s "
              f"({elapsed/60:.1f} min)")

    def _train_phase(self, phase: PhaseConfig):
        """Train one phase with fixed hyperparameters."""
        # Fresh optimizer per phase
        self.optimizer = torch.optim.Adam(
            self.deformation_net.parameters(), lr=phase.lr
        )

        M = self.config.batch_size
        K = self.config.cameras_per_point
        n_steps = self.dataset.n_steps

        best_val_loss = float("inf")

        for it in range(phase.iterations):
            self.iteration += 1

            # Sample time step
            if n_steps == 1:
                t_step = 1
                t_val = 1.0
            else:
                t_step = np.random.randint(1, n_steps + 1)
                t_val = float(t_step) / n_steps

            # ---- Training step ----
            metrics = self._train_step(
                t_val=t_val, t_step=t_step,
                patch_size=phase.patch_size,
                lambda_smooth=phase.lambda_smooth,
                M=M, K=K,
            )

            # ---- Log ----
            if (it + 1) % phase.log_interval == 0:
                msg = (
                    f"  Iter {it + 1:5d}/{phase.iterations} | "
                    f"L_dic={metrics['L_dic']:.4f} | "
                    f"L_smooth={metrics['L_smooth']:.4f} | "
                    f"L_total={metrics['L_total']:.4f} | "
                    f"pairs={metrics['valid_pairs']} | "
                    f"|grad|={metrics.get('grad_norm', 0):.3f}"
                )
                print(msg)

            # ---- Validate ----
            if (it + 1) % phase.validate_interval == 0:
                val_loss = self._evaluate(t_val, t_step, phase.patch_size, M, K)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(f"  --- VAL: ZNSSD={val_loss:.4f} (best={best_val_loss:.4f})")

            self.loss_history.append(metrics)

    # =================================================================
    # Single training step (core algorithm)
    # =================================================================

    def _train_step(
        self, t_val: float, t_step: int,
        patch_size: int, lambda_smooth: float,
        M: int, K: int,
    ) -> Dict:
        """Single training iteration — GPU-optimized.

        Key optimizations vs naive implementation:
          1. Batch projection: all cameras in ONE matmul (not N separate ones)
          2. Zero .item() calls inside per-camera loop (keeps GPU pipeline full)
          3. Active camera pre-filter: only iterate cameras with ≥5 visible pts
          4. valid_count stays on GPU as tensor until final division
        """
        # ---- 1. Sample surface points ----
        x, normals = self.surface.sample_surface_points(M, strategy="uniform")

        # Normalize for network input
        x_norm = normalize_points(x, self.surface.bbox)

        # ---- 2. Get visible cameras ----
        cam_ids = self.surface.get_visible_cameras(x, max_cams=K)

        # ---- 3. Compute deformation ----
        t_tensor = torch.full((M, 1), t_val, device=self.device, dtype=torch.float32)
        phi = self.deformation_net(x_norm, t_tensor)
        x_def_norm = x_norm + phi
        x_def = unnormalize_points(x_def_norm, self.surface.bbox)

        # ---- 4. Per-camera ZNSSD (GPU-optimized) ----
        n_cams = self.surface.num_cameras
        W, H = self.dataset.W, self.dataset.H

        # 4a. Batch project ALL cameras at once — 1 big matmul instead of N small ones
        uv_all_ref = self.surface.batch_project_all_cameras(x)      # (M, n_cams, 2)
        uv_all_def = self.surface.batch_project_all_cameras(x_def)  # (M, n_cams, 2)

        # 4b. Pre-filter: which cameras have ≥5 visible points? (1 sync, not N)
        cam_counts = torch.zeros(n_cams, dtype=torch.int32, device=self.device)
        valid_cam_mask = cam_ids >= 0
        cam_flat = cam_ids[valid_cam_mask]
        cam_counts.scatter_add_(0, cam_flat, torch.ones_like(cam_flat, dtype=torch.int32))
        active_cams = torch.where(cam_counts >= 5)[0].tolist()  # single GPU→CPU

        # 4c. Loop only over active cameras — no .item() calls inside
        total_znssd = torch.tensor(0.0, device=self.device)
        valid_count = torch.tensor(0.0, device=self.device)  # stay on GPU

        for cam_id in active_cams:
            # Points assigned to this camera
            cam_mask = (cam_ids == cam_id).any(dim=-1)  # (M,) bool

            # Get pre-computed projections for this camera
            uv_ref = uv_all_ref[cam_mask, cam_id]  # (n_vis, 2)
            uv_def = uv_all_def[cam_mask, cam_id]  # (n_vis, 2)

            # In-bounds filter
            in_bounds = (
                (uv_ref[:, 0] >= 0) & (uv_ref[:, 0] < W) &
                (uv_ref[:, 1] >= 0) & (uv_ref[:, 1] < H) &
                (uv_def[:, 0] >= 0) & (uv_def[:, 0] < W) &
                (uv_def[:, 1] >= 0) & (uv_def[:, 1] < H)
            )
            if in_bounds.sum() < 5:  # single unavoidable sync per cam
                continue

            # Extract patches
            P_ref = self.dataset.extract_patches(
                self._ref_images_gpu[cam_id],
                uv_ref[in_bounds], patch_size,
            )
            P_def = self.dataset.extract_patches(
                self._def_images_gpu[t_step][cam_id],
                uv_def[in_bounds], patch_size,
            )

            # ZNSSD — all GPU tensors, no sync
            pair_loss = znssd(P_ref, P_def)
            total_znssd = total_znssd + pair_loss * in_bounds.sum()
            valid_count = valid_count + in_bounds.sum()

        # GPU-native division (no sync needed)
        L_dic = total_znssd / valid_count.clamp(min=1)

        # ---- 5. Smoothness regularization ----
        L_smooth = deformation_smoothness_loss(
            self.deformation_net, x_norm, t_tensor
        )

        L_total = L_dic + lambda_smooth * L_smooth

        # ---- 6. Backward ----
        self.optimizer.zero_grad()
        L_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.deformation_net.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()

        return {
            "L_dic": L_dic.item(),
            "L_smooth": L_smooth.item(),
            "L_total": L_total.item(),
            "valid_pairs": int(valid_count.item()),
            "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
        }

    # =================================================================
    # Evaluation
    # =================================================================

    @torch.no_grad()
    def _evaluate(
        self, t_val: float, t_step: int, patch_size: int, M: int, K: int
    ) -> float:
        """Compute ZNSSD on a held-out set of surface points — GPU-optimized."""
        self.deformation_net.eval()

        x, normals = self.surface.sample_surface_points(M, strategy="uniform")
        x_norm = normalize_points(x, self.surface.bbox)
        cam_ids = self.surface.get_visible_cameras(x, max_cams=K)

        t_tensor = torch.full((M, 1), t_val, device=self.device, dtype=torch.float32)
        phi = self.deformation_net(x_norm, t_tensor)
        x_def = unnormalize_points(x_norm + phi, self.surface.bbox)

        # Batch project all cameras at once
        n_cams = self.surface.num_cameras
        W, H = self.dataset.W, self.dataset.H
        uv_all_ref = self.surface.batch_project_all_cameras(x)
        uv_all_def = self.surface.batch_project_all_cameras(x_def)

        # Pre-filter active cameras
        cam_counts = torch.zeros(n_cams, dtype=torch.int32, device=self.device)
        valid_cam_mask = cam_ids >= 0
        cam_flat = cam_ids[valid_cam_mask]
        cam_counts.scatter_add_(0, cam_flat, torch.ones_like(cam_flat, dtype=torch.int32))
        active_cams = torch.where(cam_counts >= 5)[0].tolist()

        total_znssd = torch.tensor(0.0, device=self.device)
        valid_count = torch.tensor(0.0, device=self.device)

        for cam_id in active_cams:
            cam_mask = (cam_ids == cam_id).any(dim=-1)
            uv_ref = uv_all_ref[cam_mask, cam_id]
            uv_def = uv_all_def[cam_mask, cam_id]

            in_bounds = (
                (uv_ref[:, 0] >= 0) & (uv_ref[:, 0] < W) &
                (uv_ref[:, 1] >= 0) & (uv_ref[:, 1] < H) &
                (uv_def[:, 0] >= 0) & (uv_def[:, 0] < W) &
                (uv_def[:, 1] >= 0) & (uv_def[:, 1] < H)
            )
            if in_bounds.sum() < 5:
                continue

            P_ref = self.dataset.extract_patches(
                self._ref_images_gpu[cam_id], uv_ref[in_bounds], patch_size,
            )
            P_def = self.dataset.extract_patches(
                self._def_images_gpu[t_step][cam_id], uv_def[in_bounds], patch_size,
            )

            total_znssd = total_znssd + znssd(P_ref, P_def) * in_bounds.sum()
            valid_count = valid_count + in_bounds.sum()

        self.deformation_net.train()
        return (total_znssd / valid_count.clamp(min=1)).item()

    # =================================================================
    # Checkpointing
    # =================================================================

    def save_checkpoint(self, path: str):
        """Save model weights + optimizer state + config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "deformation_net": self.deformation_net.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "current_phase": self.current_phase,
            "iteration": self.iteration,
            "config": self.config,
            "loss_history": self.loss_history,
        }
        torch.save(state, path)
        print(f"[Trainer] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        """Restore from checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.deformation_net.load_state_dict(state["deformation_net"])
        if state["optimizer"]:
            self.optimizer = torch.optim.Adam(self.deformation_net.parameters())
            self.optimizer.load_state_dict(state["optimizer"])
        self.current_phase = state["current_phase"]
        self.iteration = state["iteration"]
        self.config = state["config"]
        self.loss_history = state.get("loss_history", [])
        print(f"[Trainer] Checkpoint loaded ← {path} "
              f"(phase {self.current_phase}, iter {self.iteration})")

    # =================================================================
    # Inference helpers
    # =================================================================

    @torch.no_grad()
    def query_displacement(
        self, x: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Query Φ at arbitrary 3D world points.

        Args:
            x: (N, 3) world coordinates.
            t: scalar time in [0, 1].

        Returns:
            phi: (N, 3) displacement vectors in world units.
        """
        self.deformation_net.eval()
        x_norm = normalize_points(x.to(self.device), self.surface.bbox)
        t_tensor = torch.full((x.shape[0], 1), t, device=self.device)
        phi_norm = self.deformation_net(x_norm, t_tensor)
        # Convert displacement from normalized to world units:
        # phi_world = phi_norm * scale
        scale = (self.surface.bbox[1] - self.surface.bbox[0]) / 2.0
        phi_world = phi_norm * scale.unsqueeze(0)
        self.deformation_net.train()
        return phi_world

    @torch.no_grad()
    def compute_strain(
        self, x: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Compute displacement gradient (3×3 Jacobian) at given points.

        Args:
            x: (N, 3) world coordinates.
            t: scalar time.

        Returns:
            grad_phi: (N, 3, 3) displacement gradient tensor ∂Φ_i/∂x_j.
        """
        self.deformation_net.eval()
        x_dev = x.to(self.device).requires_grad_(True)
        x_norm = normalize_points(x_dev, self.surface.bbox)
        t_tensor = torch.full((x.shape[0], 1), t, device=self.device)
        phi = self.deformation_net(x_norm, t_tensor)  # (N, 3) in norm space

        # Account for normalization Jacobian: dΦ_world/dx_world = dΦ_norm/dx_norm
        # (the scale factor cancels: Φ_world = Φ_norm * scale, x_norm = x_world / scale,
        #  so dΦ_world/dx_world = d(Φ_norm*scale)/d(x_norm*scale)
        #                        = (dΦ_norm*scale)/(dx_norm*scale)
        #                        = dΦ_norm/dx_norm)

        grad_u = torch.autograd.grad(phi[:, 0].sum(), x_dev, create_graph=False, retain_graph=True)[0]
        grad_v = torch.autograd.grad(phi[:, 1].sum(), x_dev, create_graph=False, retain_graph=True)[0]
        grad_w = torch.autograd.grad(phi[:, 2].sum(), x_dev, create_graph=False, retain_graph=False)[0]

        grad_phi = torch.stack([grad_u, grad_v, grad_w], dim=-1)  # (N, 3, 3)
        self.deformation_net.train()
        return grad_phi
