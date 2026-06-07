"""
Four-stage trainer for NDeF-DIC.

Stage 0: COLMAP preprocessing (offline, not in trainer)
Stage 1: SDF surface learning from COLMAP points
Stage 2: Intensity field pre-training from reference frames
Stage 3: Deformation field training from all frames
Stage 4: Joint refinement of all parameters
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict

from .config import NDeFDICConfig
from .networks import SDFNetwork, IntensityField, DeformationField, AppearanceEmbedding
from .renderer import SurfaceRenderer
from .losses import NDeFDICLoss, smoothness_loss
from .dataset import MultiCamDataset


# ---------------------------------------------------------------------------
# Utility: random points in volume
# ---------------------------------------------------------------------------

def sample_random_volume_points(
    n_points: int,
    bounds: tuple[float, float] = (-3.0, 3.0),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample random 3D points in a bounding volume.

    Used for Eikonal regularization and off-surface SDF supervision.

    Args:
        n_points: Number of points to sample.
        bounds: (min, max) for each axis.
        device: torch device.

    Returns:
        (n_points, 3) tensor.
    """
    lo, hi = bounds
    points = torch.rand(n_points, 3, device=device) * (hi - lo) + lo
    return points


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NDeFDICTrainer:
    """
    Orchestrates the four-stage training pipeline.

    Usage:
        config = NDeFDICConfig(...)
        dataset = MultiCamDataset(config.data_dir, ...)
        trainer = NDeFDICTrainer(config, dataset)
        trainer.train()
    """

    def __init__(self, config: NDeFDICConfig, dataset: MultiCamDataset):
        self.config = config
        self.dataset = dataset
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Ensure masks exist
        dataset.ensure_masks()

        # Initialize networks
        self.sdf_net = SDFNetwork(config.sdf).to(self.device)
        self.intensity_net = IntensityField(config.intensity).to(self.device)
        self.deform_net = DeformationField(config.deformation).to(self.device)
        self.appearance_net = AppearanceEmbedding(
            config.appearance, config.n_cameras
        ).to(self.device)

        # Renderer (combines all networks)
        self.renderer = SurfaceRenderer(
            self.sdf_net,
            self.intensity_net,
            self.deform_net,
            self.appearance_net,
            config,
        )

        # Loss function
        self.loss_fn = NDeFDICLoss(config)

        # Optimizer (recreated per stage)
        self.optimizer: Optional[optim.Adam] = None

        # Logging
        self.logs: Dict[str, list] = defaultdict(list)

        # Working volume bounds for random point sampling
        # Estimate from COLMAP points if available
        self.volume_bounds = self._estimate_volume_bounds()

        print(f"[NDeF-DIC] Trainer initialized on {self.device}")
        print(f"  SDF params:       {sum(p.numel() for p in self.sdf_net.parameters()):,}")
        print(f"  Intensity params: {sum(p.numel() for p in self.intensity_net.parameters()):,}")
        print(f"  Deformation params: {sum(p.numel() for p in self.deform_net.parameters()):,}")
        print(f"  Appearance params: {sum(p.numel() for p in self.appearance_net.parameters()):,}")

    def _estimate_volume_bounds(self) -> tuple[float, float]:
        """Estimate working volume bounds from COLMAP points or use defaults."""
        pts = self.dataset.colmap_points
        if pts is not None and len(pts) > 0:
            # Expand bounds by 20% margin
            center = pts.mean(axis=0)
            extent = (pts.max(axis=0) - pts.min(axis=0)).max() * 0.6
            return (center - extent).item(), (center + extent).item()
        return (-3.0, 3.0)

    def _get_colmap_batch(self) -> Optional[torch.Tensor]:
        """Get a batch of COLMAP surface points."""
        pts = self.dataset.get_colmap_points_tensor()
        if pts is None or len(pts) == 0:
            return None
        batch_size = min(self.config.training.batch_size, len(pts))
        idx = torch.randperm(len(pts), device=self.device)[:batch_size]
        return pts[idx]

    # =======================================================================
    # Stage 1: SDF Surface Learning
    # =======================================================================

    def train_stage1(self):
        """
        Train SDF network from COLMAP sparse points + Eikonal regularization.

        Only SDF is trained; I, Φ, and appearance are frozen.
        """
        print("\n" + "=" * 60)
        print("Stage 1: SDF Surface Learning")
        print("=" * 60)

        config = self.config.training
        sdf_cfg = self.config.sdf

        # Freeze everything except SDF
        for p in self.intensity_net.parameters():
            p.requires_grad = False
        for p in self.deform_net.parameters():
            p.requires_grad = False
        for p in self.appearance_net.parameters():
            p.requires_grad = False
        for p in self.sdf_net.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(
            self.sdf_net.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
        )

        colmap_pts = self.dataset.get_colmap_points_tensor()
        if colmap_pts is None or len(colmap_pts) == 0:
            raise RuntimeError(
                "Stage 1 requires COLMAP points. "
                "Run COLMAP on reference images first."
            )
        print(f"  Training on {len(colmap_pts)} COLMAP points")

        n_epochs = config.stage1_epochs
        batch_size = min(config.batch_size, len(colmap_pts))

        for epoch in range(n_epochs):
            self.sdf_net.train()

            # Sample surface points
            idx = torch.randperm(len(colmap_pts), device=self.device)[:batch_size]
            surf_pts = colmap_pts[idx]  # (B, 3)

            # Sample random volume points for Eikonal
            rand_pts = sample_random_volume_points(
                batch_size // 2, self.volume_bounds, self.device
            )

            # Forward
            s_surf = self.sdf_net(surf_pts)  # (B, 1)

            # SDF data loss
            l_data = (s_surf ** 2).mean()

            # Near-surface loss (if we have normals)
            # For simplicity, skip near-surface and use only on-surface + Eikonal
            # A production version would compute normals

            # Eikonal loss
            rand_pts.requires_grad_(True)
            s_rand = self.sdf_net(rand_pts)
            grad = torch.autograd.grad(
                outputs=s_rand,
                inputs=rand_pts,
                grad_outputs=torch.ones_like(s_rand),
                create_graph=True,
                retain_graph=True,
            )[0]
            l_eik = ((grad.norm(p=2, dim=-1) - 1.0) ** 2).mean()

            loss = sdf_cfg.lambda_data * l_data + sdf_cfg.lambda_eikonal * l_eik

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log
            self.logs["stage1_loss"].append(loss.item())
            self.logs["stage1_data"].append(l_data.item())
            self.logs["stage1_eik"].append(l_eik.item())

            if (epoch + 1) % config.log_interval == 0:
                print(f"  Epoch {epoch + 1:5d}/{n_epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Data: {l_data.item():.4f} | "
                      f"Eik: {l_eik.item():.4f}")

            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint("stage1")

        # Final save
        self._save_checkpoint("stage1_final")
        print(f"  Stage 1 complete.")

    # =======================================================================
    # Stage 2: Intensity Field Pre-training
    # =======================================================================

    def train_stage2(self):
        """
        Train intensity field and appearance embeddings from reference frames.

        SDF is frozen; deformation is bypassed (t=0, no displacement).
        """
        print("\n" + "=" * 60)
        print("Stage 2: Intensity Field Pre-training (Reference Frames)")
        print("=" * 60)

        config = self.config.training

        # Freeze SDF and deformation
        for p in self.sdf_net.parameters():
            p.requires_grad = False
        for p in self.deform_net.parameters():
            p.requires_grad = False
        for p in self.intensity_net.parameters():
            p.requires_grad = True
        for p in self.appearance_net.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(
            list(self.intensity_net.parameters()) +
            list(self.appearance_net.parameters()),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
        )

        n_epochs = config.stage2_epochs
        batch_size = config.batch_size
        n_cameras = self.config.n_cameras

        for epoch in range(n_epochs):
            self.intensity_net.train()
            self.appearance_net.train()
            self.sdf_net.eval()

            # Random camera and random pixels
            cam_id = np.random.randint(0, n_cameras)
            pixels_uv, obs_img, mask = self.dataset.sample_pixels(
                cam_id, batch_size, step=0  # reference frame only
            )
            K, R, t = self.dataset.get_camera_tensors(cam_id)

            # Render (no deformation)
            render_out = self.renderer.render_reference_only(
                K, R, t, cam_id, pixels_uv
            )

            # Mask filtering
            valid = render_out["valid"].clone()
            if mask is not None:
                # Check mask at projected pixel locations
                proj_uv = render_out["proj_uv"]
                u_clamped = proj_uv[:, 0].long().clamp(0, self.config.image_width - 1)
                v_clamped = proj_uv[:, 1].long().clamp(0, self.config.image_height - 1)
                in_mask = mask[v_clamped, u_clamped] > 0.5
                valid = valid & in_mask
            render_out["valid"] = valid

            # Loss
            total, loss_dict = self.loss_fn(
                render_out, obs_img,
                self.sdf_net, self.appearance_net,
                colmap_points=None,
                random_points=None,
                stage="intensity",
            )

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            # Log
            for k, v in loss_dict.items():
                self.logs[f"stage2_{k}"].append(v)

            if (epoch + 1) % config.log_interval == 0:
                hit_rate = render_out["hit_mask"].float().mean().item()
                print(f"  Epoch {epoch + 1:5d}/{n_epochs} | "
                      f"Photo: {loss_dict.get('photo', 0):.4f} | "
                      f"App: {loss_dict.get('app_reg', 0):.6f} | "
                      f"Hit: {hit_rate:.3f}")

            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint("stage2")

        self._save_checkpoint("stage2_final")
        print(f"  Stage 2 complete.")

    # =======================================================================
    # Stage 3: Deformation Field Training
    # =======================================================================

    def train_stage3(self):
        """
        Train deformation field from all frames (reference + deformed).

        SDF, intensity, and appearance are frozen.
        """
        print("\n" + "=" * 60)
        print("Stage 3: Deformation Field Training (All Frames)")
        print("=" * 60)

        config = self.config.training
        def_cfg = self.config.deformation

        # Freeze SDF, intensity, appearance
        for p in self.sdf_net.parameters():
            p.requires_grad = False
        for p in self.intensity_net.parameters():
            p.requires_grad = False
        for p in self.appearance_net.parameters():
            p.requires_grad = False
        for p in self.deform_net.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(
            self.deform_net.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
        )

        n_epochs = config.stage3_epochs
        batch_size = config.batch_size
        n_cameras = self.config.n_cameras
        n_steps = self.config.n_load_steps

        for epoch in range(n_epochs):
            self.deform_net.train()
            self.sdf_net.eval()
            self.intensity_net.eval()
            self.appearance_net.eval()

            # Random camera and random load step
            cam_id = np.random.randint(0, n_cameras)
            step = np.random.randint(0, n_steps)
            load_t = step / max(n_steps - 1, 1)  # normalize to [0, 1]

            pixels_uv, obs_img, mask = self.dataset.sample_pixels(
                cam_id, batch_size, step=step
            )
            K, R, t = self.dataset.get_camera_tensors(cam_id)

            # Render with deformation
            render_out = self.renderer.render(
                K, R, t, cam_id, load_t, pixels_uv
            )
            # Inject load_step for loss computation
            render_out["load_step"] = load_t

            # Mask filtering
            valid = render_out["valid"].clone()
            if mask is not None:
                proj_uv = render_out["proj_uv"]
                u_clamped = proj_uv[:, 0].long().clamp(0, self.config.image_width - 1)
                v_clamped = proj_uv[:, 1].long().clamp(0, self.config.image_height - 1)
                in_mask = mask[v_clamped, u_clamped] > 0.5
                valid = valid & in_mask
            render_out["valid"] = valid

            # Loss
            total, loss_dict = self.loss_fn(
                render_out, obs_img,
                self.sdf_net, self.appearance_net,
                colmap_points=None,
                random_points=None,
                stage="deformation",
            )

            # Additional smoothness loss (computed directly)
            valid = render_out["valid"]
            if valid.sum() > 0:
                x_surf = render_out["x_surface"][valid]
                x_surf_grad = x_surf.detach().clone().requires_grad_(True)
                t_tensor = torch.full((x_surf_grad.shape[0], 1), load_t, device=self.device)
                l_smooth = smoothness_loss(self.deform_net, x_surf_grad, t_tensor)
                total = total + def_cfg.lambda_smooth * l_smooth
                loss_dict["smooth"] = l_smooth.item()

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            # Log
            for k, v in loss_dict.items():
                self.logs[f"stage3_{k}"].append(v)

            if (epoch + 1) % config.log_interval == 0:
                hit_rate = render_out["hit_mask"].float().mean().item()
                mean_disp = render_out["x_deformed"].norm(dim=-1).mean().item() \
                    if render_out["valid"].sum() > 0 else 0.0
                print(f"  Epoch {epoch + 1:5d}/{n_epochs} | "
                      f"Photo: {loss_dict.get('photo', 0):.4f} | "
                      f"Smooth: {loss_dict.get('smooth', 0):.4f} | "
                      f"Disp: {mean_disp:.3f} | "
                      f"Hit: {hit_rate:.3f}")

            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint("stage3")

        self._save_checkpoint("stage3_final")
        print(f"  Stage 3 complete.")

    # =======================================================================
    # Stage 4: Joint Refinement (Optional)
    # =======================================================================

    def train_stage4(self):
        """
        Joint refinement of all parameters.

        All networks unfrozen, small learning rate.
        """
        print("\n" + "=" * 60)
        print("Stage 4: Joint Refinement")
        print("=" * 60)

        config = self.config.training

        # Unfreeze all
        for p in self.sdf_net.parameters():
            p.requires_grad = True
        for p in self.intensity_net.parameters():
            p.requires_grad = True
        for p in self.deform_net.parameters():
            p.requires_grad = True
        for p in self.appearance_net.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(
            list(self.sdf_net.parameters()) +
            list(self.intensity_net.parameters()) +
            list(self.deform_net.parameters()) +
            list(self.appearance_net.parameters()),
            lr=config.lr_joint,
            betas=(config.adam_beta1, config.adam_beta2),
        )

        sdf_cfg = self.config.sdf
        def_cfg = self.config.deformation

        n_epochs = config.stage4_epochs
        batch_size = config.batch_size
        n_cameras = self.config.n_cameras
        n_steps = self.config.n_load_steps

        for epoch in range(n_epochs):
            self.sdf_net.train()
            self.intensity_net.train()
            self.deform_net.train()
            self.appearance_net.train()

            cam_id = np.random.randint(0, n_cameras)
            step = np.random.randint(0, n_steps)
            load_t = step / max(n_steps - 1, 1)

            pixels_uv, obs_img, mask = self.dataset.sample_pixels(
                cam_id, batch_size, step=step
            )
            K, R, t = self.dataset.get_camera_tensors(cam_id)

            render_out = self.renderer.render(K, R, t, cam_id, load_t, pixels_uv)
            render_out["load_step"] = load_t

            # Mask
            valid = render_out["valid"].clone()
            if mask is not None:
                proj_uv = render_out["proj_uv"]
                u_clamped = proj_uv[:, 0].long().clamp(0, self.config.image_width - 1)
                v_clamped = proj_uv[:, 1].long().clamp(0, self.config.image_height - 1)
                in_mask = mask[v_clamped, u_clamped] > 0.5
                valid = valid & in_mask
            render_out["valid"] = valid

            # SDF and Eikonal losses
            colmap_pts = self._get_colmap_batch()
            rand_pts = sample_random_volume_points(
                batch_size // 4, self.volume_bounds, self.device
            )

            total, loss_dict = self.loss_fn(
                render_out, obs_img,
                self.sdf_net, self.appearance_net,
                colmap_points=colmap_pts,
                random_points=rand_pts,
                stage="joint",
            )

            # Additional smoothness
            valid = render_out["valid"]
            if valid.sum() > 0:
                x_surf = render_out["x_surface"][valid]
                x_surf_grad = x_surf.detach().clone().requires_grad_(True)
                t_tensor = torch.full((x_surf_grad.shape[0], 1), load_t, device=self.device)
                l_smooth = smoothness_loss(self.deform_net, x_surf_grad, t_tensor)
                total = total + def_cfg.lambda_smooth * l_smooth
                loss_dict["smooth"] = l_smooth.item()

            self.optimizer.zero_grad()
            total.backward()
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(self.sdf_net.parameters(), 1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                self.logs[f"stage4_{k}"].append(v)

            if (epoch + 1) % config.log_interval == 0:
                print(f"  Epoch {epoch + 1:5d}/{n_epochs} | "
                      f"Total: {loss_dict.get('total', 0):.4f} | "
                      f"Photo: {loss_dict.get('photo', 0):.4f}")

            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint("stage4")

        self._save_checkpoint("stage4_final")
        print(f"  Stage 4 complete.")

    # =======================================================================
    # Full training
    # =======================================================================

    def train(self, stages: List[str] | None = None):
        """
        Run the full training pipeline.

        Args:
            stages: List of stages to run.
                    None = all stages.
                    e.g., ["sdf", "intensity", "deformation"]
        """
        if stages is None:
            stages = ["sdf", "intensity", "deformation", "joint"]

        stage_map = {
            "sdf": self.train_stage1,
            "intensity": self.train_stage2,
            "deformation": self.train_stage3,
            "joint": self.train_stage4,
        }

        for stage_name in stages:
            if stage_name in stage_map:
                stage_map[stage_name]()
            else:
                print(f"[WARNING] Unknown stage: {stage_name}")

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    # =======================================================================
    # Checkpointing
    # =======================================================================

    def _save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        ckpt_dir = os.path.join(self.config.work_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save({
            "sdf": self.sdf_net.state_dict(),
            "intensity": self.intensity_net.state_dict(),
            "deformation": self.deform_net.state_dict(),
            "appearance": self.appearance_net.state_dict(),
            "logs": dict(self.logs),
            "config": self.config,
        }, path)
        print(f"  [Checkpoint] Saved to {path}")

    def load_checkpoint(self, tag: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.work_dir, "checkpoints", f"{tag}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.sdf_net.load_state_dict(ckpt["sdf"])
        self.intensity_net.load_state_dict(ckpt["intensity"])
        self.deform_net.load_state_dict(ckpt["deformation"])
        self.appearance_net.load_state_dict(ckpt["appearance"])
        self.logs = defaultdict(list, ckpt.get("logs", {}))
        print(f"  [Checkpoint] Loaded from {path}")

    # =======================================================================
    # Inference
    # =======================================================================

    @torch.no_grad()
    def query_displacement(
        self,
        x_surface: torch.Tensor,
        load_step: float,
    ) -> torch.Tensor:
        """
        Query the deformation field at arbitrary surface points.

        Args:
            x_surface: (N, 3) reference surface points.
            load_step: float ∈ [0, 1].

        Returns:
            uvw: (N, 3) displacement vectors.
        """
        self.deform_net.eval()
        t = torch.full((x_surface.shape[0], 1), load_step, device=x_surface.device)
        return self.deform_net(x_surface, t)

    @torch.no_grad()
    def compute_strain(
        self,
        x_surface: torch.Tensor,
        load_step: float,
    ) -> torch.Tensor:
        """
        Compute the infinitesimal strain tensor at surface points.

        ε = (∇Φ + ∇Φᵀ) / 2

        Args:
            x_surface: (N, 3) reference surface points.
            load_step: float ∈ [0, 1].

        Returns:
            strain: (N, 6) Voigt notation: [ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]
        """
        self.deform_net.eval()
        x = x_surface.clone().requires_grad_(True)
        t = torch.full((x.shape[0], 1), load_step, device=x.device)
        uvw = self.deform_net(x, t)

        # Jacobian J_ij = ∂u_i/∂x_j
        J = torch.zeros(x.shape[0], 3, 3, device=x.device)
        for i in range(3):
            grad_i = torch.autograd.grad(
                outputs=uvw[:, i].sum(),
                inputs=x,
                create_graph=False,
                retain_graph=True,
            )[0]  # (N, 3)
            J[:, i, :] = grad_i

        # Infinitesimal strain: ε = (J + Jᵀ) / 2
        eps = (J + J.transpose(-2, -1)) / 2.0

        # Voigt notation
        strain_voigt = torch.stack([
            eps[:, 0, 0],  # ε_xx
            eps[:, 1, 1],  # ε_yy
            eps[:, 2, 2],  # ε_zz
            eps[:, 0, 1],  # ε_xy (engineering: 2*ε_xy, but we keep tensor shear)
            eps[:, 0, 2],  # ε_xz
            eps[:, 1, 2],  # ε_yz
        ], dim=-1)  # (N, 6)

        return strain_voigt
