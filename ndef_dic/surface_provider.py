"""
Step 2: SurfaceProvider — 表面采样接口。

提供统一的抽象接口供 Step 3 (Φ 网络训练) 使用，支持两种后端：
  - PointCloudSurface:   基于 Step 1 后处理的稠密点云（PatchMatch 路径）
  - NeuralStereoSurface: 基于 Step 1 训练好的 D_c 网络（PINN-Stereo 路径）

两种实现对 Step 3 完全透明 — project_to_camera() 行为相同，因为
投影只需要 K, R, t，不关心 3D 点是怎么来的。

参考: docs/step2-step3-interface.md, docs/step2-surface-representation.md
"""

import os
import json
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# =========================================================================
# 坐标归一化工具
# =========================================================================

def compute_bbox(points: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """计算世界坐标系的包围盒，带 margin。

    Args:
        points: (N, 3) 世界坐标点
        margin: 相对扩展比例 (0.05 = 5%)

    Returns:
        bbox: (2, 3) [min, max]
    """
    pmin = points.min(dim=0).values
    pmax = points.max(dim=0).values
    extent = pmax - pmin
    pmin = pmin - extent * margin
    pmax = pmax + extent * margin
    return torch.stack([pmin, pmax], dim=0)


def normalize_points(points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """将世界坐标归一化到 [-1, 1]³。

    Args:
        points: (..., 3) 世界坐标
        bbox:   (2, 3) [min, max]

    Returns:
        normalized: (..., 3) in [-1, 1]³
    """
    center = (bbox[0] + bbox[1]) / 2.0
    scale = (bbox[1] - bbox[0]) / 2.0
    return (points - center) / scale.clamp(min=1e-8)


def unnormalize_points(normalized: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """将归一化坐标恢复为世界坐标。

    Args:
        normalized: (..., 3) in [-1, 1]³
        bbox:       (2, 3) [min, max]

    Returns:
        world: (..., 3) 世界坐标
    """
    center = (bbox[0] + bbox[1]) / 2.0
    scale = (bbox[1] - bbox[0]) / 2.0
    return normalized * scale + center


# =========================================================================
# CameraParams (轻量 dataclass, 与 dataset.py 兼容)
# =========================================================================

@dataclass
class CameraParams:
    """单台相机的内外参。"""
    K: np.ndarray       # (3, 3) 内参矩阵
    R: np.ndarray       # (3, 3) world→cam 旋转
    t: np.ndarray       # (3,)   world→cam 平移
    dist: np.ndarray    # (5,)   畸变
    width: int
    height: int


# =========================================================================
# 抽象接口
# =========================================================================

class SurfaceProvider(ABC):
    """Step 2 提供给 Step 3 的唯一接口。

    Step 3 只依赖这个接口，不关心底层是 D_c 网络还是点云。
    """

    @abstractmethod
    def sample_surface_points(
        self, n: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样 n 个表面点。

        Args:
            n: 采样数量
            strategy: "uniform" | "visibility_weighted"

        Returns:
            points:  (n, 3) 世界坐标系 3D 坐标
            normals: (n, 3) 单位法向量
        """
        ...

    @abstractmethod
    def get_visible_cameras(
        self, points: torch.Tensor, max_cams: int = 3
    ) -> torch.Tensor:
        """对每个表面点返回可见相机索引。

        Args:
            points:   (n, 3) 世界坐标
            max_cams: 每个点最多返回几个相机

        Returns:
            cam_ids: (n, max_cams) int, -1 表示无效
        """
        ...

    @abstractmethod
    def project_to_camera(
        self, points: torch.Tensor, cam_id: int
    ) -> torch.Tensor:
        """将 3D 表面点投影到指定相机。

        Args:
            points: (n, 3) 世界坐标
            cam_id: 相机索引

        Returns:
            uv: (n, 2) 像素坐标 (col, row)
        """
        ...

    @property
    @abstractmethod
    def num_cameras(self) -> int:
        """相机数量"""
        ...

    @property
    @abstractmethod
    def bbox(self) -> torch.Tensor:
        """世界坐标系包围盒 (2, 3) [min, max]"""
        ...


# =========================================================================
# 实现 A: PointCloudSurface (PatchMatch / COLMAP Dense 路径)
# =========================================================================

class PointCloudSurface(SurfaceProvider):
    """基于后处理点云的表面表示。

    从 Step 1 postprocess 的输出加载:
      dense/points_norm.npy  (或 dense_points.npy)
      dense/normals.npy
      dense/vis_mask.npy

    核心优化: sample_surface_points 返回索引，get_visible_cameras
    通过索引直接查 vis_mask — O(1) 而非 O(N) 最近邻搜索。
    """

    def __init__(
        self,
        points: torch.Tensor,           # (N, 3) 世界坐标
        normals: torch.Tensor,          # (N, 3) 单位法向量
        vis_mask: torch.Tensor,         # (N, N_cam) bool
        K_list: List[np.ndarray],       # per-camera intrinsics
        R_list: List[np.ndarray],       # per-camera world→cam rotation
        t_list: List[np.ndarray],       # per-camera world→cam translation
        image_width: int,
        image_height: int,
        device: str = "cuda",
    ):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._points = points.to(self._device)
        self._normals = normals.to(self._device)
        self._vis_mask = vis_mask.to(self._device)
        self._image_width = image_width
        self._image_height = image_height

        # Camera parameters → tensors
        self._K = [torch.from_numpy(K.astype(np.float32)).to(self._device) for K in K_list]
        self._R = [torch.from_numpy(R.astype(np.float32)).to(self._device) for R in R_list]
        self._t = [torch.from_numpy(t.astype(np.float32).reshape(3)).to(self._device) for t in t_list]

        self._n_cam = len(K_list)
        self._n_points = len(points)

        # Cache per-index visible cameras as lists for fast lookup
        # Precompute: for each point, which cameras can see it (sorted by relevance)
        self._visible_cams_cache = _build_visible_cams_cache(self._vis_mask)

        # Compute bbox from points
        self._bbox = compute_bbox(self._points, margin=0.05)

        # Build KD-Tree for O(log N) nearest-neighbor queries (replaces O(N²) brute-force)
        from scipy.spatial import cKDTree
        self._kdtree = cKDTree(points.cpu().numpy().astype(np.float64))
        # Cache of last-sampled indices for zero-overhead visibility lookup
        self._last_sample_idx: Optional[torch.Tensor] = None

        # Precompute vis lookup table: (N, max_cams) → O(1) vectorized indexing
        # Each row stores up to max_cams visible camera IDs, padded with -1
        max_cams_lookup = 6  # generous default
        vis_lookup = torch.full((self._n_points, max_cams_lookup), -1, dtype=torch.long)
        for i in range(self._n_points):
            visible = self._visible_cams_cache[i]
            n_vis = min(len(visible), max_cams_lookup)
            if n_vis > 0:
                vis_lookup[i, :n_vis] = visible[:n_vis]
        self._vis_lookup = vis_lookup.to(self._device)
        self._vis_lookup_max = max_cams_lookup

    # ---- 属性 ----

    @property
    def num_cameras(self) -> int:
        return self._n_cam

    @property
    def bbox(self) -> torch.Tensor:
        return self._bbox

    @property
    def device(self) -> torch.device:
        return self._device

    # ---- 采样 ----

    def sample_surface_points(
        self, n: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从点云中随机采样。

        Args:
            n: 采样数量
            strategy:
              - "uniform": 等概率采样所有点
              - "visibility_weighted": 优先采样可见相机多的点

        Returns:
            points:  (n, 3) 世界坐标
            normals: (n, 3) 单位法向量

        After calling this, get_visible_cameras() uses the cached indices
        for O(1) lookup — no NN search needed.
        """
        n_points = self._n_points

        if strategy == "uniform":
            idx = torch.randint(0, n_points, (n,), device=self._device)

        elif strategy == "visibility_weighted":
            # 可见相机多的点有更高概率被采样
            weights = self._vis_mask.sum(dim=1).float()  # (N,)
            weights = weights / weights.sum()
            # Use numpy for weighted sampling (torch doesn't have native weighted choice)
            weights_np = weights.cpu().numpy()
            idx_np = np.random.choice(n_points, size=n, p=weights_np)
            idx = torch.from_numpy(idx_np).long().to(self._device)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Cache indices so get_visible_cameras() can do O(1) lookup
        self._last_sample_idx = idx

        return self._points[idx], self._normals[idx]

    # ---- 可见性 ----

    def get_visible_cameras(
        self, points: torch.Tensor, max_cams: int = 3
    ) -> torch.Tensor:
        """返回每个查询点的可见相机索引。

        优化路径 (按优先级):
          1. 如果 points 与最近一次 sample_surface_points 返回的点相同
             (通过 _last_sample_idx 判断)，直接用索引 O(1) 查 vis_mask。
          2. 否则用 KD-Tree 最近邻 (O(log N)) 查找对应参考点。
          3. 极端情况回退到 chunked brute-force。

        Args:
            points:   (N, 3) 世界坐标
            max_cams: 每个点最多返回几个相机

        Returns:
            cam_ids: (N, max_cams) int64, -1 表示无效
        """
        n_query = points.shape[0]
        device = points.device

        # ---- Path 1: Use cached sample indices (O(1)) ----
        if self._last_sample_idx is not None and len(self._last_sample_idx) == n_query:
            idx = self._last_sample_idx
            # Verify points match (quick shape + first-element check)
            if torch.allclose(points[0], self._points[idx[0]], atol=1e-6):
                return self._get_visible_cams_by_idx(idx, max_cams, device)

        # ---- Path 2: KD-Tree query (O(log N)) ----
        # Move query points to numpy (CPU) for scipy KD-Tree
        points_np = points.detach().cpu().numpy().astype(np.float64)
        _, nn_idx = self._kdtree.query(points_np, k=1)
        nn_idx_t = torch.from_numpy(nn_idx.astype(np.int64)).to(device)
        return self._get_visible_cams_by_idx(nn_idx_t, max_cams, device)

    def _get_visible_cams_by_idx(
        self, idx: torch.Tensor, max_cams: int, device: torch.device,
    ) -> torch.Tensor:
        """Vectorized O(1) visibility lookup — no loops, no GPU syncs.

        Args:
            idx:      (N,) int64 indices into the point cloud.
            max_cams: max cameras per point (clamped to _vis_lookup_max).
            device:   output device.

        Returns:
            cam_ids: (N, max_cams) int64, -1 for invalid slots.
        """
        max_cams = min(max_cams, self._vis_lookup_max)
        # Simple index select: (N,) → (N, max_cams) with truncation
        full = self._vis_lookup[idx]  # (N, _vis_lookup_max)
        return full[:, :max_cams]

    # ---- 投影 ----

    def project_to_camera(
        self, points: torch.Tensor, cam_id: int
    ) -> torch.Tensor:
        """World → Camera → Image 投影。纯 PyTorch，完全可微。"""
        K = self._K[cam_id]
        R = self._R[cam_id]
        t = self._t[cam_id]

        # World → Camera
        P_cam = (R @ points.T + t.unsqueeze(-1)).T  # (n, 3)

        # Camera → Image
        uv_h = (K @ P_cam.T).T  # (n, 3)
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-8)

        return uv

    def batch_project_all_cameras(
        self, points: torch.Tensor,
    ) -> torch.Tensor:
        """GPU-optimized: project N points to ALL cameras in one batched operation.

        Uses stacked (n_cams, 3, 3) matrix multiplies instead of per-camera loops.
        On GPU, the single (N, n_cams, 3) @ (n_cams, 3, 3)^T matmul is far more
        efficient than N separate (n_i, 3) @ (3, 3) operations.

        Args:
            points: (N, 3) world coordinates.

        Returns:
            uv: (N, n_cameras, 2) pixel coordinates (col, row).
        """
        device = points.device
        n_cams = self._n_cam

        # Build stacked camera matrices (cached on first call)
        if not hasattr(self, '_R_stack'):
            self._R_stack = torch.stack(self._R, dim=0)  # (n_cams, 3, 3)
            self._t_stack = torch.stack(self._t, dim=0)  # (n_cams, 3)
            # Precompute intrinsic projection params
            fx = torch.stack([K[0, 0] for K in self._K])  # (n_cams,)
            fy = torch.stack([K[1, 1] for K in self._K])  # (n_cams,)
            cx = torch.stack([K[0, 2] for K in self._K])  # (n_cams,)
            cy = torch.stack([K[1, 2] for K in self._K])  # (n_cams,)
            self._K_params = (fx, fy, cx, cy)

        fx, fy, cx, cy = self._K_params

        # World → Camera: einsum('nj,ckj->nck', points, R) → (N, n_cams, 3)
        P_cam = torch.einsum('nj,ckj->nck', points, self._R_stack)
        P_cam = P_cam + self._t_stack.unsqueeze(0)  # (N, n_cams, 3)

        # Perspective division
        depth = P_cam[..., 2:3].clamp(min=1e-8)  # (N, n_cams, 1)
        uv_norm = P_cam[..., :2] / depth  # (N, n_cams, 2)

        # Apply intrinsics
        u = uv_norm[..., 0] * fx.unsqueeze(0) + cx.unsqueeze(0)  # (N, n_cams)
        v = uv_norm[..., 1] * fy.unsqueeze(0) + cy.unsqueeze(0)  # (N, n_cams)
        uv = torch.stack([u, v], dim=-1)  # (N, n_cams, 2)

        return uv


def _build_visible_cams_cache(vis_mask: torch.Tensor) -> List[torch.Tensor]:
    """预计算每个点的可见相机列表（按可见数排序）。

    Args:
        vis_mask: (N, N_cam) bool

    Returns:
        list of (K,) tensors, each containing visible camera indices (int64)
    """
    cache = []
    vis_mask_cpu = vis_mask.cpu().numpy()
    for i in range(len(vis_mask_cpu)):
        cams = np.where(vis_mask_cpu[i])[0]
        cache.append(torch.from_numpy(cams).long())
    return cache


# =========================================================================
# 实现 B: NeuralStereoSurface (PINN-Stereo 路径)
# =========================================================================

class NeuralStereoSurface(SurfaceProvider):
    """基于训练好的 D_c 深度网络的隐式表面表示。

    表面点通过采样像素坐标 + 查询 D_c 网络 + 反投影获得。
    法向量通过 D_c 的解析梯度计算（比 PCA 点云法向量更精确）。
    可见性通过 D_j 深度自洽性在线判断。

    要求:
      - depth_networks: 每个相机一个训练好的 DepthNetwork
      - K_list, R_list, t_list: 相机参数
      - image_shapes: 每台相机的 (H, W)
      - masks (可选): 每台相机的 ROI mask
    """

    def __init__(
        self,
        depth_networks: List[torch.nn.Module],  # D_c per camera
        K_list: List[np.ndarray],
        R_list: List[np.ndarray],
        t_list: List[np.ndarray],
        image_width: int,
        image_height: int,
        masks: Optional[List[torch.Tensor]] = None,  # per-camera ROI (H, W) bool
        depth_consistency_threshold: float = 0.05,    # 5% relative depth difference
        normal_facing_threshold_deg: float = 80.0,     # max grazing angle
        device: str = "cuda",
    ):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.depth_nets = torch.nn.ModuleList(
            [net.to(self._device) for net in depth_networks]
        )
        self._n_cam = len(depth_networks)
        self._W = image_width
        self._H = image_height

        # Camera parameters → tensors
        self._K = [torch.from_numpy(K.astype(np.float32)).to(self._device) for K in K_list]
        self._R = [torch.from_numpy(R.astype(np.float32)).to(self._device) for R in R_list]
        self._t = [torch.from_numpy(t.astype(np.float32).reshape(3)).to(self._device) for t in t_list]

        # ROI masks
        if masks is not None:
            self._masks = [m.to(self._device) if m.device != self._device else m for m in masks]
        else:
            self._masks = None

        self._depth_thresh = depth_consistency_threshold
        self._cos_normal_thresh = math.cos(math.radians(normal_facing_threshold_deg))

        # Compute valid pixel counts for quota allocation
        self._valid_pixels = self._compute_valid_pixels()

        # Estimate bbox from sparse sample
        self._bbox = self._estimate_bbox()

    # ---- 属性 ----

    @property
    def num_cameras(self) -> int:
        return self._n_cam

    @property
    def bbox(self) -> torch.Tensor:
        return self._bbox

    @property
    def device(self) -> torch.device:
        return self._device

    # ---- 内部: ROI 统计 ----

    def _compute_valid_pixels(self) -> torch.Tensor:
        """每台相机有效 ROI 像素数，用于采样配额分配。"""
        counts = []
        for c in range(self._n_cam):
            if self._masks is not None and self._masks[c] is not None:
                counts.append(self._masks[c].sum().item())
            else:
                counts.append(self._H * self._W)
        return torch.tensor(counts, device=self._device)

    def _estimate_bbox(self, n_samples: int = 5000) -> torch.Tensor:
        """通过稀疏采样估计世界坐标包围盒。"""
        all_pts = []
        n_cams = self._n_cam
        n_per_cam = max(1, n_samples // n_cams)

        with torch.no_grad():
            for c in range(n_cams):
                uv = self._sample_pixels_in_mask(c, n_per_cam)
                if uv.numel() == 0:
                    continue
                uv_norm = self._uv_to_norm(uv)
                depth = self.depth_nets[c](uv_norm).squeeze(-1)
                pts = self._unproject(uv, depth, c)
                all_pts.append(pts)

        if not all_pts:
            # Fallback: unit cube
            return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=self._device)

        all_pts = torch.cat(all_pts, dim=0)
        return compute_bbox(all_pts, margin=0.05)

    # ---- 内部: 像素采样 ----

    def _sample_pixels_in_mask(self, cam_id: int, n: int) -> torch.Tensor:
        """在相机 cam_id 的 ROI 内随机采样 n 个像素坐标 (col, row)。"""
        if self._masks is not None and self._masks[cam_id] is not None:
            mask = self._masks[cam_id]
            valid_rows, valid_cols = torch.where(mask)
            if len(valid_rows) < n:
                return torch.stack([valid_cols.float(), valid_rows.float()], dim=-1)

            idx = torch.randint(0, len(valid_rows), (n,), device=self._device)
            u = valid_cols[idx].float()
            v = valid_rows[idx].float()
        else:
            u = torch.rand(n, device=self._device) * self._W
            v = torch.rand(n, device=self._device) * self._H

        return torch.stack([u, v], dim=-1)  # (n, 2)

    @staticmethod
    def _pixel_to_norm_static(uv: torch.Tensor, W: int, H: int) -> torch.Tensor:
        """静态版本 — 像素坐标 (col, row) → 归一化坐标 [-1, 1]。
        用于不需要访问实例的场景。
        """
        u_norm = 2.0 * uv[:, 0] / (W - 1) - 1.0
        v_norm = 2.0 * uv[:, 1] / (H - 1) - 1.0
        return torch.stack([u_norm, v_norm], dim=-1)

    def _uv_to_norm(self, uv: torch.Tensor) -> torch.Tensor:
        """像素坐标 (col, row) → 归一化坐标 [-1, 1]。
        用于 D_c 网络输入。
        """
        u_norm = 2.0 * uv[:, 0] / (self._W - 1) - 1.0
        v_norm = 2.0 * uv[:, 1] / (self._H - 1) - 1.0
        return torch.stack([u_norm, v_norm], dim=-1)

    # ---- 内部: 投影 / 反投影 ----

    def _unproject(
        self, uv: torch.Tensor, depth: torch.Tensor, cam_id: int
    ) -> torch.Tensor:
        """像素 + 深度 → 世界坐标系 3D 点。

        Args:
            uv:    (n, 2) 像素坐标 (col, row)
            depth: (n,) 或 (n, 1) 深度
            cam_id: 相机索引

        Returns:
            P_world: (n, 3) 世界坐标
        """
        if depth.dim() == 2:
            depth = depth.squeeze(-1)

        K = self._K[cam_id]
        R = self._R[cam_id]
        t = self._t[cam_id]

        # 像素 → 相机射线方向
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (uv[:, 0] - cx) / fx * depth
        y_cam = (uv[:, 1] - cy) / fy * depth
        z_cam = depth
        P_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (n, 3)

        # 相机 → 世界: P_world = R^T @ (P_cam - t)
        P_world = (R.T @ (P_cam - t.unsqueeze(0)).T).T  # (n, 3)
        return P_world

    def _compute_normal(self, uv: torch.Tensor, cam_id: int) -> torch.Tensor:
        """从 D_c 解析计算世界坐标系表面法向量。

        使用 autograd 计算 ∂d/∂u, ∂d/∂v,
        然后通过 unproject Jacobian 转换到世界坐标系。

        Args:
            uv: (n, 2) 像素坐标 (col, row)
            cam_id: 相机索引

        Returns:
            normal: (n, 3) 单位法向量（detached）
        """
        K = self._K[cam_id]
        R = self._R[cam_id]
        t = self._t[cam_id]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        n_pts = uv.shape[0]

        # Compute depth with grad
        uv_norm = self._uv_to_norm(uv)
        uv_norm.requires_grad_(True)

        depth = self.depth_nets[cam_id](uv_norm)  # (n, 1)

        # ∂d/∂u_norm, ∂d/∂v_norm (autograd)
        grad_outputs = torch.ones_like(depth)
        grad_u = torch.autograd.grad(
            depth, uv_norm, grad_outputs=grad_outputs,
            create_graph=False, retain_graph=True
        )[0][:, 0]  # (n,) — ∂d/∂u_norm
        grad_v = torch.autograd.grad(
            depth, uv_norm, grad_outputs=grad_outputs,
            create_graph=False, retain_graph=False
        )[0][:, 1]  # (n,) — ∂d/∂v_norm

        # Chain: ∂d/∂u = ∂d/∂u_norm * ∂u_norm/∂u
        # ∂u_norm/∂u = 2 / (W-1)
        grad_u_pix = grad_u * (2.0 / (self._W - 1))
        grad_v_pix = grad_v * (2.0 / (self._H - 1))

        depth_val = depth.squeeze(-1)  # (n,)

        # Tangent vectors in camera frame:
        # P_cam = [ (u-cx)/fx * d,  (v-cy)/fy * d,  d ]
        # ∂P_cam/∂u = [ d/fx + (u-cx)/fx * ∂d/∂u,  (v-cy)/fy * ∂d/∂u,  ∂d/∂u ]
        # ∂P_cam/∂v = [ (u-cx)/fx * ∂d/∂v,  d/fy + (v-cy)/fy * ∂d/∂v,  ∂d/∂v ]

        dx_du_cam = torch.stack([
            depth_val / fx + (uv[:, 0] - cx) / fx * grad_u_pix,
            (uv[:, 1] - cy) / fy * grad_u_pix,
            grad_u_pix,
        ], dim=-1)  # (n, 3)

        dx_dv_cam = torch.stack([
            (uv[:, 0] - cx) / fx * grad_v_pix,
            depth_val / fy + (uv[:, 1] - cy) / fy * grad_v_pix,
            grad_v_pix,
        ], dim=-1)  # (n, 3)

        # Transform tangent vectors to world
        R_T = R.T  # (3, 3) cam→world rotation
        dx_du_world = (R_T @ dx_du_cam.T).T   # (n, 3)
        dx_dv_world = (R_T @ dx_dv_cam.T).T   # (n, 3)

        # Normal = dx_du × dx_dv
        normal = torch.cross(dx_du_world, dx_dv_world, dim=-1)  # (n, 3)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-8)

        # 方向一致性: 法向量应指向相机
        cam_center = -(R_T @ t).squeeze(-1)  # (3,) world cam center
        P_world = self._unproject(uv, depth_val, cam_id)  # (n, 3)
        view_dir = cam_center.unsqueeze(0) - P_world  # (n, 3)
        view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)
        dot = (normal * view_dir).sum(dim=-1)
        flip = dot < 0
        normal[flip] = -normal[flip]

        return normal.detach()

    # ---- 采样 ----

    def sample_surface_points(
        self, n: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样 n 个表面点。

        按每台相机有效像素面积比例分配配额，
        采样像素 → D_c → unproject → 3D 点 + 解析法向量。

        strategy:
          - "uniform":       等面积分配，每台相机分配配额 ∝ 有效像素数
          - "visibility_weighted": 相同 — 等效于 uniform。
                            未来可扩展: 按点可见相机数量加权。
        """
        # 按有效像素比例分配采样配额
        total_valid = self._valid_pixels.sum()
        if strategy == "visibility_weighted":
            # 使用平方根权重，让大 ROI 相机权重高但不过于极端
            weights = self._valid_pixels.float().sqrt()
            weights = weights / weights.sum()
            quotas = (weights * n).long().clamp(min=1)
        else:
            quotas = (self._valid_pixels.float() / total_valid * n).long().clamp(min=1)

        # 调整使总数 ≈ n
        diff = n - quotas.sum().item()
        if diff > 0:
            # 随机分配给有容量的相机
            for _ in range(diff):
                c = torch.randint(0, self._n_cam, (1,)).item()
                quotas[c] += 1

        all_points, all_normals = [], []

        for cam_id in range(self._n_cam):
            q = quotas[cam_id].item()
            if q <= 0:
                continue

            uv = self._sample_pixels_in_mask(cam_id, q)  # (q, 2)
            if uv.numel() == 0:
                continue

            uv_norm = self._uv_to_norm(uv)

            with torch.no_grad():
                depth = self.depth_nets[cam_id](uv_norm).squeeze(-1)  # (q,)
                points_world = self._unproject(uv, depth, cam_id)     # (q, 3)
                normals = self._compute_normal(uv, cam_id)             # (q, 3)

            all_points.append(points_world)
            all_normals.append(normals)

        if not all_points:
            # Fallback: return zeros
            return (
                torch.zeros(n, 3, device=self._device),
                torch.zeros(n, 3, device=self._device),
            )

        points = torch.cat(all_points, dim=0)
        normals = torch.cat(all_normals, dim=0)

        # 打乱
        perm = torch.randperm(len(points), device=self._device)
        return points[perm][:n], normals[perm][:n]

    # ---- 可见性 ----

    def get_visible_cameras(
        self, points: torch.Tensor, max_cams: int = 3
    ) -> torch.Tensor:
        """对每个表面点在线判断在所有相机中的可见性。

        三重检查:
          1. FOV:     投影是否在图像范围内
          2. 深度自洽: |D_j(proj_j(x)) - depth_j(x)| / depth_j(x) < threshold
          3. 法向量朝向: n(x) · view_dir > cos(θ_max)
        """
        n_points = points.shape[0]
        n_cams = self._n_cam
        device = points.device

        scores = torch.full((n_points, n_cams), float("-inf"), device=device)

        with torch.no_grad():
            for cam_id in range(n_cams):
                K = self._K[cam_id]
                R = self._R[cam_id]
                t = self._t[cam_id]

                # Check 1: FOV
                uv = self.project_to_camera(points, cam_id)  # (n, 2)
                in_fov = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < self._W) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < self._H)
                )

                if not in_fov.any():
                    continue

                # Check 2: 深度自洽 (only for in-FOV points)
                uv_norm = self._uv_to_norm(uv)
                depth_query = self.depth_nets[cam_id](uv_norm).squeeze(-1)  # (n,)

                # 真实深度 (camera frame Z)
                P_cam = (R @ points.T + t.unsqueeze(-1)).T  # (n, 3)
                actual_depth = P_cam[:, 2]  # (n,)

                depth_ok = (
                    (torch.abs(depth_query - actual_depth) /
                     (actual_depth + 1e-8)) < self._depth_thresh
                )

                # Check 3: 法向量朝向
                # 使用 D_c 在投影位置的解析法向量
                # 先计算相机光心 + view direction
                R_inv = R.T
                cam_center = -(R_inv @ t)  # (3,)
                view_dir_all = cam_center.unsqueeze(0) - points  # (n, 3)
                view_dir_all = view_dir_all / (view_dir_all.norm(dim=-1, keepdim=True) + 1e-8)

                uv_fov = uv[in_fov]
                if uv_fov.shape[0] > 0:
                    normal_at_cam = self._compute_normal(uv_fov, cam_id)  # (m, 3)
                    view_dir_fov = view_dir_all[in_fov]                    # (m, 3)
                    cos_angle = (normal_at_cam * view_dir_fov).sum(dim=-1)
                    facing_ok_fov = cos_angle > self._cos_normal_thresh
                    facing_ok = torch.zeros(n_points, dtype=torch.bool, device=device)
                    facing_ok[in_fov] = facing_ok_fov
                else:
                    facing_ok = torch.zeros(n_points, dtype=torch.bool, device=device)

                valid = in_fov & depth_ok & facing_ok
                scores[:, cam_id] = torch.where(
                    valid,
                    torch.rand(n_points, device=device),  # 随机分 → 均匀从可见相机采样
                    torch.tensor(float("-inf"), device=device),
                )

        # Top max_cams
        _, top_cams = torch.topk(scores, max_cams, dim=-1)  # (n, max_cams)

        # 标记无效
        all_invalid = scores.max(dim=-1).values == float("-inf")
        top_cams[all_invalid] = -1

        return top_cams

    # ---- 投影 ----

    def project_to_camera(
        self, points: torch.Tensor, cam_id: int
    ) -> torch.Tensor:
        """World → Camera → Image 投影。纯 PyTorch，完全可微。"""
        K = self._K[cam_id]
        R = self._R[cam_id]
        t = self._t[cam_id]

        P_cam = (R @ points.T + t.unsqueeze(-1)).T  # (n, 3)
        uv_h = (K @ P_cam.T).T                      # (n, 3)
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-8)
        return uv


# =========================================================================
# 工厂函数
# =========================================================================

def create_surface_provider(
    data_dir: str,
    calib_dir: str,
    method: str = "point_cloud",
    device: str = "cuda",
    **kwargs,
) -> SurfaceProvider:
    """根据配置创建 SurfaceProvider 实例。

    路径 A (point_cloud):
      从 postprocess 输出加载:
        {calib_dir}/dense/dense_normals.npy
        {calib_dir}/dense/vis_mask.npy
      点云可以是 dense_points.npy (归一化后) 或 dense_points.ply。
      同时从 cameras.mat 加载相机参数。

    路径 B (neural_stereo):
      从 {calib_dir}/stereo_networks/ 加载训练好的 D_c 网络权重。
      同时从 cameras.mat 加载相机参数。

    Args:
        data_dir:   数据根目录
        calib_dir:  标定目录 (含 cameras.mat)
        method:     "point_cloud" | "neural_stereo"
        device:     "cuda" | "cpu"
        **kwargs:   传递给具体实现的额外参数

    Returns:
        SurfaceProvider 实例
    """
    # --- 加载相机参数 (复用 step1_pipeline 的实现，避免重复) ---
    cameras_path = os.path.join(calib_dir, "cameras.mat")
    if not os.path.exists(cameras_path):
        raise FileNotFoundError(
            f"cameras.mat not found at {cameras_path}. "
            f"Run Step 1 sparse calibration first."
        )

    from .step1_pipeline import load_calibration
    calib_data = load_calibration(calib_dir)
    K_list = calib_data["K_list"]
    R_list = calib_data["R_list"]
    t_list = calib_data["t_list"]
    n_cam = calib_data["num_cameras"]

    print(f"[SurfaceProvider] Loaded {n_cam} cameras from {cameras_path}")

    # --- 创建具体实现 ---
    if method == "point_cloud":
        return _create_point_cloud_surface(
            calib_dir=calib_dir,
            K_list=K_list,
            R_list=R_list,
            t_list=t_list,
            n_cam=n_cam,
            device=device,
            **kwargs,
        )

    elif method == "neural_stereo":
        return _create_neural_stereo_surface(
            data_dir=data_dir,
            calib_dir=calib_dir,
            K_list=K_list,
            R_list=R_list,
            t_list=t_list,
            n_cam=n_cam,
            device=device,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown surface method: {method}")


def _create_point_cloud_surface(
    calib_dir: str,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    n_cam: int,
    device: str,
    **kwargs,
) -> PointCloudSurface:
    """从后处理点云创建 PointCloudSurface。"""
    dense_dir = os.path.join(calib_dir, "dense")

    # 查找点云文件
    pts_path = os.path.join(dense_dir, "dense_points.ply")
    npy_path = os.path.join(dense_dir, "dense_points.npy")
    normals_path = os.path.join(dense_dir, "dense_normals.npy")
    vis_path = os.path.join(dense_dir, "vis_mask.npy")

    if not os.path.exists(normals_path):
        raise FileNotFoundError(
            f"dense_normals.npy not found at {normals_path}. "
            f"Run Step 1 dense with postprocess first."
        )
    if not os.path.exists(vis_path):
        raise FileNotFoundError(
            f"vis_mask.npy not found at {vis_path}. "
            f"Run Step 1 dense with postprocess first."
        )

    # Load points
    if os.path.exists(npy_path):
        points = torch.from_numpy(np.load(npy_path)).float()
        print(f"[SurfaceProvider] Loaded points from {npy_path}")
    elif os.path.exists(pts_path):
        from ndef_dic.dense_mvs import load_ply
        points, _ = load_ply(pts_path)
        points = torch.from_numpy(points)
        print(f"[SurfaceProvider] Loaded points from {pts_path}")
    else:
        raise FileNotFoundError(
            f"No point cloud found in {dense_dir}. "
            f"Expected dense_points.npy or dense_points.ply"
        )

    normals = torch.from_numpy(np.load(normals_path)).float()
    vis_mask = torch.from_numpy(np.load(vis_path)).bool()

    # Load image dimensions from meta or infer
    meta_path = os.path.join(dense_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    image_width = kwargs.get("image_width", 1440)
    image_height = kwargs.get("image_height", 1080)

    # 验证
    assert len(normals) == len(points), \
        f"Point/normal count mismatch: {len(points)} vs {len(normals)}"
    assert vis_mask.shape == (len(points), n_cam), \
        f"vis_mask shape {vis_mask.shape} != ({len(points)}, {n_cam})"

    print(f"[SurfaceProvider] PointCloudSurface: {len(points)} points, "
          f"mean vis: {vis_mask.sum(dim=1).float().mean():.1f}")

    return PointCloudSurface(
        points=points,
        normals=normals,
        vis_mask=vis_mask,
        K_list=K_list,
        R_list=R_list,
        t_list=t_list,
        image_width=image_width,
        image_height=image_height,
        device=device,
    )


def _create_neural_stereo_surface(
    data_dir: str,
    calib_dir: str,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    n_cam: int,
    device: str,
    **kwargs,
) -> NeuralStereoSurface:
    """从训练好的 D_c 网络创建 NeuralStereoSurface。"""
    from ndef_dic.pinn_stereo import DepthNetwork

    network_dir = os.path.join(calib_dir, "stereo_networks")
    if not os.path.isdir(network_dir):
        raise FileNotFoundError(
            f"Stereo networks dir not found: {network_dir}. "
            f"Run PINN-Stereo training first."
        )

    image_width = kwargs.get("image_width", 1440)
    image_height = kwargs.get("image_height", 1080)

    depth_nets = []
    for c in range(n_cam):
        pt_path = os.path.join(network_dir, f"depth_net_{c}.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Missing depth network weight: {pt_path}"
            )
        net = DepthNetwork()
        net.load_state_dict(torch.load(pt_path, map_location=device))
        depth_nets.append(net)
        print(f"[SurfaceProvider] Loaded depth_net_{c}.pt")

    # ROI masks (optional)
    masks = None
    masks_dir = os.path.join(data_dir, "masks")
    if os.path.isdir(masks_dir):
        import cv2
        masks = []
        for c in range(n_cam):
            mask_path = os.path.join(masks_dir, f"cam_{c}_mask.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                masks.append(torch.from_numpy(mask > 127).to(device))
            else:
                masks.append(None)
        print(f"[SurfaceProvider] Loaded {sum(1 for m in masks if m is not None)} ROI masks")

    print(f"[SurfaceProvider] NeuralStereoSurface: {n_cam} cameras, "
          f"{image_width}×{image_height}")

    return NeuralStereoSurface(
        depth_networks=depth_nets,
        K_list=K_list,
        R_list=R_list,
        t_list=t_list,
        image_width=image_width,
        image_height=image_height,
        masks=masks,
        depth_consistency_threshold=kwargs.get("depth_consistency_threshold", 0.05),
        normal_facing_threshold_deg=kwargs.get("normal_facing_threshold_deg", 80.0),
        device=device,
    )
