# Step 2 ↔ Step 3 接口对接

**日期**: 2026-06-11
**关联**: [[step2-surface-representation]], [[step3-neural-deformation-field]]

---

本文档精确描述 Step 2（SurfaceProvider）和 Step 3（Φ 网络训练）之间的接口约定和数据流。

---

## 1. 接口总览

```
┌──────────────────────────────────────────────────────────────┐
│                    Step 3: Training Loop                      │
│                                                              │
│  for iteration in range(max_iters):                          │
│                                                              │
│    # ① 从 Step 2 获取表面点                                    │
│    x, normals = surface.sample_surface_points(M)             │
│                                                              │
│    # ② 从 Step 2 获取可见性                                    │
│    cam_ids = surface.get_visible_cameras(x, max_cams=K)      │
│                                                              │
│    # ③ 计算变形场                                              │
│    phi = deformation_net(x, t)                               │
│    x_def = x + phi                                           │
│                                                              │
│    # ④ 通过 Step 2 投影                                       │
│    uv_ref = surface.project_to_camera(x,      cam_id)        │
│    uv_def = surface.project_to_camera(x_def,  cam_id)        │
│                                                              │
│    # ⑤ 从 Dataset 取图像 patch                                │
│    P_ref = dataset.extract_patch(cam_id, uv_ref, size)       │
│    P_def = dataset.extract_patch(cam_id, uv_def, size)       │
│                                                              │
│    # ⑥ 计算损失                                               │
│    loss = ZNSSD(P_ref, P_def) + λ * L_smooth(phi, x)        │
│    loss.backward()                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Step 2 的职责                         Dataset 的职责
─────────────────                     ──────────────
- 提供"训练哪里"                       - 提供"图像数据"
- surface.sample_surface_points()     - ref_images[c]
- surface.get_visible_cameras()       - def_images[c][t]
- surface.project_to_camera()         - extract_patch()
```

---

## 2. SurfaceProvider 接口（Step 2 暴露的）

```python
class SurfaceProvider(ABC):
    """Step 2 提供给 Step 3 的唯一接口。"""

    # ── 采样 ──────────────────────────────────────────

    @abstractmethod
    def sample_surface_points(
        self, n: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        → points:  (n, 3)  世界坐标系，归一化坐标
        → normals: (n, 3)  单位法向量
        """
        ...

    # ── 可见性 ────────────────────────────────────────

    @abstractmethod
    def get_visible_cameras(
        self, points: torch.Tensor, max_cams: int = 3
    ) -> torch.Tensor:
        """
        → cam_ids: (n, max_cams) int, -1 表示无效

        用法:
          cam_ids = surface.get_visible_cameras(x, max_cams=3)
          for cam in range(num_cameras):
              mask = (cam_ids == cam).any(dim=-1)
              x_cam = x[mask]
              ...
        """
        ...

    # ── 投影 ──────────────────────────────────────────

    @abstractmethod
    def project_to_camera(
        self, points: torch.Tensor, cam_id: int
    ) -> torch.Tensor:
        """
        → uv: (n, 2) 像素坐标 (col, row)

        作用在 Φ 网络的输出路径上 → 梯度可反传
        """
        ...

    # ── 元信息 ────────────────────────────────────────

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
```

---

## 3. 训练循环的完整实现

```python
class DeformationFieldTrainer:
    """
    Step 3 的训练器。
    通过 SurfaceProvider 接口消费 Step 2 的输出。
    """

    def __init__(
        self,
        surface: SurfaceProvider,               # ← Step 2 的接口
        deformation_net: torch.nn.Module,       # Φ(x,t) 网络
        dataset: MultiCamDataset,               # 图像数据
        config: Dict,
    ):
        self.surface = surface
        self.deformation_net = deformation_net
        self.dataset = dataset

        # 训练超参数
        self.M = config.get("batch_size", 1024)          # 表面点数
        self.K = config.get("cameras_per_point", 3)      # 每点相机数
        self.patch_size = config.get("patch_size", 32)   # 当前阶段 patch
        self.lambda_smooth = config.get("lambda_smooth", 1e-2)

    # ================================================================
    # 主训练循环
    # ================================================================

    def train_step(self, t: float, optimizer: torch.optim.Optimizer):
        """
        一次训练迭代。

        Args:
            t: 归一化时间 [0, 1]
        """
        # ──── ① 从 Step 2 获取训练点 ─────────────────────
        # 这是 Step 2 → Step 3 的入口
        x, normals = self.surface.sample_surface_points(
            self.M, strategy="uniform"
        )
        # x:     (M, 3) 世界坐标 (归一化)
        # normals: (M, 3) 法向量
        # 注意: x.requires_grad 应保持为 True (或在此设置)

        # ──── ② 获取可见性 ──────────────────────────────
        cam_ids = self.surface.get_visible_cameras(x, max_cams=self.K)
        # cam_ids: (M, K)  int in [0, N_cams-1] or -1

        # ──── ③ 计算变形 ────────────────────────────────
        phi = self.deformation_net(x, t)   # (M, 3)
        x_def = x + phi                     # (M, 3)

        # ──── ④⑤⑥ 逐相机计算 ZNSSD ─────────────────────
        total_znssd = 0.0
        valid_pairs = 0

        for cam_id in range(self.surface.num_cameras):
            # 找到此相机可见的点
            cam_mask = (cam_ids == cam_id).any(dim=-1)  # (M,)
            n_visible = cam_mask.sum().item()
            if n_visible == 0:
                continue

            x_c = x[cam_mask]           # 子集
            x_def_c = x_def[cam_mask]

            # ④ 投影 (通过 Step 2 接口)
            uv_ref = self.surface.project_to_camera(x_c,     cam_id)
            uv_def = self.surface.project_to_camera(x_def_c, cam_id)
            # 注意: project_to_camera 内部使用 K_c, R_c, t_c
            #       这是硬约束的体现——深度不是自由变量，x 和 x_def 是 3D 点

            # ⑤ 提取图像 patch (从 Dataset)
            P_ref = self._extract_patch(cam_id, uv_ref, self.patch_size)
            P_def = self._extract_patch(
                cam_id, uv_def, self.patch_size,
                is_deformed=True, t=t
            )

            # ⑥ 计算 ZNSSD
            znssd = self._znssd(P_ref, P_def, eps=1e-6)
            total_znssd += znssd * n_visible
            valid_pairs += n_visible

        # ──── ⑦ 汇总损失 ────────────────────────────────
        L_dic = total_znssd / max(valid_pairs, 1)
        L_smooth = self._compute_smoothness(phi, x)
        L_total = L_dic + self.lambda_smooth * L_smooth

        # ──── ⑧ 反传 ────────────────────────────────────
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        return {
            "L_dic": L_dic.item(),
            "L_smooth": L_smooth.item(),
            "L_total": L_total.item(),
            "valid_pairs": valid_pairs,
            "grad_norm": self._compute_grad_norm(),
        }

    # ================================================================
    # 辅助方法
    # ================================================================

    def _extract_patch(
        self, cam_id: int, uv: torch.Tensor,
        patch_size: int, is_deformed: bool = False, t: float = 0.0
    ) -> torch.Tensor:
        """
        从图像中提取 w×w 的 patch。

        uv: (n, 2) 像素坐标 (col, row)，范围 [0, W) × [0, H)
        Returns: (n, w, w)
        """
        H, W = self.dataset.image_height, self.dataset.image_width

        # grid_sample 需要的坐标格式: [-1, 1]
        uv_norm = torch.empty_like(uv)
        uv_norm[:, 0] = 2.0 * uv[:, 0] / (W - 1) - 1.0  # col → x
        uv_norm[:, 1] = 2.0 * uv[:, 1] / (H - 1) - 1.0  # row → y
        # grid_sample 的坐标顺序是 (x, y) 即 (col, row)
        grid = uv_norm.unsqueeze(1).unsqueeze(1)  # (n, 1, 1, 2)

        if is_deformed:
            image = self.dataset.get_def_image(cam_id, step=int(t))
        else:
            image = self.dataset.get_ref_image(cam_id)

        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # 对每个点独立提取 patch: 通过构造独立的采样网格
        # 方法: 为每个 uv 中心构造一个 w×w 的局部网格
        patches = self._grid_sample_patches(image, grid, patch_size)
        return patches  # (n, w, w)

    def _grid_sample_patches(
        self, image: torch.Tensor,
        centers: torch.Tensor,  # (n, 1, 1, 2)
        patch_size: int,
    ) -> torch.Tensor:
        """
        围绕每个中心点提取 w×w 的 patch。

        对每个 uv 中心:
          生成以 (u,v) 为中心的 w×w 均匀采样网格
          grid_sample 对这些位置做双线性插值

        Returns: (n, w, w)
        """
        n = centers.shape[0]
        H, W = self.dataset.image_height, self.dataset.image_width

        # 像素空间中的偏移量
        half = (patch_size - 1) / 2.0
        offsets = torch.linspace(-half, half, patch_size)  # (w,)

        du, dv = torch.meshgrid(offsets, offsets, indexing="xy")
        # du, dv: (w, w)

        # 中心坐标 + 偏移 = 每个 patch 像素的坐标
        centers_pix = centers.squeeze(1).squeeze(1)  # (n, 2)
        # centers_pix[:, 0] = u (col), centers_pix[:, 1] = v (row)

        # 构造 (n, w, w, 2) 的采样网格
        grid_u = centers_pix[:, 0:1, None, None] + du[None, :, :]  # (n, 1, w, w)
        grid_v = centers_pix[:, 1:2, None, None] + dv[None, :, :]

        # 归一化到 [-1, 1]
        grid_u = 2.0 * grid_u / (W - 1) - 1.0
        grid_v = 2.0 * grid_v / (H - 1) - 1.0

        grid = torch.stack([grid_u, grid_v], dim=-1)  # (n, w, w, 2)

        # 批量 grid_sample
        # image: (1, 1, H, W) → 扩展到 (n, 1, H, W)
        image_batch = image.expand(n, -1, -1, -1)
        patches = F.grid_sample(
            image_batch, grid,
            mode='bilinear', padding_mode='zeros',
            align_corners=True,
        )
        # patches: (n, 1, w, w) → (n, w, w)
        return patches.squeeze(1)

    def _znssd(self, P_ref, P_def, eps=1e-6):
        """
        Zero-mean Normalized SSD。

        P_ref, P_def: (n, w, w)
        Returns: scalar
        """
        n = P_ref.shape[0]
        patch_flat_ref = P_ref.view(n, -1)  # (n, w²)
        patch_flat_def = P_def.view(n, -1)

        # 零均值 + 归一化
        mu_ref = patch_flat_ref.mean(dim=-1, keepdim=True)
        mu_def = patch_flat_def.mean(dim=-1, keepdim=True)
        sigma_ref = patch_flat_ref.std(dim=-1, keepdim=True) + eps
        sigma_def = patch_flat_def.std(dim=-1, keepdim=True) + eps

        P_ref_norm = (patch_flat_ref - mu_ref) / sigma_ref
        P_def_norm = (patch_flat_def - mu_def) / sigma_def

        # SSD of normalized patches
        # ZNSSD ∈ [0, 4] (理论上)
        znssd = ((P_ref_norm - P_def_norm) ** 2).sum(dim=-1)  # (n,)
        return znssd.mean()

    def _compute_smoothness(self, phi, x):
        """L_smooth = ||∇Φ||_F²"""
        grad_u = torch.autograd.grad(
            phi[:, 0].sum(), x, create_graph=True, retain_graph=True
        )[0]
        grad_v = torch.autograd.grad(
            phi[:, 1].sum(), x, create_graph=True, retain_graph=True
        )[0]
        grad_w = torch.autograd.grad(
            phi[:, 2].sum(), x, create_graph=False
        )[0]

        # ||∇Φ||_F² = Σ (∂Φ_i/∂x_j)²
        smooth = (grad_u ** 2).sum() + (grad_v ** 2).sum() + (grad_w ** 2).sum()
        return smooth / phi.shape[0]
```

---

## 4. 关键数据流

### 4.1 梯度路径：Φ 网络 → ZNSSD

```
x ──→ deformation_net ──→ phi ──→ x_def = x + phi
                                       │
                         surface.project_to_camera(x_def)
                                       │
                              uv_def (pixel coords)
                                       │
                         grid_sample(def_image, uv_def)
                                       │
                              P_def (patch pixels)
                                       │
                         ZNSSD(P_ref, P_def)
                                       │
                              loss.backward()
                                       │
         梯度通过 project_to_camera 反传
         梯度通过 x_def = x + phi 反传
         梯度进入 deformation_net 参数
```

**关键**：`surface.project_to_camera()` 必须在计算图中保持可微。它内部是：
```python
P_cam = R @ points.T + t    # 线性 → 可微
uv = K @ P_cam              # 线性 → 可微
uv = uv[:2] / uv[2]         # 除法 → 可微 (autograd 自动处理)
```

### 4.2 为什么 project_to_camera 不经过 D_c

一个容易混淆的点：Step 3 用 `project_to_camera(x_def)` 时，这里的 `x_def` 已经是世界坐标系中的 3D 点了（来自 x + Φ(x,t)）。投影只需要相机参数 (K, R, t)——不需要查询深度网络 D_c。

```
Step 3 的投影:
  x_def (已知的 3D 点) ──→ project(已知的相机参数) ──→ uv_def
  不需要深度！因为我们要投影的是一个已知的 3D 点

Step 1 的投影（对比）:
  (u,v) (像素) + d (深度) ──→ unproject ──→ 3D 点 ──→ project ──→ 另一个相机
  需要深度！因为我们要从 2D 恢复 3D
```

只有在 Step 1（训练 D_c）时，才需要通过深度把 2D 变成 3D。Step 3 中，3D 点已经已知——它就是 x + Φ(x,t)。

---

## 5. 两种 SurfaceProvider 实现的差异

```python
# ── NeuralStereoSurface ──
# sample_surface_points:
#   采样 (u,v) → D_c(u,v) → unproject → x
#   x 是"在线生成"的，不是从预存数组中取的
#   法向量是从 ∇D_c 解析计算的

# ── PointCloudSurface ──
# sample_surface_points:
#   从预存的 points.npy 中随机索引 → x
#   法向量从预存的 normals.npy 中读取
#   get_visible_cameras: 查预存的 vis_mask.npy

# 但对 Step 3 来说，这两种实现完全透明——
# project_to_camera() 的行为完全相同，因为投影只需要 K, R, t
```

**唯一的性能差异**：`NeuralStereoSurface.sample_surface_points()` 每次需要做 D_c 前向传播，而 `PointCloudSurface` 是查表。但这在整体训练开销中占比很小（D_c 是小型 MLP，1024 个点只需 ~1ms）。

---

## 6. Dataset 的职责边界

Step 2 不持有图像数据。图像数据由 `MultiCamDataset` 管理：

```python
# Dataset 提供的方法 (供 Step 3 调用)

class MultiCamDataset:
    def get_ref_image(cam_id) -> torch.Tensor:
        """返回 (H, W) 参考图像 tensor"""
        ...

    def get_def_image(cam_id, step) -> torch.Tensor:
        """返回 (H, W) 变形图像 tensor"""
        ...

    def get_camera_params(cam_id) -> CameraParams:
        """返回 K, R, t, dist"""
        ...
```

分工：
- **SurfaceProvider**：知道"物体的几何"——点在哪、从哪里能看到
- **Dataset**：知道"图像数据"——图像像素值、相机参数的值
- **DeformationFieldTrainer**：协调两者，驱动训练

---

## 7. 完整文件结构

```
ndef_dic/
├── colmap_calib.py          # Step 0-1: COLMAP SfM
├── neural_stereo.py         # Step 1: D_c 网络训练 (PINN-Stereo)
│
├── surface_provider.py      # Step 2: SurfaceProvider 接口 + 两种实现
│   ├── SurfaceProvider      (ABC)
│   ├── NeuralStereoSurface  (D_c 网络方案)
│   └── PointCloudSurface    (COLMAP 点云方案)
│
├── deformation_net.py       # Step 3: Φ(x,t) 网络定义
│   ├── HashGridEncoder
│   ├── TemporalEncoder
│   └── DeformationNetwork   (Φ(x,t))
│
├── dic_losses.py            # Step 1 & 3 共用: ZNSSD, patch extraction
│   ├── znssd()
│   ├── extract_patches()
│   └── deformation_smoothness()
│
├── trainer.py               # Step 3: 训练循环
│   └── DeformationFieldTrainer
│
└── dataset.py               # 图像数据 + 相机参数加载
    └── MultiCamDataset
```

---

## 8. 初始化流程

```python
# 完整的初始化序列

def build_ndef_dic_pipeline(data_dir, config):
    """
    按顺序构建整个 NDF-DIC pipeline。
    """

    # ── Step 0/1 的输出: 相机参数 ──────────────────────
    cameras = load_camera_params(os.path.join(data_dir, "calibration"))
    # → List[CameraParams] (K, R, t, dist, width, height)

    # ── Step 1: 训练或加载深度网络 ─────────────────────
    if config["surface_method"] == "neural_stereo":
        depth_nets = train_or_load_depth_networks(data_dir, cameras, config)
        # → List[torch.nn.Module] (每台相机一个 D_c)
    else:
        depth_nets = None

    # ── Step 2: 构建表面采样接口 ──────────────────────
    surface = create_surface_provider(
        data_dir=data_dir,
        cameras=cameras,
        depth_networks=depth_nets,
        method=config["surface_method"],
    )
    # → SurfaceProvider (NeuralStereoSurface 或 PointCloudSurface)

    # ── Dataset: 图像数据 ─────────────────────────────
    dataset = MultiCamDataset(
        data_dir=data_dir,
        n_cameras=len(cameras),
        image_height=cameras[0].height,
        image_width=cameras[0].width,
        device=config["device"],
    )

    # ── Step 3: 变形场网络 + 训练器 ───────────────────
    deformation_net = DeformationNetwork(config["deformation"])

    trainer = DeformationFieldTrainer(
        surface=surface,           # ← Step 2 的接口
        deformation_net=deformation_net,
        dataset=dataset,
        config=config["training"],
    )

    return trainer
```

---

## 9. 总结：三个关键约定

1. **坐标约定**：SurfaceProvider 输出的 `points` 是世界坐标系（归一化）。`project_to_camera` 内部使用 `K, R, t` 完成 World → Camera → Image 的转换。Step 3 不需要知道坐标归一化的存在。

2. **可微性约定**：`project_to_camera` 必须保持计算图的梯度连接。它内部是线性代数 + 齐次除法，天然可微。

3. **职责约定**：SurfaceProvider 管几何，Dataset 管图像。两者不互持引用。Step 3 的 Trainer 是唯一的协调者。
