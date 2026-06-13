# Step 2：表面采样接口

**日期**: 2026-06-11
**状态**: 设计讨论
**关联**: [[step1-geometric-reconstruction]], [[neural-implicit-projection-zssd]], [[new_plan]]

---

Step 2 的职责不是"处理点云"，而是在训练好的 $D_c$ 网络（或稠密点云）之上构建一个**高效的表面采样接口**，供 Step 3 的 $\Phi(x,t)$ 训练使用。

核心洞察：

> 一旦 PINN-Stereo 的 $D_c(u,v)$ 网络训练完成，**表面就已经被隐式编码在这组网络里了**。不再需要显式的"点云 → 结构化表示"转换。

---

## 1. Step 2 在整体架构中的位置

```
Level 0: COLMAP 稀疏 SfM
  → K_c, R_c, t_c, 稀疏点 (初始化用)

Level 1: Neural Stereo {D_c}                          ← Step 1
  网络: D_c(u,v): ℝ² → ℝ⁺
  损失: 跨相机 ZNSSD
  输出: 训练好的 D_c 网络参数

Level 2: Surface Sampling Interface                   ← Step 2 (本文档)
  消费 D_c 网络 + 相机参数
  提供表面点采样、法向量查询、可见性判断
  封装为 SurfaceProvider 抽象接口

Level 3: Neural Deformation Φ                         ← Step 3
  网络: Φ(x,t): ℝ⁴ → ℝ³
  损失: 跨时间多目 ZNSSD
  表面点来源: SurfaceProvider 在线采样
  输出: 连续 3D 位移场
```

---

## 2. D_c 网络作为隐式表面表示

### 2.1 为什么 D_c 网络就是表面

D_c 训练完成后，它已经编码了完整的表面信息：

```
Step 3 需要的:                     D_c 如何提供:

① 表面点 x                        采样像素 (u,v)，查询 D_c(u,v)，
                                  unproject → x 就在表面上

② 表面法向量 n(x)                 ∂D_c/∂u, ∂D_c/∂v → 深度梯度
                                  → 通过 unproject Jacobian 转成世界法向量
                                  解析计算，比点云 PCA 更精确

③ 可见相机列表                    对 x 投影到每个相机 j:
                                  |D_j(u_j, v_j) - depth_j(x)| < ε
                                  D_j 本身就是深度参考，自洽性检查

④ 参考纹理值 (可选)               D_c 训练时的 ZNSSD 不直接给纹理
                                  但可从多相机参考图像采样后平均得到
```

### 2.2 与显式点云方案的对比

| 维度 | 显式点云 (COLMAP Dense) | D_c 隐式表面 (PINN-Stereo) |
|------|------------------------|---------------------------|
| 存储 | N×3 float32 数组 | D_c 网络参数 (~100K float32) |
| 表面点查询 | 查表 / 插值 | 网络前向传播 |
| 法向量 | PCA on KNN (近似，有噪声) | ∇D_c 解析梯度 (精确) |
| 亚点精度采样 | 需要插值 | 自然连续 |
| 可见性 | 预计算 vis_mask (离线) | 在线 D_j 深度自洽检查 |
| 与 Step 3 范式统一 | 不统一 | **完全统一** |
| 前提条件 | COLMAP Dense 成功 | D_c 网络训练收敛 |

---

## 3. SurfaceProvider 抽象接口

为了让 Step 3 的代码不依赖具体表面表示方案，定义统一接口：

```python
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import torch

class SurfaceProvider(ABC):
    """表面表示的抽象接口。

    Step 3 只依赖这个接口，不关心底层是 D_c 网络还是点云。
    """

    @abstractmethod
    def sample_surface_points(
        self, n: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样 n 个表面点。

        Args:
            n: 采样数量
            strategy: "uniform" | "visibility_weighted"

        Returns:
            points: (n, 3) 世界坐标系中的 3D 坐标
            normals: (n, 3) 单位法向量
        """
        ...

    @abstractmethod
    def get_visible_cameras(
        self, points: torch.Tensor, max_cams: int = 3
    ) -> torch.Tensor:
        """
        对每个表面点返回可见相机索引。

        Args:
            points: (n, 3)
            max_cams: 每个点最多返回几个相机

        Returns:
            cam_indices: (n, max_cams) int, -1 表示无效
        """
        ...

    @abstractmethod
    def project_to_camera(
        self, points: torch.Tensor, cam_id: int
    ) -> torch.Tensor:
        """
        将 3D 表面点投影到指定相机。

        Args:
            points: (n, 3)
            cam_id: 相机索引

        Returns:
            uv: (n, 2) 像素坐标
        """
        ...

    @property
    @abstractmethod
    def num_cameras(self) -> int:
        """返回相机数量"""
        ...

    @property
    @abstractmethod
    def bbox(self) -> torch.Tensor:
        """返回世界坐标系中的包围盒 (2, 3) [min, max]"""
        ...
```

### 3.1 实现 A：NeuralStereoSurface（D_c 网络方案）

```python
class NeuralStereoSurface(SurfaceProvider):
    """
    基于训练好的 D_c 深度网络的隐式表面表示。

    表面点通过采样像素坐标 + 查询 D_c 网络 + 反投影获得。
    法向量通过 D_c 的解析梯度计算。
    可见性通过 D_j 深度自洽性在线判断。
    """

    def __init__(
        self,
        depth_networks: List[torch.nn.Module],  # D_c per camera
        K_list: List[torch.Tensor],
        R_list: List[torch.Tensor],
        t_list: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],     # (H, W) per camera
        masks: Optional[List[torch.Tensor]] = None,
        depth_consistency_threshold: float = 0.05,
        normal_facing_threshold_deg: float = 80.0,
    ):
        self.depth_nets = depth_networks
        self.K = K_list
        self.R = R_list
        self.t = t_list
        self.image_shapes = image_shapes
        self.masks = masks
        self.depth_threshold = depth_consistency_threshold
        self.cos_normal_threshold = math.cos(
            math.radians(normal_facing_threshold_deg)
        )

    # ... 方法实现在下面展开
```

### 3.2 实现 B：PointCloudSurface（COLMAP 点云方案）

```python
class PointCloudSurface(SurfaceProvider):
    """
    基于 COLMAP Dense 点云的表面表示（备选方案）。

    当 PINN-Stereo 不可用或作为对比基线时使用。
    """

    def __init__(
        self,
        points: torch.Tensor,           # (N, 3)
        normals: torch.Tensor,          # (N, 3)
        vis_mask: torch.Tensor,         # (N, N_cam) bool
        K_list: List[torch.Tensor],
        R_list: List[torch.Tensor],
        t_list: List[torch.Tensor],
    ):
        self.points = points
        self.normals = normals
        self.vis_mask = vis_mask
        self.K = K_list
        self.R = R_list
        self.t = t_list

    # ... 查表式 sample / get_visible_cameras / project
```

**两种实现的切换**：Step 3 的代码只 import `SurfaceProvider`，通过工厂函数根据配置选择具体实现。

---

## 4. 核心操作：表面点采样

### 4.1 NeuralStereoSurface 的采样流程

```
sample_surface_points(n):

  1. 按每台相机的有效区域面积比例分配采样配额
     n_c ∝ mask_c 的有效像素数

  2. 对每台相机 c:
     a) 在 mask_c 有效区域内随机采样像素 (u, v)
     b) 查询 d = D_c(u, v)
     c) 反投影: x_world = unproject(u, v, d, K_c, R_c, t_c)
     d) 计算法向量 n = compute_normal(u, v, D_c, K_c, R_c)
        (见 §5)

  3. 汇总所有相机的采样结果，随机打乱

  4. 返回 (x_world, n)
```

```python
def sample_surface_points(
    self, n: int, strategy: str = "uniform"
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = self.K[0].device
    n_cams = len(self.depth_nets)

    # 计算每台相机的采样配额
    if self.masks is not None:
        valid_pixels = torch.tensor([
            (m > 0.5).sum().item() for m in self.masks
        ])
    else:
        valid_pixels = torch.tensor([
            h * w for h, w in self.image_shapes
        ])
    quotas = (valid_pixels / valid_pixels.sum() * n).long()
    quotas = torch.clamp(quotas, min=1)

    all_points, all_normals = [], []

    for cam_id, quota in enumerate(quotas):
        # 在 mask 内采样像素
        H, W = self.image_shapes[cam_id]
        if self.masks is not None:
            mask = self.masks[cam_id]
            valid_idx = torch.where(mask > 0.5)
            rand_idx = torch.randint(0, len(valid_idx[0]), (quota,))
            u = valid_idx[1][rand_idx].float()  # col
            v = valid_idx[0][rand_idx].float()  # row
        else:
            u = torch.rand(quota) * W
            v = torch.rand(quota) * H

        uv = torch.stack([u, v], dim=-1).to(device)

        # 查询深度
        depth = self.depth_nets[cam_id](uv)  # (quota, 1)

        # 反投影到世界坐标系
        points_world = self._unproject(uv, depth, cam_id)

        # 计算法向量
        normals = self._compute_normal(uv, cam_id)

        all_points.append(points_world)
        all_normals.append(normals)

    points = torch.cat(all_points, dim=0)
    normals = torch.cat(all_normals, dim=0)

    # 打乱
    perm = torch.randperm(len(points))
    return points[perm][:n], normals[perm][:n]
```

### 4.2 采样策略

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| `uniform` | 等概率采样所有有效像素 | 通用，无偏 |
| `visibility_weighted` | 优先采样被更多相机看到的点 | 约束更多 → 梯度更强 |
| `gradient_aware` | 优先采样图像梯度大的区域 | 聚焦纹理丰富区域 |

**默认使用 `uniform`**，简单且无偏。`visibility_weighted` 可以作为后续改进。

---

## 5. 核心操作：法向量计算

### 5.1 从 D_c 解析计算法向量

这是 D_c 方案相比点云 PCA 的显著优势——法向量是**解析梯度**，不是近邻估计。

```
世界坐标系中的表面法向量 n(x):

  给定像素 (u,v) 和深度 d = D_c(u,v)

  Step 1: 计算 3D 点对像素坐标的偏导
    ∂x/∂u, ∂x/∂v   (通过 unproject 的 Jacobian)

  Step 2: 法向量 = (∂x/∂u × ∂x/∂v) / |∂x/∂u × ∂x/∂v|
    即表面参数化的两个切向量的叉积
```

### 5.2 推导

3D 点 x 由像素 (u,v) 和深度 d 决定：

$$x = R_c^{-1} \left( d \cdot K_c^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - t_c \right)$$

其中 $d = D_c(u,v)$，所以 $d$ 是 $(u,v)$ 的函数。

切向量：

$$\frac{\partial x}{\partial u} = R_c^{-1} K_c^{-1} \left( d \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + \frac{\partial d}{\partial u} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \right)$$

$$\frac{\partial x}{\partial v} = R_c^{-1} K_c^{-1} \left( d \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} + \frac{\partial d}{\partial v} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \right)$$

法向量：

$$n = \frac{\frac{\partial x}{\partial u} \times \frac{\partial x}{\partial v}}{\left\| \frac{\partial x}{\partial u} \times \frac{\partial x}{\partial v} \right\|}$$

### 5.3 实现

```python
def _compute_normal(
    self, uv: torch.Tensor, cam_id: int
) -> torch.Tensor:
    """
    从 D_c 解析计算世界坐标系中的表面法向量。

    使用 autograd 计算 ∂d/∂u 和 ∂d/∂v，
    然后通过 unproject Jacobian 转换到世界坐标系。
    """
    u, v = uv[:, 0:1], uv[:, 1:2]  # 需要保留梯度

    u.requires_grad_(True)
    v.requires_grad_(True)

    uv_grad = torch.cat([u, v], dim=-1)
    depth = self.depth_nets[cam_id](uv_grad)

    # ∂d/∂u, ∂d/∂v
    grad_u = torch.autograd.grad(
        depth, u, grad_outputs=torch.ones_like(depth),
        create_graph=False, retain_graph=True
    )[0]
    grad_v = torch.autograd.grad(
        depth, v, grad_outputs=torch.ones_like(depth),
        create_graph=False, retain_graph=False
    )[0]

    # Unproject Jacobian
    K_inv = torch.inverse(self.K[cam_id])
    R_inv = torch.inverse(self.R[cam_id])
    ray_dir = K_inv @ torch.tensor([0., 0., 1.])  # 光轴方向在相机系

    # ∂x/∂u, ∂x/∂v
    dx_du = R_inv @ K_inv[:, 0:1] * depth + R_inv @ ray_dir * grad_u
    dx_dv = R_inv @ K_inv[:, 1:2] * depth + R_inv @ ray_dir * grad_v

    # n = dx_du × dx_dv
    normal = torch.cross(dx_du, dx_dv, dim=-1)
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-8)

    # 方向一致性: 法向量应指向相机平均方向
    cam_centers = -R_inv @ self.t[cam_id]  # 世界坐标系中的相机光心
    view_dir = cam_centers - self._unpoint(uv, depth, cam_id)
    flip = (normal * view_dir).sum(dim=-1) < 0
    normal[flip] = -normal[flip]

    return normal.detach()
```

**注意**：这里每个点需要一次 autograd 来获取 $\partial d / \partial (u,v)$。对于 batch 采样，可以用 `torch.vmap` 或在 D_c 网络中用 `torch.func.grad` 批量计算。如果性能成为瓶颈，可以用有限差分近似（在像素空间中取邻域点查询深度）。

---

## 6. 核心操作：可见性判断

### 6.1 在线可见性

与点云方案（预计算 `vis_mask`）不同，D_c 方案可以**在线判断**可见性：

```python
def get_visible_cameras(
    self, points: torch.Tensor, max_cams: int = 3
) -> torch.Tensor:
    """
    对每个表面点判断在所有相机中的可见性。

    三重检查:
      1. FOV: 投影是否在图像范围内
      2. 深度自洽: |D_j(proj_j(x)) - depth_j(x)| < threshold
      3. 法向量朝向: n(x) · view_dir > cos(θ_max)
    """
    n_points = points.shape[0]
    n_cams = len(self.depth_nets)
    device = points.device

    scores = torch.full((n_points, n_cams), float('-inf'))

    for cam_id in range(n_cams):
        K, R, t = self.K[cam_id], self.R[cam_id], self.t[cam_id]
        H, W = self.image_shapes[cam_id]

        # 投影
        uv = self.project_to_camera(points, cam_id)  # (n, 2)
        u, v = uv[:, 0], uv[:, 1]

        # Check 1: FOV
        in_fov = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Check 2: 深度自洽
        # 查询 D_c 在该像素的深度
        depth_cam = self.depth_nets[cam_id](uv)  # (n, 1)
        # 计算 x 到相机 c 的实际距离
        P_cam = R @ points.T + t  # (3, n)
        actual_depth = P_cam[2, :]  # Z 分量 = 深度
        depth_ok = (
            torch.abs(depth_cam.squeeze() - actual_depth) /
            (actual_depth + 1e-8)
        ) < self.depth_threshold

        # Check 3: 法向量朝向
        cam_center = -torch.inverse(R) @ t  # 世界坐标系
        view_dir = cam_center - points
        view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)
        cos_angle = (self._normals * view_dir).sum(dim=-1)
        facing_ok = cos_angle > self.cos_normal_threshold

        # 综合
        valid = in_fov & depth_ok & facing_ok
        scores[:, cam_id] = torch.where(
            valid,
            torch.rand(n_points, device=device),  # 随机分数用于均匀采样
            torch.tensor(float('-inf'))
        )

    # 对每个点选 max_cams 个最高分相机
    _, top_cams = torch.topk(scores, max_cams, dim=-1)

    # 标记无效 (所有相机都不可见的点)
    all_invalid = scores.max(dim=-1).values == float('-inf')
    top_cams[all_invalid] = -1

    return top_cams  # (n_points, max_cams)
```

### 6.2 可见性检查的精度讨论

| 检查项 | 精度 | 计算成本 |
|--------|------|---------|
| FOV | 精确 (矩阵运算) | 极低 |
| 深度自洽 | 依赖 D_c 精度 | 中等 (一次 D_c 前向) |
| 法向量朝向 | 依赖 D_c 梯度精度 | 低 (已有 n) |

对于 DIC 试样表面（连续、无非凸遮挡），FOV + 法向量朝向通常就足够。深度自洽检查可以设为可选——当 D_c 训练充分后开启，作为额外的遮挡过滤。

---

## 7. 核心操作：投影

```python
def project_to_camera(
    self, points: torch.Tensor, cam_id: int
) -> torch.Tensor:
    """
    将世界坐标系中的 3D 点投影到指定相机。

    Args:
        points: (n, 3)
        cam_id: 相机索引

    Returns:
        uv: (n, 2) 像素坐标 (col, row)
    """
    K = self.K[cam_id]
    R = self.R[cam_id]
    t = self.t[cam_id]

    # World → Camera
    P_cam = (R @ points.T + t).T  # (n, 3)

    # Camera → Image
    uv_h = (K @ P_cam.T).T       # (n, 3)
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    return uv
```

此函数完全可微，在 Step 3 中会大量使用。对于 NeuralStereoSurface，还需要一个对应的反向操作：

```python
def _unproject(self, uv, depth, cam_id):
    """像素 + 深度 → 世界坐标系 3D 点"""
    K_inv = torch.inverse(self.K[cam_id])
    R_inv = torch.inverse(self.R[cam_id])
    t = self.t[cam_id]

    ray_cam = (K_inv @ torch.cat([
        uv.T, torch.ones(1, uv.shape[0], device=uv.device)
    ], dim=0)).T  # (n, 3)
    P_cam = ray_cam * depth    # (n, 3)
    P_world = (R_inv @ (P_cam.T - t)).T  # (n, 3)
    return P_world
```

---

## 8. 与 Step 1 的边界

### 8.1 两种路径下的文件组织

```
路径 A: COLMAP Dense (成熟)

  Step 1 输出:
    calibration/
    ├── cameras.mat
    ├── dense/
    │   ├── dense_points.ply      → PointCloudSurface 加载
    │   ├── depth_maps/
    │   └── normal_maps/

  Step 2 处理:
    calibration/surface/
    ├── points_norm.npy           # (N, 3) 归一化坐标
    ├── normals.npy               # (N, 3) 法向量
    ├── vis_mask.npy              # (N, N_cam) 布尔
    ├── transform.json            # 归一化参数
    └── meta.json


路径 B: PINN-Stereo (神经隐式)

  Step 1 输出:
    calibration/
    ├── cameras.mat
    └── stereo_networks/
        ├── depth_net_0.pt        # D_0 网络权重
        ├── depth_net_1.pt
        ├── depth_net_2.pt
        └── ...

  Step 2 处理:
    无持久化输出——直接加载 D_c 网络 + 相机参数，
    构建 NeuralStereoSurface 实例。
```

### 8.2 工厂函数

```python
def create_surface_provider(
    data_dir: str, method: str = "neural_stereo"
) -> SurfaceProvider:
    """
    根据配置创建 SurfaceProvider 实例。

    Args:
        data_dir: 数据根目录
        method: "neural_stereo" | "point_cloud"
    """
    calib_dir = os.path.join(data_dir, "calibration")

    # 加载相机参数 (两种方法共用)
    cameras = load_camera_params(calib_dir)

    if method == "neural_stereo":
        depth_nets = load_depth_networks(
            os.path.join(calib_dir, "stereo_networks")
        )
        masks = load_masks(data_dir)
        return NeuralStereoSurface(
            depth_networks=depth_nets,
            K_list=[c.K for c in cameras],
            R_list=[c.R for c in cameras],
            t_list=[c.t for c in cameras],
            image_shapes=[(c.height, c.width) for c in cameras],
            masks=masks,
        )

    elif method == "point_cloud":
        surface_dir = os.path.join(calib_dir, "surface")
        points = torch.from_numpy(np.load(
            os.path.join(surface_dir, "points_norm.npy")
        ))
        normals = torch.from_numpy(np.load(
            os.path.join(surface_dir, "normals.npy")
        ))
        vis_mask = torch.from_numpy(np.load(
            os.path.join(surface_dir, "vis_mask.npy")
        ))
        return PointCloudSurface(
            points=points, normals=normals,
            vis_mask=vis_mask,
            K_list=[c.K for c in cameras],
            R_list=[c.R for c in cameras],
            t_list=[c.t for c in cameras],
        )

    else:
        raise ValueError(f"Unknown surface method: {method}")
```

---

## 9. 实现清单

```
□ 2.1 定义 SurfaceProvider 抽象接口
     - sample_surface_points()
     - get_visible_cameras()
     - project_to_camera()
     - 属性: num_cameras, bbox

□ 2.2 实现 NeuralStereoSurface
     - _unproject(): 像素+深度 → 3D
     - _compute_normal(): ∇D_c × unproject Jacobian
     - sample_surface_points(): 像素采样 → D_c → unproject
     - get_visible_cameras(): FOV + 深度自洽 + 法向量朝向
     - project_to_camera(): 3D → 2D 投影

□ 2.3 实现 PointCloudSurface (备选)
     - 从 .npy 加载预计算的点云、法向量、vis_mask
     - sample_surface_points(): 随机/加权查表
     - get_visible_cameras(): 查 vis_mask
     - project_to_camera(): 同上

□ 2.4 实现工厂函数 create_surface_provider()
     - 根据配置选择具体实现
     - 自动加载所需的文件

□ 2.5 单元测试
     - 采样点都在图像范围内
     - 法向量模长 ≈ 1 且方向一致
     - 可见相机数 ≥ 1
     - 投影-反投影自洽性: |unproject(project(x), D(project(x))) - x| < ε
```

---

## 10. 参考资料

- [[step1-geometric-reconstruction]] — Step 1 几何重建
- [[neural-implicit-projection-zssd]] — 神经隐式 + 投影约束 + ZNSSD
- [[new_plan]] — NDF-DIC 项目整体设计
