# NDF-DIC — Neural Deformation Field for Multi-View DIC

**用神经隐式变形场直接从多目散斑图像中恢复连续 3D 位移场。**

---

## 核心思想

传统 DIC：图像对 → 逐像素相关 → 位移  
NDF-DIC：多目图像 → Φ(x,t) → 投影 → 多目 ZNSSD → 连续位移场

```
Multi-view Images
    → Neural Deformation Field Φ(x,t): ℝ⁴ → ℝ³
    → Projection through camera models
    → Multi-view ZNSSD (photometric consistency)
    → Continuous 3D displacement (u, v, w)
```

核心损失函数是 **多目 ZNSSD**（零均值归一化平方差之和），不是图像渲染 loss。
网络学习 `u(x,t), v(x,t), w(x,t)`，不学 radiance/density/color。

---

## 架构：三阶段流水线

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Geometric Reconstruction                                │
│                                                                  │
│  COLMAP SfM → K,R,t + sparse points                             │
│  PatchMatch Stereo → dense point cloud                          │
│  or PINN-Stereo → per-camera depth networks D_c(u,v)            │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: SurfaceProvider (surface_provider.py)                   │
│                                                                  │
│  sample_surface_points(M)  → (x, normals)                       │
│  get_visible_cameras(x)    → cam_ids                            │
│  project_to_camera(x, c)   → uv (differentiable)                │
│                                                                  │
│  Two backends: PointCloudSurface · NeuralStereoSurface          │
├─────────────────────────────────────────────────────────────────┤
│ Step 3: Neural Deformation Field (deformation_net.py + trainer)  │
│                                                                  │
│  HashGrid L=16 + Temporal Encoder → MLP → tanh gate → Φ(x,t)   │
│  Coarse-to-fine: patch 32→16→8, λ_smooth 1e-2→1e-4              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
ndef_dic/                         (~5050 行)
├── __init__.py                   (   9)   package
├── colmap_calib.py               ( 334)   Step 0/1: COLMAP 稀疏 SfM
├── dense_mvs.py                  ( 317)   Step 1: COLMAP PatchMatch 稠密 MVS
├── pinn_stereo.py                (1156)   Step 1: PINN-Stereo 神经深度估计
├── postprocess.py                ( 519)   Step 1: 点云后处理 (SOR, 法向量, vis_mask)
├── step1_pipeline.py             ( 480)   Step 1: 统一编排 (sparse→dense→post)
├── surface_provider.py           (1000)   Step 2: 表面采样接口 (ABC+2 实现)
├── dic_losses.py                 ( 171)   Step 3: ZNSSD + patch 提取 + smoothness
├── deformation_net.py            ( 404)   Step 3: HashGrid + Temporal + Φ 网络
├── dataset.py                    ( 196)   Step 3: 多相机图像数据集
└── deformation_trainer.py        ( 468)   Step 3: 训练编排器

case/CylinderDIC/                         测试用例
├── images/                     12 相机 × 变形帧
├── calibration/                cameras.mat, points3D.mat
│   └── dense/                  dense_points.ply, normals.npy, vis_mask.npy
└── ground_truth/               points_ref.npy

docs/                                    设计文档
├── new_plan.md                 项目总设计
├── step0-synthetic-data-generation.md
├── step1-geometric-reconstruction.md
├── step2-surface-representation.md
├── step3-neural-deformation-field.md
├── step2-step3-interface.md    Step 2 ↔ Step 3 接口约定
└── neural-implicit-projection-zssd.md
```

---

## Φ 网络架构

```
  Input (x ∈ [-1,1]³)              Input (t ∈ [0,1])
       │                                 │
  HashGridEncoder                   TemporalEncoder
  L=16 levels, F=2                  binary | PE(L=6) | PE(L=8)
  table=2^19, 67MB                  1 | 12 | 16 dims
       │                                 │
       └──────────┬──────────────────────┘
                  │  Concat (33~48 dims)
                  ▼
    MLP: in→256→256→(skip+in)→256→256→256→3  (ReLU)
                  │
                  ▼
    Φ = tanh(α·t) · Φ_raw          ← hard constraint: Φ(t=0)=0
                  │
                  ▼
    Output: (u, v, w) ∈ ℝ³          displacement in world units
```

**关键性质**：
- `Φ(x, 0) = (0,0,0)` 精确成立（门控函数的结构保证，非学习得到）
- 17M 参数，纯 PyTorch，无外部依赖
- `project_to_camera` 完全可微 → 梯度可反传至网络权重

---

## 设计原则

此项目**不是** NeRF 项目。以下 NeRF 组件**不引入**：

- Volume Rendering · Density/Radiance Field · Sphere Tracing
- Hierarchical Sampling · View-dependent Color

设计决策的优先级：

1. 是否提升 DIC 匹配精度？
2. 是否提升位移场表达能力？
3. 是否降低计算成本？
4. 是否有利于后续 PINN 和参数反演？

---

## 快速开始

```python
from ndef_dic.surface_provider import create_surface_provider
from ndef_dic.dataset import MultiCamDataset
from ndef_dic.deformation_net import DeformationNetwork
from ndef_dic.deformation_trainer import DeformationFieldTrainer

# 1. 加载表面（Step 2）
surface = create_surface_provider(
    data_dir="case/CylinderDIC",
    calib_dir="case/CylinderDIC/calibration",
    method="point_cloud",       # "point_cloud" | "neural_stereo"
)

# 2. 加载图像数据（Step 3）
dataset = MultiCamDataset(
    data_dir="case/CylinderDIC",
    image_width=1440, image_height=1080,
)

# 3. 创建 Φ 网络
net = DeformationNetwork()

# 4. 训练
trainer = DeformationFieldTrainer(surface, net, dataset)
trainer.train()         # coarse-to-fine: 32→16→8 patch sizes
```

---

## 验证结果

| 测试项 | 结果 |
|--------|------|
| Step 2 PointCloudSurface 创建 | 200K 点, 12 cameras, mean vis 6.6 |
| Step 2 project_to_camera 梯度 | `∂L/∂x ≠ 0` — 可微 |
| Step 3 HashGridEncoder | 16.7M params, output (N,32), grad flows |
| Step 3 DeformationNetwork | `Φ(x,0)=0` 硬约束验证 (+1e-7) |
| Step 3 ZNSSD | 自相似=0, 仿射不变性验证 |
| 端到端训练 (200 iters) | ZNSSD: 872 → 26 (−96.5%) |

---

## 引用

本项目基于以下核心技术：

- Müller et al., *Instant Neural Graphics Primitives* (SIGGRAPH 2022) — Hash Grid Encoding
- Schönberger et al., *Structure-from-Motion Revisited* (CVPR 2016) — COLMAP SfM
- Pan et al., *Zero-mean Normalized Sum of Squared Differences* — DIC 标准损失
