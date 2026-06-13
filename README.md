# NDF-DIC — Neural Deformation Field for Multi-View DIC

用神经隐式变形场直接从多目散斑图像中恢复连续 3D 位移场。

---

## 1. 核心思路

### 不是 NeRF，是 DIC

这个项目不渲染图像。它用一种连续隐式表示来替代传统 DIC 的逐像素相关匹配，直接求解 3D 位移场。

```
传统 DIC:   图像对 → 逐像素相关 → 2D/3D 位移
NDF-DIC:    多目图像 → Φ(x,t) → 投影 → 多目 ZNSSD → 连续 3D 位移场
```

### 为什么这么做

传统 DIC 有三个固有问题：

| 问题 | NDF-DIC 的方案 |
|------|---------------|
| **基于网格/子区** — 位移场在离散点上定义，难以表示裂纹和应变集中 | 使用 Hash Grid Encoding（L=16 层，最高 512³ 分辨率），天然表示高频局部变形 |
| **单相机/双目** — 依赖 patch 质量，遮挡区域无解 | 多目（12 相机）冗余 — 每个表面点被 6-7 个相机同时看到，遮挡不再致命 |
| **逐帧独立** — 无法利用时域连续性 | Φ(x,t) 自带时间维度，用一个网络表示全部加载步的位移场 |

### 为什么不用 NeRF 组件

Volume Rendering、Density Field、Radiance Field、Sphere Tracing 服务于图像合成，不服务于位移测量。引入它们只会增加计算成本，稀释 DIC 信号。

**唯一借鉴 NeRF 的组件是 Hash Grid Encoding**——因为它直接提升位移场的高频表达能力。

### 核心创新

多目 ZNSSD 损失直接监督变形场，不经过渲染管线：

```
表面点 x → Φ(x,t) → x_def = x + φ
           ↓
project_to_camera(x, cam) → uv_ref    } 
project_to_camera(x_def, cam) → uv_def } 可微投影
           ↓
extract_patches(ref_img, uv_ref) → P_ref  }
extract_patches(def_img, uv_def) → P_def  } grid_sample
           ↓
ZNSSD(P_ref, P_def) → loss
           ↓
loss.backward() → 梯度经 project_to_camera 反传至 Φ 网络
```

整条梯度链没有断点——投影、patch 提取、ZNSSD 全部可微。

---

## 2. 项目架构

### 三阶段流水线

```
Step 1: Geometric Reconstruction（几何重建）
        COLMAP SfM → K,R,t + 稀疏点
        PatchMatch Stereo / PINN-Stereo → 稠密点云 / D_c 深度网络
        ↓
Step 2: SurfaceProvider（表面采样接口）
        sample_surface_points(M)   → (x, normals)
        get_visible_cameras(x)      → cam_ids
        project_to_camera(x, cam)   → uv（可微）
        ↓
Step 3: Neural Deformation Field（变形场训练）
        Φ(x,t): HashGrid + MLP + tanh gate → (u,v,w)
        Coarse-to-fine: patch 32→16→8，λ_smooth 1e-2→1e-4
```

### Step 1 设计决策

优先使用 COLMAP 成熟模块。项目核心创新不在几何重建——除非确有必要，不要自己实现 MVS。

两个稠密重建路径：
- **PatchMatch**（推荐）：COLMAP 的标准稠密重建，精度高，需要 CUDA
- **PINN-Stereo**（实验性）：每相机一个隐式深度网络 D_c(u,v)，通过跨相机 ZNSSD 联合优化。精度不如 PatchMatch，但**完全可微**

### Step 2 设计决策

Step 2 的价值不在产生数据——而在**统一接口**。两种实现对 Step 3 完全透明：

| 实现 | 数据来源 | 采样方式 | 法向量 |
|------|---------|---------|--------|
| `PointCloudSurface` | Step 1 稠密点云 (.ply + .npy) | 随机索引查表 | PCA 估计 |
| `NeuralStereoSurface` | PINN-Stereo 的 D_c 网络 | 像素采样 → D_c → unproject | ∇D_c 解析梯度 |

Step 3 不关心底层是哪种实现——`project_to_camera()` 的行为完全相同。

### Step 3 设计决策

**网络架构**：

```
  x ∈ [-1,1]³                     t ∈ [0,1]
      │                                │
  HashGridEncoder                  TemporalEncoder
  L=16 levels, F=2/level           binary | PE(L=6)
  2^19 table, 67 MB                1 | 12 dims
      │                                │
      └────────── Concat ──────────────┘
                    │
    MLP: in→256→256→(skip+in)→256→256→256→3, ReLU
                    │
    Φ = tanh(α·t) · Φ_raw      ← t=0 时 Φ 严格为零
                    │
    (u, v, w) ∈ ℝ³
```

**关键性质**：
- `Φ(x, 0) = (0, 0, 0)` 是结构保证的硬约束（tanh 门控），非学习得到
- 纯 PyTorch，17M 参数，无需 tiny-cuda-nn
- `project_to_camera` 为线性代数 + 齐次除法，天然可微

**训练策略**：Coarse-to-fine patch sizing — 大 patch 捕获大位移（吸引域大），小 patch 精细化局部应变。

---

## 3. 使用说明

### 安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch >= 2.0`、`numpy`、`scipy`、`opencv-python`、`pyyaml`。  
可选：`pycolmap`（PatchMatch 稠密重建需要 COLMAP + CUDA）。

### 数据准备

```
case/<your_case>/
├── images/
│   ├── cam_0/  001.bmp, 002.bmp, ...    ← 参考帧 + 变形帧
│   ├── cam_1/  001.bmp, 002.bmp, ...
│   └── ...
└── calibration/
    └── cameras.mat                       ← COLMAP SfM 输出（如已有）
```

### 配置

所有参数在 `config/default.yaml` 中。创建 `config/local.yaml` 覆盖默认值（已 .gitignore）。

```yaml
# config/local.yaml
device: cuda
step1:
  sparse_mode: skip_sfm       # 已有标定则跳过 SfM
step3:
  training:
    phases:                    # 缩减迭代用于快速实验
      - patch_size: 32
        iterations: 500
        lr: 0.001
      - patch_size: 16
        iterations: 1000
        lr: 0.0005
```

### 运行

```bash
# 完整流水线
python run.py

# 只跑 Step 3（已有标定和稠密点云）
python run.py --steps 3 --device cuda

# 自定义配置
python run.py --config config/local.yaml

# 清除已有结果重跑
python run.py --clean

# 查看全部选项
python run.py --help
```

### Python 接口

```python
from ndef_dic.surface_provider import create_surface_provider
from ndef_dic.dataset import MultiCamDataset
from ndef_dic.deformation_net import DeformationNetwork
from ndef_dic.deformation_trainer import DeformationFieldTrainer

# Step 2: 加载表面
surface = create_surface_provider(
    data_dir="case/CylinderDIC",
    calib_dir="case/CylinderDIC/calibration",
    method="point_cloud",
)

# Step 3: 训练变形场
dataset = MultiCamDataset(data_dir="case/CylinderDIC", image_width=1440, image_height=1080)
net = DeformationNetwork()
trainer = DeformationFieldTrainer(surface, net, dataset)
trainer.train()

# 推理
phi = trainer.query_displacement(x_points, t=1.0)
strain = trainer.compute_strain(x_points, t=1.0)
```

### 已验证结果

| 项目 | 数值 |
|------|------|
| 表面点云 | 200K 点，12 cameras，mean vis 6.6 |
| Φ 网络参数量 | 17M（16-level HashGrid + MLP） |
| 训练 200 iters | ZNSSD 872 → 26（−97%） |
| Φ(t=0) 硬约束 | max\|Φ(x,0)\| < 1e-6 精确成立 |
| ZNSSD 仿射不变性 | ZNSSD(P, aP+b) = 0 严格成立 |

---

## 设计原则

所有架构决策按以下优先级排序：

1. 是否提升 DIC 匹配精度？
2. 是否提升位移场表达能力？
3. 是否降低计算成本？
4. 是否有利于后续 PINN 和参数反演？

**本项目的核心不是 NeRF。核心是 Neural Deformation Field + Multi-view DIC。**
