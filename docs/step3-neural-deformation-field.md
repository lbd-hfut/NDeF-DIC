# Step 3：Neural Deformation Field

**日期**: 2026-06-11
**状态**: 设计讨论
**关联**: [[step1-geometric-reconstruction]], [[step2-surface-representation]], [[neural-implicit-projection-zssd]], [[new_plan]]

---

Step 3 是 NDF-DIC 的核心——训练一个神经隐式变形场 $\Phi(x,t): \mathbb{R}^4 \to \mathbb{R}^3$，输入 3D 坐标和时间，输出位移向量，通过多目 ZNSSD 损失直接监督。

这也是整个项目的主要创新所在：**不经过渲染，不经过 sphere tracing，直接用 DIC 的图像相关损失训练位移场。**

---

## 1. Step 3 在整体架构中的位置

```
Level 0: COLMAP 稀疏 SfM
  → K_c, R_c, t_c, 稀疏点

Level 1: Neural Stereo {D_c}                              ← Step 1
  → 训练好的深度网络 + 相机参数

Level 2: SurfaceProvider                                  ← Step 2
  → sample_surface_points()、get_visible_cameras()
  → project_to_camera()

Level 3: Neural Deformation Φ                             ← Step 3 (本文档)
  网络: Φ(x,t): ℝ⁴ → ℝ³
  表面点来源: SurfaceProvider 在线采样
  变形点: x_def = x + Φ(x,t)
  损失: 多目 ZNSSD (跨时间)
  输出: 连续 3D 位移场 u(x,t), v(x,t), w(x,t)
```

---

## 2. 网络结构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Deformation Network Φ(x,t)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入:                                                       │
│    x ∈ ℝ³  归一化空间坐标 (来自 SurfaceProvider)               │
│    t ∈ ℝ   归一化时间 [0, 1]                                  │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Spatial Encoding                                  │       │
│  │ Multi-Res Hash Grid (L=16 levels)                 │       │
│  │ features_per_level=2, base_res=16, finest_res=512 │       │
│  │ hash_table_size=2^19                              │       │
│  │                                                   │       │
│  │ Input: (x, y, z)                                  │       │
│  │ Output: L×F = 32 features                         │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│  ┌──────────────────────┴───────────────────────────┐       │
│  │ Temporal Encoding                                 │       │
│  │                                                   │       │
│  │ 准静态 (2个时刻): Binary indicator [1_{t>0}]        │       │
│  │ 多步加载 (≤50帧): Positional Encoding (L_t=6)      │       │
│  │ 连续采集 (>50帧): PE + Temporal Smoothness Loss    │       │
│  │                                                   │       │
│  │ Output: 2×L_t = 12 features (多步情况)              │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│         ┌───────────────┴───────────────┐                    │
│         ▼                               ▼                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Feature Concatenation: 32 + 12 = 44 dims          │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │ MLP Backbone                                      │       │
│  │                                                   │       │
│  │ Layer 0: 44 → 256, ReLU                           │       │
│  │ Layer 1: 256 → 256, ReLU                          │       │
│  │         ↓ skip: concat input(44) + Layer 1(256)   │       │
│  │ Layer 2: 300 → 256, ReLU                          │       │
│  │ Layer 3: 256 → 256, ReLU                          │       │
│  │ Layer 4: 256 → 256, ReLU                          │       │
│  │ Layer 5: 256 → 3  (u, v, w), no activation        │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Zero-Displacement Gating                          │       │
│  │                                                   │       │
│  │ Φ(x, t) = tanh(α · t) · Φ_raw(x, t)               │       │
│  │                                                   │       │
│  │ t=0 → tanh(0) = 0 → Φ = 0 (硬约束)                 │       │
│  │ t>0 → tanh→1, 位移逐渐解禁                          │       │
│  │ α: 陡峭度 (可学习或固定 ≈5.0)                       │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ▼                                    │
│  输出: (u, v, w) ∈ ℝ³  位移向量                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各组件详解

#### Spatial Hash Grid

```python
# 参数选择
hash_grid_config = {
    "num_levels": 16,           # 多分辨率层数
    "features_per_level": 2,    # 每层特征维度
    "base_resolution": 16,      # 最粗层分辨率
    "finest_resolution": 512,   # 最细层分辨率
    "hash_table_size": 2**19,   # 哈希表大小 (524,288)
    "total_features": 32,       # 16 levels × 2 features
}
```

**覆盖范围分析**（试样 100×100×10mm，归一化到 [-1, 1]³）：

| 层级 | 分辨率 | 格点间距 (mm) | 能分辨的变形特征 |
|------|--------|-------------|----------------|
| L0 | 16³ | 6.25 | 整体变形趋势 |
| L8 | 256³ | 0.39 | 局部应变集中 |
| L15 | 512³ | 0.20 | 裂纹尖端、窄颈缩 |

**设计考量**：最细层分辨率 512 在 100mm 尺度上对应 ~0.2mm 的格点间距。如果试样的厚度仅 1mm，厚度方向的格点可能只有 ~5 个。对于薄膜试样，后续可考虑 anisotropic hash grid——在薄方向降低分辨率。

#### Temporal Encoding

时间编码取决于加载步数：

```python
def encode_time(t, num_steps):
    """
    t ∈ [0, 1] 归一化时间

    策略取决于数据量:
    """
    if num_steps <= 2:
        # 只有参考 + 一个变形状态 → 时间维度退化
        # 用 binary indicator 区分参考/变形
        return torch.tensor([1.0 if t > 0 else 0.0])

    elif num_steps <= 50:
        # 中等数量加载步 → Positional Encoding
        L = 6  # 6 个频率
        enc = []
        for i in range(L):
            freq = 2.0 ** i * torch.pi
            enc.append(torch.sin(freq * t))
            enc.append(torch.cos(freq * t))
        return torch.cat(enc, dim=-1)  # 12 dims

    else:
        # 大量帧 → PE + 时域平滑正则化
        # (同上，但在 loss 中额外加 L_temp_smooth)
        return encode_time(t, L=8)  # 更高频率
```

#### Zero-Displacement Gating

三种方案对比：

| 方案 | 形式 | t=0 约束 | 表达能力 | 推荐 |
|------|------|---------|---------|------|
| A: 乘性门控 | $\Phi = h \cdot t \cdot \Phi_{raw}$ | 硬约束 | 中（t 很小时被抑制） | ❌ |
| B: tanh 门控 | $\Phi = \tanh(\alpha t) \cdot \Phi_{raw}$ | 硬约束 | 高（通过 α 控制过渡带） | ✅ |
| C: 软约束 | $\Phi = \Phi_{raw}$ + loss 惩罚 $\Phi(t=0)$ | 软约束 | 最高 | ❌ |

**推荐方案 B**（tanh gate）：
- $t=0$ 时严格为 0（硬约束，不需要 loss 项）
- $\alpha \approx 5.0$ 时，$\tanh(5 \times 0.05) \approx 0.24$，在小 t 时已部分解禁
- 通过调整 α 控制"位移从何时开始显著"

```python
def forward(self, x, t):
    phi_raw = self.mlp(self.hash_grid(x), self.pe_time(t))
    gate = torch.tanh(self.alpha * t)  # self.alpha 可学习
    return gate * phi_raw
```

---

## 3. 损失函数

### 3.1 核心损失：多目 ZNSSD

```
╔═══════════════════════════════════════════════════════════════╗
║  多目 DIC Loss — 整个框架最重要的损失函数                        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  L_dic = (1 / (M × K)) Σ_{i=1}^{M} Σ_{c∈vis(x_i)}            ║
║          ZNSSD( P_ref(i,c),  P_def(i,c) )                     ║
║                                                               ║
║  其中:                                                        ║
║    x_i:          SurfaceProvider 采样的表面点                   ║
║    vis(x_i):     可见相机集合 (来自 Step 2)                     ║
║                                                               ║
║    P_ref(i,c) = crop(ref_img[c], π_c(x_i),           w)      ║
║    P_def(i,c) = crop(def_img[c], π_c(x_i + Φ(x_i,t)), w)     ║
║                                                               ║
║    ZNSSD(P,Q) = Σ_j [(P_j-μ_P)/σ_P - (Q_j-μ_Q)/σ_Q]²         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 3.2 一次训练迭代的完整计算图

```python
def training_step(surface, deformation_net, cameras, images, t, config):
    """
    一次 training iteration 的伪代码。
    """
    M = config.batch_size          # 表面点数量, 如 1024
    K = config.cameras_per_point   # 每个点用几个相机, 如 2-3
    patch_size = config.patch_size # 当前阶段的 patch 尺寸

    # ① 采样表面点
    x, normals = surface.sample_surface_points(M)
    # x: (M, 3) 世界坐标

    # ② 查询可见相机
    cam_ids = surface.get_visible_cameras(x, max_cams=K)
    # cam_ids: (M, K) int

    # ③ 计算 Φ(x,t)
    phi = deformation_net(x, t)     # (M, 3)
    x_def = x + phi                 # (M, 3) 变形后的位置

    total_loss = 0.0

    for c in range(num_cameras):
        # 找到使用相机 c 的点
        mask_c = (cam_ids == c).any(dim=-1)  # (M,)
        if mask_c.sum() == 0:
            continue

        x_c = x[mask_c]        # 子集
        x_def_c = x_def[mask_c]

        # ④ 投影到相机 c
        uv_ref = surface.project_to_camera(x_c, c)     # 参考投影
        uv_def = surface.project_to_camera(x_def_c, c)  # 变形投影

        # ⑤ 提取 patch
        P_ref = grid_sample(ref_images[c], uv_ref, patch_size)
        P_def = grid_sample(def_images[c][t], uv_def, patch_size)

        # ⑥ ZNSSD
        total_loss += ZNSSD(P_ref, P_def)

    # ⑦ 正则化
    loss_smooth = deformation_field_smoothness(deformation_net, x, t)

    return total_loss / (M * K) + lambda_smooth * loss_smooth
```

### 3.3 正则化项

```
═══════════════════════════════════════════════════════════════
位移场平滑性 (必须)
═══════════════════════════════════════════════════════════════

L_smooth = (1/M) Σ_i ||∇_x Φ(x_i, t)||_F²

∇_x Φ ∈ ℝ^{3×3} 是位移梯度张量:
         [∂u/∂x  ∂u/∂y  ∂u/∂z]
  ∇Φ =  [∂v/∂x  ∂v/∂y  ∂v/∂z]
         [∂w/∂x  ∂w/∂y  ∂w/∂z]

||∇Φ||_F² = Σ_{i,j} (∂Φ_i/∂x_j)²

通过 torch.autograd 对每个采样点计算 Jacobian


═══════════════════════════════════════════════════════════════
不可压缩性 (可选 — 橡胶类材料)
═══════════════════════════════════════════════════════════════

L_incomp = (1/M) Σ_i (tr(∇Φ(x_i,t)))²
         = (1/M) Σ_i (∂u/∂x + ∂v/∂y + ∂w/∂z)²


═══════════════════════════════════════════════════════════════
时域平滑性 (可选 — 多帧连续加载)
═══════════════════════════════════════════════════════════════

L_temp = (1/M) Σ_i ||∂Φ/∂t (x_i, t)||²

用于加载步数 > 10 的情况，约束位移随时间平滑变化


═══════════════════════════════════════════════════════════════
总损失
═══════════════════════════════════════════════════════════════

L_total = L_dic + λ_smooth·L_smooth + λ_incomp·L_incomp + λ_temp·L_temp
```

### 3.4 超参数参考值

| 参数 | 建议值 | 说明 |
|------|--------|------|
| λ_smooth | 1e-2 (coarse) → 1e-4 (fine) | 随 patch 缩小而降低 |
| λ_incomp | 0 (默认) 或 1e-3 | 仅对橡胶/不可压缩材料 |
| λ_temp | 0 (默认) 或 1e-2 | 仅多帧 (>10) 时启用 |
| batch_size M | 1024 | 表面点采样数 |
| cameras_per_point K | 2~3 | 每个点用几个相机 |
| patch_size | 32 → 16 → 8 | 粗到精逐步缩小 |

---

## 4. 梯度流分析

### 4.1 完整的端到端梯度链

从 ZNSSD loss 到变形网络参数 θ_Φ，梯度经过以下路径：

$$\frac{\partial L}{\partial \theta_\Phi} = \frac{\partial L}{\partial Q} \cdot \frac{\partial Q}{\partial (u',v')} \cdot \frac{\partial (u',v')}{\partial \mathbf{X}_{def}} \cdot \frac{\partial \mathbf{X}_{def}}{\partial \Phi} \cdot \frac{\partial \Phi}{\partial \theta_\Phi}$$

逐项展开：

```
(1) ∂L/∂Q:
    ZNSSD 对变形 patch 像素值的梯度
    物理含义: "像素值差多少"

(2) ∂Q/∂(u',v'):
    变形图像的空间梯度 (grid_sample 反传)
    物理含义: "speckle 边缘在哪"
    关键: 这是图像纹理质量的直接体现

(3) ∂(u',v')/∂X_def:
    投影函数对 3D 点的 Jacobian
    物理含义: "3D 点移动如何影响 2D 投影"
    解析形式，无近似

(4) ∂X_def/∂Φ:
    X_def = X_ref + Φ  →  ∂X_def/∂Φ = I (恒等映射)
    物理含义: "位移变化 = 位置变化"

(5) ∂Φ/∂θ_Φ:
    标准 MLP + Hash Grid 反向传播
```

**整条梯度链没有断点，没有离散操作，没有数值近似。这是端到端可微的核心保证。**

### 4.2 梯度消失的两种情况

```
情况 A: Speckle 纹理平坦
  → ∇I_def(u',v') ≈ 0
  → (2) ≈ 0 → 整条链 ≈ 0
  → Φ 收不到梯度信号
  → 解决: 提高 speckle 质量，增大 patch_size

情况 B: 变形太大，patch 完全不匹配
  → ZNSSD 接近常数 (完全不相关 = 2)
  → (1) 梯度几乎为零 (ZNSSD 在 2 附近饱和)
  → 解决: 增大初始 patch_size (64×64) 扩大吸引域
         coarse-to-fine 策略逐步缩小
```

---

## 5. 训练策略

### 5.1 Coarse-to-Fine Patch Sizing

DIC 的一个核心经验：大 patch 吸引域大但精度低，小 patch 精度高但容易陷入局部极小。

```
Phase 1: Coarse Matching (warm-up)
  patch_size: 64×64 或 32×32
  λ_smooth:   1e-2
  epochs:     ~2000
  目标:       捕获大位移，建立大致对应关系
  备注:       此时 ZNSSD 曲面平滑，梯度信号强

Phase 2: Refinement
  patch_size: 16×16
  λ_smooth:   1e-3
  epochs:     ~5000
  目标:       精细化位移场

Phase 3: Fine Detail
  patch_size: 8×8
  λ_smooth:   1e-4
  epochs:     ~3000
  目标:       捕获局部应变集中
  备注:       只在样本充分且 speckle 质量高时启用
```

### 5.2 加载步的训练顺序

```
如果只有一个变形状态 (准静态):
  → 直接训练 t=1 时刻的 Φ

如果有多个加载步 (1, 2, ..., N):
  → 策略 A: 依次训练
    先学 t=1 (小变形)，收敛后 → t=2 → ... → t=N
    优点: 简单
    缺点: 无法利用时域连续性

  → 策略 B: 联合训练
    每个 iteration 随机采样 t ∈ {1, ..., N}
    优点: 时域正则化自然生效
    缺点: 需要更多 iteration

  → 策略 C: 课程训练
    先学小 t，逐步引入大 t
    优点: 利用连续性，逐步增加难度
    推荐用于大变形场景
```

**默认使用策略 B（联合训练）**，简单且有效。如果只有 1 个变形状态，时间维度退化为 binary flag。

### 5.3 Batch 构成

```
每个 iteration:
  采样 M = 1024 个表面点
  每个点随机选 K = 2-3 个可见相机
  总共 M × K ≈ 2000-3000 个 patch pair
  每个 patch: w × w pixels

内存估算:
  每个 patch 16×16 = 256 pixels
  3000 patches × 256 pix × 4 bytes (float32) ≈ 3 MB
  完全可以常驻 GPU，不需要梯度检查点
```

---

## 6. 位移梯度和应变计算

### 6.1 从 Φ 计算位移梯度张量

```python
def compute_displacement_gradient(phi_net, x, t):
    """
    通过 autograd 计算位移梯度张量 ∇Φ ∈ ℝ^{3×3}。

    对每个采样点 x_i，计算 Jacobian:
      ∇Φ[i] = [∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z]
    """
    x.requires_grad_(True)

    phi = phi_net(x, t)  # (M, 3)

    grad_u = torch.autograd.grad(phi[:, 0].sum(), x,
                                  create_graph=True)[0]  # (M, 3)
    grad_v = torch.autograd.grad(phi[:, 1].sum(), x,
                                  create_graph=True)[0]
    grad_w = torch.autograd.grad(phi[:, 2].sum(), x,
                                  create_graph=True)[0]

    # 组装为 (M, 3, 3)
    grad_phi = torch.stack([grad_u, grad_v, grad_w], dim=-1)
    return grad_phi
```

### 6.2 从位移梯度到应变张量

在表面切平面内计算 Green-Lagrange 应变：

```python
def compute_surface_strain(grad_phi, normal):
    """
    在表面切平面内计算应变。

    Args:
        grad_phi: (M, 3, 3) 位移梯度
        normal:   (M, 3) 表面法向量

    Returns:
        eps_surface: (M, 2, 2) 切平面内的应变张量
    """
    # 构造切平面基
    # 从法向量出发，构造两个正交的切向量
    e1 = compute_tangent_basis(normal)     # (M, 3)
    e2 = torch.cross(normal, e1, dim=-1)   # (M, 3)

    # 将 3×3 位移梯度投影到切平面
    # eps_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i + ∂u_k/∂x_i * ∂u_k/∂x_j)
    # 其中 i, j ∈ {tangent1, tangent2}

    # 变形梯度在切平面内
    F = torch.eye(3) + grad_phi           # (M, 3, 3)
    F_tangent = project_to_tangent(F, e1, e2)  # (M, 2, 2)

    # Green-Lagrange 应变
    eps = 0.5 * (F_tangent.T @ F_tangent - torch.eye(2))
    return eps
```

### 6.3 应变作为评估指标（非训练目标）

应变在初始设计中不作为训练损失的一部分（避免引入材料假设），而是作为**训练后的评估指标**：
- 可视化应变场 → 判断位移场是否物理合理
- 与应变片/DIC 软件结果对比 → 验证精度
- 检测异常区域 → 可能指示训练不充分或数据问题

---

## 7. 从 Step 2 到 Step 3 的数据流

```
Step 2 (SurfaceProvider)                    Step 3 (Φ 训练)
─────────────────────                      ──────────────

sample_surface_points(M)  ──────────────→  x (M, 3)
                                                    ↓
                                            Φ(x, t) → φ (M, 3)
                                                    ↓
                                            x_def = x + φ

project_to_camera(x, c)    ──────────────→  uv_ref (M, 2)
project_to_camera(x_def, c) ──────────────→  uv_def (M, 2)

get_visible_cameras(x)     ──────────────→  选择哪些相机参与计算

[外部 — Dataset]                           ref_img[c], def_img[c][t]
                                           grid_sample → patches
                                                    ↓
                                           ZNSSD → L_dic
```

整个接口只有三个方法调用，干净清晰。

---

## 8. 实现清单

```
□ 3.1 实现 Hash Grid Encoder
     - Multi-resolution hash encoding (可参考 tiny-cuda-nn 或纯 PyTorch)
     - L=16 levels, F=2 features, T=2^19 hash size

□ 3.2 实现 Temporal Encoding
     - Binary indicator (准静态)
     - Positional Encoding (多步)
     - 可切换的时间编码策略

□ 3.3 实现 Φ 网络 forward
     - Hash encoding → concat with time encoding → MLP → tanh gate
     - Skip connection 在第 2 层
     - Zero-displacement gating via tanh(α·t)

□ 3.4 实现 ZNSSD 模块
     - 逐 patch 计算均值 + 标准差
     - 向量化实现 (batch 计算)
     - 数值稳定性: ε > 0 防止除零

□ 3.5 实现 grid_sample patch 提取
     - 对每个表面点-相机对提取 w×w patch
     - 利用 torch.nn.functional.grid_sample
     - 注意坐标归一化: uv → [-1, 1]

□ 3.6 实现多目 DIC Loss
     - 遍历可见相机
     - 对每个 camera-point pair 计算 ZNSSD
     - 汇总所有 pair 的 loss

□ 3.7 实现位移场平滑正则化
     - autograd 计算 ∇Φ Jacobian
     - Frobenius norm
     - (可选) 不可压缩性 / 时域平滑

□ 3.8 实现训练循环
     - 从 SurfaceProvider 采样
     - Coarse-to-fine patch size schedule
     - λ_smooth 衰减 schedule
     - 学习率调度 (cosine annealing)

□ 3.9 实现应变后处理
     - 计算位移梯度 → 应变张量
     - 表面切平面投影
     - 可视化导出

□ 3.10 单元测试
     - Φ(t=0) = 0 硬约束验证
     - ZNSSD 对线性光照不变性
     - 投影-反投影自洽性
     - 梯度不爆炸/不消失 (gradient norm check)
```

---

## 9. 设计决策附录

### 9.1 为什么不加 I(x)（Intensity Field）

在之前的讨论中，I(x) 作为"多视角一致参考纹理"是一个可选组件。当前设计中不加 I(x) 的理由：

1. **ZNSSD 已经吸收了多相机曝光差异**——不需要额外的外观嵌入
2. **多相机本身的冗余已提供去噪**——单相机噪声在 ZNSSD 求和过程中被平均
3. **减少一个网络 = 减少一个训练阶段 = 减少调试复杂度**

触发加 I(x) 的条件：如果发现单相机噪声确实影响收敛，或者需要 I(x) 作为正则化项时再引入。

### 9.2 关于 Hash Grid 的选择

纯 MLP + PE 也能表示位移场，但 Hash Grid 在两个关键点上优于纯 MLP：

1. **高频表示能力**：Speckle 纹理要求 displacement field 能够捕捉与纹理匹配的局部变形（~0.1mm 尺度），纯 MLP 的谱偏置会抑制这些高频分量
2. **收敛速度**：Hash Grid 的特征查找是 O(1) + 线性插值，相比 PE 的全局正弦基函数，局部更新的效率更高

代价是增加了约 500K 的哈希表参数（16 levels × 2^19 × 2 features），但在 GPU 上完全可以接受。

### 9.3 时间维度的最小化处理

对于大多数 DIC 实验（参考 + 1~2 个变形状态），时间维度不是连续的。将 t 退化为 binary indicator 是最实用的选择。保留 Φ(x,t) 的接口（而非 Φ(x)）是为了未来的扩展性——SeqPINN-DIC、损伤演化等。

---

## 10. 参考资料

- [[step1-geometric-reconstruction]] — Step 1 几何重建（PINN-Stereo 或 COLMAP Dense）
- [[step2-surface-representation]] — Step 2 表面采样接口（SurfaceProvider）
- [[neural-implicit-projection-zssd]] — 神经隐式 + 投影约束 + ZNSSD 详细解释
- [[new_plan]] — NDF-DIC 项目整体设计
- [[2026-06-09-ndef-dic-redesign-discussion]] — 最初的重新设计讨论
