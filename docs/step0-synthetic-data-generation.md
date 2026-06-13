# Step 0：合成数据生成与基准设计

**日期**: 2026-06-12
**状态**: 设计 + 实现
**关联**: [[new_plan]], [[step1-geometric-reconstruction]], [[step2-surface-representation]], [[step3-neural-deformation-field]]

---

Step 0 的目标是生成已知 ground truth 的多目散斑图像，为 NDF-DIC 框架（Steps 1–3）提供可控的验证基准。所有输出——相机参数、三维表面几何、位移场——都是已知真值，使得 pipeline 每个环节的误差都可以被精确量化。

---

## 1. 为什么需要 Step 0

NDF-DIC 是一个由多个级联模块构成的系统：

```
Step 0 (合成数据) → Step 1 (几何重建) → Step 2 (表面表示) → Step 3 (变形场学习)
                                                        ↑
                                             已知 Ground Truth 对照
```

没有 Step 0 时，我们无法区分：
- **算法误差** vs **实验误差**（光照、振动、标定不精确）
- **模型假设误差** vs **数据不足误差**
- **模块 A 的误差** vs **模块 B 的误差**（级联传播）

Step 0 提供了**逐模块的误差溯源能力**：对每个步骤，输入和期望输出都是已知的，误差 = 算法输出 ⊖ ground truth。

---

## 2. 设计原则

### 2.1 物理一致性优先于视觉逼真度

仿真不需要完美复现真实相机（镜头暗角、散焦、镜面反射）。但必须保证：
- 三维投影关系是物理正确的（针孔模型 + 遮挡）
- 散斑图案附着在物体表面（随变形一起移动）
- 变形场满足几何约束（圆柱膨胀保持径向对称）

### 2.2 可控性优先于通用性

第一个版本的 Step 0 只支持圆柱几何体 + 参数化变形模式。不做任意几何体导入（那是后续工作）。参数化设计的价值在于：可以系统地改变单个参数（相机数量、噪声水平、变形幅度）来测试 pipeline 的鲁棒性。

### 2.3 Ground truth 完整性

Step 0 提供的 ground truth 必须覆盖 Step 1–3 的全部需求：

| 后续步骤 | 需要的 ground truth | Step 0 输出 |
|----------|-------------------|-------------|
| Step 1（SfM + MVS） | 相机内外参、稀疏/稠密点云 | `calibration/`, `ground_truth/points_ref.npy` |
| Step 2（表面表示） | 表面点云 + 法向量 + 可见性 | `ground_truth/` + 投影可重算 |
| Step 3（变形场） | 每个表面点的位移向量 u,v,w | `ground_truth/displacement.npy` |

---

## 3. 系统架构

```
                    ┌─────────────────────────┐
                    │   CylinderSimConfig      │
                    │   (所有参数集中管理)       │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌─────────────────┐
│ 相机阵列构建   │     │ 散斑表面生成   │     │ 变形场生成       │
│ build_camera  │     │ generate_     │     │ apply_         │
│ _array()      │     │ cylinder_     │     │ deformation()  │
│               │     │ surface()     │     │                │
└───────┬───────┘     └───────┬───────┘     └────────┬────────┘
        │                     │                      │
        │              ┌──────┴──────┐               │
        │              │ points_ref  │               │
        │              │ intensities │               │
        │              │ normals_ref │               │
        │              └──────┬──────┘               │
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 图像渲染         │
                    │ render_images() │
                    │ · 背景面剔除     │
                    │ · 投影 + 畸变    │
                    │ · 双线性散布     │
                    │ · Gamma 校正    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      ┌──────────────┐            ┌──────────────────┐
      │ images/       │            │ calibration/      │
      │ cam_*/001.bmp │            │ cameras.mat       │
      │ cam_*/002.bmp │            │ points3D.mat      │
      └──────────────┘            └──────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                   ┌──────────────────┐
                   │ ground_truth/     │
                   │ points_ref.npy    │
                   │ points_def.npy    │
                   │ displacement.npy  │
                   │ meta.json         │
                   └──────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ Validation Report │
                   │ validate_         │
                   │ simulation.py     │
                   └──────────────────┘
```

---

## 4. 各模块设计

### 4.1 相机阵列 (`build_camera_array`)

```
N 台相机均匀分布在 XZ 平面，相机中心位于距离原点 D = R + WD 处。

相机 i 的方位角: θ_i = 2π · i / N

世界 → 相机变换:
  R_i = [right_i; -up_i; forward_i]    (3×3 旋转矩阵)
  t_i = -R_i · C_i                     (3×1 平移向量)

其中:
  forward_i = -C_i / |C_i|              (指向原点)
  right_i   = forward_i × world_up
  up_i      = right_i × forward_i
```

**默认配置**（基于当前 CylinderDIC 算例）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_cameras` | 12 | 均匀环布 |
| `working_distance` | 300 mm | 相机中心到圆柱表面距离 |
| `image_width/height` | 1440×1080 | 像素 |
| `pixel_size` | 3.45 μm | 像元尺寸 |
| `focal_length` | 8.0 mm | 镜头焦距 |
| `k1, k2` | 0.0 | 理想针孔（可开启畸变） |

**相机距离**: D = R + WD = 80 + 300 = 380 mm

**分辨率**: 物面分辨率 = pixel_size / f × WD ≈ 0.129 mm/px

### 4.2 散斑表面生成 (`generate_cylinder_surface`)

#### 4.2.1 方法：程序化三维高斯颗粒

在圆柱表面上随机放置 `num_speckle_grains` 个三维高斯颗粒，直接计算每个表面采样点的灰度值。

**为什么不使用纹理映射？**
- 纹理映射涉及二维展开 → 三维映射，会在圆柱周向引入拉伸
- 展开纹理的高纬度区域畸变严重（极点效应）
- 平铺纹理会引入重复模式，不利于自标定（SL pattern ambiguity）
- 三维颗粒法确保散斑特征在空间上物理一致——不同相机看到的同一颗粒具有相同的形状和位置

#### 4.2.2 颗粒模型

```
每个颗粒 k 定义在圆柱表面 (θ_k, y_k):
  · 位置:    (θ_k, y_k)  — 圆柱坐标
  · 尺寸:    σ_k ∼ N(μ_σ, σ_σ²)    (mm, 测地距离)
  · 强度:    A_k ∼ U(0.3, 1.0)

对于表面点 p = (θ_p, y_p)，该点的灰度值为:
  I(p) = Σ_k A_k · exp(-d_k² / (2σ_k²))

其中 d_k 是测地距离:
  dθ = min(|θ_p - θ_k|, 2π - |θ_p - θ_k|)
  d_k² = (dθ · R)² + (y_p - y_k)²
```

**默认颗粒参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_speckle_grains` | 80,000 | 颗粒总数 |
| `grain_sigma_mean` | 0.25 mm | 平均颗粒半径 |
| `grain_sigma_std` | 0.08 mm | 颗粒尺寸变异 |
| `grain_sigma_range` | [0.1, 0.6] mm | 尺寸裁剪范围 |
| `grain_amp_range` | [0.3, 1.0] | 强度范围 |

#### 4.2.3 空间分 bin 加速

80,000 颗粒 × 15,000,000 采样点的暴力计算是 O(N_points × N_grains) ≈ 1.2×10¹² 次操作。使用 (θ, y) 空间分 bin：

```
bin_size_θ = 3·σ_max / R
bin_size_y = 3·σ_max

对于每个采样点，只查询其所在 bin 及 8 个邻接 bin 内的颗粒。
```

这使每点查询从 80,000 次距离计算减少到 ~40 次（平均每 bin 4.3 颗粒 × 9 bins），约 2000× 加速。

#### 4.2.4 采样密度

| 参数 | 值 |
|------|-----|
| `num_surface_points` | 15,000,000 |
| 圆柱表面积 | 2π·80·120 ≈ 60,319 mm² |
| 表面点密度 | ~249 pts/mm² |
| 平均点间距 | ~0.063 mm |
| 每像素对应点数（WD=300mm） | ~(0.129/0.063)² ≈ 4.2 |

采样密度需要足够高以避免投影后的像素空缺（小黑点）。经验准则：表面采样间距应不超过物面分辨率的 1/2。

### 4.3 图像渲染 (`render_images`)

#### 4.3.1 渲染管线

```
表面点 (N,3) + 灰度值 (N,) + 法向量 (N,3)
    │
    ▼
[1] 背面剔除: cos(normal, C-P) > 0.05
    │
    ▼
[2] 世界 → 相机: P_cam = R·P + t
    │
    ▼
[3] 归一化坐标: (xn, yn) = (X_cam/Z_cam, Y_cam/Z_cam)
    │
    ▼
[4] 畸变（可选）: xn_dist = xn · (1 + k1·r² + k2·r⁴)
    │
    ▼
[5] 像素坐标: (u, v) = (fx·xn_dist + cx, fy·yn_dist + cy)
    │
    ▼
[6] FOV 裁剪: (u,v) ∈ [0,W) × [0,H)
    │
    ▼
[7] 双线性散布: 每个点贡献给 4 个邻接像素
    │
    ▼
[8] 后处理: 归一化 → Gamma 校正 → 模糊 → 噪声 → 量化
```

#### 4.3.2 背面剔除

```python
view_dir = C - p           # 从表面点指向相机
cos_angle = n · view_dir   # 法向量 · 视线方向
front_facing = cos_angle > 0.05
```

仅法向量朝向相机的点被保留。`0.05` 的阈值容忍 ~87° 的掠射角——超过此角度的点由于透视压缩严重，对 DIC 匹配无贡献。

#### 4.3.3 双线性散布（Bilinear Splatting）

**动机**：简单的最近邻取整 (`u_int = round(u)`) 会产生 "死像素"——某些像素没有点落入，显示为突兀的小黑点。

**方法**：每个表面点向 4 个邻接像素贡献灰度值，权重为双线性插值系数。

```python
u0 = floor(u), v0 = floor(v)
wu1 = u - u0,  wv1 = v - v0
wu0 = 1 - wu1, wv0 = 1 - wv1

weights = [
    (u0, v0, wu0·wv0),  # 左上
    (u1, v0, wu1·wv0),  # 右上
    (u0, v1, wu0·wv1),  # 左下
    (u1, v1, wu1·wv1),  # 右下
]
```

每个像素累加 `Σ I_in × weight` 和 `Σ weight`，最终像素值 = `sum(I×w) / sum(w)`。

#### 4.3.4 后处理

| 步骤 | 操作 | 默认值 |
|------|------|--------|
| 归一化 | Linear stretch → [0, 1] | — |
| Gamma 校正 | `I^γ`, γ < 1 提亮中调 | γ = 0.55 |
| 输出映射 | Scale to `[intensity_min, intensity_max]` | [30, 250] |
| 高斯模糊 | `gaussian_filter(σ)` | σ = 0.3 px |
| 传感器噪声 | `I + N(0, σ_n²)` | σ_n = 0.0 |

### 4.4 变形模型 (`apply_deformation`)

#### 4.4.1 支持的变形模式

| 模式 | 参数 | 公式 |
|------|------|------|
| `none` | — | 恒等变换（仅生成参考状态） |
| `expansion` | Δr (mm) | r → r + Δr，均匀径向膨胀 |
| `torsion` | θ_max (°) | θ → θ + θ_max · (y / H_half)，扭转角随高度线性变化 |
| `compression` | ε (%) | y → y·(1−ε), r → r·(1+ν·ε)，轴向压缩 + 泊松效应 |
| `combined` | ε (%) | compression + torsion 叠加 |

#### 4.4.2 Expansion 的实现

```python
# 均匀径向膨胀：每个点沿径向向外移动 Δr
r = sqrt(x² + z²)
scale = 1.0 + Δr / r
x_def = x · scale
z_def = z · scale
# y 不变
```

注意：exp(Δr/r) 只在 r > 0 时有定义。圆柱轴线上 r = 0 的点不存在（表面点都在 r = R 处），所以安全。

### 4.5 验证报告 (`validate_simulation.py`)

仿真完成后自动生成验证报告，包含 6 类指标：

| 类别 | 关键指标 | 判断标准 |
|------|---------|---------|
| **散斑质量** | MIG (Mean Intensity Gradient) | > 15 gray levels/px |
| | 自相关峰宽（散斑尺寸） | 3–5 px |
| | 自相关峰锐度 | > 5 |
| **成像覆盖** | 每相机覆盖率 | > 10% |
| | 强度分布均匀性 | std/mean < 0.5 |
| **多目几何** | 相邻相机圆柱表面重叠角 | > 0°（有重叠） |
| | 交会角 | 10°–60° |
| **变形场** | 位移幅度分布 | 与输入参数一致 |
| | 径向/切向/轴向分量 | 验证变形模式正确性 |
| **表面采样** | 表面点密度 | 不低于物面分辨率 |

---

## 5. Step 0 → Step 1 接口

### 5.1 输出文件结构

```
case/CylinderDIC/
├── images/
│   ├── cam_0/
│   │   ├── 001.bmp              # 参考图像（t=0）
│   │   └── 002.bmp              # 变形图像（t=1）
│   ├── ...
│   └── cam_11/
│       ├── 001.bmp
│       └── 002.bmp
├── calibration/
│   ├── cameras.mat              # 相机参数 (COLMAT 格式)
│   │   ├── K_list               # 内参矩阵 (N_cam × 3 × 3)
│   │   ├── dist_list            # 畸变系数 (N_cam × 5)
│   │   ├── cam_from_world_R     # 旋转矩阵 (N_cam × 3 × 3)
│   │   ├── cam_from_world_t     # 平移向量 (N_cam × 3 × 1)
│   │   └── camera_models        # "PINHOLE"
│   └── points3D.mat             # 稀疏表面点 (N_sparse × 3)
├── ground_truth/
│   ├── points_ref.npy           # 参考表面点 (N × 3)
│   ├── points_def_step1.npy     # 变形表面点 (N × 3)
│   ├── displacement_step1.npy   # 位移向量 (N × 3)
│   └── meta.json                # 仿真参数
├── colmap_input/                # 所有参考图像的扁平副本（用于 COLMAP）
├── report/
│   ├── report.md                # 验证报告
│   └── metrics.json             # 量化指标
└── simulate_cylinder.py         # 仿真脚本
```

### 5.2 两种使用路径

```
路径 A（推荐验证流程）:
  Step 0 → 直接使用 calibration/ + ground_truth/
  → Step 2 (跳过 Step 1 的 SfM)
  → Step 3
  用于验证 Step 2–3 的算法正确性

路径 B（全流程验证）:
  Step 0 → images/ 送入 Step 1 (COLMAP SfM + MVS)
  → Step 2 (从重建点云出发)
  → Step 3
  用于验证 Step 1 的精度及其误差传播
```

路径 A 分离了算法验证和重建误差，是推荐的逐模块验证策略。

---

## 6. 与后续步骤的关系

### 6.1 与 Step 1 的关系

Step 1 从多目图像出发，目标是恢复相机参数和稠密点云。Step 0 提供：
- **已知相机参数**（用于跳过 SfM 或验证 SfM 精度）
- **已知三维点云**（用于验证 MVS 精度）
- **可控的视点重叠**（用于测试 SfM 在低重叠条件下的鲁棒性）

### 6.2 与 Step 2 的关系

Step 2 建立表面表示（SurfaceProvider）。Step 0 提供：
- **完整的表面点云**（验证 SurfaceProvider 的覆盖完整性）
- **法向量**（验证法向量估计精度）
- **可见性矩阵**（可通过投影重算）

### 6.3 与 Step 3 的关系

Step 3 学习变形场 Φ(x,t)。Step 0 提供：
- **每个表面点的 ground truth 位移**（u, v, w）
- **参考图像和变形图像**（用于计算 ZNSSD 监督信号）
- **相机参数**（用于投影操作）

这使得我们可以计算：
```
位移误差 = |Φ(x,t) - u_gt(x,t)|
逐点精度评估（而非仅评估图像级别的光一致性）
```

---

## 7. 已知限制与改进方向

### 7.1 当前限制

| 限制 | 影响 | 缓解 |
|------|------|------|
| 仅支持圆柱几何 | 无法测试非柱面场景 | 平板/球体可后续加入 |
| 理想针孔模型 | 无镜头畸变，实际相机存在畸变 | k1/k2 参数可开启 |
| 均匀光照 | 无曝光差异，实际多相机曝光不均 | ZNSSD 对线性曝光差鲁棒 |
| 无自遮挡变化 | 仅背面剔除，无复杂遮挡 | 对于凸几何体（圆柱）够用 |
| 静态散斑 | 散斑不随变形改变形状 | 实际散斑可能有局部拉伸 |

### 7.2 改进方向

1. **多几何体支持**：平板（面内拉伸）、带孔平板（应力集中）、球体
2. **物理光照模型**：Phong/Blinn 模型模拟镜面反射分量
3. **DIC 标准基准**：参考 DIC Challenge 2.0 的基准设计，生成标准验证集
4. **Speckle 模式变体**：支持从真实散斑图像反投影生成合成数据
5. **时序序列**：多加载步（当前仅 1 步）以测试 SeqPINN-DIC

---

## 8. 实现文件

| 文件 | 职责 |
|------|------|
| `case/CylinderDIC/simulate_cylinder.py` | 主仿真脚本：表面生成 → 渲染 → 变形 → 输出 |
| `case/CylinderDIC/validate_simulation.py` | 验证报告生成：质量指标 + 诊断图表 + Markdown 报告 |
| `docs/step0-synthetic-data-generation.md` | 本文档 |

---

## 9. 运行示例

```bash
# 默认配置运行
python case/CylinderDIC/simulate_cylinder.py

# 自定义参数
python case/CylinderDIC/simulate_cylinder.py \
    --working_distance 300 \
    --num_points 15000000 \
    --deformation expansion \
    --deformation_magnitude 0.5 \
    --noise_std 0

# 生成验证报告
python case/CylinderDIC/validate_simulation.py --output_dir case/CylinderDIC
```

---

## 参考资料

- [[new_plan]] — NDF-DIC 整体设计
- [[neural-implicit-projection-zssd]] — 核心范式详细解释
- [[step1-geometric-reconstruction]] — 几何重建（接收 Step 0 输出）
- [[step2-surface-representation]] — 表面表示
- [[step3-neural-deformation-field]] — 变形场学习
- DIC Challenge 2.0: Reu et al., Experimental Mechanics, 2022
- COLMAP: Schönberger & Frahm, CVPR 2016
