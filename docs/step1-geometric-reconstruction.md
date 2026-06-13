# Step 1：几何重建

**日期**: 2026-06-11
**状态**: 设计讨论
**关联**: [[new_plan]], [[neural-implicit-projection-zssd]], [[2026-06-09-ndef-dic-redesign-discussion]]

---

Step 1 的目标是从多目参考图像中恢复相机参数和物体表面几何，为 Step 3 的变形场学习提供空间域（表面点坐标）和投影算子（相机内外参）。

---

## 1. 输入与输出

### 输入

- 多台相机的参考图像（t=0，未变形状态）
- 每台相机 1 张，共 $N_{cam}$ 张
- 试样表面有喷涂 speckle 图案

### 输出

```
calibration/
├── cameras.mat              # 相机参数 (已有，来自稀疏 SfM)
│   ├── K_list               # 内参矩阵 (N_cam × 3 × 3)
│   ├── dist_list            # 畸变系数 (N_cam × 5)
│   ├── cam_from_world_R     # 旋转矩阵 (N_cam × 3 × 3)
│   ├── cam_from_world_t     # 平移向量 (N_cam × 3 × 1)
│   ├── P_list               # 投影矩阵 K[R|t]
│   └── camera_models        # 相机模型名称
│
├── points3D.mat             # 稀疏点云 (已有，来自 SfM)
│   └── points3D             # (N_sparse × 3)
│
└── dense/                   # 稠密重建输出 (待实现)
    ├── dense_points.ply     # 稠密点云 (N × 3, float32)
    ├── dense_normals.npy    # 法向量 (N × 3, float32)
    ├── vis_mask.npy         # 可见性矩阵 (N × N_cam, bool)
    ├── depth_maps/
    │   ├── cam_0_depth.bin  # 每台相机的深度图 (H × W, float32)
    │   ├── cam_1_depth.bin
    │   └── ...
    ├── normal_maps/
    │   ├── cam_0_normal.png # 每台相机的法向量图 (H × W × 3)
    │   └── ...
    └── meta.json            # 元数据: bounding box, scale factor, 统计信息
```

---

## 2. 当前代码状态

`ndef_dic/colmap_calib.py` 已实现稀疏 SfM 管线：

```
参考图像收集 → SIFT 特征提取 → 穷举匹配 → 增量 SfM → cameras.mat + points3D.mat
```

**缺失的是后半段（稠密 MVS）**：

```
稀疏 SfM 结果 → 图像去畸变 → PatchMatch Stereo → Stereo Fusion → 稠密点云
```

---

## 3. 稠密重建的两种实现路径

### 3.1 路径 A：COLMAP 传统 PatchMatch（成熟方案）

利用 COLMAP CLI 的标准管线：

```
Step 1a: 图像去畸变
  colmap image_undistorter
    --image_path <images>
    --input_path <sparse_sfm>
    --output_path <dense>
    → 无畸变图像 + 更新后的 K_c

Step 1b: PatchMatch Stereo
  colmap patch_match_stereo
    --workspace_path <dense>
    → 每张图像的深度图 + 法向量图

Step 1c: Stereo Fusion
  colmap stereo_fusion
    --workspace_path <dense>
    --output_path <dense>/fused.ply
    → 统一的稠密点云

Step 1d: 后处理
  - 统计离群点移除 (Statistical Outlier Removal)
  - ROI 裁剪
  - 可见性矩阵计算
```

#### PatchMatch 参数建议（speckle 场景）

| 参数 | 作用 | 建议值 | 理由 |
|------|------|--------|------|
| `window_radius` | 匹配窗口大小 | 11~15 | Speckle 是高纹理，大窗口提高判别性 |
| `num_iterations` | PatchMatch 迭代次数 | 7~8 | Speckle 可能存在误匹配，多迭代提高收敛 |
| `geom_consistency` | 几何一致性检查 | **必须开启** | 过滤 speckle 的误匹配 |
| `geom_consistency_regularizer` | 几何一致性正则化 | 0.3~0.5 | Speckle 场景建议偏高 |
| `min_num_consistent` | 最小一致性视角数 | 2~3 | 5 相机场景的合理值 |
| `max_image_size` | 处理分辨率上限 | -1 (不降采样) | 保留 DIC 所需的全分辨率 |

### 3.2 路径 B：PINN-Stereo 神经隐式方案（探索方向）

利用与 Step 3 统一的"神经隐式 + 投影约束 + ZNSSD"范式替代传统 PatchMatch。

#### 核心思想

将深度场表示为连续隐式函数 $D_c(u,v): \mathbb{R}^2 \to \mathbb{R}^+$，用多目光度一致性损失监督，通过梯度下降优化网络参数。

#### 网络结构

```
每台相机一个独立深度网络 D_c（共享架构，不共享参数）:

  Input: (u, v) ∈ ℝ²  (归一化到 [-1, 1])
    ↓
  Positional Encoding: γ(u,v), L=10 frequencies → 40 dims
    ↓
  MLP (256 × 4, skip connection at layer 2)
    ↓
  Output: d ∈ ℝ⁺  (通过 softplus 保证正值)
```

#### 损失函数

```
阶段 1: 稀疏初始化
  L_sparse = (1/N_sparse) Σ_c Σ_{(u,v,d)} |D_c(u,v) - d|²
  利用 COLMAP 稀疏点云提供初始监督

阶段 2: 多目光度一致性（核心）
  L_photo = Σ_c Σ_{j≠c} Σ_{(u,v)}
            ZNSSD(
                patch(I_c, (u,v), w),
                patch(I_j, project(unproject((u,v), D_c(u,v))), w)
            )
  
  投影操作本身嵌入了基线约束——(u_dst, v_dst) 自动在极线上

阶段 3: 正则化（可选）
  L_smooth = Σ_{(u,v)} |∇D_c(u,v)|²
  (MLP 的谱偏置已提供隐式平滑，此项通常不需要)
```

#### 硬约束嵌入

```python
def warp_pixel_to_camera(u, v, depth, K_src, R_src, t_src, K_dst, R_dst, t_dst):
    """
    从源相机的像素 (u,v) 经过 3D 空间到达目标相机。
    (u_dst, v_dst) 自动满足对极约束——无论 depth 取什么值。
    """
    # 反投影: 2D → 3D
    ray_dir = inv(K_src) @ [u, v, 1]
    P_cam = ray_dir * depth
    P_world = inv(R_src) @ (P_cam - t_src)

    # 投影: 3D → 目标相机 2D
    P_dst_cam = R_dst @ P_world + t_dst
    uv_dst_h = K_dst @ P_dst_cam
    u_dst, v_dst = uv_dst_h[:2] / uv_dst_h[2]

    return u_dst, v_dst
```

#### 两种路径对比

| 维度 | 路径 A (COLMAP PatchMatch) | 路径 B (PINN-Stereo) |
|------|---------------------------|----------------------|
| 成熟度 | 成熟、经过大量验证 | 研究阶段，需要验证 |
| 优化方式 | 逐像素独立随机搜索 | 全局梯度下降 |
| 平滑性 | 后处理（双边滤波等） | 网络谱偏置内置 |
| 多视图 | 两两匹配 + 融合 | 所有视图同时参与 |
| 子像素精度 | 二次插值 | 自然连续（网络输出） |
| GPU 利用率 | 低（逐像素，难以并行） | 高（batch 矩阵运算） |
| 可微性 | 不可微（离散搜索） | 端到端可微 |
| 空洞填充 | 需要后处理 | 网络自然插值 |
| 与 Step 3 的范式统一 | 不统一 | **完全统一**——代码高度复用 |

### 3.3 建议策略

**先走路径 A（COLMAP）跑通整个 pipeline，路径 B 作为后续改进方向。**

理由：
- COLMAP Dense 是成熟的，可以直接验证整个 NDF-DIC 框架
- 路径 B 的 PINN-Stereo 需要独立的开发、调试和验证周期
- 路径 B 的网络结构、投影模块、ZNSSD 模块可以在 Step 3 开发中复用

---

## 4. 关键衔接点

### 4.1 尺度恢复

COLMAP SfM 重建是 **up-to-scale** 的（任意尺度）。需要恢复真实物理尺度。

**方案**：拍摄棋盘格标定板

```
流程:
  1. COLMAP SfM 得到 up-to-scale 的重建
  2. 检测棋盘格角点在多目图像中的 2D 位置
  3. 三角化得到棋盘格角点的 3D 坐标 (COLMAP 尺度下)
  4. 已知棋盘格方格物理尺寸 (如 3mm)
  5. scale_factor = 物理距离 / COLMAP距离
  6. 所有 3D 坐标 × scale_factor → 真实尺度 (mm)
```

这个步骤可以在 Step 1 最后处理，不影响前面的流程。

### 4.2 稠密点云的 track 信息

稠密 MVS 的点云**没有显式的 track 信息**（COLMAP 的 track 来自 SfM 的特征匹配）。需要自己重建可见性矩阵。

```
对稠密点云中的每个点 x_i:
  for 每台相机 c:
    1. FOV Check:     (u,v) = π_c(x_i) 是否在图像范围内？
    2. 深度一致性:     |depth_map_c[u,v] - actual_depth_c(x_i)| < ε ?
    3. 法向量方向:     n_i · (o_c - x_i) > 0 ? (正面朝向相机)
  
  → vis_mask[i, c] = True 如果以上全部通过
```

**深度图必须被保存下来**（不只是融合后的点云），作为深度一致性检查的依据。

### 4.3 表面法向量

#### 必要性分析

| 使用场景 | 是否必需 | 替代方案 |
|---------|---------|---------|
| Step 1 可见性 | 辅助（非必需） | 多目光度一致性 loss 隐式处理遮挡 |
| Step 3 可见性 | 辅助（非必需） | FOV check 通常就够用 |
| Step 3 位移场正则化 | 可选但有用 | 点云局部 PCA 近似 |
| Step 3 应变计算 | 有用 | 后处理阶段再计算 |

**结论**：初期可以不做法向量判断，只用 FOV check。如果需要，用 open3d 的 `estimate_normals()` 对稠密点云做 PCA，几万点只需 ~100ms。

#### 计算方式

```
选项 A: COLMAP stereo_fusion 同时输出 normal map → 投影回点云
选项 B: 从融合后的点云计算 (open3d estimate_normals, k=20~30)
选项 C: 从深度图的梯度计算 (更直接，但噪声较大)

推荐: B 作为默认，A 作为可选增强
```

---

## 5. Speckle 模式的特殊考量

### 5.1 有利因素

- Speckle 是高对比度随机纹理 → SIFT 特征点丰富 → SfM 稳健
- 局部唯一性强 → 匹配歧义少
- 对光照变化不敏感（ZNSSD 天然鲁棒）

### 5.2 不利因素与缓解

| 风险 | 原因 | 缓解 |
|------|------|------|
| 斜面透视压缩 | Speckle 颗粒在不同视角下形状不同 | 大匹配窗口 (≥13px) 包含更多颗粒群 |
| 大 baseline 外观差异 | 镜面反射分量、倾斜模糊 | 几何一致性过滤 (≥3 视角一致) |
| 重复性幻觉 | 两个不同位置的 speckle 颗粒看起来相似 | 大窗口 + 几何一致性 + SOR 离群点去除 |
| Speckle 脱落/变形 | 加载过程中喷涂颗粒可能脱落 | 仅用参考图像做几何重建（不依赖变形图像） |

### 5.3 验证建议

在正式训练 Step 3 之前，先检查稠密点云质量：

```
□ 点云可视化：是否有明显的空洞或异常漂浮点？
□ 密度检查：mm² 内有几个点？是否足够密集？
□ 覆盖检查：ROI 区域是否被完全覆盖？
□ 法向量方向分布：是否基本一致（朝外）？
□ 重投影误差：用 COLMAP 的相机参数重投影，与原始图像对比
```

---

## 6. 实现清单

```
□ 1.1 确认 COLMAP CLI 或 pycolmap dense 模块可用
□ 1.2 实现图像去畸变 (image_undistorter)
□ 1.3 实现 PatchMatch Stereo 调用 + 参数调优
□ 1.4 实现 Stereo Fusion 调用
□ 1.5 保存深度图 (用于后续可见性判断)
□ 1.6 计算稠密点云法向量 (open3d)
□ 1.7 预计算可见性矩阵 vis_mask.npy
□ 1.8 确定物理尺度因子 (棋盘格)
□ 1.9 实现统计离群点移除 (SOR)
□ 1.10 输出元数据 meta.json
□ 1.11 在真实 speckle 图像上验证点云质量
```

---

## 7. 参考资料

- COLMAP 文档: https://colmap.github.io/
- PatchMatch Stereo: Bleyer et al., "PatchMatch Stereo - Stereo Matching with Slanted Support Windows", BMVC 2011
- [[neural-implicit-projection-zssd]] — 神经隐式 + 投影约束 + ZNSSD 详细解释
- [[new_plan]] — NDF-DIC 项目整体设计
