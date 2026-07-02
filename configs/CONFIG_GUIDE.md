# NDeF-DIC Configuration Guide

This guide explains which configuration values usually need to be changed for a new experiment, and which values should normally remain unchanged.

The main idea is simple:

- Change case identity, data location, camera count, and chessboard geometry.
- Keep output paths relative to `project.result_dir`.
- Keep dense, pre-calculation, and neural-network hyperparameters unchanged unless you are deliberately running an ablation or debugging a failure case.

## 1. Files

Configuration files are loaded in numeric order:

```text
1_project.yaml
2_sfm.yaml
3_dense.yaml
4_sfm2world.yaml
5_precalculation.yaml
6_deformation.yaml
```

The numbered order mirrors the pipeline:

```text
sfm -> sfm2world -> dense -> per-frame precalculation -> per-frame deformation
```

## 2. Parameters You Usually Must Change

### `1_project.yaml`

These are the most important case-specific fields.

```yaml
project:
  name: CylinderDIC
  data_dir: case/CylinderDIC
  result_dir: result
  image_dir: images
  calibrate_image_dir: calibrate_images
  num_cameras: 12
```

Change:

- `project.name`
  - Human-readable case name.
  - Example: `CylinderDIC`, `PlateDIC`, `BeamTest01`.

- `project.data_dir`
  - Root folder of the case.
  - This is the main path you change when switching datasets.
  - Example:
    ```yaml
    data_dir: case/MyNewCase
    ```

- `project.image_dir`
  - Folder under `project.data_dir` that stores DIC images.
  - Usually keep as `images` if your data structure follows:
    ```text
    case/MyNewCase/images/cam_0/001.bmp
    ```

- `project.calibrate_image_dir`
  - Folder under `project.data_dir` that stores chessboard images.
  - Usually keep as `calibrate_images`.

- `project.num_cameras`
  - Number of camera folders expected in the case.

Usually keep:

- `project.result_dir: result`
  - This is relative to `project.data_dir`.
  - Do not change it unless you intentionally want result files outside the case folder.

- `project.image_sequence`
  - The default convention is:
    - first image = reference image
    - following images = deformation images
    - final image becomes mask only when `dense.auto_roi.enabled=false`

## 3. Chessboard Scale Parameters

### `4_sfm2world.yaml`

These values should match the physical calibration board.

```yaml
sfm2world:
  image_dir: calibrate_images
  inner_cols: 9
  inner_rows: 7
  square_size: 10.0
  pair_selection: middle
```

Change:

- `inner_cols`
  - Number of inner chessboard corners along board width.
  - If the board has 10 squares along width, `inner_cols = 9`.

- `inner_rows`
  - Number of inner chessboard corners along board height.
  - If the board has 8 squares along height, `inner_rows = 7`.

- `square_size`
  - Physical square size in `project.units.world_length`.
  - Current unit is `mm`, so `square_size: 10.0` means 10 mm.

- `image_dir`
  - Usually keep equal to `project.calibrate_image_dir`.
  - Each `cam_x` folder should contain exactly one chessboard image.

Usually keep:

- `pair_selection: middle`
  - Selects the middle pair among cameras that see the board.
  - Good default for the current circular camera layout.

Advanced option:

- `pair_selection: max_baseline`
  - May improve triangulation stability if the chessboard is clearly visible from a wide-baseline pair.
  - Can become less reliable if the selected cameras see the board at strong oblique angles.

Usually keep:

- `subpix_window`
- `min_common_corners`
- `max_reprojection_error_px`
- `save_overlays`

Only change these if chessboard detection is unstable or if reprojection filtering removes too many corners.

## 4. SfM Parameters

### `2_sfm.yaml`

Most users should not need to change SfM parameters.

Usually change only:

- `reference_camera`
  - Default is `cam_0`.
  - This defines the SfM-world axis convention.
  - Keep `cam_0` unless you intentionally want another camera to define the coordinate orientation.

Usually keep:

- `image_dir: images`
- `ref_mode: first`
- `output_dir: sfm`
- `feature.max_features`
- `feature.first_octave`
- `matching.cross_check`
- `mapping.min_num_matches`
- `mapping.ba_global_max_refinements`
- `mapping.max_reproj_error`

When to change SfM parameters:

- Increase `feature.max_features` if the object texture is weak or camera overlap is poor.
- Try `first_octave: -1` if features are very small.
- Increase `min_num_matches` only if false matches dominate.
- Change `reference_camera` only when coordinate-axis convention matters.

## 5. Dense Reconstruction Parameters

### `3_dense.yaml`

Most dense settings should remain unchanged for normal use.

Usually change:

- `dense.auto_roi.enabled`
  - `true`: automatically infer ROI from SfM features and speckle structure.
  - `false`: use the last image in each camera folder as a mask image.

Usually keep:

- all `output_dir` values
- `model_init.model`
- `model_init.depth_normalization`
- `znssd_opt.patch_radius`
- `surface_sampler.min_visible_cameras`
- `surface_sampler.relative_sample_spacing`

When to change:

- Increase `surface_sampler.relative_sample_spacing` to produce fewer surface points.
- Decrease it to produce more surface points, at higher training cost.
- Increase `min_visible_cameras` if you want stricter multi-view consistency, but expect fewer points.

## 6. Pre-Calculation Parameters

### `5_precalculation.yaml`

The current pre-calculation module estimates displacement scale only. It is not used as sparse displacement supervision.

Usually keep:

```yaml
precalculation:
  enabled: true
  method: patch_dic_sparse
  use_as_displacement_scale: true
  use_as_sparse_initialization: false
```

Usually keep these defaults:

- `points_per_camera`
- `patch_radius`
- `cross_search_radius`
- `temporal_search_radius`
- `ncc_threshold_cross`
- `ncc_threshold_temporal`
- `match_batch_size`
- `scale_stat: mean`

When to change:

- Increase `points_per_camera` if the displacement scale estimate is noisy.
- Increase `temporal_search_radius` if deformation is larger than expected.
- Increase `patch_radius` if speckle texture is weak.
- Use `scale_stat: median` if occasional outliers remain after filtering.

Do not enable sparse initialization unless the deformation loss is explicitly redesigned to use sparse displacement supervision.

## 7. Deformation Training Parameters

### `6_deformation.yaml`

For normal use, the network architecture does not need much tuning.

Usually keep:

```yaml
network:
  hidden_dim: 32
  hidden_layers: 5
  activation: tanh
  use_positional_encoding: false
```

Why:

- This architecture is intentionally small and smooth.
- With `use_positional_encoding=false`, the MLP itself provides smoothness.
- Current CylinderDIC tests show stable displacement magnitude and photometric convergence.

Usually change:

- `training.epochs`
  - Increase if the loss is still decreasing.

- `training.lr`
  - Current good value: `0.003`.
  - If training oscillates, try `0.001`.
  - If convergence is too slow, test `0.01` carefully.

- `training.batch_size`
  - Use a fixed value if the full surface fits in memory.
  - Use `auto` for automatic memory-based selection.

- `loss.photometric`
  - `mse`: use when illumination is stable.
  - `znssd`: use when local brightness or contrast changes are significant.

Usually keep:

- `loss.patch_radius: 2`
  - This means 5x5 patches.

- `loss.invalid_patch_penalty`

- `displacement_scale.source: precalculation`

- `displacement_scale.stat: mean`

- `export.apply_sfm2world_scale: true`

Use positional encoding only when:

- deformation has higher spatial frequency,
- the low-capacity MLP underfits,
- and you are willing to enable nonzero `lambda_smooth`.

Recommended PE setting if needed:

```yaml
network:
  use_positional_encoding: true

loss:
  lambda_smooth: 1.0e-5
```

## 8. Output Paths

Most output paths should not be changed.

They are relative to `project.result_dir`, which is itself relative to `project.data_dir` by default:

```yaml
project:
  data_dir: case/CylinderDIC
  result_dir: result
```

This resolves to:

```text
case/CylinderDIC/result
```

Therefore, keep paths like:

```yaml
sfm:
  output_dir: sfm

sfm2world:
  output_dir: sfm2world

deformation:
  output_dir: deformation
```

Do not rewrite them as full case paths unless you intentionally want outputs outside the case result folder.

## 9. Minimal Checklist for a New Case

For a new dataset, usually edit only:

```yaml
# 1_project.yaml
project:
  name: MyCase
  data_dir: case/MyCase
  num_cameras: 12

# 4_sfm2world.yaml
sfm2world:
  inner_cols: 9
  inner_rows: 7
  square_size: 10.0

# 3_dense.yaml, only if needed
dense:
  auto_roi:
    enabled: true
```

Then run:

```powershell
python run.py --stages all
```

For only deformation after geometry is already available:

```powershell
python run.py --stages deformation --frames 01
```

## 10. When Not to Change Parameters

Avoid changing these unless there is a clear reason:

- output directories
- COLMAP/SfM thresholds
- neural network width/depth
- pre-calculation search radii
- dense model checkpoint names
- `sfm2world` output path
- `deformation.export.apply_sfm2world_scale`

Changing too many parameters at once makes it difficult to diagnose whether an error comes from geometry, scale recovery, surface sampling, or deformation optimization.
