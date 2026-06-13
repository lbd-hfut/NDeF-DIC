# Cylinder DIC Simulation — Validation Report

**Output directory:** `C:/01project/NDeF-DIC/case/CylinderDIC`

---

## 1. Executive Summary

**Overall Assessment:** ⚠️ **REVIEW** — Some metrics need attention (see details below).

| Category | Key Metric | Value | Status |
|----------|-----------|-------|--------|
| Speckle Quality | MIG (mean) | 54.0 | ✅ |
| Coverage | Mean ratio | 54.86% | ✅ |
| Speckle Size | Mean FWHM | 8.4 px | ⚠️ |
| View Overlap | Adjacent overlap | 0.0% | ⚠️ |
| Peak Sharpness | Correlation peak | 68.6 | — |

---

## 2. Camera Configuration

| Parameter | Value |
|-----------|-------|
| Number of cameras | 12 |
| Mean camera distance | 380.0 mm |
| Horizontal FOV per camera | 34.5° |
| Cylinder angular extent per camera | 24.3° |
| Angular step between cameras | 30.0° |
| Cylinder surface overlap angle | -5.7° |
| Overlap ratio (relative to cylinder) | 0.0% |
| Overlap arc length on surface | 0.0 mm |
| Mean adjacent baseline | 196.7 mm |
| Mean triangulation angle | 29.0° |
| Min / Max baseline | 196.7 / 760.0 mm |

### Baseline Distances (adjacent cameras)

| Camera Pair | Baseline [mm] | Triangulation Angle [°] |
|-------------|---------------|------------------------|
| 0 → 1 | 196.7 | 29.0 |
| 1 → 2 | 196.7 | 29.0 |
| 2 → 3 | 196.7 | 29.0 |
| 3 → 4 | 196.7 | 29.0 |
| 4 → 5 | 196.7 | 29.0 |
| 5 → 6 | 196.7 | 29.0 |
| 6 → 7 | 196.7 | 29.0 |
| 7 → 8 | 196.7 | 29.0 |
| 8 → 9 | 196.7 | 29.0 |
| 9 → 10 | 196.7 | 29.0 |
| 10 → 11 | 196.7 | 29.0 |
| 11 → 0 | 196.7 | 29.0 |

![Camera Geometry](camera_geometry.png)

---

## 3. Speckle Quality Analysis

### 3.1 Mean Intensity Gradient (MIG)

MIG is the primary DIC quality metric. Higher values enable more precise sub-pixel matching.
A value above 15 gray levels/pixel is considered adequate for DIC.

| Metric | Reference (001.bmp) | Deformed (002.bmp) |
|--------|---------------------|---------------------|
| Mean MIG | **54.0** | 54.3 |
| Std of means | 2.3 | — |
| Min MIG | 51.3 | — |
| Max MIG | 57.5 | — |
| Ref→Def stability | -0.34 | — |

### 3.2 Per-Camera MIG

| Camera | MIG (ref) | MIG (def) | Δ |
|--------|-----------|-----------|----|
| 0 | 52.5 | 52.6 | -0.1 |
| 1 | 51.7 | 51.7 | -0.1 |
| 2 | 51.3 | 52.8 | -1.5 |
| 3 | 53.2 | 54.0 | -0.9 |
| 4 | 57.0 | 57.1 | -0.1 |
| 5 | 56.3 | 56.8 | -0.5 |
| 6 | 56.6 | 56.1 | +0.6 |
| 7 | 57.5 | 57.5 | -0.0 |
| 8 | 55.5 | 55.4 | +0.1 |
| 9 | 53.2 | 54.5 | -1.3 |
| 10 | 51.3 | 51.7 | -0.4 |
| 11 | 51.8 | 51.7 | +0.0 |

### 3.3 Speckle Size (Autocorrelation FWHM)

| Metric | Value | Ideal Range |
|--------|-------|-------------|
| Mean speckle size | **8.4 px** | 3–5 px |
| Std of sizes | 0.4 px | — |
| Mean peak sharpness | 68.6 | > 5 |

| Camera | Speckle Size [px] | FWHM X | FWHM Y | Anisotropy | Peak Sharpness |
|--------|-------------------|--------|--------|------------|----------------|
| 0 | 8.2 | 8.0 | 8.5 | 0.95 | 62.5 |
| 1 | 8.2 | 8.4 | 7.9 | 1.06 | 65.6 |
| 2 | 8.3 | 8.0 | 8.6 | 0.94 | 69.5 |
| 3 | 8.0 | 7.9 | 8.0 | 0.99 | 71.7 |
| 4 | 8.4 | 8.4 | 8.4 | 1.00 | 68.6 |
| 5 | 8.2 | 8.1 | 8.3 | 0.98 | 67.6 |
| 6 | 8.7 | 8.7 | 8.7 | 1.00 | 64.9 |
| 7 | 7.7 | 8.0 | 7.4 | 1.08 | 82.9 |
| 8 | 8.7 | 9.0 | 8.3 | 1.08 | 69.0 |
| 9 | 8.3 | 8.1 | 8.5 | 0.95 | 67.5 |
| 10 | 8.9 | 9.1 | 8.8 | 1.03 | 65.6 |
| 11 | 9.0 | 9.4 | 8.5 | 1.10 | 67.3 |

![MIG Comparison](mig_comparison.png)

![Speckle Analysis](speckle_analysis.png)

---

## 4. Coverage & Intensity

| Metric | Value |
|--------|-------|
| Mean coverage | **54.9%** |
| Coverage std | 0.1% |
| Coverage range | [54.6%, 55.1%] |
| Mean intensity | 80.9 |
| Mean within-image std | 27.9 |
| Global dynamic range | 214.0 |

| Camera | Coverage | Mean Int. | Std Int. | Range |
|--------|----------|-----------|----------|-------|
| 0 | 54.78% | 79.4 | 27.1 | [36, 250] |
| 1 | 54.81% | 78.9 | 26.6 | [36, 250] |
| 2 | 54.78% | 78.7 | 26.4 | [36, 250] |
| 3 | 54.88% | 80.3 | 27.4 | [36, 250] |
| 4 | 55.11% | 83.7 | 29.4 | [36, 250] |
| 5 | 55.06% | 83.2 | 29.1 | [36, 250] |
| 6 | 55.00% | 83.3 | 29.3 | [36, 250] |
| 7 | 54.98% | 83.9 | 29.8 | [36, 250] |
| 8 | 54.90% | 82.2 | 28.7 | [36, 250] |
| 9 | 54.80% | 80.2 | 27.6 | [36, 250] |
| 10 | 54.63% | 78.5 | 26.6 | [36, 250] |
| 11 | 54.64% | 78.8 | 26.8 | [36, 250] |

![Coverage Overview](camera_thumbnails.png)

![Intensity Histograms](intensity_histograms.png)

---

## 5. Ground Truth Deformation

**Deformation type:** expansion
**Deformation magnitude:** 0.5

### 5.1 Displacement Statistics

| Component | Mean [mm] | Std [mm] | Min [mm] | Max [mm] |
|-----------|-----------|----------|----------|----------|
| U (X) | 0.0000 | 0.3536 | -0.5000 | 0.5000 |
| V (Y) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| W (Z) | -0.0001 | 0.3535 | -0.5000 | 0.5000 |
| Magnitude | **0.5000** | 0.0000 | 0.5000 | 0.5000 |

### 5.2 Cylindrical Decomposition

| Component | Mean | Max |
|-----------|------|-----|
| Radial dr | 0.5000 mm | 0.5000 mm |
| Tangential dθ | 0.0000° | 0.0000° |
| Axial dy | 0.0000 mm | 0.0000 mm |

### 5.3 Surface Sampling

| Metric | Value |
|--------|-------|
| Surface area | 60318.6 mm² |
| Number of points | 15,000,000 |
| Point density | 248.68 pts/mm² |
| Avg point spacing | 0.063 mm |

![Deformation Field](deformation_field.png)

![Reference vs Deformed](ref_vs_def.png)

---

## 6. Quality Checklist for Step 1 Readiness

- ✅ **MIG > 15 gray levels/px** — All cameras have sufficient texture for DIC matching (mean MIG = 54.0)
- ✅ **Coverage > 10%** — All cameras have adequate surface coverage (mean = 54.9%)
- ⚠️ **Speckle size 3–5 px** — Speckle size (8.4 px) outside ideal 3–5 px range
- ⚠️ **View overlap > 40%** — View overlap (0.0%) below 40% — may cause gaps in 3D reconstruction
- ✅ **Mean intensity > 30** — Adequate brightness for DIC (mean = 80.9)
- ✅ **Coverage uniformity (std < 15%)** — Coverage is uniform across cameras (std = 0.1%)

---

## 7. Output Files

| Path | Description |
|------|-------------|
| `images/cam_*/001.bmp` | Reference images (12 cameras) |
| `images/cam_*/002.bmp` | Deformed images (12 cameras) |
| `calibration/cameras.mat` | Camera intrinsics & extrinsics |
| `calibration/points3D.mat` | Sparse surface points |
| `ground_truth/points_ref.npy` | Ground truth reference points |
| `ground_truth/points_def_step1.npy` | Ground truth deformed points |
| `ground_truth/displacement_step1.npy` | Ground truth displacement field |
| `ground_truth/meta.json` | Simulation parameters |
