#!/usr/bin/env python
"""Config-driven NDeF-DIC pipeline entry point."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from ndef_dic.common import (
    get_by_path,
    load_config_dir,
    project_result_dir,
    resolve_data_path,
    resolve_result_path,
)


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NDeF-DIC config-driven pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing numbered YAML configs")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages: sfm,sfm2world,dense,deformation,all",
    )
    parser.add_argument("--clean-sfm", action="store_true", help="Clean SfM COLMAP database/reconstruction before running")
    parser.add_argument("--skip-sfm", action="store_true", help="Use existing SfM products")
    parser.add_argument(
        "--frames",
        type=str,
        default="all",
        help="Deformation frame ids to run, e.g. all or 01,03. Frame 01 is the first image after reference.",
    )
    return parser.parse_args()


def _natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _stage_set(raw: str) -> set[str]:
    stages = {s.strip().lower() for s in raw.split(",") if s.strip()}
    if not stages or "all" in stages:
        return {"sfm", "sfm2world", "dense", "deformation"}
    unknown = stages - {"sfm", "sfm2world", "dense", "deformation"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return stages


def _cfg(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    return get_by_path(config, dotted, default)


def _resolve_result(config: Mapping[str, Any], dotted: str, default: str | None = None) -> str:
    value = _cfg(config, dotted, default)
    if value is None:
        raise KeyError(f"Missing required config path: {dotted}")
    return str(resolve_result_path(config, value))


def _resolve_data(config: Mapping[str, Any], dotted: str, default: str | None = None) -> str:
    value = _cfg(config, dotted, default)
    if value is None:
        raise KeyError(f"Missing required config path: {dotted}")
    return str(resolve_data_path(config, value))


def discover_image_sequence(config: Mapping[str, Any]) -> tuple[str, list[dict[str, str]], str | None]:
    """Return reference stem, deformation frame records, and optional mask stem."""
    data_dir = Path(_cfg(config, "project.data_dir"))
    image_dir = data_dir / str(_cfg(config, "project.image_dir", "images"))
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    cam_dirs = sorted([p for p in image_dir.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
    if not cam_dirs:
        raise FileNotFoundError(f"No camera folders found in {image_dir}")

    sort_mode = _cfg(config, "project.image_sequence.sort_mode", "natural")
    key = (lambda p: _natural_key(p.name)) if sort_mode == "natural" else (lambda p: p.name)
    first_cam_images = sorted([p for p in cam_dirs[0].iterdir() if _is_image(p)], key=key)
    if len(first_cam_images) < 2:
        raise RuntimeError(f"Need at least reference + one deformation image in {cam_dirs[0]}")

    reference = first_cam_images[0].stem
    current_images = first_cam_images[1:]

    dense_auto_roi = bool(_cfg(config, "dense.auto_roi.enabled", True))
    last_rule = _cfg(config, "project.image_sequence.last_image_role_rule", "dense_auto_roi_enabled")
    mask_stem = None
    if last_rule == "always_mask" or (last_rule == "dense_auto_roi_enabled" and not dense_auto_roi):
        if current_images:
            mask_stem = current_images[-1].stem
            current_images = current_images[:-1]
    elif last_rule != "always_deformation" and last_rule != "dense_auto_roi_enabled":
        raise ValueError(f"Unknown last_image_role_rule: {last_rule}")

    frames = []
    for idx, image_path in enumerate(current_images, start=1):
        frames.append(
            {
                "frame_id": f"{idx:02d}",
                "reference_name": reference,
                "current_name": image_path.stem,
            }
        )

    _validate_camera_images(cam_dirs, reference, [f["current_name"] for f in frames], mask_stem)
    return reference, frames, mask_stem


def _validate_camera_images(cam_dirs: list[Path], reference: str, currents: list[str], mask: str | None) -> None:
    required = [reference, *currents]
    if mask:
        required.append(mask)
    for cam_dir in cam_dirs:
        stems = {p.stem for p in cam_dir.iterdir() if _is_image(p)}
        missing = [stem for stem in required if stem not in stems]
        if missing:
            raise FileNotFoundError(f"{cam_dir} missing image stems: {missing}")


def select_frames(frames: list[dict[str, str]], raw: str) -> list[dict[str, str]]:
    if raw.strip().lower() == "all":
        return frames
    wanted = {s.strip() for s in raw.split(",") if s.strip()}
    selected = [frame for frame in frames if frame["frame_id"] in wanted]
    missing = wanted - {frame["frame_id"] for frame in selected}
    if missing:
        raise ValueError(f"Requested frames not available: {sorted(missing)}")
    return selected


def run_sfm(config: Mapping[str, Any], *, skip: bool, clean: bool):
    from ndef_dic.sfm.reference_sfm import reference_sfm_exists, run_reference_sfm

    data_dir = str(_cfg(config, "project.data_dir"))
    out_dir = _resolve_result(config, "sfm.output_dir", "sfm")
    if skip:
        if not reference_sfm_exists(data_dir, out_dir):
            raise FileNotFoundError(f"SfM products not found in {out_dir}")
        print(f"[SfM] Existing products found in {out_dir}")
        return None

    return run_reference_sfm(
        data_dir=data_dir,
        image_dir=str(_cfg(config, "sfm.image_dir", _cfg(config, "project.image_dir", "images"))),
        ref_mode="first",
        output_dir=out_dir,
        reference_camera=str(_cfg(config, "sfm.reference_camera", "cam_0")),
        max_features=int(_cfg(config, "sfm.feature.max_features", 8192)),
        first_octave=int(_cfg(config, "sfm.feature.first_octave", 0)),
        cross_check=bool(_cfg(config, "sfm.matching.cross_check", False)),
        min_num_matches=int(_cfg(config, "sfm.mapping.min_num_matches", 8)),
        min_model_size=int(_cfg(config, "sfm.mapping.min_model_size", 3)),
        ba_global_max_refinements=int(_cfg(config, "sfm.mapping.ba_global_max_refinements", 5)),
        clean=clean or bool(_cfg(config, "sfm.output.clean", False)),
        max_reproj_error=float(_cfg(config, "sfm.mapping.max_reproj_error", 4.0)),
        dpi=int(_cfg(config, "sfm.output.dpi", 180)),
    )


def run_sfm2world(config: Mapping[str, Any]) -> dict[str, str]:
    from ndef_dic.sfm2world import ChessboardScaleConfig, run_chessboard_scale

    sfm_cfg = _cfg(config, "sfm2world", {})
    return run_chessboard_scale(
        ChessboardScaleConfig(
            data_dir=str(_cfg(config, "project.data_dir")),
            sfm_dir=_resolve_result(config, "sfm.output_dir", "sfm"),
            image_dir=str(sfm_cfg.get("image_dir", _cfg(config, "project.calibrate_image_dir", "calibrate_images"))),
            image_name=None,
            output_dir=_resolve_result(config, "sfm2world.output_dir", "sfm2world"),
            inner_cols=int(sfm_cfg.get("inner_cols", 9)),
            inner_rows=int(sfm_cfg.get("inner_rows", 7)),
            square_size=float(sfm_cfg.get("square_size", 10.0)),
            pair_selection=str(sfm_cfg.get("pair_selection", "middle")),
            subpix_window=int(sfm_cfg.get("subpix_window", 11)),
            min_common_corners=int(sfm_cfg.get("min_common_corners", 12)),
            max_reprojection_error_px=float(sfm_cfg.get("max_reprojection_error_px", 3.0)),
            save_overlays=bool(sfm_cfg.get("save_overlays", True)),
        )
    )


def run_dense(config: Mapping[str, Any]) -> dict[str, str]:
    from ndef_dic.dense import (
        DenseReconstructionConfig,
        DenseZNSSDConfig,
        DepthInitConfig,
        ReconstructionDatasetConfig,
        SurfaceSamplerConfig,
        run_dense_reconstruction,
        run_dense_znssd,
        run_model_init,
        run_reconstruction_dataset,
        run_surface_sampler,
    )

    data_dir = str(_cfg(config, "project.data_dir"))
    sfm_dir = _resolve_result(config, "sfm.output_dir", "sfm")
    dense_root = _resolve_result(config, "dense.output_dir", "dense")
    auto_roi_enabled = bool(_cfg(config, "dense.auto_roi.enabled", True))

    outputs: dict[str, str] = {}
    model_init_dir = str(Path(dense_root) / "model_init")
    roi_dir = _resolve_result(config, "dense.auto_roi.output_dir", "dense/auto_roi")
    outputs.update(
        {
            f"model_init_{k}": v
            for k, v in run_model_init(
                DepthInitConfig(
                    data_dir=data_dir,
                    sfm_dir=sfm_dir,
                    output_dir=model_init_dir,
                    roi_dir=roi_dir,
                    use_external_roi=not auto_roi_enabled,
                    external_roi_dir=None if auto_roi_enabled else roi_dir,
                    epochs=int(_cfg(config, "dense.model_init.epochs", 3000)),
                    smooth_weight=float(_cfg(config, "dense.model_init.smooth_weight", 1e-4)),
                    sparse_filter_enabled=bool(_cfg(config, "dense.model_init.sparse_filter.enabled", True)),
                    sparse_filter_min_track_length=int(
                        _cfg(config, "dense.model_init.sparse_filter.min_track_length", 2)
                    ),
                    sparse_filter_max_reproj_error=_cfg(
                        config, "dense.model_init.sparse_filter.max_reproj_error", None
                    ),
                    sparse_filter_radius_mad_thresh=float(
                        _cfg(config, "dense.model_init.sparse_filter.radius_mad_thresh", 8.0)
                    ),
                    sparse_filter_knn_k=int(_cfg(config, "dense.model_init.sparse_filter.knn_k", 8)),
                    sparse_filter_knn_mad_thresh=float(
                        _cfg(config, "dense.model_init.sparse_filter.knn_mad_thresh", 8.0)
                    ),
                )
            ).items()
        }
    )

    dataset_dir = str(Path(dense_root) / "reconstruction_dataset")
    outputs.update(
        {
            f"dataset_{k}": v
            for k, v in run_reconstruction_dataset(
                ReconstructionDatasetConfig(
                    data_dir=data_dir,
                    sfm_dir=sfm_dir,
                    model_init_dir=model_init_dir,
                    output_dir=dataset_dir,
                )
            ).items()
        }
    )

    znssd_dir = _resolve_result(config, "dense.znssd_opt.output_dir", "dense/znssd_opt")
    outputs.update(
        {
            f"znssd_{k}": v
            for k, v in run_dense_znssd(
                DenseZNSSDConfig(
                    data_dir=data_dir,
                    sfm_dir=sfm_dir,
                    model_init_dir=model_init_dir,
                    dataset_dir=dataset_dir,
                    output_dir=znssd_dir,
                    epochs=int(_cfg(config, "dense.znssd_opt.epochs", 100)),
                    max_steps_per_epoch=None,
                )
            ).items()
        }
    )

    reconstruction_dir = _resolve_result(config, "dense.reconstruction_dense.output_dir", "dense/reconstruction_dense")
    outputs.update(
        {
            f"reconstruction_{k}": v
            for k, v in run_dense_reconstruction(
                DenseReconstructionConfig(
                    data_dir=data_dir,
                    sfm_dir=sfm_dir,
                    model_init_dir=model_init_dir,
                    znssd_dir=znssd_dir,
                    output_dir=reconstruction_dir,
                )
            ).items()
        }
    )

    sampler_output = _cfg(config, "dense.surface_sampler.output", "dense/surface_sampler/deformation_surface_dataset.npz")
    sampler_dir = str(resolve_result_path(config, sampler_output).parent)
    outputs.update(
        {
            f"surface_{k}": v
            for k, v in run_surface_sampler(
                SurfaceSamplerConfig(
                    data_dir=data_dir,
                    sfm_dir=sfm_dir,
                    model_init_dir=model_init_dir,
                    reconstruction_dense_dir=reconstruction_dir,
                    output_dir=sampler_dir,
                    min_visible_cameras=int(_cfg(config, "dense.surface_sampler.min_visible_cameras", 2)),
                    relative_sample_spacing=float(
                        _cfg(config, "dense.surface_sampler.relative_sample_spacing", 0.006)
                    ),
                )
            ).items()
        }
    )
    return outputs


def run_precalculation_for_frame(config: Mapping[str, Any], frame: Mapping[str, str]) -> dict[str, str]:
    from ndef_dic.precalculation import PatchDICPrecalcConfig, run_patch_dic_precalculation

    if not bool(_cfg(config, "precalculation.enabled", True)):
        print(f"[Precalc] disabled for frame {frame['frame_id']}")
        return {}
    method = str(_cfg(config, "precalculation.method", "patch_dic_sparse"))
    if method != "patch_dic_sparse":
        raise NotImplementedError(f"Unsupported precalculation method: {method}")

    base_output = resolve_result_path(
        config,
        _cfg(config, "precalculation.patch_dic_sparse.output_dir", "deformation/precalculation/patch_dic_sparse"),
    )
    output_dir = base_output / frame["frame_id"]
    patch_cfg = _cfg(config, "precalculation.patch_dic_sparse", {})
    return run_patch_dic_precalculation(
        PatchDICPrecalcConfig(
            data_dir=str(_cfg(config, "project.data_dir")),
            sfm_dir=_resolve_result(config, "sfm.output_dir", "sfm"),
            model_init_dir=str(resolve_result_path(config, "dense/model_init")),
            surface_dataset_path=_resolve_result(
                config,
                "deformation.input.surface_dataset",
                "dense/surface_sampler/deformation_surface_dataset.npz",
            ),
            output_dir=str(output_dir),
            image_dir=str(_cfg(config, "project.image_dir", "images")),
            reference_name=frame["reference_name"],
            current_name=frame["current_name"],
            points_per_camera=int(patch_cfg.get("points_per_camera", 300)),
            patch_radius=int(patch_cfg.get("patch_radius", 10)),
            cross_search_radius=int(patch_cfg.get("cross_search_radius", 40)),
            temporal_search_radius=int(patch_cfg.get("temporal_search_radius", 8)),
            ncc_threshold_cross=float(patch_cfg.get("ncc_threshold_cross", 0.45)),
            ncc_threshold_temporal=float(patch_cfg.get("ncc_threshold_temporal", 0.55)),
            match_batch_size=int(patch_cfg.get("match_batch_size", 64)),
        )
    )


def run_deformation_for_frame(config: Mapping[str, Any], frame: Mapping[str, str]) -> dict[str, str]:
    from ndef_dic.deformation import DeformationTrainingConfig, run_deformation_training

    deformation_output = resolve_result_path(config, _cfg(config, "deformation.output_dir", "deformation")) / frame["frame_id"]
    precalc_scale = (
        resolve_result_path(
            config,
            _cfg(config, "precalculation.patch_dic_sparse.output_dir", "deformation/precalculation/patch_dic_sparse"),
        )
        / frame["frame_id"]
        / "displacement_scale.json"
    )

    source = str(_cfg(config, "deformation.displacement_scale.source", "precalculation"))
    manual_scale = _cfg(config, "deformation.displacement_scale.value")
    displacement_scale = float(manual_scale) if source == "manual" and manual_scale is not None else None
    displacement_scale_path = str(precalc_scale) if source == "precalculation" else None
    if source == "none":
        displacement_scale = 1.0
        displacement_scale_path = None

    batch_size = _cfg(config, "deformation.training.batch_size", "auto")
    if isinstance(batch_size, str) and batch_size != "auto":
        batch_size = int(batch_size)

    return run_deformation_training(
        DeformationTrainingConfig(
            data_dir=str(_cfg(config, "project.data_dir")),
            sfm_dir=_resolve_result(config, "deformation.input.sfm_dir", "sfm"),
            surface_dataset_path=_resolve_result(
                config,
                "deformation.input.surface_dataset",
                "dense/surface_sampler/deformation_surface_dataset.npz",
            ),
            output_dir=str(deformation_output),
            image_dir=str(_cfg(config, "deformation.input.image_dir", _cfg(config, "project.image_dir", "images"))),
            reference_name=frame["reference_name"],
            current_name=frame["current_name"],
            hidden_dim=int(_cfg(config, "deformation.network.hidden_dim", 32)),
            hidden_layers=int(_cfg(config, "deformation.network.hidden_layers", 5)),
            use_positional_encoding=bool(_cfg(config, "deformation.network.use_positional_encoding", False)),
            positional_encoding_frequencies=int(_cfg(config, "deformation.network.positional_encoding_frequencies", 6)),
            displacement_scale=displacement_scale,
            displacement_scale_path=displacement_scale_path,
            displacement_scale_stat=str(_cfg(config, "deformation.displacement_scale.stat", "mean")),
            sfm2world_scale_path=str(resolve_result_path(config, _cfg(config, "scale.sfm2world_scale_json", "sfm2world/sfm2world_scale.json"))),
            lambda_smooth=float(_cfg(config, "deformation.loss.lambda_smooth", 0.0)),
            photometric_loss=str(_cfg(config, "deformation.loss.photometric", "mse")),
            patch_radius=int(_cfg(config, "deformation.loss.patch_radius", 2)),
            min_valid_patch_ratio=float(_cfg(config, "deformation.loss.min_valid_patch_ratio", 1.0)),
            invalid_patch_penalty=float(_cfg(config, "deformation.loss.invalid_patch_penalty", 0.05)),
            znssd_eps=float(_cfg(config, "deformation.loss.znssd_eps", 1.0e-6)),
            epochs=int(_cfg(config, "deformation.training.epochs", 100)),
            lr=float(_cfg(config, "deformation.training.lr", 1.0e-3)),
            weight_decay=float(_cfg(config, "deformation.training.weight_decay", 0.0)),
            batch_size=batch_size,
            auto_batch_start=int(_cfg(config, "deformation.training.auto_batch_start", 1024)),
            auto_batch_max=_cfg(config, "deformation.training.auto_batch_max"),
            memory_fraction=float(_cfg(config, "deformation.training.memory_fraction", 0.80)),
            max_steps_per_epoch=_cfg(config, "deformation.training.max_steps_per_epoch"),
            log_interval=int(_cfg(config, "deformation.training.log_interval", 10)),
            max_visualization_points=int(_cfg(config, "deformation.export.max_visualization_points", 60000)),
            seed=int(_cfg(config, "deformation.training.seed", 23)),
            device=str(_cfg(config, "deformation.training.device", "auto")),
        )
    )


def main() -> None:
    args = parse_args()
    config = load_config_dir(args.config_dir)
    stages = _stage_set(args.stages)
    reference_name, frames, mask_name = discover_image_sequence(config)
    frames = select_frames(frames, args.frames)

    print("\n" + "=" * 70)
    print("  NDeF-DIC Config Pipeline")
    print(f"  Config dir: {_path_display(Path(args.config_dir))}")
    print(f"  Case:       {_cfg(config, 'project.name')}")
    print(f"  Data:       {_cfg(config, 'project.data_dir')}")
    print(f"  Result:     {project_result_dir(config)}")
    print(f"  Stages:     {', '.join(sorted(stages))}")
    print(f"  Reference:  {reference_name}")
    print(f"  Frames:     {', '.join(f['frame_id'] + ':' + f['current_name'] for f in frames) or '(none)'}")
    if mask_name:
        print(f"  Mask image:  {mask_name}")
    print("=" * 70 + "\n")

    total_start = time.time()

    if "sfm" in stages:
        _run_timed("1. SfM", lambda: run_sfm(config, skip=args.skip_sfm, clean=args.clean_sfm))

    if "sfm2world" in stages:
        _run_timed("2. SfM-to-world scale", lambda: run_sfm2world(config))

    if "dense" in stages:
        _run_timed("3. Dense reference surface", lambda: run_dense(config))

    if "deformation" in stages:
        if not frames:
            raise RuntimeError("No deformation frames found after applying image-sequence rules.")
        for frame in frames:
            print("\n" + "#" * 70)
            print(f"# Frame {frame['frame_id']} current={frame['current_name']}")
            print("#" * 70)
            _run_timed("4a. Patch-DIC precalculation", lambda f=frame: run_precalculation_for_frame(config, f))
            _run_timed("4b. Neural deformation", lambda f=frame: run_deformation_for_frame(config, f))

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  Pipeline complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)


def _run_timed(title: str, func):
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)
    start = time.time()
    outputs = func()
    print(f"\n[{title}] Done in {time.time() - start:.0f}s")
    if isinstance(outputs, Mapping):
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    elif outputs is not None:
        if hasattr(outputs, "cam_names"):
            print(f"  Cameras: {len(outputs.cam_names)}")
        if hasattr(outputs, "points3D"):
            print(f"  Sparse points: {len(outputs.points3D)}")
        if hasattr(outputs, "observations"):
            print(f"  Observations: {len(outputs.observations['uv'])}")
    return outputs


def _path_display(path: Path) -> str:
    return str(path)


if __name__ == "__main__":
    main()
