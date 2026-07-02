#!/usr/bin/env python
"""NDeF-DIC command line entry point without config files.

For the current development phase we keep parameters explicit in this file and
focus on validating the SfM module.  Project-wide configuration will be
reintroduced later after the research pipeline stabilizes.
"""

from __future__ import annotations

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="NDeF-DIC pipeline entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=str, default="1", help="Comma-separated steps to run")
    parser.add_argument("--data-dir", type=str, default="case/CylinderDIC", help="Case directory")
    parser.add_argument("--image-dir", type=str, default="images", help="Image folder inside data-dir")
    parser.add_argument("--ref-name", type=str, default="001", help="Reference image basename")
    parser.add_argument("--reference-camera", type=str, default="cam_0", help="World-axis reference camera")
    parser.add_argument(
        "--deformation-loss",
        type=str,
        default="mse",
        choices=("mse", "znssd"),
        help="Photometric loss for step 7 deformation training",
    )
    parser.add_argument("--clean", action="store_true", help="Remove previous generated outputs")
    parser.add_argument(
        "--skip-sfm",
        action="store_true",
        help="Load existing result/sfm products instead of rerunning COLMAP",
    )
    return parser.parse_args()


def run_sfm(args):
    from ndef_dic.sfm.reference_sfm import reference_sfm_exists, run_reference_sfm

    out_dir = os.path.join(args.data_dir, "result", "sfm")
    if args.skip_sfm:
        if not reference_sfm_exists(args.data_dir, out_dir):
            raise FileNotFoundError(f"SfM products not found in {out_dir}")
        print(f"[SfM] Existing products found in {out_dir}")
        return None

    return run_reference_sfm(
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        ref_mode="named",
        ref_name=args.ref_name,
        output_dir=out_dir,
        reference_camera=args.reference_camera,
        clean=args.clean,
    )


def run_dense_model_init(args):
    from ndef_dic.dense import DepthInitConfig, run_model_init

    return run_model_init(
        DepthInitConfig(
            data_dir=args.data_dir,
            sfm_dir=os.path.join(args.data_dir, "result", "sfm"),
            output_dir=os.path.join(args.data_dir, "result", "dense", "model_init"),
        )
    )


def run_reconstruction_dataset(args):
    from ndef_dic.dense import ReconstructionDatasetConfig, run_reconstruction_dataset as build_dataset

    return build_dataset(
        ReconstructionDatasetConfig(
            data_dir=args.data_dir,
            sfm_dir=os.path.join(args.data_dir, "result", "sfm"),
            model_init_dir=os.path.join(args.data_dir, "result", "dense", "model_init"),
            output_dir=os.path.join(args.data_dir, "result", "dense", "reconstruction_dataset"),
        )
    )


def run_dense_znssd(args):
    from ndef_dic.dense import DenseZNSSDConfig, run_dense_znssd as optimize_dense

    return optimize_dense(
        DenseZNSSDConfig(
            data_dir=args.data_dir,
            sfm_dir=os.path.join(args.data_dir, "result", "sfm"),
            model_init_dir=os.path.join(args.data_dir, "result", "dense", "model_init"),
            dataset_dir=os.path.join(args.data_dir, "result", "dense", "reconstruction_dataset"),
            output_dir=os.path.join(args.data_dir, "result", "dense", "znssd_opt"),
            max_steps_per_epoch=None,
        )
    )


def run_dense_reconstruction(args):
    from ndef_dic.dense import DenseReconstructionConfig, run_dense_reconstruction as export_dense

    return export_dense(
        DenseReconstructionConfig(
            data_dir=args.data_dir,
            sfm_dir=os.path.join(args.data_dir, "result", "sfm"),
            model_init_dir=os.path.join(args.data_dir, "result", "dense", "model_init"),
            znssd_dir=os.path.join(args.data_dir, "result", "dense", "znssd_opt"),
            output_dir=os.path.join(args.data_dir, "result", "dense", "reconstruction_dense"),
        )
    )


def run_surface_sampler(args):
    from ndef_dic.dense import SurfaceSamplerConfig, run_surface_sampler as sample_surface

    return sample_surface(
        SurfaceSamplerConfig(
            data_dir=args.data_dir,
            sfm_dir=os.path.join(args.data_dir, "result", "sfm"),
            model_init_dir=os.path.join(args.data_dir, "result", "dense", "model_init"),
            reconstruction_dense_dir=os.path.join(args.data_dir, "result", "dense", "reconstruction_dense"),
            output_dir=os.path.join(args.data_dir, "result", "dense", "surface_sampler"),
        )
    )


def run_deformation(args):
    from ndef_dic.deformation import DeformationTrainingConfig, run_deformation_training

    return run_deformation_training(
        DeformationTrainingConfig(
            data_dir=args.data_dir,
            sfm_dir=os.path.join(args.data_dir, "result", "sfm"),
            surface_dataset_path=os.path.join(
                args.data_dir,
                "result",
                "dense",
                "surface_sampler",
                "deformation_surface_dataset.npz",
            ),
            output_dir=os.path.join(args.data_dir, "result", "deformation"),
            image_dir=args.image_dir,
            reference_name=args.ref_name,
            current_name="002",
            use_positional_encoding=False,
            photometric_loss=args.deformation_loss,
            patch_radius=2,
            invalid_patch_penalty=0.05,
            lr=1e-4,
            displacement_scale_path=os.path.join(
                args.data_dir,
                "result",
                "deformation",
                "precalculation",
                "patch_dic_sparse",
                "displacement_scale.json",
            ),
            displacement_scale_stat="mean",
        )
    )


def main():
    args = parse_args()
    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    print("\n" + "=" * 60)
    print("  NDF-DIC Pipeline")
    print(f"  Steps: {steps}")
    print(f"  Data: {args.data_dir}")
    print("  Config files: disabled")
    print("=" * 60 + "\n")

    total_start = time.time()

    if 1 in steps:
        print("\n" + "#" * 60)
        print("# STEP 1: SfM Self-Calibration / Sparse Reconstruction")
        print("#" * 60)
        start = time.time()
        output = run_sfm(args)
        print(f"\n[Step 1] Done in {time.time() - start:.0f}s")
        if output is not None:
            print(f"  Cameras: {len(output.cam_names)}")
            print(f"  Sparse points: {len(output.points3D)}")
            print(f"  Observations: {len(output.observations['uv'])}")

    if 2 in steps:
        print("\n" + "#" * 60)
        print("# STEP 2: SfM-Guided Neural Depth Initialisation")
        print("#" * 60)
        start = time.time()
        fig_paths = run_dense_model_init(args)
        print(f"\n[Step 2] Done in {time.time() - start:.0f}s")
        for name, path in fig_paths.items():
            print(f"  {name}: {path}")

    if 3 in steps:
        print("\n" + "#" * 60)
        print("# STEP 3: Multi-View Reconstruction Dataset")
        print("#" * 60)
        start = time.time()
        dataset_paths = run_reconstruction_dataset(args)
        print(f"\n[Step 3] Done in {time.time() - start:.0f}s")
        for name, path in dataset_paths.items():
            print(f"  {name}: {path}")

    if 4 in steps:
        print("\n" + "#" * 60)
        print("# STEP 4: Dense ZNSSD Depth Optimisation")
        print("#" * 60)
        start = time.time()
        znssd_paths = run_dense_znssd(args)
        print(f"\n[Step 4] Done in {time.time() - start:.0f}s")
        for name, path in znssd_paths.items():
            print(f"  {name}: {path}")

    if 5 in steps:
        print("\n" + "#" * 60)
        print("# STEP 5: Export ZNSSD Dense Reconstruction")
        print("#" * 60)
        start = time.time()
        recon_paths = run_dense_reconstruction(args)
        print(f"\n[Step 5] Done in {time.time() - start:.0f}s")
        for name, path in recon_paths.items():
            print(f"  {name}: {path}")

    if 6 in steps:
        print("\n" + "#" * 60)
        print("# STEP 6: Visibility-Aware Reference Surface Sampler")
        print("#" * 60)
        start = time.time()
        surface_paths = run_surface_sampler(args)
        print(f"\n[Step 6] Done in {time.time() - start:.0f}s")
        for name, path in surface_paths.items():
            print(f"  {name}: {path}")

    if 7 in steps:
        print("\n" + "#" * 60)
        print("# STEP 7: Surface Neural Deformation Field")
        print("#" * 60)
        start = time.time()
        deformation_paths = run_deformation(args)
        print(f"\n[Step 7] Done in {time.time() - start:.0f}s")
        for name, path in deformation_paths.items():
            print(f"  {name}: {path}")

    unsupported = [s for s in steps if s not in {1, 2, 3, 4, 5, 6, 7}]
    if unsupported:
        raise NotImplementedError(
            f"Steps {unsupported} are not implemented in the current research pipeline."
        )

    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
