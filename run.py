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

    unsupported = [s for s in steps if s not in {1, 2}]
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
