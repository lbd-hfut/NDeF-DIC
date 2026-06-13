#!/usr/bin/env python
"""
NDF-DIC: Neural Deformation Field for Multi-View Digital Image Correlation.

Main entry point. Reads config and runs the full pipeline:
  Step 1 → Geometric Reconstruction (COLMAP SfM + Dense MVS)
  Step 2 → SurfaceProvider (point cloud or neural stereo)
  Step 3 → Deformation Field training (Φ network)

Usage:
    python run.py                          # use config/default.yaml
    python run.py --config config/exp.yaml # custom config
    python run.py --steps 1,2              # only run steps 1 and 2
    python run.py --device cpu             # override device
    python run.py --help

The pipeline can resume from any step — if calibration/cameras.mat exists,
Step 1 sparse can be skipped via config (step1.sparse_mode: skip_sfm).
"""

import os
import sys
import argparse
import time
from typing import Dict, Any


def parse_args():
    p = argparse.ArgumentParser(
        description="NDF-DIC: Neural Deformation Field DIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                              # full pipeline with defaults
  python run.py --config config/local.yaml   # custom config
  python run.py --steps 1                    # only geometric reconstruction
  python run.py --steps 2,3                  # skip reconstruction, train Phi
  python run.py --device cpu                 # run on CPU
  python run.py --clean                      # remove previous results
        """,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to config YAML (default: config/default.yaml)")
    p.add_argument("--steps", type=str, default="1,2,3",
                   help="Steps to run, comma-separated (e.g., '1,2,3')")
    p.add_argument("--device", type=str, default=None,
                   help="Override config device (cuda/cpu)")
    p.add_argument("--clean", action="store_true",
                   help="Remove previous results before starting")
    return p.parse_args()


def run_step1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run Step 1: Geometric Reconstruction."""
    from ndef_dic.step1_pipeline import run_step1

    c = cfg["step1"]
    dense_cfg = c.get("dense", {})
    post_cfg = c.get("postprocess", {})
    data = cfg["data"]

    output = run_step1(
        data_dir=data["data_dir"],
        image_dir=data["image_dir"],
        sparse_mode=c["sparse_mode"],
        dense_path=dense_cfg.get("path", "patchmatch"),
        calib_dir=os.path.join(data["data_dir"], data["calib_dir"]),
        ref_name=data.get("ref_name", "001"),
        image_width=data["image_width"],
        image_height=data["image_height"],
        post_config=post_cfg,
        clean=cfg.get("output", {}).get("clean", False),
    )
    return {
        "K_list": output.K_list,
        "R_list": output.R_list,
        "t_list": output.t_list,
        "n_cameras": output.num_cameras,
        "n_points_dense": output.num_points_dense,
        "dense_status": output.dense_status,
    }


def run_step2(cfg: Dict[str, Any]) -> Any:
    """Run Step 2: SurfaceProvider."""
    from ndef_dic.surface_provider import create_surface_provider

    data = cfg["data"]
    c = cfg["step2"]

    surface = create_surface_provider(
        data_dir=data["data_dir"],
        calib_dir=os.path.join(data["data_dir"], data["calib_dir"]),
        method=c["method"],
        device=cfg.get("device", "cuda"),
        image_width=data["image_width"],
        image_height=data["image_height"],
    )
    return surface


def run_step3(
    cfg: Dict[str, Any],
    surface: Any,
) -> Dict[str, Any]:
    """Run Step 3: Deformation Field training."""
    from ndef_dic.dataset import MultiCamDataset
    from ndef_dic.deformation_net import DeformationNetwork
    from ndef_dic.deformation_trainer import DeformationFieldTrainer
    from ndef_dic.config import (
        to_deformation_trainer_config,
        to_hash_grid_config,
        to_temporal_config,
    )

    data = cfg["data"]
    c = cfg["step3"]
    d = c.get("deformation", {})

    # Dataset
    dataset = MultiCamDataset(
        data_dir=data["data_dir"],
        image_dir=data["image_dir"],
        calib_dir=data["calib_dir"],
        ref_mode=data.get("ref_mode", "first"),
        ref_name=data.get("ref_name", "001"),
        image_width=data["image_width"],
        image_height=data["image_height"],
        device=cfg.get("device", "cuda"),
    )

    # Network
    net = DeformationNetwork(
        hash_grid_config=to_hash_grid_config(cfg),
        temporal_config=to_temporal_config(cfg),
        hidden_dim=d.get("hidden_dim", 256),
        alpha=d.get("alpha", 5.0),
        learnable_alpha=d.get("learnable_alpha", False),
    )

    # Trainer
    trainer_config = to_deformation_trainer_config(cfg)
    trainer = DeformationFieldTrainer(surface, net, dataset, trainer_config)

    # Train
    trainer.train()

    # Checkpoint
    chk_dir = c.get("training", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, "model_final.pt")
    trainer.save_checkpoint(chk_path)

    # Quick summary
    n_params = sum(p.numel() for p in net.parameters())
    final_losses = trainer.loss_history[-5:] if trainer.loss_history else []
    avg_final = (
        sum(m["L_dic"] for m in final_losses) / len(final_losses)
        if final_losses else float("nan")
    )

    return {
        "n_params": n_params,
        "final_znssd": avg_final,
        "checkpoint": chk_path,
    }


def main():
    args = parse_args()

    # Load config
    from ndef_dic.config import load_config

    cfg = load_config(args.config)

    # CLI overrides
    if args.device:
        cfg["device"] = args.device
    if args.clean:
        cfg.setdefault("output", {})["clean"] = True

    steps = [int(s.strip()) for s in args.steps.split(",")]

    print(f"\n{'='*60}")
    print(f"  NDF-DIC Pipeline")
    print(f"  Steps: {steps}")
    print(f"  Device: {cfg.get('device', 'cuda')}")
    print(f"  Data: {cfg['data']['data_dir']}")
    print(f"{'='*60}\n")

    total_start = time.time()
    surface = None

    # ---- Step 1 ----
    if 1 in steps:
        print(f"\n{'#'*60}")
        print(f"# STEP 1: Geometric Reconstruction")
        print(f"{'#'*60}")
        t0 = time.time()
        s1_result = run_step1(cfg)
        print(f"\n[Step 1] Done in {time.time() - t0:.0f}s")
        print(f"  Cameras: {s1_result['n_cameras']}")
        print(f"  Dense points: {s1_result['n_points_dense']}")
        print(f"  Status: {s1_result['dense_status']}")

    # ---- Step 2 ----
    if 2 in steps:
        print(f"\n{'#'*60}")
        print(f"# STEP 2: SurfaceProvider")
        print(f"{'#'*60}")
        t0 = time.time()
        surface = run_step2(cfg)
        print(f"\n[Step 2] Done in {time.time() - t0:.0f}s")
        print(f"  Surface: {type(surface).__name__}")
        print(f"  Cameras: {surface.num_cameras}")
        print(f"  Bbox:\n{surface.bbox}")

    # ---- Step 3 ----
    if 3 in steps:
        if surface is None:
            print("[Step 3] Creating surface from config...")
            surface = run_step2(cfg)

        print(f"\n{'#'*60}")
        print(f"# STEP 3: Deformation Field Training")
        print(f"{'#'*60}")
        t0 = time.time()
        s3_result = run_step3(cfg, surface)
        print(f"\n[Step 3] Done in {time.time() - t0:.0f}s")
        print(f"  Network params: {s3_result['n_params']:,}")
        print(f"  Final ZNSSD: {s3_result['final_znssd']:.2f}")
        print(f"  Checkpoint: {s3_result['checkpoint']}")

    # ---- Done ----
    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
