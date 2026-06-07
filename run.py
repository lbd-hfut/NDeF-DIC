#!/usr/bin/env python
"""
NDeF-DIC: Neural Deformation Field for Multi-Camera Digital Image Correlation.

Main entry point for training and inference.

Usage:
    # Train with default config
    python run.py --data_dir /path/to/data --work_dir /path/to/output

    # Train specific stages
    python run.py --data_dir /path/to/data --stages sdf intensity

    # Resume from checkpoint
    python run.py --data_dir /path/to/data --resume stage2_final

    # Generate speckle masks only
    python run.py --data_dir /path/to/data --generate_masks_only
"""

import argparse
import os
import sys
import torch
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="NDeF-DIC: Neural Deformation Field for Multi-Camera DIC"
    )

    # Paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to multi-camera data directory")
    parser.add_argument("--work_dir", type=str, default="output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--colmap_dir", type=str, default="",
                        help="Path to COLMAP results (if pre-computed)")

    # Data
    parser.add_argument("--n_cameras", type=int, default=4,
                        help="Number of cameras")
    parser.add_argument("--n_load_steps", type=int, default=10,
                        help="Number of load steps (including reference)")
    parser.add_argument("--image_height", type=int, default=1200)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--ref_name", type=str, default="ref",
                        help="Reference image filename prefix")
    parser.add_argument("--frame_pattern", type=str, default="frame_{:03d}",
                        help="Deformed frame naming pattern")
    parser.add_argument("--image_ext", type=str, default=".png")

    # Training
    parser.add_argument("--stages", type=str, nargs="+",
                        default=["sdf", "intensity", "deformation", "joint"],
                        choices=["sdf", "intensity", "deformation", "joint"],
                        help="Training stages to run")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Pixels per training batch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lr_joint", type=float, default=1e-5,
                        help="Learning rate for joint refinement")
    parser.add_argument("--stage1_epochs", type=int, default=5000)
    parser.add_argument("--stage2_epochs", type=int, default=20000)
    parser.add_argument("--stage3_epochs", type=int, default=50000)
    parser.add_argument("--stage4_epochs", type=int, default=10000)
    parser.add_argument("--photometric_loss", type=str, default="mse",
                        choices=["mse", "znssd"])

    # SDF config
    parser.add_argument("--sdf_n_freqs", type=int, default=6)
    parser.add_argument("--sdf_hidden_layers", type=int, default=6)
    parser.add_argument("--sdf_hidden_dim", type=int, default=256)
    parser.add_argument("--sdf_lambda_data", type=float, default=10.0)
    parser.add_argument("--sdf_lambda_eikonal", type=float, default=0.1)

    # Intensity config
    parser.add_argument("--intensity_n_freqs", type=int, default=10)
    parser.add_argument("--intensity_hidden_layers", type=int, default=4)
    parser.add_argument("--intensity_hidden_dim", type=int, default=256)

    # Deformation config
    parser.add_argument("--deform_n_freqs_space", type=int, default=8)
    parser.add_argument("--deform_n_freqs_time", type=int, default=4)
    parser.add_argument("--deform_hidden_layers", type=int, default=6)
    parser.add_argument("--deform_hidden_dim", type=int, default=256)
    parser.add_argument("--deform_lambda_smooth", type=float, default=0.01)

    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda | cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint tag (e.g., stage2_final)")
    parser.add_argument("--generate_masks_only", action="store_true",
                        help="Only generate speckle masks and exit")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_config_from_args(args) -> "NDeFDICConfig":
    """Build NDeFDICConfig from command-line arguments."""
    from ndef_dic.config import (
        NDeFDICConfig, SDFConfig, IntensityConfig,
        DeformationConfig, AppearanceConfig, TrainingConfig,
    )

    sdf = SDFConfig(
        n_freqs=args.sdf_n_freqs,
        hidden_layers=args.sdf_hidden_layers,
        hidden_dim=args.sdf_hidden_dim,
        lambda_data=args.sdf_lambda_data,
        lambda_eikonal=args.sdf_lambda_eikonal,
    )

    intensity = IntensityConfig(
        n_freqs=args.intensity_n_freqs,
        hidden_layers=args.intensity_hidden_layers,
        hidden_dim=args.intensity_hidden_dim,
    )

    deformation = DeformationConfig(
        n_freqs_space=args.deform_n_freqs_space,
        n_freqs_time=args.deform_n_freqs_time,
        hidden_layers=args.deform_hidden_layers,
        hidden_dim=args.deform_hidden_dim,
        lambda_smooth=args.deform_lambda_smooth,
    )

    appearance = AppearanceConfig()

    training = TrainingConfig(
        learning_rate=args.lr,
        lr_joint=args.lr_joint,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage4_epochs=args.stage4_epochs,
        photometric_loss=args.photometric_loss,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    config = NDeFDICConfig(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        colmap_dir=args.colmap_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        n_cameras=args.n_cameras,
        n_load_steps=args.n_load_steps,
        sdf=sdf,
        intensity=intensity,
        deformation=deformation,
        appearance=appearance,
        training=training,
        device=args.device,
        random_seed=args.seed,
    )

    return config


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Build config
    config = build_config_from_args(args)

    # Create output directory
    os.makedirs(config.work_dir, exist_ok=True)

    # Device check
    if config.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        config.device = "cpu"

    print("=" * 60)
    print("NDeF-DIC: Neural Deformation Field for Multi-Camera DIC")
    print("=" * 60)
    print(f"  Data:     {config.data_dir}")
    print(f"  Output:   {config.work_dir}")
    print(f"  Device:   {config.device}")
    print(f"  Cameras:  {config.n_cameras}")
    print(f"  Steps:    {config.n_load_steps}")
    print(f"  Stages:   {args.stages}")
    print(f"  Batch:    {config.training.batch_size}")
    print(f"  Photo:    {config.training.photometric_loss}")
    print(f"  Seed:     {config.random_seed}")
    print("=" * 60)

    # Load dataset
    from ndef_dic.dataset import MultiCamDataset
    dataset = MultiCamDataset(
        data_dir=config.data_dir,
        n_cameras=config.n_cameras,
        n_load_steps=config.n_load_steps,
        image_height=config.image_height,
        image_width=config.image_width,
        ref_name=args.ref_name,
        frame_pattern=args.frame_pattern,
        image_ext=args.image_ext,
        device=config.device,
    )

    print(f"\n[INFO] Loaded {len(dataset.ref_images)} reference images")
    print(f"[INFO] Loaded {len(dataset.def_images)} deformed frame sets")
    if dataset.colmap_points is not None:
        print(f"[INFO] Loaded {len(dataset.colmap_points)} COLMAP 3D points")
    else:
        print(f"[WARNING] No COLMAP 3D points found. "
              f"Stage 1 (SDF) will not work without them.")

    # Generate masks
    print(f"\n[INFO] Generating speckle masks...")
    dataset.ensure_masks()
    print(f"[INFO] Masks generated for {len(dataset.masks)} cameras")

    # Save masks for inspection
    masks_dir = os.path.join(config.work_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    from ndef_dic.speckle_mask import SpeckleMaskGenerator
    for cam_id, mask in enumerate(dataset.masks):
        SpeckleMaskGenerator.save_mask(
            mask, os.path.join(masks_dir, f"cam_{cam_id}_mask.png")
        )
    print(f"[INFO] Masks saved to {masks_dir}")

    if args.generate_masks_only:
        print(f"\n[DONE] Mask generation complete. Exiting.")
        return

    # Create trainer
    from ndef_dic.trainer import NDeFDICTrainer
    trainer = NDeFDICTrainer(config, dataset)

    # Resume if requested
    if args.resume:
        print(f"\n[INFO] Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(stages=args.stages)

    print(f"\n[DONE] All results saved to {config.work_dir}")


if __name__ == "__main__":
    main()
