"""
Configuration loader for NDF-DIC.

Loads YAML config files from config/ directory. Falls back to defaults
if no config provided. Provides typed conversion to internal Config dataclasses.
"""

import os
from typing import Any, Dict, Optional


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file. Falls back to json if yaml not available."""
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        import json
        # YAML is a superset of JSON, so try JSON first for simplicity
        with open(path, "r") as f:
            content = f.read()
        # Strip YAML comments (very basic)
        import re
        content = re.sub(r"^\s*#.*$", "", content, flags=re.MULTILINE)
        # Quick-and-dirty: try to parse as JSON-like if simple enough
        # Better: just import yaml. This fallback is for environments
        # where pyyaml is not installed.
        raise ImportError(
            "pyyaml is required for config loading. "
            "Install with: pip install pyyaml"
        )


def resolve_config_path(config_path: Optional[str] = None) -> str:
    """Resolve config file path.

    Priority:
      1. Explicit path (config_path argument)
      2. config/local.yaml (user overrides)
      3. config/default.yaml (project defaults)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if config_path and os.path.exists(config_path):
        return config_path

    local_path = os.path.join(project_root, "config", "local.yaml")
    if os.path.exists(local_path):
        return local_path

    default_path = os.path.join(project_root, "config", "default.yaml")
    if os.path.exists(default_path):
        return default_path

    raise FileNotFoundError(
        f"No config found. Create config/default.yaml or pass --config PATH."
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project configuration.

    Args:
        config_path: Optional explicit config file path.
                    If None, resolves via resolve_config_path().

    Returns:
        dict with all configuration sections.
    """
    path = resolve_config_path(config_path)
    cfg = load_yaml(path)
    print(f"[Config] Loaded from {path}")
    return cfg


# =========================================================================
# Typed conversion helpers
# =========================================================================

def to_pinn_stereo_config(cfg: Dict[str, Any]):
    """Convert config dict section to PINNStereoConfig."""
    from .pinn_stereo import PINNStereoConfig
    c = cfg.get("step1", {}).get("dense", {}).get("pinn_stereo", {})
    return PINNStereoConfig(
        n_frequencies=c.get("n_frequencies", 10),
        hidden_dim=c.get("hidden_dim", 256),
        n_layers=c.get("n_layers", 4),
        stage1_epochs=c.get("stage1_epochs", 500),
        stage1_lr=c.get("stage1_lr", 1e-3),
        stage1_batch_size=c.get("stage1_batch_size", 2048),
        stage2_epochs_max=c.get("stage2_epochs_max", 5000),
        stage2_lr=c.get("stage2_lr", 1e-4),
        stage2_patience=c.get("stage2_patience", 50),
        stage2_batch_size=c.get("stage2_batch_size", 128),
        patch_radius=c.get("patch_radius", 5),
        znssd_eps=c.get("znssd_eps", 1e-6),
        roi_dilation=c.get("roi_dilation", 15),
        device=cfg.get("device", "cuda"),
    )


def to_dense_mvs_config(cfg: Dict[str, Any]):
    """Convert config dict section to DenseMVSConfig."""
    from .dense_mvs import DenseMVSConfig
    c = cfg.get("step1", {}).get("dense", {}).get("patchmatch", {})
    return DenseMVSConfig(
        window_radius=c.get("window_radius", 7),
        window_step=c.get("window_step", 1),
        num_iterations=c.get("num_iterations", 7),
        num_samples=c.get("num_samples", 15),
        geom_consistency=c.get("geom_consistency", True),
        filter_min_ncc=c.get("filter_min_ncc", 0.1),
        filter_min_num_consistent=c.get("filter_min_num_consistent", 2),
        ncc_sigma=c.get("ncc_sigma", 0.6),
        max_depth_error=c.get("max_depth_error", 0.01),
        max_normal_error=c.get("max_normal_error", 10.0),
        max_reproj_error=c.get("max_reproj_error", 2.0),
    )


def to_deformation_trainer_config(cfg: Dict[str, Any]):
    """Convert config dict section to DeformationTrainerConfig."""
    from .deformation_trainer import (
        DeformationTrainerConfig, PhaseConfig
    )
    c = cfg.get("step3", {})
    t = c.get("training", {})

    phases = []
    for p in t.get("phases", []):
        phases.append(PhaseConfig(
            patch_size=p.get("patch_size", 32),
            lambda_smooth=p.get("lambda_smooth", 1e-2),
            iterations=p.get("iterations", 2000),
            lr=p.get("lr", 1e-3),
            log_interval=p.get("log_interval", 100),
            validate_interval=p.get("validate_interval", 500),
        ))

    return DeformationTrainerConfig(
        batch_size=t.get("batch_size", 1024),
        cameras_per_point=t.get("cameras_per_point", 3),
        phases=phases,
        load_step_strategy=t.get("load_step_strategy", "joint"),
        grad_clip_norm=t.get("grad_clip_norm", 1.0),
        device=cfg.get("device", "cuda"),
    )


def to_hash_grid_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config dict section to HashGridEncoder kwargs."""
    c = cfg.get("step3", {}).get("hash_grid", {})
    return {
        "n_levels": c.get("n_levels", 16),
        "n_features_per_level": c.get("n_features_per_level", 2),
        "base_resolution": c.get("base_resolution", 16),
        "finest_resolution": c.get("finest_resolution", 512),
        "hash_table_size": c.get("hash_table_size", 2**19),
    }


def to_temporal_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config dict section to TemporalEncoder kwargs."""
    c = cfg.get("step3", {}).get("temporal", {})
    return {
        "strategy": c.get("strategy", "binary"),
        "n_freqs": c.get("n_freqs", 6),
    }
