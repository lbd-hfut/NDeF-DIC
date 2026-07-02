"""YAML configuration loading and path resolution utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


ConfigDict = dict[str, Any]


def load_yaml_config(path: str | Path) -> ConfigDict:
    """Load one YAML config file as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two mapping objects without modifying either input."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_dir(config_dir: str | Path = "configs") -> ConfigDict:
    """Load and merge all ``*.yaml`` files in a config directory.

    Files are read in filename order, so numeric prefixes such as
    ``1_project.yaml`` and ``6_deformation.yaml`` define a stable module order.
    Later files may override earlier keys when they intentionally share a
    top-level section.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(config_dir)
    merged: ConfigDict = {}
    for path in sorted(config_dir.glob("*.yaml")):
        merged = deep_merge(merged, load_yaml_config(path))
    return merged


def project_data_dir(config: Mapping[str, Any]) -> Path:
    """Return ``project.data_dir`` as a path."""
    return Path(_required(config, "project.data_dir"))


def project_result_dir(config: Mapping[str, Any]) -> Path:
    """Return ``project.result_dir`` resolved against ``project.data_dir``.

    ``project.result_dir`` may be absolute, or relative to ``project.data_dir``.
    This keeps case configs portable: changing ``project.data_dir`` is enough
    to move all module result paths.
    """
    value = Path(_required(config, "project.result_dir"))
    if value.is_absolute():
        return value
    return project_data_dir(config) / value


def resolve_data_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to ``project.data_dir`` when not absolute."""
    return _resolve_relative(project_data_dir(config), value)


def resolve_result_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to ``project.result_dir`` when not absolute."""
    return _resolve_relative(project_result_dir(config), value)


def resolve_config_path(
    config: Mapping[str, Any],
    value: str | Path,
    *,
    base: str,
) -> Path:
    """Resolve a config path relative to either project data or result root.

    Args:
        config: Merged project configuration.
        value: Path value from a module config.
        base: ``"data"`` for paths under ``project.data_dir`` or ``"result"``
            for paths under ``project.result_dir``.
    """
    if base == "data":
        return resolve_data_path(config, value)
    if base == "result":
        return resolve_result_path(config, value)
    raise ValueError("base must be 'data' or 'result'")


def get_by_path(config: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a nested value using a dotted path such as ``project.data_dir``."""
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _required(config: Mapping[str, Any], dotted_path: str) -> Any:
    value = get_by_path(config, dotted_path)
    if value is None:
        raise KeyError(f"Missing required config value: {dotted_path}")
    return value


def _resolve_relative(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
