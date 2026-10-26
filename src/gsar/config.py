"""Configuration loading and project-relative path handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access for the existing research pipeline."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _to_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def project_root() -> Path:
    """Return the checkout root, overridable with ``GSAR_ROOT``."""
    override = os.getenv("GSAR_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve a user path relative to the project checkout."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else ((root or project_root()) / value).resolve()


def load_config(name: str, explicit_path: str | None = None) -> ConfigNode:
    """Load one YAML configuration file from ``configs/``."""
    env_name = f"GSAR_{name.upper()}_CONFIG"
    path = explicit_path or os.getenv(env_name) or str(project_root() / "configs" / f"{name}.yaml")
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"GSAR configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"GSAR configuration must be a mapping: {config_path}")
    return _to_node(data)


def get_data_synthesis_config(path: str | None = None) -> ConfigNode:
    return load_config("data_synthesis", path)


def get_verification_config(path: str | None = None) -> ConfigNode:
    return load_config("verification", path)

