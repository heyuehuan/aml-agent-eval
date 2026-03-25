"""Evaluation config loader.

Reads ``eval_config.yaml`` and resolves paths relative to the project root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_DEFAULT_CONFIG = _THIS_DIR / "eval_config.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "aml-agent-test-cases"
    csv_path: str = "data/test_cases/test_cases.csv"
    artifacts_dir: str = "output/test_cases/test_run"


@dataclass(frozen=True)
class AgentConfig:
    name: str = "real-time-aml-agent"
    temperature: float | None = None
    timeout_sec: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    max_concurrency: int = 10
    live_max_concurrency: int = 2


@dataclass(frozen=True)
class EvalConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def resolve_csv_path(self) -> Path:
        return _PROJECT_ROOT / self.dataset.csv_path

    def resolve_artifacts_dir(self) -> Path:
        return _PROJECT_ROOT / self.dataset.artifacts_dir


def load_eval_config(path: str | Path | None = None) -> EvalConfig:
    """Load evaluation config from a YAML file.

    Falls back to the bundled ``eval_config.yaml`` if *path* is ``None``.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        return EvalConfig()

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return EvalConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        experiment=ExperimentConfig(**raw.get("experiment", {})),
    )


__all__ = ["EvalConfig", "DatasetConfig", "AgentConfig", "ExperimentConfig", "load_eval_config"]
