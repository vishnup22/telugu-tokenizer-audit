from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def new_experiment_dir(run_tag: str, base_dir: str = "experiments") -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    experiment_dir = Path(base_dir) / f"{today}_{run_tag}"
    if experiment_dir.exists():
        raise FileExistsError(
            f"{experiment_dir} already exists — pick a different run_tag"
        )
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True)
    return experiment_dir


def snapshot_config(config: dict, experiment_dir: Path) -> None:
    path = experiment_dir / "config_snapshot.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def write_run_metadata(
    experiment_dir: Path,
    tokenizer_versions: dict,
    corpus_line_counts: dict[str, int],
) -> None:
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_used": "config_snapshot.yaml",
        "tokenizer_versions": tokenizer_versions,
        "corpus_line_counts": corpus_line_counts,
        "git_commit": _git_commit(),
    }
    path = experiment_dir / "run_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_tokenizer_include(config: dict, config_path: Path) -> set[str]:
    tokenizer_set = config.get("tokenizer_set", "quick_test")
    set_path = config_path.parent / "tokenizer_sets" / f"{tokenizer_set}.yaml"
    if not set_path.exists():
        set_path = Path("configs/tokenizer_sets") / f"{tokenizer_set}.yaml"
    set_cfg = load_yaml_config(set_path)
    return set(set_cfg.get("include", []))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
