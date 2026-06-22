import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from telugu_audit.run_utils import load_yaml_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "toy_corpus"


@pytest.fixture
def pipeline_env(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    experiments = tmp_path / "experiments"
    experiments.mkdir()

    cfg = load_yaml_config(ROOT / "tests" / "fixtures" / "test_config.yaml")
    # corpus_dir must be a writable tmp dir so stage-1 can build into it
    cfg["corpus_dir"] = str(processed)
    cfg["minimal_pairs_path"] = str(ROOT / cfg["minimal_pairs_path"])
    cfg["pricing_path"] = str(ROOT / cfg["pricing_path"])

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    return {
        "config_path": config_path,
        "processed": processed,
        "experiments": experiments,
        "raw": FIXTURE,
    }


def _run_script(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr + result.stdout


def test_pipeline_end_to_end(pipeline_env):
    """Orchestrator produces exactly one experiment folder containing all outputs."""
    env = pipeline_env

    _run_script(
        "00_run_full_pipeline.py",
        "--config", str(env["config_path"]),
        "--run_tag", "test",
        "--experiments-dir", str(env["experiments"]),
        "--raw-dir", str(env["raw"]),
    )

    exp_dirs = list(env["experiments"].glob("*_test"))
    assert len(exp_dirs) == 1, f"Expected 1 experiment dir, got: {exp_dirs}"
    exp_dir = exp_dirs[0]

    assert (exp_dir / "config_snapshot.yaml").exists()
    assert (exp_dir / "run_metadata.json").exists()

    fertility_cols = {
        "tokenizer",
        "register",
        "n_lines_attempted",
        "n_lines_tokenized",
        "total_tokens",
        "total_words",
        "total_bytes",
        "fertility_tokens_per_word",
        "compression_bytes_per_token",
    }
    gap_cols = {
        "tokenizer",
        "native_fertility",
        "romanized_fertility",
        "script_fertility_ratio_native_over_romanized",
    }
    mp_cols = {"word", "gloss", "morph_type", "tokenizer", "n_tokens"}

    fertility_df = pd.read_csv(exp_dir / "results" / "fertility_by_register.csv")
    gap_df = pd.read_csv(exp_dir / "results" / "script_fairness_gap.csv")
    mp_df = pd.read_csv(exp_dir / "results" / "minimal_pair_fertility.csv")

    assert fertility_cols <= set(fertility_df.columns)
    assert gap_cols <= set(gap_df.columns)
    assert mp_cols <= set(mp_df.columns)


def test_standalone_scripts_accept_experiment_dir(pipeline_env):
    """Scripts 02 and 03 can write into a pre-existing folder via --experiment-dir."""
    env = pipeline_env

    # Build corpus first
    _run_script(
        "01_build_corpus.py",
        "--config", str(env["config_path"]),
        "--raw-dir", str(env["raw"]),
        "--output-dir", str(env["processed"]),
    )

    # Script 02 creates the folder
    _run_script(
        "02_run_tokenizer_audit.py",
        "--config", str(env["config_path"]),
        "--run_tag", "standalone",
        "--experiments-dir", str(env["experiments"]),
    )

    exp_dirs = list(env["experiments"].glob("*_standalone"))
    assert len(exp_dirs) == 1
    exp_dir = exp_dirs[0]

    # Script 03 writes into it via --experiment-dir
    _run_script(
        "03_run_minimal_pair_audit.py",
        "--config", str(env["config_path"]),
        "--experiment-dir", str(exp_dir),
    )

    assert (exp_dir / "results" / "fertility_by_register.csv").exists()
    assert (exp_dir / "results" / "minimal_pair_fertility.csv").exists()
    # Only one experiment folder was created
    assert len(list(env["experiments"].glob("*_standalone*"))) == 1
