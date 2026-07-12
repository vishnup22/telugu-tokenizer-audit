import json
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
    cfg["corpus_dir"] = str(processed)
    cfg["minimal_pairs_path"] = str(ROOT / cfg["minimal_pairs_path"])
    cfg["pricing_path"] = str(ROOT / cfg["pricing_path"])

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    benchmark_scores_path = tmp_path / "benchmark_scores.yaml"
    with benchmark_scores_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "accuracy_by_model": {
                    "stub-local": 0.61,
                }
            },
            f,
        )

    return {
        "config_path": config_path,
        "processed": processed,
        "experiments": experiments,
        "raw": FIXTURE,
        "benchmark_scores_path": benchmark_scores_path,
    }


def _run_script(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr + result.stdout


def test_pipeline_end_to_end(pipeline_env):
    env = pipeline_env

    _run_script(
        "00_run_full_pipeline.py",
        "--config", str(env["config_path"]),
        "--run_tag", "test",
        "--experiments-dir", str(env["experiments"]),
        "--raw-dir", str(env["raw"]),
        "--benchmark-scores", str(env["benchmark_scores_path"]),
        "--n-bootstrap", "200",
    )

    exp_dirs = list(env["experiments"].glob("*_test"))
    assert len(exp_dirs) == 1, f"Expected 1 experiment dir, got: {exp_dirs}"
    exp_dir = exp_dirs[0]

    assert (exp_dir / "config_snapshot.yaml").exists()
    assert (exp_dir / "run_metadata.json").exists()

    fertility_cols = {
        "tokenizer",
        "register",
        "word_count_method",
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
        "word_count_method",
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

    assert (exp_dir / "results" / "script_gap_significance.json").exists()
    assert (exp_dir / "results" / "per_sentence_gaps.csv").exists()
    assert not (exp_dir / "results" / "fertility_accuracy_correlation.json").exists()


def test_multilingual_pipeline_supports_hindi(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    experiments = tmp_path / "experiments"
    experiments.mkdir()

    for src_name, dst_name in [
        ("FAKE_native_formal.txt", "te_native_formal.txt"),
        ("FAKE_native_informal.txt", "te_native_informal.txt"),
        ("FAKE_romanized_informal.txt", "te_romanized_informal.txt"),
    ]:
        (raw_dir / dst_name).write_text((FIXTURE / src_name).read_text(encoding="utf-8"), encoding="utf-8")

    (raw_dir / "hi_native_formal.txt").write_text("यह एक औपचारिक हिंदी वाक्य है।\nदूसरा औपचारिक वाक्य यहाँ है।\n", encoding="utf-8")
    (raw_dir / "hi_native_informal.txt").write_text("यह मेरी हिंदी पोस्ट है\nआज मौसम बहुत अच्छा है\n", encoding="utf-8")
    (raw_dir / "hi_romanized_informal.txt").write_text("yah meri hindi post hai\naaj mausam bahut accha hai\n", encoding="utf-8")
    (raw_dir / "english_wiki.txt").write_text("This is an English baseline sentence.\nHere is another English sentence for testing.\n", encoding="utf-8")

    config = {
        "corpus_dir": str(processed),
        "minimal_pairs_path": str(ROOT / "tests" / "fixtures" / "toy_corpus" / "FAKE_minimal_pairs.tsv"),
        "tokenizer_set": "quick_test",
        "pricing_path": str(ROOT / "configs" / "pricing.yaml"),
        "run_tag": "multi",
        "languages": [
            {"name": "telugu", "code": "te", "prefix": "te", "primary": True},
            {"name": "hindi", "code": "hi", "prefix": "hi"},
        ],
        "english_register": "english_wiki",
    }
    config_path = tmp_path / "multilang.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    _run_script(
        "00_run_full_pipeline.py",
        "--config", str(config_path),
        "--run_tag", "multi",
        "--experiments-dir", str(experiments),
        "--raw-dir", str(raw_dir),
        "--n-bootstrap", "200",
    )

    exp_dir = next(experiments.glob("*_multi"))
    gap_df = pd.read_csv(exp_dir / "results" / "script_fairness_gap.csv")
    assert set(gap_df["language"]) == {"telugu", "hindi"}

    significance = json.loads((exp_dir / "results" / "script_gap_significance.json").read_text(encoding="utf-8"))
    assert set(significance.keys()) == {"telugu", "hindi"}
    assert (exp_dir / "results" / "per_sentence_gaps.csv").exists()


def test_standalone_scripts_accept_experiment_dir(pipeline_env):
    env = pipeline_env

    _run_script(
        "01_build_corpus.py",
        "--config", str(env["config_path"]),
        "--raw-dir", str(env["raw"]),
        "--output-dir", str(env["processed"]),
    )

    _run_script(
        "02_run_tokenizer_audit.py",
        "--config", str(env["config_path"]),
        "--run_tag", "standalone",
        "--experiments-dir", str(env["experiments"]),
    )

    exp_dirs = list(env["experiments"].glob("*_standalone"))
    assert len(exp_dirs) == 1
    exp_dir = exp_dirs[0]

    _run_script(
        "03_run_minimal_pair_audit.py",
        "--config", str(env["config_path"]),
        "--experiment-dir", str(exp_dir),
    )

    assert (exp_dir / "results" / "fertility_by_register.csv").exists()
    assert (exp_dir / "results" / "minimal_pair_fertility.csv").exists()
    assert len(list(env["experiments"].glob("*_standalone*"))) == 1
