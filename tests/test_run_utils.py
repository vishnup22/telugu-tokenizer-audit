from pathlib import Path

import pytest

from telugu_audit.corpus.loaders import load_corpus, load_minimal_pairs
from telugu_audit.run_utils import new_experiment_dir


def test_new_experiment_dir_refuses_overwrite(tmp_path):
    first = new_experiment_dir("dup", base_dir=str(tmp_path))
    assert first.exists()
    with pytest.raises(FileExistsError):
        new_experiment_dir("dup", base_dir=str(tmp_path))


def test_load_minimal_pairs_rejects_bad_morph_type(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("word\tgloss\tmorph_type\nX\tY\tnot_a_real_type\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown morph_type"):
        load_minimal_pairs(bad)


def test_load_corpus_from_fixture():
    root = Path(__file__).resolve().parents[1]
                                                       
    raw = root / "tests" / "fixtures" / "toy_corpus"
    processed = root / "tests" / "fixtures" / "toy_corpus" / "_processed"
    processed.mkdir(exist_ok=True)
    for register in ("native_formal", "native_informal", "romanized_informal"):
        src = raw / f"FAKE_{register}.txt"
        dst = processed / f"{register}.txt"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    lines = load_corpus(processed, "native_formal")
    assert len(lines) >= 1
