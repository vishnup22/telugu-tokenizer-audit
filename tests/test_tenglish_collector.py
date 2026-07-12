from __future__ import annotations

import json
from pathlib import Path

from telugu_audit.corpus.collectors.tenglish_collector import collect_tenglish_informal


def test_collect_tenglish_informal_filters_and_samples(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    raw_path = sources_dir / "youtube.jsonl"

    records = [
        {
            "text": "nenu ippude chusa bagundi movie",
            "url": "https://youtube.com/watch?v=abc",
            "source": "youtube",
            "comment_id": "1",
        },
        {
            "text": "This is only English text and should be rejected",
            "url": "https://youtube.com/watch?v=def",
            "source": "youtube",
            "comment_id": "2",
        },
        {
            "text": "meeru chala bagunnaru super acting",
            "url": "https://youtube.com/watch?v=ghi",
            "source": "youtube",
            "comment_id": "3",
        },
    ]
    with raw_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    output_path = tmp_path / "tenglish_informal.txt"
    metadata_path = tmp_path / "tenglish_informal.meta.jsonl"

    n = collect_tenglish_informal(
        sources_dir=sources_dir,
        output_path=output_path,
        metadata_path=metadata_path,
        n_samples=10,
        seed=42,
    )

    assert n == 2
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert any("nenu" in line for line in lines)
    assert any("meeru" in line for line in lines)

    meta_lines = metadata_path.read_text(encoding="utf-8").splitlines()
    assert len(meta_lines) == 2
    payload = [json.loads(line) for line in meta_lines]
    assert all(row["source"] == "youtube" for row in payload)
