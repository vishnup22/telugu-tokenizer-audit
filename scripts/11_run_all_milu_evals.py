                      
"""Run configured MILU evaluations across all supported languages in a config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.benchmarks.milu import milu_config_name, milu_language_slug
from telugu_audit.run_utils import load_yaml_config

EVALUATOR_SCRIPTS = {
    "claude": "scripts/07_run_milu_eval.py",
    "sarvam": "scripts/09_run_milu_eval_sarvam.py",
    "dravidian": "scripts/08_run_milu_eval_dravidian.py",
}


def _configured_languages(config: dict) -> list[str]:
    languages = []
    for entry in config.get("languages", []):
        name = entry.get("name") or entry.get("code")
        if not name:
            continue
        languages.append(milu_language_slug(name))
    return languages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_multilingual.yaml")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--evaluators", nargs="+", default=["claude", "sarvam"])
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    languages = _configured_languages(config)

    if not languages:
        raise SystemExit("No languages found in config.")

    commands: list[list[str]] = []
    for evaluator in args.evaluators:
        script = EVALUATOR_SCRIPTS.get(evaluator)
        if script is None:
            raise SystemExit(f"Unknown evaluator '{evaluator}'. Valid options: {sorted(EVALUATOR_SCRIPTS)}")

        for language in languages:
            if evaluator == "dravidian" and language != "telugu":
                print(f"Skipping dravidian for {language}: model is Telugu-specific.")
                continue

            command = [
                sys.executable,
                script,
                "--experiment-dir",
                args.experiment_dir,
                "--language",
                language,
            ]
            if args.max_questions is not None:
                command.extend(["--max-questions", str(args.max_questions)])
            commands.append(command)

    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    supported_labels = [milu_config_name(language) for language in languages]
    print(f"Completed MILU eval scheduling for: {', '.join(supported_labels)}")


if __name__ == "__main__":
    main()
