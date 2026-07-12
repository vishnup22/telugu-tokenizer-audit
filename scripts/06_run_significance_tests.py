                      
"""Run per-sentence significance tests on the native vs romanized script gap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.analysis.posthoc import run_script_gap_significance_stage
from telugu_audit.run_utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Experiment directory to write results into"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    sig_path = run_script_gap_significance_stage(
        args.experiment_dir,
        config,
        config_path,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    results = json.loads(sig_path.read_text(encoding="utf-8"))
    for language, language_results in results.items():
        print(f"[{language}]")
        for name, stats in language_results.items():
            if "error" in stats:
                print(f"  {name} ... skipped - {stats['error']}")
            else:
                print(
                    f"  {name} ... p={stats['p_value']:.4f}  "
                    f"median_gap={stats['median_gap']:.3f}  "
                    f"CI=[{stats['bootstrap_ci_95_low']:.3f}, {stats['bootstrap_ci_95_high']:.3f}]  "
                    f"r_rb={stats['rank_biserial_correlation']:.3f}  "
                    f"direction={stats['median_direction']}"
                )

    print(f"\nWrote {sig_path}")


if __name__ == "__main__":
    main()
