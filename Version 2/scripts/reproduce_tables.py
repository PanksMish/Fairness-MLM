"""
Reads one evaluation-report JSON per method (the output of
scripts/evaluate.py) and assembles them into Table 5/8-shaped outputs
via evaluation/report.py.

    python scripts/reproduce_tables.py \\
        --reports mT5-FT=outputs/mt5_ft_eval.json ADAPT-BTS=outputs/adapt_bts_eval.json \\
        --out-dir outputs/tables

Each --reports entry is `method_name=path_to_json`. This script performs
NO metric computation itself -- it only reformats whatever
scripts/evaluate.py already computed and wrote to disk into table form.
If any of those JSON files don't exist yet, that's because no real
evaluation run has happened -- this script can't fabricate the numbers
for you, and won't try to.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.report import global_comparison_table, per_language_table, save_table

logger = logging.getLogger(__name__)


def load_reports(report_args: list[str]) -> dict[str, dict]:
    reports = {}
    for arg in report_args:
        if "=" not in arg:
            raise ValueError(f"Invalid --reports entry '{arg}', expected method_name=path.json")
        name, path = arg.split("=", 1)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No evaluation report found at '{path}' for method '{name}'. "
                "Run scripts/evaluate.py first to produce it -- this script only "
                "reformats existing evaluation output, it cannot generate it."
            )
        with open(path) as f:
            reports[name] = json.load(f)
    return reports


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Assemble Table 5/8-style outputs from evaluation reports.")
    parser.add_argument("--reports", nargs="+", required=True, help="method_name=path.json entries")
    parser.add_argument("--out-dir", default="outputs/tables")
    args = parser.parse_args()

    reports = load_reports(args.reports)
    out_dir = Path(args.out_dir)

    # Table 5: global comparison
    global_results = {name: r["global"] for name, r in reports.items()}
    global_df = global_comparison_table(global_results)
    save_table(global_df, out_dir / "table5_global_comparison.csv", fmt="csv")
    save_table(global_df, out_dir / "table5_global_comparison.md", fmt="markdown")
    logger.info("Wrote Table 5 to %s", out_dir / "table5_global_comparison.csv")

    # Table 8: per-language comparison, if exactly two methods given
    # (base vs. ours -- Table 8's actual shape needs exactly a pair)
    if len(reports) == 2:
        names = list(reports.keys())
        base_report, ours_report = reports[names[0]], reports[names[1]]
        per_lang = {}
        for lang in base_report["per_language"]:
            if lang not in ours_report["per_language"]:
                continue
            per_lang[lang] = {
                "base_f1": base_report["per_language"][lang]["task_metric"],
                "ours_f1": ours_report["per_language"][lang]["task_metric"],
                "base_bts": base_report["per_language"][lang].get("bts"),
                "ours_bts": ours_report["per_language"][lang].get("bts"),
            }
        per_lang_df = per_language_table(per_lang)
        save_table(per_lang_df, out_dir / "table8_per_language.csv", fmt="csv")
        logger.info("Wrote Table 8 to %s", out_dir / "table8_per_language.csv")
    else:
        logger.info("Skipping Table 8 (per-language comparison needs exactly 2 --reports entries: base and ours)")

    print(f"Tables written to {out_dir}")


if __name__ == "__main__":
    main()
