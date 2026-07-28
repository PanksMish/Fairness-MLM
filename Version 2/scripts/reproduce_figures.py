"""
Reads evaluation-report JSONs (scripts/evaluate.py's output) and
generates Fig. 4/5/6-style figures via visualization/plots.py. Same
"formatting only, no metric computation" scope as reproduce_tables.py.

    python scripts/reproduce_figures.py \\
        --reports mT5-FT=outputs/mt5_ft_eval.json ADAPT-BTS=outputs/adapt_bts_eval.json \\
        --highlight ADAPT-BTS --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reproduce_tables import load_reports
from visualization.plots import bar_comparison

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate Fig. 4/5-style figures from evaluation reports.")
    parser.add_argument("--reports", nargs="+", required=True, help="method_name=path.json entries")
    parser.add_argument("--highlight", default=None, help="method name to draw in a distinct color")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    reports = load_reports(args.reports)
    out_dir = Path(args.out_dir)
    names = list(reports.keys())

    # Fig. 4-style: task metric bar chart
    task_values = [reports[n]["global"].get("Macro-F1 or Span-F1") for n in names]
    if all(v is not None for v in task_values):
        path = bar_comparison(
            names, task_values, ylabel="Task metric",
            title="Global predictive performance", highlight_label=args.highlight,
            save_path=out_dir / "figure4_task_metric.png",
        )
        logger.info("Wrote %s", path)

    # Fig. 5-style: BTS bar chart (lower is better)
    bts_values = [reports[n]["global"].get("BTS") for n in names]
    if all(v is not None for v in bts_values):
        path = bar_comparison(
            names, bts_values, ylabel="BTS", title="Fairness comparison (BTS)",
            lower_is_better=True, highlight_label=args.highlight,
            save_path=out_dir / "figure5_bts_comparison.png",
        )
        logger.info("Wrote %s", path)

    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
