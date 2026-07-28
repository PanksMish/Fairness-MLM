"""
Reads an evaluation report JSON's per-language BTS/DPG (or any two
per-language metrics) and runs the Appendix A regression/residual
analysis via appendix/regression_analysis.py.

    python scripts/reproduce_appendix.py \\
        --report outputs/adapt_bts_eval.json \\
        --x-metric bts --y-metric dpg --out-dir outputs/appendix

Note: `evaluation/evaluator.py`'s `evaluate_sentiment` does populate
per-language DPG when the sentiment report has it; `evaluate_ner` does
NOT compute DPG at all (only Span-F1/BTS/CCR). This script works for any
two metrics that ARE populated per-language in whichever report you
give it (e.g. "task_metric" vs "bts", or "task_metric" vs "dpg" for a
sentiment report), and will raise a clear error listing what's actually
available if you ask for a metric that isn't.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appendix.regression_analysis import analyze_relationship, per_language_improvement_summary

logger = logging.getLogger(__name__)


def extract_per_language_metric(report: dict, metric: str) -> dict[str, float]:
    values = {}
    missing = []
    for lang, data in report["per_language"].items():
        if data.get(metric) is None:
            missing.append(lang)
            continue
        values[lang] = data[metric]
    if not values:
        available = set()
        for data in report["per_language"].values():
            available.update(k for k, v in data.items() if v is not None)
        raise ValueError(
            f"Metric '{metric}' is not populated for any language in this report. "
            f"Available non-null per-language metrics: {sorted(available)}"
        )
    if missing:
        logger.warning("Metric '%s' missing for %d/%d languages; excluded from analysis.",
                        metric, len(missing), len(report["per_language"]))
    return values


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run Appendix A-style regression analysis.")
    parser.add_argument("--report", required=True, help="Path to a scripts/evaluate.py JSON output")
    parser.add_argument("--x-metric", default="bts")
    parser.add_argument("--y-metric", default="task_metric")
    parser.add_argument("--out-dir", default="outputs/appendix")
    args = parser.parse_args()

    if not Path(args.report).exists():
        raise FileNotFoundError(f"No report at '{args.report}'. Run scripts/evaluate.py first.")

    with open(args.report) as f:
        report = json.load(f)

    x_values = extract_per_language_metric(report, args.x_metric)
    y_values = extract_per_language_metric(report, args.y_metric)
    common_langs = sorted(set(x_values) & set(y_values))
    if len(common_langs) < 3:
        raise ValueError(
            f"Only {len(common_langs)} languages have both '{args.x_metric}' and "
            f"'{args.y_metric}' populated -- need at least 3 for a meaningful regression."
        )

    import numpy as np
    x = np.array([x_values[l] for l in common_langs])
    y = np.array([y_values[l] for l in common_langs])

    result = analyze_relationship(
        x, y, x_label=args.x_metric, y_label=args.y_metric,
        output_dir=args.out_dir, file_prefix=f"{args.x_metric}_vs_{args.y_metric}",
    )
    logger.info("R^2=%.3f, Pearson r=%.3f, p=%.4g", result.r_squared, result.pearson_r, result.p_value)
    print(f"Appendix analysis written to {args.out_dir}")


if __name__ == "__main__":
    main()
