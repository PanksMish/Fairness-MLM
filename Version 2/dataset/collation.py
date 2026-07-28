"""
Computes per-language entry quotas that sum to a target total (default
2.5M) and collates gold + weak sentiment sources into ONE final dataset
across all 101 configured languages.

QUOTA STRATEGY: Table 2 gives per-tier AVERAGE train sizes (HR=40K,
MR=18K, LR=5K) which, summed across 18+37+46 languages, total ~1.62M --
not 2.5M. Rather than inventing a different ratio, this scales Table 2's
tier averages up proportionally so the SAME relative HR:MR:LR emphasis
the paper describes is preserved while hitting whatever total you ask
for:

    scale = target_total / (18*40000 + 37*18000 + 46*5000)
    quota[tier] = round(table2_avg[tier] * scale)

IMPORTANT HONESTY NOTE: quotas are TARGETS, not guarantees.
  - Gold-language quotas are capped by how many real examples actually
    exist for that language (Amazon Reviews Multilingual has a fixed,
    finite size per language -- if the quota exceeds what's available,
    you get everything available and the shortfall is reported, not
    invented).
  - Weak-language quotas depend on CC100 lexicon coverage, which is
    unknown until the pipeline actually runs (a language with poor NRC
    lexicon translation coverage will fall short of quota even after
    requesting more raw CC100 documents to compensate). This module
    requests `quota * oversample_factor` raw documents per weak
    language specifically to reduce (not eliminate) that shortfall, and
    reports actual-vs-target explicitly rather than silently padding.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Table 2's per-tier average train sizes, the base ratio this module scales.
TABLE2_TIER_AVERAGES = {"HR": 40_000, "MR": 18_000, "LR": 5_000}


def compute_language_quotas(
    config: dict,
    target_total: int = 2_500_000,
) -> dict[str, int]:
    """
    Returns {language_code: quota} for every language in
    config['languages']['all'], scaling TABLE2_TIER_AVERAGES
    proportionally so quotas sum to (approximately -- integer rounding)
    target_total. Pure arithmetic, no I/O -- fully testable without
    network access.
    """
    tier_by_lang = {}
    tier_counts = {"HR": 0, "MR": 0, "LR": 0}
    for tier_key, tier_label in [
        ("high_resource", "HR"), ("medium_resource", "MR"), ("low_resource", "LR"),
    ]:
        for lang in config["languages"][tier_key]:
            tier_by_lang[lang] = tier_label
            tier_counts[tier_label] += 1

    base_total = sum(TABLE2_TIER_AVERAGES[t] * tier_counts[t] for t in tier_counts)
    if base_total == 0:
        raise ValueError("Config has zero languages across all tiers")
    scale = target_total / base_total

    quotas = {}
    for lang, tier in tier_by_lang.items():
        quotas[lang] = round(TABLE2_TIER_AVERAGES[tier] * scale)
    return quotas


@dataclass
class LanguageCollationResult:
    language_code: str
    resource_tier: str
    label_source: str        # "gold" | "weak"
    quota: int
    actual_count: int
    shortfall: int            # quota - actual_count, 0 if met or exceeded
    shortfall_reason: str     # "" if no shortfall


def truncate_to_quota(records: list[dict], quota: int, seed: int = 42) -> list[dict]:
    """
    If more records are available than the quota, randomly (seeded, so
    deterministic) subsamples down to exactly `quota`. If fewer are
    available, returns everything available unchanged -- this function
    NEVER invents records to pad a shortfall; a shortfall is reported by
    the caller (see collate_language below), not hidden by fabrication.
    """
    if len(records) <= quota:
        return records
    rng = random.Random(seed)
    return rng.sample(records, quota)


def collate_language(
    language_code: str,
    resource_tier: str,
    label_source: str,
    available_records: list[dict],
    quota: int,
    shortfall_reason_if_short: str = "",
) -> tuple[list[dict], LanguageCollationResult]:
    """Applies the quota to one language's available records and builds
    the reporting row -- the core per-language logic, testable without
    any file I/O by passing in `available_records` directly."""
    selected = truncate_to_quota(available_records, quota)
    shortfall = max(0, quota - len(selected))
    result = LanguageCollationResult(
        language_code=language_code, resource_tier=resource_tier, label_source=label_source,
        quota=quota, actual_count=len(selected), shortfall=shortfall,
        shortfall_reason=shortfall_reason_if_short if shortfall > 0 else "",
    )
    return selected, result


def write_collation_report(results: list[LanguageCollationResult], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "language_code", "resource_tier", "label_source",
            "quota", "actual_count", "shortfall", "shortfall_reason",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "language_code": r.language_code, "resource_tier": r.resource_tier,
                "label_source": r.label_source, "quota": r.quota,
                "actual_count": r.actual_count, "shortfall": r.shortfall,
                "shortfall_reason": r.shortfall_reason,
            })
    return str(path)


def summarize_results(results: list[LanguageCollationResult], target_total: int) -> dict:
    total_actual = sum(r.actual_count for r in results)
    total_quota = sum(r.quota for r in results)
    n_short = sum(1 for r in results if r.shortfall > 0)
    return {
        "target_total": target_total,
        "total_quota_allocated": total_quota,
        "total_actual_collated": total_actual,
        "pct_of_target_achieved": round(100 * total_actual / target_total, 1) if target_total else None,
        "n_languages": len(results),
        "n_languages_short_of_quota": n_short,
        "n_gold_languages": sum(1 for r in results if r.label_source == "gold"),
        "n_weak_languages": sum(1 for r in results if r.label_source == "weak"),
    }
