import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.build_full_sentiment_dataset import build_language_plan, coverage_report, GOLD_LABELED_LANGUAGES


def _fake_config():
    return {
        "languages": {
            "high_resource": ["en", "de", "zh"],
            "medium_resource": ["hi", "sw"],
            "low_resource": ["yo", "am"],
            "all": ["en", "de", "zh", "hi", "sw", "yo", "am"],
        }
    }


def test_build_language_plan_marks_gold_languages_correctly():
    config = _fake_config()
    plan = build_language_plan(config)
    assert plan["en"] == "gold"
    assert plan["de"] == "gold"
    assert plan["zh"] == "gold"
    assert plan["hi"] == "weak"
    assert plan["sw"] == "weak"
    assert plan["yo"] == "weak"
    assert plan["am"] == "weak"


def test_build_language_plan_covers_every_configured_language():
    config = _fake_config()
    plan = build_language_plan(config)
    assert set(plan.keys()) == set(config["languages"]["all"])


def test_gold_labeled_languages_matches_real_amazon_reviews_coverage():
    assert GOLD_LABELED_LANGUAGES == {"en", "de", "es", "fr", "ja", "zh"}


def test_coverage_report_assigns_correct_tiers():
    config = _fake_config()
    plan = build_language_plan(config)
    report = coverage_report(plan, config)

    by_lang = {r["language_code"]: r for r in report}
    assert by_lang["en"]["resource_tier"] == "HR"
    assert by_lang["hi"]["resource_tier"] == "MR"
    assert by_lang["yo"]["resource_tier"] == "LR"


def test_coverage_report_marks_gold_flag_correctly():
    config = _fake_config()
    plan = build_language_plan(config)
    report = coverage_report(plan, config)
    by_lang = {r["language_code"]: r for r in report}
    assert by_lang["en"]["is_gold_label"] is True
    assert by_lang["sw"]["is_gold_label"] is False
    assert by_lang["en"]["label_source"] == "gold"
    assert by_lang["sw"]["label_source"] == "weak"


def test_coverage_report_covers_all_languages_sorted():
    config = _fake_config()
    plan = build_language_plan(config)
    report = coverage_report(plan, config)
    codes = [r["language_code"] for r in report]
    assert codes == sorted(codes)
    assert len(report) == 7


def test_real_default_config_yields_101_languages_with_6_gold():
    """Sanity-checks the actual shipped configs/default_config.yaml
    against this combiner -- catches any future drift between the
    config's language list and what the combiner expects."""
    import yaml
    import pathlib
    config_path = pathlib.Path(__file__).parent.parent / "configs" / "default_config.yaml"
    config = yaml.safe_load(open(config_path))

    plan = build_language_plan(config)
    assert len(plan) == 101

    n_gold = sum(1 for v in plan.values() if v == "gold")
    n_weak = sum(1 for v in plan.values() if v == "weak")
    assert n_gold == 6  # en, de, es, fr, ja, zh -- all must be in the config
    assert n_weak == 95

    report = coverage_report(plan, config)
    tier_counts = {}
    for row in report:
        tier_counts[row["resource_tier"]] = tier_counts.get(row["resource_tier"], 0) + 1
    assert tier_counts["HR"] == 18
    assert tier_counts["MR"] == 37
    assert tier_counts["LR"] == 46


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
