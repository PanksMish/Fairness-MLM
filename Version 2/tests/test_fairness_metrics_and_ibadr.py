import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from evaluation.fairness_metrics import (
    counterfactual_consistency_rate,
    ccr_from_logits,
    demographic_parity_gap,
    instance_weighted_global_metric,
    unweighted_group_average,
    resource_category,
)
from fairness.ibadr import IBADRScheduler, IBADRConfig


def test_ccr_all_consistent_is_one():
    preds = np.array([0, 1, 2, 1, 0])
    assert counterfactual_consistency_rate(preds, preds) == 1.0


def test_ccr_all_inconsistent_is_zero():
    a = np.array([0, 0, 0])
    b = np.array([1, 1, 1])
    assert counterfactual_consistency_rate(a, b) == 0.0


def test_ccr_partial():
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 1, 9, 9])
    assert counterfactual_consistency_rate(a, b) == 0.5


def test_ccr_from_logits_matches_argmax():
    logits_a = np.array([[0.1, 0.9], [0.8, 0.2]])
    logits_b = np.array([[0.2, 0.8], [0.3, 0.7]])  # second flips argmax
    ccr = ccr_from_logits(logits_a, logits_b)
    assert ccr == 0.5


def test_dpg_zero_when_equal_rates():
    preds = np.array([1, 0, 1, 0])
    groups = np.array(["a", "a", "b", "b"])
    dpg = demographic_parity_gap(preds, groups, positive_class=1)
    assert dpg == 0.0


def test_dpg_detects_disparity():
    preds = np.array([1, 1, 1, 0, 0, 0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    dpg = demographic_parity_gap(preds, groups, positive_class=1)
    assert dpg == 1.0  # group a: 100% positive, group b: 0%


def test_instance_weighted_global_metric_eq16():
    vals = {"en": 85.2, "sw": 78.0}
    n = {"en": 1000, "sw": 100}
    global_val = instance_weighted_global_metric(vals, n)
    expected = (1000 * 85.2 + 100 * 78.0) / 1100
    assert abs(global_val - expected) < 1e-9


def test_unweighted_group_average_matches_manual():
    vals = {"en": 85.0, "de": 83.0, "fr": 87.0, "sw": 78.0}
    hr_langs = ["en", "de", "fr"]
    avg = unweighted_group_average(vals, hr_langs)
    assert abs(avg - (85.0 + 83.0 + 87.0) / 3) < 1e-9


def test_resource_category_thresholds_sec31():
    assert resource_category(2e9) == "HR"
    assert resource_category(1.0000001e9) == "HR"
    assert resource_category(5e8) == "MR"
    assert resource_category(1.0000001e8) == "MR"
    assert resource_category(5e7) == "LR"
    assert resource_category(1e8) == "LR"  # boundary: T_l <= 1e8 is LR


def test_ibadr_selects_correct_top_fraction():
    cfg = IBADRConfig(refresh_interval=10, selection_ratio=0.2)
    sched = IBADRScheduler(cfg)
    bts = np.array([0.1, 0.9, 0.3, 0.8, 0.05, 0.7, 0.2, 0.6, 0.15, 0.5])  # n=10
    selected = sched.select_top_divergence(bts)
    assert len(selected) == 2  # 20% of 10
    # highest two values are indices 1 (0.9) and 3 (0.8)
    assert set(selected.tolist()) == {1, 3}


def test_ibadr_should_refresh_on_interval():
    sched = IBADRScheduler(IBADRConfig(refresh_interval=5))
    assert not sched.should_refresh(0)
    assert not sched.should_refresh(4)
    assert sched.should_refresh(5)
    assert sched.should_refresh(10)
    assert not sched.should_refresh(11)


def test_ibadr_refresh_step_replaces_selected_samples():
    sched = IBADRScheduler(IBADRConfig(refresh_interval=1, selection_ratio=0.5))
    dataset = ["s0", "s1", "s2", "s3"]
    bts = np.array([0.1, 0.9, 0.2, 0.8])  # top 2 (rho=0.5*4=2) -> indices 1, 3

    def regen(sample):
        return sample + "_regen"

    new_dataset, event = sched.refresh_step(step=1, dataset=dataset, per_sample_bts=bts, regenerate_fn=regen)
    assert event is not None
    assert event.n_selected == 2
    assert new_dataset[1] == "s1_regen"
    assert new_dataset[3] == "s3_regen"
    assert new_dataset[0] == "s0"  # untouched
    assert new_dataset[2] == "s2"  # untouched


def test_ibadr_no_refresh_off_interval():
    sched = IBADRScheduler(IBADRConfig(refresh_interval=100))
    dataset = ["s0", "s1"]
    bts = np.array([0.1, 0.9])
    new_dataset, event = sched.refresh_step(step=1, dataset=dataset, per_sample_bts=bts, regenerate_fn=lambda s: s)
    assert event is None
    assert new_dataset == dataset


def test_ibadr_append_mode_grows_dataset():
    sched = IBADRScheduler(IBADRConfig(refresh_interval=1, selection_ratio=0.5))
    dataset = ["s0", "s1"]
    bts = np.array([0.1, 0.9])
    new_dataset, event = sched.refresh_step(
        step=1, dataset=dataset, per_sample_bts=bts,
        regenerate_fn=lambda s: s + "_new", replace=False,
    )
    assert len(new_dataset) == 3  # original 2 + 1 appended (top 50% of 2 = 1)


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
