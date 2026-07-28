import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from baselines.losses_core import (
    cosine_similarity_matrix,
    supervised_contrastive_loss,
    build_language_fusion_positive_mask,
    build_debiasing_positive_mask,
    language_inverse_frequency_weights,
    unlearning_selection_mask,
)


def test_cosine_similarity_matrix_diagonal_is_one():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sim = cosine_similarity_matrix(embeddings)
    assert np.allclose(np.diag(sim), 1.0)


def test_cosine_similarity_matrix_symmetric():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(5, 8))
    sim = cosine_similarity_matrix(embeddings)
    assert np.allclose(sim, sim.T)


def test_supervised_contrastive_loss_hand_computed_two_positives():
    # 3 samples: anchor 0 and 1 are positives for each other (identical
    # embeddings -> sim=1); sample 2 is a negative (orthogonal).
    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],   # identical to anchor 0 -> positive, sim=1
        [0.0, 1.0],   # orthogonal -> negative, sim=0
    ])
    positive_mask = np.array([
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])
    tau = 1.0
    loss = supervised_contrastive_loss(embeddings, positive_mask, temperature=tau)

    # Hand computation for anchor 0: sim(0,1)=1, sim(0,2)=0
    # denom = exp(1/1) + exp(0/1) = e + 1
    # loss_0 = -log( exp(1)/(e+1) ) = -(1 - log(e+1)) = log(e+1) - 1
    expected_per_anchor = np.log(np.e + 1) - 1.0
    # anchors 0 and 1 are symmetric (identical embeddings), anchor 2 has no positives (excluded)
    assert abs(loss - expected_per_anchor) < 1e-6


def test_supervised_contrastive_loss_zero_when_no_positives_exist():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    positive_mask = np.zeros((2, 2), dtype=bool)
    loss = supervised_contrastive_loss(embeddings, positive_mask)
    assert loss == 0.0


def test_supervised_contrastive_loss_rejects_wrong_mask_shape():
    import pytest
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bad_mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        supervised_contrastive_loss(embeddings, bad_mask)


def test_supervised_contrastive_loss_lower_when_positives_more_similar():
    rng = np.random.default_rng(1)
    n, d = 6, 16
    labels = np.array([0, 0, 0, 1, 1, 1])

    # Case A: same-label samples have similar embeddings (small noise
    # around shared per-label centroids) -> should give LOW loss
    centroids = rng.normal(size=(2, d)) * 5  # well-separated centroids
    close_embeddings = np.array([centroids[l] + rng.normal(scale=0.01, size=d) for l in labels])

    # Case B: same-label samples have unrelated (random) embeddings -> HIGH loss
    random_embeddings = rng.normal(size=(n, d))

    positive_mask = labels[:, None] == labels[None, :]

    loss_close = supervised_contrastive_loss(close_embeddings, positive_mask, temperature=0.1)
    loss_random = supervised_contrastive_loss(random_embeddings, positive_mask, temperature=0.1)

    assert loss_close < loss_random


def test_language_fusion_positive_mask_eq2_semantics():
    labels = np.array([0, 0, 0, 1])
    languages = np.array(["en", "en", "fr", "en"])
    mask = build_language_fusion_positive_mask(labels, languages)
    # sample 0 (label 0, en): positive with sample 2 (label 0, fr) only
    # (sample 1 has same label but same language -> not a positive per Eq 2's l_t != l_i)
    assert mask[0, 1] == False
    assert mask[0, 2] == True
    assert mask[0, 3] == False  # different label


def test_debiasing_positive_mask_eq4_semantics():
    labels = np.array([0, 0, 0, 1])
    attributes = np.array(["m", "m", "f", "m"])
    mask = build_debiasing_positive_mask(labels, attributes)
    # sample 0 (label 0, m): positive with sample 2 (label 0, f) only
    assert mask[0, 1] == False  # same attribute, not a positive
    assert mask[0, 2] == True   # same label, different attribute
    assert mask[0, 3] == False  # different label


def test_language_inverse_frequency_weights_favors_low_resource():
    counts = {"en": 5e9, "sw": 5e7}  # en is 100x more resourced than sw
    weights = language_inverse_frequency_weights(counts)
    assert weights["sw"] > weights["en"]
    assert abs(weights["sw"] / weights["en"] - 100.0) < 1e-6


def test_language_inverse_frequency_weights_normalized_to_mean_one():
    counts = {"en": 1e9, "de": 2e9, "fr": 4e9}
    weights = language_inverse_frequency_weights(counts)
    assert abs(np.mean(list(weights.values())) - 1.0) < 1e-9


def test_language_inverse_frequency_weights_rejects_nonpositive_counts():
    import pytest
    with pytest.raises(ValueError):
        language_inverse_frequency_weights({"en": 0.0})


def test_unlearning_selection_mask_picks_highest_bts():
    bts = np.array([0.1, 0.9, 0.3, 0.8, 0.05])
    mask = unlearning_selection_mask(bts, top_fraction=0.4)  # top 2 of 5
    assert mask.sum() == 2
    assert mask[1] and mask[3]  # the two highest values
    assert not mask[0] and not mask[2] and not mask[4]


def test_unlearning_selection_mask_at_least_one_sample():
    bts = np.array([0.1, 0.2, 0.3])
    mask = unlearning_selection_mask(bts, top_fraction=0.01)  # rounds to 0 -> forced to 1
    assert mask.sum() == 1


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
