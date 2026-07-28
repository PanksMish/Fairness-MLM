"""
Builds the augmented dataset D_aug from Algorithm 1's outputs, Eq. (10):

    D_aug = D U { (x^(b), y, b) : (x, y, a) in D }

i.e. runs the CounterfactualDataEngine over every sample in D and keeps
only accepted candidates, unioned with the original dataset (Algorithm
1's "Add ... to D_aug" only happens inside the `if S >= gamma` branch,
line 9-11 -- rejected candidates are dropped, not added).
"""

from __future__ import annotations

from dataclasses import dataclass

from fairness.counterfactual_generation import CounterfactualDataEngine


@dataclass
class AugmentationStats:
    n_original: int
    n_attempted: int          # samples where an attribute was detected and a candidate generated
    n_accepted: int           # samples passing Eq. 9's acceptance criterion
    n_skipped_no_attribute: int
    n_skipped_no_candidate: int
    n_rejected_morphology: int
    n_rejected_score: int


def build_augmented_dataset(
    dataset: list[dict],
    engine: CounterfactualDataEngine,
    languages_by_sample: list[str],
    all_attributes: list[str],
    text_field: str = "text",
    label_field: str = "label",
    attribute_field: str = "attribute",
) -> tuple[list[dict], AugmentationStats]:
    """
    Args:
        dataset: list of {text_field: x, label_field: y, attribute_field: a (optional)}
        engine: a configured CounterfactualDataEngine
        languages_by_sample: parallel list giving each sample's language
            (needed for the engine's resource-tier dispatch)
        all_attributes: the full attribute label space A (e.g. ["m", "f"])

    Returns:
        (D_aug, stats) where D_aug = D U {accepted counterfactuals}, each
        counterfactual entry carrying the SAME label y and the NEW
        attribute b (Eq. 10), plus bookkeeping on why samples were or
        weren't augmented.
    """
    if len(dataset) != len(languages_by_sample):
        raise ValueError("dataset and languages_by_sample must be the same length")

    d_aug = list(dataset)  # D_aug <- D  (line 1)
    n_attempted = n_accepted = 0
    n_skipped_no_attribute = n_skipped_no_candidate = 0
    n_rejected_morphology = n_rejected_score = 0

    for sample, language in zip(dataset, languages_by_sample):
        text = sample[text_field]
        declared_attribute = sample.get(attribute_field)

        candidate = engine.generate_one(
            text, language, all_attributes, attribute_from=declared_attribute,
        )

        if candidate is None:
            # Could be no-attribute-detected OR no-candidate-generated;
            # the engine doesn't currently distinguish these in its
            # return value, so we re-check cheaply for bookkeeping.
            if declared_attribute is None and engine.detector(text) is None:
                n_skipped_no_attribute += 1
            else:
                n_skipped_no_candidate += 1
            continue

        n_attempted += 1

        if candidate.score == float("-inf"):
            n_rejected_morphology += 1
            continue

        if not candidate.accepted:
            n_rejected_score += 1
            continue

        n_accepted += 1
        new_sample = dict(sample)
        new_sample[text_field] = candidate.candidate_text
        new_sample[attribute_field] = candidate.attribute_to
        # label_field (y) is carried over UNCHANGED, per Eq. 10 and
        # assumption A1 (Semantic Invariance): y(x) = y(x^(b))
        d_aug.append(new_sample)

    stats = AugmentationStats(
        n_original=len(dataset),
        n_attempted=n_attempted,
        n_accepted=n_accepted,
        n_skipped_no_attribute=n_skipped_no_attribute,
        n_skipped_no_candidate=n_skipped_no_candidate,
        n_rejected_morphology=n_rejected_morphology,
        n_rejected_score=n_rejected_score,
    )
    return d_aug, stats
