import numpy as np
import pytest

from expengine.inference import itt

TRUE_ABSOLUTE_EFFECT = 0.004
TRUE_RELATIVE_LIFT = 0.2


def test_recovers_a_known_absolute_effect(simple_experiment):
    treatment, outcome = simple_experiment()
    effect = itt.binary_effect_from_arrays(outcome, treatment)
    assert abs(effect["absolute_effect"] - TRUE_ABSOLUTE_EFFECT) < 3.0 * effect["standard_error"]
    assert effect["confidence_low"] < TRUE_ABSOLUTE_EFFECT < effect["confidence_high"]
    assert effect["p_value"] < 0.01


def test_relative_lift_interval_covers_the_truth(simple_experiment):
    treatment, outcome = simple_experiment()
    effect = itt.binary_effect_from_arrays(outcome, treatment)
    assert effect["relative_low"] < TRUE_RELATIVE_LIFT < effect["relative_high"]


def test_relative_interval_is_not_symmetric(simple_experiment):
    treatment, outcome = simple_experiment()
    effect = itt.binary_effect_from_arrays(outcome, treatment)
    upper = effect["relative_high"] - effect["relative_lift"]
    lower = effect["relative_lift"] - effect["relative_low"]
    assert upper > lower


def test_interval_coverage_is_close_to_nominal():
    replicates = 200
    covered = 0
    for index in range(replicates):
        stream = np.random.default_rng(4000 + index)
        rows = 40_000
        treatment = (stream.random(rows) < 0.5).astype(np.int8)
        probability = np.where(treatment == 1, 0.104, 0.1)
        outcome = (stream.random(rows) < probability).astype(np.int8)
        effect = itt.binary_effect_from_arrays(outcome, treatment)
        if effect["confidence_low"] <= TRUE_ABSOLUTE_EFFECT <= effect["confidence_high"]:
            covered += 1
    assert 0.90 <= covered / replicates <= 0.99


def test_relative_lift_is_undefined_without_control_successes():
    effect = itt.two_proportion_effect(10.0, 1_000, 0.0, 1_000)
    assert np.isnan(effect["relative_lift"])


def test_tiny_arms_are_rejected():
    with pytest.raises(ValueError):
        itt.difference_in_means(0.5, 0.25, 1, 0.4, 0.24, 1_000)
