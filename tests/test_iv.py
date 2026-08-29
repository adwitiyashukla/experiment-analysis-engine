import numpy as np
import pytest

from expengine.inference import itt, iv

TRUE_CACE = 0.05
TRUE_COMPLIANCE = 0.2


def _naive(treatment, exposure, outcome):
    treated = treatment == 1
    control = ~treated
    cross_treated = float(np.count_nonzero((outcome[treated] == 1) & (exposure[treated] == 1)))
    return iv.naive_exposed_comparison(
        cross_treated,
        float(exposure[treated].sum(dtype=np.float64)),
        float(outcome[control].sum(dtype=np.float64)),
        int(control.sum(dtype=np.int64)),
    )


def test_wald_estimator_recovers_a_known_cace(targeted_non_compliance):
    treatment, exposure, outcome = targeted_non_compliance()
    result = iv.cace_from_arrays(outcome, exposure, treatment)
    assert abs(result["cace"] - TRUE_CACE) < 3.0 * result["cace_standard_error"]
    assert result["cace_confidence_low"] < TRUE_CACE < result["cace_confidence_high"]
    assert abs(result["compliance_rate"] - TRUE_COMPLIANCE) < 0.005
    assert result["first_stage_f"] > 1_000.0


def test_naive_exposed_comparison_is_biased_while_the_wald_estimator_is_not(
    targeted_non_compliance,
):
    treatment, exposure, outcome = targeted_non_compliance()
    wald = iv.cace_from_arrays(outcome, exposure, treatment)
    naive = _naive(treatment, exposure, outcome)
    assert naive["absolute_effect"] > TRUE_CACE + 5.0 * naive["standard_error"]
    assert abs(wald["cace"] - TRUE_CACE) < abs(naive["absolute_effect"] - TRUE_CACE)


def test_itt_is_much_smaller_than_the_cace_under_low_compliance(targeted_non_compliance):
    treatment, exposure, outcome = targeted_non_compliance()
    result = iv.cace_from_arrays(outcome, exposure, treatment)
    assert result["itt_outcome"] < result["cace"]
    assert abs(result["cace_over_itt"] - 1.0 / TRUE_COMPLIANCE) < 0.5


def test_an_instrument_that_moves_nothing_is_rejected():
    with pytest.raises(ValueError):
        iv.cace_from_counts(1_000, 1_000, 50.0, 40.0, 0.0, 0.0, 0.0, 0.0)


def test_incrementality_matches_the_counterfactual(simple_experiment):
    treatment, outcome = simple_experiment()
    effect = itt.binary_effect_from_arrays(outcome, treatment)
    result = iv.incrementality(effect, effect["successes_treated"] * 0.5)
    assert result["incremental_conversions"] == pytest.approx(
        result["conversions_treated_arm"] - result["counterfactual_conversions"], rel=1e-9
    )
    assert result["overstatement_treated_arm"] > 1.0
    assert result["incremental_low"] < result["incremental_conversions"]
    assert result["incremental_high"] > result["incremental_conversions"]


def test_incrementality_rejects_a_non_positive_effect():
    effect = {
        "rows_treated": 1_000,
        "mean_control": 0.05,
        "absolute_effect": -0.01,
        "successes_treated": 40.0,
        "confidence_low": -0.02,
        "confidence_high": 0.0,
    }
    with pytest.raises(ValueError):
        iv.incrementality(effect, 10.0)
