import numpy as np
import pandas as pd
import pytest

from expengine.guardrails import balance, srm


def test_srm_passes_a_fair_split():
    result = srm.sample_ratio_mismatch(8_500_000, 1_500_000, 0.85)
    assert result["passes"]
    assert result["p_value"] > 0.001
    assert result["observed_treatment_share"] == pytest.approx(0.85)


def test_srm_detects_a_broken_split():
    result = srm.sample_ratio_mismatch(8_000_000, 2_000_000, 0.85)
    assert not result["passes"]
    assert result["p_value"] < 1e-10
    assert result["chi_square"] > 100.0


def test_srm_rejects_an_impossible_intended_share():
    with pytest.raises(ValueError):
        srm.sample_ratio_mismatch(10, 10, 0.0)


def test_srm_rejects_an_empty_experiment():
    with pytest.raises(ValueError):
        srm.sample_ratio_mismatch(0, 0)


def test_balance_sits_near_zero_under_randomisation(generator):
    rows = 400_000
    treatment = (generator.random(rows) < 0.85).astype(np.int8)
    values = generator.normal(size=rows).astype(np.float32)
    result = balance.feature_balance("f0", values, treatment)
    assert abs(result["standardised_mean_difference"]) < 0.02
    assert result["rows_treated"] + result["rows_control"] == rows


def test_balance_flags_a_shifted_feature(generator):
    rows = 200_000
    treatment = (generator.random(rows) < 0.5).astype(np.int8)
    values = generator.normal(size=rows) + 0.3 * treatment
    result = balance.feature_balance("f0", values, treatment)
    assert result["standardised_mean_difference"] > 0.25
    assert result["p_value"] < 1e-10


def test_balance_summary_counts_features_above_the_threshold():
    table = pd.DataFrame(
        {
            "standardised_mean_difference": [0.001, -0.02, 0.3],
            "z_statistic": [0.4, -1.1, 42.0],
        }
    )
    summary = balance.balance_summary(table, threshold=0.1, z_threshold=3.0)
    assert summary["features"] == 3
    assert summary["features_above_threshold"] == 1
    assert not summary["passes"]
    assert summary["max_absolute_smd"] == pytest.approx(0.3)
    assert summary["features_beyond_chance"] == 1
    assert summary["max_absolute_z"] == pytest.approx(42.0)


def test_balance_needs_both_arms():
    values = np.arange(10, dtype=np.float32)
    treatment = np.ones(10, dtype=np.int8)
    with pytest.raises(ValueError):
        balance.feature_balance("f0", values, treatment)
