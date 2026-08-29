import math

import pytest

from expengine.inference import power


def test_minimum_detectable_effect_falls_as_the_sample_grows():
    small = power.minimum_detectable_effect(0.02, 10_000, 10_000)
    large = power.minimum_detectable_effect(0.02, 1_000_000, 1_000_000)
    assert large < small


def test_variance_reduction_shrinks_the_minimum_detectable_effect():
    base = power.minimum_detectable_effect(0.02, 100_000, 100_000)
    reduced = power.minimum_detectable_effect(0.02, 100_000, 100_000, variance_ratio=0.5)
    assert reduced == pytest.approx(base * math.sqrt(0.5), rel=1e-9)


def test_required_rows_round_trips_through_the_minimum_detectable_effect():
    baseline = 0.02
    effect = 0.001
    total = power.required_rows(baseline, effect, 0.5)
    rows_treated = total // 2
    achieved = power.minimum_detectable_effect(baseline, rows_treated, total - rows_treated)
    assert achieved == pytest.approx(effect, rel=0.01)


def test_uneven_allocation_needs_more_users_than_a_balanced_one():
    balanced = power.required_rows(0.02, 0.001, 0.5)
    skewed = power.required_rows(0.02, 0.001, 0.85)
    assert skewed > balanced


def test_curve_is_monotone_and_never_worse_with_variance_reduction():
    totals = power.curve_grid(20_000, 10_000_000, 25)
    table = power.mde_curve(0.02, 0.85, totals, variance_ratio=0.6)
    assert table["mde_absolute"].is_monotonic_decreasing
    assert (table["mde_absolute_cuped"] < table["mde_absolute"]).all()
    assert (table["mde_relative"] > 0).all()


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        power.minimum_detectable_effect(0.0, 100, 100)
    with pytest.raises(ValueError):
        power.minimum_detectable_effect(0.02, 0, 100)
    with pytest.raises(ValueError):
        power.required_rows(0.02, 0.0)
