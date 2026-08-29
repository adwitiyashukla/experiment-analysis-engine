import numpy as np
import pandas as pd

from expengine.hte import segments

TRUE_SEGMENT_EFFECT = 0.02


def test_correction_never_rejects_more_than_the_uncorrected_rule():
    stream = np.random.default_rng(1)
    p_values = np.concatenate([stream.uniform(size=90), stream.uniform(0.0, 0.0005, size=10)])
    uncorrected = int((p_values < 0.05).sum())
    corrected = int(segments.benjamini_hochberg(p_values, 0.05).sum())
    assert corrected <= uncorrected
    assert corrected >= 10


def test_correction_finds_almost_nothing_under_a_pure_null():
    stream = np.random.default_rng(2)
    discoveries = 0
    for _ in range(200):
        discoveries += int(segments.benjamini_hochberg(stream.uniform(size=50), 0.05).sum())
    assert discoveries / 200.0 < 0.5


def test_adjusted_values_are_monotone_and_never_smaller_than_the_raw_values():
    p_values = np.array([0.001, 0.01, 0.02, 0.04, 0.9])
    adjusted = segments.benjamini_hochberg_adjusted(p_values)
    assert np.all(np.diff(adjusted) >= -1e-12)
    assert np.all(adjusted >= p_values - 1e-12)
    assert np.all(adjusted <= 1.0)


def test_empty_input_is_handled():
    assert segments.benjamini_hochberg(np.array([])).size == 0
    assert segments.benjamini_hochberg_adjusted(np.array([])).size == 0


def test_quantile_bins_split_the_sample_evenly(generator):
    values = generator.normal(size=100_000)
    bins = segments.assign_bins(values, segments.quantile_edges(values, 4))
    counts = np.bincount(bins)
    assert counts.size == 4
    assert counts.max() - counts.min() < 100


def test_segment_effects_recover_an_effect_confined_to_one_quantile():
    stream = np.random.default_rng(7)
    rows = 400_000
    feature = stream.random(rows)
    treatment = (stream.random(rows) < 0.5).astype(np.int8)
    effect = np.where(feature > 0.75, TRUE_SEGMENT_EFFECT, 0.0)
    outcome = (stream.random(rows) < 0.05 + effect * treatment).astype(np.int8)
    bins = segments.assign_bins(feature, segments.quantile_edges(feature, 4))
    table = segments.apply_multiple_testing(
        pd.DataFrame.from_records(segments.segment_effects("f0", bins, outcome, treatment))
    )
    top = table.loc[table["segment"] == 3].iloc[0]
    assert abs(top["absolute_effect"] - TRUE_SEGMENT_EFFECT) < 3.0 * top["standard_error"]
    assert bool(top["significant_corrected"])
    assert int(table["significant_corrected"].sum()) == 1


def test_multiple_testing_summary_reports_what_the_correction_removed():
    table = pd.DataFrame({"p_value": [0.0001, 0.03, 0.04, 0.2]})
    table = segments.apply_multiple_testing(table, alpha=0.05)
    summary = segments.multiple_testing_summary(table, alpha=0.05)
    assert summary["segments"] == 4
    assert summary["significant_uncorrected"] == 3
    assert summary["significant_corrected"] <= summary["significant_uncorrected"]
    assert summary["dropped_by_correction"] >= 0
