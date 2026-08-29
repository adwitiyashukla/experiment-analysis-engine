import numpy as np

from expengine import config
from expengine.inference import cuped


def test_theta_recovers_the_regression_slope():
    stream = np.random.default_rng(11)
    rows = 200_000
    covariate = stream.normal(size=rows)
    treatment = (stream.random(rows) < 0.5).astype(np.int8)
    outcome = 3.0 * covariate + 0.5 * treatment + stream.normal(size=rows)
    result = cuped.cuped_effect(outcome, covariate, treatment)
    assert abs(result["theta"] - 3.0) < 0.05
    assert result["variance_reduction"] > 0.8
    assert result["effective_sample_multiplier"] > 5.0
    assert result["adjusted_standard_error"] < result["raw_standard_error"]
    assert abs(result["adjusted_absolute_effect"] - 0.5) < 3.0 * result["adjusted_standard_error"]


def test_binary_outcome_keeps_the_effect_and_loses_variance():
    stream = np.random.default_rng(12)
    rows = 400_000
    replications = 4
    effects = []
    standard_errors = []
    reductions = []
    gaps = []
    for _ in range(replications):
        baseline = 0.05 + 0.45 * stream.random(rows)
        treatment = (stream.random(rows) < 0.5).astype(np.int8)
        outcome = (stream.random(rows) < baseline + 0.02 * treatment).astype(np.int8)
        result = cuped.cuped_effect(outcome, baseline, treatment)
        effects.append(result["adjusted_absolute_effect"])
        standard_errors.append(result["adjusted_standard_error"])
        reductions.append(result["variance_reduction"])
        gaps.append(abs(result["raw_absolute_effect"] - result["adjusted_absolute_effect"]))
    standard_error_of_mean = float(np.mean(standard_errors)) / np.sqrt(replications)
    assert min(reductions) > 0.05
    assert abs(float(np.mean(effects)) - 0.02) < 3.0 * standard_error_of_mean
    assert max(gaps) < 0.002


def test_a_useless_covariate_changes_nothing():
    stream = np.random.default_rng(13)
    rows = 100_000
    covariate = stream.normal(size=rows)
    treatment = (stream.random(rows) < 0.5).astype(np.int8)
    outcome = (stream.random(rows) < 0.1 + 0.01 * treatment).astype(np.int8)
    result = cuped.cuped_effect(outcome, covariate, treatment)
    assert abs(result["variance_reduction"]) < 0.01
    assert abs(result["theta"]) < 0.01


def test_moments_combine_the_same_way_as_a_single_pass():
    stream = np.random.default_rng(14)
    outcome = stream.normal(size=5_000)
    covariate = stream.normal(size=5_000)
    whole = cuped.update_moments(cuped.empty_moments(), outcome, covariate)
    first = cuped.update_moments(cuped.empty_moments(), outcome[:2_000], covariate[:2_000])
    second = cuped.update_moments(cuped.empty_moments(), outcome[2_000:], covariate[2_000:])
    combined = cuped.combine_moments(first, second)
    for key in cuped.MOMENT_KEYS:
        assert abs(whole[key] - combined[key]) < 1e-6


def test_cross_fitted_control_models_do_not_absorb_the_treatment_effect(monkeypatch):
    monkeypatch.setattr(config, "CUPED_MAX_ITER", 40)
    stream = np.random.default_rng(15)
    rows = 60_000
    features = stream.normal(size=(rows, 3)).astype(np.float32)
    treatment = (stream.random(rows) < 0.5).astype(np.int8)
    signal = 0.15 + 0.04 * features[:, 0] + 0.03 * features[:, 1]
    probability = np.clip(signal + 0.02 * treatment, 0.001, 0.999)
    outcome = (stream.random(rows) < probability).astype(np.int8)
    control = treatment == 0
    control_features = features[control]
    assignment = cuped.fold_assignment(
        int(control.sum(dtype=np.int64)), config.CUPED_FOLDS, config.RANDOM_SEED
    )
    models = cuped.fit_control_models(
        control_features, outcome[control].astype(np.float32), assignment
    )
    covariate = cuped.ensemble_predictions(models, features)
    covariate[control] = cuped.out_of_fold_predictions(models, control_features, assignment)
    result = cuped.cuped_effect(outcome, covariate, treatment)
    assert abs(result["adjusted_absolute_effect"] - 0.02) < 3.0 * result["adjusted_standard_error"]
    assert result["variance_reduction"] > 0.0
