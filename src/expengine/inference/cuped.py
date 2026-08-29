import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from expengine import config
from expengine.inference import itt

MOMENT_KEYS = (
    "rows",
    "sum_outcome",
    "sum_outcome_square",
    "sum_covariate",
    "sum_covariate_square",
    "sum_cross",
)


def empty_moments() -> dict[str, float]:
    return dict.fromkeys(MOMENT_KEYS, 0.0)


def update_moments(moments: dict[str, float], outcome, covariate) -> dict[str, float]:
    outcome_values = np.asarray(outcome, dtype=np.float64)
    covariate_values = np.asarray(covariate, dtype=np.float64)
    if outcome_values.size != covariate_values.size:
        raise ValueError("outcome and covariate must have the same length")
    moments["rows"] += float(outcome_values.size)
    moments["sum_outcome"] += float(outcome_values.sum())
    moments["sum_outcome_square"] += float(np.dot(outcome_values, outcome_values))
    moments["sum_covariate"] += float(covariate_values.sum())
    moments["sum_covariate_square"] += float(np.dot(covariate_values, covariate_values))
    moments["sum_cross"] += float(np.dot(outcome_values, covariate_values))
    return moments


def combine_moments(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    return {key: first[key] + second[key] for key in MOMENT_KEYS}


def moment_statistics(moments: dict[str, float]) -> dict[str, float]:
    rows = moments["rows"]
    if rows < 2:
        raise ValueError("moments need at least two rows")
    mean_outcome = moments["sum_outcome"] / rows
    mean_covariate = moments["sum_covariate"] / rows
    variance_outcome = (moments["sum_outcome_square"] - rows * mean_outcome**2) / (rows - 1.0)
    variance_covariate = (moments["sum_covariate_square"] - rows * mean_covariate**2) / (rows - 1.0)
    covariance = (moments["sum_cross"] - rows * mean_outcome * mean_covariate) / (rows - 1.0)
    return {
        "rows": rows,
        "mean_outcome": mean_outcome,
        "mean_covariate": mean_covariate,
        "variance_outcome": max(variance_outcome, 0.0),
        "variance_covariate": max(variance_covariate, 0.0),
        "covariance": covariance,
    }


def theta_from_moments(moments: dict[str, float]) -> float:
    statistics = moment_statistics(moments)
    if statistics["variance_covariate"] <= 0.0:
        return 0.0
    return float(statistics["covariance"] / statistics["variance_covariate"])


def cuped_from_moments(
    treated_moments: dict[str, float],
    control_moments: dict[str, float],
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> dict[str, float]:
    pooled = combine_moments(treated_moments, control_moments)
    pooled_statistics = moment_statistics(pooled)
    theta = theta_from_moments(pooled)
    treated_statistics = moment_statistics(treated_moments)
    control_statistics = moment_statistics(control_moments)
    adjusted = {}
    for label, statistics in (("treated", treated_statistics), ("control", control_statistics)):
        adjusted[label] = {
            "mean": statistics["mean_outcome"]
            - theta * (statistics["mean_covariate"] - pooled_statistics["mean_covariate"]),
            "variance": max(
                statistics["variance_outcome"]
                + theta**2 * statistics["variance_covariate"]
                - 2.0 * theta * statistics["covariance"],
                0.0,
            ),
        }
    rows_treated = int(treated_statistics["rows"])
    rows_control = int(control_statistics["rows"])
    raw = itt.difference_in_means(
        treated_statistics["mean_outcome"],
        treated_statistics["variance_outcome"],
        rows_treated,
        control_statistics["mean_outcome"],
        control_statistics["variance_outcome"],
        rows_control,
        confidence_level,
    )
    adjusted_effect = itt.difference_in_means(
        adjusted["treated"]["mean"],
        adjusted["treated"]["variance"],
        rows_treated,
        adjusted["control"]["mean"],
        adjusted["control"]["variance"],
        rows_control,
        confidence_level,
    )
    denominator = math.sqrt(
        pooled_statistics["variance_outcome"] * pooled_statistics["variance_covariate"]
    )
    correlation = pooled_statistics["covariance"] / denominator if denominator > 0.0 else math.nan
    if raw["standard_error"] > 0.0:
        variance_ratio = (adjusted_effect["standard_error"] / raw["standard_error"]) ** 2
    else:
        variance_ratio = math.nan
    return {
        "rows_treated": rows_treated,
        "rows_control": rows_control,
        "theta": theta,
        "covariate_correlation": float(correlation),
        "raw_absolute_effect": raw["absolute_effect"],
        "raw_standard_error": raw["standard_error"],
        "raw_confidence_low": raw["confidence_low"],
        "raw_confidence_high": raw["confidence_high"],
        "adjusted_absolute_effect": adjusted_effect["absolute_effect"],
        "adjusted_standard_error": adjusted_effect["standard_error"],
        "adjusted_confidence_low": adjusted_effect["confidence_low"],
        "adjusted_confidence_high": adjusted_effect["confidence_high"],
        "adjusted_p_value": adjusted_effect["p_value"],
        "variance_ratio": float(variance_ratio),
        "variance_reduction": float(1.0 - variance_ratio),
        "effective_sample_multiplier": float(1.0 / variance_ratio),
    }


def cuped_effect(
    outcome, covariate, treatment, confidence_level: float = config.CONFIDENCE_LEVEL
) -> dict[str, float]:
    outcome_values = np.asarray(outcome)
    covariate_values = np.asarray(covariate)
    treated = np.asarray(treatment) == 1
    treated_moments = update_moments(
        empty_moments(), outcome_values[treated], covariate_values[treated]
    )
    control_moments = update_moments(
        empty_moments(), outcome_values[~treated], covariate_values[~treated]
    )
    return cuped_from_moments(treated_moments, control_moments, confidence_level)


def build_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.CUPED_MAX_ITER,
        learning_rate=config.CUPED_LEARNING_RATE,
        max_leaf_nodes=config.CUPED_MAX_LEAF_NODES,
        early_stopping=False,
        random_state=seed,
    )


def fold_assignment(rows: int, folds: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, folds, size=rows).astype(np.int8)


def fit_control_models(
    features,
    outcome,
    assignment,
    folds: int = config.CUPED_FOLDS,
    seed: int = config.RANDOM_SEED,
    max_rows: int = config.CUPED_FIT_MAX_ROWS,
) -> list[HistGradientBoostingRegressor]:
    generator = np.random.default_rng(seed)
    models = []
    for fold in range(folds):
        indices = np.nonzero(assignment != fold)[0]
        if indices.size > max_rows:
            indices = generator.choice(indices, size=max_rows, replace=False)
        model = build_regressor(seed + fold)
        model.fit(features[indices], outcome[indices])
        models.append(model)
    return models


def out_of_fold_predictions(models, features, assignment) -> np.ndarray:
    predictions = np.zeros(features.shape[0], dtype=np.float64)
    for fold, model in enumerate(models):
        selected = assignment == fold
        if selected.any():
            predictions[selected] = model.predict(features[selected])
    return predictions


def ensemble_predictions(models, features) -> np.ndarray:
    total = np.zeros(features.shape[0], dtype=np.float64)
    for model in models:
        total += model.predict(features)
    return total / float(len(models))
