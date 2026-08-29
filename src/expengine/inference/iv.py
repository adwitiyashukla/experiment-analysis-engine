import math

import numpy as np

from expengine import config
from expengine.inference import itt


def cace_from_counts(
    rows_treated: int,
    rows_control: int,
    outcome_treated: float,
    outcome_control: float,
    exposure_treated: float,
    exposure_control: float,
    outcome_and_exposure_treated: float,
    outcome_and_exposure_control: float,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> dict[str, float]:
    if rows_treated < 2 or rows_control < 2:
        raise ValueError("each arm needs at least two rows")
    rate_outcome_treated = outcome_treated / rows_treated
    rate_outcome_control = outcome_control / rows_control
    rate_exposure_treated = exposure_treated / rows_treated
    rate_exposure_control = exposure_control / rows_control
    itt_outcome = rate_outcome_treated - rate_outcome_control
    itt_exposure = rate_exposure_treated - rate_exposure_control
    if itt_exposure <= 0.0:
        raise ValueError("assignment does not shift exposure so the Wald estimator is undefined")
    variance_outcome = (
        rate_outcome_treated * (1.0 - rate_outcome_treated) / rows_treated
        + rate_outcome_control * (1.0 - rate_outcome_control) / rows_control
    )
    variance_exposure = (
        rate_exposure_treated * (1.0 - rate_exposure_treated) / rows_treated
        + rate_exposure_control * (1.0 - rate_exposure_control) / rows_control
    )
    covariance = (
        outcome_and_exposure_treated / rows_treated - rate_outcome_treated * rate_exposure_treated
    ) / rows_treated + (
        outcome_and_exposure_control / rows_control - rate_outcome_control * rate_exposure_control
    ) / rows_control
    cace = itt_outcome / itt_exposure
    variance_cace = (
        variance_outcome / itt_exposure**2
        + itt_outcome**2 * variance_exposure / itt_exposure**4
        - 2.0 * itt_outcome * covariance / itt_exposure**3
    )
    standard_error = math.sqrt(max(variance_cace, 0.0))
    critical = itt.critical_value(confidence_level)
    return {
        "rows_treated": int(rows_treated),
        "rows_control": int(rows_control),
        "itt_outcome": float(itt_outcome),
        "itt_outcome_standard_error": float(math.sqrt(variance_outcome)),
        "itt_exposure": float(itt_exposure),
        "itt_exposure_standard_error": float(math.sqrt(variance_exposure)),
        "compliance_rate": float(rate_exposure_treated),
        "first_stage_f": float(itt_exposure**2 / variance_exposure),
        "cace": float(cace),
        "cace_standard_error": standard_error,
        "cace_confidence_low": float(cace - critical * standard_error),
        "cace_confidence_high": float(cace + critical * standard_error),
        "cace_over_itt": float(cace / itt_outcome) if itt_outcome != 0.0 else math.nan,
    }


def cace_from_arrays(
    outcome, exposure, treatment, confidence_level: float = config.CONFIDENCE_LEVEL
) -> dict[str, float]:
    outcome_values = np.asarray(outcome)
    exposure_values = np.asarray(exposure)
    treated = np.asarray(treatment) == 1
    control = ~treated
    return cace_from_counts(
        int(treated.sum(dtype=np.int64)),
        int(control.sum(dtype=np.int64)),
        float(outcome_values[treated].sum(dtype=np.float64)),
        float(outcome_values[control].sum(dtype=np.float64)),
        float(exposure_values[treated].sum(dtype=np.float64)),
        float(exposure_values[control].sum(dtype=np.float64)),
        float((outcome_values[treated] * exposure_values[treated]).sum(dtype=np.float64)),
        float((outcome_values[control] * exposure_values[control]).sum(dtype=np.float64)),
        confidence_level,
    )


def naive_exposed_comparison(
    outcome_and_exposure_treated: float,
    exposure_treated: float,
    outcome_control: float,
    rows_control: int,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> dict[str, float]:
    return itt.two_proportion_effect(
        outcome_and_exposure_treated,
        int(exposure_treated),
        outcome_control,
        rows_control,
        confidence_level,
    )


def incrementality(
    effect: dict[str, float], outcome_and_exposure_treated: float
) -> dict[str, float]:
    rows_treated = float(effect["rows_treated"])
    counterfactual = rows_treated * effect["mean_control"]
    incremental = effect["absolute_effect"] * rows_treated
    if incremental <= 0.0:
        raise ValueError("no incremental conversions to attribute")
    return {
        "conversions_treated_arm": float(effect["successes_treated"]),
        "counterfactual_conversions": float(counterfactual),
        "incremental_conversions": float(incremental),
        "incremental_low": float(effect["confidence_low"] * rows_treated),
        "incremental_high": float(effect["confidence_high"] * rows_treated),
        "conversions_among_exposed": float(outcome_and_exposure_treated),
        "overstatement_treated_arm": float(effect["successes_treated"] / incremental),
        "overstatement_exposed_only": float(outcome_and_exposure_treated / incremental),
    }
