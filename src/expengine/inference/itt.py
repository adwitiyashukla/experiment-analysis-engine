import math

import numpy as np
from scipy import stats

from expengine import config


def critical_value(confidence_level: float = config.CONFIDENCE_LEVEL) -> float:
    return float(stats.norm.ppf(0.5 + confidence_level / 2.0))


def difference_in_means(
    mean_treated: float,
    variance_treated: float,
    rows_treated: int,
    mean_control: float,
    variance_control: float,
    rows_control: int,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> dict[str, float]:
    if rows_treated < 2 or rows_control < 2:
        raise ValueError("each arm needs at least two rows")
    absolute = mean_treated - mean_control
    standard_error = math.sqrt(variance_treated / rows_treated + variance_control / rows_control)
    critical = critical_value(confidence_level)
    if standard_error > 0.0:
        z_statistic = absolute / standard_error
        p_value = float(2.0 * stats.norm.sf(abs(z_statistic)))
    else:
        z_statistic = math.nan
        p_value = math.nan
    return {
        "rows_treated": int(rows_treated),
        "rows_control": int(rows_control),
        "mean_treated": float(mean_treated),
        "mean_control": float(mean_control),
        "absolute_effect": float(absolute),
        "standard_error": float(standard_error),
        "confidence_low": float(absolute - critical * standard_error),
        "confidence_high": float(absolute + critical * standard_error),
        "z_statistic": float(z_statistic),
        "p_value": p_value,
    }


def relative_lift(
    mean_treated: float,
    variance_treated: float,
    rows_treated: int,
    mean_control: float,
    variance_control: float,
    rows_control: int,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> dict[str, float]:
    if mean_treated <= 0.0 or mean_control <= 0.0:
        return {
            "relative_lift": math.nan,
            "relative_low": math.nan,
            "relative_high": math.nan,
            "log_ratio_standard_error": math.nan,
        }
    ratio = mean_treated / mean_control
    log_standard_error = math.sqrt(
        variance_treated / (rows_treated * mean_treated**2)
        + variance_control / (rows_control * mean_control**2)
    )
    critical = critical_value(confidence_level)
    return {
        "relative_lift": float(ratio - 1.0),
        "relative_low": float(math.exp(math.log(ratio) - critical * log_standard_error) - 1.0),
        "relative_high": float(math.exp(math.log(ratio) + critical * log_standard_error) - 1.0),
        "log_ratio_standard_error": float(log_standard_error),
    }


def two_proportion_effect(
    successes_treated: float,
    rows_treated: int,
    successes_control: float,
    rows_control: int,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> dict[str, float]:
    rate_treated = successes_treated / rows_treated
    rate_control = successes_control / rows_control
    variance_treated = rate_treated * (1.0 - rate_treated)
    variance_control = rate_control * (1.0 - rate_control)
    result = difference_in_means(
        rate_treated,
        variance_treated,
        rows_treated,
        rate_control,
        variance_control,
        rows_control,
        confidence_level,
    )
    result.update(
        relative_lift(
            rate_treated,
            variance_treated,
            rows_treated,
            rate_control,
            variance_control,
            rows_control,
            confidence_level,
        )
    )
    result["successes_treated"] = float(successes_treated)
    result["successes_control"] = float(successes_control)
    return result


def binary_effect_from_arrays(
    outcome, treatment, confidence_level: float = config.CONFIDENCE_LEVEL
) -> dict[str, float]:
    outcome = np.asarray(outcome)
    treated = np.asarray(treatment) == 1
    return two_proportion_effect(
        float(outcome[treated].sum(dtype=np.float64)),
        int(treated.sum(dtype=np.int64)),
        float(outcome[~treated].sum(dtype=np.float64)),
        int((~treated).sum(dtype=np.int64)),
        confidence_level,
    )
