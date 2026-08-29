import numpy as np
import pandas as pd
from scipy import stats

from expengine import config


def standardised_mean_difference(
    mean_treated: float,
    variance_treated: float,
    mean_control: float,
    variance_control: float,
) -> float:
    pooled = np.sqrt((variance_treated + variance_control) / 2.0)
    if pooled <= 0.0:
        return 0.0
    return float((mean_treated - mean_control) / pooled)


def feature_balance(name: str, values, treatment) -> dict[str, float]:
    values = np.asarray(values)
    treated = np.asarray(treatment) == 1
    treated_values = values[treated]
    control_values = values[~treated]
    if treated_values.size < 2 or control_values.size < 2:
        raise ValueError(f"feature {name} needs at least two rows in each arm")
    mean_treated = float(treated_values.mean(dtype=np.float64))
    mean_control = float(control_values.mean(dtype=np.float64))
    variance_treated = float(treated_values.var(ddof=1, dtype=np.float64))
    variance_control = float(control_values.var(ddof=1, dtype=np.float64))
    standard_error = np.sqrt(
        variance_treated / treated_values.size + variance_control / control_values.size
    )
    if standard_error > 0.0:
        z_statistic = (mean_treated - mean_control) / standard_error
        p_value = float(2.0 * stats.norm.sf(abs(z_statistic)))
    else:
        z_statistic = 0.0
        p_value = 1.0
    return {
        "feature": name,
        "rows_treated": int(treated_values.size),
        "rows_control": int(control_values.size),
        "mean_treated": mean_treated,
        "mean_control": mean_control,
        "standard_deviation_treated": float(np.sqrt(variance_treated)),
        "standard_deviation_control": float(np.sqrt(variance_control)),
        "standardised_mean_difference": standardised_mean_difference(
            mean_treated, variance_treated, mean_control, variance_control
        ),
        "z_statistic": float(z_statistic),
        "p_value": p_value,
    }


def balance_summary(
    table: pd.DataFrame,
    threshold: float = config.BALANCE_SMD_THRESHOLD,
    z_threshold: float = config.BALANCE_Z_THRESHOLD,
) -> dict[str, float]:
    absolute = table["standardised_mean_difference"].abs()
    absolute_z = table["z_statistic"].abs()
    return {
        "features": int(table.shape[0]),
        "max_absolute_smd": float(absolute.max()),
        "mean_absolute_smd": float(absolute.mean()),
        "threshold": float(threshold),
        "features_above_threshold": int((absolute > threshold).sum()),
        "passes": bool((absolute <= threshold).all()),
        "z_threshold": float(z_threshold),
        "max_absolute_z": float(absolute_z.max()),
        "features_beyond_chance": int((absolute_z > z_threshold).sum()),
    }
