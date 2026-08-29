import math

import numpy as np
import pandas as pd
from scipy import stats

from expengine import config


def z_values(alpha: float, power: float) -> tuple[float, float]:
    return float(stats.norm.ppf(1.0 - alpha / 2.0)), float(stats.norm.ppf(power))


def minimum_detectable_effect(
    baseline_rate: float,
    rows_treated: int,
    rows_control: int,
    power: float = config.TARGET_POWER,
    alpha: float = config.AA_ALPHA,
    variance_ratio: float = 1.0,
) -> float:
    if rows_treated < 1 or rows_control < 1:
        raise ValueError("each arm needs at least one row")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline rate must sit strictly between 0 and 1")
    z_alpha, z_power = z_values(alpha, power)
    variance = baseline_rate * (1.0 - baseline_rate) * variance_ratio
    return float(
        (z_alpha + z_power) * math.sqrt(variance * (1.0 / rows_treated + 1.0 / rows_control))
    )


def required_rows(
    baseline_rate: float,
    absolute_effect: float,
    treatment_share: float = config.INTENDED_TREATMENT_SHARE,
    power: float = config.TARGET_POWER,
    alpha: float = config.AA_ALPHA,
    variance_ratio: float = 1.0,
) -> int:
    if absolute_effect <= 0.0:
        raise ValueError("absolute effect must be positive")
    z_alpha, z_power = z_values(alpha, power)
    variance = baseline_rate * (1.0 - baseline_rate) * variance_ratio
    allocation = 1.0 / treatment_share + 1.0 / (1.0 - treatment_share)
    total = variance * (z_alpha + z_power) ** 2 * allocation / absolute_effect**2
    return int(math.ceil(total))


def curve_grid(minimum_total: int, maximum_total: int, points: int) -> np.ndarray:
    return np.unique(np.geomspace(minimum_total, maximum_total, points).round().astype(np.int64))


def mde_curve(
    baseline_rate: float,
    treatment_share: float,
    totals,
    power: float = config.TARGET_POWER,
    alpha: float = config.AA_ALPHA,
    variance_ratio: float = 1.0,
) -> pd.DataFrame:
    records = []
    for total in totals:
        rows_treated = int(round(int(total) * treatment_share))
        rows_control = int(total) - rows_treated
        if rows_treated < 1 or rows_control < 1:
            continue
        base = minimum_detectable_effect(
            baseline_rate, rows_treated, rows_control, power, alpha, 1.0
        )
        adjusted = minimum_detectable_effect(
            baseline_rate, rows_treated, rows_control, power, alpha, variance_ratio
        )
        records.append(
            {
                "total_rows": int(total),
                "rows_treated": rows_treated,
                "rows_control": rows_control,
                "mde_absolute": base,
                "mde_relative": base / baseline_rate,
                "mde_absolute_cuped": adjusted,
                "mde_relative_cuped": adjusted / baseline_rate,
            }
        )
    return pd.DataFrame.from_records(records)
