import math

import numpy as np
import pandas as pd

from expengine import config
from expengine.inference import itt


def rho_for_horizon(alpha: float, horizon: int) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must sit strictly between 0 and 1")
    term = -2.0 * math.log(alpha) + math.log(-2.0 * math.log(alpha) + 1.0)
    return math.sqrt(term / horizon)


def confidence_sequence_multiplier(rows, alpha: float, rho: float):
    scaled = np.asarray(rows, dtype=np.float64) * rho**2
    inner = scaled + 1.0
    return np.sqrt((2.0 * inner / scaled) * np.log(np.sqrt(inner) / alpha))


def confidence_sequence_radius(standard_error, rows, alpha: float, rho: float):
    return standard_error * confidence_sequence_multiplier(rows, alpha, rho)


def peek_schedule(arm_size: int, peeks: int) -> np.ndarray:
    if peeks < 1:
        raise ValueError("need at least one peek")
    schedule = np.unique(np.linspace(arm_size / peeks, arm_size, peeks).round().astype(np.int64))
    return schedule[schedule > 1]


def simulate_peeking(
    base_rate: float,
    arm_size: int = config.AA_ARM_SIZE,
    peeks: int = config.AA_PEEKS,
    simulations: int = config.AA_SIMULATIONS,
    alpha: float = config.AA_ALPHA,
    seed: int = config.RANDOM_SEED,
    true_effect: float = 0.0,
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    schedule = peek_schedule(arm_size, peeks)
    increments = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), schedule)))
    treated = generator.binomial(
        increments, base_rate + true_effect, size=(simulations, schedule.size)
    ).cumsum(axis=1)
    control = generator.binomial(increments, base_rate, size=(simulations, schedule.size)).cumsum(
        axis=1
    )
    rows = schedule.astype(np.float64)
    rate_treated = treated / rows
    rate_control = control / rows
    standard_error = np.sqrt(
        rate_treated * (1.0 - rate_treated) / rows + rate_control * (1.0 - rate_control) / rows
    )
    difference = rate_treated - rate_control
    usable = standard_error > 0.0
    critical = itt.critical_value(1.0 - alpha)
    rho = rho_for_horizon(alpha, 2 * arm_size)
    radius = confidence_sequence_radius(standard_error, 2.0 * rows, alpha, rho)
    fixed = np.zeros_like(difference, dtype=bool)
    sequential = np.zeros_like(difference, dtype=bool)
    fixed[usable] = np.abs(difference[usable]) > critical * standard_error[usable]
    sequential[usable] = np.abs(difference[usable]) > radius[usable]
    return pd.DataFrame(
        {
            "peek": np.arange(1, schedule.size + 1),
            "rows_per_arm": schedule,
            "fixed_horizon_single_look_rate": fixed.mean(axis=0),
            "fixed_horizon_any_look_rate": np.maximum.accumulate(fixed, axis=1).mean(axis=0),
            "sequential_any_look_rate": np.maximum.accumulate(sequential, axis=1).mean(axis=0),
            "nominal_alpha": alpha,
        }
    )


def arrival_order(rows: int, seed: int = config.RANDOM_SEED) -> np.ndarray:
    if rows < 1:
        raise ValueError("need at least one row to place in an arrival order")
    return np.random.default_rng(seed).permutation(rows)


def monitor(
    outcome,
    treatment,
    checkpoints,
    alpha: float = config.AA_ALPHA,
    rho: float | None = None,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    outcome_values = np.asarray(outcome)
    treated_flags = np.asarray(treatment) == 1
    total_rows = outcome_values.size
    if rho is None:
        rho = rho_for_horizon(alpha, total_rows)
    critical = itt.critical_value(confidence_level)
    rows_treated = 0
    rows_control = 0
    successes_treated = 0.0
    successes_control = 0.0
    previous = 0
    records = []
    for checkpoint in checkpoints:
        point = int(checkpoint)
        if point <= previous or point > total_rows:
            continue
        segment_treated = treated_flags[previous:point]
        segment_outcome = outcome_values[previous:point]
        rows_treated += int(segment_treated.sum(dtype=np.int64))
        rows_control += int((~segment_treated).sum(dtype=np.int64))
        successes_treated += float(segment_outcome[segment_treated].sum(dtype=np.float64))
        successes_control += float(segment_outcome[~segment_treated].sum(dtype=np.float64))
        previous = point
        if rows_treated < 2 or rows_control < 2:
            continue
        rate_treated = successes_treated / rows_treated
        rate_control = successes_control / rows_control
        standard_error = math.sqrt(
            rate_treated * (1.0 - rate_treated) / rows_treated
            + rate_control * (1.0 - rate_control) / rows_control
        )
        if standard_error <= 0.0:
            continue
        difference = rate_treated - rate_control
        radius = float(confidence_sequence_radius(standard_error, point, alpha, rho))
        records.append(
            {
                "rows_seen": point,
                "rows_treated": rows_treated,
                "rows_control": rows_control,
                "absolute_effect": difference,
                "standard_error": standard_error,
                "fixed_low": difference - critical * standard_error,
                "fixed_high": difference + critical * standard_error,
                "sequential_low": difference - radius,
                "sequential_high": difference + radius,
                "sequential_excludes_zero": bool(abs(difference) > radius),
                "fixed_excludes_zero": bool(abs(difference) > critical * standard_error),
            }
        )
    return pd.DataFrame.from_records(records)


def first_crossing(track: pd.DataFrame, column: str) -> int:
    crossed = track.loc[track[column], "rows_seen"]
    if crossed.empty:
        return -1
    return int(crossed.iloc[0])
