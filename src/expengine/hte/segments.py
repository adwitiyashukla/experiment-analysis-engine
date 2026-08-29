import numpy as np
import pandas as pd

from expengine import config
from expengine.inference import itt


def benjamini_hochberg(p_values, alpha: float = config.BENJAMINI_HOCHBERG_ALPHA) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    count = values.size
    if count == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(values)
    thresholds = alpha * np.arange(1, count + 1) / count
    passing = np.nonzero(values[order] <= thresholds)[0]
    rejected = np.zeros(count, dtype=bool)
    if passing.size > 0:
        rejected[order[: passing.max() + 1]] = True
    return rejected


def benjamini_hochberg_adjusted(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    count = values.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values)
    scaled = values[order] * count / np.arange(1, count + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    return adjusted


def quantile_edges(values, quantiles: int = config.SEGMENT_QUANTILES) -> np.ndarray:
    if quantiles < 2:
        raise ValueError("need at least two quantiles")
    probabilities = np.linspace(0.0, 1.0, quantiles + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(values), probabilities))


def assign_bins(values, edges) -> np.ndarray:
    return np.searchsorted(
        np.asarray(edges, dtype=np.float64), np.asarray(values), side="right"
    ).astype(np.int16)


def segment_effects(
    feature_name: str,
    bins,
    outcome,
    treatment,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> list[dict[str, float]]:
    bin_values = np.asarray(bins)
    outcome_values = np.asarray(outcome)
    treated = np.asarray(treatment) == 1
    records = []
    for bin_index in np.unique(bin_values):
        selected = bin_values == bin_index
        selected_treated = selected & treated
        selected_control = selected & ~treated
        rows_treated = int(selected_treated.sum(dtype=np.int64))
        rows_control = int(selected_control.sum(dtype=np.int64))
        if rows_treated < 2 or rows_control < 2:
            continue
        effect = itt.two_proportion_effect(
            float(outcome_values[selected_treated].sum(dtype=np.float64)),
            rows_treated,
            float(outcome_values[selected_control].sum(dtype=np.float64)),
            rows_control,
            confidence_level,
        )
        effect["feature"] = feature_name
        effect["segment"] = int(bin_index)
        effect["segment_label"] = f"{feature_name} q{int(bin_index) + 1}"
        records.append(effect)
    return records


def apply_multiple_testing(
    table: pd.DataFrame, alpha: float = config.BENJAMINI_HOCHBERG_ALPHA
) -> pd.DataFrame:
    result = table.copy()
    p_values = result["p_value"].to_numpy(dtype=np.float64)
    result["significant_uncorrected"] = p_values < alpha
    result["benjamini_hochberg_q"] = benjamini_hochberg_adjusted(p_values)
    result["significant_corrected"] = benjamini_hochberg(p_values, alpha)
    return result


def multiple_testing_summary(
    table: pd.DataFrame, alpha: float = config.BENJAMINI_HOCHBERG_ALPHA
) -> dict[str, float]:
    uncorrected = int(table["significant_uncorrected"].sum())
    corrected = int(table["significant_corrected"].sum())
    return {
        "segments": int(table.shape[0]),
        "alpha": float(alpha),
        "significant_uncorrected": uncorrected,
        "significant_corrected": corrected,
        "dropped_by_correction": uncorrected - corrected,
    }
