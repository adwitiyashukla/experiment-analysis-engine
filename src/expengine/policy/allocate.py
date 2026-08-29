import math

import numpy as np
import pandas as pd


def even_allocation(capacity, budget: float) -> np.ndarray:
    capacity_values = np.asarray(capacity, dtype=np.float64)
    count = capacity_values.size
    allocation = np.zeros(count, dtype=np.float64)
    remaining = min(float(budget), float(capacity_values.sum()))
    active = capacity_values > 0.0
    for _ in range(count + 1):
        if remaining <= 1e-9 or not active.any():
            break
        share = remaining / float(active.sum())
        take = np.where(active, np.minimum(capacity_values - allocation, share), 0.0)
        allocation += take
        remaining -= float(take.sum())
        active = active & ((capacity_values - allocation) > 1e-9)
    return allocation


def greedy_allocation(value_per_impression, capacity, budget: float) -> np.ndarray:
    values = np.asarray(value_per_impression, dtype=np.float64)
    capacity_values = np.asarray(capacity, dtype=np.float64)
    allocation = np.zeros(values.size, dtype=np.float64)
    remaining = float(budget)
    for index in np.argsort(-values):
        if remaining <= 0.0 or values[index] <= 0.0:
            break
        take = min(float(capacity_values[index]), remaining)
        allocation[index] = take
        remaining -= take
    return allocation


def allocation_value(allocation, value_per_impression) -> float:
    return float(
        np.dot(
            np.asarray(allocation, dtype=np.float64),
            np.asarray(value_per_impression, dtype=np.float64),
        )
    )


def compare_allocations(value_per_impression, capacity, budget: float) -> dict[str, float]:
    even = even_allocation(capacity, budget)
    greedy = greedy_allocation(value_per_impression, capacity, budget)
    even_value = allocation_value(even, value_per_impression)
    greedy_value = allocation_value(greedy, value_per_impression)
    return {
        "budget": float(budget),
        "even_impressions": float(even.sum()),
        "greedy_impressions": float(greedy.sum()),
        "even_incremental_conversions": even_value,
        "greedy_incremental_conversions": greedy_value,
        "gain": greedy_value - even_value,
        "gain_multiple": greedy_value / even_value if even_value > 0.0 else math.nan,
    }


def allocation_table(labels, value_per_impression, capacity, budget: float) -> pd.DataFrame:
    even = even_allocation(capacity, budget)
    greedy = greedy_allocation(value_per_impression, capacity, budget)
    values = np.asarray(value_per_impression, dtype=np.float64)
    return pd.DataFrame(
        {
            "segment": list(labels),
            "value_per_impression": values,
            "capacity_impressions": np.asarray(capacity, dtype=np.float64),
            "even_impressions": even,
            "greedy_impressions": greedy,
            "even_incremental_conversions": even * values,
            "greedy_incremental_conversions": greedy * values,
        }
    )
