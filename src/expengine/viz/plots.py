import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_SIZE = (8.4, 4.8)
COLOR_PRIMARY = "#2F6FED"
COLOR_SECONDARY = "#8A94A6"
COLOR_ALERT = "#D1495B"
COLOR_ACCENT = "#2A9D8F"
POINTS_PER_UNIT = 100.0


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_balance(table: pd.DataFrame, path: Path, threshold: float) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    ordered = table.sort_values("feature")
    positions = np.arange(ordered.shape[0])
    axes.barh(positions, ordered["standardised_mean_difference"], color=COLOR_PRIMARY)
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["feature"])
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.axvline(threshold, color=COLOR_ALERT, linestyle="--", linewidth=0.9)
    axes.axvline(-threshold, color=COLOR_ALERT, linestyle="--", linewidth=0.9)
    axes.set_xlabel("standardised mean difference, treated minus control")
    axes.set_title("Randomisation check on the 12 pre-treatment features")
    axes.grid(axis="x", alpha=0.3, linewidth=0.6)
    return _save(figure, path)


def plot_itt_versus_cace(records: list[dict], path: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    labels = [record["label"] for record in records]
    values = np.array([record["value"] for record in records]) * POINTS_PER_UNIT
    low = values - np.array([record["low"] for record in records]) * POINTS_PER_UNIT
    high = np.array([record["high"] for record in records]) * POINTS_PER_UNIT - values
    positions = np.arange(len(labels))
    axes.bar(positions, values, color=[COLOR_SECONDARY, COLOR_PRIMARY], width=0.55)
    axes.errorbar(positions, values, yerr=[low, high], fmt="none", ecolor="black", capsize=5)
    for position, value in zip(positions, values, strict=False):
        axes.text(position, value, f"{value:.3f} pp", ha="center", va="bottom")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylabel("effect on conversion, percentage points")
    axes.set_title("Assignment effect versus effect on users who saw an ad")
    axes.grid(axis="y", alpha=0.3, linewidth=0.6)
    axes.set_ylim(0.0, max(values + high) * 1.25)
    return _save(figure, path)


def plot_peeking(table: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.plot(
        table["peek"],
        table["fixed_horizon_any_look_rate"],
        color=COLOR_ALERT,
        marker="o",
        markersize=3.5,
        label="fixed horizon test, rejected at any peek so far",
    )
    axes.plot(
        table["peek"],
        table["sequential_any_look_rate"],
        color=COLOR_ACCENT,
        marker="s",
        markersize=3.5,
        label="confidence sequence, rejected at any peek so far",
    )
    axes.axhline(
        float(table["nominal_alpha"].iloc[0]),
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="nominal 5 percent",
    )
    axes.set_xlabel("number of times the experiment was checked")
    axes.set_ylabel("false positive rate under a true null")
    axes.set_title("Peeking at a fixed horizon test inflates false positives")
    axes.grid(alpha=0.3, linewidth=0.6)
    axes.legend(loc="upper left", frameon=False)
    return _save(figure, path)


def plot_confidence_sequence(track: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    rows = track["rows_seen"]
    axes.fill_between(
        rows,
        track["sequential_low"] * POINTS_PER_UNIT,
        track["sequential_high"] * POINTS_PER_UNIT,
        color=COLOR_ACCENT,
        alpha=0.25,
        label="always valid confidence sequence",
    )
    axes.plot(
        rows,
        track["fixed_low"] * POINTS_PER_UNIT,
        color=COLOR_SECONDARY,
        linewidth=1.0,
        linestyle="--",
        label="fixed horizon interval",
    )
    axes.plot(
        rows,
        track["fixed_high"] * POINTS_PER_UNIT,
        color=COLOR_SECONDARY,
        linewidth=1.0,
        linestyle="--",
    )
    axes.plot(
        rows,
        track["absolute_effect"] * POINTS_PER_UNIT,
        color=COLOR_PRIMARY,
        linewidth=1.6,
        label="running estimate",
    )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xscale("log")
    tail = track.iloc[len(track) // 3 :]
    span = float(np.nanmax(np.abs(tail["sequential_high"] - tail["sequential_low"]))) or 0.001
    axes.set_ylim(-2.0 * span * POINTS_PER_UNIT, 2.5 * span * POINTS_PER_UNIT)
    axes.set_xlabel("users observed")
    axes.set_ylabel("conversion effect, percentage points")
    axes.set_title("Monitoring the conversion effect continuously")
    axes.grid(alpha=0.3, linewidth=0.6)
    axes.legend(loc="lower right", frameon=False)
    return _save(figure, path)


def plot_mde_curve(table: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.plot(
        table["total_rows"],
        table["mde_relative"] * POINTS_PER_UNIT,
        color=COLOR_PRIMARY,
        label="unadjusted",
    )
    axes.plot(
        table["total_rows"],
        table["mde_relative_cuped"] * POINTS_PER_UNIT,
        color=COLOR_ACCENT,
        label="with variance reduction",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("total users in the experiment")
    axes.set_ylabel("minimum detectable relative lift, percent")
    axes.set_title("How large an effect this design can detect at 80 percent power")
    axes.grid(alpha=0.3, linewidth=0.6, which="both")
    axes.legend(frameon=False)
    return _save(figure, path)


def plot_cuped(table: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    positions = np.arange(table.shape[0])
    width = 0.36
    axes.bar(
        positions - width / 2,
        table["raw_standard_error"] * POINTS_PER_UNIT,
        width,
        color=COLOR_SECONDARY,
        label="unadjusted",
    )
    axes.bar(
        positions + width / 2,
        table["adjusted_standard_error"] * POINTS_PER_UNIT,
        width,
        color=COLOR_ACCENT,
        label="after variance reduction",
    )
    for position, reduction in zip(positions, table["variance_reduction"], strict=False):
        axes.text(
            position,
            0.0,
            f"{reduction * 100.0:.1f} percent less variance",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(table["outcome"])
    axes.set_ylabel("standard error of the effect, percentage points")
    axes.set_title("Variance reduction from a control variate fitted on control users")
    axes.grid(axis="y", alpha=0.3, linewidth=0.6)
    axes.legend(frameon=False)
    return _save(figure, path)


def plot_segments(table: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(8.4, 6.4))
    ordered = table.sort_values("absolute_effect").reset_index(drop=True)
    positions = np.arange(ordered.shape[0])
    values = ordered["absolute_effect"] * POINTS_PER_UNIT
    low = values - ordered["confidence_low"] * POINTS_PER_UNIT
    high = ordered["confidence_high"] * POINTS_PER_UNIT - values
    colours = np.where(ordered["significant_corrected"], COLOR_PRIMARY, COLOR_SECONDARY)
    axes.errorbar(values, positions, xerr=[low, high], fmt="none", ecolor="#C7CDD8", capsize=2)
    axes.scatter(values, positions, color=colours, s=18, zorder=3)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["segment_label"], fontsize=7)
    axes.set_xlabel("conversion effect, percentage points")
    axes.set_title("Segment effects, filled markers survive the Benjamini-Hochberg correction")
    axes.grid(axis="x", alpha=0.3, linewidth=0.6)
    return _save(figure, path)


def plot_allocation(table: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    positions = np.arange(table.shape[0])
    width = 0.38
    axes.bar(
        positions - width / 2,
        table["even_impressions"],
        width,
        color=COLOR_SECONDARY,
        label="budget spread evenly",
    )
    axes.bar(
        positions + width / 2,
        table["greedy_impressions"],
        width,
        color=COLOR_PRIMARY,
        label="budget follows incremental value",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(table["segment"], rotation=45, ha="right", fontsize=8)
    axes.set_xlabel("baseline propensity decile, lowest to highest")
    axes.set_ylabel("impressions allocated")
    axes.set_title("Where a limited impression budget should go")
    axes.grid(axis="y", alpha=0.3, linewidth=0.6)
    axes.legend(frameon=False)
    return _save(figure, path)
