import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

from expengine import config
from expengine.data import loader
from expengine.guardrails import balance, srm
from expengine.hte import segments
from expengine.inference import cuped, itt, iv, power, sequential
from expengine.policy import allocate
from expengine.viz import plots

SUMMARY_FILENAME = "run_summary.json"
MONITOR_CHECKPOINTS = 60


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_table(table: pd.DataFrame, name: str) -> None:
    table.to_csv(config.ARTIFACTS_DIR / name, index=False)


def figure_path(path) -> str:
    return path.as_posix()


def move_to_front(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    remaining = [name for name in table.columns if name not in columns]
    return table.loc[:, columns + remaining]


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialise a value of type {type(value).__name__}")


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def collect_control_features(rows_control: int) -> np.ndarray:
    columns = [*config.FEATURE_COLUMNS, config.TREATMENT_COLUMN]
    blocks = []
    for batch in loader.iter_batches(columns):
        selected = batch[config.TREATMENT_COLUMN].to_numpy() == 0
        if selected.any():
            blocks.append(
                batch.loc[selected, list(config.FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
            )
    features = np.concatenate(blocks)
    if features.shape[0] != rows_control:
        raise ValueError("the control feature block does not match the control row count")
    return features


def build_control_variates(models_by_outcome, assignment, total_rows: int) -> dict[str, np.ndarray]:
    columns = [*config.FEATURE_COLUMNS, config.TREATMENT_COLUMN]
    covariates = {name: np.zeros(total_rows, dtype=np.float32) for name in models_by_outcome}
    control_position = 0
    row_position = 0
    for batch in loader.iter_batches(columns):
        features = batch[list(config.FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        treated = batch[config.TREATMENT_COLUMN].to_numpy() == 1
        control_positions = np.nonzero(~treated)[0]
        folds = assignment[control_position : control_position + control_positions.size]
        size = features.shape[0]
        for name, models in models_by_outcome.items():
            stacked = np.vstack([model.predict(features) for model in models])
            values = stacked.mean(axis=0)
            values[control_positions] = stacked[folds, control_positions]
            covariates[name][row_position : row_position + size] = values
        control_position += control_positions.size
        row_position += size
    return covariates


def load_cached_control_variates(total_rows: int) -> dict[str, np.ndarray] | None:
    path = config.control_variate_path()
    if not path.exists():
        return None
    stored = np.load(path)
    names = list(config.OUTCOME_COLUMNS)
    if any(name not in stored for name in names):
        return None
    if any(stored[name].size != total_rows for name in names):
        return None
    return {name: stored[name] for name in names}


def control_variates(
    frame: pd.DataFrame, control, rows_control: int, total_rows: int, refit: bool
) -> dict[str, np.ndarray]:
    if not refit:
        cached = load_cached_control_variates(total_rows)
        if cached is not None:
            log("reusing the cached control variate")
            return cached
    log("fitting the control variate on control users only")
    control_features = collect_control_features(rows_control)
    assignment = cuped.fold_assignment(rows_control, config.CUPED_FOLDS, config.RANDOM_SEED)
    models_by_outcome = {}
    for name in config.OUTCOME_COLUMNS:
        control_outcome = frame[name].to_numpy()[control].astype(np.float32)
        models_by_outcome[name] = cuped.fit_control_models(
            control_features, control_outcome, assignment
        )
        log(f"fitted {config.CUPED_FOLDS} cross fitted models for {name}")
    del control_features
    log("predicting the control variate for every user")
    covariates = build_control_variates(models_by_outcome, assignment, total_rows)
    np.savez(config.control_variate_path(), **covariates)
    log(f"cached the control variate at {config.control_variate_path()}")
    return covariates


def run(skip_figures: bool = False, refit: bool = False) -> dict:
    started = time.time()
    config.ensure_output_dirs()

    log("building the processed cache")
    loader.build_cache()

    log("loading assignment, exposure and outcomes")
    frame = loader.load_analysis_frame()
    dataset = loader.summarise(frame)
    write_table(loader.require_expected_facts(dataset), "dataset_facts.csv")
    log(f"verified {dataset['rows']} rows against the published facts")

    treatment = frame[config.TREATMENT_COLUMN].to_numpy()
    exposure = frame[config.EXPOSURE_COLUMN].to_numpy()
    treated = treatment == 1
    control = ~treated
    exposure_treated = exposure[treated]
    exposure_control = exposure[control]
    rows_treated = int(dataset["treated_rows"])
    rows_control = int(dataset["control_rows"])
    total_rows = int(dataset["rows"])

    log("guardrail one, sample ratio mismatch")
    srm_result = srm.sample_ratio_mismatch(rows_treated, rows_control)
    write_table(srm.srm_table(srm_result), "srm.csv")
    log(f"srm p value {srm_result['p_value']:.6f}, passes {srm_result['passes']}")

    log("guardrail two, randomisation balance across the 12 features")
    balance_rows = []
    for name in config.FEATURE_COLUMNS:
        values = loader.load_columns([name])[name].to_numpy()
        balance_rows.append(balance.feature_balance(name, values, treatment))
    balance_table = pd.DataFrame.from_records(balance_rows)
    write_table(balance_table, "balance_smd.csv")
    balance_result = balance.balance_summary(balance_table)
    log(f"largest absolute smd {balance_result['max_absolute_smd']:.5f}")

    log("intention to treat effects")
    itt_results = {}
    for name in config.OUTCOME_COLUMNS:
        values = frame[name].to_numpy()
        effect = itt.two_proportion_effect(
            float(values[treated].sum(dtype=np.float64)),
            rows_treated,
            float(values[control].sum(dtype=np.float64)),
            rows_control,
        )
        effect["outcome"] = name
        itt_results[name] = effect
        log(
            f"{name} itt {effect['absolute_effect'] * 100:.4f} pp, "
            f"relative {effect['relative_lift'] * 100:.2f} percent"
        )
    write_table(
        move_to_front(pd.DataFrame.from_records(list(itt_results.values())), ["outcome"]),
        "itt_estimates.csv",
    )

    covariates = control_variates(frame, control, rows_control, total_rows, refit)
    cuped_results = {}
    for name in config.OUTCOME_COLUMNS:
        values = frame[name].to_numpy()
        covariate = covariates[name]
        result = cuped.cuped_from_moments(
            cuped.update_moments(cuped.empty_moments(), values[treated], covariate[treated]),
            cuped.update_moments(cuped.empty_moments(), values[control], covariate[control]),
        )
        result["outcome"] = name
        cuped_results[name] = result
        log(
            f"{name} variance reduction {result['variance_reduction'] * 100:.2f} percent, "
            f"effective sample multiplier {result['effective_sample_multiplier']:.3f}"
        )
    write_table(
        move_to_front(pd.DataFrame.from_records(list(cuped_results.values())), ["outcome"]),
        "cuped_estimates.csv",
    )

    log("always valid sequential inference")
    aa_table = sequential.simulate_peeking(base_rate=float(dataset["visit_control_rate"]))
    write_table(aa_table, "sequential_aa.csv")
    primary = frame[config.PRIMARY_OUTCOME].to_numpy()
    checkpoints = np.unique(
        np.geomspace(config.MDE_CURVE_MIN_TOTAL, total_rows, MONITOR_CHECKPOINTS)
        .round()
        .astype(np.int64)
    )
    order = sequential.arrival_order(total_rows)
    track = sequential.monitor(primary[order], treatment[order], checkpoints)
    write_table(track, "confidence_sequence.csv")
    sequential_result = {
        "simulations": config.AA_SIMULATIONS,
        "peeks": int(aa_table.shape[0]),
        "nominal_alpha": config.AA_ALPHA,
        "fixed_horizon_single_look_rate": float(
            aa_table["fixed_horizon_single_look_rate"].iloc[-1]
        ),
        "fixed_horizon_false_positive_rate": float(
            aa_table["fixed_horizon_any_look_rate"].iloc[-1]
        ),
        "sequential_false_positive_rate": float(aa_table["sequential_any_look_rate"].iloc[-1]),
        "first_crossing_sequential": sequential.first_crossing(track, "sequential_excludes_zero"),
        "first_crossing_fixed": sequential.first_crossing(track, "fixed_excludes_zero"),
    }
    log(
        f"peeking false positive rate {sequential_result['fixed_horizon_false_positive_rate']:.3f} "
        f"versus sequential {sequential_result['sequential_false_positive_rate']:.3f}"
    )

    log("instrumental variables under one sided non compliance")
    exposure_treated_total = float(exposure_treated.sum(dtype=np.float64))
    exposure_control_total = float(exposure_control.sum(dtype=np.float64))
    iv_results = {}
    for name in config.OUTCOME_COLUMNS:
        values = frame[name].to_numpy()
        outcome_control_total = float(values[control].sum(dtype=np.float64))
        cross_treated = float(np.count_nonzero((values[treated] == 1) & (exposure_treated == 1)))
        cross_control = float(np.count_nonzero((values[control] == 1) & (exposure_control == 1)))
        result = iv.cace_from_counts(
            rows_treated,
            rows_control,
            float(values[treated].sum(dtype=np.float64)),
            outcome_control_total,
            exposure_treated_total,
            exposure_control_total,
            cross_treated,
            cross_control,
        )
        naive = iv.naive_exposed_comparison(
            cross_treated, exposure_treated_total, outcome_control_total, rows_control
        )
        result["outcome"] = name
        result["outcome_among_exposed"] = cross_treated
        result["naive_exposed_effect"] = naive["absolute_effect"]
        result["naive_exposed_low"] = naive["confidence_low"]
        result["naive_exposed_high"] = naive["confidence_high"]
        result["naive_over_cace"] = naive["absolute_effect"] / result["cace"]
        iv_results[name] = result
        log(
            f"{name} cace {result['cace'] * 100:.4f} pp, "
            f"{result['cace_over_itt']:.1f} times the itt"
        )
    write_table(
        move_to_front(pd.DataFrame.from_records(list(iv_results.values())), ["outcome"]),
        "iv_estimates.csv",
    )

    log("incrementality and attribution")
    incrementality_result = iv.incrementality(
        itt_results[config.PRIMARY_OUTCOME],
        iv_results[config.PRIMARY_OUTCOME]["outcome_among_exposed"],
    )
    write_table(pd.DataFrame.from_records([incrementality_result]), "incrementality.csv")
    log(
        "attribution over the treated arm overstates impact by "
        f"{incrementality_result['overstatement_treated_arm']:.2f} times"
    )

    log("power and minimum detectable effect")
    baseline = float(dataset[f"{config.PRIMARY_OUTCOME}_control_rate"])
    treatment_share = float(dataset["treatment_share"])
    variance_ratio = float(cuped_results[config.PRIMARY_OUTCOME]["variance_ratio"])
    observed_effect = abs(float(itt_results[config.PRIMARY_OUTCOME]["absolute_effect"]))
    totals = power.curve_grid(config.MDE_CURVE_MIN_TOTAL, total_rows, config.MDE_CURVE_POINTS)
    write_table(
        power.mde_curve(baseline, treatment_share, totals, variance_ratio=variance_ratio),
        "power_mde.csv",
    )
    power_result = {
        "baseline_rate": baseline,
        "target_power": config.TARGET_POWER,
        "alpha": config.AA_ALPHA,
        "mde_absolute": power.minimum_detectable_effect(baseline, rows_treated, rows_control),
        "mde_absolute_cuped": power.minimum_detectable_effect(
            baseline, rows_treated, rows_control, variance_ratio=variance_ratio
        ),
        "rows_required_for_observed_effect": power.required_rows(
            baseline, observed_effect, treatment_share
        ),
        "rows_required_with_cuped": power.required_rows(
            baseline, observed_effect, treatment_share, variance_ratio=variance_ratio
        ),
    }
    power_result["mde_relative"] = power_result["mde_absolute"] / baseline
    power_result["mde_relative_cuped"] = power_result["mde_absolute_cuped"] / baseline
    log(f"minimum detectable relative lift {power_result['mde_relative'] * 100:.2f} percent")

    log("segment effects with multiple testing control")
    segment_rows = []
    for name in config.FEATURE_COLUMNS:
        values = loader.load_columns([name])[name].to_numpy()
        bins = segments.assign_bins(values, segments.quantile_edges(values))
        segment_rows.extend(segments.segment_effects(name, bins, primary, treatment))
    segment_table = segments.apply_multiple_testing(pd.DataFrame.from_records(segment_rows))
    segment_table = move_to_front(segment_table, ["feature", "segment", "segment_label"])
    write_table(segment_table, "segment_effects.csv")
    segment_result = segments.multiple_testing_summary(segment_table)
    log(
        f"{segment_result['significant_uncorrected']} segments significant before correction, "
        f"{segment_result['significant_corrected']} after"
    )

    log("budget allocation across baseline propensity bins")
    covariate = covariates[config.PRIMARY_OUTCOME]
    bins = segments.assign_bins(
        covariate, segments.quantile_edges(covariate, config.ALLOCATION_BINS)
    )
    allocation_rows = []
    for bin_index in np.unique(bins):
        selected = bins == bin_index
        selected_treated = selected & treated
        selected_control = selected & control
        rows_bin_treated = int(selected_treated.sum(dtype=np.int64))
        rows_bin_control = int(selected_control.sum(dtype=np.int64))
        if rows_bin_treated < 2 or rows_bin_control < 2:
            continue
        impressions = float(np.count_nonzero(exposure[selected_treated] == 1))
        compliance = impressions / rows_bin_treated
        effect = itt.two_proportion_effect(
            float(primary[selected_treated].sum(dtype=np.float64)),
            rows_bin_treated,
            float(primary[selected_control].sum(dtype=np.float64)),
            rows_bin_control,
        )
        allocation_rows.append(
            {
                "segment": f"bin {int(bin_index) + 1}",
                "rows_treated": rows_bin_treated,
                "rows_control": rows_bin_control,
                "impressions": impressions,
                "compliance_rate": compliance,
                "itt": effect["absolute_effect"],
                "itt_standard_error": effect["standard_error"],
                "cace": effect["absolute_effect"] / compliance if compliance > 0.0 else 0.0,
            }
        )
    allocation_bins = pd.DataFrame.from_records(allocation_rows)
    budget = float(allocation_bins["impressions"].sum() * config.ALLOCATION_BUDGET_FRACTION)
    allocation_table = pd.concat(
        [
            allocate.allocation_table(
                allocation_bins["segment"],
                allocation_bins["cace"],
                allocation_bins["impressions"],
                budget,
            ),
            allocation_bins.drop(columns=["segment", "impressions"]),
        ],
        axis=1,
    )
    write_table(allocation_table, "allocation.csv")
    allocation_result = allocate.compare_allocations(
        allocation_bins["cace"], allocation_bins["impressions"], budget
    )
    allocation_result["bins"] = int(allocation_bins.shape[0])
    allocation_result["budget_fraction"] = config.ALLOCATION_BUDGET_FRACTION
    log(f"targeted allocation gain multiple {allocation_result['gain_multiple']:.2f}")

    figures = {}
    if not skip_figures:
        log("writing figures")
        primary_itt = itt_results[config.PRIMARY_OUTCOME]
        primary_iv = iv_results[config.PRIMARY_OUTCOME]
        comparison = [
            {
                "label": "ITT, everyone assigned",
                "value": primary_itt["absolute_effect"],
                "low": primary_itt["confidence_low"],
                "high": primary_itt["confidence_high"],
            },
            {
                "label": "CACE, users who saw an ad",
                "value": primary_iv["cace"],
                "low": primary_iv["cace_confidence_low"],
                "high": primary_iv["cace_confidence_high"],
            },
        ]
        figures = {
            "balance": figure_path(
                plots.plot_balance(
                    balance_table,
                    config.FIGURES_DIR / "balance_smd.png",
                    config.BALANCE_SMD_THRESHOLD,
                )
            ),
            "itt_versus_cace": figure_path(
                plots.plot_itt_versus_cace(comparison, config.FIGURES_DIR / "itt_versus_cace.png")
            ),
            "peeking": figure_path(
                plots.plot_peeking(aa_table, config.FIGURES_DIR / "aa_peeking.png")
            ),
            "confidence_sequence": figure_path(
                plots.plot_confidence_sequence(
                    track, config.FIGURES_DIR / "confidence_sequence.png"
                )
            ),
            "cuped": figure_path(
                plots.plot_cuped(
                    pd.DataFrame.from_records(list(cuped_results.values())),
                    config.FIGURES_DIR / "cuped_variance.png",
                )
            ),
            "mde": figure_path(
                plots.plot_mde_curve(
                    power.mde_curve(
                        baseline, treatment_share, totals, variance_ratio=variance_ratio
                    ),
                    config.FIGURES_DIR / "mde_curve.png",
                )
            ),
            "segments": figure_path(
                plots.plot_segments(segment_table, config.FIGURES_DIR / "segment_effects.png")
            ),
            "allocation": figure_path(
                plots.plot_allocation(allocation_table, config.FIGURES_DIR / "allocation.png")
            ),
        }

    payload = {
        "dataset": dataset,
        "guardrails": {"srm": srm_result, "balance": balance_result},
        "itt": itt_results,
        "cuped": cuped_results,
        "sequential": sequential_result,
        "iv": iv_results,
        "incrementality": incrementality_result,
        "power": power_result,
        "segments": segment_result,
        "allocation": allocation_result,
        "figures": figures,
        "library_versions": library_versions(),
        "runtime_seconds": round(time.time() - started, 1),
    }
    summary_path = config.ARTIFACTS_DIR / SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    log(f"wrote {summary_path} in {payload['runtime_seconds']} seconds")
    return payload
