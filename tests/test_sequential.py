import numpy as np

from expengine.inference import itt, sequential

TRUE_EFFECT = 0.01


def test_confidence_sequence_is_wider_than_a_fixed_horizon_interval():
    alpha = 0.05
    horizon = 100_000
    rho = sequential.rho_for_horizon(alpha, horizon)
    multiplier = float(sequential.confidence_sequence_multiplier(horizon, alpha, rho))
    assert multiplier > itt.critical_value(1.0 - alpha)


def test_confidence_sequence_tightens_as_data_arrives():
    alpha = 0.05
    rho = sequential.rho_for_horizon(alpha, 100_000)
    values = sequential.confidence_sequence_multiplier(
        np.array([1_000.0, 10_000.0, 100_000.0]), alpha, rho
    )
    assert values[0] > values[1] > values[2]


def test_peek_schedule_ends_at_the_arm_size():
    schedule = sequential.peek_schedule(20_000, 10)
    assert schedule[-1] == 20_000
    assert schedule.size == 10
    assert np.all(np.diff(schedule) > 0)


def test_peeking_inflates_false_positives_while_the_sequence_holds():
    table = sequential.simulate_peeking(
        base_rate=0.05, arm_size=20_000, peeks=15, simulations=800, alpha=0.05, seed=3
    )
    assert table["fixed_horizon_single_look_rate"].iloc[-1] < 0.09
    assert table["fixed_horizon_any_look_rate"].iloc[-1] > 0.10
    assert table["sequential_any_look_rate"].iloc[-1] <= 0.05
    assert table["fixed_horizon_any_look_rate"].is_monotonic_increasing


def test_monitor_covers_a_real_effect(simple_experiment):
    treatment, outcome = simple_experiment(
        rows=200_000, control_rate=0.05, absolute_effect=TRUE_EFFECT
    )
    track = sequential.monitor(outcome, treatment, [20_000, 60_000, 120_000, 200_000])
    assert track.shape[0] == 4
    final = track.iloc[-1]
    assert final["sequential_low"] < final["fixed_low"]
    assert final["sequential_high"] > final["fixed_high"]
    assert final["sequential_low"] <= TRUE_EFFECT <= final["sequential_high"]


def test_first_crossing_reports_minus_one_when_nothing_crosses(simple_experiment):
    treatment, outcome = simple_experiment(rows=20_000, control_rate=0.05, absolute_effect=0.0)
    track = sequential.monitor(outcome, treatment, [5_000, 10_000, 20_000])
    assert sequential.first_crossing(track, "sequential_excludes_zero") == -1


def test_arrival_order_is_a_permutation():
    order = sequential.arrival_order(50_000, seed=5)
    assert order.size == 50_000
    assert np.array_equal(np.sort(order), np.arange(50_000))
    assert not np.array_equal(order, np.arange(50_000))


def test_a_file_sorted_by_arm_needs_the_arrival_order(simple_experiment):
    treatment, outcome = simple_experiment(
        rows=200_000, control_rate=0.05, absolute_effect=TRUE_EFFECT
    )
    checkpoints = [20_000, 60_000, 120_000, 200_000]
    sorted_positions = np.argsort(-treatment, kind="stable")
    sorted_track = sequential.monitor(
        outcome[sorted_positions], treatment[sorted_positions], checkpoints
    )
    order = sequential.arrival_order(treatment.size)
    shuffled_track = sequential.monitor(outcome[order], treatment[order], checkpoints)
    assert sorted_track.shape[0] < len(checkpoints)
    assert shuffled_track.shape[0] == len(checkpoints)
