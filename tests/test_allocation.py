import numpy as np
import pytest

from expengine.policy import allocate


def test_following_incremental_value_beats_spreading_the_budget():
    values = np.array([0.001, 0.01, 0.05])
    capacity = np.array([1_000.0, 1_000.0, 1_000.0])
    comparison = allocate.compare_allocations(values, capacity, 1_500.0)
    assert comparison["greedy_incremental_conversions"] > comparison["even_incremental_conversions"]
    assert comparison["gain_multiple"] > 1.0
    assert comparison["gain"] > 0.0


def test_both_rules_respect_capacity_and_budget():
    values = np.array([0.02, 0.01, 0.03])
    capacity = np.array([100.0, 500.0, 50.0])
    greedy = allocate.greedy_allocation(values, capacity, 400.0)
    even = allocate.even_allocation(capacity, 400.0)
    assert greedy.sum() == pytest.approx(400.0)
    assert even.sum() == pytest.approx(400.0)
    assert np.all(greedy <= capacity + 1e-9)
    assert np.all(even <= capacity + 1e-9)


def test_even_allocation_spills_into_the_larger_bins():
    capacity = np.array([10.0, 1_000.0])
    even = allocate.even_allocation(capacity, 400.0)
    assert even[0] == pytest.approx(10.0)
    assert even[1] == pytest.approx(390.0)


def test_a_budget_larger_than_capacity_is_capped():
    capacity = np.array([10.0, 20.0])
    even = allocate.even_allocation(capacity, 1_000.0)
    assert even.sum() == pytest.approx(30.0)


def test_bins_with_no_incremental_value_are_skipped():
    values = np.array([0.05, -0.01])
    capacity = np.array([100.0, 100.0])
    greedy = allocate.greedy_allocation(values, capacity, 150.0)
    assert greedy[1] == 0.0
    assert greedy[0] == pytest.approx(100.0)


def test_allocation_table_reports_both_rules():
    values = np.array([0.05, 0.01])
    capacity = np.array([100.0, 100.0])
    table = allocate.allocation_table(["high", "low"], values, capacity, 120.0)
    assert list(table["segment"]) == ["high", "low"]
    assert table["greedy_impressions"].sum() == pytest.approx(120.0)
    assert table["greedy_incremental_conversions"].sum() > (
        table["even_incremental_conversions"].sum()
    )
