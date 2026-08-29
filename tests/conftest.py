import numpy as np
import pytest

SEED = 20260824


@pytest.fixture
def generator() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def simple_experiment():
    def build(
        rows: int = 400_000,
        treatment_share: float = 0.5,
        control_rate: float = 0.02,
        absolute_effect: float = 0.004,
        seed: int = SEED,
    ):
        stream = np.random.default_rng(seed)
        treatment = (stream.random(rows) < treatment_share).astype(np.int8)
        probability = np.where(treatment == 1, control_rate + absolute_effect, control_rate)
        outcome = (stream.random(rows) < probability).astype(np.int8)
        return treatment, outcome

    return build


@pytest.fixture
def targeted_non_compliance():
    def build(
        rows: int = 600_000,
        treatment_share: float = 0.5,
        compliance: float = 0.2,
        base_rate: float = 0.05,
        complier_effect: float = 0.05,
        seed: int = SEED,
    ):
        stream = np.random.default_rng(seed)
        quality = stream.random(rows)
        treatment = (stream.random(rows) < treatment_share).astype(np.int8)
        exposure = ((treatment == 1) & (quality > 1.0 - compliance)).astype(np.int8)
        probability = base_rate * (0.5 + quality) + complier_effect * exposure
        outcome = (stream.random(rows) < probability).astype(np.int8)
        return treatment, exposure, outcome

    return build
