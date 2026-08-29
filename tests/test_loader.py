import gzip

import numpy as np
import pandas as pd
import pytest

from expengine import config
from expengine.data import loader

SYNTHETIC_ROWS = 5_000


@pytest.fixture
def synthetic_raw(tmp_path, monkeypatch):
    stream = np.random.default_rng(4)
    columns = {name: stream.normal(size=SYNTHETIC_ROWS) for name in config.FEATURE_COLUMNS}
    treatment = (stream.random(SYNTHETIC_ROWS) < 0.85).astype(np.int8)
    columns[config.TREATMENT_COLUMN] = treatment
    columns["conversion"] = (stream.random(SYNTHETIC_ROWS) < 0.01).astype(np.int8)
    columns["visit"] = (stream.random(SYNTHETIC_ROWS) < 0.05).astype(np.int8)
    columns[config.EXPOSURE_COLUMN] = (
        (treatment == 1) & (stream.random(SYNTHETIC_ROWS) < 0.2)
    ).astype(np.int8)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with gzip.open(raw_dir / config.RAW_FILENAME, "wt", newline="") as handle:
        pd.DataFrame(columns).to_csv(handle, index=False)
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    return treatment


def test_cache_round_trip(synthetic_raw):
    treatment = synthetic_raw
    assert loader.build_cache().exists()
    frame = loader.load_analysis_frame()
    assert frame.shape == (SYNTHETIC_ROWS, len(config.ANALYSIS_COLUMNS))
    summary = loader.summarise(frame)
    assert summary["rows"] == SYNTHETIC_ROWS
    assert summary["treated_rows"] == int(treatment.sum(dtype=np.int64))
    assert summary["exposed_control_rows"] == 0
    assert 0.0 < summary["compliance_rate"] < 1.0


def test_cache_is_reused_until_it_is_forced(synthetic_raw):
    first = loader.build_cache()
    stamp = first.stat().st_mtime_ns
    assert loader.build_cache().stat().st_mtime_ns == stamp


def test_batches_cover_every_row(synthetic_raw):
    loader.build_cache()
    seen = sum(
        batch.shape[0] for batch in loader.iter_batches([config.TREATMENT_COLUMN], batch_rows=512)
    )
    assert seen == SYNTHETIC_ROWS


def test_feature_columns_load_one_at_a_time(synthetic_raw):
    loader.build_cache()
    column = loader.load_columns([config.FEATURE_COLUMNS[0]])
    assert list(column.columns) == [config.FEATURE_COLUMNS[0]]
    assert column.shape[0] == SYNTHETIC_ROWS


def test_a_missing_raw_file_is_reported_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "absent")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    with pytest.raises(FileNotFoundError):
        loader.build_cache()


def test_synthetic_data_fails_the_published_fact_check(synthetic_raw):
    loader.build_cache()
    summary = loader.summarise(loader.load_analysis_frame())
    table = loader.check_facts(summary)
    assert not bool(table["matches"].all())
    with pytest.raises(ValueError):
        loader.require_expected_facts(summary)
