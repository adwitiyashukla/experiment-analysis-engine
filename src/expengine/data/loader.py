from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from expengine import config

FEATURE_DTYPE = "float32"
FLAG_DTYPE = "int8"


def _normalise(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.loc[:, list(config.ALL_COLUMNS)].copy()
    for name in config.ANALYSIS_COLUMNS:
        values = frame[name].to_numpy()
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"column {name} holds values outside 0 and 1")
        frame[name] = values.astype(FLAG_DTYPE)
    return frame


def build_cache(force: bool = False) -> Path:
    target = config.processed_path()
    if target.exists() and not force:
        return target
    source = config.raw_path()
    if not source.exists():
        raise FileNotFoundError(
            f"raw file not found at {source}. Set EXPENGINE_RAW_DIR to the folder that holds "
            f"{config.RAW_FILENAME}"
        )
    config.ensure_output_dirs()
    temporary = target.parent / (target.name + ".tmp")
    reader = pd.read_csv(
        source,
        usecols=list(config.ALL_COLUMNS),
        dtype=FEATURE_DTYPE,
        chunksize=config.READ_CHUNK_ROWS,
    )
    writer = None
    try:
        for chunk in reader:
            table = pa.Table.from_pandas(_normalise(chunk), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression=config.PARQUET_COMPRESSION
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no rows were read from {source}")
    temporary.replace(target)
    return target


def load_columns(columns: Sequence[str], path: Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else config.processed_path()
    return pq.read_table(source, columns=list(columns)).to_pandas()


def iter_batches(
    columns: Sequence[str],
    batch_rows: int = config.READ_CHUNK_ROWS,
    path: Path | None = None,
) -> Iterator[pd.DataFrame]:
    source = Path(path) if path is not None else config.processed_path()
    handle = pq.ParquetFile(source)
    for batch in handle.iter_batches(batch_size=batch_rows, columns=list(columns)):
        yield batch.to_pandas()


def load_analysis_frame(path: Path | None = None) -> pd.DataFrame:
    return load_columns(config.ANALYSIS_COLUMNS, path=path)


def summarise(frame: pd.DataFrame) -> dict[str, float]:
    treatment = frame[config.TREATMENT_COLUMN].to_numpy()
    exposure = frame[config.EXPOSURE_COLUMN].to_numpy()
    treated = treatment == 1
    control = ~treated
    summary: dict[str, float] = {
        "rows": int(frame.shape[0]),
        "treated_rows": int(treated.sum(dtype=np.int64)),
        "control_rows": int(control.sum(dtype=np.int64)),
        "exposed_treated_rows": int(exposure[treated].sum(dtype=np.int64)),
        "exposed_control_rows": int(exposure[control].sum(dtype=np.int64)),
    }
    summary["treatment_share"] = summary["treated_rows"] / summary["rows"]
    summary["compliance_rate"] = summary["exposed_treated_rows"] / summary["treated_rows"]
    for outcome in config.OUTCOME_COLUMNS:
        values = frame[outcome].to_numpy()
        summary[f"{outcome}_treated_count"] = int(values[treated].sum(dtype=np.int64))
        summary[f"{outcome}_control_count"] = int(values[control].sum(dtype=np.int64))
        summary[f"{outcome}_treated_rate"] = float(values[treated].mean(dtype=np.float64))
        summary[f"{outcome}_control_rate"] = float(values[control].mean(dtype=np.float64))
    return summary


def expected_facts() -> dict[str, float]:
    return {
        "rows": config.EXPECTED_ROWS,
        "control_rows": config.EXPECTED_CONTROL_ROWS,
        "treated_rows": config.EXPECTED_TREATED_ROWS,
        "exposed_control_rows": config.EXPECTED_EXPOSED_CONTROL_ROWS,
        "treatment_share": config.INTENDED_TREATMENT_SHARE,
        "compliance_rate": config.EXPECTED_COMPLIANCE_RATE,
        "visit_control_rate": config.EXPECTED_CONTROL_VISIT_RATE,
        "visit_treated_rate": config.EXPECTED_TREATED_VISIT_RATE,
        "conversion_control_rate": config.EXPECTED_CONTROL_CONVERSION_RATE,
        "conversion_treated_rate": config.EXPECTED_TREATED_CONVERSION_RATE,
        "conversion_treated_count": config.EXPECTED_TREATED_CONVERSIONS,
    }


def fact_tolerances() -> dict[str, float]:
    return {
        "treatment_share": config.RATE_TOLERANCE,
        "compliance_rate": config.COMPLIANCE_TOLERANCE,
        "visit_control_rate": config.RATE_TOLERANCE,
        "visit_treated_rate": config.RATE_TOLERANCE,
        "conversion_control_rate": config.RATE_TOLERANCE,
        "conversion_treated_rate": config.RATE_TOLERANCE,
    }


def check_facts(summary: dict[str, float]) -> pd.DataFrame:
    tolerances = fact_tolerances()
    records = []
    for fact, expected in expected_facts().items():
        observed = summary[fact]
        tolerance = tolerances.get(fact, 0.0)
        records.append(
            {
                "fact": fact,
                "expected": expected,
                "observed": observed,
                "tolerance": tolerance,
                "matches": bool(abs(float(observed) - float(expected)) <= tolerance),
            }
        )
    return pd.DataFrame.from_records(records)


def require_expected_facts(summary: dict[str, float]) -> pd.DataFrame:
    table = check_facts(summary)
    failures = table.loc[~table["matches"], "fact"].tolist()
    if failures:
        raise ValueError(f"dataset facts do not match the published values: {failures}")
    return table
