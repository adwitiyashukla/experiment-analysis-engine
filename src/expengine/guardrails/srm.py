import pandas as pd
from scipy import stats

from expengine import config


def sample_ratio_mismatch(
    rows_treated: int,
    rows_control: int,
    intended_treatment_share: float = config.INTENDED_TREATMENT_SHARE,
    threshold: float = config.SRM_P_VALUE_THRESHOLD,
) -> dict[str, float]:
    total = rows_treated + rows_control
    if total <= 0:
        raise ValueError("sample ratio mismatch needs at least one row")
    if not 0.0 < intended_treatment_share < 1.0:
        raise ValueError("intended treatment share must sit strictly between 0 and 1")
    expected_treated = total * intended_treatment_share
    expected_control = total * (1.0 - intended_treatment_share)
    chi_square = (rows_treated - expected_treated) ** 2 / expected_treated + (
        rows_control - expected_control
    ) ** 2 / expected_control
    p_value = float(stats.chi2.sf(chi_square, df=1))
    return {
        "rows_total": int(total),
        "rows_treated": int(rows_treated),
        "rows_control": int(rows_control),
        "expected_treated": float(expected_treated),
        "expected_control": float(expected_control),
        "observed_treatment_share": float(rows_treated / total),
        "intended_treatment_share": float(intended_treatment_share),
        "chi_square": float(chi_square),
        "p_value": p_value,
        "threshold": float(threshold),
        "passes": bool(p_value >= threshold),
    }


def srm_table(result: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame.from_records([result])
