"""

Feature engineering for the Rates → Credit dispersion models.
"""

from __future__ import annotations

import pandas as pd


def build_weekly_dataset(
    bb_spread_bp: pd.Series,
    b_spread_bp: pd.Series,
    ccc_spread_bp: pd.Series,
    sofr_3m_pct: pd.Series,
    sofr_2y_pct: pd.Series,
    sofr_10y_pct: pd.Series,
) -> pd.DataFrame:
    """Join the three credit series + SOFR tenors into one aligned DataFrame.

    Notes:
    - Inputs can be daily/weekly; we align on exact dates present in *all* series (inner join).
    - Output columns are in bp/% and include derived dispersion + curve features.

    Returns
    -------
    DataFrame indexed by date with columns:
    - bb_stm_bp, b_stm_bp, ccc_stm_bp
    - sofr_3m_pct, sofr_2y_pct, sofr_10y_pct
    - disp_b_bb_bp, disp_ccc_bb_bp
    - slope_2s10s_bp, level_2y_bp, front_3m_bp
    """
    credit = pd.DataFrame(
        {
            "bb_stm_bp": bb_spread_bp,
            "b_stm_bp": b_spread_bp,
            "ccc_stm_bp": ccc_spread_bp,
        }
    ).dropna()

    rates = pd.DataFrame(
        {
            "sofr_3m_pct": sofr_3m_pct,
            "sofr_2y_pct": sofr_2y_pct,
            "sofr_10y_pct": sofr_10y_pct,
        }
    ).dropna()

    out = credit.join(rates, how="inner").dropna()
    out = out.sort_index()

    # Credit dispersion (bp)
    out["disp_b_bb_bp"] = out["b_stm_bp"] - out["bb_stm_bp"]
    out["disp_ccc_bb_bp"] = out["ccc_stm_bp"] - out["bb_stm_bp"]

    # Rate features (bp)
    out["slope_2s10s_bp"] = (out["sofr_10y_pct"] - out["sofr_2y_pct"]) * 100.0
    out["level_2y_bp"] = out["sofr_2y_pct"] * 100.0
    out["front_3m_bp"] = out["sofr_3m_pct"] * 100.0

    return out


def trim_discontinuous_head(df: pd.DataFrame, *, max_gap_days: int = 180) -> pd.DataFrame:
    """Drop an early 'island' segment if the history starts with a large time gap.

    Example: the provided data has a single 2018 observation and then resumes in 2021.
    For real-time monitoring, keeping the continuous history is usually preferable.
    """
    if df.empty:
        return df

    s = df.sort_index()
    gaps = s.index.to_series().diff().dt.days
    if gaps.isna().all():
        return s

    # First date that follows a large gap (keep from there onward)
    big = gaps[gaps > max_gap_days]
    if len(big) == 0:
        return s

    start_date = big.index[0]
    return s.loc[start_date:].copy()


def add_piecewise_regime_features(
    df: pd.DataFrame,
    *,
    slope_col: str = "slope_2s10s_bp",
    threshold_bp: float = 0.0,
    dummy_col: str = "inv_dummy",
    interaction_col: str = "slope_x_inv",
) -> pd.DataFrame:
    out = df.copy()
    out[dummy_col] = (out[slope_col] < threshold_bp).astype(int)
    out[interaction_col] = out[slope_col] * out[dummy_col]
    return out
