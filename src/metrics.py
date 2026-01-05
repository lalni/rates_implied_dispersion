from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_mean_baseline(y: pd.Series, *, warmup: int) -> pd.Series:
    y_arr = y.to_numpy(dtype=float)
    out = np.full(len(y_arr), np.nan, dtype=float)
    for i in range(len(y_arr)):
        if i < warmup:
            continue
        past = y_arr[:i]
        past = past[np.isfinite(past)]
        if past.size == 0:
            continue
        out[i] = float(past.mean())
    return pd.Series(out, index=y.index, name="pred_naive_mean")


def compute_oos_metrics(
    df: pd.DataFrame,
    *,
    y_col: str,
    pred: pd.Series,
    warmup: int,
) -> dict:
    
    y = df[y_col].astype(float)
    mask = pred.notna() & y.notna()
    y_eval = y[mask]
    pred_eval = pred[mask].astype(float)
    resid = y_eval - pred_eval

    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))

    naive = expanding_mean_baseline(y, warmup=warmup)
    naive_eval = naive[mask].astype(float)

    sse_model = float(np.sum((y_eval - pred_eval) ** 2))
    sse_naive = float(np.sum((y_eval - naive_eval) ** 2))
    oos_r2 = float(1.0 - sse_model / sse_naive) if sse_naive > 0 else float("nan")

    return {"rmse_bp": rmse, "mae_bp": mae, "oos_r2_vs_expanding_mean": oos_r2, "n_eval": int(mask.sum())}


def compute_z_metrics(z: pd.Series) -> dict:
    z = z.dropna().astype(float)
    if len(z) == 0:
        return {"z_mean": float("nan"), "z_std": float("nan"), "pct_|z|>=2": float("nan"), "n": 0}

    return {
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "pct_|z|>=2": float((z.abs() >= 2).mean() * 100.0),
        "pct_z>=2": float((z >= 2).mean() * 100.0),
        "pct_z>=3": float((z >= 3).mean() * 100.0),
        "pct_z<=-2": float((z <= -2).mean() * 100.0),
        "n": int(len(z)),
    }
