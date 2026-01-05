from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ExpandingOLSResult:
    pred: pd.Series
    resid: pd.Series
    zscore: pd.Series
    percentile: pd.Series
    coef: pd.DataFrame


def _percentile_rank(past: np.ndarray, x: float) -> float:
    past = past[np.isfinite(past)]
    if past.size == 0 or not np.isfinite(x):
        return np.nan
    s = np.sort(past)
    k = np.searchsorted(s, x, side="right")
    return 100.0 * k / s.size


def expanding_ols_signal(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Iterable[str],
    warmup: int = 104,
) -> ExpandingOLSResult:
   
    x_cols = list(x_cols)
    y = df[y_col].to_numpy(dtype=float)
    X = df[x_cols].to_numpy(dtype=float)

    preds = np.full(len(df), np.nan, dtype=float)
    resid = np.full(len(df), np.nan, dtype=float)
    z = np.full(len(df), np.nan, dtype=float)
    pct = np.full(len(df), np.nan, dtype=float)

    coef_rows = []
    coef_idx = []

    lr = LinearRegression()

    for i in range(len(df)):
        if i < warmup:
            continue

        # Train strictly on the past
        X_train = X[:i]
        y_train = y[:i]
        m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        if m.sum() < max(10, len(x_cols) + 5):
            continue

        lr.fit(X_train[m], y_train[m])

        # Predict current point
        x_i = X[i]
        if not np.isfinite(x_i).all() or not np.isfinite(y[i]):
            continue

        yhat = float(lr.predict(x_i.reshape(1, -1))[0])
        preds[i] = yhat
        resid[i] = float(y[i] - yhat)

        past_r = resid[:i]
        past_r = past_r[np.isfinite(past_r)]
        if past_r.size >= 20:
            mu = float(past_r.mean())
            sd = float(past_r.std(ddof=1))
            if sd > 0:
                z[i] = (resid[i] - mu) / sd
            pct[i] = _percentile_rank(past_r, resid[i])

        # Coefs
        row = {"alpha": float(lr.intercept_)}
        row.update({f"beta_{c}": float(b) for c, b in zip(x_cols, lr.coef_)})
        coef_rows.append(row)
        coef_idx.append(df.index[i])

    pred_s = pd.Series(preds, index=df.index, name="pred")
    resid_s = pd.Series(resid, index=df.index, name="resid")
    z_s = pd.Series(z, index=df.index, name="zscore")
    pct_s = pd.Series(pct, index=df.index, name="percentile")
    coef_df = pd.DataFrame(coef_rows, index=pd.Index(coef_idx, name="date")).sort_index()

    return ExpandingOLSResult(pred=pred_s, resid=resid_s, zscore=z_s, percentile=pct_s, coef=coef_df)


def in_sample_ols_stats(df: pd.DataFrame, *, y_col: str, x_cols: Iterable[str]) -> dict:
    x_cols = list(x_cols)
    y = df[y_col].to_numpy(dtype=float)
    X = df[x_cols].to_numpy(dtype=float)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)

    lr = LinearRegression()
    lr.fit(X[m], y[m])

    return {
        "alpha": float(lr.intercept_),
        **{f"beta_{c}": float(b) for c, b in zip(x_cols, lr.coef_)},
        "r2": float(lr.score(X[m], y[m])),
        "n_obs": int(m.sum()),
    }
