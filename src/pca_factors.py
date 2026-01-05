from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def tenor_to_years(tenor: str) -> float:
    t = str(tenor).strip().upper()
    if t.endswith("D"):
        n = float(t[:-1])
        return n / 365.0
    if t.endswith("W"):
        n = float(t[:-1])
        return (n * 7.0) / 365.0
    if t.endswith("M"):
        n = float(t[:-1])
        return n / 12.0
    if t.endswith("Y"):
        n = float(t[:-1])
        return n
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"Unsupported tenor format: {tenor!r}")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else (v / n)


def _templates(maturities_years: np.ndarray) -> np.ndarray:
    m = maturities_years.astype(float)
    level = np.ones_like(m)

    slope = m - m.mean()
    slope = slope - (slope @ level) / (level @ level) * level

    curvature = -((m - m.mean()) ** 2)

    curvature = curvature - (curvature @ level) / (level @ level) * level
    curvature = curvature - (curvature @ slope) / (slope @ slope) * slope

    T = np.vstack([_normalize(level), _normalize(slope), _normalize(curvature)])
    return T


@dataclass(frozen=True)
class PcaFit:
    tenors: list[str]
    maturities_years: np.ndarray
    mean_curve: np.ndarray  
    components: np.ndarray  
    explained_var_ratio: np.ndarray  


def fit_static_curve_pca(
    curve: pd.DataFrame,
    *,
    train_size: int,
    n_components: int = 3,
) -> PcaFit:
    if n_components != 3:
        raise ValueError("This project currently supports n_components=3 only (Level/Slope/Curvature).")

    c = curve.dropna().copy()
    if len(c) < max(train_size, 30):
        raise ValueError(f"Not enough curve observations to fit PCA: have {len(c)}, need >= {max(train_size,30)}")

    tenors = list(c.columns)
    maturities = np.array([tenor_to_years(t) for t in tenors], dtype=float)

    train = c.iloc[:train_size]
    mean_curve = train.mean(axis=0).to_numpy(dtype=float)

    X_train = (train.to_numpy(dtype=float) - mean_curve)

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    comps = pca.components_.copy()  # (3, n)

    T = _templates(maturities)  # (3, n)
    sim = comps @ T.T  # (3,3)

    best_perm = None
    best_score = -np.inf
    for perm in permutations(range(3)):
        score = sum(abs(sim[perm[j], j]) for j in range(3))
        if score > best_score:
            best_score = score
            best_perm = perm

    assert best_perm is not None
    aligned = np.vstack([comps[best_perm[j]] for j in range(3)])  
    sim2 = aligned @ T.T
    for j in range(3):
        sgn = 1.0 if sim2[j, j] >= 0 else -1.0
        aligned[j, :] *= sgn

    return PcaFit(
        tenors=tenors,
        maturities_years=maturities,
        mean_curve=mean_curve,
        components=aligned,
        explained_var_ratio=pca.explained_variance_ratio_.copy(),
    )


def project_pca_factors(curve: pd.DataFrame, fit: PcaFit) -> pd.DataFrame:
    missing = [t for t in fit.tenors if t not in curve.columns]
    if missing:
        raise ValueError(f"Curve is missing tenors required by PCA fit: {missing}")

    c = curve[fit.tenors].copy()
    X = c.to_numpy(dtype=float) - fit.mean_curve
    scores_pct = X @ fit.components.T  
    scores_bp = scores_pct * 100.0

    return pd.DataFrame(
        scores_bp,
        index=c.index,
        columns=["pc_level_bp", "pc_slope_bp", "pc_curvature_bp"],
    )


def pca_loadings_table(fit: PcaFit) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "tenor": fit.tenors,
                "maturity_years": fit.maturities_years,
                "Level": fit.components[0, :],
                "Slope": fit.components[1, :],
                "Curvature": fit.components[2, :],
            }
        )
        .set_index("tenor")
        .sort_values("maturity_years")
    )
