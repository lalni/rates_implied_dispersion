from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: str | Path, dpi: int = 160) -> None:
    p = _ensure_parent(path)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()


def plot_actual_vs_pred(
    df: pd.DataFrame,
    outpath: str | Path,
    title: str,
    *,
    y_col: str = "disp_ccc_bb_bp",
    pred_col: str = "pred",
    actual_label: str = "Actual",
    pred_label: str = "Predicted",
) -> None:
    
    plt.figure()
    plt.plot(df.index, df[y_col], label=actual_label)
    plt.plot(df.index, df[pred_col], label=pred_label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("bp")
    plt.legend()
    savefig(outpath)


def plot_actual_vs_pred_multi(
    df: pd.DataFrame,
    outpath: str | Path,
    title: str,
    *,
    y_col: str,
    pred_cols: Sequence[str],
    labels: Sequence[str],
    actual_label: str = "Actual",
) -> None:
    plt.figure()
    plt.plot(df.index, df[y_col], label=actual_label)
    for c, lab in zip(pred_cols, labels):
        plt.plot(df.index, df[c], label=lab)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("bp")
    plt.legend()
    savefig(outpath)


def plot_residual_zscore(
    df: pd.DataFrame,
    outpath: str | Path,
    title: str,
    *,
    z_col: str = "zscore",
    label: str = "Residual z-score",
) -> None:
    plt.figure()
    plt.plot(df.index, df[z_col], label=label)
    plt.axhline(2.0, linestyle="--")
    plt.axhline(-2.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("z")
    plt.legend()
    savefig(outpath)


def plot_residual_zscore_multi(
    df: pd.DataFrame,
    outpath: str | Path,
    title: str,
    *,
    z_cols: Sequence[str],
    labels: Sequence[str],
) -> None:
    plt.figure()
    for c, lab in zip(z_cols, labels):
        plt.plot(df.index, df[c], label=lab)
    plt.axhline(2.0, linestyle="--")
    plt.axhline(-2.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("z")
    plt.legend()
    savefig(outpath)


def plot_scatter(df: pd.DataFrame, outpath: str | Path, title: str) -> None:
    plt.figure()
    plt.scatter(df["slope_2s10s_bp"], df["disp_ccc_bb_bp"], s=14)
    plt.title(title)
    plt.xlabel("SOFR curve slope (10Y-2Y, bp)")
    plt.ylabel("CCC-BB dispersion (bp)")
    savefig(outpath)




def plot_pca_loadings(loadings: pd.DataFrame, outpath: str | Path, title: str = "PCA loadings") -> None:
    df = loadings.copy()
    for c in ["Level", "Slope", "Curvature"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} in loadings table")

    if "maturity_years" in df.columns:
        df = df.sort_values("maturity_years")

    tenors = list(df.index.astype(str))

    plt.figure(figsize=(10, 4))
    for c in ["Level", "Slope", "Curvature"]:
        plt.plot(range(len(tenors)), df[c].values, marker="o", label=c)

    plt.xticks(range(len(tenors)), tenors, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Loading (unitless)")
    plt.tight_layout()
    plt.legend()
    savefig(outpath)
