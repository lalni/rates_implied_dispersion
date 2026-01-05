
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SofrSpec:

    curve_name: str = "USD SOFR"  
    date_col: int = 1           
    value_col: int = 2        
    tenor_col: int = 6         
    curve_col: int = 14          


@dataclass(frozen=True)
class BenchSpec:

    region: str = "LCD-US"                
    metric: str = "Spread to Maturity"    
    region_col: int = 0
    category_col: int = 2                 # BB Loans, B Loans, CCC Only
    metric_col: int = 3
    rating_col: int = 5                   
    date_col: int = 10
    value_col: int = 11                   # bp


def load_sofr_csv(path: str | Path, *, low_memory: bool = False) -> pd.DataFrame:

    path = Path(path).expanduser()
    df = pd.read_csv(path, header=None, low_memory=low_memory)
    df = df.rename(columns={1: "date", 2: "rate_pct", 6: "tenor", 14: "curve"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rate_pct"] = pd.to_numeric(df["rate_pct"], errors="coerce")
    df["tenor"] = df["tenor"].astype(str)
    df["curve"] = df["curve"].astype(str)
    return df


def load_benchmark_csv(path: str | Path, *, low_memory: bool = False) -> pd.DataFrame:

    path = Path(path).expanduser()
    df = pd.read_csv(path, header=None, low_memory=low_memory)
    df = df.rename(columns={0: "region", 2: "category", 3: "metric", 5: "rating", 10: "date", 11: "spread_bp"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["spread_bp"] = pd.to_numeric(df["spread_bp"], errors="coerce")
    df["region"] = df["region"].astype(str)
    df["category"] = df["category"].astype(str)
    df["metric"] = df["metric"].astype(str)
    df["rating"] = df["rating"].astype(str)
    return df


def extract_sofr_tenor_series(
    sofr_raw: pd.DataFrame,
    curve_name: str,
    tenor: str,
) -> pd.Series:
    """Extract a SOFR tenor time series (percent), indexed by date."""
    sub = sofr_raw[(sofr_raw["curve"] == curve_name) & (sofr_raw["tenor"] == tenor)].copy()
    sub = sub.dropna(subset=["date", "rate_pct"])
    s = sub.sort_values("date").groupby("date")["rate_pct"].last()
    s.name = f"sofr_{tenor.lower()}_pct"
    return s


def extract_benchmark_series(
    bench_raw: pd.DataFrame,
    *,
    region: str,
    category: str,
    metric: str,
) -> pd.Series:

    sub = bench_raw[(bench_raw["region"] == region) & (bench_raw["category"] == category) & (bench_raw["metric"] == metric)].copy()
    sub = sub.dropna(subset=["date", "spread_bp"])
    s = sub.sort_values("date").groupby("date")["spread_bp"].last()
    s.name = category.lower().replace(" ", "_")
    return s
