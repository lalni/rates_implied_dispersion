from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.io import (
    BenchSpec,
    SofrSpec,
    extract_benchmark_series,
    extract_sofr_tenor_series,
    load_benchmark_csv,
    load_sofr_csv,
)
from src.features import add_piecewise_regime_features, build_weekly_dataset, trim_discontinuous_head
from src.metrics import compute_oos_metrics, compute_z_metrics
from src.model import expanding_ols_signal, in_sample_ols_stats
from src.pca_factors import fit_static_curve_pca, pca_loadings_table, project_pca_factors
from src.plotting import (
    plot_actual_vs_pred,
    plot_actual_vs_pred_multi,
    plot_pca_loadings,
    plot_residual_zscore,
    plot_residual_zscore_multi,
    plot_scatter,
)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def build_curve_matrix(sofr_raw: pd.DataFrame, *, curve_name: str, tenors: list[str]) -> pd.DataFrame:
    """Pivot SOFR raw history into a (date x tenor) curve matrix (rate in percent)."""
    sub = sofr_raw[(sofr_raw["curve"] == curve_name) & (sofr_raw["tenor"].isin(tenors))].copy()
    if sub.empty:
        raise ValueError(f"No SOFR rows found for curve={curve_name!r} and tenors={tenors}")
    pivot = (
        sub.pivot_table(index="date", columns="tenor", values="rate_pct", aggfunc="last")
        .sort_index()
        .reindex(columns=tenors)
    )
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofr", required=True, help="Path to SOFR CSV (headerless)")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark CSV (headerless)")
    parser.add_argument("--out", default="outputs", help="Output folder (default: outputs)")
    parser.add_argument("--warmup", type=int, default=104, help="Warm-up obs for expanding OLS (default: 104)")
    parser.add_argument(
        "--gap_days",
        type=int,
        default=180,
        help="Trim an early discontinuous 'island' if a gap exceeds this (default: 180 days)",
    )
    parser.add_argument(
        "--curve_name",
        default=None,
        help="SOFR curve name in the export (default: SofrSpec.curve_name)",
    )
    parser.add_argument(
        "--pca_tenors",
        default="1M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,30Y",
        help="Comma-separated tenors for PCA curve matrix",
    )
    parser.add_argument(
        "--pca_train",
        type=int,
        default=None,
        help="Calibration window size for PCA loadings (default: warmup)",
    )

    args = parser.parse_args()
    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    sofr_raw = load_sofr_csv(args.sofr)
    bench_raw = load_benchmark_csv(args.benchmark)

    sofr_spec = SofrSpec()
    curve_name = args.curve_name or sofr_spec.curve_name

    # Extract SOFR tenors 
    sofr_3m = extract_sofr_tenor_series(sofr_raw, curve_name, "3M")
    sofr_2y = extract_sofr_tenor_series(sofr_raw, curve_name, "2Y")
    sofr_10y = extract_sofr_tenor_series(sofr_raw, curve_name, "10Y")

    # Extract benchmark time series (bp)
    bench_spec = BenchSpec()
    bb = extract_benchmark_series(bench_raw, region=bench_spec.region, category="BB Loans", metric=bench_spec.metric)
    b = extract_benchmark_series(bench_raw, region=bench_spec.region, category="B Loans", metric=bench_spec.metric)
    ccc = extract_benchmark_series(bench_raw, region=bench_spec.region, category="CCC Only", metric=bench_spec.metric)

    df = build_weekly_dataset(bb, b, ccc, sofr_3m, sofr_2y, sofr_10y)
    df = trim_discontinuous_head(df, max_gap_days=args.gap_days)
    df = add_piecewise_regime_features(df)

    # ----------------------------
    # PCA factors 
    # ----------------------------
    tenors = [t.strip() for t in str(args.pca_tenors).split(",") if t.strip()]
    curve = build_curve_matrix(sofr_raw, curve_name=curve_name, tenors=tenors)

    # Use only dates that exist in the dispersion dataset
    curve = curve.loc[curve.index.intersection(df.index)].sort_index()
    if len(curve) < max(args.warmup, 60):
        raise ValueError(f"Not enough overlapping dates for PCA: have {len(curve)}")

    pca_train = int(args.pca_train) if args.pca_train is not None else int(args.warmup)
    pca_train = min(pca_train, len(curve) - 1)

    pca_fit = fit_static_curve_pca(curve, train_size=pca_train, n_components=3)
    factors = project_pca_factors(curve, pca_fit)

    df = df.join(factors, how="inner").dropna()

    loadings = pca_loadings_table(pca_fit)
    loadings.to_csv(outdir / "pca_loadings.csv", index_label="tenor")
    plot_pca_loadings(loadings, outdir / "pca_loadings.png", "SOFR PCA loadings (v3)")
    write_json(
        outdir / "pca_meta.json",
        {
            "curve_name": curve_name,
            "tenors": tenors,
            "train_size": int(pca_train),
            "explained_variance_ratio": [float(x) for x in pca_fit.explained_var_ratio],
        },
    )

    # ----------------------------
    # Model variants
    # ----------------------------
    models = [
        {"name": "1f_slope", "x_cols": ["slope_2s10s_bp"]},
        {"name": "2f_level_slope", "x_cols": ["level_2y_bp", "slope_2s10s_bp"]},
        {"name": "piecewise_regime_2f", "x_cols": ["level_2y_bp", "slope_2s10s_bp", "inv_dummy", "slope_x_inv"]},
        {"name": "pca3_lsc", "x_cols": ["pc_level_bp", "pc_slope_bp", "pc_curvature_bp"]},
    ]

    target = "disp_ccc_bb_bp"
    control = "disp_b_bb_bp"

    signals = df.copy()

    summary: dict = {
        "dataset": {
            "n_obs": int(len(df)),
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "warmup": int(args.warmup),
            "gap_days_trimmed": int(args.gap_days),
        },
        "pca": json.loads((outdir / "pca_meta.json").read_text()),
        "latest": {},
        "models": {},
        "control_models": {},
    }

    latest_row = df.iloc[-1]
    summary["latest"] = {
        "date": str(df.index.max().date()),
        "sofr_3m_pct": float(latest_row["sofr_3m_pct"]),
        "sofr_2y_pct": float(latest_row["sofr_2y_pct"]),
        "sofr_10y_pct": float(latest_row["sofr_10y_pct"]),
        "level_2y_bp": float(latest_row["level_2y_bp"]),
        "slope_2s10s_bp": float(latest_row["slope_2s10s_bp"]),
        "pc_level_bp": float(latest_row["pc_level_bp"]),
        "pc_slope_bp": float(latest_row["pc_slope_bp"]),
        "pc_curvature_bp": float(latest_row["pc_curvature_bp"]),
        "bb_stm_bp": float(latest_row["bb_stm_bp"]),
        "b_stm_bp": float(latest_row["b_stm_bp"]),
        "ccc_stm_bp": float(latest_row["ccc_stm_bp"]),
        "disp_b_bb_bp": float(latest_row["disp_b_bb_bp"]),
        "disp_ccc_bb_bp": float(latest_row["disp_ccc_bb_bp"]),
    }

    # Run all models 
    for spec in models:
        name = spec["name"]
        x_cols = spec["x_cols"]

        sig = expanding_ols_signal(df, y_col=target, x_cols=x_cols, warmup=args.warmup)
        signals[f"pred_{name}"] = sig.pred
        signals[f"resid_{name}"] = sig.resid
        signals[f"z_{name}"] = sig.zscore
        signals[f"pct_{name}"] = sig.percentile

        sig_c = expanding_ols_signal(df, y_col=control, x_cols=x_cols, warmup=args.warmup)
        signals[f"pred_{control}_{name}"] = sig_c.pred
        signals[f"resid_{control}_{name}"] = sig_c.resid
        signals[f"z_{control}_{name}"] = sig_c.zscore
        signals[f"pct_{control}_{name}"] = sig_c.percentile
        sig.coef.to_csv(outdir / f"coef_time_series_{name}.csv", index_label="date")
        sig_c.coef.to_csv(outdir / f"coef_time_series_{control}_{name}.csv", index_label="date")


        # Metrics blocks
        ins = in_sample_ols_stats(df, y_col=target, x_cols=x_cols)
        oos = compute_oos_metrics(df, y_col=target, pred=signals[f"pred_{name}"], warmup=args.warmup)
        zmet = compute_z_metrics(signals[f"z_{name}"])

        latest = {
            "date": str(df.index.max().date()),
            "pred": float(sig.pred.iloc[-1]) if pd.notna(sig.pred.iloc[-1]) else None,
            "resid": float(sig.resid.iloc[-1]) if pd.notna(sig.resid.iloc[-1]) else None,
            "zscore": float(sig.zscore.iloc[-1]) if pd.notna(sig.zscore.iloc[-1]) else None,
            "percentile": float(sig.percentile.iloc[-1]) if pd.notna(sig.percentile.iloc[-1]) else None,
        }

        summary["models"][name] = {
            "x_cols": x_cols,
            "in_sample": ins,
            "oos": oos,
            "z_diagnostics": zmet,
            "latest_signal": latest,
        }

        ins_c = in_sample_ols_stats(df, y_col=control, x_cols=x_cols)
        oos_c = compute_oos_metrics(df, y_col=control, pred=signals[f"pred_{control}_{name}"], warmup=args.warmup)
        zmet_c = compute_z_metrics(signals[f"z_{control}_{name}"])

        latest_c = {
            "date": str(df.index.max().date()),
            "pred": float(sig_c.pred.iloc[-1]) if pd.notna(sig_c.pred.iloc[-1]) else None,
            "resid": float(sig_c.resid.iloc[-1]) if pd.notna(sig_c.resid.iloc[-1]) else None,
            "zscore": float(sig_c.zscore.iloc[-1]) if pd.notna(sig_c.zscore.iloc[-1]) else None,
            "percentile": float(sig_c.percentile.iloc[-1]) if pd.notna(sig_c.percentile.iloc[-1]) else None,
        }

        summary["control_models"][name] = {
            "x_cols": x_cols,
            "in_sample": ins_c,
            "oos": oos_c,
            "z_diagnostics": zmet_c,
            "latest_signal": latest_c,
        }

        # Top dislocations for the target
        top = signals.dropna(subset=[f"resid_{name}"]).sort_values(f"resid_{name}", ascending=False).head(15)
        top_cols = [
            target,
            "level_2y_bp",
            "slope_2s10s_bp",
            "pc_level_bp",
            "pc_slope_bp",
            "pc_curvature_bp",
            f"pred_{name}",
            f"resid_{name}",
            f"z_{name}",
            f"pct_{name}",
        ]
        top[top_cols].to_csv(outdir / f"top_dislocations_{name}.csv", index_label="date")

        plot_actual_vs_pred(
            signals,
            outdir / f"actual_vs_pred_{name}.png",
            f"CCC-BB dispersion vs Rates-implied ({name})",
            y_col=target,
            pred_col=f"pred_{name}",
            pred_label=name,
        )
        plot_residual_zscore(
            signals,
            outdir / f"residual_zscore_{name}.png",
            f"Residual z-score ({name})",
            z_col=f"z_{name}",
            label=name,
        )

    # Save the full signal table
    signals.to_csv(outdir / "signals_all_models.csv", index_label="date")

    # Comparison tables (target + control)
    def _comparison_table(which: str) -> pd.DataFrame:
        rows = []
        for spec in models:
            name = spec["name"]
            block = summary[which][name]
            rows.append({"model": name, **block["in_sample"], **block["oos"], **block["z_diagnostics"]})
        return pd.DataFrame(rows).set_index("model")

    comp_target = _comparison_table("models")
    comp_ctrl = _comparison_table("control_models")
    comp_target.to_csv(outdir / "model_comparison_ccc_bb.csv")
    comp_ctrl.to_csv(outdir / "model_comparison_b_bb.csv")

    # Multi-model overlay plots (target)
    pred_cols = [f"pred_{m['name']}" for m in models]
    z_cols = [f"z_{m['name']}" for m in models]
    labels = [m["name"] for m in models]

    plot_actual_vs_pred_multi(
        signals,
        outdir / "actual_vs_pred_models.png",
        "CCC-BB dispersion: Actual vs all model predictions",
        y_col=target,
        pred_cols=pred_cols,
        labels=labels,
    )
    plot_residual_zscore_multi(
        signals,
        outdir / "zscore_models.png",
        "Residual z-score: all models",
        z_cols=z_cols,
        labels=labels,
    )
    plot_scatter(signals, outdir / "scatter_slope_vs_dispersion.png", "SOFR slope vs CCC-BB dispersion")

    # Summary JSON
    write_json(outdir / "summary_latest.json", summary)

    print("Done. Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
