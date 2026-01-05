# Rates-implied Credit Dispersion (v3: SOFR Curve PCA → CCC tail dislocation)

This repo builds a **client-friendly “rates-implied” fair-value** for leveraged-loan credit dispersion and turns the **residual** into a monitoring / trading signal.

- **Target**: `CCC – BB` Spread-to-Maturity (STM) dispersion (bp)
- **Rates engine**: SOFR curve compressed into **Level / Slope / Curvature** via **PCA (3 factors)**
- **Signal**: residual (actual − predicted), plus **real-time z-score** and **percentile** (computed using only past residuals)

---

## Why PCA on the SOFR curve?

The v1/v2 baselines use hand-crafted factors:
- `Level` ≈ 2Y (bp)
- `Slope` ≈ 10Y − 2Y (bp)

In v3 we add a more complete curve representation:
- **PC1 ≈ Level** (parallel shifts)
- **PC2 ≈ Slope** (twists / steepening–flattening)
- **PC3 ≈ Curvature** (belly vs wings)

That extra curvature factor is often where “curve shape” regimes show up (e.g., belly-led rallies/selloffs) and can materially change how “fair-value” responds to macro.

---

## Data inputs (2 CSV files)

### 1) `sofrUS.csv` (headerless)
SOFR curve history. We use:
- `date` (col 1)
- `rate_pct` (col 2)
- `tenor` (col 6, e.g. `3M`, `2Y`, `10Y`, …)
- `curve` (col 14, default `USD SOFR`)

### 2) `benchmark.csv` (headerless)
LCD discounted spreads / benchmarks. We use:
- `region` (col 0, default `LCD-US`)
- `category` (col 2, e.g. `BB Loans`, `B Loans`, `CCC Only`)
- `metric` (col 3, default `Spread to Maturity`)
- `date` (col 10)
- `spread_bp` (col 11)

---

## Methodology (end-to-end)

### Step 0 — Align + clean
1. Extract 3 loan STM series (BB / B / CCC).
2. Extract SOFR tenors (baseline: `3M`, `2Y`, `10Y`).
3. Inner-join on common dates.
4. Drop an early “island” segment if the history starts with a large time gap (default `--gap_days=180`).

### Step 1 — Build dispersion targets
- Control spread:  
  `disp_b_bb_bp = B_STM − BB_STM`
- Target dispersion:  
  `disp_ccc_bb_bp = CCC_STM − BB_STM`

### Step 2 — Build SOFR PCA factors (Level/Slope/Curvature)
1. Pivot SOFR into a **curve matrix**: `(date × tenor)` using `--pca_tenors`  
   default: `1M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,30Y`.
2. **No look-ahead**: fit PCA on the **first `--pca_train` observations**  
   (default: `--warmup`, i.e. the same warm-up used by the expanding regression).
3. PCA loadings are **re-ordered and sign-fixed** to match canonical templates:
   - Level: all tenors same sign
   - Slope: long − short
   - Curvature: belly vs wings
4. Project every day’s curve onto these fixed loadings to get:
   - `pc_level_bp`, `pc_slope_bp`, `pc_curvature_bp`

Artifacts written:
- `outputs/pca_loadings.csv`
- `outputs/pca_loadings.png`
- `outputs/pca_meta.json` (explained variance ratio, tenors, training size)

### Step 3 — “Rates-implied” model (expanding OLS)
For each day `t` (after warm-up):
1. Fit OLS using **only data strictly before `t`**
2. Predict fair-value for `disp_ccc_bb_bp[t]`
3. Compute residual: `resid[t] = actual[t] − pred[t]`
4. Compute **z-score & percentile** vs **past residuals only**

This is implemented in `src/model.py` as `expanding_ols_signal(...)`.

### Step 4 — Dislocation signal
- Positive residual (large): “CCC tail is **wider** than rates imply” → stress / cheap tail protection
- Negative residual (large): “CCC tail is **tighter** than rates imply” → tail rich / complacency

---

## Model variants included
You’ll get all of these in the output table for comparison:

- `1f_slope`: `disp_ccc_bb_bp ~ slope_2s10s_bp`
- `2f_level_slope`: `~ level_2y_bp + slope_2s10s_bp`
- `piecewise_regime_2f`: allows a different slope/intercept in inversion regime
- `pca3_lsc` (v3): `~ pc_level_bp + pc_slope_bp + pc_curvature_bp`

---

## Outputs (what gets generated)
Key files under `outputs/`:
- `signals_all_models.csv`
- `model_comparison_ccc_bb.csv`, `model_comparison_b_bb.csv`
- `summary_latest.json`
- `top_dislocations_<model>.csv`
- `actual_vs_pred_<model>.png`
- `residual_zscore_<model>.png`
- `actual_vs_pred_models.png`, `zscore_models.png`
- `pca_loadings.csv`, `pca_loadings.png`, `pca_meta.json`

---



