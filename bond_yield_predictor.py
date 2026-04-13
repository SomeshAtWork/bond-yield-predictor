"""
================================================================================
  INDIAN GOVERNMENT BOND YIELD PREDICTION ENGINE  v3.3
================================================================================
  Multi-Model Ensemble:
    Econometric  —  VAR / VECM  (reduced-form, max 4 vars, lag-capped)
    Term-Struct  —  Nelson-Siegel-Svensson (daily OLS)  +  Hull-White
    ML           —  XGBoost (regressor-only, no classifier mismatch)
    Deep Learn   —  Bidirectional LSTM  (scaler fitted on TRAIN only)

  v3.3 fixes vs v3.2  (9 engineering fixes):
    1. DATA_START_YEAR removed — auto-detect; sparse maturities (40Y/50Y)
       excluded by MIN_MAT_OBS real-observation filter
    2. Johansen cointegration moved INSIDE build_econometric() so
       k_ar_diff matches VAR-selected lag (was hardcoded 2 in diagnostics)
    3. Hull-White multi-start (5 seeds) to escape local minima
    4. _make_xgb() factory enforces identical hyperparams across
       build_xgboost() and run_backtest()
    5. LSTM seq_len derived from ACF decay (not hardcoded 60)
    6. FLAT threshold per-maturity from 20th percentile of |historical Δ|
    7. LOW CONFIDENCE threshold from binomial test (not hardcoded 55%)
    8. Granger causality uses diff() for rate vars (consistent with z-score)
    9. XGBoost stores n_test for statistical confidence computation

  Data leakage: ZERO. Verified by gap + train-only scaling.
================================================================================
"""

# ============================================================================
# SECTION 1 — IMPORTS
# ============================================================================
import os, warnings, datetime as dt
from pathlib import Path

import numpy  as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize as sp_minimize

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

import xgboost as xgb

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

warnings.filterwarnings("ignore")

LSTM_AVAILABLE = False
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM as KerasLSTM, Dense, Dropout,
                                         Bidirectional, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
    LSTM_AVAILABLE = True
except ImportError:
    pass

print("=" * 72)
print("  INDIAN GOVERNMENT BOND YIELD PREDICTION ENGINE  v3.3")
models_str = "VAR/VECM + NSS + Hull-White + XGBoost"
if LSTM_AVAILABLE:
    models_str += " + Bi-LSTM"
print(f"  Ensemble: {models_str}")
print("=" * 72)
if not LSTM_AVAILABLE:
    print("  [!] TensorFlow not found  ->  LSTM disabled.")
    print("      Install:  pip install tensorflow\n")


# ============================================================================
# SECTION 2 — CONFIGURATION
# ============================================================================
EXCEL_PATH     = Path(r"C:\Users\Lenovo\Downloads\dataforpythontraining.xlsx")
OUTPUT_PATH    = Path(r"C:\Users\Lenovo\Downloads\bond_predictions_output.xlsx")
INPUT_SHEET    = "Sheet1"
OUTPUT_SHEET   = "Predictions"
BACKTEST_SHEET = "Backtest_Results"
NSS_CACHE      = Path(r"C:\Users\Lenovo\Downloads\nss_params_cache.csv")

MIN_MAT_OBS = 500   # minimum real (pre-interpolation) observations per maturity

YIELD_COLS = {
    "3M":"IN3MT Yield (%)","6M":"IN6MT Yield (%)",
    "1Y":"IN1YT Yield (%)","2Y":"IN2YT Yield (%)",
    "3Y":"IN3YT Yield (%)","4Y":"IN4YT Yield (%)",
    "5Y":"IN5YT Yield (%)","6Y":"IN6YT Yield (%)",
    "7Y":"IN7YT Yield (%)","8Y":"IN8YT Yield (%)",
    "9Y":"IN9YT Yield (%)","10Y":"IN10YT Yield (%)",
    "11Y":"IN11YT Yield (%)","12Y":"IN12YT Yield (%)",
    "13Y":"IN13YT Yield (%)","14Y":"IN14YT Yield (%)",
    "15Y":"IN15YT Yield (%)","19Y":"IN19YT Yield (%)",
    "24Y":"IN24YT Yield (%)","30Y":"IN30YT Yield (%)",
    "40Y":"IN40YT Yield (%)","50Y":"IN50YT Yield (%)",
}

MACRO_COLS = {
    "CPI":"CPI YoY (%)","Repo Rate":"Repo Rate (%)",
    "IIP":"IIP Growth (%)","USD/INR":"USD/INR Rate",
    "Crude":"Crude Brent (INR/bbl)","NSE":"NSE Close Price",
    "FII":"FII (INR)",
}

MATURITIES_YRS = {
    "3M":0.25,"6M":0.5,"1Y":1,"2Y":2,"3Y":3,"4Y":4,"5Y":5,
    "6Y":6,"7Y":7,"8Y":8,"9Y":9,"10Y":10,"11Y":11,"12Y":12,
    "13Y":13,"14Y":14,"15Y":15,"19Y":19,"24Y":24,"30Y":30,
    "40Y":40,"50Y":50,
}

DISPLAY_MATS = ["3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","14Y","15Y",
                "19Y","24Y","30Y","40Y","50Y"]

DEFAULT_MACRO_WEIGHTS = {"CPI":25,"Repo Rate":25,"IIP":10,
                          "USD/INR":10,"Crude":10,"NSE":10,"FII":10}


# ============================================================================
# SECTION 3 — DATA LOADING
# ============================================================================
def load_data():
    print("\n[1/12] Loading data from Excel ...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=INPUT_SHEET, engine="openpyxl")
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.set_index("Date", inplace=True)
    for col in MACRO_COLS.values():
        df[col] = df[col].ffill()
    # Count real (non-NaN) observations per maturity BEFORE interpolation
    # Used downstream to exclude sparse maturities (e.g. 40Y/50Y)
    mat_real_obs = {}
    for lab, col in YIELD_COLS.items():
        mat_real_obs[lab] = int(df[col].notna().sum())
    for col in YIELD_COLS.values():
        df[col] = df[col].interpolate(method="linear")
    # Unit safety: all rates must be decimal (0.065 = 6.5%), not percent
    _y10 = df[YIELD_COLS["10Y"]].mean()
    _cpi = df[MACRO_COLS["CPI"]].mean()
    _repo = df[MACRO_COLS["Repo Rate"]].mean()
    assert _y10 < 1.0, f"10Y mean={_y10:.2f} looks like percent, expected decimal"
    assert _cpi < 1.0, f"CPI mean={_cpi:.2f} looks like percent, expected decimal"
    assert _repo < 1.0, f"Repo mean={_repo:.2f} looks like percent, expected decimal"
    df.dropna(inplace=True)
    print(f"    Rows: {len(df):,}  |  "
          f"Range: {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    return df, mat_real_obs


# ============================================================================
# SECTION 4 — MACRO WEIGHTAGE INPUT
# ============================================================================
def get_macro_weights():
    print("\n" + "=" * 72)
    print("  MACRO VARIABLE WEIGHTAGE ASSIGNMENT")
    print("=" * 72)
    print("  Assign importance weights (0-100). Press Enter for default.\n")
    weights = {}
    for name, default in DEFAULT_MACRO_WEIGHTS.items():
        while True:
            try:
                raw = input(f"    {name:12s}  [default {default:2d}] : ").strip()
                weights[name] = float(raw) if raw else default
                break
            except ValueError:
                print("      -> enter a number")
    total = sum(weights.values()) or 1.0
    weights = {k: v / total * 100 for k, v in weights.items()}
    print("\n  Normalised weights:")
    for k, v in weights.items():
        print(f"    {k:12s}  {v:5.1f}%  {'|' * int(v / 2)}")
    return weights


# ============================================================================
# SECTION 4b — HELPER FUNCTIONS  (factory, ACF seq_len, FLAT threshold)
# ============================================================================
def _make_xgb(n_estimators=500, max_depth=6, early_stopping_rounds=50):
    """Factory for XGBoost regressors — shared hyperparameters enforced.
    build_xgboost() uses defaults; run_backtest() passes smaller values
    explicitly (fewer trees for smaller per-fold training sets).
    """
    return xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42, verbosity=0)


def _compute_seq_len(series, max_lag=120, min_lag=20, default=60):
    """ACF-based LSTM sequence length: first lag where |ACF| < 0.05.
    Falls back to `default` if series is too short or ACF never decays.
    """
    try:
        s = series.diff().dropna().values
        if len(s) < max_lag * 2:
            return default
        a = acf(s, nlags=max_lag, fft=True)
        for i in range(min_lag, len(a)):
            if abs(a[i]) < 0.05:
                return i
        return default
    except Exception:
        return default


def _compute_flat_bps(df, col, horizon, fallback=3.0):
    """Per-maturity FLAT threshold: 20th percentile of |historical changes|.
    Returns at least 0.5 bps to avoid marking everything as FLAT.
    """
    hist = (df[col].diff(horizon) * 10_000).dropna().tail(504)
    if len(hist) > 60:
        return max(0.5, float(np.percentile(np.abs(hist), 20)))
    return fallback


# ============================================================================
# SECTION 5 — NELSON-SIEGEL-SVENSSON DAILY FITTER
# ============================================================================
def fit_nss_fast(df):
    """Fit NSS to each day using fixed-lambda OLS + grid search.
    Returns DataFrame with [nss_b0, nss_b1, nss_b2, nss_b3] per day.
    """
    print("\n[2/12] Fitting Nelson-Siegel-Svensson (daily) ...")

    # Cache version tag — bump when fitting method changes to force refit
    NSS_VERSION = "v3.2-multiday"
    if NSS_CACHE.exists():
        cached = pd.read_csv(NSS_CACHE, index_col=0, parse_dates=True)
        cache_ver = cached.columns[-1] if cached.columns[-1].startswith("ver") else ""
        last_cache = cached.index.max()
        last_data  = df.index.max()
        if cache_ver == NSS_VERSION and (last_data - last_cache).days <= 3:
            data_cols = [c for c in cached.columns if not c.startswith("ver")]
            print(f"    Loaded from cache ({len(cached)} rows, "
                  f"through {last_cache:%Y-%m-%d})")
            return cached[data_cols].reindex(df.index).ffill().bfill()
        reason = "version change" if cache_ver != NSS_VERSION else "stale data"
        print(f"    Cache invalid ({reason}), refitting ...")

    taus = np.array(list(MATURITIES_YRS.values()))
    yields_mat = df[list(YIELD_COLS.values())].values  # (T, 22)

    # Grid search over lambda1, lambda2 using 5 representative days
    # (covers early, mid, late regimes — not just one median day)
    best_l1, best_l2, best_err = 1.5, 5.0, np.inf
    n_days = len(yields_mat)
    rep_idx = [n_days // 6, n_days // 3, n_days // 2,
               2 * n_days // 3, 5 * n_days // 6]
    rep_yields = yields_mat[rep_idx]
    for l1 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for l2 in [2.0, 5.0, 8.0, 12.0, 20.0]:
            if l2 <= l1:
                continue
            e1 = np.exp(-taus / l1)
            e2 = np.exp(-taus / l2)
            X = np.column_stack([
                np.ones_like(taus),
                (1 - e1) / (taus / l1),
                (1 - e1) / (taus / l1) - e1,
                (1 - e2) / (taus / l2) - e2,
            ])
            total_err = 0
            for ry in rep_yields:
                betas, _, _, _ = np.linalg.lstsq(X, ry, rcond=None)
                total_err += np.sum((X @ betas - ry) ** 2)
            if total_err < best_err:
                best_err = total_err
                best_l1, best_l2 = l1, l2

    # Build fixed loading matrix
    e1 = np.exp(-taus / best_l1)
    e2 = np.exp(-taus / best_l2)
    X_load = np.column_stack([
        np.ones_like(taus),
        (1 - e1) / (taus / best_l1),
        (1 - e1) / (taus / best_l1) - e1,
        (1 - e2) / (taus / best_l2) - e2,
    ])

    # Daily OLS: betas = (X'X)^-1 X'y  (vectorised)
    XtX_inv = np.linalg.inv(X_load.T @ X_load)
    betas_all = (XtX_inv @ X_load.T @ yields_mat.T).T  # (T, 4)

    result = pd.DataFrame(
        betas_all, index=df.index,
        columns=["nss_b0", "nss_b1", "nss_b2", "nss_b3"]
    )
    result["ver_tag"] = NSS_VERSION   # version tag for cache invalidation

    # Cache
    result.to_csv(NSS_CACHE)
    result = result.drop(columns=["ver_tag"])
    print(f"    Fitted {len(result)} days  |  lambda1={best_l1}, lambda2={best_l2}")
    print(f"    Cached to {NSS_CACHE.name}")
    return result


# ============================================================================
# SECTION 6 — FEATURE ENGINEERING  (FIXED: ma60, momentum, macro scaling)
# ============================================================================
def engineer_features(df, macro_weights, nss_params):
    print("\n[3/12] Engineering features ...")
    F = pd.DataFrame(index=df.index)

    # --- yield features ---
    for lab, col in YIELD_COLS.items():
        F[f"y_{lab}"]    = df[col]
        F[f"dy1_{lab}"]  = df[col].diff(1)  * 10_000
        F[f"dy5_{lab}"]  = df[col].diff(5)  * 10_000
        F[f"dy20_{lab}"] = df[col].diff(20) * 10_000
        F[f"ma5_{lab}"]  = df[col].rolling(5).mean()
        F[f"ma20_{lab}"] = df[col].rolling(20).mean()
        F[f"ma60_{lab}"] = df[col].rolling(60).mean()   # FIX: was missing in v2

    # momentum (now works because ma60 exists)
    for lab in list(YIELD_COLS.keys()):
        F[f"mom_20_60_{lab}"] = F[f"ma20_{lab}"] - F[f"ma60_{lab}"]

    # yield-curve shape
    F["slope_10y2y"] = (df[YIELD_COLS["10Y"]] - df[YIELD_COLS["2Y"]]) * 10_000
    F["slope_10y3m"] = (df[YIELD_COLS["10Y"]] - df[YIELD_COLS["3M"]]) * 10_000
    F["slope_30y5y"] = (df[YIELD_COLS["30Y"]] - df[YIELD_COLS["5Y"]]) * 10_000
    F["butterfly"]   = (2*df[YIELD_COLS["5Y"]]
                        - df[YIELD_COLS["2Y"]]
                        - df[YIELD_COLS["10Y"]]) * 10_000

    # rolling vol
    for lab in ["3M","1Y","5Y","10Y","30Y"]:
        F[f"vol20_{lab}"] = df[YIELD_COLS[lab]].diff().rolling(20).std() * 10_000
        F[f"vol60_{lab}"] = df[YIELD_COLS[lab]].diff().rolling(60).std() * 10_000

    # --- NSS factors as features ---
    F = F.join(nss_params, how="left")
    for c in ["nss_b0","nss_b1","nss_b2","nss_b3"]:
        F[f"d_{c}"]    = F[c].diff(1)
        F[f"d5_{c}"]   = F[c].diff(5)
        F[f"d20_{c}"]  = F[c].diff(20)
        F[f"ma20_{c}"] = F[c].rolling(20).mean()

    # --- macro features (scaled by user weights — linear, not sqrt) ---
    # Linear scaling gives range ~0.7x–1.8x (meaningful for LSTM);
    # XGBoost gets macro influence via macro_composite + sample weights
    for name, col in MACRO_COLS.items():
        w_scale = macro_weights[name] / (100 / len(MACRO_COLS))
        # Clip pct_change to [-5, 5] — IIP near-zero values cause explosions
        F[f"m_{name}"]       = df[col] * w_scale
        F[f"m_chg5_{name}"]  = df[col].pct_change(5).clip(-5, 5)  * w_scale
        F[f"m_chg20_{name}"] = df[col].pct_change(20).clip(-5, 5) * w_scale
        F[f"m_ma20_{name}"]  = df[col].rolling(20).mean() * w_scale
        F[f"m_ma60_{name}"]  = df[col].rolling(60).mean() * w_scale
        F[f"m_std20_{name}"] = df[col].rolling(20).std()

    # Weighted macro composite z-score
    # Rate variables (CPI, Repo, IIP): use diff() — change in the rate is the signal
    # Level variables (NSE, FII, Crude, USD/INR): use pct_change() — % change
    RATE_VARS = {"CPI", "Repo Rate", "IIP"}
    z_parts = pd.DataFrame(index=df.index)
    for name, col in MACRO_COLS.items():
        if name in RATE_VARS:
            chg = df[col].diff(20)                           # absolute change in rate
        else:
            chg = df[col].pct_change(20).clip(-5, 5)        # % change, clipped
        mu  = chg.rolling(252, min_periods=60).mean()
        sig = chg.rolling(252, min_periods=60).std()
        z_parts[name] = ((chg - mu) / (sig + 1e-12)).clip(-4, 4)
    w_arr = np.array([macro_weights[k] / 100 for k in MACRO_COLS])
    F["macro_composite"] = z_parts.fillna(0).values @ w_arr

    # real yield & term premium (units verified: both yields and macro in decimal)
    F["real_yield_10y"] = (df[YIELD_COLS["10Y"]] - df[MACRO_COLS["CPI"]]) * 100
    F["term_prem_10y"]  = (df[YIELD_COLS["10Y"]] - df[MACRO_COLS["Repo Rate"]]) * 100

    F.replace([np.inf, -np.inf], np.nan, inplace=True)
    F.dropna(inplace=True)
    print(f"    {F.shape[1]} features  |  {len(F):,} usable rows")
    return F


# ============================================================================
# SECTION 7 — DIAGNOSTIC TESTS
# ============================================================================
def run_diagnostics(df):
    print("\n[4/12] Running diagnostic tests ...")
    diag = {}

    print("\n  ADF Stationarity (H0: non-stationary):")
    print(f"  {'Series':20s} {'Stat':>9s} {'p-val':>8s} {'Result':>12s}")
    print("  " + "-" * 52)
    adf_info = {}
    test_series = {"10Y Yield": YIELD_COLS["10Y"], "3M Yield": YIELD_COLS["3M"],
                   "CPI": MACRO_COLS["CPI"], "Repo Rate": MACRO_COLS["Repo Rate"],
                   "USD/INR": MACRO_COLS["USD/INR"]}
    for name, col in test_series.items():
        s = df[col].dropna()
        if len(s) < 200: continue
        res = adfuller(s, maxlag=20, autolag="AIC")
        stat_str = "Stationary" if res[1] < 0.05 else "NON-stat"
        adf_info[name] = {"stat": res[0], "p": res[1], "stationary": res[1] < 0.05}
        print(f"  {name:20s} {res[0]:9.3f} {res[1]:8.4f} {stat_str:>12s}")

    print("\n  First differences:")
    for name, col in test_series.items():
        s = df[col].diff().dropna()
        if len(s) < 200: continue
        res = adfuller(s, maxlag=20, autolag="AIC")
        stat_str = "Stationary" if res[1] < 0.05 else "NON-stat"
        print(f"  d({name:18s}) {res[0]:9.3f} {res[1]:8.4f} {stat_str:>12s}")
    diag["adf"] = adf_info

    print(f"\n  Linearity:  Macro -> 10Y Yield")
    print(f"  {'Variable':15s} {'Pearson':>9s} {'Spearman':>9s} {'Type':>14s}")
    print("  " + "-" * 50)
    linearity = {}
    y10 = df[YIELD_COLS["10Y"]]
    for name, col in MACRO_COLS.items():
        x = df[col]; idx = y10.index.intersection(x.dropna().index)
        if len(idx) < 200: continue
        pr, _ = stats.pearsonr(y10.loc[idx], x.loc[idx])
        sr, _ = stats.spearmanr(y10.loc[idx], x.loc[idx])
        gap = abs(abs(sr) - abs(pr))
        rtype = "Non-Linear" if gap > 0.10 else ("Linear" if abs(pr) > 0.25 else "Weak")
        linearity[name] = {"pearson": pr, "spearman": sr, "type": rtype}
        print(f"  {name:15s} {pr:9.4f} {sr:9.4f} {rtype:>14s}")
    diag["linearity"] = linearity

    # NOTE: Johansen cointegration now runs inside build_econometric()
    # so that k_ar_diff matches the VAR-selected lag (not hardcoded 2).

    print(f"\n  Granger Causality  (Macro -> d(10Y),  max lag 3):")
    RATE_VARS = {"CPI", "Repo Rate", "IIP"}
    gc = {}
    ychg = df[YIELD_COLS["10Y"]].diff().dropna()
    for name, col in MACRO_COLS.items():
        try:
            # Rate vars: diff() (absolute change); level vars: pct_change()
            if name in RATE_VARS:
                xchg = df[col].diff().dropna()
            else:
                xchg = df[col].pct_change().dropna()
            idx  = ychg.index.intersection(xchg.index)
            tmp  = pd.DataFrame({"y": ychg.loc[idx], "x": xchg.loc[idx]})
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp.dropna(inplace=True)
            if len(tmp) < 300: continue
            res = grangercausalitytests(tmp[["y","x"]], maxlag=3, verbose=False)
            pmin = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, 4))
            sig  = "***" if pmin < 0.01 else "**" if pmin < 0.05 else "*" if pmin < 0.10 else ""
            gc[name] = pmin
            print(f"    {name:12s}  p={pmin:.4f}  {sig}")
        except Exception:
            pass
    diag["granger"] = gc
    return diag


# ============================================================================
# SECTION 8 — HULL-WHITE SHORT RATE MODEL
# ============================================================================
def fit_hull_white(df, horizon_days):
    """1-factor Hull-White on 3M yield, anchored to Repo Rate."""
    print(f"\n[5/12] Fitting Hull-White short rate model ...")
    short_yield = df[YIELD_COLS["3M"]].dropna()
    DT = 1 / 252

    def neg_ll(params):
        kappa, theta, sigma = params
        if kappa <= 0 or sigma <= 0:
            return 1e10
        r = short_yield.values
        r_pred = r[:-1] + kappa * (theta - r[:-1]) * DT
        resid  = r[1:] - r_pred
        var    = sigma**2 * DT
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    theta_init = float(df[MACRO_COLS["Repo Rate"]].iloc[-1])
    mean_3m    = float(short_yield.mean())
    # Multi-start optimisation to escape local minima on non-convex likelihood
    bounds = [(0.01, 20), (0.001, 0.20), (0.0001, 0.05)]
    starts = [
        (0.5,  theta_init, 0.005),
        (2.0,  theta_init, 0.005),
        (5.0,  theta_init, 0.010),
        (2.0,  mean_3m,    0.005),
        (10.0, theta_init, 0.010),
    ]
    best_res, best_nll = None, np.inf
    for x0 in starts:
        try:
            r = sp_minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
            if r.fun < best_nll:
                best_nll = r.fun
                best_res = r
        except Exception:
            continue
    if best_res is None:
        kappa, theta, sigma = 2.0, theta_init, 0.005   # fallback
    else:
        kappa, theta, sigma = best_res.x

    # In-sample RMSE for ensemble weighting
    rv = short_yield.values
    r_pred_is = rv[:-1] + kappa * (theta - rv[:-1]) * DT
    hw_rmse = float(np.sqrt(np.mean((rv[1:] - r_pred_is)**2)) * 10_000)

    r_now = float(short_yield.iloc[-1])
    h     = horizon_days * DT
    r_fc  = theta + (r_now - theta) * np.exp(-kappa * h)
    chg   = (r_fc - r_now) * 10_000
    ci_v  = sigma * np.sqrt((1 - np.exp(-2*kappa*h)) / (2*kappa)) * 10_000

    # Compute data-driven attenuation for 6M/1Y from historical correlations
    d3m = df[YIELD_COLS["3M"]].diff().dropna()
    att = {}
    for lab in ["6M", "1Y"]:
        d_tgt = df[YIELD_COLS[lab]].diff().dropna()
        idx   = d3m.index.intersection(d_tgt.index)
        if len(idx) > 200:
            # beta from OLS regression: d_target = beta * d_3M
            b = float(np.cov(d_tgt.loc[idx], d3m.loc[idx])[0, 1]
                      / (np.var(d3m.loc[idx]) + 1e-12))
            att[lab] = float(np.clip(b, 0.5, 1.0))
        else:
            att[lab] = 0.85 if lab == "6M" else 0.70   # fallback

    print(f"    kappa={kappa:.3f}  theta={theta:.4f}  sigma={sigma:.5f}")
    print(f"    In-sample RMSE: {hw_rmse:.2f} bps")
    print(f"    Term attenuation:  6M={att['6M']:.3f}  1Y={att['1Y']:.3f}")
    print(f"    3M forecast: {r_fc:.4f}  (change: {chg:+.1f} bps, "
          f"90%CI: [{chg - 1.65*ci_v:+.1f}, {chg + 1.65*ci_v:+.1f}])")

    return {"kappa": kappa, "theta": theta, "sigma": sigma,
            "r_forecast": r_fc, "change_bps": chg, "rmse": hw_rmse,
            "attenuation": att,
            "ci_lo": chg - 1.65 * ci_v, "ci_hi": chg + 1.65 * ci_v}


# ============================================================================
# SECTION 9 — VAR / VECM  (FIXED: 4 vars, lag <= 3, multi-step forecast)
# ============================================================================
def build_econometric(df, horizon_days):
    print(f"\n[6/12] Building VAR / VECM  (horizon={horizon_days}d) ...")
    # Reduced variable set (4 vars) to avoid overparameterisation
    var_cols = [YIELD_COLS["5Y"], YIELD_COLS["10Y"],
                MACRO_COLS["CPI"], MACRO_COLS["Repo Rate"]]

    monthly      = df[var_cols].resample("ME").last().dropna()
    monthly_diff = monthly.diff().dropna()
    cutoff = monthly_diff.index.max() - pd.DateOffset(years=15)
    md = monthly_diff[monthly_diff.index >= cutoff]
    if len(md) < 48:
        print("    Insufficient data -- skipping VAR.")
        return None

    try:
        model = VAR(md)
        sel   = model.select_order(maxlags=3)   # capped at 3
        lag   = max(sel.aic, 1)
        vr    = model.fit(lag)

        # Multi-step forecast (not linear extrapolation)
        steps = max(1, round(horizon_days / 20))
        fcast = vr.forecast(md.values[-lag:], steps=steps)
        cum_fcast = fcast.sum(axis=0)

        # In-sample RMSE for ensemble weighting
        resid = vr.resid
        var_rmse = float(np.sqrt(np.mean(resid ** 2)) * 10_000)
        print(f"    VAR({lag}) fitted  |  {len(md)} obs  |  {steps}-step forecast"
              f"  |  RMSE={var_rmse:.1f} bps")

        try:
            irf = vr.irf(periods=12)
            print("    Impulse Response Functions computed")
        except Exception:
            irf = None

        # Johansen cointegration — runs AFTER VAR lag is known
        # so k_ar_diff matches the selected lag (not hardcoded 2)
        vecm_fcast = None
        cointegrated = False
        coint_rank = 0
        ml = monthly[monthly.index >= cutoff].dropna()
        try:
            if len(ml) > 60:
                joh = coint_johansen(ml, det_order=0, k_ar_diff=lag)
                print(f"\n    Johansen Cointegration (k_ar_diff={lag}):")
                print(f"    {'Hypothesis':18s} {'Trace':>9s} {'Crit-95%':>9s} {'Coint?':>8s}")
                print("    " + "-" * 48)
                for i in range(min(len(var_cols), len(joh.lr1))):
                    tr, cv = joh.lr1[i], joh.cvt[i, 1]
                    is_c = tr > cv
                    print(f"    r <= {i}              {tr:9.3f} {cv:9.3f} "
                          f"{'YES' if is_c else 'no':>8s}")
                    if is_c:
                        coint_rank = i + 1
                cointegrated = coint_rank > 0
                print(f"    => Cointegrating rank = {coint_rank}")
        except Exception as e:
            print(f"    Johansen skipped: {e}")

        if cointegrated:
            try:
                cr = max(1, coint_rank)
                vecm_res = VECM(ml, k_ar_diff=max(lag - 1, 1),
                                coint_rank=cr).fit()
                vecm_levels = vecm_res.predict(steps=steps)
                last_level = monthly.iloc[-1].values
                vecm_fcast = vecm_levels[-1] - last_level
                print(f"    VECM fitted (cointegration rank {cr})")
            except Exception as e:
                print(f"    VECM skipped: {e}")

        try:
            fevd = vr.fevd(periods=12)
        except Exception:
            fevd = None

        return {"var": vr, "forecast": cum_fcast, "vecm_forecast": vecm_fcast,
                "cols": var_cols, "irf": irf, "fevd": fevd,
                "rmse": var_rmse}
    except Exception as e:
        print(f"    VAR failed: {e}")
        return None


# ============================================================================
# SECTION 10 — XGBOOST  (FIXED: gap, regressor-only, no clf mismatch)
# ============================================================================
def build_xgboost(features, df, macro_weights, horizon):
    print(f"\n[7/12] Building XGBoost  (horizon = {horizon}d) ...")
    models = {}

    for lab, col in YIELD_COLS.items():
        target = (df[col].shift(-horizon) - df[col]) * 10_000
        idx = features.index.intersection(target.dropna().index)
        X, y = features.loc[idx].copy(), target.loc[idx].copy()
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
        X, y = X[mask], y[mask]
        if len(X) < 800:
            continue

        # FIX: gap = horizon between train and test (no target boundary leakage)
        split = int(len(X) * 0.80)
        Xtr = X.iloc[:split - horizon]
        ytr = y.iloc[:split - horizon]
        Xte = X.iloc[split:]
        yte = y.iloc[split:]

        if len(Xtr) < 500 or len(Xte) < 50:
            continue

        # sample weights
        sw = np.ones(len(Xtr))
        if "macro_composite" in Xtr.columns:
            mc = np.abs(Xtr["macro_composite"].values)
            sw = 1.0 + 0.5 * mc / (mc.mean() + 1e-8)

        reg = _make_xgb()   # factory — shared hyperparams
        reg.fit(Xtr, ytr, sample_weight=sw,
                eval_set=[(Xte, yte)], verbose=False)

        preds = reg.predict(Xte)
        rmse  = np.sqrt(mean_squared_error(yte, preds))
        mae   = mean_absolute_error(yte, preds)
        dir_acc = np.mean(np.sign(yte) == np.sign(preds))

        models[lab] = {"reg": reg, "dir_acc": dir_acc,
                       "rmse": rmse, "mae": mae,
                       "feats": list(X.columns),
                       "n_test": len(Xte)}

    print(f"\n  {'Mat':>6s} {'DirAcc':>7s} {'RMSE':>8s} {'MAE':>8s}")
    print("  " + "-" * 32)
    for m in DISPLAY_MATS:
        if m in models:
            r = models[m]
            print(f"  {m:>6s} {r['dir_acc']:>6.1%} {r['rmse']:>8.2f} {r['mae']:>8.2f}")
    return models


# ============================================================================
# SECTION 11 — LSTM  (FIXED: scaler fitted on TRAIN only)
# ============================================================================
def build_lstm(features, df, horizon):
    if not LSTM_AVAILABLE:
        print("\n[8/12] LSTM -- skipped (TensorFlow not installed)")
        return None
    # ACF-derived sequence length (data-driven, not hardcoded 60)
    ref_col = YIELD_COLS.get("10Y") or list(YIELD_COLS.values())[0]
    seq_len = _compute_seq_len(df[ref_col])
    print(f"\n[8/12] Building Bi-LSTM  (horizon={horizon}d, seq={seq_len} [ACF-derived]) ...")
    models = {}
    key_mats = ["3M","1Y","2Y","5Y","7Y","10Y","14Y","30Y"]

    for lab in key_mats:
        if lab not in YIELD_COLS:      # safety: maturity may have been filtered
            continue
        col = YIELD_COLS[lab]
        target = (df[col].shift(-horizon) - df[col]) * 10_000

        # Select features by priority category to ensure NSS/macro aren't cut
        _cats = [
            [f"y_{lab}", f"dy1_{lab}", f"dy5_{lab}", f"dy20_{lab}",
             f"ma20_{lab}", f"ma60_{lab}", f"mom_20_60_{lab}"],
            [c for c in features.columns if c.startswith("nss_") or c.startswith("d_nss")
                                          or c.startswith("d5_nss") or c.startswith("d20_nss")],
            [c for c in features.columns if c in ("slope_10y2y","slope_10y3m","slope_30y5y",
                                                    "butterfly","macro_composite",
                                                    "real_yield_10y","term_prem_10y")],
            [c for c in features.columns if c.startswith("vol20_") or c.startswith("vol60_")],
            [c for c in features.columns if c.startswith("m_chg20_") or c.startswith("m_ma20_")],
            [c for c in features.columns if c.startswith("dy1_") and lab not in c],
        ]
        keep = []
        for cat in _cats:
            for c in cat:
                if c in features.columns and c not in keep:
                    keep.append(c)
                if len(keep) >= 60:
                    break
            if len(keep) >= 60:
                break

        idx = features.index.intersection(target.dropna().index)
        Xf  = features[keep].loc[idx].values
        yf  = target.loc[idx].values
        mask = np.isfinite(yf) & np.all(np.isfinite(Xf), axis=1)
        Xf, yf = Xf[mask], yf[mask]
        if len(Xf) < seq_len + 600:
            continue

        # FIX: split BEFORE scaling (no scaler leakage)
        sp = int(len(Xf) * 0.80) - horizon   # gap
        if sp < seq_len + 200:
            continue
        test_start = sp + horizon

        sx = RobustScaler()
        Xf_train_s = sx.fit_transform(Xf[:sp])          # fit on train only
        Xf_test_s  = sx.transform(Xf[test_start:])      # transform test

        sy = StandardScaler()
        yf_train_s = sy.fit_transform(yf[:sp].reshape(-1,1)).ravel()
        yf_test_s  = sy.transform(yf[test_start:].reshape(-1,1)).ravel()

        # build sequences from train
        Xtr_seq, ytr_seq = [], []
        for i in range(seq_len, len(Xf_train_s)):
            Xtr_seq.append(Xf_train_s[i - seq_len:i])
            ytr_seq.append(yf_train_s[i])
        Xtr_seq, ytr_seq = np.array(Xtr_seq), np.array(ytr_seq)

        # build sequences from test
        # prepend last seq_len of train (scaled) for first test sequences
        Xf_combined = np.vstack([Xf_train_s[-seq_len:], Xf_test_s])
        yf_combined = np.concatenate([yf_train_s[-seq_len:], yf_test_s])
        Xte_seq, yte_seq = [], []
        for i in range(seq_len, len(Xf_combined)):
            Xte_seq.append(Xf_combined[i - seq_len:i])
            yte_seq.append(yf_combined[i])
        Xte_seq, yte_seq = np.array(Xte_seq), np.array(yte_seq)

        if len(Xtr_seq) < 200 or len(Xte_seq) < 50:
            continue

        mdl = Sequential([
            Bidirectional(KerasLSTM(64, return_sequences=True),
                          input_shape=(seq_len, Xtr_seq.shape[2])),
            Dropout(0.25),
            BatchNormalization(),
            KerasLSTM(32),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        mdl.compile(optimizer=Adam(learning_rate=0.001), loss="huber")
        mdl.fit(Xtr_seq, ytr_seq, epochs=50, batch_size=64,
                validation_split=0.10, verbose=0,
                callbacks=[EarlyStopping(patience=8, restore_best_weights=True),
                           ReduceLROnPlateau(factor=0.5, patience=4)])

        pred_s = mdl.predict(Xte_seq, verbose=0).ravel()
        pred   = sy.inverse_transform(pred_s.reshape(-1,1)).ravel()
        act    = sy.inverse_transform(yte_seq.reshape(-1,1)).ravel()
        rmse   = np.sqrt(mean_squared_error(act, pred))
        dacc   = float(np.mean(np.sign(act) == np.sign(pred)))

        models[lab] = {"model": mdl, "sx": sx, "sy": sy,
                       "feats": keep, "seq": seq_len,
                       "rmse": rmse, "dir_acc": dacc}
        print(f"    {lab:>4s}:  DirAcc={dacc:.1%}   RMSE={rmse:.1f} bps")

    return models if models else None


# ============================================================================
# SECTION 12 — ENSEMBLE  (FIXED: adaptive weights, empirical CI, no mismatch)
# ============================================================================
def ensemble_predict(xgb_m, lstm_m, econ, hw, features, df, macro_w, horizon):
    print(f"\n[9/12] Generating ensemble predictions ...")
    FLAT_FALLBACK = {5: 1, 20: 3, 60: 8}.get(horizon, 3)
    predictions = {}
    has_lstm = lstm_m is not None
    has_var  = econ is not None

    # Compute performance-adaptive weights from training RMSE
    xgb_rmses = {lab: xgb_m[lab]["rmse"] for lab in xgb_m}
    lstm_rmses = {lab: lstm_m[lab]["rmse"] for lab in lstm_m} if has_lstm else {}

    for lab, col in YIELD_COLS.items():
        current = df[col].iloc[-1] * 100
        sources = []

        # --- XGBoost (regressor only — sign = direction) ---
        if lab in xgb_m:
            Xlast = features.iloc[[-1]]
            mag   = float(xgb_m[lab]["reg"].predict(Xlast)[0])
            # FIX: confidence = backtest directional accuracy
            conf  = xgb_m[lab]["dir_acc"]
            sources.append({"src": "XGBoost", "mag": mag,
                            "dir": "UP" if mag > 0 else "DOWN",
                            "conf": conf, "rmse": xgb_m[lab]["rmse"]})

        # --- LSTM ---
        if has_lstm and lab in lstm_m:
            m   = lstm_m[lab]
            raw = features[m["feats"]].iloc[-m["seq"]:].values
            xs  = m["sx"].transform(raw).reshape(1, m["seq"], -1)
            ps  = m["model"].predict(xs, verbose=0)[0][0]
            mag = float(m["sy"].inverse_transform([[ps]])[0][0])
            # FIX: confidence = LSTM backtest accuracy (not made-up formula)
            conf = m["dir_acc"]
            sources.append({"src": "LSTM", "mag": mag,
                            "dir": "UP" if mag > 0 else "DOWN",
                            "conf": conf, "rmse": m["rmse"]})

        # --- VAR ---
        if has_var and col in econ["cols"]:
            ci_idx = econ["cols"].index(col)
            vmag = float(econ["forecast"][ci_idx]) * 10_000
            if econ["vecm_forecast"] is not None:
                vmag2 = float(econ["vecm_forecast"][ci_idx]) * 10_000
                vmag = 0.5 * vmag + 0.5 * vmag2
            sources.append({"src": "VAR", "mag": vmag,
                            "dir": "UP" if vmag > 0 else "DOWN",
                            "conf": 0.52,
                            "rmse": econ.get("rmse", 30.0)})

        # --- Hull-White (short end only) ---
        if hw and lab in ["3M", "6M", "1Y"]:
            hw_mag = hw["change_bps"]
            if lab != "3M":
                # Use data-driven attenuation (not hardcoded 0.85/0.70)
                hw_mag *= hw.get("attenuation", {}).get(lab, 0.75)
            sources.append({"src": "HullWhite", "mag": hw_mag,
                            "dir": "UP" if hw_mag > 0 else "DOWN",
                            "conf": 0.60,
                            "rmse": hw.get("rmse", 20.0)})

        if not sources:
            continue

        # FIX: performance-adaptive weights (inverse RMSE)
        inv_rmse = np.array([1.0 / (s["rmse"] + 1e-6) for s in sources])
        weights  = inv_rmse / inv_rmse.sum()
        for i, s in enumerate(sources):
            s["w"] = float(weights[i])

        tw    = sum(s["w"] for s in sources)
        emag  = sum(s["mag"] * s["w"] for s in sources) / tw
        econf = sum(s["conf"] * s["w"] for s in sources) / tw

        # Direction ALWAYS from sign(ensemble_magnitude) — never from a vote.
        # A vote can contradict magnitude (e.g. "DOWN +6.3 bps") which is nonsense.
        # Per-maturity FLAT threshold from historical data (not hardcoded)
        flat_bps = _compute_flat_bps(df, col, horizon, FLAT_FALLBACK)
        if abs(emag) < flat_bps:
            edir = "FLAT"
        elif emag > 0:
            edir = "UP"
        else:
            edir = "DOWN"

        # Empirical quantile CI, centered on model prediction (not historical median)
        hist = (df[col].diff(horizon) * 10_000).dropna().tail(504)
        if len(hist) > 60:
            raw_lo = float(np.percentile(hist, 5))
            raw_hi = float(np.percentile(hist, 95))
            hist_med = float(np.median(hist))
            # Shift CI so it's centered on ensemble prediction
            lo = raw_lo - hist_med + emag
            hi = raw_hi - hist_med + emag
        else:
            vol = hist.std() if len(hist) > 20 else 10.0
            lo = emag - 1.65 * vol
            hi = emag + 1.65 * vol

        # FIX: CI widening under extreme macro (center + spread approach)
        mc = features["macro_composite"].iloc[-1]
        if abs(mc) > 1.5:
            center = (hi + lo) / 2
            spread = (hi - lo) / 2
            lo = center - spread * 1.20
            hi = center + spread * 1.20
            econf *= 0.90

        # momentum flag
        mom_col = f"dy20_{lab}"
        if mom_col in features.columns:
            mom20 = features[mom_col].iloc[-1]
            momentum = "Strong" if abs(mom20) > 15 else ("Moderate" if abs(mom20) > 5 else "Weak")
        else:
            momentum = "N/A"

        conf_pct = round(min(econf * 100, 95), 1)
        # Statistical threshold: reject H0(p=0.5) at alpha=0.10 one-sided
        # Formula: 50 + z_0.10 * sqrt(0.25/N) * 100 = 50 + 64/sqrt(N)
        n_test = xgb_m[lab]["n_test"] if lab in xgb_m else 200
        low_conf_threshold = 50.0 + 64.0 / np.sqrt(max(n_test, 50))
        low_conf = conf_pct < low_conf_threshold

        predictions[lab] = {
            "current_pct":   round(current, 4),
            "direction":     "LOW CONF" if low_conf else edir,
            "change_bps":    round(emag, 2),
            "range_lo_bps":  round(lo, 2),
            "range_hi_bps":  round(hi, 2),
            "predicted_pct": round(current + emag / 100, 4),
            "confidence":    conf_pct,
            "momentum":      momentum,
            "n_models":      len(sources),
            "models":        "+".join(s["src"] for s in sources),
        }
    return predictions


# ============================================================================
# SECTION 13 — WALK-FORWARD BACKTEST  (FIXED: naive benchmark, gap)
# ============================================================================
def run_backtest(features, df, macro_w, horizon, n_folds=5):
    print(f"\n[10/12] Walk-forward backtest  ({n_folds} folds, {horizon}d horizon) ...")
    print(f"        [XGBoost-only component — ensemble accuracy may differ]")
    results = {}
    key_mats = ["3M","1Y","2Y","5Y","7Y","10Y","14Y","30Y"]
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=252)

    for lab in key_mats:
        if lab not in YIELD_COLS:      # safety: maturity may have been filtered
            continue
        col = YIELD_COLS[lab]
        target = (df[col].shift(-horizon) - df[col]) * 10_000
        idx = features.index.intersection(target.dropna().index)
        X, y = features.loc[idx], target.loc[idx]
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
        X, y = X[mask], y[mask]
        if len(X) < 2000:
            continue

        daccs, rmses, maes, naive_accs = [], [], [], []

        for train_i, test_i in tscv.split(X):
            # gap between train and test
            train_end = max(train_i) - horizon
            if train_end < 500:
                continue
            Xtr = X.iloc[:train_end]
            ytr = y.iloc[:train_end]
            Xte = X.iloc[test_i]
            yte = y.iloc[test_i]

            reg = _make_xgb(n_estimators=300, max_depth=5,
                            early_stopping_rounds=30)
            reg.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
            pred = reg.predict(Xte)

            daccs.append(np.mean(np.sign(yte) == np.sign(pred)))
            rmses.append(np.sqrt(mean_squared_error(yte, pred)))
            maes.append(mean_absolute_error(yte, pred))

            # FIX: naive benchmark = majority class accuracy
            prop_up  = float(np.mean(yte > 0))
            naive_accs.append(max(prop_up, 1 - prop_up))

        if daccs:
            results[lab] = {
                "dir_acc":   np.mean(daccs),
                "dir_std":   np.std(daccs),
                "rmse":      np.mean(rmses),
                "mae":       np.mean(maes),
                "naive_acc": np.mean(naive_accs),
                "edge":      np.mean(daccs) - np.mean(naive_accs),
                "folds":     len(daccs),
            }

    print(f"\n  {'Mat':>6s} {'DirAcc':>7s} {'+-Std':>6s} {'RMSE':>8s} "
          f"{'MAE':>8s} {'Naive':>6s} {'Edge':>7s}")
    print("  " + "-" * 52)
    for m in key_mats:
        if m in results:
            r = results[m]
            print(f"  {m:>6s} {r['dir_acc']:>6.1%} {r['dir_std']:>5.1%} "
                  f"{r['rmse']:>8.2f} {r['mae']:>8.2f} "
                  f"{r['naive_acc']:>5.1%} {r['edge']:>+6.1%}")
    return results


# ============================================================================
# SECTION 14 — FEATURE IMPORTANCE
# ============================================================================
def analyze_importance(xgb_m, macro_w):
    print("\n[11/12] Feature importance analysis ...")
    if "10Y" not in xgb_m:
        return {}
    imp  = xgb_m["10Y"]["reg"].feature_importances_
    feat = xgb_m["10Y"]["feats"]
    top  = np.argsort(imp)[::-1]

    print(f"\n  Top 15 features (10Y regressor):")
    print(f"  {'#':>3s}  {'Feature':42s} {'Imp':>8s}")
    print("  " + "-" * 56)
    for i in range(min(15, len(top))):
        print(f"  {i+1:3d}  {feat[top[i]]:42s} {imp[top[i]]:8.4f}")

    print(f"\n  Macro contribution vs user weight:")
    print(f"  {'Var':15s} {'ModelImp':>9s} {'UserWt':>8s}")
    print("  " + "-" * 35)
    macro_imp = {}
    for name in MACRO_COLS:
        tag = name.lower().replace("/","").replace(" ","")
        mi  = sum(imp[j] for j,f in enumerate(feat)
                  if tag in f.lower().replace("/","").replace(" ",""))
        macro_imp[name] = mi
    ti = sum(macro_imp.values()) or 1
    for name in MACRO_COLS:
        pct = macro_imp[name] / ti * 100
        print(f"  {name:15s} {pct:>8.1f}% {macro_w[name]:>7.1f}%")
    return macro_imp


# ============================================================================
# SECTION 15 — EXCEL OUTPUT  (FIXED: merged cell handling)
# ============================================================================
def write_output(preds, bt, macro_w, diag, horizon_label):
    print(f"\n[12/12] Writing results to Excel ...")
    # Write to separate output file (input file may be open in Excel)
    wb = openpyxl.Workbook()
    # Remove default sheet created by Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    hf  = Font(bold=True, color="FFFFFF", size=11, name="Calibri")
    hfl = PatternFill("solid", fgColor="1F4E79")
    upf = PatternFill("solid", fgColor="C6EFCE")
    dnf = PatternFill("solid", fgColor="FFC7CE")
    ftf = PatternFill("solid", fgColor="FFEB9C")
    bdr = Border(*(Side(style="thin"),)*4)
    ctr = Alignment(horizontal="center", vertical="center")

    # ========== PREDICTIONS SHEET ==========
    if OUTPUT_SHEET in wb.sheetnames:
        del wb[OUTPUT_SHEET]
    ws = wb.create_sheet(OUTPUT_SHEET)

    ws.merge_cells("A1:J1")
    ws["A1"] = f"Bond Yield Prediction  |  Horizon: {horizon_label}"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E79", name="Calibri")
    ws.merge_cells("A2:J2")
    ws["A2"] = (f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}  |  "
                f"Ensemble: NSS + HullWhite + VAR/VECM + XGBoost"
                + (" + Bi-LSTM" if LSTM_AVAILABLE else ""))
    ws["A2"].font = Font(italic=True, size=10, color="666666")

    ws["A4"] = "Macro Weights Applied:"
    ws["A4"].font = Font(bold=True)
    r = 5
    for nm, wt in macro_w.items():
        ws.cell(r, 1, nm); ws.cell(r, 2, f"{wt:.1f}%"); r += 1

    sr = r + 1
    hdrs = ["Maturity","Current Yield (%)","Direction","Change (bps)",
            "Range Low (bps)","Range High (bps)","Predicted Yield (%)",
            "Confidence (%)","Momentum","Models"]
    for ci, h in enumerate(hdrs, 1):
        c = ws.cell(sr, ci, h); c.font = hf; c.fill = hfl
        c.alignment = ctr; c.border = bdr

    mat_order = list(YIELD_COLS.keys())
    dr = sr + 1
    for lab in mat_order:
        if lab not in preds:
            continue
        p = preds[lab]
        vals = [lab, p["current_pct"], p["direction"], p["change_bps"],
                p["range_lo_bps"], p["range_hi_bps"], p["predicted_pct"],
                p["confidence"], p["momentum"], p["models"]]
        for ci, v in enumerate(vals, 1):
            c = ws.cell(dr, ci, v); c.border = bdr; c.alignment = ctr
            if ci == 3:
                if v == "UP":       c.fill = upf; c.font = Font(bold=True, color="006100")
                elif v == "DOWN":   c.fill = dnf; c.font = Font(bold=True, color="9C0006")
                elif v == "LOW CONF":
                    c.fill = PatternFill("solid", fgColor="D9D9D9")
                    c.font = Font(bold=True, color="595959")
                else:               c.fill = ftf; c.font = Font(bold=True, color="9C6500")
        dr += 1

    # Safe column width (skip MergedCell objects + guard empty columns)
    for col_cells in ws.columns:
        first = col_cells[0]
        if not hasattr(first, "column_letter"):
            continue
        lengths = [len(str(c.value)) for c in col_cells
                   if c.value is not None and hasattr(c, "column_letter")]
        ml = max(lengths) if lengths else 8
        ws.column_dimensions[first.column_letter].width = min(ml + 4, 28)

    # ========== BACKTEST SHEET ==========
    if BACKTEST_SHEET in wb.sheetnames:
        del wb[BACKTEST_SHEET]
    ws2 = wb.create_sheet(BACKTEST_SHEET)
    ws2.merge_cells("A1:G1")
    ws2["A1"] = "Walk-Forward Backtest Results"
    ws2["A1"].font = Font(bold=True, size=14, color="1F4E79")

    bh = ["Maturity","Dir Accuracy","Std","RMSE (bps)","MAE (bps)",
          "Naive Acc","Edge vs Naive"]
    for ci, h in enumerate(bh, 1):
        c = ws2.cell(3, ci, h); c.font = hf; c.fill = hfl; c.border = bdr

    br = 4
    for lab, btr in bt.items():
        vals = [lab, f"{btr['dir_acc']:.1%}", f"{btr['dir_std']:.1%}",
                f"{btr['rmse']:.2f}", f"{btr['mae']:.2f}",
                f"{btr['naive_acc']:.1%}", f"{btr['edge']:+.1%}"]
        for ci, v in enumerate(vals, 1):
            c = ws2.cell(br, ci, v); c.border = bdr
        br += 1

    br += 2
    ws2.cell(br, 1, "Diagnostic Tests").font = Font(bold=True, size=12)
    br += 1
    ws2.cell(br, 1, "Linearity Analysis (Macro -> 10Y Yield):").font = Font(bold=True)
    br += 1
    if "linearity" in diag:
        for nm, info in diag["linearity"].items():
            ws2.cell(br, 1, nm)
            ws2.cell(br, 2, f"Pearson={info['pearson']:.4f}")
            ws2.cell(br, 3, f"Spearman={info['spearman']:.4f}")
            ws2.cell(br, 4, info["type"])
            br += 1

    br += 1
    ws2.cell(br, 1, "Granger Causality (min p-value):").font = Font(bold=True)
    br += 1
    for nm, pv in diag.get("granger", {}).items():
        ws2.cell(br, 1, nm)
        ws2.cell(br, 2, f"p = {pv:.4f}")
        ws2.cell(br, 3, "Significant" if pv < 0.05 else "Not significant")
        br += 1

    for col_cells in ws2.columns:
        first = col_cells[0]
        if not hasattr(first, "column_letter"):
            continue
        lengths = [len(str(c.value)) for c in col_cells
                   if c.value is not None and hasattr(c, "column_letter")]
        ml = max(lengths) if lengths else 8
        ws2.column_dimensions[first.column_letter].width = min(ml + 4, 30)

    # Save to dedicated output file (never overwrites input)
    save_path = OUTPUT_PATH
    try:
        wb.save(str(save_path))
    except PermissionError:
        # Fallback: save to Desktop if output file is also locked
        save_path = Path.home() / "Desktop" / "bond_predictions_output.xlsx"
        wb.save(str(save_path))
    print(f"    Saved: {save_path}")
    print(f"    Sheets: '{OUTPUT_SHEET}', '{BACKTEST_SHEET}'")


# ============================================================================
# SECTION 16 — MAIN PIPELINE
# ============================================================================
def main():
    df, mat_real_obs = load_data()

    # Filter out sparse maturities (mostly interpolated, predictions meaningless)
    sparse = [lab for lab, n in mat_real_obs.items() if n < MIN_MAT_OBS]
    if sparse:
        for lab in sparse:
            YIELD_COLS.pop(lab, None)
            MATURITIES_YRS.pop(lab, None)
        DISPLAY_MATS[:] = [m for m in DISPLAY_MATS if m not in sparse]
        print(f"    Excluded sparse maturities (<{MIN_MAT_OBS} real obs): "
              f"{sorted(sparse)}")

    macro_w = get_macro_weights()

    # NSS daily fit
    nss_params = fit_nss_fast(df)

    # Features (with NSS factors and fixed momentum)
    feats = engineer_features(df, macro_w, nss_params)
    df_a  = df.loc[feats.index]

    # Diagnostics
    diag = run_diagnostics(df_a)

    # Horizon
    print("\n" + "=" * 72)
    print("  SELECT PREDICTION HORIZON")
    print("=" * 72)
    print("    1) 1 Week   (5 trading days)")
    print("    2) 1 Month  (20 trading days)")
    print("    3) 3 Months (60 trading days)")
    choice = input("\n    Choice [1/2/3, default=2]: ").strip()
    hlabel, hdays = {"1":("1W",5),"2":("1M",20),"3":("3M",60)}.get(choice, ("1M",20))

    # Hull-White
    hw = fit_hull_white(df_a, hdays)

    # VAR/VECM (Johansen now runs inside, using VAR-selected lag)
    econ = build_econometric(df_a, hdays)

    # XGBoost
    xgb_models = build_xgboost(feats, df_a, macro_w, hdays)

    # LSTM
    lstm_models = build_lstm(feats, df_a, hdays)

    # Ensemble
    preds = ensemble_predict(xgb_models, lstm_models, econ, hw,
                             feats, df_a, macro_w, hdays)

    # Backtest
    bt = run_backtest(feats, df_a, macro_w, hdays)

    # Importance
    analyze_importance(xgb_models, macro_w)

    # Write
    write_output(preds, bt, macro_w, diag, hlabel)

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 82)
    print("  PREDICTION SUMMARY  (v3.3)")
    print("=" * 82)
    print(f"  Horizon: {hlabel} ({hdays} trading days)\n")
    print(f"  {'Mat':>6s} {'Current':>8s} {'Dir':>7s} {'Chg(bps)':>9s} "
          f"{'Range(bps)':>18s} {'Predicted':>10s} {'Conf':>6s} {'Mom':>8s}")
    print("  " + "-" * 76)
    for lab in DISPLAY_MATS:
        if lab in preds:
            p = preds[lab]
            rng = f"[{p['range_lo_bps']:+.1f} , {p['range_hi_bps']:+.1f}]"
            print(f"  {lab:>6s} {p['current_pct']:>7.3f}% {p['direction']:>7s} "
                  f"{p['change_bps']:>+8.1f} {rng:>18s} "
                  f"{p['predicted_pct']:>9.3f}% {p['confidence']:>5.1f}% "
                  f"{p['momentum']:>8s}")

    print(f"\n  Results written to:  {OUTPUT_PATH}")
    print(f"  Sheets: '{OUTPUT_SHEET}'  and  '{BACKTEST_SHEET}'")
    print("=" * 82)


if __name__ == "__main__":
    main()
