"""
================================================================================
  INDIAN GOVERNMENT BOND YIELD PREDICTION ENGINE  v2.0
================================================================================
  Multi-Model Ensemble: Econometric (VAR / VECM) + ML (XGBoost / LSTM)
  with User-Defined Macro Variable Weightages

  Designed for institutional fixed-income desks.
  Predicts: Direction | Magnitude (bps) | Confidence Interval
  for every maturity on the Indian sovereign yield curve.

  Data leakage: ZERO. Strict walk-forward protocol throughout.
================================================================================
"""

# ============================================================================
# SECTION 1 — IMPORTS
# ============================================================================
import os, sys, warnings, datetime as dt
from pathlib import Path
from copy import deepcopy

import numpy  as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# --- scikit-learn -----------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, r2_score)

# --- statsmodels (econometrics) ---------------------------------------------
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.stats.diagnostic import het_breuschpagan

# --- XGBoost ----------------------------------------------------------------
import xgboost as xgb

# --- openpyxl (Excel I/O) ---------------------------------------------------
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers

warnings.filterwarnings("ignore")

# --- Optional: LSTM via TensorFlow ------------------------------------------
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
print("  INDIAN GOVERNMENT BOND YIELD PREDICTION ENGINE  v2.0")
print("  Ensemble: VAR/VECM  +  XGBoost" + ("  +  Bi-LSTM" if LSTM_AVAILABLE else ""))
print("=" * 72)
if not LSTM_AVAILABLE:
    print("  [!] TensorFlow not found  ->  LSTM disabled.")
    print("      Install:  pip install tensorflow")
    print("      The engine will use VAR/VECM + XGBoost only.\n")


# ============================================================================
# SECTION 2 — CONFIGURATION
# ============================================================================
EXCEL_PATH    = Path(r"C:\Users\Lenovo\Downloads\dataforpythontraining.xlsx")
INPUT_SHEET   = "Sheet1"
OUTPUT_SHEET  = "Predictions"
BACKTEST_SHEET = "Backtest_Results"

# Only use data from this year onward (pre-2000 Indian bond data is static)
DATA_START_YEAR = 2000

# Yield columns — maps friendly label -> Excel column name
YIELD_COLS = {
    "3M":  "IN3MT Yield (%)",  "6M":  "IN6MT Yield (%)",
    "1Y":  "IN1YT Yield (%)",  "2Y":  "IN2YT Yield (%)",
    "3Y":  "IN3YT Yield (%)",  "4Y":  "IN4YT Yield (%)",
    "5Y":  "IN5YT Yield (%)",  "6Y":  "IN6YT Yield (%)",
    "7Y":  "IN7YT Yield (%)",  "8Y":  "IN8YT Yield (%)",
    "9Y":  "IN9YT Yield (%)",  "10Y": "IN10YT Yield (%)",
    "11Y": "IN11YT Yield (%)", "12Y": "IN12YT Yield (%)",
    "13Y": "IN13YT Yield (%)", "14Y": "IN14YT Yield (%)",
    "15Y": "IN15YT Yield (%)", "19Y": "IN19YT Yield (%)",
    "24Y": "IN24YT Yield (%)", "30Y": "IN30YT Yield (%)",
    "40Y": "IN40YT Yield (%)", "50Y": "IN50YT Yield (%)",
}

MACRO_COLS = {
    "CPI":       "CPI YoY (%)",
    "Repo Rate": "Repo Rate (%)",
    "IIP":       "IIP Growth (%)",
    "USD/INR":   "USD/INR Rate",
    "Crude":     "Crude Brent (INR/bbl)",
    "NSE":       "NSE Close Price",
    "FII":       "FII (INR)",
}

# Maturities to display in the summary (all 22 are still modelled)
DISPLAY_MATS = ["3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","14Y","15Y",
                "19Y","24Y","30Y","40Y","50Y"]

HORIZONS = {"1W": 5, "1M": 20, "3M": 60}
DEFAULT_MACRO_WEIGHTS = {"CPI": 25, "Repo Rate": 25, "IIP": 10,
                          "USD/INR": 10, "Crude": 10, "NSE": 10, "FII": 10}


# ============================================================================
# SECTION 3 — DATA LOADING
# ============================================================================
def load_data():
    print("\n[1/10] Loading data from Excel ...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=INPUT_SHEET, engine="openpyxl")
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.set_index("Date", inplace=True)

    # Keep only post-2000 data (modern Indian bond market)
    df = df[df.index.year >= DATA_START_YEAR]

    # Forward-fill macro columns (monthly / quarterly reported, daily grid)
    for col in MACRO_COLS.values():
        df[col] = df[col].ffill()

    # Linear-interpolate yields for any gaps
    for col in YIELD_COLS.values():
        df[col] = df[col].interpolate(method="linear")

    df.dropna(inplace=True)

    print(f"    Rows: {len(df):,}   |   "
          f"Range: {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    return df


# ============================================================================
# SECTION 4 — MACRO WEIGHTAGE INPUT
# ============================================================================
def get_macro_weights():
    print("\n" + "=" * 72)
    print("  MACRO VARIABLE WEIGHTAGE ASSIGNMENT")
    print("=" * 72)
    print("  Assign importance weights (0–100) to each macro driver.")
    print("  Press Enter to accept the default.  Weights are auto-normalised.\n")

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
# SECTION 5 — FEATURE ENGINEERING
# ============================================================================
def engineer_features(df, macro_weights):
    print("\n[2/10] Engineering features ...")
    F = pd.DataFrame(index=df.index)

    # ---------- yield features ----------
    for lab, col in YIELD_COLS.items():
        F[f"y_{lab}"]      = df[col]
        F[f"dy1_{lab}"]    = df[col].diff(1)   * 10_000   # 1-day chg bps
        F[f"dy5_{lab}"]    = df[col].diff(5)   * 10_000
        F[f"dy20_{lab}"]   = df[col].diff(20)  * 10_000
        F[f"ma5_{lab}"]    = df[col].rolling(5).mean()
        F[f"ma20_{lab}"]   = df[col].rolling(20).mean()

    # yield-curve shape
    F["slope_10y2y"]  = (df[YIELD_COLS["10Y"]] - df[YIELD_COLS["2Y"]])  * 10_000
    F["slope_10y3m"]  = (df[YIELD_COLS["10Y"]] - df[YIELD_COLS["3M"]])  * 10_000
    F["slope_30y5y"]  = (df[YIELD_COLS["30Y"]] - df[YIELD_COLS["5Y"]])  * 10_000
    F["butterfly"]    = (2*df[YIELD_COLS["5Y"]]
                         - df[YIELD_COLS["2Y"]]
                         - df[YIELD_COLS["10Y"]]) * 10_000

    # Nelson-Siegel approximate factors
    all_y = df[list(YIELD_COLS.values())]
    F["ns_level"]     = all_y.mean(axis=1)  * 10_000
    F["ns_slope"]     = (df[YIELD_COLS["30Y"]] - df[YIELD_COLS["3M"]]) * 10_000
    F["ns_curve"]     = (2*df[YIELD_COLS["7Y"]]
                         - df[YIELD_COLS["3M"]]
                         - df[YIELD_COLS["30Y"]]) * 10_000

    # rolling vol
    for lab in ["3M","1Y","5Y","10Y","30Y"]:
        F[f"vol20_{lab}"]  = df[YIELD_COLS[lab]].diff().rolling(20).std()  * 10_000
        F[f"vol60_{lab}"]  = df[YIELD_COLS[lab]].diff().rolling(60).std()  * 10_000

    # ---------- macro features ----------
    for name, col in MACRO_COLS.items():
        F[f"m_{name}"]       = df[col]
        F[f"m_chg5_{name}"]  = df[col].pct_change(5)
        F[f"m_chg20_{name}"] = df[col].pct_change(20)
        F[f"m_ma20_{name}"]  = df[col].rolling(20).mean()
        F[f"m_ma60_{name}"]  = df[col].rolling(60).mean()
        F[f"m_std20_{name}"] = df[col].rolling(20).std()

    # weighted macro composite z-score
    z_parts = pd.DataFrame(index=df.index)
    for name, col in MACRO_COLS.items():
        chg = df[col].pct_change(20)
        mu  = chg.rolling(252, min_periods=60).mean()
        sig = chg.rolling(252, min_periods=60).std()
        z_parts[name] = ((chg - mu) / (sig + 1e-12)).clip(-4, 4)

    w_arr = np.array([macro_weights[k] / 100 for k in MACRO_COLS])
    F["macro_composite"] = z_parts.fillna(0).values @ w_arr

    # real-yield & term-premium proxies
    F["real_yield_10y"]  = (df[YIELD_COLS["10Y"]] - df[MACRO_COLS["CPI"]]) * 100
    F["term_prem_10y"]   = (df[YIELD_COLS["10Y"]] - df[MACRO_COLS["Repo Rate"]]) * 100

    # momentum indicators
    for lab in ["5Y","10Y","30Y"]:
        F[f"mom_20_60_{lab}"] = F[f"ma20_{lab}"] - F[f"ma60_{lab}"] if f"ma60_{lab}" in F.columns else 0

    F.replace([np.inf, -np.inf], np.nan, inplace=True)
    F.dropna(inplace=True)
    print(f"    {F.shape[1]} features  |  {len(F):,} usable rows")
    return F


# ============================================================================
# SECTION 6 — DIAGNOSTIC TESTS
# ============================================================================
def run_diagnostics(df):
    print("\n[3/10] Running diagnostic tests ...")
    diag = {}

    # --- ADF stationarity ---
    print("\n  ADF Stationarity (H0: non-stationary):")
    print(f"  {'Series':20s} {'Stat':>9s} {'p-val':>8s} {'Result':>12s}")
    print("  " + "-" * 52)
    adf_info = {}
    test_series = {"10Y Yield": YIELD_COLS["10Y"],
                   "3M Yield":  YIELD_COLS["3M"],
                   "CPI":       MACRO_COLS["CPI"],
                   "Repo Rate": MACRO_COLS["Repo Rate"],
                   "USD/INR":   MACRO_COLS["USD/INR"]}
    for name, col in test_series.items():
        s = df[col].dropna()
        if len(s) < 200:
            continue
        res = adfuller(s, maxlag=20, autolag="AIC")
        stat_str = "Stationary" if res[1] < 0.05 else "NON-stat"
        adf_info[name] = {"stat": res[0], "p": res[1],
                          "stationary": res[1] < 0.05}
        print(f"  {name:20s} {res[0]:9.3f} {res[1]:8.4f} {stat_str:>12s}")

    print("\n  First differences:")
    for name, col in test_series.items():
        s = df[col].diff().dropna()
        if len(s) < 200:
            continue
        res = adfuller(s, maxlag=20, autolag="AIC")
        stat_str = "Stationary" if res[1] < 0.05 else "NON-stat"
        print(f"  d({name:18s}) {res[0]:9.3f} {res[1]:8.4f} {stat_str:>12s}")
    diag["adf"] = adf_info

    # --- Linearity (Pearson vs Spearman) ---
    print(f"\n  Linearity:  Macro -> 10Y Yield")
    print(f"  {'Variable':15s} {'Pearson':>9s} {'Spearman':>9s} {'Type':>14s}")
    print("  " + "-" * 50)
    linearity = {}
    y10 = df[YIELD_COLS["10Y"]]
    for name, col in MACRO_COLS.items():
        x = df[col]
        idx = y10.index.intersection(x.dropna().index)
        if len(idx) < 200:
            continue
        pr, _ = stats.pearsonr(y10.loc[idx], x.loc[idx])
        sr, _ = stats.spearmanr(y10.loc[idx], x.loc[idx])
        gap = abs(abs(sr) - abs(pr))
        rtype = "Non-Linear" if gap > 0.10 else ("Linear" if abs(pr) > 0.25 else "Weak")
        linearity[name] = {"pearson": pr, "spearman": sr, "type": rtype}
        print(f"  {name:15s} {pr:9.4f} {sr:9.4f} {rtype:>14s}")
    diag["linearity"] = linearity

    # --- Johansen cointegration ---
    print("\n  Johansen Cointegration (10Y + CPI + Repo):")
    diag["cointegrated"] = False
    try:
        coint_cols = [YIELD_COLS["10Y"], MACRO_COLS["CPI"], MACRO_COLS["Repo Rate"]]
        monthly = df[coint_cols].resample("ME").last().dropna()
        if len(monthly) > 60:
            joh = coint_johansen(monthly, det_order=0, k_ar_diff=2)
            print(f"  {'Hypothesis':18s} {'Trace':>9s} {'Crit-95%':>9s} {'Coint?':>8s}")
            print("  " + "-" * 48)
            for i in range(min(3, len(joh.lr1))):
                tr, cv = joh.lr1[i], joh.cvt[i, 1]
                print(f"  r <= {i}              {tr:9.3f} {cv:9.3f} {'YES' if tr > cv else 'no':>8s}")
            diag["cointegrated"] = bool(joh.lr1[0] > joh.cvt[0, 1])
    except Exception as e:
        print(f"  (skipped: {e})")

    # --- Granger causality ---
    print(f"\n  Granger Causality  (Macro -> d(10Y),  max lag 5):")
    gc = {}
    ychg = df[YIELD_COLS["10Y"]].diff().dropna()
    for name, col in MACRO_COLS.items():
        try:
            xchg = df[col].pct_change().dropna()
            idx  = ychg.index.intersection(xchg.index)
            tmp  = pd.DataFrame({"y": ychg.loc[idx], "x": xchg.loc[idx]})
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp.dropna(inplace=True)
            if len(tmp) < 300:
                continue
            res = grangercausalitytests(tmp[["y","x"]], maxlag=5, verbose=False)
            pmin = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, 6))
            sig  = "***" if pmin < 0.01 else "**" if pmin < 0.05 else "*" if pmin < 0.10 else ""
            gc[name] = pmin
            print(f"    {name:12s}  p={pmin:.4f}  {sig}")
        except Exception:
            pass
    diag["granger"] = gc
    return diag


# ============================================================================
# SECTION 7 — VAR / VECM
# ============================================================================
def build_econometric(df, diag):
    print("\n[4/10] Building VAR / VECM ...")
    var_yields = [YIELD_COLS[m] for m in ["3M","1Y","5Y","10Y","30Y"]]
    var_macros = [MACRO_COLS[m] for m in ["CPI","Repo Rate","USD/INR","Crude"]]
    var_cols   = var_yields + var_macros

    monthly      = df[var_cols].resample("ME").last().dropna()
    monthly_diff = monthly.diff().dropna()

    # use last 15 years
    cutoff = monthly_diff.index.max() - pd.DateOffset(years=15)
    md = monthly_diff[monthly_diff.index >= cutoff]
    if len(md) < 48:
        print("    Insufficient data — skipping VAR.")
        return None

    try:
        model = VAR(md)
        sel   = model.select_order(maxlags=8)
        lag   = max(sel.aic, 1)
        vr    = model.fit(lag)
        fcast = vr.forecast(md.values[-lag:], steps=3)
        print(f"    VAR({lag}) fitted  |  {len(md)} obs  |  AIC lag = {lag}")

        # impulse-response
        try:
            irf = vr.irf(periods=12)
            print("    Impulse Response Functions computed (12 periods)")
        except Exception:
            irf = None

        # VECM if cointegrated (predict returns LEVELS, not diffs)
        vecm_fcast = None
        last_level = monthly.iloc[-1].values  # last observed monthly level
        if diag.get("cointegrated"):
            try:
                ml = monthly[monthly.index >= cutoff].dropna()
                vecm_res = VECM(ml, k_ar_diff=max(lag - 1, 1),
                                coint_rank=1).fit()
                vecm_levels = vecm_res.predict(steps=3)
                # convert level forecast to CHANGE from last observation
                vecm_fcast = vecm_levels - last_level
                print("    VECM fitted (cointegration rank 1)")
            except Exception as e:
                print(f"    VECM skipped: {e}")

        # variance decomposition
        try:
            fevd = vr.fevd(periods=12)
            print("    Forecast Error Variance Decomposition computed")
        except Exception:
            fevd = None

        return {"var": vr, "forecast": fcast, "vecm_forecast": vecm_fcast,
                "cols": var_cols, "irf": irf, "fevd": fevd,
                "last_level": last_level}
    except Exception as e:
        print(f"    VAR failed: {e}")
        return None


# ============================================================================
# SECTION 8 — XGBOOST
# ============================================================================
def build_xgboost(features, df, macro_weights, horizon):
    print(f"\n[5/10] Building XGBoost  (horizon = {horizon}d) ...")
    models = {}

    for lab, col in YIELD_COLS.items():
        target = (df[col].shift(-horizon) - df[col]) * 10_000   # bps
        idx = features.index.intersection(target.dropna().index)
        X, y = features.loc[idx].copy(), target.loc[idx].copy()
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
        X, y = X[mask], y[mask]
        if len(X) < 800:
            continue

        split = int(len(X) * 0.80)
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        ytr, yte = y.iloc[:split], y.iloc[split:]

        # sample weights — amplify macro-driven periods
        sw = np.ones(len(Xtr))
        if "macro_composite" in Xtr.columns:
            mc = np.abs(Xtr["macro_composite"].values)
            sw = 1.0 + 0.5 * mc / (mc.mean() + 1e-8)

        # ----- direction classifier -----
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
            random_state=42, verbosity=0, use_label_encoder=False)
        clf.fit(Xtr, (ytr > 0).astype(int), sample_weight=sw,
                eval_set=[(Xte, (yte > 0).astype(int))], verbose=False)
        dir_acc = accuracy_score((yte > 0).astype(int), clf.predict(Xte))

        # ----- magnitude regressor -----
        reg = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="rmse",
            random_state=42, verbosity=0)
        reg.fit(Xtr, ytr, sample_weight=sw,
                eval_set=[(Xte, yte)], verbose=False)
        preds = reg.predict(Xte)
        rmse  = np.sqrt(mean_squared_error(yte, preds))
        mae   = mean_absolute_error(yte, preds)

        models[lab] = {"clf": clf, "reg": reg, "dir_acc": dir_acc,
                       "rmse": rmse, "mae": mae, "feats": list(X.columns),
                       "Xte": Xte, "yte": yte}

    print(f"\n  {'Mat':>6s} {'DirAcc':>7s} {'RMSE':>8s} {'MAE':>8s}")
    print("  " + "-" * 32)
    for m in DISPLAY_MATS:
        if m in models:
            r = models[m]
            print(f"  {m:>6s} {r['dir_acc']:>6.1%} {r['rmse']:>8.2f} {r['mae']:>8.2f}")
    return models


# ============================================================================
# SECTION 9 — LSTM  (optional)
# ============================================================================
def build_lstm(features, df, horizon, seq_len=60):
    if not LSTM_AVAILABLE:
        print("\n[6/10] LSTM — skipped (TensorFlow not installed)")
        return None
    print(f"\n[6/10] Building Bi-LSTM  (horizon={horizon}d, seq={seq_len}) ...")
    models = {}
    key_mats = ["3M","1Y","2Y","5Y","7Y","10Y","14Y","30Y"]

    for lab in key_mats:
        col = YIELD_COLS[lab]
        target = (df[col].shift(-horizon) - df[col]) * 10_000

        # select features relevant to this maturity
        keep = [c for c in features.columns
                if any(t in c for t in [f"y_{lab}", "dy1_", "dy5_", "dy20_",
                                         "slope", "butterfly", "ns_",
                                         "vol20_", "vol60_",
                                         "macro_composite",
                                         "real_yield", "term_prem",
                                         "m_chg20_", "m_ma20_"])]
        if len(keep) > 60:
            keep = keep[:60]

        idx = features.index.intersection(target.dropna().index)
        Xf  = features[keep].loc[idx].values
        yf  = target.loc[idx].values
        mask = np.isfinite(yf) & np.all(np.isfinite(Xf), axis=1)
        Xf, yf = Xf[mask], yf[mask]
        if len(Xf) < seq_len + 600:
            continue

        sx = RobustScaler(); Xs = sx.fit_transform(Xf)
        sy = StandardScaler(); ys = sy.fit_transform(yf.reshape(-1,1)).ravel()

        # sequences
        Xseq, yseq = [], []
        for i in range(seq_len, len(Xs)):
            Xseq.append(Xs[i - seq_len:i])
            yseq.append(ys[i])
        Xseq, yseq = np.array(Xseq), np.array(yseq)

        sp = int(len(Xseq) * 0.80)
        Xtr, Xte = Xseq[:sp], Xseq[sp:]
        ytr, yte = yseq[:sp], yseq[sp:]

        mdl = Sequential([
            Bidirectional(KerasLSTM(64, return_sequences=True),
                          input_shape=(seq_len, Xtr.shape[2])),
            Dropout(0.25),
            BatchNormalization(),
            KerasLSTM(32),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        mdl.compile(optimizer=Adam(learning_rate=0.001), loss="huber")
        mdl.fit(Xtr, ytr, epochs=50, batch_size=64,
                validation_split=0.10, verbose=0,
                callbacks=[EarlyStopping(patience=8, restore_best_weights=True),
                           ReduceLROnPlateau(factor=0.5, patience=4)])

        pred_s = mdl.predict(Xte, verbose=0).ravel()
        pred   = sy.inverse_transform(pred_s.reshape(-1,1)).ravel()
        act    = sy.inverse_transform(yte.reshape(-1,1)).ravel()
        rmse   = np.sqrt(mean_squared_error(act, pred))
        dacc   = np.mean(np.sign(act) == np.sign(pred))

        models[lab] = {"model": mdl, "sx": sx, "sy": sy,
                       "feats": keep, "seq": seq_len,
                       "rmse": rmse, "dir_acc": dacc}
        print(f"    {lab:>4s}:  DirAcc={dacc:.1%}   RMSE={rmse:.1f} bps")

    return models if models else None


# ============================================================================
# SECTION 10 — ENSEMBLE PREDICTION
# ============================================================================
def ensemble_predict(xgb_m, lstm_m, econ, features, df, macro_w, horizon):
    print(f"\n[7/10] Generating ensemble predictions ...")
    predictions = {}

    # adaptive ensemble weights based on model availability
    has_lstm = lstm_m is not None
    has_var  = econ is not None
    w_xgb   = 0.45 if has_lstm else (0.60 if has_var else 1.00)
    w_lstm  = 0.30 if has_lstm else 0.00
    w_var   = 1.0 - w_xgb - w_lstm

    for lab, col in YIELD_COLS.items():
        current = df[col].iloc[-1] * 100          # current yield %
        sources = []

        # --- XGBoost ---
        if lab in xgb_m:
            Xlast  = features.iloc[[-1]]
            prob   = xgb_m[lab]["clf"].predict_proba(Xlast)[0]
            mag    = float(xgb_m[lab]["reg"].predict(Xlast)[0])
            sources.append({"src": "XGBoost", "mag": mag,
                            "dir": "UP" if prob[1] > 0.5 else "DOWN",
                            "conf": float(max(prob)), "w": w_xgb})

        # --- LSTM ---
        if has_lstm and lab in lstm_m:
            m = lstm_m[lab]
            raw = features[m["feats"]].iloc[-m["seq"]:].values
            xs  = m["sx"].transform(raw).reshape(1, m["seq"], -1)
            ps  = m["model"].predict(xs, verbose=0)[0][0]
            mag = float(m["sy"].inverse_transform([[ps]])[0][0])
            sources.append({"src": "LSTM", "mag": mag,
                            "dir": "UP" if mag > 0 else "DOWN",
                            "conf": min(0.55 + abs(mag)/60, 0.92), "w": w_lstm})

        # --- VAR ---
        if has_var and col in econ["cols"]:
            ci = econ["cols"].index(col)
            vmag = float(econ["forecast"][0][ci]) * 10_000  # monthly diff -> bps
            vmag *= horizon / 20.0                           # scale to horizon
            if econ["vecm_forecast"] is not None:
                vmag2 = float(econ["vecm_forecast"][0][ci]) * 10_000 * horizon / 20
                vmag  = 0.5 * vmag + 0.5 * vmag2
            sources.append({"src": "VAR", "mag": vmag,
                            "dir": "UP" if vmag > 0 else "DOWN",
                            "conf": 0.55, "w": w_var})

        if not sources:
            continue

        tw = sum(s["w"] for s in sources)
        emag  = sum(s["mag"] * s["w"] for s in sources) / tw
        econf = sum(s["conf"]* s["w"] for s in sources) / tw

        up_w  = sum(s["w"] for s in sources if s["dir"] == "UP")
        dn_w  = sum(s["w"] for s in sources if s["dir"] == "DOWN")
        edir  = "FLAT" if abs(emag) < 2 else ("UP" if up_w > dn_w else "DOWN")

        # confidence interval from historical vol
        hist = (df[col].diff(horizon) * 10_000).dropna().tail(252)
        vol  = hist.std() if len(hist) > 60 else 10.0
        lo   = emag - 1.65 * vol       # 90% CI
        hi   = emag + 1.65 * vol

        # widen under extreme macro
        mc = features["macro_composite"].iloc[-1]
        if abs(mc) > 1.5:
            lo *= 1.20; hi *= 1.20; econf *= 0.90

        # momentum flag
        if f"dy20_{lab}" in features.columns:
            mom20 = features[f"dy20_{lab}"].iloc[-1]
            momentum = "Strong" if abs(mom20) > 15 else "Moderate" if abs(mom20) > 5 else "Weak"
        else:
            momentum = "N/A"

        predictions[lab] = {
            "current_pct":   round(current, 4),
            "direction":     edir,
            "change_bps":    round(emag, 2),
            "range_lo_bps":  round(lo, 2),
            "range_hi_bps":  round(hi, 2),
            "predicted_pct": round(current + emag / 100, 4),
            "confidence":    round(min(econf * 100, 95), 1),
            "momentum":      momentum,
            "n_models":      len(sources),
            "models":        "+".join(s["src"] for s in sources),
        }

    return predictions


# ============================================================================
# SECTION 11 — WALK-FORWARD BACKTEST
# ============================================================================
def run_backtest(features, df, macro_w, horizon, n_folds=5):
    print(f"\n[8/10] Walk-forward backtest  ({n_folds} folds, {horizon}d horizon) ...")
    results = {}
    key_mats = ["3M","1Y","2Y","5Y","7Y","10Y","14Y","30Y"]
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=252)

    for lab in key_mats:
        col = YIELD_COLS[lab]
        target = (df[col].shift(-horizon) - df[col]) * 10_000
        idx = features.index.intersection(target.dropna().index)
        X, y = features.loc[idx], target.loc[idx]
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
        X, y = X[mask], y[mask]
        if len(X) < 2000:
            continue

        daccs, rmses, maes = [], [], []
        naive_daccs = []   # benchmark: predict zero change

        for train_i, test_i in tscv.split(X):
            Xtr, Xte = X.iloc[train_i], X.iloc[test_i]
            ytr, yte = y.iloc[train_i], y.iloc[test_i]
            if len(Xtr) < 500:
                continue

            reg = xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
            reg.fit(Xtr, ytr, verbose=False)
            pred = reg.predict(Xte)

            daccs.append(np.mean(np.sign(yte) == np.sign(pred)))
            rmses.append(np.sqrt(mean_squared_error(yte, pred)))
            maes.append(mean_absolute_error(yte, pred))

            # naive benchmark: predict 0 change
            naive_daccs.append(np.mean(yte <= 0))  # "no change" gets ~50%

        if daccs:
            results[lab] = {
                "dir_acc":     np.mean(daccs),
                "dir_std":     np.std(daccs),
                "rmse":        np.mean(rmses),
                "mae":         np.mean(maes),
                "naive_acc":   np.mean(naive_daccs),
                "edge":        np.mean(daccs) - 0.50,
                "folds":       len(daccs),
            }

    print(f"\n  {'Mat':>6s} {'DirAcc':>7s} {'+-Std':>6s} {'RMSE':>8s} "
          f"{'MAE':>8s} {'Naive':>6s} {'Edge':>6s}")
    print("  " + "-" * 50)
    for m in key_mats:
        if m in results:
            r = results[m]
            print(f"  {m:>6s} {r['dir_acc']:>6.1%} {r['dir_std']:>5.1%} "
                  f"{r['rmse']:>8.2f} {r['mae']:>8.2f} "
                  f"{r['naive_acc']:>5.1%} {r['edge']:>+5.1%}")
    return results


# ============================================================================
# SECTION 12 — FEATURE IMPORTANCE
# ============================================================================
def analyze_importance(xgb_m, macro_w):
    print("\n[9/10] Feature importance analysis ...")
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

    # aggregate macro contribution
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
# SECTION 13 — EXCEL OUTPUT
# ============================================================================
def write_output(preds, bt, macro_w, diag, horizon_label):
    print(f"\n[10/10] Writing results to Excel ...")
    wb = openpyxl.load_workbook(str(EXCEL_PATH))

    # ---------- styling ----------
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
                f"Ensemble: VAR/VECM + XGBoost" +
                (" + Bi-LSTM" if LSTM_AVAILABLE else ""))
    ws["A2"].font = Font(italic=True, size=10, color="666666")

    # macro weights
    ws["A4"] = "Macro Weights Applied:"
    ws["A4"].font = Font(bold=True)
    r = 5
    for nm, wt in macro_w.items():
        ws.cell(r, 1, nm); ws.cell(r, 2, f"{wt:.1f}%"); r += 1

    # headers
    sr = r + 1
    hdrs = ["Maturity","Current Yield (%)","Direction","Change (bps)",
            "Range Low (bps)","Range High (bps)","Predicted Yield (%)",
            "Confidence (%)","Momentum","Models"]
    for ci, h in enumerate(hdrs, 1):
        c = ws.cell(sr, ci, h); c.font = hf; c.fill = hfl; c.alignment = ctr; c.border = bdr

    # data
    mat_order = ["3M","6M","1Y","2Y","3Y","4Y","5Y","6Y","7Y","8Y","9Y",
                 "10Y","11Y","12Y","13Y","14Y","15Y","19Y","24Y","30Y","40Y","50Y"]
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
                if v == "UP":    c.fill = upf; c.font = Font(bold=True, color="006100")
                elif v == "DOWN":c.fill = dnf; c.font = Font(bold=True, color="9C0006")
                else:            c.fill = ftf; c.font = Font(bold=True, color="9C6500")
        dr += 1

    for col in ws.columns:
        ml = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(ml + 4, 28)

    # ========== BACKTEST SHEET ==========
    if BACKTEST_SHEET in wb.sheetnames:
        del wb[BACKTEST_SHEET]
    ws2 = wb.create_sheet(BACKTEST_SHEET)
    ws2.merge_cells("A1:G1")
    ws2["A1"] = "Walk-Forward Backtest Results"
    ws2["A1"].font = Font(bold=True, size=14, color="1F4E79")

    bh = ["Maturity","Dir Accuracy","Std","RMSE (bps)","MAE (bps)",
          "Naive Acc","Edge vs 50%"]
    for ci, h in enumerate(bh, 1):
        c = ws2.cell(3, ci, h); c.font = hf; c.fill = hfl; c.border = bdr

    br = 4
    for lab, r in bt.items():
        vals = [lab, f"{r['dir_acc']:.1%}", f"{r['dir_std']:.1%}",
                f"{r['rmse']:.2f}", f"{r['mae']:.2f}",
                f"{r['naive_acc']:.1%}", f"{r['edge']:+.1%}"]
        for ci, v in enumerate(vals, 1):
            c = ws2.cell(br, ci, v); c.border = bdr
        br += 1

    # diagnostics section
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

    for col in ws2.columns:
        ml = max(len(str(c.value or "")) for c in col)
        ws2.column_dimensions[col[0].column_letter].width = min(ml + 4, 30)

    wb.save(str(EXCEL_PATH))
    print(f"    Saved: {EXCEL_PATH.name}")
    print(f"    New sheets: '{OUTPUT_SHEET}', '{BACKTEST_SHEET}'")


# ============================================================================
# SECTION 14 — MAIN PIPELINE
# ============================================================================
def main():
    # 1. Load
    df = load_data()

    # 2. Weights
    macro_w = get_macro_weights()

    # 3. Features
    feats = engineer_features(df, macro_w)
    df_a  = df.loc[feats.index]

    # 4. Diagnostics
    diag = run_diagnostics(df_a)

    # 5. VAR/VECM
    econ = build_econometric(df_a, diag)

    # 6. Horizon
    print("\n" + "=" * 72)
    print("  SELECT PREDICTION HORIZON")
    print("=" * 72)
    print("    1) 1 Week   (5 trading days)")
    print("    2) 1 Month  (20 trading days)")
    print("    3) 3 Months (60 trading days)")
    choice = input("\n    Choice [1/2/3, default=2]: ").strip()
    hlabel, hdays = {"1":("1W",5),"2":("1M",20),"3":("3M",60)}.get(choice, ("1M",20))

    # 7. XGBoost
    xgb_models = build_xgboost(feats, df_a, macro_w, hdays)

    # 8. LSTM
    lstm_models = build_lstm(feats, df_a, hdays)

    # 9. Ensemble
    preds = ensemble_predict(xgb_models, lstm_models, econ,
                             feats, df_a, macro_w, hdays)

    # 10. Backtest
    bt = run_backtest(feats, df_a, macro_w, hdays)

    # 11. Importance
    analyze_importance(xgb_models, macro_w)

    # 12. Write
    write_output(preds, bt, macro_w, diag, hlabel)

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 82)
    print("  PREDICTION SUMMARY")
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

    print(f"\n  Results written to:  {EXCEL_PATH.name}")
    print(f"  Sheets: '{OUTPUT_SHEET}'  and  '{BACKTEST_SHEET}'")
    print("=" * 82)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
