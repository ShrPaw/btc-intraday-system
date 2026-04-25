"""
SETUP VALIDATION ENGINE
=======================
Transforms the BTC intraday system from a backtest equity engine into a
SETUP QUALITY VALIDATION ENGINE. No equity, no leverage, no position sizing.
Every setup is measured purely in R-multiples with structural stops.

Fixes ALL data leakage identified in the strict audit:
  - Rolling past-only quantiles (no global)
  - Next-candle entry (signal close → entry next open)
  - Worst-case intracandle (SL before TP when ambiguous)
  - Structural stop (swing/EMA invalidation, not fixed %)
  - No equity/leverage/position sizing in output

Phases:
  1. Audit data and timing fixes
  2. Remove equity dependency
  3. Setup validation engine
  4. Structural stop
  5. TP levels in R
  6. Worst-case intracandle logic
  7. Required metrics (grouped)
  8. Expectancy in R
  9. Random baselines
  10. Short focus analysis
  11. Signal output format
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG — identical to original, NO parameter tuning
# =========================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

RSI_PERIOD = 14
EMA_PERIOD = 20
STRUCTURE_GATE_WINDOW = 5
ADX_PERIOD = 14
ADX_TRENDING_BONUS = 25.0
ADX_CHOPPY_PENALTY = 15.0
ENABLE_ADX_FILTER = True

NO_TRADE_THRESHOLD = 0.72
MILD_THRESHOLD = 0.78
MID_THRESHOLD = 0.80
PREMIUM_THRESHOLD = 0.82
HIGH_THRESHOLD = 0.88

ALLOWED_MODES = {"MILD", "MID", "HIGH", "PREMIUM", "ELITE"}
VOLUME_PERCENTILE = 0.30
COOLDOWN_BARS_5M = 3
LOOKBACK_SWING = 12
SWING_STOP_LOOKBACK = 4

# Structural stop bounds
STOP_FLOOR_PCT = 0.004   # 0.4% minimum
STOP_CAP_PCT = 0.020     # 2.0% maximum

# TP levels in R multiples
TP_R_MULTIPLES = [1, 2, 3, 4]

# Walk-forward
TRAIN_CUTOFF = pd.Timestamp("2025-12-01")

# Horizon for tracking outcomes (max bars to look ahead)
OUTCOME_HORIZON_BARS = 200  # ~16.7 hours at 5m


# =========================================================
# AUDIT TABLE — documents every leakage fix
# =========================================================

AUDIT_FIXES = [
    {
        "id": "F1",
        "issue": "Global volume quantile",
        "original": "df['volume'].quantile(0.30) on full dataset",
        "fix": "Rolling expanding quantile with shift(1): only past data",
        "severity": "P0",
    },
    {
        "id": "F2",
        "issue": "Same-candle entry",
        "original": "entry_price = df['close'].iloc[i] (signal candle)",
        "fix": "entry_price = df['open'].iloc[i+1] (next candle open)",
        "severity": "P0",
    },
    {
        "id": "F3",
        "issue": "TP-before-SL in same candle",
        "original": "TP checked before SL (optimistic fill)",
        "fix": "If both touched in same candle, SL assumed first (worst case)",
        "severity": "P1",
    },
    {
        "id": "F4",
        "issue": "Global quantile in confidence",
        "original": "bounded_positive_score() uses df.abs().quantile(0.8)",
        "fix": "Rolling expanding quantile with shift(1), window=5000",
        "severity": "P1",
    },
    {
        "id": "F5",
        "issue": "Global EMA dist reference",
        "original": "ema_dist_pct.abs().quantile(0.8) on full dataset",
        "fix": "Rolling expanding quantile with shift(1), window=5000",
        "severity": "P1",
    },
    {
        "id": "F6",
        "issue": "Multi-timeframe future candle",
        "original": "merge_asof uses currently-forming HTF candle",
        "fix": "Higher-TF indicators use only CLOSED candles via shift on merge",
        "severity": "P1",
    },
    {
        "id": "F7",
        "issue": "Equity/leverage in output",
        "original": "Signals require equity, position size, margin",
        "fix": "Output is pure: direction, entry, stop, R-multiples, confidence",
        "severity": "P2",
    },
    {
        "id": "F8",
        "issue": "Fixed % TP levels",
        "original": "TP1=0.5%, TP2=0.7%, etc. regardless of structure",
        "fix": "TP levels at 1R/2R/3R/4R from structural stop distance",
        "severity": "P2",
    },
    {
        "id": "F9",
        "issue": "Volume threshold uses current candle",
        "original": "volume >= volume.quantile(0.30) includes current bar",
        "fix": "expanding().quantile().shift(1) — only past candles",
        "severity": "P0",
    },
]


def print_audit_table():
    print("\n" + "=" * 100)
    print("  AUDIT TABLE — DATA LEAKAGE FIXES")
    print("=" * 100)
    print(f"  {'ID':<5s} {'Severity':<9s} {'Issue':<35s} {'Original → Fixed'}")
    print(f"  {'-'*5}-{'-'*9}-{'-'*35}-{'-'*60}")
    for f in AUDIT_FIXES:
        print(f"  {f['id']:<5s} {f['severity']:<9s} {f['issue']:<35s} {f['fix']}")
    print()


# =========================================================
# LOAD DATA
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "delta" not in df.columns:
        df["delta"] = 0.0
    if "trade_count" not in df.columns:
        df["trade_count"] = 1.0
    return df


# =========================================================
# RESAMPLE
# =========================================================

def resample_bars(ticks: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = ticks.set_index("timestamp")
    bars = pd.DataFrame({
        "open": x["price"].resample(rule).first(),
        "high": x["price"].resample(rule).max(),
        "low": x["price"].resample(rule).min(),
        "close": x["price"].resample(rule).last(),
        "volume": x["volume"].resample(rule).sum(),
        "delta": x["delta"].resample(rule).sum(),
        "trade_count": x["trade_count"].resample(rule).sum(),
    }).dropna().reset_index()
    return bars


# =========================================================
# INDICATORS — identical to original, no changes
# =========================================================

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_rsi_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = rsi_wilder(df["close"], period)
    df["rsi_prev"] = df["rsi"].shift(1)
    df["cross_up_50"] = (df["rsi_prev"] <= 50) & (df["rsi"] > 50)
    df["cross_dn_50"] = (df["rsi_prev"] >= 50) & (df["rsi"] < 50)
    cross_dir = np.where(df["cross_up_50"], 1, np.where(df["cross_dn_50"], -1, 0))
    df["cross_dir"] = cross_dir

    last_cross_idx = -1
    last_cross_dir = 0
    bars_since, last_dir = [], []
    for i, d in enumerate(cross_dir):
        if d != 0:
            last_cross_idx = i
            last_cross_dir = int(d)
        if last_cross_idx == -1:
            bars_since.append(np.nan)
            last_dir.append(0)
        else:
            bars_since.append(i - last_cross_idx)
            last_dir.append(last_cross_dir)

    df["bars_since_cross"] = bars_since
    df["last_cross_dir"] = last_dir
    df["fresh_long"] = (df["last_cross_dir"] == 1) & (df["bars_since_cross"] <= 4)
    df["fresh_short"] = (df["last_cross_dir"] == -1) & (df["bars_since_cross"] <= 4)
    df["rsi_slope_1"] = df["rsi"] - df["rsi"].shift(1)
    df["rsi_slope_2"] = df["rsi"] - df["rsi"].shift(2)
    return df


def add_ema_bos_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_dist_pct"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["ema_reclaim_long"] = (
        (df["close"].shift(1) <= df["ema20"].shift(1)) &
        (df["close"] > df["ema20"])
    )
    df["ema_reclaim_short"] = (
        (df["close"].shift(1) >= df["ema20"].shift(1)) &
        (df["close"] < df["ema20"])
    )
    lookback = LOOKBACK_SWING
    df["prev_swing_high"] = df["high"].rolling(lookback).max().shift(1)
    df["prev_swing_low"] = df["low"].rolling(lookback).min().shift(1)
    df["bos_long"] = df["high"] > df["prev_swing_high"]
    df["bos_short"] = df["low"] < df["prev_swing_low"]
    df["structure_trigger_long"] = (
        df["ema_reclaim_long"].rolling(STRUCTURE_GATE_WINDOW, min_periods=1).max().fillna(0).astype(bool) &
        df["bos_long"].rolling(STRUCTURE_GATE_WINDOW, min_periods=1).max().fillna(0).astype(bool)
    )
    df["structure_trigger_short"] = (
        df["ema_reclaim_short"].rolling(STRUCTURE_GATE_WINDOW, min_periods=1).max().fillna(0).astype(bool) &
        df["bos_short"].rolling(STRUCTURE_GATE_WINDOW, min_periods=1).max().fillna(0).astype(bool)
    )
    return df


def add_extra_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["rv_6"] = df["ret_1"].rolling(6).std()
    df["delta_3"] = df["delta"].rolling(3).sum()
    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    alpha = 1.0 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    df["plus_di"] = plus_di.values
    df["minus_di"] = minus_di.values
    df["adx"] = adx.values
    return df


# =========================================================
# HELPERS — FIX F4/F5: rolling quantiles only
# =========================================================

def rolling_minmax_score(series: pd.Series, window: int = 200) -> pd.Series:
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return ((series - roll_min) / denom).clip(0, 1)


def bounded_positive_score_rolling(series: pd.Series, ref_quantile: float = 0.8,
                                    window: int = 5000) -> pd.Series:
    """FIX F4: Rolling quantile — only uses past data, never future."""
    abs_s = series.abs()
    ref = abs_s.rolling(window, min_periods=200).quantile(ref_quantile).shift(1)
    ref = ref.replace(0, np.nan)
    return (abs_s / ref).clip(0, 1).fillna(0)


def merge_asof_feature(base: pd.DataFrame, other: pd.DataFrame, prefix: str,
                       cols: list) -> pd.DataFrame:
    """FIX F6: Use only closed higher-TF candles."""
    rhs = other[["timestamp"] + cols].copy()
    # Shift higher-TF data by 1 to ensure we use only CLOSED candles
    for c in cols:
        rhs[c] = rhs[c].shift(1)
    rhs = rhs.dropna(subset=[cols[0]])
    rhs = rhs.rename(columns={c: f"{prefix}_{c}" for c in cols})
    return pd.merge_asof(
        base.sort_values("timestamp"),
        rhs.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


# =========================================================
# SETUP ENGINE — identical logic to original
# =========================================================

def build_setup_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    scalp_long_context = (
        df["h4_fresh_long"] &
        (df["m15_rsi"] > 50) &
        (df["m15_rsi"].shift(1) > 50) &
        (df["h6_rsi"] > 50) &
        (df["h12_rsi"] > 50)
    )
    scalp_short_context = (
        df["h4_fresh_short"] &
        (df["m15_rsi"] < 50) &
        (df["m15_rsi"].shift(1) < 50) &
        (df["h6_rsi"] < 50) &
        (df["h12_rsi"] < 50)
    )
    trend_long_context = (
        (df["h4_rsi"] > 55) &
        (df["h4_rsi_slope_1"] > 0) &
        (df["h4_rsi_slope_2"] > 0) &
        (df["m15_rsi"] > 55) &
        (df["m15_rsi_slope_1"] >= 0) &
        (df["h12_rsi"] > 50)
    )
    trend_short_context = (
        (df["h4_rsi"] < 45) &
        (df["h4_rsi_slope_1"] < 0) &
        (df["h4_rsi_slope_2"] < 0) &
        (df["m15_rsi"] < 45) &
        (df["m15_rsi_slope_1"] <= 0) &
        (df["h12_rsi"] < 50)
    )

    df["setup_type"] = "none"
    df["direction"] = 0
    df["stage"] = 0

    df.loc[scalp_long_context, ["setup_type", "direction"]] = ["RSI_SCALP", 1]
    df.loc[scalp_short_context, ["setup_type", "direction"]] = ["RSI_SCALP", -1]
    df.loc[trend_long_context, ["setup_type", "direction"]] = ["RSI_TREND", 1]
    df.loc[trend_short_context, ["setup_type", "direction"]] = ["RSI_TREND", -1]

    bsc = df["h4_bars_since_cross"].fillna(99).astype(int)
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc <= 1), "stage"] = 1
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 2) & (bsc <= 3), "stage"] = 2
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 4), "stage"] = 3

    return df


# =========================================================
# STRUCTURE GATE — identical logic
# =========================================================

def add_structure_gate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trigger_ok"] = False
    out.loc[(out["setup_type"] == "RSI_SCALP") & (out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["setup_type"] == "RSI_SCALP") & (out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    out.loc[(out["setup_type"] == "RSI_TREND") & (out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["setup_type"] == "RSI_TREND") & (out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    return out


# =========================================================
# CONFIDENCE ENGINE — FIX F4/F5: rolling quantiles only
# =========================================================

def build_confidence_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    h4_align_long = (out["h4_rsi"] > 50).astype(float)
    h4_align_short = (out["h4_rsi"] < 50).astype(float)
    h6_align_long = (out["h6_rsi"] > 50).astype(float)
    h6_align_short = (out["h6_rsi"] < 50).astype(float)
    h12_align_long = (out["h12_rsi"] > 50).astype(float)
    h12_align_short = (out["h12_rsi"] < 50).astype(float)

    alignment_long = 0.4 * h4_align_long + 0.3 * h6_align_long + 0.3 * h12_align_long
    alignment_short = 0.4 * h4_align_short + 0.3 * h6_align_short + 0.3 * h12_align_short

    fresh_base = 1 - (out["h4_bars_since_cross"] / 4.0)
    fresh_base = fresh_base.clip(0, 1).fillna(0)

    # FIX F4: rolling quantile for slope scores
    h4_slope_score = bounded_positive_score_rolling(out["h4_rsi_slope_1"], 0.8, window=5000)
    m15_slope_score = bounded_positive_score_rolling(out["m15_rsi_slope_1"], 0.8, window=5000)
    ret_score = bounded_positive_score_rolling(out["ret_3"], 0.8, window=5000)
    momentum_score = (0.4 * h4_slope_score + 0.4 * m15_slope_score + 0.2 * ret_score).clip(0, 1)

    # FIX F5: rolling ema_dist reference
    ema_dist_rolling_ref = out["ema_dist_pct"].abs().rolling(5000, min_periods=200).quantile(0.8).shift(1)
    ema_dist_rolling_ref = ema_dist_rolling_ref.fillna(1e-9).clip(lower=1e-9)
    ema_dist_quality = (1 - (out["ema_dist_pct"].abs() / ema_dist_rolling_ref)).clip(0, 1)

    structure_long = (
        0.4 * out["ema_reclaim_long"].astype(float) +
        0.4 * out["bos_long"].astype(float) +
        0.2 * ema_dist_quality
    ).clip(0, 1)
    structure_short = (
        0.4 * out["ema_reclaim_short"].astype(float) +
        0.4 * out["bos_short"].astype(float) +
        0.2 * ema_dist_quality
    ).clip(0, 1)

    rv_score = rolling_minmax_score(out["rv_6"].fillna(0), 200).fillna(0.5)

    cfx_long = (out["m30_rsi"] > 50).astype(float)
    cfx_short = (out["m30_rsi"] < 50).astype(float)
    cfx_align = np.where(out["direction"] == 1, cfx_long, cfx_short)
    out["cfx_score"] = cfx_align

    out["confidence_raw"] = 0.0

    scalp_mask = out["setup_type"] == "RSI_SCALP"
    scalp_alignment = np.where(out["direction"] == 1, alignment_long, alignment_short)
    scalp_structure = np.where(out["direction"] == 1, structure_long, structure_short)
    scalp_conf = (
        0.30 * scalp_alignment + 0.25 * fresh_base + 0.20 * momentum_score +
        0.15 * rv_score + 0.10 * scalp_structure
    )

    trend_mask = out["setup_type"] == "RSI_TREND"
    trend_alignment = np.where(out["direction"] == 1, alignment_long, alignment_short)
    trend_structure = np.where(out["direction"] == 1, structure_long, structure_short)
    trend_conf = (
        0.25 * trend_alignment + 0.20 * momentum_score + 0.30 * trend_structure +
        0.15 * rv_score + 0.10 * cfx_align
    )

    out.loc[scalp_mask, "confidence_raw"] = scalp_conf[scalp_mask]
    out.loc[trend_mask, "confidence_raw"] = trend_conf[trend_mask]
    out["confidence_raw"] = out["confidence_raw"].clip(0, 1)

    if ENABLE_ADX_FILTER and "h4_adx" in out.columns:
        adx_val = out["h4_adx"].fillna(20)
        adx_modifier = pd.Series(0.0, index=out.index)
        adx_modifier[adx_val >= ADX_TRENDING_BONUS] = 0.03
        adx_modifier[(adx_val >= 15) & (adx_val < 25)] = 0.0
        adx_modifier[adx_val < ADX_CHOPPY_PENALTY] = -0.03
        out["confidence_raw"] = (out["confidence_raw"] + adx_modifier).clip(0, 1)
        out["adx_modifier"] = adx_modifier
    else:
        out["adx_modifier"] = 0.0

    premium_fresh = fresh_base >= 0.80
    premium_momentum = momentum_score >= 0.70
    premium_structure = np.where(out["direction"] == 1, structure_long, structure_short) >= 0.85
    premium_vol = rv_score >= 0.55
    premium_dist = ema_dist_quality >= 0.55
    elite_gate = premium_fresh & premium_momentum & premium_structure & premium_vol & premium_dist

    out["confidence_mode"] = "NO_TRADE"
    out.loc[(out["confidence_raw"] >= NO_TRADE_THRESHOLD) & (out["confidence_raw"] < MILD_THRESHOLD), "confidence_mode"] = "MILD"
    out.loc[(out["confidence_raw"] >= MILD_THRESHOLD) & (out["confidence_raw"] < MID_THRESHOLD), "confidence_mode"] = "MID"
    out.loc[(out["confidence_raw"] >= MID_THRESHOLD) & (out["confidence_raw"] < PREMIUM_THRESHOLD), "confidence_mode"] = "PREMIUM"
    out.loc[(out["confidence_raw"] >= PREMIUM_THRESHOLD) & (out["confidence_raw"] < HIGH_THRESHOLD), "confidence_mode"] = "HIGH"
    out.loc[(out["confidence_raw"] >= HIGH_THRESHOLD) & elite_gate, "confidence_mode"] = "ELITE"

    return out


def short_validation_layer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["short_valid"] = True
    mid_short_mask = (out["confidence_mode"] == "MID") & (out["direction"] == -1)
    if not mid_short_mask.any():
        return out
    htf_bearish_count = (
        (out["h4_rsi"] < 45).astype(int) +
        (out["h6_rsi"] < 45).astype(int) +
        (out["h12_rsi"] < 45).astype(int)
    )
    c1 = htf_bearish_count >= 2
    m15_slope_neg = out["m15_rsi_slope_1"] < -0.2
    consecutive_neg = m15_slope_neg.rolling(3).sum() == 3
    c2 = consecutive_neg.fillna(False)
    rolling_low_1h = out["low"].rolling(12).min()
    c3 = (out["close"] - rolling_low_1h) / out["close"] > 0.01
    c4 = out["ema20"] < out["ema50"]
    c5 = (out["close"] < out["ema20"]) & (out["close"] < out["ema50"])
    all_conditions = c1 & c2 & c3 & c4 & c5
    out.loc[mid_short_mask & ~all_conditions, "short_valid"] = False
    return out


# =========================================================
# TRADE DECISION — FIX F1/F9: rolling volume threshold
# =========================================================

def add_trade_decision(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mode_direction_ok = (
        out["confidence_mode"].isin(ALLOWED_MODES) &
        (
            ((out["direction"] == 1) & True) |
            ((out["direction"] == -1) & True)
        )
    )

    # FIX F1/F9: expanding quantile with shift(1) — past-only
    volume_rolling_threshold = (
        out["volume"]
        .expanding(min_periods=500)
        .quantile(VOLUME_PERCENTILE)
        .shift(1)
    )
    volume_ok = out["volume"] >= volume_rolling_threshold.fillna(0)

    out["take_trade"] = (
        (out["setup_type"] != "none") &
        out["trigger_ok"] &
        (out["confidence_mode"] != "NO_TRADE") &
        mode_direction_ok &
        volume_ok
    )
    return out


# =========================================================
# STRUCTURAL STOP — Phase 4
# =========================================================

def compute_structural_stop(df: pd.DataFrame, signal_idx: int,
                            direction: int, strategy: str) -> dict:
    """
    Stop = structural invalidation level.
    For LONG: below recent swing low, BOS invalidation, EMA20 reclaim failure
    For SHORT: above recent swing high, BOS invalidation, EMA20 loss failure
    Returns: stop_price, stop_distance_pct, stop_source, is_stop_valid
    """
    row = df.iloc[signal_idx]
    signal_price = float(row["close"])

    start = max(0, signal_idx - SWING_STOP_LOOKBACK)
    window = df.iloc[start:signal_idx + 1]

    if direction == 1:
        # LONG stop candidates
        swing_low = float(window["low"].min())
        ema20_val = float(row["ema20"])

        # Pick the tighter (higher) of swing low or EMA20 for RSI_TREND
        if strategy == "RSI_TREND":
            stop_candidates = [swing_low, ema20_val]
            stop_price = max(stop_candidates)  # tighter stop
            if stop_price == swing_low:
                stop_source = "swing_low"
            else:
                stop_source = "ema20_reclaim"
        else:
            stop_price = swing_low
            stop_source = "swing_low"

        stop_distance_pct = (signal_price - stop_price) / signal_price
    else:
        # SHORT stop candidates
        swing_high = float(window["high"].max())
        ema20_val = float(row["ema20"])

        if strategy == "RSI_TREND":
            stop_candidates = [swing_high, ema20_val]
            stop_price = min(stop_candidates)  # tighter stop
            if stop_price == swing_high:
                stop_source = "swing_high"
            else:
                stop_source = "ema20_loss"
        else:
            stop_price = swing_high
            stop_source = "swing_high"

        stop_distance_pct = (stop_price - signal_price) / signal_price

    # Clamp to floor/cap
    original_pct = stop_distance_pct
    stop_distance_pct = max(stop_distance_pct, STOP_FLOOR_PCT)
    stop_distance_pct = min(stop_distance_pct, STOP_CAP_PCT)

    # Recompute stop price from clamped distance
    if direction == 1:
        stop_price = signal_price * (1 - stop_distance_pct)
    else:
        stop_price = signal_price * (1 + stop_distance_pct)

    is_stop_valid = (STOP_FLOOR_PCT <= original_pct <= STOP_CAP_PCT)

    return {
        "stop_price": stop_price,
        "stop_distance_pct": stop_distance_pct,
        "stop_source": stop_source,
        "is_stop_valid": is_stop_valid,
    }


# =========================================================
# R-MULTIPLE TRACKER — Phase 5 + Phase 6
# =========================================================

def compute_tp_levels(entry_price: float, stop_price: float, direction: int) -> dict:
    """Phase 5: TP levels at 1R, 2R, 3R, 4R from structural stop."""
    R_abs = abs(entry_price - stop_price)
    tps = {}
    for n in TP_R_MULTIPLES:
        if direction == 1:
            tp_price = entry_price + n * R_abs
        else:
            tp_price = entry_price - n * R_abs
        tps[f"TP{n}"] = tp_price
    tps["R_abs"] = R_abs
    return tps


def track_setup_outcome(df: pd.DataFrame, entry_idx: int, direction: int,
                         stop_price: float, tp_levels: dict) -> dict:
    """
    Phase 6: Worst-case intracandle logic.
    If SL and TP both touched in same candle, assume SL hit first.
    Tracks MFE, MAE, time to each level.
    """
    R_abs = tp_levels["R_abs"]
    if R_abs <= 0:
        return {"valid": False}

    tp_prices = {n: tp_levels[f"TP{n}"] for n in TP_R_MULTIPLES}

    hit_1R = False
    hit_2R = False
    hit_3R = False
    hit_4R = False
    time_to_1R = np.nan
    time_to_2R = np.nan
    time_to_3R = np.nan
    time_to_4R = np.nan
    time_to_SL = np.nan
    sl_hit = False
    ambiguous_1R = False
    ambiguous_2R = False
    ambiguous_3R = False
    ambiguous_4R = False

    max_favorable = 0.0
    max_adverse = 0.0

    end_idx = min(entry_idx + OUTCOME_HORIZON_BARS, len(df) - 1)

    for j in range(entry_idx + 1, end_idx + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        # Compute excursion in R
        if direction == 1:
            favorable = (high - tp_levels["entry_price"]) / R_abs if "entry_price" in tp_levels else (high - df.iloc[entry_idx]["close"]) / R_abs
            adverse = (df.iloc[entry_idx]["close"] - low) / R_abs if "entry_price" not in tp_levels else (tp_levels["entry_price"] - low) / R_abs
        else:
            favorable = (df.iloc[entry_idx]["close"] - low) / R_abs
            adverse = (high - df.iloc[entry_idx]["close"]) / R_abs

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        # ──────────────────────────────────────────────────────
        # FIX P0-1: Check SL FIRST, then TP. If both touched in
        # the same candle, SL wins. TP must NOT count as hit.
        # ──────────────────────────────────────────────────────

        # Step 1: Determine if SL is touched on this bar
        if direction == 1:
            sl_touched = low <= stop_price
        else:
            sl_touched = high >= stop_price

        # Step 2: If SL touched, check for same-candle TP ambiguity
        if sl_touched and not sl_hit:
            sl_hit = True
            time_to_SL = j - entry_idx

            # Check if any TP was also touched in this same candle
            # If so, flag ambiguity — TP must NOT count as hit
            for n in TP_R_MULTIPLES:
                tp_p = tp_prices[n]
                if direction == 1:
                    tp_also_touched = high >= tp_p
                else:
                    tp_also_touched = low <= tp_p

                if tp_also_touched:
                    if n == 1 and not hit_1R:
                        ambiguous_1R = True
                        # Do NOT set hit_1R — SL wins
                    elif n == 2 and not hit_2R:
                        ambiguous_2R = True
                    elif n == 3 and not hit_3R:
                        ambiguous_3R = True
                    elif n == 4 and not hit_4R:
                        ambiguous_4R = True
            # SL occurred — stop tracking
            break

        # Step 3: Only check TP if SL was NOT touched on this bar
        if not sl_touched:
            for n in TP_R_MULTIPLES:
                tp_p = tp_prices[n]
                if direction == 1:
                    tp_touched = high >= tp_p
                else:
                    tp_touched = low <= tp_p

                if tp_touched:
                    if n == 1 and not hit_1R:
                        hit_1R = True
                        time_to_1R = j - entry_idx
                    elif n == 2 and not hit_2R:
                        hit_2R = True
                        time_to_2R = j - entry_idx
                    elif n == 3 and not hit_3R:
                        hit_3R = True
                        time_to_3R = j - entry_idx
                    elif n == 4 and not hit_4R:
                        hit_4R = True
                        time_to_4R = j - entry_idx

    expired = not sl_hit and not hit_1R

    return {
        "valid": True,
        "hit_1R": hit_1R,
        "hit_2R": hit_2R,
        "hit_3R": hit_3R,
        "hit_4R": hit_4R,
        "sl_hit": sl_hit,
        "time_to_1R": time_to_1R,
        "time_to_2R": time_to_2R,
        "time_to_3R": time_to_3R,
        "time_to_4R": time_to_4R,
        "time_to_SL": time_to_SL,
        "max_favorable_excursion_R": max_favorable,
        "max_adverse_excursion_R": max_adverse,
        "expired_without_resolution": expired,
        "ambiguous_1R": ambiguous_1R,
        "ambiguous_2R": ambiguous_2R,
        "ambiguous_3R": ambiguous_3R,
        "ambiguous_4R": ambiguous_4R,
    }


# =========================================================
# VALIDATION ENGINE — runs all setups, builds result table
# =========================================================

def run_validation(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Run validation on all qualifying setups. Returns one row per setup."""
    results = []

    # FIX F2: entry at next candle open
    for i in range(len(df)):
        row = df.iloc[i]
        if row["setup_type"] == "none" or row["direction"] == 0 or not bool(row["take_trade"]):
            continue

        # Entry is next candle open
        entry_idx = i + 1
        if entry_idx >= len(df):
            continue

        entry_price = float(df.iloc[entry_idx]["open"])
        signal_time = row["timestamp"]
        entry_time = df.iloc[entry_idx]["timestamp"]
        direction = int(row["direction"])
        strategy = str(row["setup_type"])
        confidence_mode = str(row["confidence_mode"])
        confidence_raw = float(row["confidence_raw"])

        # Structural stop
        stop_info = compute_structural_stop(df, i, direction, strategy)
        if not stop_info["is_stop_valid"]:
            continue

        stop_price = stop_info["stop_price"]
        stop_distance_pct = stop_info["stop_distance_pct"]

        # TP levels in R
        tp_levels = compute_tp_levels(entry_price, stop_price, direction)
        tp_levels["entry_price"] = entry_price

        # Track outcome
        outcome = track_setup_outcome(df, entry_idx, direction, stop_price, tp_levels)
        if not outcome.get("valid", False):
            continue

        # Build result row — NO equity, NO leverage
        result = {
            "symbol": symbol,
            "signal_time": signal_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "dir_label": "LONG" if direction == 1 else "SHORT",
            "setup_type": strategy,
            "confidence_mode": confidence_mode,
            "confidence_raw": confidence_raw,
            "stage": int(row.get("stage", 0)),
            "cfx_score": float(row.get("cfx_score", 0)),
            "stop_price": stop_price,
            "stop_distance_pct": stop_distance_pct,
            "stop_source": stop_info["stop_source"],
            "is_stop_valid": stop_info["is_stop_valid"],
            "R_abs": tp_levels["R_abs"],
            "TP1_price": tp_levels["TP1"],
            "TP2_price": tp_levels["TP2"],
            "TP3_price": tp_levels["TP3"],
            "TP4_price": tp_levels["TP4"],
        }

        # Add outcome data
        for key, val in outcome.items():
            if key != "valid":
                result[key] = val

        # Regime classification
        h4_rsi = float(row.get("h4_rsi", 50))
        rv_6 = float(row.get("rv_6", 0))
        hour = signal_time.hour

        if h4_rsi > 55:
            result["htf_regime"] = "bullish"
        elif h4_rsi < 45:
            result["htf_regime"] = "bearish"
        else:
            result["htf_regime"] = "neutral"

        if 0 <= hour < 8:
            result["session"] = "Asian"
        elif 8 <= hour < 16:
            result["session"] = "European"
        else:
            result["session"] = "US"

        result["month"] = signal_time.strftime("%Y-%m")
        result["hour_utc"] = hour

        # HTF fields for all directions (used for regime filtering)
        result["h4_rsi_entry"] = h4_rsi
        result["h6_rsi_entry"] = float(row.get("h6_rsi", 50))
        result["h12_rsi_entry"] = float(row.get("h12_rsi", 50))

        # Short-specific fields
        if direction == -1:
            result["price_vs_ema20"] = float(row.get("ema_dist_pct", 0))
            result["ema20_val"] = float(row.get("ema20", 0))
            result["ema50_val"] = float(row.get("ema50", 0))

        results.append(result)

    return pd.DataFrame(results)


# =========================================================
# METRICS REPORTER — Phase 7
# =========================================================

def compute_group_metrics(subset: pd.DataFrame) -> dict:
    """Compute all metrics for a subset of setups."""
    n = len(subset)
    if n == 0:
        return None

    # Hit rates
    hit_1R = subset["hit_1R"].sum()
    hit_2R = subset["hit_2R"].sum()
    hit_3R = subset["hit_3R"].sum()
    hit_4R = subset["hit_4R"].sum()
    sl_count = subset["sl_hit"].sum()

    pct_hit_1R = hit_1R / n
    pct_hit_2R = hit_2R / n
    pct_hit_3R = hit_3R / n
    pct_hit_4R = hit_4R / n

    # MFE / MAE
    median_mfe = subset["max_favorable_excursion_R"].median()
    mean_mfe = subset["max_favorable_excursion_R"].mean()
    median_mae = subset["max_adverse_excursion_R"].median()
    mean_mae = subset["max_adverse_excursion_R"].mean()

    # Time to levels (in 5m bars)
    t1_valid = subset["time_to_1R"].dropna()
    tsl_valid = subset["time_to_SL"].dropna()
    avg_time_1R = t1_valid.mean() if len(t1_valid) > 0 else np.nan
    avg_time_SL = tsl_valid.mean() if len(tsl_valid) > 0 else np.nan

    # Expiration
    expired = subset["expired_without_resolution"].sum()
    expiration_rate = expired / n

    # Ambiguous
    amb_count = subset[["ambiguous_1R", "ambiguous_2R", "ambiguous_3R", "ambiguous_4R"]].any(axis=1).sum()
    ambiguous_rate = amb_count / n

    # Phase 8: Expectancy in R
    # P(hit_1R) * 1 - P(SL_before_1R) * 1
    # SL_before_1R = SL hit AND NOT hit_1R
    sl_before_1R = (subset["sl_hit"] & ~subset["hit_1R"]).sum()
    sl_before_2R = (subset["sl_hit"] & ~subset["hit_2R"]).sum()
    sl_before_3R = (subset["sl_hit"] & ~subset["hit_3R"]).sum()

    exp_1R = (hit_1R / n) * 1.0 - (sl_before_1R / n) * 1.0
    exp_2R = (hit_2R / n) * 2.0 - (sl_before_2R / n) * 1.0
    exp_3R = (hit_3R / n) * 3.0 - (sl_before_3R / n) * 1.0

    return {
        "count": n,
        "pct_hit_1R": pct_hit_1R,
        "pct_hit_2R": pct_hit_2R,
        "pct_hit_3R": pct_hit_3R,
        "pct_hit_4R": pct_hit_4R,
        "sl_rate": sl_count / n,
        "median_MFE_R": median_mfe,
        "mean_MFE_R": mean_mfe,
        "median_MAE_R": median_mae,
        "mean_MAE_R": mean_mae,
        "avg_time_to_1R_bars": avg_time_1R,
        "avg_time_to_SL_bars": avg_time_SL,
        "expiration_rate": expiration_rate,
        "ambiguous_rate": ambiguous_rate,
        "expectancy_1R": exp_1R,
        "expectancy_2R": exp_2R,
        "expectancy_3R": exp_3R,
    }


def print_metrics_table(title: str, groups: dict):
    """Print a formatted metrics table."""
    if not groups:
        print(f"\n  {title}: No data")
        return

    print(f"\n  {'=' * 120}")
    print(f"  {title}")
    print(f"  {'=' * 120}")
    header = (
        f"  {'Group':<25s} {'N':>5s} {'1R':>6s} {'2R':>6s} {'3R':>6s} {'4R':>6s} "
        f"{'MedMFE':>7s} {'MedMAE':>7s} {'AvgT1R':>7s} {'AvgTSL':>7s} "
        f"{'Exp%':>6s} {'Amb%':>6s} {'Exp1R':>7s} {'Exp2R':>7s} {'Exp3R':>7s}"
    )
    print(header)
    print(f"  {'-' * 25}-{'-' * 5}-{'-' * 6}-{'-' * 6}-{'-' * 6}-{'-' * 6}-"
          f"{'-' * 7}-{'-' * 7}-{'-' * 7}-{'-' * 7}-{'-' * 6}-{'-' * 6}-"
          f"{'-' * 7}-{'-' * 7}-{'-' * 7}")

    for group_name, m in groups.items():
        if m is None:
            continue
        t1r = f"{m['avg_time_to_1R_bars']:.1f}" if pd.notna(m['avg_time_to_1R_bars']) else "  -"
        tsl = f"{m['avg_time_to_SL_bars']:.1f}" if pd.notna(m['avg_time_to_SL_bars']) else "  -"
        print(
            f"  {group_name:<25s} {m['count']:>5d} "
            f"{m['pct_hit_1R']:>5.1%} {m['pct_hit_2R']:>5.1%} "
            f"{m['pct_hit_3R']:>5.1%} {m['pct_hit_4R']:>5.1%} "
            f"{m['median_MFE_R']:>6.2f}R {m['median_MAE_R']:>6.2f}R "
            f"{t1r:>7s} {tsl:>7s} "
            f"{m['expiration_rate']:>5.1%} {m['ambiguous_rate']:>5.1%} "
            f"{m['expectancy_1R']:>+6.3f} {m['expectancy_2R']:>+6.3f} {m['expectancy_3R']:>+6.3f}"
        )


def compute_all_metrics(results: pd.DataFrame) -> dict:
    """Phase 7: Compute metrics for all required groupings."""
    all_groups = {}

    # 1. Direction
    dir_groups = {}
    for d in ["LONG", "SHORT"]:
        sub = results[results["dir_label"] == d]
        dir_groups[d] = compute_group_metrics(sub)
    all_groups["Direction"] = dir_groups

    # 2. Confidence
    conf_groups = {}
    for c in ["MILD", "MID", "HIGH", "PREMIUM", "ELITE"]:
        sub = results[results["confidence_mode"] == c]
        conf_groups[c] = compute_group_metrics(sub)
    all_groups["Confidence"] = conf_groups

    # 3. Direction + Confidence
    dc_groups = {}
    for d in ["LONG", "SHORT"]:
        for c in ["MILD", "MID", "HIGH", "PREMIUM", "ELITE"]:
            sub = results[(results["dir_label"] == d) & (results["confidence_mode"] == c)]
            dc_groups[f"{d}/{c}"] = compute_group_metrics(sub)
    all_groups["Direction+Confidence"] = dc_groups

    # 4. Month
    month_groups = {}
    for m in sorted(results["month"].unique()):
        sub = results[results["month"] == m]
        month_groups[m] = compute_group_metrics(sub)
    all_groups["Month"] = month_groups

    # 5. Regime
    regime_groups = {}
    for r in ["bullish", "bearish", "neutral"]:
        sub = results[results["htf_regime"] == r]
        regime_groups[r] = compute_group_metrics(sub)
    all_groups["HTF Regime"] = regime_groups

    # 6. Session
    session_groups = {}
    for s in ["Asian", "European", "US"]:
        sub = results[results["session"] == s]
        session_groups[s] = compute_group_metrics(sub)
    all_groups["Session"] = session_groups

    return all_groups


# =========================================================
# PHASE 9: RANDOM BASELINES
# =========================================================

def build_random_baselines(results: pd.DataFrame, df: pd.DataFrame,
                           symbol: str) -> dict:
    """Build random baselines to test if edge is real.
    FIX P0-2: Reset index so positional indexing matches label indexing.
    Also adds full-dataset and same-session baselines."""
    baselines = {}
    rng = np.random.RandomState(42 + hash(symbol) % 10000)

    # FIX: Reset index so iloc positions match index labels
    df = df.reset_index(drop=True).copy()
    n_bars = len(df)

    # All bars with setup_type (strategy's filtered universe)
    tradeable_indices = df[df["setup_type"] != "none"].index.tolist()
    if len(tradeable_indices) < 50:
        print(f"    [SKIP] {symbol}: only {len(tradeable_indices)} tradeable bars")
        return baselines

    n_setups = len(results)
    if n_setups == 0:
        return baselines

    # Helper: run one random entry through outcome tracking
    def _run_one(sig_idx, rand_dir, setup_type):
        entry_idx = sig_idx + 1
        if entry_idx >= n_bars:
            return None
        entry_price = float(df.iloc[entry_idx]["open"])
        stop_info = compute_structural_stop(df, sig_idx, rand_dir, setup_type)
        if not stop_info["is_stop_valid"]:
            return None
        stop_price = stop_info["stop_price"]
        tp_levels = compute_tp_levels(entry_price, stop_price, rand_dir)
        tp_levels["entry_price"] = entry_price
        outcome = track_setup_outcome(df, entry_idx, rand_dir, stop_price, tp_levels)
        if not outcome.get("valid", False):
            return None
        r = {
            "direction": rand_dir,
            "dir_label": "LONG" if rand_dir == 1 else "SHORT",
            "setup_type": setup_type,
        }
        r.update({k: v for k, v in outcome.items() if k != "valid"})
        return r

    # ─────────────────────────────────────────────────────
    # Baseline 0: Full-dataset random (any valid bar)
    # ─────────────────────────────────────────────────────
    bl0_results = []
    bl0_attempted = 0
    bl0_skipped = 0
    # Exclude early bars without enough indicator history
    min_start = 500
    valid_bars = list(range(min_start, n_bars - 1))  # -1 because we need next open
    sample_size = min(n_setups * 4, len(valid_bars))
    sampled = rng.choice(valid_bars, size=sample_size, replace=False)
    for sig_idx in sampled:
        bl0_attempted += 1
        rand_dir = rng.choice([1, -1])
        r = _run_one(sig_idx, rand_dir, "RSI_SCALP")
        if r is None:
            bl0_skipped += 1
            continue
        bl0_results.append(r)
    if bl0_results:
        baselines["full_dataset_random"] = pd.DataFrame(bl0_results)
    print(f"    Baseline 0 (full-dataset random): {bl0_attempted} attempted, {bl0_skipped} skipped, {len(bl0_results)} valid")

    # ─────────────────────────────────────────────────────
    # Baseline 1: Same timestamps, random direction
    # ─────────────────────────────────────────────────────
    bl1_results = []
    bl1_attempted = 0
    bl1_skipped = 0
    for _, row in results.iterrows():
        bl1_attempted += 1
        sig_time = row["signal_time"]
        mask = df["timestamp"] == sig_time
        if not mask.any():
            bl1_skipped += 1
            continue
        sig_idx = int(df.index[mask][0])
        entry_idx = sig_idx + 1
        if entry_idx >= n_bars:
            bl1_skipped += 1
            continue

        rand_dir = rng.choice([1, -1])
        entry_price = float(df.iloc[entry_idx]["open"])

        stop_info = compute_structural_stop(df, sig_idx, rand_dir, row["setup_type"])
        if not stop_info["is_stop_valid"]:
            bl1_skipped += 1
            continue
        stop_price = stop_info["stop_price"]
        tp_levels = compute_tp_levels(entry_price, stop_price, rand_dir)
        tp_levels["entry_price"] = entry_price

        outcome = track_setup_outcome(df, entry_idx, rand_dir, stop_price, tp_levels)
        if not outcome.get("valid", False):
            bl1_skipped += 1
            continue

        r = {
            "direction": rand_dir,
            "dir_label": "LONG" if rand_dir == 1 else "SHORT",
            "setup_type": row["setup_type"],
            "confidence_mode": row["confidence_mode"],
            "htf_regime": row["htf_regime"],
            "session": row["session"],
            "month": row["month"],
        }
        r.update({k: v for k, v in outcome.items() if k != "valid"})
        bl1_results.append(r)

    if bl1_results:
        baselines["same_time_random_dir"] = pd.DataFrame(bl1_results)
    print(f"    Baseline 1 (same-time random-dir): {bl1_attempted} attempted, {bl1_skipped} skipped, {len(bl1_results)} valid")

    # ─────────────────────────────────────────────────────
    # Baseline 2: Same regime, random timestamp
    # ─────────────────────────────────────────────────────
    bl2_results = []
    bl2_attempted = 0
    bl2_skipped = 0
    # Derive regime from h4_rsi in df (not from setup_type)
    df["_regime"] = "neutral"
    if "h4_rsi" in df.columns:
        df.loc[df["h4_rsi"] > 55, "_regime"] = "bullish"
        df.loc[df["h4_rsi"] < 45, "_regime"] = "bearish"

    regime_map = results.groupby("htf_regime").apply(lambda x: x.index.tolist()).to_dict()
    for regime, indices in regime_map.items():
        # Sample from ALL bars in this regime, not just setup bars
        regime_bars = df[
            (df["_regime"] == regime) &
            (df.index >= 500) &
            (df.index < n_bars - 1)
        ]
        if len(regime_bars) < 20:
            continue
        sample_size = min(len(indices) * 3, len(regime_bars))
        sampled = regime_bars.sample(n=sample_size, random_state=rng)
        for _, bar in sampled.iterrows():
            bl2_attempted += 1
            sig_idx = int(bar.name)
            rand_dir = rng.choice([1, -1])
            r = _run_one(sig_idx, rand_dir, "RSI_SCALP")
            if r is None:
                bl2_skipped += 1
                continue
            r["htf_regime"] = regime
            bl2_results.append(r)

    if bl2_results:
        baselines["same_regime_random_time"] = pd.DataFrame(bl2_results)
    print(f"    Baseline 2 (same-regime random-time): {bl2_attempted} attempted, {bl2_skipped} skipped, {len(bl2_results)} valid")

    # ─────────────────────────────────────────────────────
    # Baseline 3: Same direction distribution
    # ─────────────────────────────────────────────────────
    long_ratio = (results["direction"] == 1).mean()
    bl3_results = []
    bl3_attempted = 0
    bl3_skipped = 0
    valid_bars_all = list(range(500, n_bars - 1))
    sample_size = min(n_setups * 3, len(valid_bars_all))
    sampled_indices = rng.choice(valid_bars_all, size=sample_size, replace=False)
    for sig_idx in sampled_indices:
        bl3_attempted += 1
        rand_dir = 1 if rng.random() < long_ratio else -1
        r = _run_one(int(sig_idx), rand_dir, "RSI_SCALP")
        if r is None:
            bl3_skipped += 1
            continue
        bl3_results.append(r)

    if bl3_results:
        baselines["same_dir_dist"] = pd.DataFrame(bl3_results)
    print(f"    Baseline 3 (same-dir-distrib): {bl3_attempted} attempted, {bl3_skipped} skipped, {len(bl3_results)} valid")

    # ─────────────────────────────────────────────────────
    # Baseline 4: Same holding window (random entry)
    # ─────────────────────────────────────────────────────
    bl4_results = []
    bl4_attempted = 0
    bl4_skipped = 0
    sample_size = min(n_setups * 3, len(valid_bars_all))
    sampled_indices = rng.choice(valid_bars_all, size=sample_size, replace=False)
    for sig_idx in sampled_indices:
        bl4_attempted += 1
        rand_dir = rng.choice([1, -1])
        r = _run_one(int(sig_idx), rand_dir, "RSI_SCALP")
        if r is None:
            bl4_skipped += 1
            continue
        bl4_results.append(r)

    if bl4_results:
        baselines["same_holding_window"] = pd.DataFrame(bl4_results)
    print(f"    Baseline 4 (same-holding-window): {bl4_attempted} attempted, {bl4_skipped} skipped, {len(bl4_results)} valid")

    # ─────────────────────────────────────────────────────
    # Baseline 5: Same session, random timestamp
    # ─────────────────────────────────────────────────────
    bl5_results = []
    bl5_attempted = 0
    bl5_skipped = 0
    df["_session"] = "US"
    df.loc[(df["timestamp"].dt.hour >= 0) & (df["timestamp"].dt.hour < 8), "_session"] = "Asian"
    df.loc[(df["timestamp"].dt.hour >= 8) & (df["timestamp"].dt.hour < 16), "_session"] = "European"

    session_map = results.groupby("session").apply(lambda x: x.index.tolist()).to_dict()
    for session, indices in session_map.items():
        session_bars = df[
            (df["_session"] == session) &
            (df.index >= 500) &
            (df.index < n_bars - 1)
        ]
        if len(session_bars) < 20:
            continue
        sample_size = min(len(indices) * 3, len(session_bars))
        sampled = session_bars.sample(n=sample_size, random_state=rng)
        for _, bar in sampled.iterrows():
            bl5_attempted += 1
            sig_idx = int(bar.name)
            rand_dir = rng.choice([1, -1])
            r = _run_one(sig_idx, rand_dir, "RSI_SCALP")
            if r is None:
                bl5_skipped += 1
                continue
            r["session"] = session
            bl5_results.append(r)

    if bl5_results:
        baselines["same_session_random_time"] = pd.DataFrame(bl5_results)
    print(f"    Baseline 5 (same-session random-time): {bl5_attempted} attempted, {bl5_skipped} skipped, {len(bl5_results)} valid")

    # Cleanup temp columns
    df.drop(columns=["_regime", "_session"], inplace=True, errors="ignore")

    return baselines


def print_baseline_comparison(results: pd.DataFrame, baselines: dict):
    """Phase 9: Compare system vs random baselines."""
    print(f"\n  {'=' * 100}")
    print(f"  PHASE 9: BASELINE COMPARISON — SYSTEM vs RANDOM")
    print(f"  {'=' * 100}")

    sys_metrics = compute_group_metrics(results)
    if sys_metrics is None:
        print("  No system setups to compare.")
        return

    header = f"  {'Baseline':<30s} {'N':>6s} {'1R%':>6s} {'2R%':>6s} {'3R%':>6s} {'Exp1R':>7s} {'Exp2R':>7s} {'Exp3R':>7s} {'MedMFE':>7s} {'MedMAE':>7s}"
    print(header)
    print(f"  {'-'*30}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")

    # System row
    print(
        f"  {'SYSTEM (actual)':<30s} {sys_metrics['count']:>6d} "
        f"{sys_metrics['pct_hit_1R']:>5.1%} {sys_metrics['pct_hit_2R']:>5.1%} "
        f"{sys_metrics['pct_hit_3R']:>5.1%} "
        f"{sys_metrics['expectancy_1R']:>+6.3f} {sys_metrics['expectancy_2R']:>+6.3f} "
        f"{sys_metrics['expectancy_3R']:>+6.3f} "
        f"{sys_metrics['median_MFE_R']:>6.2f}R {sys_metrics['median_MAE_R']:>6.2f}R"
    )

    for bl_name, bl_df in baselines.items():
        bl_m = compute_group_metrics(bl_df)
        if bl_m is None:
            continue
        print(
            f"  {bl_name:<30s} {bl_m['count']:>6d} "
            f"{bl_m['pct_hit_1R']:>5.1%} {bl_m['pct_hit_2R']:>5.1%} "
            f"{bl_m['pct_hit_3R']:>5.1%} "
            f"{bl_m['expectancy_1R']:>+6.3f} {bl_m['expectancy_2R']:>+6.3f} "
            f"{bl_m['expectancy_3R']:>+6.3f} "
            f"{bl_m['median_MFE_R']:>6.2f}R {bl_m['median_MAE_R']:>6.2f}R"
        )


# =========================================================
# PHASE 10: SHORT FOCUS ANALYSIS
# =========================================================

def classify_short_failure(row: pd.Series) -> str:
    """Classify why a SHORT setup failed (hit SL)."""
    if not row.get("sl_hit", False):
        return "success"

    h4_rsi = row.get("h4_rsi_entry", 50)
    h6_rsi = row.get("h6_rsi_entry", 50)
    h12_rsi = row.get("h12_rsi_entry", 50)
    ema_dist = row.get("price_vs_ema20", 0)
    mfe = row.get("max_favorable_excursion_R", 0)
    mae = row.get("max_adverse_excursion_R", 0)

    # 1. Pullback short inside bullish HTF
    if h4_rsi > 55 and h6_rsi > 50:
        return "pullback_in_bullish_htf"

    # 2. Late short after move already extended
    if mae > 1.5 and mfe < 0.3:
        return "late_after_extended_move"

    # 3. Short into support (low MFE, tight stop)
    if mfe < 0.2 and row.get("stop_distance_pct", 0) < 0.006:
        return "short_into_support"

    # 4. Weak BOS / fake breakdown
    if mfe < 0.5 and mae > 0.8:
        return "weak_bos_fake_breakdown"

    # 5. Low-volatility chop
    if mfe < 0.3 and mae < 0.5:
        return "low_vol_chop"

    # 6. Stop too tight
    if row.get("stop_distance_pct", 0) < 0.005 and mae > 0.5:
        return "stop_too_tight"

    # 7. Entry too late
    if mae > 1.0:
        return "entry_too_late"

    return "other_failure"


def analyze_short_setups(results: pd.DataFrame):
    """Phase 10: Deep analysis of SHORT setups."""
    shorts = results[results["direction"] == -1].copy()
    if len(shorts) == 0:
        print("\n  No SHORT setups to analyze.")
        return

    print(f"\n  {'=' * 100}")
    print(f"  PHASE 10: SHORT FOCUS ANALYSIS")
    print(f"  {'=' * 100}")

    # Overall short metrics
    sm = compute_group_metrics(shorts)
    print(f"\n  OVERALL SHORTS: {sm['count']} setups")
    print(f"    Hit 1R: {sm['pct_hit_1R']:.1%} | Hit 2R: {sm['pct_hit_2R']:.1%} | Hit 3R: {sm['pct_hit_3R']:.1%}")
    print(f"    Expectancy: 1R={sm['expectancy_1R']:+.3f} | 2R={sm['expectancy_2R']:+.3f} | 3R={sm['expectancy_3R']:+.3f}")

    # By confidence mode
    print(f"\n  SHORT BY CONFIDENCE:")
    for mode in ["MILD", "MID", "HIGH", "PREMIUM", "ELITE"]:
        sub = shorts[shorts["confidence_mode"] == mode]
        if len(sub) == 0:
            continue
        m = compute_group_metrics(sub)
        print(
            f"    {mode:<10s}: {m['count']:>4d} setups | "
            f"1R: {m['pct_hit_1R']:.1%} | 2R: {m['pct_hit_2R']:.1%} | "
            f"Exp1R: {m['expectancy_1R']:+.3f} | Exp2R: {m['expectancy_2R']:+.3f} | "
            f"MedMFE: {m['median_MFE_R']:.2f}R"
        )

    # Failure classification for SHORT/MID
    mid_shorts = shorts[shorts["confidence_mode"] == "MID"]
    if len(mid_shorts) > 0:
        failures = mid_shorts[mid_shorts["sl_hit"]]
        if len(failures) > 0:
            print(f"\n  SHORT/MID FAILURE CLASSIFICATION ({len(failures)} failures):")
            failures = failures.copy()
            failures["failure_class"] = failures.apply(classify_short_failure, axis=1)
            for fc, count in failures["failure_class"].value_counts().items():
                pct = count / len(failures)
                print(f"    {fc:<30s}: {count:>4d} ({pct:.1%})")

    # All shorts failure classification
    all_short_failures = shorts[shorts["sl_hit"]].copy()
    if len(all_short_failures) > 0:
        print(f"\n  ALL SHORT FAILURE CLASSIFICATION ({len(all_short_failures)} failures):")
        all_short_failures["failure_class"] = all_short_failures.apply(classify_short_failure, axis=1)
        for fc, count in all_short_failures["failure_class"].value_counts().items():
            pct = count / len(all_short_failures)
            print(f"    {fc:<30s}: {count:>4d} ({pct:.1%})")

    # Short/MID verdict
    mid_m = compute_group_metrics(mid_shorts)
    if mid_m is not None:
        print(f"\n  SHORT/MID VERDICT:")
        if mid_m["expectancy_1R"] > 0 and mid_m["pct_hit_1R"] > 0.45:
            print(f"    ✅ VALID — SHORT/MID shows positive expectancy ({mid_m['expectancy_1R']:+.3f} at 1R)")
        elif mid_m["expectancy_1R"] > -0.1:
            print(f"    ⚠️  MARGINAL — SHORT/MID expectancy near zero ({mid_m['expectancy_1R']:+.3f} at 1R)")
        else:
            print(f"    ❌ DISABLE — SHORT/MID has negative expectancy ({mid_m['expectancy_1R']:+.3f} at 1R)")

    # HTF state at short entry
    print(f"\n  SHORT HTF STATE AT ENTRY:")
    for regime in ["bullish", "bearish", "neutral"]:
        sub = shorts[shorts["htf_regime"] == regime]
        if len(sub) == 0:
            continue
        m = compute_group_metrics(sub)
        print(
            f"    {regime:<10s}: {m['count']:>4d} setups | "
            f"1R: {m['pct_hit_1R']:.1%} | Exp1R: {m['expectancy_1R']:+.3f}"
        )


# =========================================================
# PHASE 11: SIGNAL OUTPUT FORMAT
# =========================================================

def print_sample_signals(results: pd.DataFrame, n: int = 5):
    """Phase 11: Print sample signals in the required format."""
    print(f"\n  {'=' * 80}")
    print(f"  PHASE 11: SAMPLE SIGNALS (top {n} by confidence)")
    print(f"  {'=' * 80}")

    # Pick top signals by confidence, diverse across symbols
    top = results.nlargest(n * 2, "confidence_raw")
    # Deduplicate by symbol, keep top per symbol
    seen_symbols = set()
    selected = []
    for _, row in top.iterrows():
        if row["symbol"] not in seen_symbols and len(selected) < n:
            selected.append(row)
            seen_symbols.add(row["symbol"])
    if len(selected) < n:
        for _, row in top.iterrows():
            if len(selected) >= n:
                break
            # Check if this row is already selected (by index)
            if row.name not in [s.name for s in selected]:
                selected.append(row)

    # Also get historical quality for each signal
    for row in selected:
        setup_desc = f"{row['setup_type']}"
        if row["direction"] == 1:
            setup_desc += " + EMA_RECLAIM + BOS_UP"
        else:
            setup_desc += " + EMA_LOSS + BOS_DOWN"

        # Historical stats from same confidence/setup
        same_type = results[
            (results["setup_type"] == row["setup_type"]) &
            (results["confidence_mode"] == row["confidence_mode"]) &
            (results["dir_label"] == row["dir_label"])
        ]
        hist_count = len(same_type)
        hist_1R = same_type["hit_1R"].mean() if hist_count > 0 else 0
        hist_2R = same_type["hit_2R"].mean() if hist_count > 0 else 0
        hist_3R = same_type["hit_3R"].mean() if hist_count > 0 else 0
        hist_mfe = same_type["max_favorable_excursion_R"].median() if hist_count > 0 else 0
        hist_mae = same_type["max_adverse_excursion_R"].median() if hist_count > 0 else 0

        # Verdict
        exp_1R = compute_group_metrics(same_type)
        if exp_1R and exp_1R["expectancy_1R"] > 0.05:
            verdict = "VALID SETUP"
        elif exp_1R and exp_1R["expectancy_1R"] > -0.05:
            verdict = "MARGINAL SETUP"
        else:
            verdict = "WEAK SETUP"

        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │ SIGNAL DETECTED                                         │
  │ Asset:     {row['symbol']:<15s}                          │
  │ Direction: {row['dir_label']:<15s}                          │
  │ Time:      {str(row['signal_time'])[:-6]:<25s} UTC        │
  │ Setup:     {setup_desc:<40s}│
  │                                                         │
  │ Entry:     {row['entry_price']:>12,.2f} (next open)               │
  │ Stop:      {row['stop_price']:>12,.2f} ({row['stop_distance_pct']:.2%}, {row['stop_source']})│
  │ TP1:       {row['TP1_price']:>12,.2f} | 1R                          │
  │ TP2:       {row['TP2_price']:>12,.2f} | 2R                          │
  │ TP3:       {row['TP3_price']:>12,.2f} | 3R                          │
  │ TP4:       {row['TP4_price']:>12,.2f} | 4R                          │
  │                                                         │
  │ Confidence: {row['confidence_mode']:<10s} ({row['confidence_raw']:.3f})               │
  │                                                         │
  │ Historical Quality ({hist_count} similar setups):               │
  │ Hit 1R: {hist_1R:.0%} | Hit 2R: {hist_2R:.0%} | Hit 3R: {hist_3R:.0%}          │
  │ Median MFE: {hist_mfe:.2f}R | Median MAE: {hist_mae:.2f}R          │
  │ Verdict: {verdict}                                      │
  └─────────────────────────────────────────────────────────┘""")


# =========================================================
# MASTER DATASET BUILDER
# =========================================================

def build_master_dataset(path: str) -> pd.DataFrame:
    ticks = load_data(path)
    bars_5m = resample_bars(ticks, "5min")
    bars_15m = resample_bars(ticks, "15min")
    bars_30m = resample_bars(ticks, "30min")
    bars_4h = resample_bars(ticks, "4h")
    bars_6h = resample_bars(ticks, "6h")
    bars_12h = resample_bars(ticks, "12h")

    bars_5m = add_rsi_features(bars_5m, RSI_PERIOD)
    bars_5m = add_ema_bos_features_5m(bars_5m)
    bars_5m = add_extra_features_5m(bars_5m)

    bars_15m = add_rsi_features(bars_15m, RSI_PERIOD)
    bars_30m = add_rsi_features(bars_30m, RSI_PERIOD)
    bars_4h = add_rsi_features(bars_4h, RSI_PERIOD)
    bars_4h = compute_adx(bars_4h, ADX_PERIOD)
    bars_6h = add_rsi_features(bars_6h, RSI_PERIOD)
    bars_12h = add_rsi_features(bars_12h, RSI_PERIOD)

    df = bars_5m.copy()
    # FIX F6: merge_asof_feature now shifts higher-TF data
    df = merge_asof_feature(df, bars_15m, "m15", ["rsi", "rsi_slope_1"])
    df = merge_asof_feature(df, bars_30m, "m30", ["rsi"])
    df = merge_asof_feature(df, bars_4h, "h4", ["rsi", "rsi_slope_1", "rsi_slope_2",
                                                   "fresh_long", "fresh_short",
                                                   "bars_since_cross", "adx", "plus_di", "minus_di"])
    df = merge_asof_feature(df, bars_6h, "h6", ["rsi"])
    df = merge_asof_feature(df, bars_12h, "h12", ["rsi"])

    for c in ["h4_fresh_long", "h4_fresh_short"]:
        df[c] = df[c].fillna(False).astype(bool)

    df = df.dropna().reset_index(drop=True)
    df = build_setup_engine(df)
    df = add_structure_gate(df)
    df = build_confidence_engine(df)
    df = short_validation_layer(df)
    df = add_trade_decision(df)

    return df


# =========================================================
# MAIN
# =========================================================

def main():
    start_time = datetime.now()
    print("=" * 100)
    print("  SETUP VALIDATION ENGINE")
    print("  Primary validation: BTCUSDT only")
    print("  Multi-symbol: secondary robustness checks")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)

    # Phase 1: Audit table
    print_audit_table()

    # ─────────────────────────────────────────────────────
    # Collect per-symbol results separately
    # ─────────────────────────────────────────────────────
    symbol_results = {}   # symbol → {"train": df, "test": df}
    symbol_baselines = {} # symbol → baselines dict

    for symbol in SYMBOLS:
        data_path = f"data/features/{symbol.lower()}_1m.csv"
        if not os.path.exists(data_path):
            print(f"\n  [SKIP] {symbol}: {data_path} not found")
            continue

        label = "PRIMARY" if symbol == "BTCUSDT" else "ROBUSTNESS"
        print(f"\n{'─' * 80}")
        print(f"  [{label}] Processing {symbol}")
        print(f"{'─' * 80}")

        df = build_master_dataset(data_path)

        total_bars = len(df)
        setups = (df["setup_type"] != "none").sum()
        triggered = ((df["setup_type"] != "none") & df["trigger_ok"]).sum()
        tradeable = df["take_trade"].sum()
        print(f"    Bars: {total_bars:,} | Setups: {setups:,} | Triggered: {triggered:,} | Tradeable: {tradeable:,}")

        df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy()
        df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy()

        print(f"\n  TRAIN ({df_train['timestamp'].min().strftime('%Y-%m-%d')} → {df_train['timestamp'].max().strftime('%Y-%m-%d')}):")
        train_results = run_validation(df_train, symbol)
        if len(train_results) > 0:
            train_results["period"] = "TRAIN"
            tm = compute_group_metrics(train_results)
            print(f"    {len(train_results)} setups | Hit 1R: {tm['pct_hit_1R']:.1%} | Exp1R: {tm['expectancy_1R']:+.3f}")
        else:
            train_results = pd.DataFrame()

        print(f"\n  TEST ({df_test['timestamp'].min().strftime('%Y-%m-%d')} → {df_test['timestamp'].max().strftime('%Y-%m-%d')}):")
        test_results = run_validation(df_test, symbol)
        if len(test_results) > 0:
            test_results["period"] = "TEST"
            ttm = compute_group_metrics(test_results)
            print(f"    {len(test_results)} setups | Hit 1R: {ttm['pct_hit_1R']:.1%} | Exp1R: {ttm['expectancy_1R']:+.3f}")
        else:
            test_results = pd.DataFrame()

        symbol_results[symbol] = {"train": train_results, "test": test_results}

        # Baselines on test period
        if len(test_results) > 0:
            print(f"\n  Building baselines for {symbol}...")
            baselines = build_random_baselines(test_results, df_test, symbol)
            symbol_baselines[symbol] = baselines

    # ═══════════════════════════════════════════════════════
    # PRIMARY VALIDATION — BTCUSDT ONLY
    # ═══════════════════════════════════════════════════════
    btc_train = symbol_results.get("BTCUSDT", {}).get("train", pd.DataFrame())
    btc_test = symbol_results.get("BTCUSDT", {}).get("test", pd.DataFrame())
    btc_all = pd.concat([btc_train, btc_test], ignore_index=True) if len(btc_train) > 0 or len(btc_test) > 0 else pd.DataFrame()
    btc_baselines = symbol_baselines.get("BTCUSDT", {})

    print(f"\n\n{'═' * 100}")
    print(f"  ██████████████████████████████████████████████████████████████████████████████████████")
    print(f"  ██  PRIMARY VALIDATION — BTCUSDT ONLY                                           ██")
    print(f"  ██████████████████████████████████████████████████████████████████████████████████████")
    print(f"{'═' * 100}")

    if len(btc_all) == 0:
        print("\n  [ERROR] No BTCUSDT results. Cannot validate.")
        return

    # BTC overall
    btc_stats_all = compute_group_metrics(btc_all)
    print(f"\n  BTCUSDT ALL PERIODS: {btc_stats_all['count']} setups")
    print(f"    Hit 1R: {btc_stats_all['pct_hit_1R']:.1%} | Hit 2R: {btc_stats_all['pct_hit_2R']:.1%} | Hit 3R: {btc_stats_all['pct_hit_3R']:.1%}")
    print(f"    Exp 1R: {btc_stats_all['expectancy_1R']:+.3f} | Exp 2R: {btc_stats_all['expectancy_2R']:+.3f} | Exp 3R: {btc_stats_all['expectancy_3R']:+.3f}")
    print(f"    Ambiguity: {btc_stats_all['ambiguous_rate']:.1%} | Expired: {btc_stats_all['expiration_rate']:.1%}")

    # BTC test period (headline)
    if len(btc_test) > 0:
        btc_stats_test = compute_group_metrics(btc_test)
        print(f"\n  BTCUSDT TEST PERIOD (HEADLINE): {btc_stats_test['count']} setups")
        print(f"    Hit 1R: {btc_stats_test['pct_hit_1R']:.1%} | Hit 2R: {btc_stats_test['pct_hit_2R']:.1%} | Hit 3R: {btc_stats_test['pct_hit_3R']:.1%}")
        print(f"    Exp 1R: {btc_stats_test['expectancy_1R']:+.3f} | Exp 2R: {btc_stats_test['expectancy_2R']:+.3f} | Exp 3R: {btc_stats_test['expectancy_3R']:+.3f}")
        print(f"    Median MFE: {btc_stats_test['median_MFE_R']:.2f}R | Median MAE: {btc_stats_test['median_MAE_R']:.2f}R")

    # BTC by direction
    print(f"\n  BTCUSDT BY DIRECTION:")
    for d in ["LONG", "SHORT"]:
        sub_all = btc_all[btc_all["dir_label"] == d]
        m_all = compute_group_metrics(sub_all)
        if m_all:
            sub_test = btc_test[btc_test["dir_label"] == d] if len(btc_test) > 0 else pd.DataFrame()
            m_test = compute_group_metrics(sub_test) if len(sub_test) > 0 else None
            test_str = f" | TEST: {m_test['count']} setups, Exp1R={m_test['expectancy_1R']:+.3f}" if m_test else ""
            print(f"    {d}: {m_all['count']} setups | Exp1R={m_all['expectancy_1R']:+.3f} | Hit1R={m_all['pct_hit_1R']:.1%}{test_str}")

    # BTC by confidence
    print(f"\n  BTCUSDT BY CONFIDENCE TIER:")
    for mode in ["MILD", "MID", "HIGH", "PREMIUM", "ELITE"]:
        sub_all = btc_all[btc_all["confidence_mode"] == mode]
        m_all = compute_group_metrics(sub_all)
        if m_all:
            sub_test = btc_test[btc_test["confidence_mode"] == mode] if len(btc_test) > 0 else pd.DataFrame()
            m_test = compute_group_metrics(sub_test) if len(sub_test) > 0 else None
            test_str = f" | TEST: {m_test['count']}, Exp1R={m_test['expectancy_1R']:+.3f}" if m_test else ""
            print(f"    {mode:<10s}: {m_all['count']:>3d} setups | Exp1R={m_all['expectancy_1R']:+.3f} | Hit1R={m_all['pct_hit_1R']:.1%}{test_str}")

    # BTC by direction+confidence
    print(f"\n  BTCUSDT BY DIRECTION+CONFIDENCE:")
    for d in ["LONG", "SHORT"]:
        for mode in ["MILD", "MID", "HIGH", "PREMIUM"]:
            sub_all = btc_all[(btc_all["dir_label"] == d) & (btc_all["confidence_mode"] == mode)]
            m_all = compute_group_metrics(sub_all)
            if m_all and m_all["count"] >= 3:
                sub_test = btc_test[(btc_test["dir_label"] == d) & (btc_test["confidence_mode"] == mode)] if len(btc_test) > 0 else pd.DataFrame()
                m_test = compute_group_metrics(sub_test) if len(sub_test) > 0 else None
                test_str = f" | TEST: {m_test['count']}, Exp1R={m_test['expectancy_1R']:+.3f}" if m_test else ""
                print(f"    {d}/{mode:<8s}: {m_all['count']:>3d} setups | Exp1R={m_all['expectancy_1R']:+.3f} | Hit1R={m_all['pct_hit_1R']:.1%}{test_str}")

    # BTC by month
    print(f"\n  BTCUSDT BY MONTH:")
    for month in sorted(btc_all["month"].unique()):
        sub = btc_all[btc_all["month"] == month]
        m = compute_group_metrics(sub)
        if m:
            print(f"    {month}: {m['count']:>3d} setups | Hit1R={m['pct_hit_1R']:.1%} | Exp1R={m['expectancy_1R']:+.3f}")

    # BTC by regime
    print(f"\n  BTCUSDT BY HTF REGIME:")
    for regime in ["bullish", "bearish", "neutral"]:
        sub = btc_all[btc_all["htf_regime"] == regime]
        m = compute_group_metrics(sub)
        if m:
            print(f"    {regime:<10s}: {m['count']:>3d} setups | Hit1R={m['pct_hit_1R']:.1%} | Exp1R={m['expectancy_1R']:+.3f}")

    # BTC by session
    print(f"\n  BTCUSDT BY SESSION:")
    for session in ["Asian", "European", "US"]:
        sub = btc_all[btc_all["session"] == session]
        m = compute_group_metrics(sub)
        if m:
            print(f"    {session:<10s}: {m['count']:>3d} setups | Hit1R={m['pct_hit_1R']:.1%} | Exp1R={m['expectancy_1R']:+.3f}")

    # BTC baseline comparison (headline)
    if btc_baselines:
        print(f"\n  {'─' * 80}")
        print(f"  BTCUSDT BASELINE COMPARISON (TEST PERIOD)")
        print(f"  {'─' * 80}")
        btc_sys = compute_group_metrics(btc_test) if len(btc_test) > 0 else None
        if btc_sys:
            print(f"\n  {'Baseline':<35s} {'N':>5s} {'1R%':>6s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s} {'Edge':>8s}")
            print(f"  {'─'*35} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*7} {'─'*8}")
            print(f"  {'BTCUSDT (system)':<35s} {btc_sys['count']:>5d} {btc_sys['pct_hit_1R']:>5.1%} {btc_sys['expectancy_1R']:>+7.3f}R {btc_sys['expectancy_2R']:>+7.3f}R {btc_sys['median_MFE_R']:>6.2f}R")
            for bl_name, bl_df in btc_baselines.items():
                bl_m = compute_group_metrics(bl_df)
                if bl_m:
                    edge = btc_sys["expectancy_1R"] - bl_m["expectancy_1R"]
                    print(f"  {bl_name:<35s} {bl_m['count']:>5d} {bl_m['pct_hit_1R']:>5.1%} {bl_m['expectancy_1R']:>+7.3f}R {bl_m['expectancy_2R']:>+7.3f}R {bl_m['median_MFE_R']:>6.2f}R {edge:>+7.3f}R")

    # ═══════════════════════════════════════════════════════
    # SECONDARY — MULTI-SYMBOL ROBUSTNESS
    # ═══════════════════════════════════════════════════════
    other_symbols = [s for s in SYMBOLS if s != "BTCUSDT" and s in symbol_results]
    if other_symbols:
        print(f"\n\n{'═' * 100}")
        print(f"  SECONDARY — MULTI-SYMBOL ROBUSTNESS CHECKS")
        print(f"  These results test cross-asset consistency. They do NOT establish edge.")
        print(f"{'═' * 100}")

        # Per-symbol summary table
        print(f"\n  {'Symbol':<12s} {'Period':<8s} {'N':>5s} {'1R%':>6s} {'2R%':>6s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s} {'Amb%':>5s}")
        print(f"  {'─'*12} {'─'*8} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*7} {'─'*5}")

        # BTC first
        if len(btc_test) > 0:
            m = compute_group_metrics(btc_test)
            print(f"  {'BTCUSDT':<12s} {'TEST':<8s} {m['count']:>5d} {m['pct_hit_1R']:>5.1%} {m['pct_hit_2R']:>5.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R {m['ambiguous_rate']:>4.1%}")

        # Others
        for symbol in other_symbols:
            sr = symbol_results[symbol]
            for period_name, period_df in [("TRAIN", sr["train"]), ("TEST", sr["test"])]:
                if len(period_df) > 0:
                    m = compute_group_metrics(period_df)
                    print(f"  {symbol:<12s} {period_name:<8s} {m['count']:>5d} {m['pct_hit_1R']:>5.1%} {m['pct_hit_2R']:>5.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R {m['ambiguous_rate']:>4.1%}")

        # Robustness baseline comparison per symbol
        print(f"\n  PER-SYMBOL BASELINE COMPARISON (TEST PERIOD):")
        for symbol in ["BTCUSDT"] + other_symbols:
            if symbol not in symbol_baselines or not symbol_baselines[symbol]:
                continue
            sr = symbol_results[symbol]
            test_df = sr["test"]
            if len(test_df) == 0:
                continue
            sys_m = compute_group_metrics(test_df)
            print(f"\n    {symbol}: {sys_m['count']} setups | System Exp1R={sys_m['expectancy_1R']:+.3f}")
            for bl_name, bl_df in symbol_baselines[symbol].items():
                bl_m = compute_group_metrics(bl_df)
                if bl_m:
                    edge = sys_m["expectancy_1R"] - bl_m["expectancy_1R"]
                    print(f"      {bl_name:<30s}: N={bl_m['count']:>4d} | Exp1R={bl_m['expectancy_1R']:+.3f} | Edge={edge:+.3f}")

        # Combined basket (clearly labeled)
        all_test_frames = []
        for symbol in SYMBOLS:
            if symbol in symbol_results and len(symbol_results[symbol]["test"]) > 0:
                all_test_frames.append(symbol_results[symbol]["test"])
        if all_test_frames:
            basket_test = pd.concat(all_test_frames, ignore_index=True)
            basket_m = compute_group_metrics(basket_test)
            print(f"\n  {'─' * 80}")
            print(f"  BASKET (ALL SYMBOLS COMBINED) — ROBUSTNESS ONLY, NOT HEADLINE")
            print(f"  {'─' * 80}")
            print(f"    {basket_m['count']} setups | Hit1R={basket_m['pct_hit_1R']:.1%} | Exp1R={basket_m['expectancy_1R']:+.3f} | Exp2R={basket_m['expectancy_2R']:+.3f}")

    # ═══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════
    all_frames = []
    for symbol in SYMBOLS:
        if symbol in symbol_results:
            for period_df in [symbol_results[symbol]["train"], symbol_results[symbol]["test"]]:
                if len(period_df) > 0:
                    all_frames.append(period_df)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        output_path = "data/features/setup_validation_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n  [SAVED] {output_path} ({len(combined)} setups)")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'═' * 100}")
    print(f"  SETUP VALIDATION ENGINE COMPLETE")
    print(f"  Time: {elapsed:.1f}s | Primary: BTCUSDT | Robustness: {len(other_symbols)} other symbols")
    print(f"{'═' * 100}")

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)


if __name__ == "__main__":
    main()
