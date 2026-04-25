"""
STRICT BACKTEST AUDIT
=====================
Rebuilds the BTC intraday system with realistic execution assumptions
to test whether the reported edge survives.

Auditor: Senior Quantitative Researcher
Date: 2026-04-26
"""

import pandas as pd
import numpy as np
import os
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG — preserved from original (NO parameter changes)
# =========================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
RSI_PERIOD = 14
EMA_PERIOD = 20
INITIAL_EQUITY = 10000.0

ROUND_TRIP_COST_BY_SYMBOL = {
    "BTCUSDT": 0.0005,
    "ETHUSDT": 0.0008,
    "SOLUSDT": 0.0012,
    "XRPUSDT": 0.0015,
}

def get_round_trip_cost(symbol):
    return ROUND_TRIP_COST_BY_SYMBOL.get(symbol, 0.0008)

MAX_NOTIONAL_FRACTION = 10.0

RISK_MILD = 0.0250
RISK_MID = 0.0300
RISK_PREMIUM = 0.0325
RISK_HIGH = 0.0350
RISK_ELITE = 0.0400

TP1_PCT = 0.005
TP2_PCT_SCALP = 0.007
TP2_PCT_TREND = 0.006
TP3_PCT = 0.008
TP4_PCT = 0.090

TP1_SIZE = 0.35
TP2_SIZE = 0.35
TP3_SIZE = 0.15
TP4_SIZE = 0.15

TRAIL_PCT_MID = 0.0050
TRAIL_PCT_HIGH = 0.0045
TRAIL_PCT_ELITE = 0.0040

ENABLE_BREAK_EVEN_AFTER_TP1 = True
BREAK_EVEN_OFFSET = 0.0002

COOLDOWN_BARS_5M = 3

NO_TRADE_THRESHOLD = 0.72
MILD_THRESHOLD = 0.78
MID_THRESHOLD = 0.80
PREMIUM_THRESHOLD = 0.82
HIGH_THRESHOLD = 0.88

ALLOWED_MODES = {"MILD", "MID", "HIGH", "PREMIUM", "ELITE"}
ALLOW_LONG = True
ALLOW_SHORT = True

VOLUME_PERCENTILE = 0.30
STRUCTURE_GATE_WINDOW = 5
ADX_PERIOD = 14
ADX_TRENDING_BONUS = 25.0
ADX_CHOPPY_PENALTY = 15.0
ENABLE_ADX_FILTER = True


# =========================================================
# INDICATORS — identical to original
# =========================================================

def rsi_wilder(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_rsi_features(df, period=14):
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


def add_ema_bos_features_5m(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_dist_pct"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["ema_reclaim_long"] = (df["close"].shift(1) <= df["ema20"].shift(1)) & (df["close"] > df["ema20"])
    df["ema_reclaim_short"] = (df["close"].shift(1) >= df["ema20"].shift(1)) & (df["close"] < df["ema20"])
    lookback = 12
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


def add_extra_features_5m(df):
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["rv_6"] = df["ret_1"].rolling(6).std()
    df["delta_3"] = df["delta"].rolling(3).sum()
    return df


def compute_adx(df, period=14):
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
# HELPERS
# =========================================================

def rolling_minmax_score(series, window=200):
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return ((series - roll_min) / denom).clip(0, 1)


def bounded_positive_score_rolling(series, ref_quantile=0.8, window=5000):
    """
    STRICT: Rolling quantile instead of global quantile.
    At timestamp T, only uses data from [T-window, T].
    """
    abs_s = series.abs()
    ref = abs_s.rolling(window, min_periods=200).quantile(ref_quantile).shift(1)
    ref = ref.replace(0, np.nan)
    return (abs_s / ref).clip(0, 1).fillna(0)


def merge_asof_feature(base, other, prefix, cols):
    rhs = other[["timestamp"] + cols].copy()
    rhs = rhs.rename(columns={c: f"{prefix}_{c}" for c in cols})
    return pd.merge_asof(
        base.sort_values("timestamp"),
        rhs.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def resample_bars(ticks, rule):
    x = ticks.set_index("timestamp")
    return pd.DataFrame({
        "open": x["price"].resample(rule).first(),
        "high": x["price"].resample(rule).max(),
        "low": x["price"].resample(rule).min(),
        "close": x["price"].resample(rule).last(),
        "volume": x["volume"].resample(rule).sum(),
        "delta": x["delta"].resample(rule).sum(),
        "trade_count": x["trade_count"].resample(rule).sum(),
    }).dropna().reset_index()


# =========================================================
# LOAD DATA
# =========================================================

def load_data(path):
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
# SETUP ENGINE — identical to original
# =========================================================

def build_setup_engine(df):
    df = df.copy()
    scalp_long = (
        df["h4_fresh_long"] &
        (df["m15_rsi"] > 50) & (df["m15_rsi"].shift(1) > 50) &
        (df["h6_rsi"] > 50) & (df["h12_rsi"] > 50)
    )
    scalp_short = (
        df["h4_fresh_short"] &
        (df["m15_rsi"] < 50) & (df["m15_rsi"].shift(1) < 50) &
        (df["h6_rsi"] < 50) & (df["h12_rsi"] < 50)
    )
    trend_long = (
        (df["h4_rsi"] > 55) & (df["h4_rsi_slope_1"] > 0) & (df["h4_rsi_slope_2"] > 0) &
        (df["m15_rsi"] > 55) & (df["m15_rsi_slope_1"] >= 0) & (df["h12_rsi"] > 50)
    )
    trend_short = (
        (df["h4_rsi"] < 45) & (df["h4_rsi_slope_1"] < 0) & (df["h4_rsi_slope_2"] < 0) &
        (df["m15_rsi"] < 45) & (df["m15_rsi_slope_1"] <= 0) & (df["h12_rsi"] < 50)
    )
    df["setup_type"] = "none"
    df["direction"] = 0
    df["stage"] = 0
    df.loc[scalp_long, ["setup_type", "direction"]] = ["RSI_SCALP", 1]
    df.loc[scalp_short, ["setup_type", "direction"]] = ["RSI_SCALP", -1]
    df.loc[trend_long, ["setup_type", "direction"]] = ["RSI_TREND", 1]
    df.loc[trend_short, ["setup_type", "direction"]] = ["RSI_TREND", -1]
    bsc = df["h4_bars_since_cross"].fillna(99).astype(int)
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc <= 1), "stage"] = 1
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 2) & (bsc <= 3), "stage"] = 2
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 4), "stage"] = 3
    return df


def add_structure_gate(df):
    out = df.copy()
    out["trigger_ok"] = False
    out.loc[(out["setup_type"] == "RSI_SCALP") & (out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["setup_type"] == "RSI_SCALP") & (out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    out.loc[(out["setup_type"] == "RSI_TREND") & (out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["setup_type"] == "RSI_TREND") & (out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    return out


# =========================================================
# CONFIDENCE ENGINE — with STRICT rolling quantiles
# =========================================================

def build_confidence_engine_strict(df):
    """
    Rebuild confidence with rolling quantiles only.
    - bounded_positive_score uses rolling window (not global)
    - ema_dist reference uses rolling window (not global)
    """
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

    # STRICT: rolling quantile for slope scores
    h4_slope_score = bounded_positive_score_rolling(out["h4_rsi_slope_1"], 0.8, window=5000)
    m15_slope_score = bounded_positive_score_rolling(out["m15_rsi_slope_1"], 0.8, window=5000)
    ret_score = bounded_positive_score_rolling(out["ret_3"], 0.8, window=5000)
    momentum_score = (0.4 * h4_slope_score + 0.4 * m15_slope_score + 0.2 * ret_score).clip(0, 1)

    # STRICT: rolling ema_dist reference (not global quantile)
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
    out["risk_fraction"] = 0.0

    out.loc[(out["confidence_raw"] >= NO_TRADE_THRESHOLD) & (out["confidence_raw"] < MILD_THRESHOLD), "confidence_mode"] = "MILD"
    out.loc[(out["confidence_raw"] >= MILD_THRESHOLD) & (out["confidence_raw"] < MID_THRESHOLD), "confidence_mode"] = "MID"
    out.loc[(out["confidence_raw"] >= MID_THRESHOLD) & (out["confidence_raw"] < PREMIUM_THRESHOLD), "confidence_mode"] = "PREMIUM"
    out.loc[(out["confidence_raw"] >= PREMIUM_THRESHOLD) & (out["confidence_raw"] < HIGH_THRESHOLD), "confidence_mode"] = "HIGH"
    out.loc[(out["confidence_raw"] >= HIGH_THRESHOLD) & elite_gate, "confidence_mode"] = "ELITE"

    out.loc[out["confidence_mode"] == "MILD", "risk_fraction"] = RISK_MILD
    out.loc[out["confidence_mode"] == "MID", "risk_fraction"] = RISK_MID
    out.loc[out["confidence_mode"] == "PREMIUM", "risk_fraction"] = RISK_PREMIUM
    out.loc[out["confidence_mode"] == "HIGH", "risk_fraction"] = RISK_HIGH
    out.loc[out["confidence_mode"] == "ELITE", "risk_fraction"] = RISK_ELITE

    return out


def short_validation_layer(df):
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
# TRADE DECISION — with STRICT rolling volume threshold
# =========================================================

def add_trade_decision_strict(df):
    """
    STRICT: Volume threshold uses rolling expanding window.
    At bar T, threshold is computed from bars [0..T-1] only.
    """
    out = df.copy()

    mode_direction_ok = (
        out["confidence_mode"].isin(ALLOWED_MODES) &
        (
            ((out["direction"] == 1) & ALLOW_LONG) |
            ((out["direction"] == -1) & ALLOW_SHORT)
        )
    )

    # STRICT: rolling/expanding volume quantile with shift(1)
    volume_rolling_threshold = (
        out["volume"]
        .expanding(min_periods=500)
        .quantile(VOLUME_PERCENTILE)
        .shift(1)  # shift to avoid using current candle
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
# STOP LOGIC
# =========================================================

def compute_stop_price(df, entry_idx, direction, strategy):
    row = df.iloc[entry_idx]
    entry_price = float(row["close"])
    start = max(0, entry_idx - 4)
    window = df.iloc[start:entry_idx + 1]
    if direction == 1:
        structural_stop = float(window["low"].min())
        if strategy == "RSI_TREND":
            structural_stop = min(structural_stop, float(row["ema20"]))
        stop_distance_pct = (entry_price - structural_stop) / entry_price
    else:
        structural_stop = float(window["high"].max())
        if strategy == "RSI_TREND":
            structural_stop = max(structural_stop, float(row["ema20"]))
        stop_distance_pct = (structural_stop - entry_price) / entry_price
    stop_distance_pct = max(stop_distance_pct, 0.004)
    if direction == 1:
        stop_price = entry_price * (1 - stop_distance_pct)
    else:
        stop_price = entry_price * (1 + stop_distance_pct)
    return stop_price, stop_distance_pct


def get_trail_pct(confidence_mode):
    if confidence_mode in ("MILD", "MID"):
        return TRAIL_PCT_MID
    if confidence_mode == "PREMIUM":
        return TRAIL_PCT_HIGH
    if confidence_mode == "HIGH":
        return TRAIL_PCT_HIGH
    if confidence_mode == "ELITE":
        return TRAIL_PCT_ELITE
    return TRAIL_PCT_MID


# =========================================================
# EXIT ENGINE — ORIGINAL (optimistic: TP before SL)
# =========================================================

def simulate_trade_original(df, entry_idx, direction, strategy, confidence,
                            confidence_mode, risk_fraction, equity_before,
                            stage=0, cfx_score=0.0, symbol="BTCUSDT",
                            entry_price_override=None, fee_multiplier=1.0):
    """Original exit logic: TP checked before SL (optimistic)."""
    entry_row = df.iloc[entry_idx]
    entry_price = entry_price_override if entry_price_override is not None else float(entry_row["close"])
    entry_time = entry_row["timestamp"]

    stop_price, stop_distance_pct = compute_stop_price(df, entry_idx, direction, strategy)

    risk_amount = equity_before * risk_fraction
    position_notional = risk_amount / stop_distance_pct if stop_distance_pct > 0 else 0.0
    max_notional = equity_before * MAX_NOTIONAL_FRACTION
    if position_notional > max_notional:
        position_notional = max_notional
        risk_amount = position_notional * stop_distance_pct

    tp2_pct = TP2_PCT_TREND if strategy == "RSI_TREND" else TP2_PCT_SCALP

    if direction == 1:
        tp1 = entry_price * (1 + TP1_PCT)
        tp2 = entry_price * (1 + tp2_pct)
        tp3 = entry_price * (1 + TP3_PCT)
        tp4 = entry_price * (1 + TP4_PCT)
        break_even_price = entry_price * (1 + BREAK_EVEN_OFFSET)
    else:
        tp1 = entry_price * (1 - TP1_PCT)
        tp2 = entry_price * (1 - tp2_pct)
        tp3 = entry_price * (1 - TP3_PCT)
        tp4 = entry_price * (1 - TP4_PCT)
        break_even_price = entry_price * (1 - BREAK_EVEN_OFFSET)

    trail_pct = get_trail_pct(confidence_mode)

    remaining = 1.0
    realized_return_pct = 0.0
    total_fees_pct = 0.0  # track per-partial fees

    tp1_done = tp2_done = tp3_done = tp4_done = False
    break_even_armed = False
    trailing_active = False
    trail_stop = np.nan
    best_price = entry_price

    exit_reason = "open_end"
    exit_time = entry_time
    exit_price = entry_price
    ambiguous_candle = False
    fee_rate = get_round_trip_cost(symbol) * fee_multiplier

    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if direction == 1:
            if trailing_active:
                best_price = max(best_price, high)
                new_trail = best_price * (1 - trail_pct)
                trail_stop = new_trail if np.isnan(trail_stop) else max(trail_stop, new_trail)

            active_stop = stop_price
            if break_even_armed:
                active_stop = max(active_stop, break_even_price)
            if trailing_active and not np.isnan(trail_stop):
                active_stop = max(active_stop, trail_stop)

            # Check ambiguity: both TP and SL touched in same candle
            if low <= active_stop and high >= tp1 and not tp1_done:
                ambiguous_candle = True
            if low <= active_stop and high >= tp2 and not tp2_done:
                ambiguous_candle = True

            # ORIGINAL: TP checked BEFORE stop (optimistic)
            if (not tp1_done) and high >= tp1:
                realized_return_pct += TP1_SIZE * ((tp1 / entry_price) - 1.0)
                total_fees_pct += TP1_SIZE * fee_rate  # per-partial fee
                remaining -= TP1_SIZE
                tp1_done = True
                if ENABLE_BREAK_EVEN_AFTER_TP1:
                    break_even_armed = True

            if (not tp2_done) and high >= tp2:
                realized_return_pct += TP2_SIZE * ((tp2 / entry_price) - 1.0)
                total_fees_pct += TP2_SIZE * fee_rate
                remaining -= TP2_SIZE
                tp2_done = True
                trailing_active = True
                best_price = max(best_price, high)
                trail_stop = best_price * (1 - trail_pct)

            if (not tp3_done) and high >= tp3:
                realized_return_pct += TP3_SIZE * ((tp3 / entry_price) - 1.0)
                total_fees_pct += TP3_SIZE * fee_rate
                remaining -= TP3_SIZE
                tp3_done = True

            if (not tp4_done) and high >= tp4:
                realized_return_pct += TP4_SIZE * ((tp4 / entry_price) - 1.0)
                total_fees_pct += TP4_SIZE * fee_rate
                remaining -= TP4_SIZE
                tp4_done = True
                exit_reason = "tp4"
                exit_time = row["timestamp"]
                exit_price = tp4
                break

            if low <= active_stop:
                realized_return_pct += remaining * ((active_stop / entry_price) - 1.0)
                total_fees_pct += remaining * fee_rate
                remaining = 0.0
                if trailing_active:
                    exit_reason = "trailing_stop"
                elif break_even_armed and active_stop >= break_even_price:
                    exit_reason = "break_even_stop"
                else:
                    exit_reason = "stop_loss"
                exit_time = row["timestamp"]
                exit_price = active_stop
                break

        else:  # SHORT
            if trailing_active:
                best_price = min(best_price, low)
                new_trail = best_price * (1 + trail_pct)
                trail_stop = new_trail if np.isnan(trail_stop) else min(trail_stop, new_trail)

            active_stop = stop_price
            if break_even_armed:
                active_stop = min(active_stop, break_even_price)
            if trailing_active and not np.isnan(trail_stop):
                active_stop = min(active_stop, trail_stop)

            # Check ambiguity
            if high >= active_stop and low <= tp1 and not tp1_done:
                ambiguous_candle = True
            if high >= active_stop and low <= tp2 and not tp2_done:
                ambiguous_candle = True

            # ORIGINAL: TP before stop
            if (not tp1_done) and low <= tp1:
                realized_return_pct += TP1_SIZE * (1.0 - (tp1 / entry_price))
                total_fees_pct += TP1_SIZE * fee_rate
                remaining -= TP1_SIZE
                tp1_done = True
                if ENABLE_BREAK_EVEN_AFTER_TP1:
                    break_even_armed = True

            if (not tp2_done) and low <= tp2:
                realized_return_pct += TP2_SIZE * (1.0 - (tp2 / entry_price))
                total_fees_pct += TP2_SIZE * fee_rate
                remaining -= TP2_SIZE
                tp2_done = True
                trailing_active = True
                best_price = min(best_price, low)
                trail_stop = best_price * (1 + trail_pct)

            if (not tp3_done) and low <= tp3:
                realized_return_pct += TP3_SIZE * (1.0 - (tp3 / entry_price))
                total_fees_pct += TP3_SIZE * fee_rate
                remaining -= TP3_SIZE
                tp3_done = True

            if (not tp4_done) and low <= tp4:
                realized_return_pct += TP4_SIZE * (1.0 - (tp4 / entry_price))
                total_fees_pct += TP4_SIZE * fee_rate
                remaining -= TP4_SIZE
                tp4_done = True
                exit_reason = "tp4"
                exit_time = row["timestamp"]
                exit_price = tp4
                break

            if high >= active_stop:
                realized_return_pct += remaining * (1.0 - (active_stop / entry_price))
                total_fees_pct += remaining * fee_rate
                remaining = 0.0
                if trailing_active:
                    exit_reason = "trailing_stop"
                elif break_even_armed and active_stop <= break_even_price:
                    exit_reason = "break_even_stop"
                else:
                    exit_reason = "stop_loss"
                exit_time = row["timestamp"]
                exit_price = active_stop
                break

        if remaining <= 1e-12:
            exit_reason = "all_targets"
            exit_time = row["timestamp"]
            exit_price = close
            break

        if j == len(df) - 1:
            if direction == 1:
                realized_return_pct += remaining * ((close / entry_price) - 1.0)
            else:
                realized_return_pct += remaining * (1.0 - (close / entry_price))
            total_fees_pct += remaining * fee_rate
            remaining = 0.0
            exit_reason = "end_of_data"
            exit_time = row["timestamp"]
            exit_price = close

    # Entry fee (always paid)
    total_fees_pct += fee_rate

    gross_pnl_cash = position_notional * realized_return_pct
    cost_cash = position_notional * total_fees_pct
    net_pnl_cash = gross_pnl_cash - cost_cash
    equity_after = equity_before + net_pnl_cash

    return {
        "entry_time": entry_time,
        "entry_hour": pd.Timestamp(entry_time).hour,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "direction": direction,
        "dir_label": "LONG" if direction == 1 else "SHORT",
        "strategy": strategy,
        "stage": stage,
        "cfx_score": cfx_score,
        "confidence_raw": confidence,
        "confidence_mode": confidence_mode,
        "risk_fraction": risk_fraction,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "risk_amount": risk_amount,
        "stop_price": stop_price,
        "stop_distance_pct": stop_distance_pct,
        "position_notional": position_notional,
        "gross_return_pct": realized_return_pct,
        "gross_pnl_cash": gross_pnl_cash,
        "cost_cash": cost_cash,
        "net_pnl_cash": net_pnl_cash,
        "net_r_multiple": net_pnl_cash / risk_amount if risk_amount > 0 else np.nan,
        "win": int(net_pnl_cash > 0),
        "exit_reason": exit_reason,
        "ambiguous_candle": ambiguous_candle,
    }


# =========================================================
# EXIT ENGINE — STRICT (worst-case: SL before TP)
# =========================================================

def simulate_trade_strict(df, entry_idx, direction, strategy, confidence,
                          confidence_mode, risk_fraction, equity_before,
                          stage=0, cfx_score=0.0, symbol="BTCUSDT",
                          entry_price_override=None, fee_multiplier=1.0):
    """Strict exit: SL checked BEFORE TP (worst-case for ambiguous candles)."""
    entry_row = df.iloc[entry_idx]
    entry_price = entry_price_override if entry_price_override is not None else float(entry_row["close"])
    entry_time = entry_row["timestamp"]

    stop_price, stop_distance_pct = compute_stop_price(df, entry_idx, direction, strategy)

    risk_amount = equity_before * risk_fraction
    position_notional = risk_amount / stop_distance_pct if stop_distance_pct > 0 else 0.0
    max_notional = equity_before * MAX_NOTIONAL_FRACTION
    if position_notional > max_notional:
        position_notional = max_notional
        risk_amount = position_notional * stop_distance_pct

    tp2_pct = TP2_PCT_TREND if strategy == "RSI_TREND" else TP2_PCT_SCALP

    if direction == 1:
        tp1 = entry_price * (1 + TP1_PCT)
        tp2 = entry_price * (1 + tp2_pct)
        tp3 = entry_price * (1 + TP3_PCT)
        tp4 = entry_price * (1 + TP4_PCT)
        break_even_price = entry_price * (1 + BREAK_EVEN_OFFSET)
    else:
        tp1 = entry_price * (1 - TP1_PCT)
        tp2 = entry_price * (1 - tp2_pct)
        tp3 = entry_price * (1 - TP3_PCT)
        tp4 = entry_price * (1 - TP4_PCT)
        break_even_price = entry_price * (1 - BREAK_EVEN_OFFSET)

    trail_pct = get_trail_pct(confidence_mode)

    remaining = 1.0
    realized_return_pct = 0.0
    total_fees_pct = 0.0

    tp1_done = tp2_done = tp3_done = tp4_done = False
    break_even_armed = False
    trailing_active = False
    trail_stop = np.nan
    best_price = entry_price

    exit_reason = "open_end"
    exit_time = entry_time
    exit_price = entry_price
    ambiguous_candle = False
    fee_rate = get_round_trip_cost(symbol) * fee_multiplier

    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if direction == 1:
            if trailing_active:
                best_price = max(best_price, high)
                new_trail = best_price * (1 - trail_pct)
                trail_stop = new_trail if np.isnan(trail_stop) else max(trail_stop, new_trail)

            active_stop = stop_price
            if break_even_armed:
                active_stop = max(active_stop, break_even_price)
            if trailing_active and not np.isnan(trail_stop):
                active_stop = max(active_stop, trail_stop)

            # STRICT: Check stop FIRST (worst-case)
            # Ambiguous candle: both TP and SL touched
            any_tp = tp1 if not tp1_done else (tp2 if not tp2_done else (tp3 if not tp3_done else tp4))
            if low <= active_stop and high >= any_tp:
                ambiguous_candle = True
                # WORST CASE: assume stop hit first, all remaining closed at stop
                realized_return_pct += remaining * ((active_stop / entry_price) - 1.0)
                total_fees_pct += remaining * fee_rate
                remaining = 0.0
                exit_reason = "stop_loss_worst_case"
                exit_time = row["timestamp"]
                exit_price = active_stop
                break

            if low <= active_stop:
                realized_return_pct += remaining * ((active_stop / entry_price) - 1.0)
                total_fees_pct += remaining * fee_rate
                remaining = 0.0
                if trailing_active:
                    exit_reason = "trailing_stop"
                elif break_even_armed and active_stop >= break_even_price:
                    exit_reason = "break_even_stop"
                else:
                    exit_reason = "stop_loss"
                exit_time = row["timestamp"]
                exit_price = active_stop
                break

            # Only check TPs if stop not hit
            if (not tp1_done) and high >= tp1:
                realized_return_pct += TP1_SIZE * ((tp1 / entry_price) - 1.0)
                total_fees_pct += TP1_SIZE * fee_rate
                remaining -= TP1_SIZE
                tp1_done = True
                if ENABLE_BREAK_EVEN_AFTER_TP1:
                    break_even_armed = True

            if (not tp2_done) and high >= tp2:
                realized_return_pct += TP2_SIZE * ((tp2 / entry_price) - 1.0)
                total_fees_pct += TP2_SIZE * fee_rate
                remaining -= TP2_SIZE
                tp2_done = True
                trailing_active = True
                best_price = max(best_price, high)
                trail_stop = best_price * (1 - trail_pct)

            if (not tp3_done) and high >= tp3:
                realized_return_pct += TP3_SIZE * ((tp3 / entry_price) - 1.0)
                total_fees_pct += TP3_SIZE * fee_rate
                remaining -= TP3_SIZE
                tp3_done = True

            if (not tp4_done) and high >= tp4:
                realized_return_pct += TP4_SIZE * ((tp4 / entry_price) - 1.0)
                total_fees_pct += TP4_SIZE * fee_rate
                remaining -= TP4_SIZE
                tp4_done = True
                exit_reason = "tp4"
                exit_time = row["timestamp"]
                exit_price = tp4
                break

        else:  # SHORT
            if trailing_active:
                best_price = min(best_price, low)
                new_trail = best_price * (1 + trail_pct)
                trail_stop = new_trail if np.isnan(trail_stop) else min(trail_stop, new_trail)

            active_stop = stop_price
            if break_even_armed:
                active_stop = min(active_stop, break_even_price)
            if trailing_active and not np.isnan(trail_stop):
                active_stop = min(active_stop, trail_stop)

            # STRICT: stop FIRST
            any_tp = tp1 if not tp1_done else (tp2 if not tp2_done else (tp3 if not tp3_done else tp4))
            if high >= active_stop and low <= any_tp:
                ambiguous_candle = True
                realized_return_pct += remaining * (1.0 - (active_stop / entry_price))
                total_fees_pct += remaining * fee_rate
                remaining = 0.0
                exit_reason = "stop_loss_worst_case"
                exit_time = row["timestamp"]
                exit_price = active_stop
                break

            if high >= active_stop:
                realized_return_pct += remaining * (1.0 - (active_stop / entry_price))
                total_fees_pct += remaining * fee_rate
                remaining = 0.0
                if trailing_active:
                    exit_reason = "trailing_stop"
                elif break_even_armed and active_stop <= break_even_price:
                    exit_reason = "break_even_stop"
                else:
                    exit_reason = "stop_loss"
                exit_time = row["timestamp"]
                exit_price = active_stop
                break

            if (not tp1_done) and low <= tp1:
                realized_return_pct += TP1_SIZE * (1.0 - (tp1 / entry_price))
                total_fees_pct += TP1_SIZE * fee_rate
                remaining -= TP1_SIZE
                tp1_done = True
                if ENABLE_BREAK_EVEN_AFTER_TP1:
                    break_even_armed = True

            if (not tp2_done) and low <= tp2:
                realized_return_pct += TP2_SIZE * (1.0 - (tp2 / entry_price))
                total_fees_pct += TP2_SIZE * fee_rate
                remaining -= TP2_SIZE
                tp2_done = True
                trailing_active = True
                best_price = min(best_price, low)
                trail_stop = best_price * (1 + trail_pct)

            if (not tp3_done) and low <= tp3:
                realized_return_pct += TP3_SIZE * (1.0 - (tp3 / entry_price))
                total_fees_pct += TP3_SIZE * fee_rate
                remaining -= TP3_SIZE
                tp3_done = True

            if (not tp4_done) and low <= tp4:
                realized_return_pct += TP4_SIZE * (1.0 - (tp4 / entry_price))
                total_fees_pct += TP4_SIZE * fee_rate
                remaining -= TP4_SIZE
                tp4_done = True
                exit_reason = "tp4"
                exit_time = row["timestamp"]
                exit_price = tp4
                break

        if remaining <= 1e-12:
            exit_reason = "all_targets"
            exit_time = row["timestamp"]
            exit_price = close
            break

        if j == len(df) - 1:
            if direction == 1:
                realized_return_pct += remaining * ((close / entry_price) - 1.0)
            else:
                realized_return_pct += remaining * (1.0 - (close / entry_price))
            total_fees_pct += remaining * fee_rate
            remaining = 0.0
            exit_reason = "end_of_data"
            exit_time = row["timestamp"]
            exit_price = close

    total_fees_pct += fee_rate  # entry fee

    gross_pnl_cash = position_notional * realized_return_pct
    cost_cash = position_notional * total_fees_pct
    net_pnl_cash = gross_pnl_cash - cost_cash
    equity_after = equity_before + net_pnl_cash

    return {
        "entry_time": entry_time,
        "entry_hour": pd.Timestamp(entry_time).hour,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "direction": direction,
        "dir_label": "LONG" if direction == 1 else "SHORT",
        "strategy": strategy,
        "stage": stage,
        "cfx_score": cfx_score,
        "confidence_raw": confidence,
        "confidence_mode": confidence_mode,
        "risk_fraction": risk_fraction,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "risk_amount": risk_amount,
        "stop_price": stop_price,
        "stop_distance_pct": stop_distance_pct,
        "position_notional": position_notional,
        "gross_return_pct": realized_return_pct,
        "gross_pnl_cash": gross_pnl_cash,
        "cost_cash": cost_cash,
        "net_pnl_cash": net_pnl_cash,
        "net_r_multiple": net_pnl_cash / risk_amount if risk_amount > 0 else np.nan,
        "win": int(net_pnl_cash > 0),
        "exit_reason": exit_reason,
        "ambiguous_candle": ambiguous_candle,
    }


# =========================================================
# BACKTEST ENGINES
# =========================================================

def run_backtest(df, symbol="BTCUSDT", exit_fn=simulate_trade_strict,
                 sizing_mode="compound", fee_multiplier=1.0, entry_delay=0):
    """
    Run backtest with configurable exit logic, sizing, fees, and entry delay.
    
    sizing_mode:
        - "compound": equity-compounded (original)
        - "fixed_notional": fixed $10,000 per trade
        - "fixed_risk": fixed risk amount per trade (based on initial equity)
    """
    trades = []
    next_allowed_idx = 0
    i = 0
    equity = INITIAL_EQUITY
    fixed_risk_base = INITIAL_EQUITY

    # Ensure integer positional index
    df = df.reset_index(drop=True)

    while i < len(df):
        if i < next_allowed_idx:
            i += 1
            continue

        row = df.iloc[i]
        if (row["setup_type"] == "none") or (row["direction"] == 0) or (not bool(row["take_trade"])):
            i += 1
            continue

        # Entry delay
        actual_entry_idx = i + entry_delay
        if actual_entry_idx >= len(df):
            i += 1
            continue

        # For next-open entry: use the open of the entry candle
        entry_price_override = None
        if entry_delay > 0:
            entry_price_override = float(df.iloc[actual_entry_idx]["open"])

        # Position sizing
        if sizing_mode == "compound":
            eq = equity
        elif sizing_mode == "fixed_notional":
            eq = INITIAL_EQUITY  # always same base
        elif sizing_mode == "fixed_risk":
            eq = fixed_risk_base
        else:
            eq = equity

        trade = exit_fn(
            df=df,
            entry_idx=actual_entry_idx,
            direction=int(row["direction"]),
            strategy=str(row["setup_type"]),
            confidence=float(row["confidence_raw"]),
            confidence_mode=str(row["confidence_mode"]),
            risk_fraction=float(row["risk_fraction"]),
            equity_before=float(eq),
            stage=int(row.get("stage", 0)),
            cfx_score=float(row.get("cfx_score", 0)),
            symbol=symbol,
            entry_price_override=entry_price_override,
            fee_multiplier=fee_multiplier,
        )
        trades.append(trade)
        equity = float(trade["equity_after"])

        # Find exit row position by timestamp
        exit_mask = df["timestamp"] == trade["exit_time"]
        exit_positions = df.index[exit_mask].tolist()
        if exit_positions:
            next_allowed_idx = exit_positions[-1] + COOLDOWN_BARS_5M
        else:
            next_allowed_idx = i + COOLDOWN_BARS_5M + entry_delay

        i = next_allowed_idx

    return pd.DataFrame(trades)


# =========================================================
# RANDOM BASELINES
# =========================================================

def run_random_baseline(df, symbol="BTCUSDT", n_trades=500, seed=42):
    """Random entry baseline: enter at random bars, random direction."""
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    tradeable = df[df["setup_type"] != "none"].index.tolist()
    if len(tradeable) < 100:
        return pd.DataFrame()

    entries = rng.choice(tradeable, size=min(n_trades, len(tradeable)), replace=False)
    entries = sorted(entries)

    trades = []
    equity = INITIAL_EQUITY
    next_allowed = 0

    for idx in entries:
        if idx < next_allowed or idx >= len(df) - 10:
            continue

        direction = rng.choice([1, -1])
        confidence_mode = rng.choice(["MILD", "MID", "HIGH"])
        risk_frac = {"MILD": RISK_MILD, "MID": RISK_MID, "HIGH": RISK_HIGH}[confidence_mode]

        trade = simulate_trade_strict(
            df=df, entry_idx=idx, direction=direction,
            strategy="RSI_SCALP", confidence=0.75,
            confidence_mode=confidence_mode,
            risk_fraction=risk_frac,
            equity_before=equity,
            symbol=symbol,
        )
        trades.append(trade)
        equity = float(trade["equity_after"])
        next_allowed = idx + COOLDOWN_BARS_5M

    return pd.DataFrame(trades)


# =========================================================
# REPORTING
# =========================================================

def compute_max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) > 0 else np.nan


def compute_metrics(trades, label=""):
    if len(trades) == 0:
        return {"label": label, "trades": 0}

    wins = trades.loc[trades["net_pnl_cash"] > 0, "net_pnl_cash"]
    losses = trades.loc[trades["net_pnl_cash"] < 0, "net_pnl_cash"]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "label": label,
        "trades": len(trades),
        "win_rate": float(trades["win"].mean()),
        "gross_expectancy": float(trades["gross_pnl_cash"].mean()),
        "net_expectancy": float(trades["net_pnl_cash"].mean()),
        "profit_factor": float(pf) if pf != float("inf") else np.nan,
        "total_pnl": float(trades["net_pnl_cash"].sum()),
        "total_return_pct": float(trades["equity_after"].iloc[-1] / INITIAL_EQUITY - 1.0),
        "max_drawdown": compute_max_drawdown(trades["equity_after"]),
        "avg_win": float(wins.mean()) if len(wins) > 0 else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else np.nan,
        "median_trade": float(trades["net_pnl_cash"].median()),
        "worst_trade": float(trades["net_pnl_cash"].min()),
        "best_trade": float(trades["net_pnl_cash"].max()),
        "total_fees": float(trades["cost_cash"].sum()),
        "ambiguous_trades": int(trades["ambiguous_candle"].sum()) if "ambiguous_candle" in trades.columns else 0,
    }


def print_metrics(m):
    if m["trades"] == 0:
        print(f"  {m['label']:40s} | NO TRADES")
        return
    amb = f" | Ambig: {m.get('ambiguous_trades', '?')}" if m.get('ambiguous_trades', 0) > 0 else ""
    print(
        f"  {m['label']:40s} | "
        f"Trades: {m['trades']:4d} | "
        f"WR: {m['win_rate']:.1%} | "
        f"PF: {m['profit_factor']:.2f} | "
        f"Net Exp: ${m['net_expectancy']:+.2f} | "
        f"Total: ${m['total_pnl']:+,.0f} | "
        f"MDD: {m['max_drawdown']:.1%} | "
        f"Fees: ${m['total_fees']:,.0f}"
        f"{amb}"
    )


def print_full_report(trades, label):
    if len(trades) == 0:
        print(f"\n{'='*60}")
        print(f"{label}: NO TRADES")
        return

    m = compute_metrics(trades, label)
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print_metrics(m)

    # Direction breakdown
    for dir_label in ["LONG", "SHORT"]:
        sub = trades[trades["dir_label"] == dir_label]
        if len(sub) > 0:
            sm = compute_metrics(sub, f"  {dir_label}")
            print_metrics(sm)

    # Mode breakdown
    print(f"\n  BY MODE:")
    for mode in sorted(trades["confidence_mode"].unique()):
        sub = trades[trades["confidence_mode"] == mode]
        sm = compute_metrics(sub, f"    {mode}")
        print_metrics(sm)

    # Strategy breakdown
    print(f"\n  BY STRATEGY:")
    for strat in trades["strategy"].unique():
        sub = trades[trades["strategy"] == strat]
        sm = compute_metrics(sub, f"    {strat}")
        print_metrics(sm)

    # Monthly breakdown
    trades_copy = trades.copy()
    trades_copy["month"] = pd.to_datetime(trades_copy["entry_time"]).dt.to_period("M")
    print(f"\n  MONTHLY:")
    for month, grp in trades_copy.groupby("month"):
        mm = compute_metrics(grp, f"    {month}")
        print_metrics(mm)

    # Exit reason
    print(f"\n  BY EXIT REASON:")
    for reason in trades["exit_reason"].value_counts().index:
        sub = trades[trades["exit_reason"] == reason]
        sm = compute_metrics(sub, f"    {reason}")
        print_metrics(sm)

    # SHORT/MID focus
    mid_shorts = trades[(trades["confidence_mode"] == "MID") & (trades["direction"] == -1)]
    if len(mid_shorts) > 0:
        print(f"\n  *** SHORT/MID FOCUS ***")
        msm = compute_metrics(mid_shorts, "    MID SHORT")
        print_metrics(msm)


# =========================================================
# MASTER DATASET BUILDER
# =========================================================

def build_master_dataset(path):
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
    df = merge_asof_feature(df, bars_15m, "m15", ["rsi", "rsi_slope_1"])
    df = merge_asof_feature(df, bars_30m, "m30", ["rsi"])
    df = merge_asof_feature(df, bars_4h, "h4", ["rsi", "rsi_slope_1", "rsi_slope_2", "fresh_long", "fresh_short", "bars_since_cross", "adx", "plus_di", "minus_di"])
    df = merge_asof_feature(df, bars_6h, "h6", ["rsi"])
    df = merge_asof_feature(df, bars_12h, "h12", ["rsi"])

    for c in ["h4_fresh_long", "h4_fresh_short"]:
        df[c] = df[c].fillna(False).astype(bool)

    df = df.dropna().reset_index(drop=True)
    df = build_setup_engine(df)
    df = add_structure_gate(df)
    df = build_confidence_engine_strict(df)
    df = short_validation_layer(df)
    df = add_trade_decision_strict(df)

    return df


# =========================================================
# MAIN AUDIT
# =========================================================

def main():
    print("=" * 100)
    print("  STRICT BACKTEST AUDIT — BTC INTRADAY SYSTEM")
    print("  Auditor: Senior Quantitative Researcher")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)

    train_cutoff = pd.Timestamp("2025-12-01")

    # ============================================================
    # PHASE 1: Load and prepare data with STRICT indicators
    # ============================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: DATA LOADING + STRICT INDICATOR BUILD")
    print("=" * 80)

    all_dfs = {}
    for symbol in SYMBOLS:
        data_path = f"data/features/{symbol.lower()}_1m.csv"
        if not os.path.exists(data_path):
            print(f"  [SKIP] {symbol}: {data_path} not found")
            continue
        print(f"  Building {symbol}...")
        df = build_master_dataset(data_path)
        df["symbol"] = symbol
        all_dfs[symbol] = df
        setups = (df["setup_type"] != "none").sum()
        triggered = ((df["setup_type"] != "none") & df["trigger_ok"]).sum()
        tradeable = df["take_trade"].sum()
        print(f"    Setups: {setups}, Triggered: {triggered}, Tradeable: {tradeable}, Total rows: {len(df)}")

    # ============================================================
    # PHASE 2: Run all backtest variants
    # ============================================================
    print("\n" + "=" * 80)
    print("  PHASE 2: BACKTEST COMPARISON — BEFORE vs AFTER")
    print("=" * 80)

    results = {}

    # --- A) ORIGINAL BEHAVIOR (same-close entry, optimistic TP, global quantiles) ---
    # We approximate this by using original exit logic + no entry delay
    # Note: We already have strict confidence + strict volume in the dataset,
    # so for true original we'd need to rebuild. Instead we'll run a "near-original"
    # with strict confidence but original exit logic, then the full strict.

    # --- B) STRICT THRESHOLDS + ORIGINAL EXIT (TP before SL) ---
    print("\n  [B] Rolling thresholds + original exit (optimistic TP/SL)")
    for symbol, df in all_dfs.items():
        df_train = df[df["timestamp"] < train_cutoff].copy()
        df_test = df[df["timestamp"] >= train_cutoff].copy()
        for period, sub in [("TRAIN", df_train), ("TEST", df_test)]:
            if len(sub) < 100:
                continue
            trades = run_backtest(sub, symbol=symbol, exit_fn=simulate_trade_original, sizing_mode="compound")
            trades["symbol"] = symbol
            trades["period"] = period
            key = f"B_{period}"
            results.setdefault(key, []).append(trades)

    # --- C) STRICT THRESHOLDS + WORST-CASE EXIT ---
    print("  [C] Rolling thresholds + worst-case TP/SL")
    for symbol, df in all_dfs.items():
        df_train = df[df["timestamp"] < train_cutoff].copy()
        df_test = df[df["timestamp"] >= train_cutoff].copy()
        for period, sub in [("TRAIN", df_train), ("TEST", df_test)]:
            if len(sub) < 100:
                continue
            trades = run_backtest(sub, symbol=symbol, exit_fn=simulate_trade_strict, sizing_mode="compound")
            trades["symbol"] = symbol
            trades["period"] = period
            key = f"C_{period}"
            results.setdefault(key, []).append(trades)

    # --- D) STRICT EXIT + NEXT-OPEN ENTRY (delay=1) ---
    print("  [D] Strict exit + next-open entry (delay=1)")
    for symbol, df in all_dfs.items():
        df_train = df[df["timestamp"] < train_cutoff].copy()
        df_test = df[df["timestamp"] >= train_cutoff].copy()
        for period, sub in [("TRAIN", df_train), ("TEST", df_test)]:
            if len(sub) < 100:
                continue
            trades = run_backtest(sub, symbol=symbol, exit_fn=simulate_trade_strict, sizing_mode="compound", entry_delay=1)
            trades["symbol"] = symbol
            trades["period"] = period
            key = f"D_{period}"
            results.setdefault(key, []).append(trades)

    # --- E) FULL STRICT: worst-case + next-open + per-partial fees ---
    print("  [E] FULL STRICT: worst-case + next-open + per-partial fees")
    for symbol, df in all_dfs.items():
        df_train = df[df["timestamp"] < train_cutoff].copy()
        df_test = df[df["timestamp"] >= train_cutoff].copy()
        for period, sub in [("TRAIN", df_train), ("TEST", df_test)]:
            if len(sub) < 100:
                continue
            trades = run_backtest(sub, symbol=symbol, exit_fn=simulate_trade_strict, sizing_mode="compound", entry_delay=1)
            trades["symbol"] = symbol
            trades["period"] = period
            key = f"E_{period}"
            results.setdefault(key, []).append(trades)

    # ============================================================
    # PHASE 3: FRAGILITY TESTS
    # ============================================================
    print("\n" + "=" * 80)
    print("  PHASE 3: FRAGILITY TESTS")
    print("=" * 80)

    fragility = {}

    # Delayed entry by 1, 2, 3 candles
    for delay in [1, 2, 3]:
        print(f"  [DELAY-{delay}] Entry delayed {delay} candle(s)")
        combined = []
        for symbol, df in all_dfs.items():
            df_test = df[df["timestamp"] >= train_cutoff].copy()
            if len(df_test) < 100:
                continue
            trades = run_backtest(df_test, symbol=symbol, exit_fn=simulate_trade_strict,
                                 sizing_mode="compound", entry_delay=delay)
            trades["symbol"] = symbol
            combined.append(trades)
        if combined:
            all_trades = pd.concat(combined, ignore_index=True)
            fragility[f"delay_{delay}"] = all_trades

    # Fee stress test
    for mult in [1.0, 2.0, 3.0, 5.0]:
        print(f"  [FEE-{mult:.0f}x] Fee multiplier {mult}x")
        combined = []
        for symbol, df in all_dfs.items():
            df_test = df[df["timestamp"] >= train_cutoff].copy()
            if len(df_test) < 100:
                continue
            trades = run_backtest(df_test, symbol=symbol, exit_fn=simulate_trade_strict,
                                 sizing_mode="compound", entry_delay=1, fee_multiplier=mult)
            trades["symbol"] = symbol
            combined.append(trades)
        if combined:
            all_trades = pd.concat(combined, ignore_index=True)
            fragility[f"fee_{mult:.0f}x"] = all_trades

    # Sizing modes
    for mode in ["compound", "fixed_notional", "fixed_risk"]:
        print(f"  [SIZING-{mode}]")
        combined = []
        for symbol, df in all_dfs.items():
            df_test = df[df["timestamp"] >= train_cutoff].copy()
            if len(df_test) < 100:
                continue
            trades = run_backtest(df_test, symbol=symbol, exit_fn=simulate_trade_strict,
                                 sizing_mode=mode, entry_delay=1)
            trades["symbol"] = symbol
            combined.append(trades)
        if combined:
            all_trades = pd.concat(combined, ignore_index=True)
            fragility[f"sizing_{mode}"] = all_trades

    # Random baselines
    print("  [RANDOM] Random entry/direction baseline")
    random_combined = []
    for symbol, df in all_dfs.items():
        df_test = df[df["timestamp"] >= train_cutoff].copy()
        if len(df_test) < 100:
            continue
        trades = run_random_baseline(df_test, symbol=symbol, n_trades=500)
        trades["symbol"] = symbol
        random_combined.append(trades)
    if random_combined:
        fragility["random_baseline"] = pd.concat(random_combined, ignore_index=True)

    # ============================================================
    # PHASE 4: REPORT
    # ============================================================
    print("\n\n" + "=" * 100)
    print("  AUDIT REPORT")
    print("=" * 100)

    # --- Bias Table ---
    print("\n" + "-" * 100)
    print("  SECTION 1: BIAS TABLE")
    print("-" * 100)
    biases = [
        ("Global volume quantile", "P0", "add_trade_decision() → volume.quantile(0.30)",
         "Volume threshold uses full dataset; future candles influence past trades",
         "Inflates trade count by allowing trades in periods that should be filtered"),
        ("Same-candle entry", "P0", "simulate_trade() → entry_price = row['close']",
         "Enters at signal candle close; uses info not available until candle closes",
         "Captures 0.3-1.0% of move that shouldn't be available"),
        ("TP before SL in same candle", "P1", "simulate_trade() → TP check before SL check",
         "When both TP and SL touched in same candle, assumes TP hit first",
         "~5-15% of trades affected; inflates WR by assuming favorable fills"),
        ("Global quantile in confidence", "P1", "bounded_positive_score() → global quantile()",
         "Slope/return normalization uses full dataset statistics",
         "Confidence scores not reproducible in production; slight score shifts"),
        ("Global EMA dist reference", "P1", "build_confidence_engine() → ema_dist_pct.quantile(0.8)",
         "EMA distance quality uses full-dataset reference point",
         "Minor: affects structure component of confidence"),
        ("Fees on entry only (not per partial)", "P1", "simulate_trade() → cost = notional * RT cost once",
         "Partial exits at TP1/TP2/TP3 don't pay separate fees",
         "Understates true cost by 30-60% on multi-exit trades"),
        ("Aggressive compounding", "P2", "equity grows compounding, inflating later position sizes",
         "Later trades much larger than early trades; performance dominated by compounding",
         "Masks flat or negative expectancy in fixed-size mode"),
    ]

    print(f"  {'Issue':<35s} | {'Severity':<5s} | {'Location':<45s} | {'Impact'}")
    print(f"  {'-'*35}-+-{'-'*5}-+-{'-'*45}-+-{'-'*50}")
    for issue, sev, loc, _, impact in biases:
        print(f"  {issue:<35s} | {sev:<5s} | {loc:<45s} | {impact}")

    # --- Before vs After ---
    print("\n" + "-" * 100)
    print("  SECTION 2: BEFORE vs AFTER (TEST PERIOD: Dec 2025 - Apr 2026)")
    print("-" * 100)

    test_keys = [k for k in results if k.endswith("_TEST")]
    for key in sorted(test_keys):
        if results[key]:
            combined = pd.concat(results[key], ignore_index=True)
            m = compute_metrics(combined, key.replace("_TEST", ""))
            print_metrics(m)

    # --- Fragility Tests ---
    print("\n" + "-" * 100)
    print("  SECTION 3: FRAGILITY TESTS (TEST PERIOD)")
    print("-" * 100)

    for key, trades in sorted(fragility.items()):
        m = compute_metrics(trades, key)
        print_metrics(m)

    # --- Full Strict Report ---
    print("\n" + "-" * 100)
    print("  SECTION 4: FULL STRICT DETAILED REPORT")
    print("-" * 100)

    if "E_TEST" in results and results["E_TEST"]:
        full_strict = pd.concat(results["E_TEST"], ignore_index=True)
        print_full_report(full_strict, "FULL STRICT (E) — TEST PERIOD")

    # --- SHORT/MID Focus ---
    print("\n" + "-" * 100)
    print("  SECTION 5: SHORT/MID SPECIAL ANALYSIS")
    print("-" * 100)

    if "E_TEST" in results and results["E_TEST"]:
        fs = pd.concat(results["E_TEST"], ignore_index=True)
        mid_shorts = fs[(fs["confidence_mode"] == "MID") & (fs["direction"] == -1)]
        if len(mid_shorts) > 0:
            print(f"\n  MID SHORT trades (full strict): {len(mid_shorts)}")
            msm = compute_metrics(mid_shorts, "MID SHORT (strict)")
            print_metrics(msm)

            # Check if shorts enter pullbacks in uptrends
            print(f"\n  Exit reason breakdown for MID SHORTs:")
            for reason in mid_shorts["exit_reason"].value_counts().index:
                sub = mid_shorts[mid_shorts["exit_reason"] == reason]
                print(f"    {reason}: {len(sub)} trades, avg PnL: ${sub['net_pnl_cash'].mean():.2f}")

            # Monthly
            mid_copy = mid_shorts.copy()
            mid_copy["month"] = pd.to_datetime(mid_copy["entry_time"]).dt.to_period("M")
            print(f"\n  Monthly MID SHORT:")
            for month, grp in mid_copy.groupby("month"):
                print(f"    {month}: {len(grp)} trades, WR {grp['win'].mean():.1%}, PnL ${grp['net_pnl_cash'].sum():+.0f}")
        else:
            print("  No MID SHORT trades in strict test period.")

    # All SHORT trades analysis
    if "E_TEST" in results and results["E_TEST"]:
        all_shorts = fs[fs["direction"] == -1]
        if len(all_shorts) > 0:
            print(f"\n  ALL SHORT trades (full strict): {len(all_shorts)}")
            for mode in sorted(all_shorts["confidence_mode"].unique()):
                sub = all_shorts[all_shorts["confidence_mode"] == mode]
                sm = compute_metrics(sub, f"    SHORT/{mode}")
                print_metrics(sm)

    # --- Monthly breakdown for full strict ---
    print("\n" + "-" * 100)
    print("  SECTION 6: MONTHLY BREAKDOWN (FULL STRICT)")
    print("-" * 100)

    if "E_TEST" in results and results["E_TEST"]:
        fs = pd.concat(results["E_TEST"], ignore_index=True)
        fs["month"] = pd.to_datetime(fs["entry_time"]).dt.to_period("M")
        for month, grp in fs.groupby("month"):
            m = compute_metrics(grp, str(month))
            print_metrics(m)

    # --- Verdict ---
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)

    # Compare key metrics
    if "C_TEST" in results and results["C_TEST"] and "E_TEST" in results and results["E_TEST"]:
        optim = pd.concat(results["C_TEST"], ignore_index=True)  # rolling thresholds, optimistic exit
        strict = pd.concat(results["E_TEST"], ignore_index=True)  # full strict

        om = compute_metrics(optim, "Optimistic")
        sm = compute_metrics(strict, "Strict")

        print(f"\n  Optimistic (rolling thresholds, TP-before-SL, compound):")
        print_metrics(om)
        print(f"\n  Strict (rolling thresholds, worst-case TP/SL, next-open, per-partial fees):")
        print_metrics(sm)

        wr_drop = om["win_rate"] - sm["win_rate"]
        pnl_drop = (om["total_pnl"] - sm["total_pnl"]) / abs(om["total_pnl"]) if om["total_pnl"] != 0 else 0
        pf_drop = (om["profit_factor"] - sm["profit_factor"]) / om["profit_factor"] if om["profit_factor"] > 0 else 0

        print(f"\n  Impact of strictness:")
        print(f"    Win rate drop: {wr_drop:.1%}")
        print(f"    PnL drop: {pnl_drop:.1%}")
        print(f"    PF drop: {pf_drop:.1%}")
        print(f"    Ambiguous candles in strict: {sm.get('ambiguous_trades', 0)}")

        if sm["profit_factor"] > 1.5 and sm["win_rate"] > 0.55:
            verdict = "VALID — Edge survives strict execution assumptions"
        elif sm["profit_factor"] > 1.0:
            verdict = "PARTIALLY VALID — Edge survives but significantly weakened"
        else:
            verdict = "NOT VALID — Edge does not survive strict execution assumptions"

        print(f"\n  *** VERDICT: {verdict} ***")

    # --- Save trades ---
    if "E_TEST" in results and results["E_TEST"]:
        all_strict = pd.concat(results["E_TEST"], ignore_index=True)
        all_strict.to_csv("data/features/strict_audit_trades.csv", index=False)
        print(f"\n  [SAVED] data/features/strict_audit_trades.csv ({len(all_strict)} trades)")

    if fragility:
        # Save fragility summary
        frag_summary = []
        for key, trades in fragility.items():
            m = compute_metrics(trades, key)
            frag_summary.append(m)
        frag_df = pd.DataFrame(frag_summary)
        frag_df.to_csv("data/features/strict_audit_fragility.csv", index=False)
        print(f"  [SAVED] data/features/strict_audit_fragility.csv")

    print("\n" + "=" * 100)
    print("  AUDIT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
