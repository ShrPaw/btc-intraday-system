#!/usr/bin/env python3
"""
BTC Intraday Live Signal Bot
Connects to Binance WebSocket → resamples to multi-timeframe → runs backtester logic → sends Telegram signals.
"""

import json
import time
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import websocket
import pandas as pd
import numpy as np
import requests

# =========================================================
# CONFIG — EDIT THESE
# =========================================================

TELEGRAM_BOT_TOKEN = "8648848839:AAFSTnGaA6qTN9Jrm1HJ3mkFqt0vNszwBgA"
TELEGRAM_CHAT_ID = "8457186616"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# How many 1m candles to keep in memory (rolling window)
# 12H = 720 bars, but we need more for indicator warmup → use ~2 days
ROLLING_WINDOW = 10080  # 7 days of 1m candles (enough for 4H RSI-14 warmup)

# How often to check for new setups (in seconds)
CHECK_INTERVAL = 60  # check every 1m bar close

# =========================================================
# STRATEGY PARAMS (must match btc_intraday_system.py exactly)
# =========================================================

RSI_PERIOD = 14
EMA_PERIOD = 20

# Confidence thresholds
NO_TRADE_THRESHOLD = 0.72
MILD_THRESHOLD = 0.78
MID_THRESHOLD = 0.80
PREMIUM_THRESHOLD = 0.82
HIGH_THRESHOLD = 0.88

# ADX
ADX_PERIOD = 14
ADX_TRENDING_BONUS = 25.0
ADX_CHOPPY_PENALTY = 15.0
ENABLE_ADX_FILTER = True

# TP structure
TP1_PCT = 0.005
TP2_PCT_SCALP = 0.007
TP2_PCT_TREND = 0.006
TP3_PCT = 0.008
TP4_PCT = 0.090

# Cooldown (5m bars)
COOLDOWN_BARS_5M = 3

# Allowed modes
ALLOWED_MODES = {"MILD", "PREMIUM", "HIGH", "ELITE"}

# Trailing stop
TRAIL_PCT_MID = 0.0050
TRAIL_PCT_HIGH = 0.0045
TRAIL_PCT_ELITE = 0.0040

# =========================================================
# STATE
# =========================================================

# Per-symbol rolling 1m candle buffers
candle_buffers = defaultdict(list)  # symbol → list of {timestamp, price, volume, delta, trade_count}

# Track last signal time per symbol to enforce cooldown
last_signal_time = {}  # symbol → datetime of last signal

# Track active signals (to not re-send)
sent_signals = set()  # set of (symbol, timestamp, direction) tuples

# Shutdown flag
running = True

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_bot")


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(text: str) -> bool:
    """Send a message to the configured Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if data.get("ok"):
            log.info(f"Telegram sent: {text[:60]}...")
            return True
        else:
            log.error(f"Telegram error: {data}")
            return False
    except Exception as e:
        log.error(f"Telegram exception: {e}")
        return False


# =========================================================
# INDICATORS (copied from btc_intraday_system.py)
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
    bars_since = []
    last_dir = []

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
        (df["close"].shift(1) <= df["ema20"].shift(1)) & (df["close"] > df["ema20"])
    )
    df["ema_reclaim_short"] = (
        (df["close"].shift(1) >= df["ema20"].shift(1)) & (df["close"] < df["ema20"])
    )

    lookback = 12
    df["prev_swing_high"] = df["high"].rolling(lookback).max().shift(1)
    df["prev_swing_low"] = df["low"].rolling(lookback).min().shift(1)
    df["bos_long"] = df["high"] > df["prev_swing_high"]
    df["bos_short"] = df["low"] < df["prev_swing_low"]

    df["structure_trigger_long"] = (
        df["ema_reclaim_long"].rolling(3).max().fillna(0).astype(bool) &
        df["bos_long"].rolling(3).max().fillna(0).astype(bool)
    )
    df["structure_trigger_short"] = (
        df["ema_reclaim_short"].rolling(3).max().fillna(0).astype(bool) &
        df["bos_short"].rolling(3).max().fillna(0).astype(bool)
    )
    return df


def add_extra_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["rv_6"] = df["ret_1"].rolling(6).std()
    df["delta_3"] = df["delta"].rolling(3).sum()
    df["delta_6"] = df["delta"].rolling(6).sum()
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
    df["dx"] = dx.values
    df["adx"] = adx.values
    return df


# =========================================================
# HELPERS
# =========================================================

def rolling_minmax_score(series: pd.Series, window: int = 200) -> pd.Series:
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    score = (series - roll_min) / denom
    return score.clip(0, 1)


def bounded_positive_score(series: pd.Series, ref_quantile: float = 0.8) -> pd.Series:
    ref = series.abs().quantile(ref_quantile)
    if pd.isna(ref) or ref == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series.abs() / ref).clip(0, 1)


def merge_asof_feature(base: pd.DataFrame, other: pd.DataFrame, prefix: str, cols: list) -> pd.DataFrame:
    rhs = other[["timestamp"] + cols].copy()
    rhs = rhs.rename(columns={c: f"{prefix}_{c}" for c in cols})
    return pd.merge_asof(
        base.sort_values("timestamp"),
        rhs.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


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


def get_trail_pct(confidence_mode: str) -> float:
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
# CORE: Build analysis for last bar
# =========================================================

def build_analysis(symbol: str) -> pd.DataFrame | None:
    """Build the full master dataset from the rolling buffer. Returns None if not enough data."""
    buffer = candle_buffers[symbol]
    if len(buffer) < 100:
        return None

    # Convert buffer to DataFrame
    df = pd.DataFrame(buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure numeric types
    for col in ["price", "volume", "delta", "trade_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Resample
    bars_5m = resample_bars(df, "5min")
    bars_15m = resample_bars(df, "15min")
    bars_30m = resample_bars(df, "30min")
    bars_4h = resample_bars(df, "4h")
    bars_6h = resample_bars(df, "6h")
    bars_12h = resample_bars(df, "12h")

    # Need minimum bars for indicators
    if len(bars_5m) < 60 or len(bars_4h) < 10:
        return None

    # Add indicators
    bars_5m = add_rsi_features(bars_5m, RSI_PERIOD)
    bars_5m = add_ema_bos_features_5m(bars_5m)
    bars_5m = add_extra_features_5m(bars_5m)

    bars_15m = add_rsi_features(bars_15m, RSI_PERIOD)
    bars_30m = add_rsi_features(bars_30m, RSI_PERIOD)
    bars_4h = add_rsi_features(bars_4h, RSI_PERIOD)
    bars_4h = compute_adx(bars_4h, ADX_PERIOD)
    bars_6h = add_rsi_features(bars_6h, RSI_PERIOD)
    bars_12h = add_rsi_features(bars_12h, RSI_PERIOD)

    # Merge
    merged = bars_5m.copy()
    merged = merge_asof_feature(merged, bars_15m, "m15", ["rsi", "rsi_slope_1"])
    merged = merge_asof_feature(merged, bars_30m, "m30", ["rsi"])
    merged = merge_asof_feature(
        merged, bars_4h, "h4",
        ["rsi", "rsi_slope_1", "rsi_slope_2", "fresh_long", "fresh_short", "bars_since_cross", "adx", "plus_di", "minus_di"]
    )
    merged = merge_asof_feature(merged, bars_6h, "h6", ["rsi"])
    merged = merge_asof_feature(merged, bars_12h, "h12", ["rsi"])

    for c in ["h4_fresh_long", "h4_fresh_short"]:
        merged[c] = merged[c].fillna(False).astype(bool)

    merged = merged.dropna().reset_index(drop=True)

    # Setup engine
    merged = _build_setup_engine(merged)
    # Structure gate
    merged = _add_structure_gate(merged)
    # Confidence
    merged = _build_confidence(merged)
    # Trade decision
    merged = _add_trade_decision(merged)

    return merged


def _build_setup_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scalp_long = (
        df["h4_fresh_long"] &
        (df["m15_rsi"] > 50) &
        (df["m15_rsi"].shift(1) > 50) &
        (df["h6_rsi"] > 50) &
        (df["h12_rsi"] > 50)
    )
    scalp_short = (
        df["h4_fresh_short"] &
        (df["m15_rsi"] < 50) &
        (df["m15_rsi"].shift(1) < 50) &
        (df["h6_rsi"] < 50) &
        (df["h12_rsi"] < 50)
    )
    trend_long = (
        (df["h4_rsi"] > 55) &
        (df["h4_rsi_slope_1"] > 0) &
        (df["h4_rsi_slope_2"] > 0) &
        (df["m15_rsi"] > 55) &
        (df["m15_rsi_slope_1"] >= 0) &
        (df["h12_rsi"] > 50)
    )
    trend_short = (
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

    df.loc[scalp_long, ["setup_type", "direction"]] = ["RSI_SCALP", 1]
    df.loc[scalp_short, ["setup_type", "direction"]] = ["RSI_SCALP", -1]
    df.loc[trend_long, ["setup_type", "direction"]] = ["RSI_TREND", 1]
    df.loc[trend_short, ["setup_type", "direction"]] = ["RSI_TREND", -1]

    bsc = df["h4_bars_since_cross"].fillna(99).astype(int)
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc <= 1), "stage"] = 1
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 2) & (bsc <= 3), "stage"] = 2
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 4), "stage"] = 3

    return df


def _add_structure_gate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trigger_ok"] = False
    out.loc[(out["setup_type"] == "RSI_SCALP") & (out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["setup_type"] == "RSI_SCALP") & (out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    out.loc[(out["setup_type"] == "RSI_TREND") & (out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["setup_type"] == "RSI_TREND") & (out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    return out


def _build_confidence(df: pd.DataFrame) -> pd.DataFrame:
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

    h4_slope_score = bounded_positive_score(out["h4_rsi_slope_1"], 0.8)
    m15_slope_score = bounded_positive_score(out["m15_rsi_slope_1"], 0.8)
    ret_score = bounded_positive_score(out["ret_3"], 0.8)
    momentum_score = (0.4 * h4_slope_score + 0.4 * m15_slope_score + 0.2 * ret_score).clip(0, 1)

    ema_dist_ref = max(out["ema_dist_pct"].abs().quantile(0.8), 1e-9)
    ema_dist_quality = (1 - (out["ema_dist_pct"].abs() / ema_dist_ref)).clip(0, 1)

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
        0.30 * scalp_alignment +
        0.25 * fresh_base +
        0.20 * momentum_score +
        0.15 * rv_score +
        0.10 * scalp_structure
    )

    trend_mask = out["setup_type"] == "RSI_TREND"
    trend_alignment = np.where(out["direction"] == 1, alignment_long, alignment_short)
    trend_structure = np.where(out["direction"] == 1, structure_long, structure_short)
    trend_conf = (
        0.25 * trend_alignment +
        0.20 * momentum_score +
        0.30 * trend_structure +
        0.15 * rv_score +
        0.10 * cfx_align
    )

    out.loc[scalp_mask, "confidence_raw"] = scalp_conf[scalp_mask]
    out.loc[trend_mask, "confidence_raw"] = trend_conf[trend_mask]
    out["confidence_raw"] = out["confidence_raw"].clip(0, 1)

    # ADX modifier
    if ENABLE_ADX_FILTER and "h4_adx" in out.columns:
        adx_val = out["h4_adx"].fillna(20)
        adx_modifier = pd.Series(0.0, index=out.index)
        adx_modifier[adx_val >= ADX_TRENDING_BONUS] = 0.03
        adx_modifier[(adx_val >= 15) & (adx_val < 25)] = 0.0
        adx_modifier[adx_val < ADX_CHOPPY_PENALTY] = -0.03
        out["confidence_raw"] = (out["confidence_raw"] + adx_modifier).clip(0, 1)

    # Elite gate
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


def _add_trade_decision(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mode_direction_ok = (
        out["confidence_mode"].isin(ALLOWED_MODES) &
        out["direction"].isin({1, -1})
    )

    vol_threshold = out["volume"].quantile(0.30)
    volume_ok = out["volume"] >= vol_threshold

    out["take_trade"] = (
        (out["setup_type"] != "none") &
        out["trigger_ok"] &
        (out["confidence_mode"] != "NO_TRADE") &
        mode_direction_ok &
        volume_ok
    )
    return out


# =========================================================
# SIGNAL FORMATTING
# =========================================================

def format_signal(symbol: str, row: pd.Series) -> str:
    direction = "🟢 LONG" if row["direction"] == 1 else "🔴 SHORT"
    entry = float(row["close"])
    conf = float(row["confidence_raw"])
    mode = str(row["confidence_mode"])
    strategy = str(row["setup_type"])
    h4_rsi = float(row["h4_rsi"])
    m15_rsi = float(row["m15_rsi"])

    # Compute stop for display
    if row["direction"] == 1:
        structural_stop = float(row["low"])
        stop_pct = max((entry - structural_stop) / entry, 0.004)
        stop = entry * (1 - stop_pct)
    else:
        structural_stop = float(row["high"])
        stop_pct = max((structural_stop - entry) / entry, 0.004)
        stop = entry * (1 + stop_pct)

    tp2_pct = TP2_PCT_TREND if strategy == "RSI_TREND" else TP2_PCT_SCALP
    trail_pct = get_trail_pct(mode)

    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")

    lines = [
        f"{'━' * 30}",
        f"📊 <b>{symbol}</b> — {direction}",
        f"⏰ {now} GMT+8",
        f"",
        f"🎯 <b>Strategy:</b> {strategy}",
        f"💪 <b>Confidence:</b> {conf:.1%} ({mode})",
        f"",
        f"📍 <b>Entry:</b> ${entry:,.2f}",
        f"🛑 <b>Stop:</b> ${stop:,.2f} ({stop_pct:.2%})",
        f"",
        f"🎯 <b>Take Profits:</b>",
        f"  TP1: ${entry * (1 + TP1_PCT) if row['direction'] == 1 else entry * (1 - TP1_PCT):,.2f} (+{TP1_PCT:.1%}) → 35%",
        f"  TP2: ${entry * (1 + tp2_pct) if row['direction'] == 1 else entry * (1 - tp2_pct):,.2f} (+{tp2_pct:.1%}) → 35%",
        f"  TP3: ${entry * (1 + TP3_PCT) if row['direction'] == 1 else entry * (1 - TP3_PCT):,.2f} (+{TP3_PCT:.1%}) → 15%",
        f"  TP4: ${entry * (1 + TP4_PCT) if row['direction'] == 1 else entry * (1 - TP4_PCT):,.2f} (+{TP4_PCT:.1%}) → 15%",
        f"",
        f"🔄 <b>Trail:</b> {trail_pct:.2%} (after TP2)",
        f"⚖️ <b>BE:</b> +0.02% (after TP1)",
        f"",
        f"📈 <b>Context:</b>",
        f"  4H RSI: {h4_rsi:.1f} | 15M RSI: {m15_rsi:.1f}",
        f"  ADX: {float(row.get('h4_adx', 0)):.1f}",
    ]

    return "\n".join(lines)


# =========================================================
# SIGNAL CHECK
# =========================================================

def check_for_signals():
    """Run analysis on each symbol and send Telegram signals for new setups."""
    for symbol in SYMBOLS:
        try:
            analysis = build_analysis(symbol)
            if analysis is None:
                continue

            # Only look at the latest bar
            latest = analysis.iloc[-1]

            if not bool(latest.get("take_trade", False)):
                continue

            # Create unique key to avoid re-sending
            ts = str(latest["timestamp"])
            direction = int(latest["direction"])
            sig_key = (symbol, ts, direction)

            if sig_key in sent_signals:
                continue

            # Cooldown check
            if symbol in last_signal_time:
                elapsed = (datetime.now(timezone.utc) - last_signal_time[symbol]).total_seconds()
                if elapsed < COOLDOWN_BARS_5M * 5 * 60:  # 3 × 5m bars = 15 min
                    continue

            # SEND SIGNAL
            msg = format_signal(symbol, latest)
            if send_telegram(msg):
                sent_signals.add(sig_key)
                last_signal_time[symbol] = datetime.now(timezone.utc)
                log.info(f"🚀 SIGNAL SENT: {symbol} {latest['setup_type']} {'LONG' if direction == 1 else 'SHORT'} conf={latest['confidence_raw']:.1%}")

            # Cleanup old signals (keep last 100)
            if len(sent_signals) > 100:
                old_keys = sorted(sent_signals)[:50]
                for k in old_keys:
                    sent_signals.discard(k)

        except Exception as e:
            log.error(f"Error checking {symbol}: {e}", exc_info=True)


# =========================================================
# BINANCE WEBSOCKET
# =========================================================

def on_message(ws, message):
    """Handle incoming WebSocket messages from Binance."""
    global candle_buffers

    try:
        data = json.loads(message)
        kline = data.get("k", {})
        symbol = kline.get("s", "")

        if not kline.get("x", False):
            return  # Only process closed klines

        candle = {
            "timestamp": pd.Timestamp(kline["t"], unit="ms", tz="UTC"),
            "price": float(kline["c"]),
            "volume": float(kline["v"]),
            "delta": float(kline["V"]) - (float(kline["v"]) - float(kline["V"])),
            "trade_count": float(kline["n"]),
        }

        candle_buffers[symbol].append(candle)

        # Trim to rolling window
        if len(candle_buffers[symbol]) > ROLLING_WINDOW:
            candle_buffers[symbol] = candle_buffers[symbol][-ROLLING_WINDOW:]

        log.debug(f"New 1m candle: {symbol} @ {candle['timestamp']} price=${candle['price']:,.2f}")

    except Exception as e:
        log.error(f"WebSocket parse error: {e}")


def on_error(ws, error):
    log.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    log.warning(f"WebSocket closed: {close_status_code} {close_msg}")
    if running:
        log.info("Reconnecting in 5 seconds...")
        time.sleep(5)
        start_websocket()


def on_open(ws):
    log.info("WebSocket connected!")
    # Subscribe to kline streams for all symbols
    streams = [f"{s.lower()}@kline_1m" for s in SYMBOLS]
    subscribe_msg = {
        "method": "SUBSCRIBE",
        "params": streams,
        "id": 1,
    }
    ws.send(json.dumps(subscribe_msg))
    log.info(f"Subscribed to: {', '.join(streams)}")


def start_websocket():
    """Connect to Binance combined WebSocket stream."""
    streams = "/".join([f"{s.lower()}@kline_1m" for s in SYMBOLS])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = ws.run_forever(ping_interval=30, ping_timeout=10)
    return ws


# =========================================================
# MAIN LOOP
# =========================================================

def signal_handler(sig, frame):
    global running
    log.info("Shutdown signal received...")
    running = False


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("=" * 60)
    log.info("BTC Intraday Live Signal Bot")
    log.info(f"Symbols: {', '.join(SYMBOLS)}")
    log.info(f"Check interval: {CHECK_INTERVAL}s")
    log.info("=" * 60)

    # Send startup notification
    send_telegram("🤖 <b>BTC Signal Bot Online</b>\nWatching: " + ", ".join(SYMBOLS) + "\nSignals will appear here.")

    # Start WebSocket in background thread
    import threading
    ws_thread = threading.Thread(target=start_websocket, daemon=True)
    ws_thread.start()

    # Main loop: check for signals every minute
    log.info("Waiting for WebSocket data to accumulate...")
    time.sleep(300)  # Wait 5 min for initial data

    while running:
        try:
            check_for_signals()
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(10)

    log.info("Bot stopped.")


if __name__ == "__main__":
    main()
