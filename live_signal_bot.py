#!/usr/bin/env python3
"""
WRAITH — Intraday Crypto Signal Bot
- Binance WebSocket → multi-timeframe analysis → Telegram signals
- Inline buttons for instant trade execution
- Position monitoring with TP/SL alerts
- ccxt integration for live trading (or paper mode)
"""

import json
import os
import time
import logging
import signal
import sys
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

import websocket
import pandas as pd
import numpy as np
import requests

# Load .env
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

load_env()

# =========================================================
# CONFIG (from .env)
# =========================================================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
PAPER_MODE = os.environ.get("PAPER_MODE", "true").lower() == "true"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
# Generate signals on ALL symbols (was XRP-only, which killed 75% of signals)
TRADING_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

ROLLING_WINDOW = 10080  # 7 days of 1m candles
CHECK_INTERVAL = 60

RSI_PERIOD = 14
EMA_PERIOD = 20

NO_TRADE_THRESHOLD = 0.72
MILD_THRESHOLD = 0.78
MID_THRESHOLD = 0.80
PREMIUM_THRESHOLD = 0.82
HIGH_THRESHOLD = 0.88

ADX_PERIOD = 14
ADX_TRENDING_BONUS = 25.0
ADX_CHOPPY_PENALTY = 15.0
ENABLE_ADX_FILTER = True

TP1_PCT = 0.005
TP2_PCT_SCALP = 0.007
TP2_PCT_TREND = 0.006
TP3_PCT = 0.008
TP4_PCT = 0.090

TP1_SIZE = 0.35
TP2_SIZE = 0.35
TP3_SIZE = 0.15
TP4_SIZE = 0.15

COOLDOWN_BARS_5M = 3
ALLOWED_MODES = {"MILD", "PREMIUM", "HIGH", "ELITE"}

TRAIL_PCT_MID = 0.0050
TRAIL_PCT_HIGH = 0.0045
TRAIL_PCT_ELITE = 0.0040

# Slippage by symbol
SLIPPAGE = {"BTCUSDT": 0.0005, "ETHUSDT": 0.0008, "SOLUSDT": 0.0012, "XRPUSDT": 0.0015}

# Default leverage per trade
DEFAULT_LEVERAGE = 5

# =========================================================
# STATE
# =========================================================

candle_buffers = defaultdict(list)
last_signal_time = {}
sent_signals = set()

# Active positions: {symbol: {direction, entry_price, stop_price, tp1..tp4, size_usd, remaining, mode, strategy, ...}}
active_positions = {}

# Pending callbacks to avoid double-execution
pending_callbacks = set()

running = True
lock = threading.Lock()

# ccxt exchange (lazy init)
_exchange = None

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
# CCXT EXCHANGE
# =========================================================

def get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "options": {"defaultType": "future"},  # USDT futures
            "enableRateLimit": True,
        })
        _exchange.load_markets()
    return _exchange


def get_account_balance() -> float:
    """Fetch available USDT balance from Binance Futures. Falls back to PAPER_EQUITY."""
    if PAPER_MODE:
        return float(os.environ.get("PAPER_EQUITY", "10000"))
    try:
        ex = get_exchange()
        balance = ex.fetch_balance()
        usdt_free = balance.get("USDT", {}).get("free", 0)
        return float(usdt_free)
    except Exception as e:
        log.error(f"Failed to fetch balance: {e}")
        return 0.0


def execute_order(symbol: str, direction: str, size_usd: float, leverage: int = DEFAULT_LEVERAGE) -> dict | None:
    """Place a market order on Binance Futures. Returns order dict or None."""
    if PAPER_MODE:
        log.info(f"[PAPER] Would execute: {direction} {symbol} size=${size_usd:.2f} lev={leverage}x")
        return {"id": f"paper_{int(time.time())}", "status": "filled", "paper": True}

    try:
        ex = get_exchange()
        # Set leverage
        ex.set_leverage(leverage, symbol)

        side = "buy" if direction == "LONG" else "sell"
        ticker = ex.fetch_ticker(symbol)
        price = ticker["last"]
        amount = size_usd / price

        # Check minimum order size
        market = ex.market(symbol)
        min_notional = market.get("limits", {}).get("cost", {}).get("min", 5.0)
        if size_usd < min_notional:
            log.warning(f"⚠️ {symbol} order ${size_usd:.2f} below min notional ${min_notional:.2f} — skipping")
            return None

        # Round amount to exchange precision
        min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
        if amount < min_qty:
            log.warning(f"⚠️ {symbol} qty {amount:.8f} below min {min_qty} — skipping")
            return None

        amount = ex.amount_to_precision(symbol, amount)

        order = ex.create_order(symbol, "market", side, amount)
        log.info(f"✅ ORDER EXECUTED: {side} {symbol} {amount} @ market")
        return order
    except Exception as e:
        log.error(f"Order execution failed: {e}", exc_info=True)
        return None


def close_position(symbol: str, fraction: float = 1.0) -> dict | None:
    """Close a position (full or partial)."""
    if symbol not in active_positions:
        return None

    pos = active_positions[symbol]
    if PAPER_MODE:
        log.info(f"[PAPER] Would close {fraction:.0%} of {symbol} {pos['direction']}")
        return {"id": f"paper_close_{int(time.time())}", "status": "filled", "paper": True}

    try:
        ex = get_exchange()
        # Fetch current position
        positions = ex.fetch_positions([symbol])
        pos_data = next((p for p in positions if float(p.get("contracts", 0)) > 0), None)
        if not pos_data:
            log.warning(f"No open position found for {symbol}")
            return None

        close_amount = float(pos_data["contracts"]) * fraction
        close_side = "sell" if pos["direction"] == "LONG" else "buy"
        close_amount = ex.amount_to_precision(symbol, close_amount)

        order = ex.create_order(symbol, "market", close_side, close_amount, params={"reduceOnly": True})
        log.info(f"✅ POSITION CLOSED: {close_side} {symbol} {close_amount} ({fraction:.0%})")
        return order
    except Exception as e:
        log.error(f"Close position failed: {e}", exc_info=True)
        return None


# =========================================================
# TELEGRAM HELPERS
# =========================================================

def send_telegram(text: str, reply_markup: dict = None) -> dict | None:
    """Send message to Telegram. Returns message dict."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if data.get("ok"):
            return data.get("result")
        log.error(f"Telegram error: {data}")
    except Exception as e:
        log.error(f"Telegram exception: {e}")
    return None


def edit_telegram(message_id: int, text: str, reply_markup: dict = None) -> bool:
    """Edit an existing Telegram message."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.json().get("ok", False)
    except Exception as e:
        log.error(f"Edit message error: {e}")
    return False


def answer_callback(callback_id: str, text: str = ""):
    """Answer a callback query (removes loading spinner)."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    try:
        requests.post(url, json={"callback_query_id": callback_id, "text": text}, timeout=5)
    except Exception:
        pass


# =========================================================
# SIGNAL FORMATTING
# =========================================================

def get_trail_pct(mode: str) -> float:
    if mode in ("MILD", "MID"):
        return TRAIL_PCT_MID
    if mode == "PREMIUM":
        return TRAIL_PCT_HIGH
    if mode in ("HIGH", "ELITE"):
        return TRAIL_PCT_ELITE if mode == "ELITE" else TRAIL_PCT_HIGH
    return TRAIL_PCT_MID


def format_signal(symbol: str, row: pd.Series) -> tuple[str, dict]:
    """Returns (text, inline_keyboard) for Telegram."""
    direction = "LONG" if row["direction"] == 1 else "SHORT"
    emoji = "🟢" if direction == "LONG" else "🔴"
    entry = float(row["close"])
    conf = float(row["confidence_raw"])
    mode = str(row["confidence_mode"])
    strategy = str(row["setup_type"])
    h4_rsi = float(row["h4_rsi"])
    m15_rsi = float(row["m15_rsi"])

    # Structural stop: 4-bar lookback (matches backtester v5.5)
    if row["direction"] == 1:
        structural_stop = float(row.get("structural_low", row["low"]))
        if strategy == "RSI_TREND":
            structural_stop = min(structural_stop, float(row["ema20"]))
        stop_pct = max((entry - structural_stop) / entry, 0.004)
        stop = entry * (1 - stop_pct)
    else:
        structural_stop = float(row.get("structural_high", row["high"]))
        if strategy == "RSI_TREND":
            structural_stop = max(structural_stop, float(row["ema20"]))
        stop_pct = max((structural_stop - entry) / entry, 0.004)
        stop = entry * (1 + stop_pct)

    tp2_pct = TP2_PCT_TREND if strategy == "RSI_TREND" else TP2_PCT_SCALP
    trail_pct = get_trail_pct(mode)
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M GMT+8")

    sign = 1 if direction == "LONG" else -1

    text = (
        f"{'━' * 24}\n"
        f"📊 <b>{symbol}</b> — {emoji} {direction}\n"
        f"⏰ {now}\n\n"
        f"🎯 Strategy: {strategy}\n"
        f"💪 Confidence: {conf:.1%} ({mode})\n\n"
        f"📍 Entry: <b>${entry:,.2f}</b>\n"
        f"🛑 Stop: ${stop:,.2f} ({stop_pct:.2%})\n\n"
        f"🎯 Take Profits:\n"
        f"  TP1: ${entry * (1 + sign * TP1_PCT):,.2f} (+{TP1_PCT:.1%}) → 35%\n"
        f"  TP2: ${entry * (1 + sign * tp2_pct):,.2f} (+{tp2_pct:.1%}) → 35%\n"
        f"  TP3: ${entry * (1 + sign * TP3_PCT):,.2f} (+{TP3_PCT:.1%}) → 15%\n"
        f"  TP4: ${entry * (1 + sign * TP4_PCT):,.2f} (+{TP4_PCT:.1%}) → 15%\n\n"
        f"🔄 Trail: {trail_pct:.2%} (after TP2)\n"
        f"📈 4H RSI: {h4_rsi:.1f} | 15M RSI: {m15_rsi:.1f}\n"
        f"{'━' * 24}"
    )

    callback_data = f"go_{direction.lower()}_{symbol}"

    keyboard = {
        "inline_keyboard": [
            [
                {"text": f"⚡ {direction} {symbol.split('USDT')[0]} ({DEFAULT_LEVERAGE}x)", "callback_data": callback_data},
            ],
            [
                {"text": "📊 3x", "callback_data": f"go_{direction.lower()}_{symbol}_3x"},
                {"text": "📊 5x", "callback_data": f"go_{direction.lower()}_{symbol}_5x"},
                {"text": "📊 10x", "callback_data": f"go_{direction.lower()}_{symbol}_10x"},
            ],
            [
                {"text": "❌ Pass", "callback_data": f"pass_{symbol}"},
            ],
        ]
    }

    return text, keyboard


def format_position_status(symbol: str) -> tuple[str, dict]:
    """Format active position with TP/SL close buttons."""
    pos = active_positions[symbol]
    direction = pos["direction"]
    emoji = "🟢" if direction == "LONG" else "🔴"

    # Get current price from buffer
    buffer = candle_buffers.get(symbol, [])
    current_price = buffer[-1]["price"] if buffer else pos["entry_price"]

    if direction == "LONG":
        pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
    else:
        pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

    pnl_emoji = "💰" if pnl_pct > 0 else "📉"

    text = (
        f"{'━' * 24}\n"
        f"📊 <b>{symbol}</b> — {emoji} {direction} (OPEN)\n\n"
        f"📍 Entry: ${pos['entry_price']:,.2f}\n"
        f"💲 Current: ${current_price:,.2f}\n"
        f"{pnl_emoji} PnL: {pnl_pct:+.2%}\n\n"
        f"🛑 Stop: ${pos['stop_price']:,.2f}\n"
        f"🎯 TP1: ${pos['tp1']:,.2f} {'✅' if pos.get('tp1_hit') else ''}\n"
        f"🎯 TP2: ${pos['tp2']:,.2f} {'✅' if pos.get('tp2_hit') else ''}\n"
        f"🎯 TP3: ${pos['tp3']:,.2f} {'✅' if pos.get('tp3_hit') else ''}\n"
        f"🎯 TP4: ${pos['tp4']:,.2f} {'✅' if pos.get('tp4_hit') else ''}\n\n"
        f"Remaining: {pos['remaining']:.0%}\n"
        f"{'━' * 24}"
    )

    keyboard = {
        "inline_keyboard": [
            [
                {"text": "❌ Close 50%", "callback_data": f"close_{symbol}_half"},
                {"text": "❌ Close All", "callback_data": f"close_{symbol}_all"},
            ],
        ]
    }

    return text, keyboard


# =========================================================
# INDICATORS (same as v1)
# =========================================================

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
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
    df["ema_reclaim_long"] = (df["close"].shift(1) <= df["ema20"].shift(1)) & (df["close"] > df["ema20"])
    df["ema_reclaim_short"] = (df["close"].shift(1) >= df["ema20"].shift(1)) & (df["close"] < df["ema20"])
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
    # Structural stop: 4-bar lookback (matches backtester v5.5)
    stop_lookback = 4
    df["structural_low"] = df["low"].rolling(stop_lookback).min()
    df["structural_high"] = df["high"].rolling(stop_lookback).max()
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
# HELPERS
# =========================================================

def rolling_minmax_score(s: pd.Series, w: int = 200) -> pd.Series:
    rmin = s.rolling(w).min()
    rmax = s.rolling(w).max()
    return ((s - rmin) / (rmax - rmin).replace(0, np.nan)).clip(0, 1)

def bounded_positive_score(s: pd.Series, q: float = 0.8) -> pd.Series:
    ref = s.abs().quantile(q)
    if pd.isna(ref) or ref == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s.abs() / ref).clip(0, 1)

def merge_asof_feature(base, other, prefix, cols):
    rhs = other[["timestamp"] + cols].copy()
    rhs = rhs.rename(columns={c: f"{prefix}_{c}" for c in cols})
    return pd.merge_asof(base.sort_values("timestamp"), rhs.sort_values("timestamp"), on="timestamp", direction="backward")

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
# ANALYSIS PIPELINE
# =========================================================

def build_analysis(symbol: str) -> pd.DataFrame | None:
    buffer = candle_buffers[symbol]
    if len(buffer) < 100:
        return None
    df = pd.DataFrame(buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["price", "volume", "delta", "trade_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    bars_5m = resample_bars(df, "5min")
    bars_15m = resample_bars(df, "15min")
    bars_30m = resample_bars(df, "30min")
    bars_4h = resample_bars(df, "4h")
    bars_6h = resample_bars(df, "6h")
    bars_12h = resample_bars(df, "12h")

    if len(bars_5m) < 60 or len(bars_4h) < 10:
        return None

    bars_5m = add_rsi_features(bars_5m)
    bars_5m = add_ema_bos_features_5m(bars_5m)
    bars_5m = add_extra_features_5m(bars_5m)
    bars_15m = add_rsi_features(bars_15m)
    bars_30m = add_rsi_features(bars_30m)
    bars_4h = add_rsi_features(bars_4h)
    bars_4h = compute_adx(bars_4h)
    bars_6h = add_rsi_features(bars_6h)
    bars_12h = add_rsi_features(bars_12h)

    merged = bars_5m.copy()
    merged = merge_asof_feature(merged, bars_15m, "m15", ["rsi", "rsi_slope_1"])
    merged = merge_asof_feature(merged, bars_30m, "m30", ["rsi"])
    merged = merge_asof_feature(merged, bars_4h, "h4", ["rsi", "rsi_slope_1", "rsi_slope_2", "fresh_long", "fresh_short", "bars_since_cross", "adx", "plus_di", "minus_di"])
    merged = merge_asof_feature(merged, bars_6h, "h6", ["rsi"])
    merged = merge_asof_feature(merged, bars_12h, "h12", ["rsi"])

    for c in ["h4_fresh_long", "h4_fresh_short"]:
        merged[c] = merged[c].fillna(False).astype(bool)
    merged = merged.dropna().reset_index(drop=True)

    merged = _setup_engine(merged)
    merged = _structure_gate(merged)
    merged = _confidence(merged)
    merged = _trade_decision(merged)
    return merged


def _setup_engine(df):
    df = df.copy()
    sl = df["h4_fresh_long"] & (df["m15_rsi"] > 50) & (df["m15_rsi"].shift(1) > 50) & (df["h6_rsi"] > 50) & (df["h12_rsi"] > 50)
    ss = df["h4_fresh_short"] & (df["m15_rsi"] < 50) & (df["m15_rsi"].shift(1) < 50) & (df["h6_rsi"] < 50) & (df["h12_rsi"] < 50)
    tl = (df["h4_rsi"] > 55) & (df["h4_rsi_slope_1"] > 0) & (df["h4_rsi_slope_2"] > 0) & (df["m15_rsi"] > 55) & (df["m15_rsi_slope_1"] >= 0) & (df["h12_rsi"] > 50)
    ts = (df["h4_rsi"] < 45) & (df["h4_rsi_slope_1"] < 0) & (df["h4_rsi_slope_2"] < 0) & (df["m15_rsi"] < 45) & (df["m15_rsi_slope_1"] <= 0) & (df["h12_rsi"] < 50)

    df["setup_type"] = "none"
    df["direction"] = 0
    df["stage"] = 0
    df.loc[sl, ["setup_type", "direction"]] = ["RSI_SCALP", 1]
    df.loc[ss, ["setup_type", "direction"]] = ["RSI_SCALP", -1]
    df.loc[tl, ["setup_type", "direction"]] = ["RSI_TREND", 1]
    df.loc[ts, ["setup_type", "direction"]] = ["RSI_TREND", -1]

    bsc = df["h4_bars_since_cross"].fillna(99).astype(int)
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc <= 1), "stage"] = 1
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 2) & (bsc <= 3), "stage"] = 2
    return df


def _structure_gate(df):
    out = df.copy()
    out["trigger_ok"] = False
    out.loc[(out["direction"] == 1), "trigger_ok"] = out["structure_trigger_long"]
    out.loc[(out["direction"] == -1), "trigger_ok"] = out["structure_trigger_short"]
    return out


def _confidence(df):
    out = df.copy()
    h4a_l = (out["h4_rsi"] > 50).astype(float)
    h4a_s = (out["h4_rsi"] < 50).astype(float)
    h6a_l = (out["h6_rsi"] > 50).astype(float)
    h6a_s = (out["h6_rsi"] < 50).astype(float)
    h12a_l = (out["h12_rsi"] > 50).astype(float)
    h12a_s = (out["h12_rsi"] < 50).astype(float)
    align_l = 0.4*h4a_l + 0.3*h6a_l + 0.3*h12a_l
    align_s = 0.4*h4a_s + 0.3*h6a_s + 0.3*h12a_s

    fresh = (1 - out["h4_bars_since_cross"]/4.0).clip(0, 1).fillna(0)
    mom = (0.4*bounded_positive_score(out["h4_rsi_slope_1"]) + 0.4*bounded_positive_score(out["m15_rsi_slope_1"]) + 0.2*bounded_positive_score(out["ret_3"])).clip(0, 1)

    edr = max(out["ema_dist_pct"].abs().quantile(0.8), 1e-9)
    edq = (1 - out["ema_dist_pct"].abs()/edr).clip(0, 1)

    str_l = (0.4*out["ema_reclaim_long"].astype(float) + 0.4*out["bos_long"].astype(float) + 0.2*edq).clip(0, 1)
    str_s = (0.4*out["ema_reclaim_short"].astype(float) + 0.4*out["bos_short"].astype(float) + 0.2*edq).clip(0, 1)

    rv = rolling_minmax_score(out["rv_6"].fillna(0)).fillna(0.5)
    cfx_l = (out["m30_rsi"] > 50).astype(float)
    cfx_s = (out["m30_rsi"] < 50).astype(float)
    cfx = np.where(out["direction"] == 1, cfx_l, cfx_s)

    out["confidence_raw"] = 0.0
    sc_mask = out["setup_type"] == "RSI_SCALP"
    tr_mask = out["setup_type"] == "RSI_TREND"

    sc_a = np.where(out["direction"] == 1, align_l, align_s)
    sc_s = np.where(out["direction"] == 1, str_l, str_s)
    out.loc[sc_mask, "confidence_raw"] = (0.30*sc_a + 0.25*fresh + 0.20*mom + 0.15*rv + 0.10*sc_s)[sc_mask]

    tr_a = np.where(out["direction"] == 1, align_l, align_s)
    tr_s = np.where(out["direction"] == 1, str_l, str_s)
    out.loc[tr_mask, "confidence_raw"] = (0.25*tr_a + 0.20*mom + 0.30*tr_s + 0.15*rv + 0.10*cfx)[tr_mask]

    out["confidence_raw"] = out["confidence_raw"].clip(0, 1)

    if ENABLE_ADX_FILTER and "h4_adx" in out.columns:
        adx = out["h4_adx"].fillna(20)
        mod = pd.Series(0.0, index=out.index)
        mod[adx >= ADX_TRENDING_BONUS] = 0.03
        mod[adx < ADX_CHOPPY_PENALTY] = -0.03
        out["confidence_raw"] = (out["confidence_raw"] + mod).clip(0, 1)

    eg = (fresh >= 0.8) & (mom >= 0.7) & (np.where(out["direction"]==1, str_l, str_s) >= 0.85) & (rv >= 0.55) & (edq >= 0.55)

    out["confidence_mode"] = "NO_TRADE"
    out.loc[(out["confidence_raw"] >= NO_TRADE_THRESHOLD) & (out["confidence_raw"] < MILD_THRESHOLD), "confidence_mode"] = "MILD"
    out.loc[(out["confidence_raw"] >= MILD_THRESHOLD) & (out["confidence_raw"] < MID_THRESHOLD), "confidence_mode"] = "MID"
    out.loc[(out["confidence_raw"] >= MID_THRESHOLD) & (out["confidence_raw"] < PREMIUM_THRESHOLD), "confidence_mode"] = "PREMIUM"
    out.loc[(out["confidence_raw"] >= PREMIUM_THRESHOLD) & (out["confidence_raw"] < HIGH_THRESHOLD), "confidence_mode"] = "HIGH"
    out.loc[(out["confidence_raw"] >= HIGH_THRESHOLD) & eg, "confidence_mode"] = "ELITE"
    return out


def _trade_decision(df):
    out = df.copy()
    ok = out["confidence_mode"].isin(ALLOWED_MODES) & out["direction"].isin({1, -1})
    vt = out["volume"].quantile(0.30)
    out["take_trade"] = (out["setup_type"] != "none") & out["trigger_ok"] & (out["confidence_mode"] != "NO_TRADE") & ok & (out["volume"] >= vt)
    return out


# =========================================================
# SIGNAL CHECK
# =========================================================

def check_for_signals():
    for symbol in SYMBOLS:
        # Only trade on whitelisted symbols
        if symbol not in TRADING_SYMBOLS:
            continue
        # Skip if already in a position
        if symbol in active_positions:
            continue

        try:
            analysis = build_analysis(symbol)
            if analysis is None:
                log.debug(f"[{symbol}] No analysis (not enough data)")
                continue

            latest = analysis.iloc[-1]

            # ── Diagnostic logging (every check cycle) ──
            setup = str(latest.get("setup_type", "none"))
            conf = float(latest.get("confidence_raw", 0))
            mode = str(latest.get("confidence_mode", "NO_TRADE"))
            trigger = bool(latest.get("trigger_ok", False))
            take = bool(latest.get("take_trade", False))
            h4_rsi = float(latest.get("h4_rsi", 0))
            h4_adx = float(latest.get("h4_adx", 0)) if pd.notna(latest.get("h4_adx")) else 0
            dir_val = int(latest.get("direction", 0))

            if setup != "none":
                reason = ""
                if not trigger:
                    reason = "NO_TRIGGER (EMA reclaim + BOS not aligned in 3 bars)"
                elif mode == "NO_TRADE":
                    reason = f"LOW_CONF ({conf:.1%} < 0.72 threshold)"
                elif mode not in ALLOWED_MODES:
                    reason = f"MODE_BLOCKED ({mode} not in {ALLOWED_MODES})"
                else:
                    reason = "PASS"

                log.info(
                    f"[{symbol}] setup={setup} dir={'LONG' if dir_val==1 else 'SHORT' if dir_val==-1 else '?'} "
                    f"conf={conf:.1%} mode={mode} trigger={'✓' if trigger else '✗'} "
                    f"H4_RSI={h4_rsi:.1f} ADX={h4_adx:.1f} → {reason}"
                )
            else:
                # Only log periodically to avoid spam (every ~10 min for idle symbols)
                log.debug(f"[{symbol}] No setup — H4_RSI={h4_rsi:.1f} ADX={h4_adx:.1f}")

            if not take:
                continue

            ts = str(latest["timestamp"])
            direction = int(latest["direction"])
            sig_key = (symbol, ts, direction)
            if sig_key in sent_signals:
                continue

            if symbol in last_signal_time:
                elapsed = (datetime.now(timezone.utc) - last_signal_time[symbol]).total_seconds()
                if elapsed < COOLDOWN_BARS_5M * 5 * 60:
                    continue

            text, keyboard = format_signal(symbol, latest)
            result = send_telegram(text, keyboard)
            if result:
                sent_signals.add(sig_key)
                last_signal_time[symbol] = datetime.now(timezone.utc)
                log.info(f"🚀 SIGNAL: {symbol} {latest['setup_type']} {'LONG' if direction==1 else 'SHORT'} conf={latest['confidence_raw']:.1%}")

            if len(sent_signals) > 100:
                for k in sorted(sent_signals)[:50]:
                    sent_signals.discard(k)

        except Exception as e:
            log.error(f"Error checking {symbol}: {e}", exc_info=True)


# =========================================================
# CALLBACK HANDLER (Telegram button presses)
# =========================================================

def poll_telegram_updates():
    """Long-poll Telegram for callback queries."""
    offset = 0
    while running:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            resp = requests.get(url, params={"offset": offset, "timeout": 10}, timeout=15)
            data = resp.json()

            if not data.get("ok"):
                time.sleep(2)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1

                if "callback_query" in update:
                    handle_callback(update["callback_query"])

        except Exception as e:
            log.error(f"Poll error: {e}")
            time.sleep(3)


def handle_callback(cq: dict):
    """Process a button press."""
    callback_id = cq["id"]
    data = cq.get("data", "")
    msg_id = cq.get("message", {}).get("message_id")

    if data in pending_callbacks:
        answer_callback(callback_id, "Already processing...")
        return

    pending_callbacks.add(data)

    try:
        if data.startswith("go_long_") or data.startswith("go_short_"):
            parts = data.split("_")
            direction = "LONG" if parts[1] == "long" else "SHORT"
            # Check for leverage suffix
            leverage = DEFAULT_LEVERAGE
            symbol = parts[2]
            if len(parts) > 3 and parts[3].endswith("x"):
                leverage = int(parts[3].replace("x", ""))
                symbol = f"{parts[2]}USDT"
            else:
                symbol = f"{parts[2]}USDT"

            execute_trade(symbol, direction, leverage, msg_id, callback_id)

        elif data.startswith("pass_"):
            answer_callback(callback_id, "Signal passed ✋")
            if msg_id:
                edit_telegram(msg_id, cq["message"]["text"] + "\n\n<i>❌ Passed by user</i>")

        elif data.startswith("close_"):
            parts = data.split("_")
            symbol = f"{parts[1]}USDT"
            fraction = 0.5 if parts[2] == "half" else 1.0
            close_trade(symbol, fraction, msg_id, callback_id)

    finally:
        pending_callbacks.discard(data)


def execute_trade(symbol: str, direction: str, leverage: int, msg_id: int | None, callback_id: str):
    """Execute a trade from Telegram button press."""
    with lock:
        if symbol in active_positions:
            answer_callback(callback_id, f"Already in {symbol} position!")
            return

        buffer = candle_buffers.get(symbol, [])
        if not buffer:
            answer_callback(callback_id, "No price data yet!")
            return

        entry_price = buffer[-1]["price"]

        # Compute stop
        analysis = build_analysis(symbol)
        if analysis is None:
            answer_callback(callback_id, "Analysis unavailable!")
            return

        latest = analysis.iloc[-1]
        # Structural stop: 4-bar lookback + EMA20 for RSI_TREND (matches backtester v5.5)
        if direction == "LONG":
            structural_stop = float(latest.get("structural_low", latest["low"]))
            if str(latest["setup_type"]) == "RSI_TREND":
                structural_stop = min(structural_stop, float(latest["ema20"]))
            stop_pct = max((entry_price - structural_stop) / entry_price, 0.004)
            stop_price = entry_price * (1 - stop_pct)
        else:
            structural_stop = float(latest.get("structural_high", latest["high"]))
            if str(latest["setup_type"]) == "RSI_TREND":
                structural_stop = max(structural_stop, float(latest["ema20"]))
            stop_pct = max((structural_stop - entry_price) / entry_price, 0.004)
            stop_price = entry_price * (1 + stop_pct)

        tp2_pct = TP2_PCT_TREND if latest["setup_type"] == "RSI_TREND" else TP2_PCT_SCALP
        sign = 1 if direction == "LONG" else -1

        pos = {
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "stop_pct": stop_pct,
            "tp1": entry_price * (1 + sign * TP1_PCT),
            "tp2": entry_price * (1 + sign * tp2_pct),
            "tp3": entry_price * (1 + sign * TP3_PCT),
            "tp4": entry_price * (1 + sign * TP4_PCT),
            "tp1_hit": False,
            "tp2_hit": False,
            "tp3_hit": False,
            "tp4_hit": False,
            "remaining": 1.0,
            "mode": str(latest["confidence_mode"]),
            "strategy": str(latest["setup_type"]),
            "confidence": float(latest["confidence_raw"]),
            "leverage": leverage,
            "size_usd": 0,  # will be set
            "entry_time": datetime.now(timezone.utc),
            "msg_id": msg_id,
            "break_even_armed": False,
            "trailing_active": False,
            "best_price": entry_price,
            "trail_stop": None,
        }

        # Position sizing based on actual account balance
        equity = get_account_balance()
        mode_risk = {"MILD": 0.025, "MID": 0.03, "PREMIUM": 0.0325, "HIGH": 0.035, "ELITE": 0.04}
        risk_pct = mode_risk.get(pos["mode"], 0.03)
        size_usd = (risk_pct * equity) / stop_pct
        pos["size_usd"] = min(size_usd, equity * 10)  # cap at 10x leverage
        log.info(f"📊 Position sizing: equity=${equity:.2f} risk={risk_pct:.1%} stop={stop_pct:.2%} → size=${size_usd:.2f}")

        # Safety: reject if equity too low or position too small
        if equity < 1.0:
            answer_callback(callback_id, f"❌ Insufficient balance: ${equity:.2f}")
            return
        if size_usd < 5.0:
            answer_callback(callback_id, f"❌ Position too small: ${size_usd:.2f} (min $5)")
            return

        # Execute
        order = execute_order(symbol, direction, pos["size_usd"], leverage)
        if order is None:
            answer_callback(callback_id, "❌ Order failed!")
            return

        active_positions[symbol] = pos
        answer_callback(callback_id, f"✅ {direction} {symbol} @ ${entry_price:,.2f}")

        # Edit message to show it's been executed
        if msg_id:
            mode = text if (text := cq_text_safe(cq_from_callback_id(callback_id))) else ""
            edit_telegram(
                msg_id,
                f"{'━' * 24}\n"
                f"✅ <b>EXECUTED</b>\n"
                f"{'🟢' if direction == 'LONG' else '🔴'} {direction} {symbol}\n"
                f"📊 {leverage}x | ${pos['size_usd']:,.0f}\n"
                f"📍 Entry: ${entry_price:,.2f}\n"
                f"🛑 Stop: ${stop_price:,.2f}\n"
                f"{'━' * 24}"
            )

        # Send position status with close buttons
        text, keyboard = format_position_status(symbol)
        send_telegram(text, keyboard)

        log.info(f"✅ TRADE OPENED: {direction} {symbol} @ ${entry_price:,.2f} lev={leverage}x size=${pos['size_usd']:,.0f}")


def cq_text_safe(cq):
    return cq.get("message", {}).get("text", "") if cq else ""

def cq_from_callback_id(cid):
    return None  # not stored, skip


def close_trade(symbol: str, fraction: float, msg_id: int | None, callback_id: str):
    """Close position from button press."""
    with lock:
        if symbol not in active_positions:
            answer_callback(callback_id, "No open position!")
            return

        order = close_position(symbol, fraction)
        if order is None:
            answer_callback(callback_id, "❌ Close failed!")
            return

        pos = active_positions[symbol]
        pos["remaining"] -= fraction

        if pos["remaining"] <= 0.01:
            del active_positions[symbol]
            answer_callback(callback_id, f"✅ Closed {symbol}")
            send_telegram(f"✅ <b>{symbol} position fully closed</b>")
        else:
            answer_callback(callback_id, f"✅ Closed {fraction:.0%} of {symbol}")
            text, keyboard = format_position_status(symbol)
            send_telegram(text, keyboard)


# =========================================================
# POSITION MONITOR (runs every 15s)
# =========================================================

def monitor_positions():
    """Check active positions for TP/SL hits and send alerts."""
    while running:
        try:
            with lock:
                for symbol in list(active_positions.keys()):
                    pos = active_positions[symbol]
                    buffer = candle_buffers.get(symbol, [])
                    if not buffer:
                        continue

                    current = buffer[-1]
                    price = current["price"]
                    direction = pos["direction"]
                    sign = 1 if direction == "LONG" else -1

                    # Check stop loss
                    if direction == "LONG" and price <= pos["stop_price"]:
                        _close_and_notify(symbol, "🛑 STOP LOSS hit", price)
                        continue
                    elif direction == "SHORT" and price >= pos["stop_price"]:
                        _close_and_notify(symbol, "🛑 STOP LOSS hit", price)
                        continue

                    # Check TP1
                    if not pos["tp1_hit"]:
                        if (direction == "LONG" and price >= pos["tp1"]) or (direction == "SHORT" and price <= pos["tp1"]):
                            pos["tp1_hit"] = True
                            pos["remaining"] -= TP1_SIZE
                            pos["break_even_armed"] = True
                            close_position(symbol, TP1_SIZE)
                            send_telegram(f"🎯 <b>TP1 HIT</b> — {symbol} {direction} @ ${pos['tp1']:,.2f}\n+{TP1_PCT:.1%} on 35% | BE armed")

                    # Check TP2
                    if not pos["tp2_hit"]:
                        if (direction == "LONG" and price >= pos["tp2"]) or (direction == "SHORT" and price <= pos["tp2"]):
                            pos["tp2_hit"] = True
                            pos["remaining"] -= TP2_SIZE
                            pos["trailing_active"] = True
                            pos["best_price"] = price
                            close_position(symbol, TP2_SIZE)
                            send_telegram(f"🎯 <b>TP2 HIT</b> — {symbol} {direction} @ ${pos['tp2']:,.2f}\n+{TP2_PCT_SCALP if pos['strategy']=='RSI_SCALP' else TP2_PCT_TREND:.1%} on 35% | Trailing ON")

                    # Check TP3
                    if not pos["tp3_hit"]:
                        if (direction == "LONG" and price >= pos["tp3"]) or (direction == "SHORT" and price <= pos["tp3"]):
                            pos["tp3_hit"] = True
                            pos["remaining"] -= TP3_SIZE
                            close_position(symbol, TP3_SIZE)
                            send_telegram(f"🎯 <b>TP3 HIT</b> — {symbol} {direction} @ ${pos['tp3']:,.2f}\n+{TP3_PCT:.1%} on 15%")

                    # Check TP4
                    if not pos["tp4_hit"]:
                        if (direction == "LONG" and price >= pos["tp4"]) or (direction == "SHORT" and price <= pos["tp4"]):
                            pos["tp4_hit"] = True
                            close_position(symbol, 1.0)
                            send_telegram(f"🚀 <b>TP4 MOON HIT</b> — {symbol} {direction} @ ${pos['tp4']:,.2f}\n+{TP4_PCT:.1%}!")
                            del active_positions[symbol]
                            continue

                    # Break-even stop after TP1
                    if pos["break_even_armed"] and not pos["trailing_active"]:
                        be_price = pos["entry_price"] * (1 + 0.0002 * sign)
                        if (direction == "LONG" and price <= be_price) or (direction == "SHORT" and price >= be_price):
                            _close_and_notify(symbol, "⚖️ BREAK-EVEN stop", price)
                            continue

                    # Trailing stop after TP2
                    if pos["trailing_active"]:
                        trail_pct = get_trail_pct(pos["mode"])
                        if direction == "LONG":
                            pos["best_price"] = max(pos["best_price"], price)
                            trail_stop = pos["best_price"] * (1 - trail_pct)
                            if pos["trail_stop"] is None or trail_stop > pos["trail_stop"]:
                                pos["trail_stop"] = trail_stop
                            if price <= pos["trail_stop"]:
                                _close_and_notify(symbol, "🔄 TRAILING STOP", price)
                                continue
                        else:
                            pos["best_price"] = min(pos["best_price"], price)
                            trail_stop = pos["best_price"] * (1 + trail_pct)
                            if pos["trail_stop"] is None or trail_stop < pos["trail_stop"]:
                                pos["trail_stop"] = trail_stop
                            if price >= pos["trail_stop"]:
                                _close_and_notify(symbol, "🔄 TRAILING STOP", price)
                                continue

        except Exception as e:
            log.error(f"Monitor error: {e}", exc_info=True)

        time.sleep(15)


def _close_and_notify(symbol: str, reason: str, price: float):
    """Close full position and notify."""
    pos = active_positions.get(symbol)
    if not pos:
        return

    if pos["remaining"] > 0.01:
        close_position(symbol, 1.0)

    direction = pos["direction"]
    if direction == "LONG":
        pnl = (price - pos["entry_price"]) / pos["entry_price"]
    else:
        pnl = (pos["entry_price"] - price) / pos["entry_price"]

    send_telegram(
        f"{reason}\n"
        f"{'🟢' if direction == 'LONG' else '🔴'} {direction} {symbol}\n"
        f"📍 Entry: ${pos['entry_price']:,.2f}\n"
        f"💲 Exit: ${price:,.2f}\n"
        f"{'💰' if pnl > 0 else '📉'} PnL: {pnl:+.2%}"
    )

    del active_positions[symbol]


# =========================================================
# WEBSOCKET
# =========================================================

def on_message(ws, message):
    try:
        data = json.loads(message)
        k = data.get("k", {})
        symbol = k.get("s", "")
        if not k.get("x", False):
            return
        candle_buffers[symbol].append({
            "timestamp": pd.Timestamp(k["t"], unit="ms", tz="UTC"),
            "price": float(k["c"]),
            "volume": float(k["v"]),
            "delta": float(k["V"]) - (float(k["v"]) - float(k["V"])),
            "trade_count": float(k["n"]),
        })
        if len(candle_buffers[symbol]) > ROLLING_WINDOW:
            candle_buffers[symbol] = candle_buffers[symbol][-ROLLING_WINDOW:]
    except Exception as e:
        log.error(f"WS parse error: {e}")


def on_error(ws, error):
    log.error(f"WS error: {error}")


def on_close(ws, code, msg):
    log.warning(f"WS closed: {code} {msg}")
    if running:
        time.sleep(5)
        start_websocket()


def on_open(ws):
    log.info("WebSocket connected!")
    streams = [f"{s.lower()}@kline_1m" for s in SYMBOLS]
    ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1}))
    log.info(f"Subscribed: {', '.join(streams)}")


def start_websocket():
    streams = "/".join(f"{s.lower()}@kline_1m" for s in SYMBOLS)
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=30, ping_timeout=10)


# =========================================================
# MAIN
# =========================================================

def signal_handler(sig, frame):
    global running
    log.info("Shutdown...")
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mode_label = "📝 PAPER" if PAPER_MODE else "🔴 LIVE"
    log.info("=" * 60)
    log.info(f"WRAITH — {mode_label}")
    log.info(f"Symbols: {', '.join(SYMBOLS)}")
    log.info("=" * 60)

    send_telegram(
        f"🪦 <b>WRAITH Online</b>\n"
        f"Mode: {mode_label}\n"
        f"Watching: {', '.join(SYMBOLS)}\n"
        f"{'⚡ Ready to trade!' if not PAPER_MODE else '📝 Paper mode — no real orders'}"
    )

    # Start WebSocket
    ws_thread = threading.Thread(target=start_websocket, daemon=True)
    ws_thread.start()

    # Start Telegram poller
    poll_thread = threading.Thread(target=poll_telegram_updates, daemon=True)
    poll_thread.start()

    # Start position monitor
    monitor_thread = threading.Thread(target=monitor_positions, daemon=True)
    monitor_thread.start()

    # Wait for data
    log.info("Waiting 5 min for data accumulation...")
    time.sleep(300)

    # Main signal loop
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
