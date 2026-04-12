import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = "data/features/research_dataset.csv"
# Multi-asset: run backtest on each symbol and combine
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

RSI_PERIOD = 14
EMA_PERIOD = 20

INITIAL_EQUITY = 10000.0

# ── Fix #5: Symbol-specific slippage ──
ROUND_TRIP_COST_DEFAULT = 0.0008
ROUND_TRIP_COST_BY_SYMBOL = {
    "BTCUSDT": 0.0005,
    "ETHUSDT": 0.0008,
    "SOLUSDT": 0.0012,
    "XRPUSDT": 0.0015,
    "CHZUSDT": 0.0015,   # low-liquidity alt
}

def get_round_trip_cost(symbol: str) -> float:
    return ROUND_TRIP_COST_BY_SYMBOL.get(symbol, ROUND_TRIP_COST_DEFAULT)

# ── Fix #4: Position sizing config ──
# Max notional as fraction of equity (prevents oversized positions on expensive assets)
MAX_NOTIONAL_FRACTION = 10.0   # position_notional <= equity * 10 (10x leverage cap)
# If position_notional > equity * MAX_NOTIONAL_FRACTION, scale risk_fraction down

# Risk by mode
RISK_LOW = 0.0000
RISK_MILD = 0.0250     # NEW: smaller size for marginal setups (0.72-0.78)
RISK_MID = 0.0300
RISK_PREMIUM = 0.0325   # between MID and HIGH
RISK_HIGH = 0.0350
RISK_ELITE = 0.0400

# TP structure
TP1_PCT = 0.005
TP2_PCT_SCALP = 0.007    # RSI_SCALP: +0.7%
TP2_PCT_TREND = 0.006    # RSI_TREND: +0.6% (confirmed from live signals)
TP3_PCT = 0.008
TP4_PCT = 0.090

TP1_SIZE = 0.35
TP2_SIZE = 0.35
TP3_SIZE = 0.15
TP4_SIZE = 0.15

# Trailing by mode
TRAIL_PCT_MID = 0.0050
TRAIL_PCT_HIGH = 0.0045
TRAIL_PCT_ELITE = 0.0040

# Break-even after TP1
ENABLE_BREAK_EVEN_AFTER_TP1 = True
BREAK_EVEN_OFFSET = 0.0002

# ── Fix #2: Reduced cooldown + wider bands ──
COOLDOWN_BARS_5M = 3   # was 6 — more trades per week

# Confidence bands — lowered to generate more trades
NO_TRADE_THRESHOLD = 0.72
MILD_THRESHOLD = 0.78    # NEW: small-size tier for marginal setups
MID_THRESHOLD = 0.80
PREMIUM_THRESHOLD = 0.82  # was 0.85 — widened
HIGH_THRESHOLD = 0.88     # was 0.90 — widened

USE_HOUR_FILTER = True
# Crypto trades 24/7 but some hours have better liquidity/moves:
# US market hours + Asia overlap + London open
ALLOWED_HOURS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
# All hours for now — filter disabled functionally but framework is in place
# To actually filter: set specific hours like {13, 14, 15, 16, 17, 18} for US session

# =========================================================
# ACTIVE BASELINE — HYBRID v2
# =========================================================
# HIGH -> LONG + SHORT (all)
# MID -> LONG + SHORT (gated by CFX=1 only)
# ELITE paused

BASELINE_MODE = "HYBRID"

if BASELINE_MODE == "HIGH_ONLY":
    ALLOWED_MODES = {"HIGH"}
    ALLOW_LONG = True
    ALLOW_SHORT = True

elif BASELINE_MODE == "MID_ONLY":
    ALLOWED_MODES = {"MID"}
    ALLOW_LONG = True
    ALLOW_SHORT = True

elif BASELINE_MODE == "HYBRID":
    # MILD (0.72-0.78): small size, all directions — was being skipped entirely
    # PREMIUM (0.80-0.82): active
    # HIGH (0.82-0.88): active
    # ELITE (0.88+): active
    # MID (0.78-0.80): still disabled (confirmed drag)
    ALLOWED_MODES = {"MILD", "HIGH", "PREMIUM", "ELITE"}
    ALLOW_LONG = True
    ALLOW_SHORT = True

else:
    raise ValueError(f"Invalid BASELINE_MODE: {BASELINE_MODE}")

# Volume/delta filter
ENABLE_VOLUME_FILTER = True
VOLUME_PERCENTILE = 0.30   # require volume above 30th percentile (not dead candles)
DELTA_ALIGN_REQUIRED = False  # if True, delta must agree with direction


# =========================================================
# LOAD
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = ["timestamp", "price"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

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
# INDICATORS
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
        (df["close"].shift(1) <= df["ema20"].shift(1)) &
        (df["close"] > df["ema20"])
    )
    df["ema_reclaim_short"] = (
        (df["close"].shift(1) >= df["ema20"].shift(1)) &
        (df["close"] < df["ema20"])
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


# =========================================================
# FIX #1: ADX REGIME FILTER
# =========================================================

# Config
ADX_PERIOD = 14
ADX_TRENDING_BONUS = 25.0    # ADX > 25 = strong trend → confidence bonus
ADX_CHOPPY_PENALTY = 15.0    # ADX < 15 = weak trend → confidence penalty
ENABLE_ADX_FILTER = True     # now a confidence modifier, not a gate

def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """
    Compute ADX (Average Directional Index) from OHLC bars.
    Returns the dataframe with +DI, -DI, DX, ADX columns added.
    """
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed (Wilder's method = EMA with alpha=1/period)
    alpha = 1.0 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Directional Indicators
    plus_di = 100.0 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

    # DX and ADX
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


def merge_asof_feature(base: pd.DataFrame, other: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
    rhs = other[["timestamp"] + cols].copy()
    rhs = rhs.rename(columns={c: f"{prefix}_{c}" for c in cols})

    return pd.merge_asof(
        base.sort_values("timestamp"),
        rhs.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def direction_label(direction: int) -> str:
    return "LONG" if direction == 1 else "SHORT"


# =========================================================
# SETUP ENGINE
# =========================================================

def build_setup_engine(df5: pd.DataFrame) -> pd.DataFrame:
    df = df5.copy()

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

    # Stage: derived from h4_bars_since_cross (only meaningful for RSI_SCALP)
    # Stage 1 = very fresh (0-1 bars), Stage 2 = still fresh (2-3 bars), Stage 3 = aging (4+)
    bsc = df["h4_bars_since_cross"].fillna(99).astype(int)
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc <= 1), "stage"] = 1
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 2) & (bsc <= 3), "stage"] = 2
    df.loc[(df["setup_type"] == "RSI_SCALP") & (bsc >= 4), "stage"] = 3

    return df


# =========================================================
# STRUCTURE GATE
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
# CONFIDENCE ENGINE
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

    # CFX: 30M RSI alignment (confirmed from live signals: CFX 30M)
    cfx_long = (out["m30_rsi"] > 50).astype(float)
    cfx_short = (out["m30_rsi"] < 50).astype(float)
    cfx_align = np.where(out["direction"] == 1, cfx_long, cfx_short)
    out["cfx_score"] = cfx_align  # binary: 1 if 30M confirms, 0 if not

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

    # ── Fix #1: ADX as confidence modifier (not a gate) ──
    # ADX confirms trend strength for the direction the system already chose
    if ENABLE_ADX_FILTER and "h4_adx" in out.columns:
        adx_val = out["h4_adx"].fillna(20)  # default to neutral if missing
        adx_modifier = pd.Series(0.0, index=out.index)
        adx_modifier[adx_val >= ADX_TRENDING_BONUS] = 0.03    # strong trend: +3%
        adx_modifier[(adx_val >= 15) & (adx_val < 25)] = 0.0  # normal: no change
        adx_modifier[adx_val < ADX_CHOPPY_PENALTY] = -0.03    # weak/choppy: -3%
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

    # MILD: 0.72-0.78 — small size, marginal setups (new tier for more trades)
    out.loc[
        (out["confidence_raw"] >= NO_TRADE_THRESHOLD) &
        (out["confidence_raw"] < MILD_THRESHOLD),
        "confidence_mode"
    ] = "MILD"

    # MID: 0.78-0.80 — still disabled (confirmed drag in backtest)
    out.loc[
        (out["confidence_raw"] >= MILD_THRESHOLD) &
        (out["confidence_raw"] < MID_THRESHOLD),
        "confidence_mode"
    ] = "MID"

    # PREMIUM: 0.80-0.82
    out.loc[
        (out["confidence_raw"] >= MID_THRESHOLD) &
        (out["confidence_raw"] < PREMIUM_THRESHOLD),
        "confidence_mode"
    ] = "PREMIUM"

    # HIGH: 0.82-0.88
    out.loc[
        (out["confidence_raw"] >= PREMIUM_THRESHOLD) &
        (out["confidence_raw"] < HIGH_THRESHOLD),
        "confidence_mode"
    ] = "HIGH"

    # ELITE: 0.88+ with elite_gate
    out.loc[
        (out["confidence_raw"] >= HIGH_THRESHOLD) & elite_gate,
        "confidence_mode"
    ] = "ELITE"

    out.loc[out["confidence_mode"] == "MILD", "risk_fraction"] = RISK_MILD
    out.loc[out["confidence_mode"] == "MID", "risk_fraction"] = RISK_MID
    out.loc[out["confidence_mode"] == "PREMIUM", "risk_fraction"] = RISK_PREMIUM
    out.loc[out["confidence_mode"] == "HIGH", "risk_fraction"] = RISK_HIGH
    out.loc[out["confidence_mode"] == "ELITE", "risk_fraction"] = RISK_ELITE

    return out


# =========================================================
# SHORT VALIDATION LAYER (MID ONLY)
# =========================================================

def short_validation_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies additional bearish evidence gates for MID SHORT trades.
    Does NOT touch HIGH, ELITE, or any LONG trades.

    ALL conditions must pass:
      1. HTF Bearish Strength: 2+ of {h4, h6, h12} RSI < 45
      2. 15M Momentum: m15 RSI slope negative for 3+ consecutive bars
      3. Not Near Recent Low: price > 1% above rolling 1H low
      4. EMA Stack Bearish: ema20(5m) < ema50(5m)
      5. Price Below Both EMAs: close < ema20 AND close < ema50
    """
    out = df.copy()
    out["short_valid"] = True  # default: pass for non-MID-short rows

    mid_short_mask = (out["confidence_mode"] == "MID") & (out["direction"] == -1)

    if not mid_short_mask.any():
        return out

    # --- Condition 1: HTF Bearish Strength ---
    # Need 2+ of 3 HTF RSI genuinely bearish (below 45, not just 50)
    htf_bearish_count = (
        (out["h4_rsi"] < 45).astype(int) +
        (out["h6_rsi"] < 45).astype(int) +
        (out["h12_rsi"] < 45).astype(int)
    )
    c1 = htf_bearish_count >= 2

    # --- Condition 2: 15M Momentum Sustained ---
    # m15 RSI slope negative for at least 3 consecutive bars
    # Catches: counter-trend dips that look like shorts but reverse fast
    m15_slope_neg = out["m15_rsi_slope_1"] < -0.2
    consecutive_neg = m15_slope_neg.rolling(3).sum() == 3
    c2 = consecutive_neg.fillna(False)

    # --- Condition 3: Not Near Recent Low ---
    # Price must be > 1% above 12-bar rolling low (~1H at 5m)
    # Catches: shorting into support/squeeze zones
    rolling_low_1h = out["low"].rolling(12).min()
    c3 = (out["close"] - rolling_low_1h) / out["close"] > 0.01

    # --- Condition 4: EMA Stack Bearish ---
    # ema20 must be below ema50 on 5m (short-term structure confirms bearish)
    c4 = out["ema20"] < out["ema50"]

    # --- Condition 5: Price Below Both EMAs ---
    # Price must be trading below both EMAs (not just dipping below ema20)
    c5 = (out["close"] < out["ema20"]) & (out["close"] < out["ema50"])

    # Apply: only mid_short_mask rows are filtered
    all_conditions = c1 & c2 & c3 & c4 & c5
    out.loc[mid_short_mask & ~all_conditions, "short_valid"] = False

    return out


# =========================================================
# TRADE DECISION
# =========================================================

def add_trade_decision(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    hour_ok = True
    if USE_HOUR_FILTER:
        hour_ok = out["timestamp"].dt.hour.isin(ALLOWED_HOURS)

    if BASELINE_MODE == "HYBRID":
        # HIGH: all directions
        high_ok = (out["confidence_mode"] == "HIGH") & out["direction"].isin({1, -1})
        # PREMIUM: new tier 0.82-0.88
        premium_ok = (out["confidence_mode"] == "PREMIUM") & out["direction"].isin({1, -1})
        # ELITE: treated as HIGH
        elite_ok = (out["confidence_mode"] == "ELITE") & out["direction"].isin({1, -1})
        # MILD: small size, marginal setups (0.72-0.78)
        mild_ok = (out["confidence_mode"] == "MILD") & out["direction"].isin({1, -1})
        # MID: still disabled (0.78-0.80)
        mode_direction_ok = high_ok | premium_ok | elite_ok | mild_ok
    else:
        mode_direction_ok = (
            out["confidence_mode"].isin(ALLOWED_MODES) &
            (
                ((out["direction"] == 1) & ALLOW_LONG) |
                ((out["direction"] == -1) & ALLOW_SHORT)
            )
        )

    # Volume filter: skip dead candles
    volume_ok = True
    if ENABLE_VOLUME_FILTER:
        vol_threshold = out["volume"].quantile(VOLUME_PERCENTILE)
        volume_ok = out["volume"] >= vol_threshold

    # Delta alignment filter (optional)
    delta_ok = True
    if DELTA_ALIGN_REQUIRED:
        delta_ok = (
            ((out["direction"] == 1) & (out["delta_3"] > 0)) |
            ((out["direction"] == -1) & (out["delta_3"] < 0))
        )

    out["take_trade"] = (
        (out["setup_type"] != "none") &
        out["trigger_ok"] &
        (out["confidence_mode"] != "NO_TRADE") &
        mode_direction_ok &
        hour_ok &
        volume_ok &
        delta_ok
    )

    return out


# =========================================================
# STOP LOGIC
# =========================================================

def compute_stop_price(df: pd.DataFrame, entry_idx: int, direction: int, strategy: str) -> tuple[float, float]:
    row = df.iloc[entry_idx]
    entry_price = float(row["close"])

    recent_lookback = 6
    start = max(0, entry_idx - recent_lookback)
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

    stop_distance_pct = max(stop_distance_pct, 0.005)

    if direction == 1:
        stop_price = entry_price * (1 - stop_distance_pct)
    else:
        stop_price = entry_price * (1 + stop_distance_pct)

    return stop_price, stop_distance_pct


def get_trail_pct(confidence_mode: str) -> float:
    if confidence_mode == "MILD":
        return TRAIL_PCT_MID   # wider trail for lower confidence
    if confidence_mode == "MID":
        return TRAIL_PCT_MID
    if confidence_mode == "PREMIUM":
        return TRAIL_PCT_HIGH  # use HIGH trail for PREMIUM
    if confidence_mode == "HIGH":
        return TRAIL_PCT_HIGH
    if confidence_mode == "ELITE":
        return TRAIL_PCT_ELITE
    return TRAIL_PCT_MID


# =========================================================
# EXIT ENGINE
# =========================================================

def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    strategy: str,
    confidence: float,
    confidence_mode: str,
    risk_fraction: float,
    equity_before: float,
    stage: int = 0,
    cfx_score: float = 0.0,
    symbol: str = "BTCUSDT",
) -> dict:
    entry_row = df.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    entry_time = entry_row["timestamp"]

    stop_price, stop_distance_pct = compute_stop_price(df, entry_idx, direction, strategy)

    risk_amount = equity_before * risk_fraction
    position_notional = risk_amount / stop_distance_pct if stop_distance_pct > 0 else 0.0

    # ── Fix #4: Position sizing cap ──
    max_notional = equity_before * MAX_NOTIONAL_FRACTION
    if position_notional > max_notional:
        # Scale risk down so notional doesn't exceed cap
        position_notional = max_notional
        risk_amount = position_notional * stop_distance_pct

    # Strategy-specific TP2
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

    tp1_done = False
    tp2_done = False
    tp3_done = False
    tp4_done = False

    break_even_armed = False
    trailing_active = False
    trail_stop = np.nan
    best_price = entry_price

    exit_reason = "open_end"
    exit_time = entry_time
    exit_price = entry_price

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

            if low <= active_stop:
                realized_return_pct += remaining * ((active_stop / entry_price) - 1.0)
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

            if (not tp1_done) and high >= tp1:
                realized_return_pct += TP1_SIZE * ((tp1 / entry_price) - 1.0)
                remaining -= TP1_SIZE
                tp1_done = True
                if ENABLE_BREAK_EVEN_AFTER_TP1:
                    break_even_armed = True

            if (not tp2_done) and high >= tp2:
                realized_return_pct += TP2_SIZE * ((tp2 / entry_price) - 1.0)
                remaining -= TP2_SIZE
                tp2_done = True
                trailing_active = True
                best_price = max(best_price, high)
                trail_stop = best_price * (1 - trail_pct)

            if (not tp3_done) and high >= tp3:
                realized_return_pct += TP3_SIZE * ((tp3 / entry_price) - 1.0)
                remaining -= TP3_SIZE
                tp3_done = True

            if (not tp4_done) and high >= tp4:
                realized_return_pct += TP4_SIZE * ((tp4 / entry_price) - 1.0)
                remaining -= TP4_SIZE
                tp4_done = True
                exit_reason = "tp4"
                exit_time = row["timestamp"]
                exit_price = tp4
                break

        else:
            if trailing_active:
                best_price = min(best_price, low)
                new_trail = best_price * (1 + trail_pct)
                trail_stop = new_trail if np.isnan(trail_stop) else min(trail_stop, new_trail)

            active_stop = stop_price

            if break_even_armed:
                active_stop = min(active_stop, break_even_price)

            if trailing_active and not np.isnan(trail_stop):
                active_stop = min(active_stop, trail_stop)

            if high >= active_stop:
                realized_return_pct += remaining * (1.0 - (active_stop / entry_price))
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
                remaining -= TP1_SIZE
                tp1_done = True
                if ENABLE_BREAK_EVEN_AFTER_TP1:
                    break_even_armed = True

            if (not tp2_done) and low <= tp2:
                realized_return_pct += TP2_SIZE * (1.0 - (tp2 / entry_price))
                remaining -= TP2_SIZE
                tp2_done = True
                trailing_active = True
                best_price = min(best_price, low)
                trail_stop = best_price * (1 + trail_pct)

            if (not tp3_done) and low <= tp3:
                realized_return_pct += TP3_SIZE * (1.0 - (tp3 / entry_price))
                remaining -= TP3_SIZE
                tp3_done = True

            if (not tp4_done) and low <= tp4:
                realized_return_pct += TP4_SIZE * (1.0 - (tp4 / entry_price))
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
            remaining = 0.0
            exit_reason = "end_of_data"
            exit_time = row["timestamp"]
            exit_price = close

    gross_pnl_cash = position_notional * realized_return_pct
    # ── Fix #5: Symbol-specific slippage ──
    round_trip_cost = get_round_trip_cost(symbol)
    cost_cash = position_notional * round_trip_cost
    net_pnl_cash = gross_pnl_cash - cost_cash
    equity_after = equity_before + net_pnl_cash

    return {
        "entry_time": entry_time,
        "entry_hour": pd.Timestamp(entry_time).hour,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "direction": direction,
        "dir_label": direction_label(direction),
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
        "gross_return_pct_on_notional": realized_return_pct,
        "gross_pnl_cash": gross_pnl_cash,
        "cost_cash": cost_cash,
        "net_pnl_cash": net_pnl_cash,
        "net_r_multiple": net_pnl_cash / risk_amount if risk_amount > 0 else np.nan,
        "win": int(net_pnl_cash > 0),
        "exit_reason": exit_reason,
    }


def run_backtest(df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
    trades = []
    next_allowed_idx = 0
    i = 0
    equity = INITIAL_EQUITY

    while i < len(df):
        if i < next_allowed_idx:
            i += 1
            continue

        row = df.iloc[i]
        if (row["setup_type"] == "none") or (row["direction"] == 0) or (not bool(row["take_trade"])):
            i += 1
            continue

        trade = simulate_trade(
            df=df,
            entry_idx=i,
            direction=int(row["direction"]),
            strategy=str(row["setup_type"]),
            confidence=float(row["confidence_raw"]),
            confidence_mode=str(row["confidence_mode"]),
            risk_fraction=float(row["risk_fraction"]),
            equity_before=float(equity),
            stage=int(row.get("stage", 0)),
            cfx_score=float(row.get("cfx_score", 0)),
            symbol=symbol,
        )
        trades.append(trade)
        equity = float(trade["equity_after"])

        exit_loc = df.index[df["timestamp"] == trade["exit_time"]]
        if len(exit_loc) > 0:
            next_allowed_idx = int(exit_loc[0]) + COOLDOWN_BARS_5M
        else:
            next_allowed_idx = i + COOLDOWN_BARS_5M

        i = next_allowed_idx

    return pd.DataFrame(trades)


# =========================================================
# REPORT
# =========================================================

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) > 0 else np.nan


def print_core_report(trades: pd.DataFrame, setups: pd.DataFrame) -> None:
    print("=" * 80)
    print("BASELINE_MODE:", BASELINE_MODE, "(MILD + PREMIUM + HIGH + ELITE | MID DISABLED)")
    print("ADX FILTER:", "ON" if ENABLE_ADX_FILTER else "OFF", f"(trending >{ADX_TRENDING_THRESHOLD}, choppy <{ADX_CHOPPY_THRESHOLD})")
    print("SHORT VALIDATION: STANDBY (no MID shorts allowed)")
    print("=" * 80)

    total_contexts = int((setups["setup_type"] != "none").sum())
    total_triggered = int(((setups["setup_type"] != "none") & (setups["trigger_ok"])).sum())
    total_tradeable_all = int(((setups["setup_type"] != "none") & (setups["trigger_ok"]) & (setups["confidence_mode"] != "NO_TRADE")).sum())

    mid_short_all = int(((setups["confidence_mode"] == "MID") & (setups["direction"] == -1) & (setups["setup_type"] != "none") & setups["trigger_ok"]).sum())
    mid_short_valid = int(((setups["confidence_mode"] == "MID") & (setups["direction"] == -1) & (setups["setup_type"] != "none") & setups["trigger_ok"] & setups.get("short_valid", True)).sum())

    total_tradeable_filtered = int(((setups["setup_type"] != "none") & (setups["trigger_ok"]) & (setups["take_trade"])).sum())

    print("Total Setups:", total_contexts)
    print("Structure Trigger Passed:", total_triggered)
    print("Tradeable After Confidence (all modes):", total_tradeable_all)
    print("MID SHORT candidates (before validation):", mid_short_all)
    print("MID SHORT passed validation:", mid_short_valid)
    print("MID SHORT filtered out:", mid_short_all - mid_short_valid)
    print("Tradeable After All Filters:", total_tradeable_filtered)
    print("Trades Taken:", len(trades))

    if len(trades) == 0:
        print("\nNo trades.")
        return

    wins = trades.loc[trades["net_pnl_cash"] > 0, "net_pnl_cash"]
    losses = trades.loc[trades["net_pnl_cash"] < 0, "net_pnl_cash"]

    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    avg_win = wins.mean() if len(wins) > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan
    expectancy = trades["net_pnl_cash"].mean()
    total_return = trades["equity_after"].iloc[-1] / INITIAL_EQUITY - 1.0
    max_drawdown = compute_max_drawdown(trades["equity_after"])

    print("\nCORE METRICS")
    print("Trades:", len(trades))
    print("Winrate:", float(trades["win"].mean()))
    print("Expectancy ($):", float(expectancy))
    print("Profit Factor:", float(profit_factor) if pd.notna(profit_factor) else np.nan)
    print("Total Return (%):", float(total_return))
    print("Avg Win ($):", float(avg_win) if pd.notna(avg_win) else np.nan)
    print("Avg Loss ($):", float(avg_loss) if pd.notna(avg_loss) else np.nan)
    print("Max Drawdown (%):", float(max_drawdown))

    print("\nBY MODE")
    print(
        trades.groupby("confidence_mode", observed=True)["net_pnl_cash"]
        .agg(["count", "mean", "median", "sum"])
        .to_string()
    )

    print("\nBY DIRECTION")
    print(
        trades.groupby("dir_label", observed=True)["net_pnl_cash"]
        .agg(["count", "mean", "median", "sum"])
        .to_string()
    )

    print("\nBY STRATEGY")
    print(
        trades.groupby("strategy", observed=True)["net_pnl_cash"]
        .agg(["count", "mean", "median", "sum"])
        .to_string()
    )

    # Stage breakdown (RSI_SCALP only)
    scalp_trades = trades[trades["strategy"] == "RSI_SCALP"]
    if len(scalp_trades) > 0 and "stage" in scalp_trades.columns:
        print("\nBY STAGE (RSI_SCALP only)")
        print(
            scalp_trades.groupby("stage", observed=True)["net_pnl_cash"]
            .agg(["count", "mean", "median", "sum"])
            .to_string()
        )

    # CFX breakdown (RSI_TREND only)
    trend_trades = trades[trades["strategy"] == "RSI_TREND"]
    if len(trend_trades) > 0 and "cfx_score" in trend_trades.columns:
        print("\nBY CFX (RSI_TREND only)")
        print(
            trend_trades.groupby("cfx_score", observed=True)["net_pnl_cash"]
            .agg(["count", "mean", "median", "sum"])
            .to_string()
        )

    print("\nBY EXIT REASON")
    print(
        trades.groupby("exit_reason", observed=True)["net_pnl_cash"]
        .agg(["count", "mean", "median", "sum"])
        .to_string()
    )

    # Short-specific breakdown
    short_trades = trades[trades["direction"] == -1]
    long_trades = trades[trades["direction"] == 1]

    print("\n" + "=" * 40)
    print("SHORT vs LONG COMPARISON")
    print("=" * 40)

    for label, subset in [("LONG", long_trades), ("SHORT", short_trades)]:
        if len(subset) == 0:
            print(f"\n{label}: No trades")
            continue
        w = subset.loc[subset["net_pnl_cash"] > 0, "net_pnl_cash"]
        l = subset.loc[subset["net_pnl_cash"] < 0, "net_pnl_cash"]
        gp = w.sum()
        gl = -l.sum()
        pf = gp / gl if gl > 0 else np.nan
        print(f"\n{label} ({len(subset)} trades)")
        print(f"  Winrate: {float(subset['win'].mean()):.3f}")
        print(f"  PF: {float(pf) if pd.notna(pf) else np.nan:.3f}")
        print(f"  Expectancy: ${float(subset['net_pnl_cash'].mean()):.2f}")
        print(f"  Total PnL: ${float(subset['net_pnl_cash'].sum()):.2f}")
        if label == "SHORT":
            mid_short = subset[subset["confidence_mode"] == "MID"]
            high_short = subset[subset["confidence_mode"] == "HIGH"]
            if len(mid_short) > 0:
                mw = mid_short.loc[mid_short["net_pnl_cash"] > 0, "net_pnl_cash"]
                ml = -mid_short.loc[mid_short["net_pnl_cash"] < 0, "net_pnl_cash"].sum()
                mpf = mw.sum() / ml if ml > 0 else np.nan
                print(f"    MID SHORT ({len(mid_short)}): WR={float(mid_short['win'].mean()):.3f} PF={float(mpf) if pd.notna(mpf) else np.nan:.3f}")
            if len(high_short) > 0:
                hw = high_short.loc[high_short["net_pnl_cash"] > 0, "net_pnl_cash"]
                hl = -high_short.loc[high_short["net_pnl_cash"] < 0, "net_pnl_cash"].sum()
                hpf = hw.sum() / hl if hl > 0 else np.nan
                print(f"    HIGH SHORT ({len(high_short)}): WR={float(high_short['win'].mean()):.3f} PF={float(hpf) if pd.notna(hpf) else np.nan:.3f}")


# =========================================================
# BUILD DATASET
# =========================================================

def build_master_dataset(path: str = None) -> pd.DataFrame:
    ticks = load_data(path or INPUT_PATH)

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

    # ── Fix #1: ADX regime filter ──
    # Compute ADX on 4H bars (regime is a higher-TF concept)
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
    df = build_confidence_engine(df)
    df = short_validation_layer(df)
    df = add_trade_decision(df)

    return df


# =========================================================
# MAIN
# =========================================================

def main():
    import os

    # ── Walk-Forward Validation ──
    print("=" * 80)
    print("MULTI-ASSET WALK-FORWARD VALIDATION")
    print("=" * 80)

    train_cutoff = pd.Timestamp("2025-12-01")  # Train: Jun-Nov 2025, Test: Dec 2025-Apr 2026
    all_trades_train = []
    all_trades_test = []
    all_setups = []

    for symbol in SYMBOLS:
        data_path = f"data/features/{symbol.lower()}_1m.csv"
        if not os.path.exists(data_path):
            print(f"\n[SKIP] {symbol}: {data_path} not found. Run fetch_btc_data.py first.")
            continue

        print(f"\n{'─' * 60}")
        print(f"Processing {symbol}")
        print(f"{'─' * 60}")

        df = build_master_dataset(data_path)
        df["symbol"] = symbol
        all_setups.append(df)

        df_train = df[df["timestamp"] < train_cutoff].copy()
        df_test = df[df["timestamp"] >= train_cutoff].copy()

        if len(df_train) > 100:
            trades_train = run_backtest(df_train, symbol=symbol)
            trades_train["symbol"] = symbol
            trades_train["period"] = "TRAIN"
            all_trades_train.append(trades_train)
            print(f"  Train: {len(trades_train)} trades, WR {trades_train['win'].mean():.1%}" if len(trades_train) > 0 else f"  Train: 0 trades")

        if len(df_test) > 100:
            trades_test = run_backtest(df_test, symbol=symbol)
            trades_test["symbol"] = symbol
            trades_test["period"] = "TEST"
            all_trades_test.append(trades_test)
            print(f"  Test:  {len(trades_test)} trades, WR {trades_test['win'].mean():.1%}" if len(trades_test) > 0 else f"  Test:  0 trades")

    # Combine
    combined_train = pd.concat(all_trades_train, ignore_index=True) if all_trades_train else pd.DataFrame()
    combined_test = pd.concat(all_trades_test, ignore_index=True) if all_trades_test else pd.DataFrame()
    combined_all = pd.concat([combined_train, combined_test], ignore_index=True) if not combined_train.empty or not combined_test.empty else pd.DataFrame()

    # Reports
    for label, trades in [("FULL PERIOD", combined_all), ("TRAIN (Jan-Feb)", combined_train), ("TEST (Mar-Apr)", combined_test)]:
        print(f"\n{'=' * 80}")
        print(f"{label}")
        print(f"{'=' * 80}")
        if len(trades) == 0:
            print("No trades.")
            continue

        wins = trades.loc[trades["net_pnl_cash"] > 0, "net_pnl_cash"]
        losses = trades.loc[trades["net_pnl_cash"] < 0, "net_pnl_cash"]
        pf = wins.sum() / (-losses.sum()) if losses.sum() < 0 else float("inf")

        print(f"Trades: {len(trades)}")
        print(f"Winrate: {trades['win'].mean():.1%}")
        print(f"Profit Factor: {pf:.2f}")
        print(f"Total PnL: ${trades['net_pnl_cash'].sum():+.2f}")
        print(f"Expectancy: ${trades['net_pnl_cash'].mean():+.2f}")

        print(f"\nBY SYMBOL:")
        print(trades.groupby("symbol")["net_pnl_cash"].agg(["count", "mean", "sum"]).to_string())

        print(f"\nBY STRATEGY:")
        print(trades.groupby("strategy")["net_pnl_cash"].agg(["count", "mean", "sum"]).to_string())

        print(f"\nBY MODE:")
        print(trades.groupby("confidence_mode")["net_pnl_cash"].agg(["count", "mean", "sum"]).to_string())

        print(f"\nBY EXIT:")
        print(trades.groupby("exit_reason")["net_pnl_cash"].agg(["count", "mean", "sum"]).to_string())

    # Walk-forward comparison
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD COMPARISON")
    print(f"{'=' * 80}")
    for label, t in [("FULL", combined_all), ("TRAIN", combined_train), ("TEST", combined_test)]:
        if len(t) == 0:
            print(f"{label}: No trades")
            continue
        wr = t["win"].mean()
        pw = t.loc[t["net_pnl_cash"] > 0, "net_pnl_cash"].sum()
        pl = -t.loc[t["net_pnl_cash"] < 0, "net_pnl_cash"].sum()
        pf = pw / pl if pl > 0 else float("inf")
        pnl = t["net_pnl_cash"].sum()
        print(f"{label}: {len(t):3d} trades | WR {wr:.1%} | PF {pf:.2f} | PnL ${pnl:+.0f}")

    # Save
    if len(combined_all) > 0:
        combined_all.to_csv("data/features/baseline_trades.csv", index=False)
        print(f"\n[SAVED] data/features/baseline_trades.csv")


if __name__ == "__main__":
    main()
