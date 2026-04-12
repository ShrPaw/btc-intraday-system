import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = "data/features/research_dataset.csv"

RSI_PERIOD = 14
EMA_PERIOD = 20

INITIAL_EQUITY = 10000.0
ROUND_TRIP_COST = 0.0008

# Risk by mode
RISK_LOW = 0.0000
RISK_MID = 0.0300
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

COOLDOWN_BARS_5M = 6

# Confidence bands
NO_TRADE_THRESHOLD = 0.72
MID_THRESHOLD = 0.80
HIGH_THRESHOLD = 0.90

USE_HOUR_FILTER = False
ALLOWED_HOURS = {5, 6, 8, 9, 10, 12, 18}

# =========================================================
# ACTIVE BASELINE
# =========================================================
# HYBRID with short validation layer:
# HIGH -> LONG + SHORT (unchanged)
# MID -> LONG (unchanged) + SHORT (gated by validation layer)
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
    # MID mode disabled — empirical analysis shows it's a -$1,120 drag
    # Only HIGH mode trades are profitable
    ALLOWED_MODES = {"HIGH"}
    ALLOW_LONG = True
    ALLOW_SHORT = True

else:
    raise ValueError(f"Invalid BASELINE_MODE: {BASELINE_MODE}")


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
        0.30 * trend_alignment +
        0.20 * momentum_score +
        0.35 * trend_structure +
        0.15 * rv_score
    )

    out.loc[scalp_mask, "confidence_raw"] = scalp_conf[scalp_mask]
    out.loc[trend_mask, "confidence_raw"] = trend_conf[trend_mask]
    out["confidence_raw"] = out["confidence_raw"].clip(0, 1)

    premium_fresh = fresh_base >= 0.80
    premium_momentum = momentum_score >= 0.70
    premium_structure = np.where(out["direction"] == 1, structure_long, structure_short) >= 0.85
    premium_vol = rv_score >= 0.55
    premium_dist = ema_dist_quality >= 0.55

    elite_gate = premium_fresh & premium_momentum & premium_structure & premium_vol & premium_dist

    out["confidence_mode"] = "NO_TRADE"
    out["risk_fraction"] = 0.0

    out.loc[
        (out["confidence_raw"] >= NO_TRADE_THRESHOLD) &
        (out["confidence_raw"] < MID_THRESHOLD),
        "confidence_mode"
    ] = "MID"

    out.loc[
        out["confidence_raw"] >= MID_THRESHOLD,
        "confidence_mode"
    ] = "HIGH"

    out.loc[
        (out["confidence_raw"] >= HIGH_THRESHOLD) & elite_gate,
        "confidence_mode"
    ] = "ELITE"

    out.loc[out["confidence_mode"] == "MID", "risk_fraction"] = RISK_MID
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
        # HIGH only — MID mode disabled (empirical drag: -$1,120 over 3.5 months)
        mode_direction_ok = (
            (out["confidence_mode"] == "HIGH") & out["direction"].isin({1, -1})
        )
    else:
        mode_direction_ok = (
            out["confidence_mode"].isin(ALLOWED_MODES) &
            (
                ((out["direction"] == 1) & ALLOW_LONG) |
                ((out["direction"] == -1) & ALLOW_SHORT)
            )
        )

    out["take_trade"] = (
        (out["setup_type"] != "none") &
        out["trigger_ok"] &
        (out["confidence_mode"] != "NO_TRADE") &
        mode_direction_ok &
        hour_ok
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
    if confidence_mode == "MID":
        return TRAIL_PCT_MID
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
) -> dict:
    entry_row = df.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    entry_time = entry_row["timestamp"]

    stop_price, stop_distance_pct = compute_stop_price(df, entry_idx, direction, strategy)

    risk_amount = equity_before * risk_fraction
    position_notional = risk_amount / stop_distance_pct if stop_distance_pct > 0 else 0.0

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
    cost_cash = position_notional * ROUND_TRIP_COST
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


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
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
    print("BASELINE_MODE:", BASELINE_MODE, "(HIGH ONLY — MID DISABLED)")
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

def build_master_dataset() -> pd.DataFrame:
    ticks = load_data(INPUT_PATH)

    bars_5m = resample_bars(ticks, "5min")
    bars_15m = resample_bars(ticks, "15min")
    bars_4h = resample_bars(ticks, "4h")
    bars_6h = resample_bars(ticks, "6h")
    bars_12h = resample_bars(ticks, "12h")

    bars_5m = add_rsi_features(bars_5m, RSI_PERIOD)
    bars_5m = add_ema_bos_features_5m(bars_5m)
    bars_5m = add_extra_features_5m(bars_5m)

    bars_15m = add_rsi_features(bars_15m, RSI_PERIOD)
    bars_4h = add_rsi_features(bars_4h, RSI_PERIOD)
    bars_6h = add_rsi_features(bars_6h, RSI_PERIOD)
    bars_12h = add_rsi_features(bars_12h, RSI_PERIOD)

    df = bars_5m.copy()

    df = merge_asof_feature(df, bars_15m, "m15", ["rsi", "rsi_slope_1"])
    df = merge_asof_feature(df, bars_4h, "h4", ["rsi", "rsi_slope_1", "rsi_slope_2", "fresh_long", "fresh_short", "bars_since_cross"])
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
    df = build_master_dataset()
    trades = run_backtest(df)

    print_core_report(trades, df)

    setups_path = "data/features/baseline_setups.csv"
    trades_path = "data/features/baseline_trades.csv"

    df.to_csv(setups_path, index=False)
    trades.to_csv(trades_path, index=False)

    print(f"\n[SAVED] {setups_path}")
    print(f"[SAVED] {trades_path}")


if __name__ == "__main__":
    main()
