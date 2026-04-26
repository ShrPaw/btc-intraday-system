"""
FUNDING RATE EVENT STUDY
========================
Treat funding extremes as discrete events. Study the PRICE PATH around them.

NOT testing returns. Testing BEHAVIOR.
- Does price first go against the crowd?
- How far? How fast?
- Does it revert fully or partially?

For each event:
  1. MAE (max adverse excursion) — how far does price go against the crowd?
  2. MFE (max favorable excursion) — how far does price eventually move with the crowd?
  3. Time to mean reversion — bars until price moves back toward event level
  4. Time to continuation — bars until price moves in the "crowd right" direction

Output:
  - % of events where reversal happens first
  - median reversal size
  - median time to reversal
  - % continuation vs reversal
  - All split by regime (neutral / trending via ADX)
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "features")

# =========================================================
# CONFIG
# =========================================================

# Funding extreme thresholds (percentiles of rolling distribution)
FUNDING_HIGH_PCTL = 90   # "too long" — crowd is paying to be long
FUNDING_LOW_PCTL = 10    # "too short" — crowd is paying to be short

# How far forward to track price after each event (in 1m bars)
TRACK_HORIZON_BARS = 720  # 12 hours

# ADX regime threshold
ADX_TRENDING_THRESHOLD = 25
ADX_PERIOD = 14

# Rolling window for funding percentile (shift 1 to avoid lookahead)
FUNDING_ROLLING_WINDOW = 30  # ~10 days of 8h funding

# Mean reversion definition: price returns to within X% of event price
REVERSION_THRESHOLD_PCT = 0.001  # 0.1% — "mostly reverted"

# =========================================================
# DATA LOADING
# =========================================================

def load_1m_data():
    """Load 1m BTC data, resample to 4H for ADX, keep 1m for tracking."""
    path = os.path.join(DATA_DIR, "btcusdt_1m.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_funding_data():
    """Load 8h funding rates."""
    path = os.path.join(DATA_DIR, "btcusdt_funding.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# =========================================================
# INDICATORS
# =========================================================

def compute_adx(df_1m):
    """Compute ADX on 4H bars, return as time series."""
    # Resample to 4H
    df_4h = df_1m.set_index("timestamp").resample("4h").agg({
        "price": ["first", "max", "min", "last"],
        "volume": "sum"
    }).dropna()
    df_4h.columns = ["open", "high", "low", "close", "volume"]
    df_4h = df_4h.reset_index()

    # True Range
    df_4h["prev_close"] = df_4h["close"].shift(1)
    df_4h["tr"] = np.maximum(
        df_4h["high"] - df_4h["low"],
        np.maximum(
            abs(df_4h["high"] - df_4h["prev_close"]),
            abs(df_4h["low"] - df_4h["prev_close"])
        )
    )

    # Directional movement
    df_4h["up_move"] = df_4h["high"] - df_4h["high"].shift(1)
    df_4h["down_move"] = df_4h["low"].shift(1) - df_4h["low"]
    df_4h["plus_dm"] = np.where(
        (df_4h["up_move"] > df_4h["down_move"]) & (df_4h["up_move"] > 0),
        df_4h["up_move"], 0
    )
    df_4h["minus_dm"] = np.where(
        (df_4h["down_move"] > df_4h["up_move"]) & (df_4h["down_move"] > 0),
        df_4h["down_move"], 0
    )

    # Smoothed averages (Wilder's)
    alpha = 1 / ADX_PERIOD
    df_4h["atr"] = df_4h["tr"].ewm(alpha=alpha, min_periods=ADX_PERIOD).mean()
    df_4h["plus_di"] = 100 * df_4h["plus_dm"].ewm(alpha=alpha, min_periods=ADX_PERIOD).mean() / df_4h["atr"]
    df_4h["minus_di"] = 100 * df_4h["minus_dm"].ewm(alpha=alpha, min_periods=ADX_PERIOD).mean() / df_4h["atr"]

    dx = 100 * abs(df_4h["plus_di"] - df_4h["minus_di"]) / (df_4h["plus_di"] + df_4h["minus_di"])
    df_4h["adx"] = dx.ewm(alpha=alpha, min_periods=ADX_PERIOD).mean()

    return df_4h[["timestamp", "adx", "plus_di", "minus_di"]].dropna()


# =========================================================
# EVENT DETECTION
# =========================================================

def detect_events(funding_df, adx_df):
    """
    Detect funding extreme events.
    A high funding event: funding > rolling p90 → crowd is paying to be LONG → expect downward pressure
    A low funding event:  funding < rolling p10 → crowd is paying to be SHORT → expect upward pressure

    Uses rolling percentile with shift(1) to avoid lookahead.
    """
    # Rolling percentile thresholds (past-only)
    funding_df = funding_df.copy()
    funding_df["p90"] = funding_df["fundingRate"].rolling(
        FUNDING_ROLLING_WINDOW, min_periods=10
    ).quantile(0.90).shift(1)
    funding_df["p10"] = funding_df["fundingRate"].rolling(
        FUNDING_ROLLING_WINDOW, min_periods=10
    ).quantile(0.10).shift(1)

    # Detect extremes
    funding_df["is_high"] = funding_df["fundingRate"] > funding_df["p90"]
    funding_df["is_low"] = funding_df["fundingRate"] < funding_df["p10"]

    # Merge ADX at each funding timestamp
    funding_df = pd.merge_asof(
        funding_df.sort_values("timestamp"),
        adx_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    funding_df["regime"] = np.where(
        funding_df["adx"] >= ADX_TRENDING_THRESHOLD, "trending", "neutral"
    )

    events = []

    for _, row in funding_df.iterrows():
        if row["is_high"]:
            events.append({
                "timestamp": row["timestamp"],
                "type": "HIGH_EXTREME",
                "funding_rate": row["fundingRate"],
                "funding_pctile": "p90+",
                "crowd_direction": "LONG",
                "expected_price_direction": "DOWN",  # crowd is wrong → expect down
                "adx": row["adx"],
                "regime": row["regime"],
            })
        if row["is_low"]:
            events.append({
                "timestamp": row["timestamp"],
                "type": "LOW_EXTREME",
                "funding_rate": row["fundingRate"],
                "funding_pctile": "p10-",
                "crowd_direction": "SHORT",
                "expected_price_direction": "UP",  # crowd is wrong → expect up
                "adx": row["adx"],
                "regime": row["regime"],
            })

    return pd.DataFrame(events)


# =========================================================
# PATH TRACKING
# =========================================================

def track_event_path(event, price_1m):
    """
    For a single event, track the price path for TRACK_HORIZON_BARS 1m bars.

    Returns:
      - mfe: max favorable excursion (price moves in expected direction)
      - mae: max adverse excursion (price moves against expected direction)
      - time_to_first_reversal: bars until price first moves against crowd
                                  (i.e., adverse move > REVERSION_THRESHOLD)
      - time_to_first_continuation: bars until price first moves with crowd
                                      (i.e., favorable move > REVERSION_THRESHOLD)
      - reversal_happened_first: did adverse move happen before favorable?
      - price_at_horizon: price after TRACK_HORIZON_BARS
      - full_path: array of price changes (% from event price)
    """
    event_time = event["timestamp"]
    expected_dir = event["expected_price_direction"]  # "UP" or "DOWN"

    # Find the nearest 1m bar at or after the event
    mask = price_1m["timestamp"] >= event_time
    if mask.sum() == 0:
        return None

    start_idx = price_1m.index[mask][0]
    end_idx = start_idx + TRACK_HORIZON_BARS

    if end_idx >= len(price_1m):
        return None

    path_prices = price_1m.iloc[start_idx:end_idx + 1]["price"].values
    event_price = path_prices[0]

    if event_price == 0:
        return None

    # Price changes in % from event price
    pct_changes = (path_prices - event_price) / event_price  # positive = price went up

    # Determine favorable direction
    if expected_dir == "UP":
        favorable_sign = 1   # price going up is favorable
    else:
        favorable_sign = -1  # price going down is favorable

    favorable_pcts = pct_changes * favorable_sign
    adverse_pcts = -pct_changes * favorable_sign

    # MFE = max favorable excursion
    mfe = np.max(favorable_pcts)

    # MAE = max adverse excursion
    mae = np.max(adverse_pcts)

    # Time to first reversal (adverse move > threshold)
    reversal_bars = None
    for i in range(1, len(pct_changes)):
        if adverse_pcts[i] >= REVERSION_THRESHOLD_PCT:
            reversal_bars = i
            break

    # Time to first continuation (favorable move > threshold)
    continuation_bars = None
    for i in range(1, len(pct_changes)):
        if favorable_pcts[i] >= REVERSION_THRESHOLD_PCT:
            continuation_bars = i
            break

    # Which happened first?
    if reversal_bars is not None and continuation_bars is not None:
        reversal_first = reversal_bars < continuation_bars
    elif reversal_bars is not None:
        reversal_first = True
    elif continuation_bars is not None:
        reversal_first = False
    else:
        reversal_first = None  # neither happened

    # Partial/full reversion tracking
    # "Full reversion" = price returns to within 0.1% of event price from adverse side
    reversion_pct = None
    if reversal_bars is not None:
        # Max adverse move during the reversal window
        max_adverse_in_window = np.max(adverse_pcts[:reversal_bars + 1]) if reversal_bars > 0 else 0
        # How much of that adverse move was recovered?
        # After reversal peak, track the recovery
        adverse_peak_idx = np.argmax(adverse_pcts[:reversal_bars + 1]) if reversal_bars > 0 else 0
        if adverse_peak_idx > 0:
            peak_adverse = adverse_pcts[adverse_peak_idx]
            # After the peak, what's the minimum adverse (i.e., max recovery)?
            if adverse_peak_idx < len(adverse_pcts) - 1:
                remaining = adverse_pcts[adverse_peak_idx + 1:]
                min_remaining = np.min(remaining)
                reversion_pct = max(0, (peak_adverse - min_remaining) / peak_adverse) if peak_adverse > 0 else 0
            else:
                reversion_pct = 0
        else:
            reversion_pct = 0

    return {
        "mfe_pct": mfe,
        "mae_pct": mae,
        "time_to_reversal_bars": reversal_bars,
        "time_to_continuation_bars": continuation_bars,
        "reversal_happened_first": reversal_first,
        "reversion_pct": reversion_pct,
        "price_at_horizon_pct": pct_changes[-1],
        "path": pct_changes,
    }


# =========================================================
# ANALYSIS
# =========================================================

def analyze_results(results_df):
    """Analyze event study results by type and regime."""

    print("\n" + "=" * 80)
    print("FUNDING RATE EVENT STUDY — BEHAVIORAL ANALYSIS")
    print("=" * 80)

    for group_name, group_df in results_df.groupby(["type", "regime"]):
        event_type, regime = group_name
        n = len(group_df)
        if n < 5:
            print(f"\n--- {event_type} / {regime} (n={n}, TOO FEW) ---")
            continue

        print(f"\n{'='*60}")
        print(f"  {event_type}  |  Regime: {regime}  |  n = {n}")
        print(f"  (Crowd is {group_df.iloc[0]['crowd_direction']}, expect {group_df.iloc[0]['expected_price_direction']})")
        print(f"{'='*60}")

        # Reversal vs continuation
        reversal_first = group_df["reversal_happened_first"].dropna()
        n_valid = len(reversal_first)
        n_reversal = (reversal_first == True).sum()
        n_continuation = (reversal_first == False).sum()

        print(f"\n  PATH BEHAVIOR:")
        print(f"    Reversal happened first:   {n_reversal}/{n_valid} = {n_reversal/n_valid*100:.1f}%")
        print(f"    Continuation happened first: {n_continuation}/{n_valid} = {n_continuation/n_valid*100:.1f}%")

        # MAE — how far does price go against the crowd?
        mae = group_df["mae_pct"].dropna()
        print(f"\n  ADVERSE EXCURSION (against expected direction):")
        print(f"    Median MAE:  {mae.median()*100:.3f}%")
        print(f"    Mean MAE:    {mae.mean()*100:.3f}%")
        print(f"    p75 MAE:     {mae.quantile(0.75)*100:.3f}%")
        print(f"    p90 MAE:     {mae.quantile(0.90)*100:.3f}%")
        print(f"    Max MAE:     {mae.max()*100:.3f}%")

        # MFE — how far does price eventually move with the crowd?
        mfe = group_df["mfe_pct"].dropna()
        print(f"\n  FAVORABLE EXCURSION (with expected direction):")
        print(f"    Median MFE:  {mfe.median()*100:.3f}%")
        print(f"    Mean MFE:    {mfe.mean()*100:.3f}%")
        print(f"    p75 MFE:     {mfe.quantile(0.75)*100:.3f}%")
        print(f"    p90 MFE:     {mfe.quantile(0.90)*100:.3f}%")

        # Time to reversal
        rev_times = group_df["time_to_reversal_bars"].dropna()
        if len(rev_times) > 0:
            print(f"\n  TIME TO REVERSAL (bars = minutes):")
            print(f"    Median:  {rev_times.median():.0f} bars ({rev_times.median()/60:.1f} hours)")
            print(f"    Mean:    {rev_times.mean():.0f} bars ({rev_times.mean()/60:.1f} hours)")
            print(f"    p75:     {rev_times.quantile(0.75):.0f} bars")
            print(f"    p90:     {rev_times.quantile(0.90):.0f} bars")

        # Time to continuation
        cont_times = group_df["time_to_continuation_bars"].dropna()
        if len(cont_times) > 0:
            print(f"\n  TIME TO CONTINUATION (bars = minutes):")
            print(f"    Median:  {cont_times.median():.0f} bars ({cont_times.median()/60:.1f} hours)")
            print(f"    Mean:    {cont_times.mean():.0f} bars ({cont_times.mean()/60:.1f} hours)")
            print(f"    p75:     {cont_times.quantile(0.75):.0f} bars")
            print(f"    p90:     {cont_times.quantile(0.90):.0f} bars")

        # Reversion extent
        rev_pct = group_df["reversion_pct"].dropna()
        if len(rev_pct) > 0:
            print(f"\n  REVERSION EXTENT (% of adverse move recovered):")
            print(f"    Median:  {rev_pct.median()*100:.1f}%")
            print(f"    Mean:    {rev_pct.mean()*100:.1f}%")

        # Final outcome at horizon
        horizon = group_df["price_at_horizon_pct"].dropna()
        print(f"\n  PRICE AT 12H HORIZON:")
        print(f"    Median:  {horizon.median()*100:.3f}%")
        print(f"    Mean:    {horizon.mean()*100:.3f}%")
        pct_positive = (horizon > 0).sum() / len(horizon) * 100
        print(f"    % positive: {pct_positive:.1f}%")

    # Cross-regime comparison
    print(f"\n\n{'='*80}")
    print("  CROSS-REGIME COMPARISON")
    print(f"{'='*80}")

    for event_type in results_df["type"].unique():
        subset = results_df[results_df["type"] == event_type]
        trending = subset[subset["regime"] == "trending"]
        neutral = subset[subset["regime"] == "neutral"]

        if len(trending) < 3 or len(neutral) < 3:
            continue

        print(f"\n  {event_type}:")
        print(f"  {'Metric':<30} {'Neutral':>12} {'Trending':>12} {'Delta':>12}")
        print(f"  {'-'*66}")

        for metric, label in [
            ("mae_pct", "Median MAE %"),
            ("mfe_pct", "Median MFE %"),
            ("time_to_reversal_bars", "Median Reversal bars"),
            ("time_to_continuation_bars", "Median Continuation bars"),
            ("price_at_horizon_pct", "Median Horizon %"),
        ]:
            n_val = neutral[metric].median()
            t_val = trending[metric].median()
            delta = t_val - n_val
            print(f"  {label:<30} {n_val*100 if 'pct' in metric else n_val:>12.3f} {t_val*100 if 'pct' in metric else t_val:>12.3f} {delta*100 if 'pct' in metric else delta:>12.3f}")

        # Reversal rate
        n_rev_n = neutral["reversal_happened_first"].dropna()
        n_rev_t = trending["reversal_happened_first"].dropna()
        rev_n = (n_rev_n == True).sum() / len(n_rev_n) * 100 if len(n_rev_n) > 0 else 0
        rev_t = (n_rev_t == True).sum() / len(n_rev_t) * 100 if len(n_rev_t) > 0 else 0
        print(f"  {'Reversal first %':<30} {rev_n:>11.1f}% {rev_t:>11.1f}% {rev_t-rev_n:>+11.1f}%")


# =========================================================
# MAIN
# =========================================================

def main():
    print("Loading data...")
    price_1m = load_1m_data()
    funding = load_funding_data()
    print(f"  1m candles: {len(price_1m):,} ({price_1m['timestamp'].min()} → {price_1m['timestamp'].max()})")
    print(f"  Funding:    {len(funding):,} ({funding['timestamp'].min()} → {funding['timestamp'].max()})")

    print("\nComputing ADX on 4H bars...")
    adx_df = compute_adx(price_1m)
    print(f"  ADX series: {len(adx_df):,} bars")

    print("\nDetecting funding extreme events...")
    events = detect_events(funding, adx_df)
    print(f"  Total events: {len(events)}")
    print(f"    HIGH_EXTREME (crowd long, expect down): {(events['type'] == 'HIGH_EXTREME').sum()}")
    print(f"    LOW_EXTREME  (crowd short, expect up):  {(events['type'] == 'LOW_EXTREME').sum()}")
    print(f"    By regime: {events['regime'].value_counts().to_dict()}")

    print(f"\nTracking price paths ({TRACK_HORIZON_BARS} bars = {TRACK_HORIZON_BARS/60:.1f} hours per event)...")
    results = []
    for _, event in events.iterrows():
        result = track_event_path(event, price_1m)
        if result is None:
            continue
        results.append({**event.to_dict(), **{k: v for k, v in result.items() if k != "path"}})

    results_df = pd.DataFrame(results)
    print(f"  Tracked events: {len(results_df)}")

    # Save detailed results
    out_path = os.path.join(DATA_DIR, "funding_event_study_results.csv")
    results_df.drop(columns=["path"], errors="ignore").to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")

    analyze_results(results_df)

    # Print all events for inspection
    print(f"\n\n{'='*80}")
    print("  ALL EVENTS (chronological)")
    print(f"{'='*80}")
    print(f"  {'Timestamp':<25} {'Type':<15} {'Funding':>10} {'Regime':>10} {'MAE%':>8} {'MFE%':>8} {'RevFirst':>10} {'Horizon%':>10}")
    print(f"  {'-'*106}")
    for _, r in results_df.sort_values("timestamp").iterrows():
        rev_str = "YES" if r["reversal_happened_first"] == True else ("NO" if r["reversal_happened_first"] == False else "—")
        print(f"  {str(r['timestamp'])[:25]:<25} {r['type']:<15} {r['funding_rate']:>+10.6f} {r['regime']:>10} {r['mae_pct']*100:>8.3f} {r['mfe_pct']*100:>8.3f} {rev_str:>10} {r['price_at_horizon_pct']*100:>+10.3f}")


if __name__ == "__main__":
    main()
