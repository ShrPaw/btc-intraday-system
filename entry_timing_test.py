"""
ENTRY TIMING TEST — LONG SETUPS (BTCUSDT)
==========================================
Tests whether pullback-to-EMA20 entries improve LONG edge.

Hypothesis: After a BOS triggers, waiting for price to pull back to EMA20
before entering gives a better risk/reward entry.

Rules:
  - No parameter tuning
  - No new indicators
  - No changes to RSI/EMA/BOS/ADX logic
  - BTC-only primary validation
  - R-multiples, worst-case intracandle, closed candles
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF, OUTCOME_HORIZON_BARS
)


def find_pullback_entry(df, signal_idx, direction, max_wait_bars=20):
    """
    After a signal fires, wait for price to pull back near EMA20.
    Returns: entry_idx, entry_price, bars_waited, or None if no pullback.

    For LONG: entry when low touches/crosses EMA20 (pullback to support)
    """
    signal_bar = df.iloc[signal_idx]
    ema20_at_signal = float(signal_bar["ema20"])

    # Look ahead up to max_wait_bars for a pullback
    for j in range(signal_idx + 1, min(signal_idx + 1 + max_wait_bars, len(df))):
        bar = df.iloc[j]
        low = float(bar["low"])
        close = float(bar["close"])

        if direction == 1:
            # LONG: pullback to EMA20 — low touches or goes below EMA20
            if low <= ema20_at_signal:
                # Enter at EMA20 level (conservative: enter at EMA20, not at low)
                entry_price = ema20_at_signal
                return j, entry_price, j - signal_idx

    return None


def run_validation_with_pullback(df, symbol, max_wait=20):
    """
    Run validation but with pullback-to-EMA20 entry timing.
    Uses the same stop/TP/outcome logic but entry is at pullback.
    """
    from setup_validation_engine import compute_structural_stop, compute_tp_levels, track_setup_outcome

    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        if row["setup_type"] == "none" or row["direction"] == 0 or not bool(row["take_trade"]):
            continue

        direction = int(row["direction"])
        if direction != 1:  # LONG only
            continue

        strategy = str(row["setup_type"])
        signal_time = row["timestamp"]

        # Find pullback entry
        pullback = find_pullback_entry(df, i, direction, max_wait_bars=max_wait)
        if pullback is None:
            continue  # No pullback within window — skip

        entry_idx, entry_price, bars_waited = pullback
        entry_time = df.iloc[entry_idx]["timestamp"]

        # Structural stop (from signal bar, same as engine)
        stop_info = compute_structural_stop(df, i, direction, strategy)
        if not stop_info["is_stop_valid"]:
            continue

        stop_price = stop_info["stop_price"]
        stop_distance_pct = stop_info["stop_distance_pct"]

        # TP levels in R (from pullback entry price)
        tp_levels = compute_tp_levels(entry_price, stop_price, direction)
        tp_levels["entry_price"] = entry_price

        # Track outcome from pullback entry
        outcome = track_setup_outcome(df, entry_idx, direction, stop_price, tp_levels)
        if not outcome.get("valid", False):
            continue

        result = {
            "symbol": symbol,
            "signal_time": signal_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "setup_type": strategy,
            "confidence_mode": str(row["confidence_mode"]),
            "confidence_raw": float(row["confidence_raw"]),
            "stop_price": stop_price,
            "stop_distance_pct": stop_distance_pct,
            "stop_source": stop_info["stop_source"],
            "R_abs": tp_levels["R_abs"],
            "bars_waited": bars_waited,
        }

        for key, val in outcome.items():
            if key != "valid":
                result[key] = val

        # Regime
        h4_rsi = float(row.get("h4_rsi", 50))
        result["h4_rsi_entry"] = h4_rsi
        if h4_rsi > 55:
            result["htf_regime"] = "bullish"
        elif h4_rsi < 45:
            result["htf_regime"] = "bearish"
        else:
            result["htf_regime"] = "neutral"

        hour = signal_time.hour
        if 0 <= hour < 8:
            result["session"] = "Asian"
        elif 8 <= hour < 16:
            result["session"] = "European"
        else:
            result["session"] = "US"

        result["month"] = signal_time.strftime("%Y-%m")
        results.append(result)

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("  ENTRY TIMING TEST — LONG SETUPS (BTCUSDT)")
    print("  Testing pullback-to-EMA20 entry for LONGs")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Build master dataset
    print("\n  Building master dataset...")
    df = build_master_dataset("data/features/btcusdt_1m.csv")

    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

    print(f"  Train: {len(df_train):,} bars")
    print(f"  Test:  {len(df_test):,} bars")

    # ─────────────────────────────────────────
    # A. Standard next-open entry (baseline)
    # ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  A. BASELINE — Next-candle-open entry (LONG only)")
    print(f"{'─' * 60}")

    train_std = run_validation(df_train, "BTCUSDT")
    test_std = run_validation(df_test, "BTCUSDT")
    train_std_long = train_std[train_std["direction"] == 1]
    test_std_long = test_std[test_std["direction"] == 1]

    m_train_std = compute_group_metrics(train_std_long)
    m_test_std = compute_group_metrics(test_std_long)
    if m_train_std:
        print(f"  Train: {m_train_std['count']} setups | Hit1R={m_train_std['pct_hit_1R']:.1%} | Exp1R={m_train_std['expectancy_1R']:+.3f}")
    if m_test_std:
        print(f"  Test:  {m_test_std['count']} setups | Hit1R={m_test_std['pct_hit_1R']:.1%} | Exp1R={m_test_std['expectancy_1R']:+.3f}")

    # ─────────────────────────────────────────
    # B. Pullback-to-EMA20 entry
    # ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  B. PULLBACK-TO-EMA20 entry (LONG only)")
    print(f"{'─' * 60}")

    for max_wait in [5, 10, 15, 20]:
        print(f"\n    Max wait: {max_wait} bars")
        train_pb = run_validation_with_pullback(df_train, "BTCUSDT", max_wait=max_wait)
        test_pb = run_validation_with_pullback(df_test, "BTCUSDT", max_wait=max_wait)

        m_train_pb = compute_group_metrics(train_pb)
        m_test_pb = compute_group_metrics(test_pb)

        if m_train_pb:
            print(f"      Train: {m_train_pb['count']} setups | Hit1R={m_train_pb['pct_hit_1R']:.1%} | Exp1R={m_train_pb['expectancy_1R']:+.3f}")
        else:
            print(f"      Train: 0 setups")
        if m_test_pb:
            print(f"      Test:  {m_test_pb['count']} setups | Hit1R={m_test_pb['pct_hit_1R']:.1%} | Exp1R={m_test_pb['expectancy_1R']:+.3f}")
        else:
            print(f"      Test: 0 setups")

    # ─────────────────────────────────────────
    # C. Detailed pullback analysis (max_wait=10)
    # ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  C. DETAILED PULLBACK ANALYSIS (max_wait=10)")
    print(f"{'─' * 60}")

    for period_label, period_df in [("TRAIN", df_train), ("TEST", df_test)]:
        pb = run_validation_with_pullback(period_df, "BTCUSDT", max_wait=10)
        if len(pb) == 0:
            print(f"\n  {period_label}: No pullback entries found")
            continue

        print(f"\n  {period_label} — {len(pb)} pullback entries:")
        print(f"    Median bars waited: {pb['bars_waited'].median():.0f}")
        print(f"    Mean bars waited:   {pb['bars_waited'].mean():.1f}")

        # Compare R distribution
        m_pb = compute_group_metrics(pb)
        # Baseline for same period
        std_results = run_validation(period_df, "BTCUSDT")
        std_longs = std_results[std_results["direction"] == 1]
        m_std = compute_group_metrics(std_longs)

        if m_std and m_pb:
            print(f"\n    COMPARISON:")
            print(f"    {'Metric':<20s} {'Standard':>12s} {'Pullback':>12s} {'Delta':>10s}")
            print(f"    {'─'*20} {'─'*12} {'─'*12} {'─'*10}")
            print(f"    {'Setups':<20s} {m_std['count']:>12d} {m_pb['count']:>12d}")
            print(f"    {'Hit 1R':<20s} {m_std['pct_hit_1R']:>11.1%} {m_pb['pct_hit_1R']:>11.1%} {m_pb['pct_hit_1R']-m_std['pct_hit_1R']:>+9.1%}")
            print(f"    {'Hit 2R':<20s} {m_std['pct_hit_2R']:>11.1%} {m_pb['pct_hit_2R']:>11.1%} {m_pb['pct_hit_2R']-m_std['pct_hit_2R']:>+9.1%}")
            print(f"    {'Exp 1R':<20s} {m_std['expectancy_1R']:>+11.3f}R {m_pb['expectancy_1R']:>+11.3f}R {m_pb['expectancy_1R']-m_std['expectancy_1R']:>+9.3f}R")
            print(f"    {'Exp 2R':<20s} {m_std['expectancy_2R']:>+11.3f}R {m_pb['expectancy_2R']:>+11.3f}R {m_pb['expectancy_2R']-m_std['expectancy_2R']:>+9.3f}R")
            print(f"    {'Med MFE':<20s} {m_std['median_MFE_R']:>11.2f}R {m_pb['median_MFE_R']:>11.2f}R {m_pb['median_MFE_R']-m_std['median_MFE_R']:>+9.2f}R")

    # ─────────────────────────────────────────
    # D. Pullback per-setup details (test)
    # ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  D. PULLBACK ENTRIES — TEST PERIOD DETAILS")
    print(f"{'─' * 60}")

    test_pb = run_validation_with_pullback(df_test, "BTCUSDT", max_wait=10)
    if len(test_pb) > 0:
        for _, row in test_pb.iterrows():
            hit = "WIN" if row["hit_1R"] else "LOSS"
            print(f"    {row['signal_time']} | {row['setup_type']:<10s} | {row['confidence_mode']:<8s} | h4_rsi={row['h4_rsi_entry']:.1f} | wait={row['bars_waited']}bars | {hit} | MFE={row['max_favorable_excursion_R']:.2f}R")
    else:
        print("    No pullback entries in test period.")

    # ─────────────────────────────────────────
    # E. Missed setups (no pullback within window)
    # ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  E. MISSED SETUPS — No pullback within 10 bars (test)")
    print(f"{'─' * 60}")

    std_test = run_validation(df_test, "BTCUSDT")
    std_longs = std_test[std_test["direction"] == 1]
    pb_test = run_validation_with_pullback(df_test, "BTCUSDT", max_wait=10)

    # Setups that fired but didn't get a pullback
    std_times = set(std_longs["signal_time"].values)
    pb_times = set(pb_test["signal_time"].values) if len(pb_test) > 0 else set()
    missed_times = std_times - pb_times

    if missed_times:
        missed = std_longs[std_longs["signal_time"].isin(missed_times)]
        m_missed = compute_group_metrics(missed)
        print(f"    Missed: {len(missed)} setups (no pullback within 10 bars)")
        if m_missed:
            print(f"    These setups (standard entry): Hit1R={m_missed['pct_hit_1R']:.1%} | Exp1R={m_missed['expectancy_1R']:+.3f}")
        for _, row in missed.iterrows():
            hit = "WIN" if row["hit_1R"] else "LOSS"
            print(f"      {row['signal_time']} | {row['setup_type']:<10s} | {row['confidence_mode']:<8s} | {hit}")
    else:
        print("    All setups had a pullback within 10 bars.")

    print(f"\n{'═' * 80}")
    print(f"  ENTRY TIMING TEST COMPLETE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
