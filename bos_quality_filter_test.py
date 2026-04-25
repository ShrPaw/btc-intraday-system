"""
BOS QUALITY FILTER TEST — LONG SETUPS (BTCUSDT)
================================================
Tests whether filtering weak BOS events improves LONG edge.

BOS quality metrics (all derived from existing OHLCV — no new indicators):
  1. Break magnitude: how far price moved beyond the swing level
  2. Close position: did the candle close beyond the broken level (not just wick)
  3. Body-to-range ratio: strong body = conviction, wick = rejection risk

Rules:
  - No parameter tuning or threshold optimization
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
    TRAIN_CUTOFF, LOOKBACK_SWING, STRUCTURE_GATE_WINDOW
)


def compute_bos_quality_at_signals(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    For each validated setup, compute BOS quality metrics at signal time.
    Uses the same swing high/low computation as the engine (LOOKBACK_SWING=12).
    """
    results = results.copy()

    # Pre-compute swing levels on the full df (same as engine)
    df = df.copy()
    df["prev_swing_high"] = df["high"].rolling(LOOKBACK_SWING).max().shift(1)
    df["prev_swing_low"] = df["low"].rolling(LOOKBACK_SWING).min().shift(1)

    # For each setup, find the signal bar and compute BOS quality
    bos_mag_list = []
    bos_close_pos_list = []
    bos_body_ratio_list = []
    bos_vol_ratio_list = []
    bos_quality_list = []

    for _, row in results.iterrows():
        sig_time = row["signal_time"]
        direction = int(row["direction"])

        # Find signal bar in df
        bar_mask = df["timestamp"] == sig_time
        if not bar_mask.any():
            bos_mag_list.append(np.nan)
            bos_close_pos_list.append(np.nan)
            bos_body_ratio_list.append(np.nan)
            bos_vol_ratio_list.append(np.nan)
            bos_quality_list.append(np.nan)
            continue

        sig_idx = df.index[bar_mask][0]
        bar = df.iloc[sig_idx]

        swing_high = bar["prev_swing_high"]
        swing_low = bar["prev_swing_low"]

        candle_high = float(bar["high"])
        candle_low = float(bar["low"])
        candle_close = float(bar["close"])
        candle_open = float(bar["open"])
        candle_vol = float(bar["volume"])

        candle_range = candle_high - candle_low
        body = abs(candle_close - candle_open)

        if direction == 1:
            # LONG BOS: high broke above swing_high
            if pd.notna(swing_high) and candle_high > swing_high:
                # Break magnitude: how far above the swing level
                break_mag = (candle_high - swing_high) / swing_high * 100  # in %

                # Close position: did it close above the swing level?
                # 1.0 = closed at high, 0.0 = closed at swing level, negative = closed below
                if candle_range > 0:
                    close_pos = (candle_close - swing_high) / candle_range
                else:
                    close_pos = 0.0

                # Body-to-range ratio
                body_ratio = body / candle_range if candle_range > 0 else 0.0

                # Volume ratio: this candle vs recent average
                recent_vol = df.iloc[max(0, sig_idx-20):sig_idx]["volume"].mean()
                vol_ratio = candle_vol / recent_vol if recent_vol > 0 else 1.0

                # Composite quality score (simple, not optimized)
                # Break magnitude: bigger is better (capped at 0.5%)
                mag_score = min(break_mag / 0.5, 1.0)
                # Close position: higher is better (close above swing)
                close_score = max(0, min(close_pos, 1.0))
                # Body ratio: higher is better (strong body, not wick)
                body_score = body_ratio
                # Volume: higher is better (capped at 2x)
                vol_score = min(vol_ratio / 2.0, 1.0)

                quality = 0.30 * mag_score + 0.35 * close_score + 0.20 * body_score + 0.15 * vol_score
            else:
                break_mag = 0.0
                close_pos = 0.0
                body_ratio = 0.0
                vol_ratio = 1.0
                quality = 0.0

        else:
            # SHORT BOS: low broke below swing_low
            if pd.notna(swing_low) and candle_low < swing_low:
                break_mag = (swing_low - candle_low) / swing_low * 100  # in %

                if candle_range > 0:
                    close_pos = (swing_low - candle_close) / candle_range
                else:
                    close_pos = 0.0

                body_ratio = body / candle_range if candle_range > 0 else 0.0

                recent_vol = df.iloc[max(0, sig_idx-20):sig_idx]["volume"].mean()
                vol_ratio = candle_vol / recent_vol if recent_vol > 0 else 1.0

                mag_score = min(break_mag / 0.5, 1.0)
                close_score = max(0, min(close_pos, 1.0))
                body_score = body_ratio
                vol_score = min(vol_ratio / 2.0, 1.0)

                quality = 0.30 * mag_score + 0.35 * close_score + 0.20 * body_score + 0.15 * vol_score
            else:
                break_mag = 0.0
                close_pos = 0.0
                body_ratio = 0.0
                vol_ratio = 1.0
                quality = 0.0

        bos_mag_list.append(break_mag)
        bos_close_pos_list.append(close_pos)
        bos_body_ratio_list.append(body_ratio)
        bos_vol_ratio_list.append(vol_ratio)
        bos_quality_list.append(quality)

    results["bos_break_mag"] = bos_mag_list
    results["bos_close_pos"] = bos_close_pos_list
    results["bos_body_ratio"] = bos_body_ratio_list
    results["bos_vol_ratio"] = bos_vol_ratio_list
    results["bos_quality"] = bos_quality_list

    return results


def analyze_bos_quality(results: pd.DataFrame, label: str):
    """Analyze BOS quality distribution and its relationship to outcomes."""
    print(f"\n{'═' * 80}")
    print(f"  BOS QUALITY ANALYSIS — {label}")
    print(f"{'═' * 80}")

    # Only setups with valid BOS quality
    valid = results[results["bos_quality"].notna()].copy()
    if len(valid) == 0:
        print("  No valid BOS quality data.")
        return

    print(f"\n  Total setups with BOS quality: {len(valid)}")

    # Distribution stats
    print(f"\n  BOS QUALITY DISTRIBUTION:")
    print(f"    Mean:   {valid['bos_quality'].mean():.3f}")
    print(f"    Median: {valid['bos_quality'].median():.3f}")
    print(f"    Std:    {valid['bos_quality'].std():.3f}")
    print(f"    Min:    {valid['bos_quality'].min():.3f}")
    print(f"    Max:    {valid['bos_quality'].max():.3f}")

    # Component stats
    print(f"\n  COMPONENT STATS:")
    print(f"    Break magnitude (%%): mean={valid['bos_break_mag'].mean():.3f}%, median={valid['bos_break_mag'].median():.3f}%")
    print(f"    Close position:      mean={valid['bos_close_pos'].mean():.3f}, median={valid['bos_close_pos'].median():.3f}")
    print(f"    Body ratio:          mean={valid['bos_body_ratio'].mean():.3f}, median={valid['bos_body_ratio'].median():.3f}")
    print(f"    Volume ratio:        mean={valid['bos_vol_ratio'].mean():.2f}x, median={valid['bos_vol_ratio'].median():.2f}x")

    # BOS quality quartile analysis
    print(f"\n  BOS QUALITY QUARTILE → OUTCOME:")
    print(f"  {'Quartile':<12s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s} {'MedMAE':>7s}")
    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")

    for q_label, q_mask in [
        ("Q1 (worst)", valid["bos_quality"] <= valid["bos_quality"].quantile(0.25)),
        ("Q2", (valid["bos_quality"] > valid["bos_quality"].quantile(0.25)) & (valid["bos_quality"] <= valid["bos_quality"].quantile(0.50))),
        ("Q3", (valid["bos_quality"] > valid["bos_quality"].quantile(0.50)) & (valid["bos_quality"] <= valid["bos_quality"].quantile(0.75))),
        ("Q4 (best)", valid["bos_quality"] > valid["bos_quality"].quantile(0.75)),
    ]:
        subset = valid[q_mask]
        m = compute_group_metrics(subset)
        if m:
            print(f"  {q_label:<12s} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R {m['median_MAE_R']:>6.2f}R")

    # Threshold sweep: what happens as we raise the quality floor?
    print(f"\n  QUALITY FLOOR SWEEP (LONG only):")
    print(f"  {'Floor':>8s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s} {'Filtered':>9s}")
    print(f"  {'─'*8} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*9}")

    total = len(valid)
    for floor in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        filtered = valid[valid["bos_quality"] >= floor]
        m = compute_group_metrics(filtered)
        if m and m["count"] >= 3:
            pct_kept = len(filtered) / total * 100
            print(f"  ≥{floor:>6.2f} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R {pct_kept:>7.1f}%")


def analyze_component_impact(results: pd.DataFrame, label: str):
    """Analyze each BOS component independently."""
    print(f"\n{'═' * 80}")
    print(f"  BOS COMPONENT ANALYSIS — {label}")
    print(f"{'═' * 80}")

    valid = results[results["bos_quality"].notna()].copy()
    if len(valid) == 0:
        print("  No valid data.")
        return

    for comp_name, comp_col in [
        ("Break Magnitude", "bos_break_mag"),
        ("Close Position", "bos_close_pos"),
        ("Body Ratio", "bos_body_ratio"),
        ("Volume Ratio", "bos_vol_ratio"),
    ]:
        print(f"\n  {comp_name}:")
        print(f"  {'Bin':<20s} {'N':>5s} {'Hit1R':>7s} {'Exp1R':>8s} {'MedMFE':>7s}")
        print(f"  {'─'*20} {'─'*5} {'─'*7} {'─'*8} {'─'*7}")

        if comp_col == "bos_break_mag":
            bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 999)]
            labels = ["0-0.05%", "0.05-0.1%", "0.1-0.2%", "0.2-0.3%", ">0.3%"]
        elif comp_col == "bos_close_pos":
            bins = [(-999, 0), (0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 999)]
            labels = ["<0 (below)", "0-0.3", "0.3-0.6", "0.6-1.0", ">1.0"]
        elif comp_col == "bos_body_ratio":
            bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 999)]
            labels = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">0.9"]
        else:  # vol_ratio
            bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 999)]
            labels = ["<0.5x", "0.5-1.0x", "1.0-1.5x", "1.5-2.0x", ">2.0x"]

        for (lo, hi), lbl in zip(bins, labels):
            subset = valid[(valid[comp_col] >= lo) & (valid[comp_col] < hi)]
            m = compute_group_metrics(subset)
            if m and m["count"] >= 2:
                print(f"  {lbl:<20s} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R")


def train_test_comparison(results: pd.DataFrame):
    """Compare BOS quality effect in train vs test."""
    print(f"\n{'═' * 80}")
    print(f"  TRAIN vs TEST — BOS QUALITY EFFECT")
    print(f"{'═' * 80}")

    for period in ["TRAIN", "TEST"]:
        sub = results[results["period"] == period].copy()
        valid = sub[sub["bos_quality"].notna()]
        if len(valid) == 0:
            continue

        print(f"\n  {period} ({len(valid)} setups):")
        print(f"    Mean BOS quality: {valid['bos_quality'].mean():.3f}")

        # Low vs high quality
        median_q = valid["bos_quality"].median()
        low_q = valid[valid["bos_quality"] <= median_q]
        high_q = valid[valid["bos_quality"] > median_q]

        m_low = compute_group_metrics(low_q)
        m_high = compute_group_metrics(high_q)

        if m_low and m_high:
            print(f"    Below median (≤{median_q:.3f}): {m_low['count']} setups, Exp1R={m_low['expectancy_1R']:+.3f}, Hit1R={m_low['pct_hit_1R']:.1%}")
            print(f"    Above median (>{median_q:.3f}): {m_high['count']} setups, Exp1R={m_high['expectancy_1R']:+.3f}, Hit1R={m_high['pct_hit_1R']:.1%}")
            print(f"    Delta: {m_high['expectancy_1R'] - m_low['expectancy_1R']:+.3f}R")


def main():
    print("=" * 80)
    print("  BOS QUALITY FILTER TEST — LONG SETUPS (BTCUSDT)")
    print("  Testing if BOS strength predicts LONG outcome")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Build master dataset
    print("\n  Building master dataset...")
    df = build_master_dataset("data/features/btcusdt_1m.csv")

    # Split train/test
    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

    print(f"  Train: {len(df_train):,} bars ({df_train['timestamp'].min().strftime('%Y-%m-%d')} → {df_train['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Test:  {len(df_test):,} bars ({df_test['timestamp'].min().strftime('%Y-%m-%d')} → {df_test['timestamp'].max().strftime('%Y-%m-%d')})")

    # Run validation
    print("\n  Running validation (train)...")
    train_results = run_validation(df_train, "BTCUSDT")
    train_results["period"] = "TRAIN"

    print("  Running validation (test)...")
    test_results = run_validation(df_test, "BTCUSDT")
    test_results["period"] = "TEST"

    all_results = pd.concat([train_results, test_results], ignore_index=True)

    # Filter LONG only
    longs = all_results[all_results["direction"] == 1].copy()
    print(f"\n  LONG setups: {len(longs)} total ({(longs['period']=='TRAIN').sum()} train, {(longs['period']=='TEST').sum()} test)")

    # Compute BOS quality
    print("\n  Computing BOS quality metrics...")
    # Need full df for volume ratio computation
    longs_train = compute_bos_quality_at_signals(df_train, longs[longs["period"] == "TRAIN"])
    longs_test = compute_bos_quality_at_signals(df_test, longs[longs["period"] == "TEST"])
    longs = pd.concat([longs_train, longs_test], ignore_index=True)

    # Baseline (all LONGs)
    print(f"\n{'─' * 80}")
    print(f"  BASELINE — ALL LONG SETUPS")
    print(f"{'─' * 80}")
    m_all = compute_group_metrics(longs)
    m_train = compute_group_metrics(longs[longs["period"] == "TRAIN"])
    m_test = compute_group_metrics(longs[longs["period"] == "TEST"])
    if m_all:
        print(f"  ALL:   {m_all['count']:>3d} setups | Exp1R={m_all['expectancy_1R']:+.3f} | Hit1R={m_all['pct_hit_1R']:.1%}")
    if m_train:
        print(f"  TRAIN: {m_train['count']:>3d} setups | Exp1R={m_train['expectancy_1R']:+.3f} | Hit1R={m_train['pct_hit_1R']:.1%}")
    if m_test:
        print(f"  TEST:  {m_test['count']:>3d} setups | Exp1R={m_test['expectancy_1R']:+.3f} | Hit1R={m_test['pct_hit_1R']:.1%}")

    # Full analysis
    analyze_bos_quality(longs, "ALL LONGS (TRAIN + TEST)")
    analyze_bos_quality(longs[longs["period"] == "TEST"], "LONGS — TEST PERIOD ONLY")
    analyze_component_impact(longs, "ALL LONGS")
    train_test_comparison(longs)

    # Also test SHORTS for comparison (same BOS quality filter)
    shorts = all_results[all_results["direction"] == -1].copy()
    if len(shorts) > 0:
        shorts_train = compute_bos_quality_at_signals(df_train, shorts[shorts["period"] == "TRAIN"])
        shorts_test = compute_bos_quality_at_signals(df_test, shorts[shorts["period"] == "TEST"])
        shorts = pd.concat([shorts_train, shorts_test], ignore_index=True)
        analyze_bos_quality(shorts[shorts["period"] == "TEST"], "SHORTS — TEST PERIOD (for comparison)")

    # Save enriched results
    output_cols = [
        "symbol", "signal_time", "entry_time", "direction", "dir_label",
        "setup_type", "confidence_mode", "period",
        "hit_1R", "hit_2R", "hit_3R", "sl_hit",
        "max_favorable_excursion_R", "max_adverse_excursion_R",
        "bos_break_mag", "bos_close_pos", "bos_body_ratio", "bos_vol_ratio", "bos_quality",
        "h4_rsi_entry", "htf_regime", "session", "month"
    ]
    save_cols = [c for c in output_cols if c in longs.columns]
    longs[save_cols].to_csv("data/features/bos_quality_long_results.csv", index=False)
    print(f"\n  [SAVED] data/features/bos_quality_long_results.csv ({len(longs)} rows)")

    print(f"\n{'═' * 80}")
    print(f"  BOS QUALITY FILTER TEST COMPLETE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
