"""
TRAIN/TEST STABILITY INVESTIGATION — BTCUSDT LONG
==================================================
Diagnoses WHY the train period (Jun-Nov 2025) is deeply negative
while the test period (Dec 2025-Apr 2026) is strongly positive.

This is a diagnostic, not an optimization. We are trying to understand
whether the LONG edge is real but regime-dependent, or simply luck.

Approach:
  1. Compare market structure between train and test periods
  2. Check if the setup conditions fire in different regimes
  3. Walk-forward splits to find where the edge appears/disappears
  4. Check if the setup detection itself is biased
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF
)


def market_regime_stats(df, label):
    """Characterize the market regime of a period."""
    print(f"\n  MARKET REGIME — {label}")
    print(f"  {'─' * 60}")

    # Price action
    start_price = float(df.iloc[0]["close"])
    end_price = float(df.iloc[-1]["close"])
    ret = (end_price - start_price) / start_price * 100
    print(f"    Price: {start_price:.0f} → {end_price:.0f} ({ret:+.1f}%)")

    # Volatility
    ret_1m = df["close"].pct_change().dropna()
    ann_vol = ret_1m.std() * np.sqrt(525600) * 100  # annualized from 1m
    print(f"    Annualized vol: {ann_vol:.1f}%")

    # H4 RSI distribution
    h4_rsi = df["h4_rsi"].dropna()
    print(f"    H4 RSI: mean={h4_rsi.mean():.1f}, median={h4_rsi.median():.1f}, std={h4_rsi.std():.1f}")
    print(f"    H4 RSI distribution:")
    for lo, hi in [(30, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 80)]:
        pct = ((h4_rsi >= lo) & (h4_rsi < hi)).mean() * 100
        bar = "█" * int(pct / 2)
        print(f"      [{lo},{hi}): {pct:5.1f}% {bar}")

    # H12 RSI distribution
    h12_rsi = df["h12_rsi"].dropna()
    print(f"    H12 RSI: mean={h12_rsi.mean():.1f}, median={h12_rsi.median():.1f}")

    # ADX
    if "h4_adx" in df.columns:
        adx = df["h4_adx"].dropna()
        print(f"    H4 ADX: mean={adx.mean():.1f}, >25={((adx>25).mean()*100):.1f}%, >30={((adx>30).mean()*100):.1f}%")

    # Trend: count bars where H4 RSI > 50 vs < 50
    bull_pct = (h4_rsi > 50).mean() * 100
    print(f"    H4 RSI > 50: {bull_pct:.1f}% of bars (bullish bias)")

    # Monthly returns
    df_m = df.set_index("timestamp")["close"].resample("ME").last().pct_change().dropna() * 100
    print(f"    Monthly returns:")
    for ts, ret_m in df_m.items():
        direction = "▲" if ret_m > 0 else "▼"
        print(f"      {ts.strftime('%Y-%m')}: {ret_m:+.1f}% {direction}")


def setup_fire_analysis(df, results, label):
    """Analyze WHERE and WHEN setups fire."""
    print(f"\n  SETUP FIRING — {label}")
    print(f"  {'─' * 60}")

    longs = results[results["direction"] == 1]
    print(f"    Total LONG setups: {len(longs)}")

    if len(longs) == 0:
        return

    # H4 RSI at signal time
    h4_vals = longs["h4_rsi_entry"].dropna()
    print(f"    H4 RSI at signal: mean={h4_vals.mean():.1f}, range=[{h4_vals.min():.1f}, {h4_vals.max():.1f}]")

    # Distribution of H4 RSI at signal time
    print(f"    H4 RSI distribution at signal:")
    for lo, hi in [(45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 80)]:
        n = ((h4_vals >= lo) & (h4_vals < hi)).sum()
        wins = longs[(longs["h4_rsi_entry"] >= lo) & (longs["h4_rsi_entry"] < hi)]["hit_1R"].astype(bool).sum()
        if n > 0:
            print(f"      [{lo},{hi}): {n} setups, {wins} wins ({wins/n:.0%})")

    # Setup type breakdown
    for st in ["RSI_SCALP", "RSI_TREND"]:
        st_longs = longs[longs["setup_type"] == st]
        if len(st_longs) > 0:
            m = compute_group_metrics(st_longs)
            if m:
                print(f"    {st}: {m['count']} setups, Hit1R={m['pct_hit_1R']:.0%}, Exp1R={m['expectancy_1R']:+.3f}")

    # Confidence distribution
    print(f"    Confidence tier distribution:")
    for tier in ["MILD", "MID", "HIGH", "PREMIUM"]:
        n = (longs["confidence_mode"] == tier).sum()
        if n > 0:
            wins = longs[longs["confidence_mode"] == tier]["hit_1R"].astype(bool).sum()
            print(f"      {tier}: {n} setups, {wins} wins ({wins/n:.0%})")

    # Monthly firing
    print(f"    Monthly:")
    for month in sorted(longs["month"].unique()):
        m_longs = longs[longs["month"] == month]
        n = len(m_longs)
        wins = m_longs["hit_1R"].astype(bool).sum()
        r = compute_r_series(m_longs)
        print(f"      {month}: {n} setups, {wins} wins, {r.sum():+.1f}R")


def compute_r_series(subset):
    """Compute R per trade."""
    r_list = []
    for _, row in subset.iterrows():
        if row["sl_hit"] and not row["hit_1R"]:
            r = -1.0
        elif row["hit_1R"] and row["sl_hit"]:
            r = 0.0
        elif row["hit_1R"] and not row["sl_hit"]:
            if row.get("hit_4R", False):
                r = 4.0
            elif row.get("hit_3R", False):
                r = 3.0
            elif row.get("hit_2R", False):
                r = 2.0
            else:
                r = 1.0
        else:
            r = 0.0
        r_list.append(r)
    return np.array(r_list)


def walk_forward_splits(df, results):
    """Test edge across multiple walk-forward splits."""
    print(f"\n  WALK-FORWARD SPLITS")
    print(f"  {'─' * 60}")

    # 3-month windows, stepping by 1 month
    months = sorted(df["timestamp"].dt.to_period("M").unique())
    print(f"    Available months: {len(months)}")

    print(f"\n    {'Window':<30s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"    {'─'*30} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")

    for i in range(len(months) - 2):
        start_month = months[i]
        end_month = months[i + 2]

        window_mask = (results["month"] >= str(start_month)) & (results["month"] <= str(end_month))
        window = results[window_mask & (results["direction"] == 1)]

        if len(window) > 0:
            m = compute_group_metrics(window)
            r = compute_r_series(window)
            label = f"{start_month} → {end_month}"
            marker = ""
            if m:
                print(f"    {label:<30s} {m['count']:>4d} {m['pct_hit_1R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {r.sum():>+7.1f}R{marker}")

    # Also: 2-month windows
    print(f"\n    2-month windows:")
    print(f"    {'Window':<30s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"    {'─'*30} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")

    for i in range(len(months) - 1):
        start_month = months[i]
        end_month = months[i + 1]

        window_mask = (results["month"] >= str(start_month)) & (results["month"] <= str(end_month))
        window = results[window_mask & (results["direction"] == 1)]

        if len(window) > 0:
            m = compute_group_metrics(window)
            r = compute_r_series(window)
            label = f"{start_month} → {end_month}"
            if m:
                print(f"    {label:<30s} {m['count']:>4d} {m['pct_hit_1R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {r.sum():>+7.1f}R")


def stop_analysis(results, label):
    """Analyze stop distances and outcomes."""
    print(f"\n  STOP ANALYSIS — {label}")
    print(f"  {'─' * 60}")

    longs = results[results["direction"] == 1]
    if len(longs) == 0:
        print("    No LONG setups.")
        return

    # Stop distance distribution
    sd = longs["stop_distance_pct"] * 100
    print(f"    Stop distance: mean={sd.mean():.2f}%, median={sd.median():.2f}%, range=[{sd.min():.2f}%, {sd.max():.2f}%]")

    # Stop source breakdown
    for source in longs["stop_source"].unique():
        n = (longs["stop_source"] == source).sum()
        wins = longs[longs["stop_source"] == source]["hit_1R"].astype(bool).sum()
        print(f"    {source}: {n} setups, {wins} wins ({wins/n:.0%})")

    # Stop distance vs outcome
    print(f"    Stop distance vs outcome:")
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 3.0)]:
        mask = (sd >= lo) & (sd < hi)
        n = mask.sum()
        if n > 0:
            wins = longs[mask]["hit_1R"].astype(bool).sum()
            print(f"      [{lo:.1f}%, {hi:.1f}%): {n} setups, {wins} wins ({wins/n:.0%})")


def main():
    print("=" * 80)
    print("  TRAIN/TEST STABILITY INVESTIGATION — BTCUSDT LONG")
    print("  Diagnosing why train is negative, test is positive")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Build dataset
    print("\n  Building master dataset...")
    df = build_master_dataset("data/features/btcusdt_1m.csv")

    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

    print(f"  Train: {len(df_train):,} bars ({df_train['timestamp'].min().strftime('%Y-%m-%d')} → {df_train['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Test:  {len(df_test):,} bars ({df_test['timestamp'].min().strftime('%Y-%m-%d')} → {df_test['timestamp'].max().strftime('%Y-%m-%d')})")

    # Run validation
    print("\n  Running validation...")
    train_results = run_validation(df_train, "BTCUSDT")
    test_results = run_validation(df_test, "BTCUSDT")
    train_results["period"] = "TRAIN"
    test_results["period"] = "TEST"
    all_results = pd.concat([train_results, test_results], ignore_index=True)

    # ═══════════════════════════════════════════════════════
    # 1. MARKET REGIME COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  1. MARKET REGIME COMPARISON")
    print(f"{'═' * 80}")

    market_regime_stats(df_train, "TRAIN (Jun-Nov 2025)")
    market_regime_stats(df_test, "TEST (Dec 2025-Apr 2026)")

    # ═══════════════════════════════════════════════════════
    # 2. SETUP FIRING PATTERNS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  2. SETUP FIRING PATTERNS")
    print(f"{'═' * 80}")

    setup_fire_analysis(df_train, train_results, "TRAIN")
    setup_fire_analysis(df_test, test_results, "TEST")

    # ═══════════════════════════════════════════════════════
    # 3. STOP ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  3. STOP ANALYSIS")
    print(f"{'═' * 80}")

    stop_analysis(train_results, "TRAIN")
    stop_analysis(test_results, "TEST")

    # ═══════════════════════════════════════════════════════
    # 4. WALK-FORWARD SPLITS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  4. WALK-FORWARD SPLITS (ALL LONGS)")
    print(f"{'═' * 80}")

    walk_forward_splits(df, all_results)

    # ═══════════════════════════════════════════════════════
    # 5. EVERY SINGLE TRADE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  5. EVERY LONG TRADE (chronological)")
    print(f"{'═' * 80}")

    longs = all_results[all_results["direction"] == 1].sort_values("signal_time")
    print(f"\n  {'#':>3s} {'Date':<20s} {'Type':<10s} {'Conf':<8s} {'H4_RSI':>7s} {'H12_RSI':>7s} {'Stop%':>6s} {'Result':>8s} {'R':>5s} {'CumR':>7s}")
    print(f"  {'─'*3} {'─'*20} {'─'*10} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*8} {'─'*5} {'─'*7}")

    cum_r = 0
    for i, (_, row) in enumerate(longs.iterrows()):
        if row["sl_hit"] and not row["hit_1R"]:
            r = -1.0
            result = "LOSS"
        elif row["hit_1R"] and row["sl_hit"]:
            r = 0.0
            result = "BREAKEVEN"
        elif row["hit_1R"] and not row["sl_hit"]:
            if row.get("hit_4R", False):
                r = 4.0
            elif row.get("hit_3R", False):
                r = 3.0
            elif row.get("hit_2R", False):
                r = 2.0
            else:
                r = 1.0
            result = f"WIN({r:.0f}R)"
        else:
            r = 0.0
            result = "EXPIRED"
        cum_r += r

        h4 = row.get("h4_rsi_entry", 0)
        h12 = row.get("h12_rsi_entry", 0)
        stop_pct = row["stop_distance_pct"] * 100
        date_str = str(row["signal_time"])[:19]

        print(f"  {i+1:>3d} {date_str:<20s} {row['setup_type']:<10s} {row['confidence_mode']:<8s} {h4:>6.1f} {h12:>6.1f} {stop_pct:>5.2f}% {result:>8s} {r:>+4.0f}R {cum_r:>+6.1f}R")

    # ═══════════════════════════════════════════════════════
    # 6. DIAGNOSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  6. DIAGNOSIS")
    print(f"{'═' * 80}")

    train_longs = all_results[(all_results["direction"] == 1) & (all_results["period"] == "TRAIN")]
    test_longs = all_results[(all_results["direction"] == 1) & (all_results["period"] == "TEST")]

    m_train = compute_group_metrics(train_longs)
    m_test = compute_group_metrics(test_longs)

    print(f"\n  TRAIN: {m_train['count']} setups, Exp1R={m_train['expectancy_1R']:+.3f}, Hit1R={m_train['pct_hit_1R']:.1%}")
    print(f"  TEST:  {m_test['count']} setups, Exp1R={m_test['expectancy_1R']:+.3f}, Hit1R={m_test['pct_hit_1R']:.1%}")

    # Key differences
    print(f"\n  KEY DIFFERENCES:")

    # H4 RSI distribution at signal time
    train_h4 = train_longs["h4_rsi_entry"].dropna()
    test_h4 = test_longs["h4_rsi_entry"].dropna()
    print(f"    H4 RSI at signal: train mean={train_h4.mean():.1f}, test mean={test_h4.mean():.1f}")

    # Confidence distribution
    train_conf = train_longs["confidence_raw"].dropna()
    test_conf = test_longs["confidence_raw"].dropna()
    print(f"    Confidence: train mean={train_conf.mean():.3f}, test mean={test_conf.mean():.3f}")

    # Stop distance
    train_stop = train_longs["stop_distance_pct"].dropna() * 100
    test_stop = test_longs["stop_distance_pct"].dropna() * 100
    print(f"    Stop distance: train mean={train_stop.mean():.2f}%, test mean={test_stop.mean():.2f}%")

    # MFE/MAE
    train_mfe = train_longs["max_favorable_excursion_R"].dropna()
    test_mfe = test_longs["max_favorable_excursion_R"].dropna()
    print(f"    Median MFE: train={train_mfe.median():.2f}R, test={test_mfe.median():.2f}R")

    train_mae = train_longs["max_adverse_excursion_R"].dropna()
    test_mae = test_longs["max_adverse_excursion_R"].dropna()
    print(f"    Median MAE: train={train_mae.median():.2f}R, test={test_mae.median():.2f}R")

    # Setup type mix
    for st in ["RSI_SCALP", "RSI_TREND"]:
        train_n = (train_longs["setup_type"] == st).sum()
        test_n = (test_longs["setup_type"] == st).sum()
        print(f"    {st}: train={train_n}, test={test_n}")

    # Regime mix
    for regime in ["bullish", "bearish", "neutral"]:
        train_n = (train_longs["htf_regime"] == regime).sum()
        test_n = (test_longs["htf_regime"] == regime).sum()
        print(f"    {regime}: train={train_n}, test={test_n}")

    print(f"\n{'═' * 80}")
    print(f"  INVESTIGATION COMPLETE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
