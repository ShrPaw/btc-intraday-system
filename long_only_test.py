"""
LONG-ONLY & DIRECTIONLESS EXPANSION TEST
=========================================
Tests whether LONG-only or directionless breakout captures the timing edge.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    compute_structural_stop, compute_tp_levels, TRAIN_CUTOFF,
    OUTCOME_HORIZON_BARS, TP_R_MULTIPLES, SWING_STOP_LOOKBACK,
    STOP_FLOOR_PCT, STOP_CAP_PCT
)


def load_btc():
    df = build_master_dataset("data/features/btcusdt_1m.csv")
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    test_results = run_validation(df_test, "BTCUSDT")
    test_results["period"] = "TEST"
    train_results = run_validation(df_train, "BTCUSDT")
    train_results["period"] = "TRAIN"
    all_results = pd.concat([train_results, test_results], ignore_index=True)
    return df_train, df_test, all_results


def directionless_expansion(df, indices, horizons=[5, 10, 20, 50]):
    """
    For each setup index, measure the MAX absolute move from entry within horizon.
    Simulates: enter at next open, capture the biggest move in either direction.
    """
    results = {h: [] for h in horizons}
    best_directions = {h: [] for h in horizons}  # which direction had the max move

    for idx in indices:
        if idx + 1 >= len(df):
            continue
        entry_price = float(df.iloc[idx + 1]["open"])

        for h in horizons:
            end_idx = min(idx + 1 + h, len(df) - 1)
            # Scan all bars in horizon for max high and min low
            window = df.iloc[idx + 2:end_idx + 1]  # bars after entry
            if len(window) == 0:
                results[h].append(0.0)
                best_directions[h].append("flat")
                continue

            max_high = float(window["high"].max())
            min_low = float(window["low"].min())

            # Max favorable excursion in either direction
            up_move = (max_high - entry_price) / entry_price
            down_move = (entry_price - min_low) / entry_price
            max_abs = max(up_move, down_move)
            best_dir = "long" if up_move > down_move else "short"

            results[h].append(max_abs)
            best_directions[h].append(best_dir)

    return results, best_directions


def directionless_with_stop(df, indices, horizons=[20, 50], stop_pct=0.006):
    """
    Directionless breakout with a stop:
    - Enter at next open
    - Stop at entry * (1 - stop_pct) for long-side, entry * (1 + stop_pct) for short-side
    - Take the side that triggers first
    - Measure outcome in R where R = entry * stop_pct
    """
    results = {h: [] for h in horizons}
    taken_directions = {h: [] for h in horizons}

    for idx in indices:
        if idx + 1 >= len(df):
            continue
        entry_price = float(df.iloc[idx + 1]["open"])
        R = entry_price * stop_pct

        for h in horizons:
            window = df.iloc[idx + 2:idx + 2 + h]
            if len(window) == 0:
                results[h].append(0.0)
                taken_directions[h].append("none")
                continue

            # Check which direction triggers first
            long_stop = entry_price - R
            short_stop = entry_price + R
            long_tp = entry_price + R  # 1R target
            short_tp = entry_price - R

            long_triggered = False
            short_triggered = False
            outcome = 0.0
            taken = "none"

            for _, bar in window.iterrows():
                high = float(bar["high"])
                low = float(bar["low"])

                if not long_triggered and not short_triggered:
                    # Neither triggered yet — check if either breaks
                    if low <= long_stop:
                        long_triggered = True
                        outcome = -1.0  # stopped out
                        taken = "long_stopped"
                        break
                    elif high >= short_stop:
                        short_triggered = True
                        outcome = -1.0
                        taken = "short_stopped"
                        break
                    elif high >= long_tp:
                        long_triggered = True
                        outcome = 1.0
                        taken = "long_win"
                        break
                    elif low <= short_tp:
                        short_triggered = True
                        outcome = 1.0
                        taken = "short_win"
                        break

            if taken == "none":
                # Neither triggered — measure current P&L
                last_close = float(window.iloc[-1]["close"])
                long_pnl = (last_close - entry_price) / R
                short_pnl = (entry_price - last_close) / R
                if abs(long_pnl) > abs(short_pnl):
                    outcome = long_pnl
                    taken = "long_held"
                else:
                    outcome = short_pnl
                    taken = "short_held"

            results[h].append(outcome)
            taken_directions[h].append(taken)

    return results, taken_directions


def monthly_breakdown(results_df, label=""):
    """Show Exp1R by month."""
    months = sorted(results_df["month"].unique())
    print(f"\n    {'Month':<10s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'Exp2R':>8s}")
    print(f"    {'─'*10} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")
    for m in months:
        sub = results_df[results_df["month"] == m]
        metrics = compute_group_metrics(sub)
        if metrics:
            print(f"    {m:<10s} {metrics['count']:>4d} {metrics['pct_hit_1R']:>6.1%} {metrics['expectancy_1R']:>+7.3f}R {metrics['expectancy_2R']:>+7.3f}R")


def drawdown_analysis(results_df, label=""):
    """Simulate equity curve in R and measure drawdown."""
    # Sort by entry time
    sorted_df = results_df.sort_values("entry_time").reset_index(drop=True)

    # Compute cumulative R
    r_per_trade = []
    for _, row in sorted_df.iterrows():
        if row["hit_1R"] and not row["sl_hit"]:
            r_per_trade.append(1.0)
        elif row["hit_2R"] and row["sl_hit"]:
            # hit 1R then stopped later — net depends on timing
            r_per_trade.append(1.0)  # simplified: 1R captured
        elif row["sl_hit"] and not row["hit_1R"]:
            r_per_trade.append(-1.0)
        elif row["sl_hit"] and row["hit_1R"]:
            # hit 1R first then stopped — net = +1R -1R = 0R? Actually depends on R tracking
            # In this system, if hit_1R=True and sl_hit=True, 1R was hit on earlier candle
            r_per_trade.append(1.0 - 1.0)  # net 0
        else:
            r_per_trade.append(0.0)  # expired

    equity = np.cumsum(r_per_trade)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    total_r = equity[-1] if len(equity) > 0 else 0
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    avg_r = np.mean(r_per_trade) if r_per_trade else 0

    # Count consecutive losses
    max_consec_loss = 0
    current_consec = 0
    for r in r_per_trade:
        if r < 0:
            current_consec += 1
            max_consec_loss = max(max_consec_loss, current_consec)
        else:
            current_consec = 0

    print(f"\n    {label} EQUITY ANALYSIS:")
    print(f"      Total R:       {total_r:+.1f}R")
    print(f"      Trades:        {len(r_per_trade)}")
    print(f"      Avg R/trade:   {avg_r:+.3f}R")
    print(f"      Max drawdown:  {max_dd:.1f}R")
    print(f"      Max consec loss: {max_consec_loss}")

    # Rolling 10-trade window
    if len(r_per_trade) >= 10:
        rolling_10 = pd.Series(r_per_trade).rolling(10).sum()
        worst_10 = rolling_10.min()
        best_10 = rolling_10.max()
        print(f"      Worst 10-trade window: {worst_10:+.1f}R")
        print(f"      Best 10-trade window:  {best_10:+.1f}R")

    return r_per_trade


def main():
    start_time = datetime.now()
    df_train, df_test, all_results = load_btc()

    print("=" * 100)
    print("  LONG-ONLY & DIRECTIONLESS EXPANSION TEST — BTCUSDT")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)

    # ═══════════════════════════════════════════════════════
    # TEST 1: LONG-ONLY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print(f"  TEST 1: LONG-ONLY")
    print(f"{'═' * 100}")

    long_all = all_results[all_results["dir_label"] == "LONG"]
    long_train = long_all[long_all["period"] == "TRAIN"]
    long_test = long_all[long_all["period"] == "TEST"]
    short_all = all_results[all_results["dir_label"] == "SHORT"]
    short_test = short_all[short_all["period"] == "TEST"]

    m_long_all = compute_group_metrics(long_all)
    m_long_test = compute_group_metrics(long_test)
    m_short_test = compute_group_metrics(short_test)
    m_all_test = compute_group_metrics(all_results[all_results["period"] == "TEST"])

    print(f"\n  {'Category':<25s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s} {'MedMAE':>7s}")
    print(f"  {'─'*25} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")

    for label, m in [
        ("ALL (all periods)", compute_group_metrics(all_results)),
        ("ALL (test)", m_all_test),
        ("LONG (all periods)", m_long_all),
        ("LONG (test)", m_long_test),
        ("SHORT (test)", m_short_test),
    ]:
        if m:
            print(f"  {label:<25s} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R {m['median_MAE_R']:>6.2f}R")

    # LONG by confidence
    print(f"\n  LONG BY CONFIDENCE (all periods):")
    for mode in ["MILD", "MID", "HIGH", "PREMIUM"]:
        sub = long_all[long_all["confidence_mode"] == mode]
        m = compute_group_metrics(sub)
        if m:
            sub_test = long_test[long_test["confidence_mode"] == mode]
            m_t = compute_group_metrics(sub_test) if len(sub_test) > 0 else None
            test_str = f" | TEST: {m_t['count']}, Exp1R={m_t['expectancy_1R']:+.3f}" if m_t else ""
            print(f"    {mode:<10s}: {m['count']:>3d} setups | Exp1R={m['expectancy_1R']:+.3f} | Hit1R={m['pct_hit_1R']:.1%}{test_str}")

    # LONG monthly
    print(f"\n  LONG MONTHLY BREAKDOWN (all periods):")
    monthly_breakdown(long_all)

    # LONG drawdown
    print(f"\n  LONG DRAWDOWN:")
    r_long = drawdown_analysis(long_all, "LONG (all)")
    if len(long_test) > 0:
        r_long_test = drawdown_analysis(long_test, "LONG (test)")

    # ═══════════════════════════════════════════════════════
    # TEST 2: DIRECTIONLESS EXPANSION
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'═' * 100}")
    print(f"  TEST 2: DIRECTIONLESS EXPANSION")
    print(f"  Measures max |move| from entry within horizon, regardless of direction")
    print(f"{'═' * 100}")

    # Get validated setup indices for test period
    test_setups = all_results[all_results["period"] == "TEST"]
    test_times = set(test_setups["signal_time"].tolist())
    test_indices = df_test[df_test["timestamp"].isin(test_times)].index.tolist()

    # Random baseline
    rng = np.random.RandomState(42)
    valid_bars = list(range(500, len(df_test) - 55))
    random_indices = sorted(rng.choice(valid_bars, size=min(len(test_indices) * 5, len(valid_bars)), replace=False))

    horizons = [5, 10, 20, 50]
    setup_exp, setup_dirs = directionless_expansion(df_test, test_indices, horizons)
    random_exp, random_dirs = directionless_expansion(df_test, random_indices, horizons)

    print(f"\n  MAX ABSOLUTE RETURN FROM ENTRY (mean):")
    print(f"  {'Horizon':<12s} {'Setup':>10s} {'Random':>10s} {'Ratio':>8s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8}")
    for h in horizons:
        s = np.mean(setup_exp[h])
        r = np.mean(random_exp[h])
        ratio = s / r if r > 0 else 0
        print(f"  {h:>3d} bars     {s:>9.4f}  {r:>9.4f}  {ratio:>7.2f}x")

    print(f"\n  MAX ABSOLUTE RETURN FROM ENTRY (median):")
    print(f"  {'Horizon':<12s} {'Setup':>10s} {'Random':>10s} {'Ratio':>8s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8}")
    for h in horizons:
        s = np.median(setup_exp[h])
        r = np.median(random_exp[h])
        ratio = s / r if r > 0 else 0
        print(f"  {h:>3d} bars     {s:>9.4f}  {r:>9.4f}  {ratio:>7.2f}x")

    # Best direction chosen
    print(f"\n  BEST DIRECTION AT EACH HORIZON (setup timestamps):")
    for h in horizons:
        dirs = setup_dirs[h]
        n_long = sum(1 for d in dirs if d == "long")
        n_short = sum(1 for d in dirs if d == "short")
        n = len(dirs)
        print(f"    {h:>3d} bars: {n_long}/{n} long-optimal ({n_long/n:.1%}), {n_short}/{n} short-optimal ({n_short/n:.1%})")

    # ═══════════════════════════════════════════════════════
    # TEST 3: COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'═' * 100}")
    print(f"  TEST 3: COMPARISON — LONG-ONLY vs ORIGINAL vs RANDOM")
    print(f"{'═' * 100}")

    # Random LONG-only baseline
    rng2 = np.random.RandomState(99)
    random_long_indices = sorted(rng2.choice(valid_bars, size=len(test_indices) * 3, replace=False))
    random_long_results = []
    for idx in random_long_indices:
        if idx + 1 >= len(df_test):
            continue
        entry_price = float(df_test.iloc[idx + 1]["open"])
        rand_dir = 1  # force LONG
        stop_info = compute_structural_stop(df_test, idx, rand_dir, "RSI_SCALP")
        if not stop_info["is_stop_valid"]:
            continue
        tp_levels = compute_tp_levels(entry_price, stop_info["stop_price"], rand_dir)
        tp_levels["entry_price"] = entry_price
        from setup_validation_engine import track_setup_outcome
        outcome = track_setup_outcome(df_test, idx + 1, rand_dir, stop_info["stop_price"], tp_levels)
        if outcome.get("valid", False):
            r = {"direction": 1, "dir_label": "LONG"}
            r.update({k: v for k, v in outcome.items() if k != "valid"})
            r["month"] = "random"
            random_long_results.append(r)

    random_long_df = pd.DataFrame(random_long_results)
    m_random_long = compute_group_metrics(random_long_df) if len(random_long_df) > 0 else None

    print(f"\n  {'Category':<25s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s}")
    print(f"  {'─'*25} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7}")

    rows = [
        ("ALL signals (test)", m_all_test),
        ("LONG-only (test)", m_long_test),
        ("SHORT-only (test)", m_short_test),
        ("Random LONG", m_random_long),
    ]
    for label, m in rows:
        if m:
            print(f"  {label:<25s} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R")

    # Monthly comparison: LONG-only vs ALL
    print(f"\n  MONTHLY Exp1R COMPARISON (test period):")
    print(f"    {'Month':<10s} {'ALL':>8s} {'LONG':>8s} {'SHORT':>8s}")
    print(f"    {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    test_period = all_results[all_results["period"] == "TEST"]
    for month in sorted(test_period["month"].unique()):
        all_m = compute_group_metrics(test_period[test_period["month"] == month])
        long_m = compute_group_metrics(test_period[(test_period["month"] == month) & (test_period["dir_label"] == "LONG")])
        short_m = compute_group_metrics(test_period[(test_period["month"] == month) & (test_period["dir_label"] == "SHORT")])
        all_str = f"{all_m['expectancy_1R']:+.3f}" if all_m else "  -"
        long_str = f"{long_m['expectancy_1R']:+.3f}" if long_m else "  -"
        short_str = f"{short_m['expectancy_1R']:+.3f}" if short_m else "  -"
        print(f"    {month:<10s} {all_str:>8s} {long_str:>8s} {short_str:>8s}")

    # Drawdown comparison
    print(f"\n  DRAWDOWN COMPARISON:")
    r_all = drawdown_analysis(test_period, "ALL signals (test)")
    if len(long_test) > 0:
        r_long_only = drawdown_analysis(long_test, "LONG-only (test)")
    if len(short_test) > 0:
        r_short_only = drawdown_analysis(short_test, "SHORT-only (test)")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'═' * 100}")
    print(f"  COMPLETE — {elapsed:.1f}s")
    print(f"{'═' * 100}")


if __name__ == "__main__":
    main()
