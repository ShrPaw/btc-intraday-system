"""
LONG-ONLY FINAL VALIDATION — BTCUSDT
=====================================
Fixed 1R risk per trade. No compounding. No shorts.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF
)


def equity_curve(results_df):
    """Fixed 1R risk per trade. Win = +1R or +2R etc. Loss = -1R."""
    records = results_df.sort_values("entry_time").reset_index(drop=True)
    trades = []
    for _, row in records.iterrows():
        if row["sl_hit"] and not row["hit_1R"]:
            pnl = -1.0
        elif row["hit_1R"] and row["sl_hit"]:
            # Hit 1R on earlier candle, then stopped — net 0
            pnl = 0.0
        elif row["hit_1R"] and not row["sl_hit"]:
            pnl = 1.0
            if row.get("hit_2R", False) and not row.get("hit_3R", False):
                pnl = 2.0
            elif row.get("hit_3R", False) and not row.get("hit_4R", False):
                pnl = 3.0
            elif row.get("hit_4R", False):
                pnl = 4.0
        else:
            pnl = 0.0  # expired
        trades.append({
            "time": row["entry_time"],
            "month": row["month"],
            "regime": row.get("htf_regime", "unknown"),
            "session": row.get("session", "unknown"),
            "confidence": row["confidence_mode"],
            "pnl_R": pnl,
            "hit_1R": row["hit_1R"],
            "hit_2R": row.get("hit_2R", False),
            "hit_3R": row.get("hit_3R", False),
            "sl_hit": row["sl_hit"],
            "mfe": row["max_favorable_excursion_R"],
            "mae": row["max_adverse_excursion_R"],
        })
    df = pd.DataFrame(trades)
    df["cumulative_R"] = df["pnl_R"].cumsum()
    df["peak_R"] = df["cumulative_R"].cummax()
    df["drawdown_R"] = df["cumulative_R"] - df["peak_R"]
    return df


def report(label, eq):
    """Print full report for an equity curve."""
    n = len(eq)
    if n == 0:
        print(f"\n  {label}: No trades")
        return

    wins = eq[eq["pnl_R"] > 0]
    losses = eq[eq["pnl_R"] < 0]
    breakeven = eq[eq["pnl_R"] == 0]

    total_r = eq["pnl_R"].sum()
    avg_r = eq["pnl_R"].mean()
    hit_1r = eq["hit_1R"].sum()
    hit_2r = eq["hit_2R"].sum() if "hit_2R" in eq.columns else 0
    sl = eq["sl_hit"].sum()

    max_dd = eq["drawdown_R"].min()
    max_consec = 0
    cur_consec = 0
    for p in eq["pnl_R"]:
        if p < 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    print(f"\n  {'─' * 70}")
    print(f"  {label}")
    print(f"  {'─' * 70}")
    print(f"    Trades:        {n}")
    print(f"    Wins (>0R):    {len(wins)} ({len(wins)/n:.1%})")
    print(f"    Losses (-1R):  {len(losses)} ({len(losses)/n:.1%})")
    print(f"    Breakeven (0R):{len(breakeven)} ({len(breakeven)/n:.1%})")
    print(f"    Hit 1R:        {hit_1r} ({hit_1r/n:.1%})")
    print(f"    Hit 2R:        {hit_2r} ({hit_2r/n:.1%})")
    print(f"    SL hit:        {sl} ({sl/n:.1%})")
    print(f"    Total R:       {total_r:+.1f}R")
    print(f"    Avg R/trade:   {avg_r:+.3f}R")
    print(f"    Max drawdown:  {max_dd:.1f}R")
    print(f"    Max consec loss: {max_consec}")

    # Rolling windows
    if n >= 10:
        roll10 = eq["pnl_R"].rolling(10).sum()
        print(f"    Worst 10-trade: {roll10.min():+.1f}R")
        print(f"    Best 10-trade:  {roll10.max():+.1f}R")
    if n >= 5:
        roll5 = eq["pnl_R"].rolling(5).sum()
        print(f"    Worst 5-trade:  {roll5.min():+.1f}R")
        print(f"    Best 5-trade:   {roll5.max():+.1f}R")

    # MFE/MAE
    print(f"    Median MFE:    {eq['mfe'].median():.2f}R")
    print(f"    Median MAE:    {eq['mae'].median():.2f}R")

    # Monthly breakdown
    print(f"\n    MONTHLY:")
    print(f"    {'Month':<10s} {'N':>4s} {'Wins':>5s} {'Losses':>6s} {'TotalR':>8s} {'AvgR':>7s} {'CumulR':>8s}")
    print(f"    {'─'*10} {'─'*4} {'─'*5} {'─'*6} {'─'*8} {'─'*7} {'─'*8}")
    cumul = 0
    for month in sorted(eq["month"].unique()):
        m = eq[eq["month"] == month]
        n_m = len(m)
        w = (m["pnl_R"] > 0).sum()
        l = (m["pnl_R"] < 0).sum()
        t = m["pnl_R"].sum()
        a = m["pnl_R"].mean()
        cumul += t
        print(f"    {month:<10s} {n_m:>4d} {w:>5d} {l:>6d} {t:>+7.1f}R {a:>+6.3f}R {cumul:>+7.1f}R")

    # Regime breakdown
    if "regime" in eq.columns:
        print(f"\n    REGIME:")
        print(f"    {'Regime':<12s} {'N':>4s} {'Hit1R':>7s} {'TotalR':>8s} {'AvgR':>7s}")
        print(f"    {'─'*12} {'─'*4} {'─'*7} {'─'*8} {'─'*7}")
        for regime in ["bullish", "bearish", "neutral"]:
            r = eq[eq["regime"] == regime]
            if len(r) > 0:
                print(f"    {regime:<12s} {len(r):>4d} {r['hit_1R'].sum()/len(r):>6.1%} {r['pnl_R'].sum():>+7.1f}R {r['pnl_R'].mean():>+6.3f}R")

    # Session breakdown
    if "session" in eq.columns:
        print(f"\n    SESSION:")
        print(f"    {'Session':<12s} {'N':>4s} {'Hit1R':>7s} {'TotalR':>8s} {'AvgR':>7s}")
        print(f"    {'─'*12} {'─'*4} {'─'*7} {'─'*8} {'─'*7}")
        for session in ["Asian", "European", "US"]:
            s = eq[eq["session"] == session]
            if len(s) > 0:
                print(f"    {session:<12s} {len(s):>4d} {s['hit_1R'].sum()/len(s):>6.1%} {s['pnl_R'].sum():>+7.1f}R {s['pnl_R'].mean():>+6.3f}R")

    # Confidence breakdown
    print(f"\n    CONFIDENCE:")
    print(f"    {'Tier':<12s} {'N':>4s} {'Hit1R':>7s} {'TotalR':>8s} {'AvgR':>7s}")
    print(f"    {'─'*12} {'─'*4} {'─'*7} {'─'*8} {'─'*7}")
    for conf in ["MILD", "MID", "HIGH", "PREMIUM", "ELITE"]:
        c = eq[eq["confidence"] == conf]
        if len(c) > 0:
            print(f"    {conf:<12s} {len(c):>4d} {c['hit_1R'].sum()/len(c):>6.1%} {c['pnl_R'].sum():>+7.1f}R {c['pnl_R'].mean():>+6.3f}R")


def main():
    print("=" * 80)
    print("  LONG-ONLY FINAL VALIDATION — BTCUSDT")
    print("  Fixed 1R risk per trade. No compounding. No shorts.")
    print("=" * 80)

    df = build_master_dataset("data/features/btcusdt_1m.csv")
    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

    # Run validation (includes all the fixes: next-open, worst-case, rolling)
    train_results = run_validation(df_train, "BTCUSDT")
    test_results = run_validation(df_test, "BTCUSDT")

    # Filter LONG only
    train_long = train_results[train_results["direction"] == 1].copy()
    test_long = test_results[test_results["direction"] == 1].copy()
    all_long = pd.concat([train_long, test_long], ignore_index=True)

    # Add period label
    train_long["period"] = "TRAIN"
    test_long["period"] = "TEST"
    all_labeled = pd.concat([train_long, test_long], ignore_index=True)

    print(f"\n  LONG-ONLY SETUPS: {len(train_long)} train + {len(test_long)} test = {len(all_long)} total")

    # Standard R metrics
    m_all = compute_group_metrics(all_long)
    m_train = compute_group_metrics(train_long)
    m_test = compute_group_metrics(test_long)

    print(f"\n  R-MULTIPLE METRICS:")
    print(f"  {'Period':<10s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Hit3R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'Exp3R':>8s} {'MedMFE':>7s} {'MedMAE':>7s}")
    print(f"  {'─'*10} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    for label, m in [("ALL", m_all), ("TRAIN", m_train), ("TEST", m_test)]:
        if m:
            print(f"  {label:<10s} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['pct_hit_3R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {m['expectancy_3R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R {m['median_MAE_R']:>6.2f}R")

    # Equity curves (fixed 1R risk)
    eq_all = equity_curve(all_labeled)
    eq_train = equity_curve(train_long)
    eq_test = equity_curve(test_long)

    report("ALL PERIODS — LONG-ONLY (fixed 1R risk)", eq_all)
    report("TRAIN PERIOD — LONG-ONLY (fixed 1R risk)", eq_train)
    report("TEST PERIOD — LONG-ONLY (fixed 1R risk)", eq_test)

    # Final summary
    print(f"\n{'═' * 80}")
    print(f"  FINAL VERDICT — LONG-ONLY BTCUSDT")
    print(f"{'═' * 80}")
    if m_test:
        print(f"  Test period: {m_test['count']} setups")
        print(f"  Exp1R = {m_test['expectancy_1R']:+.3f}R")
        print(f"  Hit1R = {m_test['pct_hit_1R']:.1%}")
        print(f"  Max drawdown = {eq_test['drawdown_R'].min():.1f}R")
        print(f"  Total R (test) = {eq_test['pnl_R'].sum():+.1f}R")
        if m_test['expectancy_1R'] > 0.1 and m_test['pct_hit_1R'] > 0.55:
            print(f"\n  ✅ LONG-ONLY VALIDATED: Positive expectancy with manageable drawdown.")
        elif m_test['expectancy_1R'] > 0:
            print(f"\n  ⚠️  LONG-ONLY PROMISING: Positive but thin edge. Needs more data.")
        else:
            print(f"\n  ❌ LONG-ONLY NOT VALIDATED: Negative or zero expectancy.")


if __name__ == "__main__":
    main()
