"""
LONG-ONLY + HTF REGIME FILTER — BTCUSDT
========================================
Tests: only trade LONG when HTF is bullish.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF
)


def equity_curve(results_df):
    """Fixed 1R risk per trade. No compounding."""
    records = results_df.sort_values("entry_time").reset_index(drop=True)
    trades = []
    for _, row in records.iterrows():
        if row["sl_hit"] and not row["hit_1R"]:
            pnl = -1.0
        elif row["hit_1R"] and row["sl_hit"]:
            pnl = 0.0
        elif row["hit_1R"] and not row["sl_hit"]:
            pnl = 1.0
            if row.get("hit_2R", False) and not row.get("hit_3R", False):
                pnl = 2.0
            elif row.get("hit_3R", False):
                pnl = 3.0
        else:
            pnl = 0.0
        trades.append({
            "time": row["entry_time"],
            "month": row["month"],
            "regime": row.get("htf_regime", "unknown"),
            "pnl_R": pnl,
            "hit_1R": row["hit_1R"],
        })
    df = pd.DataFrame(trades)
    df["cumulative_R"] = df["pnl_R"].cumsum()
    df["peak_R"] = df["cumulative_R"].cummax()
    df["drawdown_R"] = df["cumulative_R"] - df["peak_R"]
    return df


def print_comparison(label_a, eq_a, m_a, label_b, eq_b, m_b):
    """Side-by-side comparison."""
    print(f"\n  {'─' * 80}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"  {'─' * 80}")
    print(f"  {'Metric':<25s} {label_a:>20s} {label_b:>20s}")
    print(f"  {'─'*25} {'─'*20} {'─'*20}")

    for metric, getter in [
        ("Trades", lambda e, m: f"{m['count']}"),
        ("Hit 1R", lambda e, m: f"{m['pct_hit_1R']:.1%}"),
        ("Hit 2R", lambda e, m: f"{m['pct_hit_2R']:.1%}"),
        ("Exp 1R", lambda e, m: f"{m['expectancy_1R']:+.3f}R"),
        ("Exp 2R", lambda e, m: f"{m['expectancy_2R']:+.3f}R"),
        ("Median MFE", lambda e, m: f"{m['median_MFE_R']:.2f}R"),
        ("Total R", lambda e, m: f"{e['pnl_R'].sum():+.1f}R"),
        ("Max drawdown", lambda e, m: f"{e['drawdown_R'].min():.1f}R"),
        ("Max consec loss", lambda e, m: str(max(0, max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(e['pnl_R'], lambda x: x < 0) if k), default=0)))),
        ("Avg R/trade", lambda e, m: f"{e['pnl_R'].mean():+.3f}R"),
    ]:
        try:
            va = getter(eq_a, m_a)
            vb = getter(eq_b, m_b)
        except:
            va = vb = "-"
        print(f"  {metric:<25s} {va:>20s} {vb:>20s}")


def monthly_table(eq, label=""):
    """Monthly breakdown."""
    print(f"\n    {label} MONTHLY:")
    print(f"    {'Month':<10s} {'N':>4s} {'Hit1R':>7s} {'TotalR':>8s} {'CumulR':>8s}")
    print(f"    {'─'*10} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")
    cumul = 0
    for month in sorted(eq["month"].unique()):
        m = eq[eq["month"] == month]
        n = len(m)
        h = m["hit_1R"].sum()
        t = m["pnl_R"].sum()
        cumul += t
        print(f"    {month:<10s} {n:>4d} {h/n:>6.1%} {t:>+7.1f}R {cumul:>+7.1f}R")


def max_consecutive_loss(eq):
    """Count max consecutive losses."""
    max_c = 0
    cur = 0
    for p in eq["pnl_R"]:
        if p < 0:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def main():
    print("=" * 80)
    print("  LONG-ONLY + HTF REGIME FILTER — BTCUSDT")
    print("  Fixed 1R risk per trade. No compounding.")
    print("=" * 80)

    # Build and validate
    df = build_master_dataset("data/features/btcusdt_1m.csv")
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)
    all_results = run_validation(df_test, "BTCUSDT")
    all_results["period"] = "TEST"

    # LONG only
    longs = all_results[all_results["direction"] == 1].copy()
    print(f"\n  LONG setups (test period): {len(longs)}")

    # Filter variants
    # 1. All LONGs (no filter)
    # 2. Bullish regime (htf_regime == "bullish", i.e., H4 RSI > 55)
    # 3. Strict bullish: H4 > 50 AND H6 > 50 AND H12 > 50
    # 4. Very strict: H4 > 55 AND H6 > 50 AND H12 > 50
    # 5. RSI + slope: H4 > 50 AND H4 slope positive

    filters = {
        "LONG (all)": longs.copy(),
        "LONG/bullish": longs[longs["htf_regime"] == "bullish"].copy(),
        "LONG/H4>50 & H6>50 & H12>50": longs[
            (longs["h4_rsi_entry"] > 50) &
            (longs["h6_rsi_entry"] > 50) &
            (longs["h12_rsi_entry"] > 50)
        ].copy(),
        "LONG/H4>55 & H6>50 & H12>50": longs[
            (longs["h4_rsi_entry"] > 55) &
            (longs["h6_rsi_entry"] > 50) &
            (longs["h12_rsi_entry"] > 50)
        ].copy(),
        "LONG/H4>50 & H6>50": longs[
            (longs["h4_rsi_entry"] > 50) &
            (longs["h6_rsi_entry"] > 50)
        ].copy(),
    }

    # Also test with bearish excluded (same as bullish-only since no bearish longs in test)
    # But add: what if we exclude neutral?
    filters["LONG/not-neutral"] = longs[longs["htf_regime"] != "neutral"].copy()

    # Results table
    print(f"\n  {'Filter':<35s} {'N':>4s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'TotalR':>8s} {'MaxDD':>7s} {'MaxCL':>6s} {'MedMFE':>7s}")
    print(f"  {'─'*35} {'─'*4} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*6} {'─'*7}")

    for label, subset in filters.items():
        if len(subset) == 0:
            print(f"  {label:<35s} {'0':>4s}")
            continue
        m = compute_group_metrics(subset)
        eq = equity_curve(subset)
        total_r = eq["pnl_R"].sum()
        max_dd = eq["drawdown_R"].min()
        max_cl = max_consecutive_loss(eq)
        print(f"  {label:<35s} {m['count']:>4d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['expectancy_2R']:>+7.3f}R {total_r:>+7.1f}R {max_dd:>6.1f}R {max_cl:>6d} {m['median_MFE_R']:>6.2f}R")

    # Detailed: best filter vs baseline
    # The best filter is likely LONG/bullish or LONG/H4>50 & H6>50
    best_label = max(filters.keys(), key=lambda k: compute_group_metrics(filters[k])["expectancy_1R"] if len(filters[k]) > 0 else -999)
    best = filters[best_label]
    baseline = filters["LONG (all)"]

    m_base = compute_group_metrics(baseline)
    m_best = compute_group_metrics(best)
    eq_base = equity_curve(baseline)
    eq_best = equity_curve(best)

    print_comparison("LONG (all)", eq_base, m_base, best_label, eq_best, m_best)

    # Monthly for best
    monthly_table(eq_best, best_label)

    # Monthly for baseline
    monthly_table(eq_base, "LONG (all)")

    # Regime breakdown for best
    if len(best) > 0:
        print(f"\n    {best_label} REGIME BREAKDOWN:")
        for regime in ["bullish", "bearish", "neutral"]:
            r = best[best["htf_regime"] == regime]
            if len(r) > 0:
                m = compute_group_metrics(r)
                print(f"      {regime:<10s}: {m['count']} setups | Hit1R={m['pct_hit_1R']:.1%} | Exp1R={m['expectancy_1R']:+.3f}")

    # Also show: what happens to LONGs in neutral/bearish?
    print(f"\n  LONG IN NON-BULLISH REGIME:")
    non_bull = longs[longs["htf_regime"] != "bullish"]
    if len(non_bull) > 0:
        m = compute_group_metrics(non_bull)
        eq = equity_curve(non_bull)
        print(f"    {m['count']} setups | Hit1R={m['pct_hit_1R']:.1%} | Exp1R={m['expectancy_1R']:+.3f} | TotalR={eq['pnl_R'].sum():+.1f}R")

    neutral = longs[longs["htf_regime"] == "neutral"]
    if len(neutral) > 0:
        m = compute_group_metrics(neutral)
        eq = equity_curve(neutral)
        print(f"    Neutral only: {m['count']} setups | Hit1R={m['pct_hit_1R']:.1%} | Exp1R={m['expectancy_1R']:+.3f} | TotalR={eq['pnl_R'].sum():+.1f}R")

    print(f"\n{'═' * 80}")
    print(f"  VERDICT")
    print(f"{'═' * 80}")
    print(f"  Baseline LONG (all):   {m_base['count']} setups | Exp1R={m_base['expectancy_1R']:+.3f}R | TotalR={eq_base['pnl_R'].sum():+.1f}R | MaxDD={eq_base['drawdown_R'].min():.1f}R")
    if m_best:
        print(f"  Best filter ({best_label}): {m_best['count']} setups | Exp1R={m_best['expectancy_1R']:+.3f}R | TotalR={eq_best['pnl_R'].sum():+.1f}R | MaxDD={eq_best['drawdown_R'].min():.1f}R")
        if m_best['expectancy_1R'] > m_base['expectancy_1R'] and m_best['count'] >= 5:
            print(f"\n  ✅ REGIME FILTER IMPROVES: +{m_best['expectancy_1R'] - m_base['expectancy_1R']:.3f}R Exp1R improvement")
        elif m_best['count'] < 5:
            print(f"\n  ⚠️  TOO FEW TRADES: {m_best['count']} setups — not statistically meaningful")
        else:
            print(f"\n  ❌ NO IMPROVEMENT: Filter does not add value")


if __name__ == "__main__":
    main()
