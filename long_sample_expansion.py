"""
LONG SAMPLE EXPANSION — MULTI-SYMBOL
=====================================
Expands LONG-only validation across BTC, ETH, SOL, XRP to get
more data points. Each symbol validated independently.

Rules:
  - BTC is primary headline
  - Other symbols are robustness checks only
  - Never aggregate into one headline metric
  - R-multiples, worst-case intracandle, closed candles
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF, SYMBOLS
)


def equity_curve_fixed(results_df):
    """Fixed 1R risk per trade."""
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
            elif row.get("hit_3R", False) and not row.get("hit_4R", False):
                pnl = 3.0
            elif row.get("hit_4R", False):
                pnl = 4.0
        else:
            pnl = 0.0
        trades.append({"pnl_R": pnl, "month": row["month"], "hit_1R": row["hit_1R"],
                        "hit_2R": row.get("hit_2R", False), "sl_hit": row["sl_hit"],
                        "mfe": row["max_favorable_excursion_R"], "mae": row["max_adverse_excursion_R"]})
    df = pd.DataFrame(trades)
    df["cumulative_R"] = df["pnl_R"].cumsum()
    df["peak_R"] = df["cumulative_R"].cummax()
    df["drawdown_R"] = df["cumulative_R"] - df["peak_R"]
    return df


def report_compact(label, eq, m):
    """Compact report."""
    n = len(eq)
    if n == 0:
        print(f"    {label}: No trades")
        return
    wins = (eq["pnl_R"] > 0).sum()
    losses = (eq["pnl_R"] < 0).sum()
    max_dd = eq["drawdown_R"].min()
    max_consec = 0
    cur = 0
    for p in eq["pnl_R"]:
        if p < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    print(f"    {label}: {n} trades | Wins {wins} ({wins/n:.0%}) | Losses {losses} | Total {eq['pnl_R'].sum():+.1f}R | Avg {eq['pnl_R'].mean():+.3f}R | MaxDD {max_dd:.1f}R | MaxConsecLoss {max_consec}")


def main():
    print("=" * 80)
    print("  LONG SAMPLE EXPANSION — MULTI-SYMBOL")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    symbol_data = {}

    for symbol in SYMBOLS:
        data_path = f"data/features/{symbol.lower()}_1m.csv"
        print(f"\n{'─' * 60}")
        print(f"  Processing {symbol}")
        print(f"{'─' * 60}")

        df = build_master_dataset(data_path)
        df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
        df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

        train_results = run_validation(df_train, symbol)
        test_results = run_validation(df_test, symbol)
        train_results["period"] = "TRAIN"
        test_results["period"] = "TEST"

        all_results = pd.concat([train_results, test_results], ignore_index=True)
        longs = all_results[all_results["direction"] == 1].copy()
        longs_train = longs[longs["period"] == "TRAIN"]
        longs_test = longs[longs["period"] == "TEST"]

        label = "PRIMARY" if symbol == "BTCUSDT" else "ROBUSTNESS"
        print(f"  [{label}]")
        print(f"    Train: {len(longs_train)} LONG setups")
        print(f"    Test:  {len(longs_test)} LONG setups")

        m_train = compute_group_metrics(longs_train)
        m_test = compute_group_metrics(longs_test)
        m_all = compute_group_metrics(longs)

        if m_train:
            print(f"    Train: Hit1R={m_train['pct_hit_1R']:.1%} | Exp1R={m_train['expectancy_1R']:+.3f} | Total={longs_train.apply(lambda r: -1.0 if (r['sl_hit'] and not r['hit_1R']) else (1.0 if r['hit_1R'] else 0.0), axis=1).sum():+.1f}R")
        if m_test:
            print(f"    Test:  Hit1R={m_test['pct_hit_1R']:.1%} | Exp1R={m_test['expectancy_1R']:+.3f} | Total={longs_test.apply(lambda r: -1.0 if (r['sl_hit'] and not r['hit_1R']) else (1.0 if r['hit_1R'] else 0.0), axis=1).sum():+.1f}R")
        if m_all:
            print(f"    All:   Hit1R={m_all['pct_hit_1R']:.1%} | Exp1R={m_all['expectancy_1R']:+.3f}")

        # Monthly breakdown
        if len(longs_test) > 0:
            print(f"\n    MONTHLY (test):")
            for month in sorted(longs_test["month"].unique()):
                m_long = longs_test[longs_test["month"] == month]
                mm = compute_group_metrics(m_long)
                if mm:
                    eq = equity_curve_fixed(m_long)
                    print(f"      {month}: {mm['count']:>2d} setups | Hit1R={mm['pct_hit_1R']:.0%} | Exp1R={mm['expectancy_1R']:+.3f} | Total={eq['pnl_R'].sum():+.1f}R")

        # H4 RSI zone breakdown
        print(f"\n    H4 RSI ZONE (test):")
        for zone_name, lo, hi in [("50-55", 50, 55), ("55-60", 55, 60), ("60-70", 60, 70), ("70+", 70, 100)]:
            zone_longs = longs_test[(longs_test["h4_rsi_entry"] >= lo) & (longs_test["h4_rsi_entry"] < hi)]
            if len(zone_longs) > 0:
                zm = compute_group_metrics(zone_longs)
                if zm:
                    print(f"      RSI {zone_name}: {zm['count']:>2d} setups | Hit1R={zm['pct_hit_1R']:.0%} | Exp1R={zm['expectancy_1R']:+.3f}")

        symbol_data[symbol] = {"train": longs_train, "test": longs_test, "all": longs}

    # ═══════════════════════════════════════════════════════
    # CROSS-SYMBOL SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'═' * 80}")
    print(f"  CROSS-SYMBOL SUMMARY — LONG ONLY (TEST PERIOD)")
    print(f"{'═' * 80}")
    print(f"  {'Symbol':<12s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'TotalR':>8s} {'MaxDD':>7s}")
    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7}")

    total_n = 0
    total_r = 0
    for symbol in SYMBOLS:
        if symbol in symbol_data:
            test_longs = symbol_data[symbol]["test"]
            m = compute_group_metrics(test_longs)
            if m:
                eq = equity_curve_fixed(test_longs)
                label = "PRIMARY" if symbol == "BTCUSDT" else ""
                print(f"  {symbol:<12s} {m['count']:>5d} {m['pct_hit_1R']:>6.1%} {m['pct_hit_2R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {eq['pnl_R'].sum():>+7.1f}R {eq['drawdown_R'].min():>6.1f}R {label}")
                total_n += m["count"]
                total_r += eq["pnl_R"].sum()

    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7}")
    print(f"  {'TOTAL':<12s} {total_n:>5d} {'':>7s} {'':>7s} {'':>8s} {total_r:>+7.1f}R {'':>7s} (robustness only)")

    # Combined ALL symbols test period (labeled as robustness)
    all_test_longs = pd.concat([symbol_data[s]["test"] for s in SYMBOLS if s in symbol_data], ignore_index=True)
    m_combined = compute_group_metrics(all_test_longs)
    if m_combined:
        eq_combined = equity_curve_fixed(all_test_longs)
        print(f"\n  COMBINED (all 4 symbols, test period) — ROBUSTNESS METRIC, NOT HEADLINE:")
        print(f"    {m_combined['count']} setups | Hit1R={m_combined['pct_hit_1R']:.1%} | Exp1R={m_combined['expectancy_1R']:+.3f} | Total={eq_combined['pnl_R'].sum():+.1f}R | MaxDD={eq_combined['drawdown_R'].min():.1f}R")

    # Train/test stability per symbol
    print(f"\n\n{'═' * 80}")
    print(f"  TRAIN/TEST STABILITY — LONG ONLY")
    print(f"{'═' * 80}")
    print(f"  {'Symbol':<12s} {'Train_N':>7s} {'Train_Hit':>10s} {'Train_Exp':>10s} {'Test_N':>7s} {'Test_Hit':>10s} {'Test_Exp':>10s} {'Stable?':>8s}")
    print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*10} {'─'*7} {'─'*10} {'─'*10} {'─'*8}")

    for symbol in SYMBOLS:
        if symbol not in symbol_data:
            continue
        sd = symbol_data[symbol]
        mt = compute_group_metrics(sd["train"])
        mv = compute_group_metrics(sd["test"])
        if mt and mv:
            stable = "YES" if (mt["expectancy_1R"] > 0 and mv["expectancy_1R"] > 0) else ("TEST+TRAIN-" if mt["expectancy_1R"] < 0 and mv["expectancy_1R"] > 0 else "NO")
            print(f"  {symbol:<12s} {mt['count']:>7d} {mt['pct_hit_1R']:>9.1%} {mt['expectancy_1R']:>+9.3f}R {mv['count']:>7d} {mv['pct_hit_1R']:>9.1%} {mv['expectancy_1R']:>+9.3f}R {stable:>8s}")

    print(f"\n{'═' * 80}")
    print(f"  LONG SAMPLE EXPANSION COMPLETE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
