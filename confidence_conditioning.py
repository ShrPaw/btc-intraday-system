"""
CONFIDENCE CONDITIONING LAYER — LONG SETUPS (BTCUSDT)
=====================================================
Maps where the edge lives by conditioning on HTF RSI, slope,
EMA distance, volatility, and trend strength.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF
)


def load_long_setups():
    """Load BTCUSDT LONG setups with HTF RSI data."""
    df = build_master_dataset("data/features/btcusdt_1m.csv")

    # We need HTF RSI and slope for ALL bars, not just shorts
    # The engine now stores h4_rsi_entry etc for all directions
    # But we also need slope and other features from the master dataset

    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)

    # Run validation to get setup-level results
    test_results = run_validation(df_test, "BTCUSDT")
    test_results["period"] = "TEST"
    train_results = run_validation(df_train, "BTCUSDT")
    train_results["period"] = "TRAIN"
    all_results = pd.concat([train_results, test_results], ignore_index=True)

    # Filter LONG only
    longs = all_results[all_results["direction"] == 1].copy()

    # Ensure boolean columns are bool type
    for col in ["hit_1R", "hit_2R", "hit_3R", "hit_4R", "sl_hit"]:
        if col in longs.columns:
            longs[col] = longs[col].astype(bool)

    # Merge additional features from the master dataset
    # For each setup, find the corresponding bar in the master dataset
    # to get slope, EMA distance, volatility, etc.

    # Build a lookup from the master dataset
    # The signal_time corresponds to a bar in df
    for period_label, period_df, period_results in [
        ("TRAIN", df_train, train_results),
        ("TEST", df_test, test_results),
    ]:
        long_mask = period_results["direction"] == 1
        if not long_mask.any():
            continue

        for idx, row in period_results[long_mask].iterrows():
            sig_time = row["signal_time"]
            bar_mask = period_df["timestamp"] == sig_time
            if not bar_mask.any():
                continue
            bar_idx = period_df.index[bar_mask][0]
            bar = period_df.iloc[bar_idx]

            # Store features
            for col in ["h4_rsi_slope_1", "h4_rsi_slope_2", "m15_rsi", "m15_rsi_slope_1",
                         "ema_dist_pct", "rv_6", "h4_adx", "ret_3"]:
                if col in bar.index:
                    longs.loc[idx, col] = float(bar[col]) if pd.notna(bar[col]) else np.nan

    return longs


def bin_and_measure(df, col, bins, labels, label_col=None):
    """Bin a column and measure metrics per bin."""
    if label_col is None:
        label_col = f"{col}_bin"

    df = df.copy()
    # Drop rows where col is NaN
    df = df[df[col].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df[label_col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

    results = []
    for label in labels:
        subset = df[df[label_col] == label].copy()
        # Re-ensure bool types after filtering
        for c in ["hit_1R", "hit_2R", "hit_3R", "hit_4R", "sl_hit"]:
            if c in subset.columns:
                subset[c] = subset[c].fillna(False).astype(bool)
        m = compute_group_metrics(subset)
        if m:
            results.append({
                "bin": label,
                "n": m["count"],
                "hit_1R": m["pct_hit_1R"],
                "hit_2R": m["pct_hit_2R"],
                "exp_1R": m["expectancy_1R"],
                "exp_2R": m["expectancy_2R"],
                "med_mfe": m["median_MFE_R"],
                "med_mae": m["median_MAE_R"],
            })
    return pd.DataFrame(results)


def print_table(title, df, sort_col="exp_1R"):
    """Print a formatted table."""
    if df is None or len(df) == 0:
        print(f"\n  {title}: No data")
        return

    df = df.sort_values(sort_col, ascending=False)
    print(f"\n  {title}")
    print(f"  {'Bin':<25s} {'N':>4s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s} {'MedMAE':>7s}")
    print(f"  {'─'*25} {'─'*4} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    for _, row in df.iterrows():
        print(f"  {row['bin']:<25s} {row['n']:>4d} {row['hit_1R']:>6.1%} {row['hit_2R']:>6.1%} {row['exp_1R']:>+7.3f}R {row['exp_2R']:>+7.3f}R {row['med_mfe']:>6.2f}R {row['med_mae']:>6.2f}R")


def condition_ranking(longs):
    """Rank all conditions by expectancy."""
    all_conditions = []

    # H4 RSI level
    t = bin_and_measure(longs, "h4_rsi_entry",
                        [0, 40, 50, 55, 60, 70, 100],
                        ["<40", "40-50", "50-55", "55-60", "60-70", ">70"])
    for _, r in t.iterrows():
        all_conditions.append({"condition": f"H4_RSI {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # H6 RSI level
    t = bin_and_measure(longs, "h6_rsi_entry",
                        [0, 40, 50, 55, 60, 70, 100],
                        ["<40", "40-50", "50-55", "55-60", "60-70", ">70"])
    for _, r in t.iterrows():
        all_conditions.append({"condition": f"H6_RSI {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # H12 RSI level
    t = bin_and_measure(longs, "h12_rsi_entry",
                        [0, 40, 50, 55, 60, 70, 100],
                        ["<40", "40-50", "50-55", "55-60", "60-70", ">70"])
    for _, r in t.iterrows():
        all_conditions.append({"condition": f"H12_RSI {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # H4 RSI slope
    if "h4_rsi_slope_1" in longs.columns:
        t = bin_and_measure(longs, "h4_rsi_slope_1",
                            [-100, -2, -0.5, 0, 0.5, 2, 100],
                            ["<-2", "-2 to -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 2", ">2"])
        for _, r in t.iterrows():
            all_conditions.append({"condition": f"H4_slope {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # EMA distance
    if "ema_dist_pct" in longs.columns:
        t = bin_and_measure(longs, "ema_dist_pct",
                            [-0.1, -0.01, -0.003, 0, 0.003, 0.01, 0.1],
                            ["<-1%", "-1% to -0.3%", "-0.3% to 0%", "0% to 0.3%", "0.3% to 1%", ">1%"])
        for _, r in t.iterrows():
            all_conditions.append({"condition": f"EMA_dist {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # Volatility (rv_6)
    if "rv_6" in longs.columns:
        t = bin_and_measure(longs, "rv_6",
                            [0, 0.001, 0.002, 0.003, 0.005, 0.1],
                            ["Low (<0.1%)", "0.1-0.2%", "0.2-0.3%", "0.3-0.5%", ">0.5%"])
        for _, r in t.iterrows():
            all_conditions.append({"condition": f"Volatility {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # ADX
    if "h4_adx" in longs.columns:
        t = bin_and_measure(longs, "h4_adx",
                            [0, 15, 20, 25, 30, 100],
                            ["<15 (weak)", "15-20", "20-25", "25-30", ">30 (strong)"])
        for _, r in t.iterrows():
            all_conditions.append({"condition": f"ADX {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # M15 RSI
    if "m15_rsi" in longs.columns:
        t = bin_and_measure(longs, "m15_rsi",
                            [0, 40, 50, 55, 60, 70, 100],
                            ["<40", "40-50", "50-55", "55-60", "60-70", ">70"])
        for _, r in t.iterrows():
            all_conditions.append({"condition": f"M15_RSI {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # Recent return (momentum)
    if "ret_3" in longs.columns:
        t = bin_and_measure(longs, "ret_3",
                            [-0.1, -0.005, -0.001, 0, 0.001, 0.005, 0.1],
                            ["<-0.5%", "-0.5% to -0.1%", "-0.1% to 0%", "0% to 0.1%", "0.1% to 0.5%", ">0.5%"])
        for _, r in t.iterrows():
            all_conditions.append({"condition": f"Recent return {r['bin']}", "exp_1R": r["exp_1R"], "n": r["n"], "hit_1R": r["hit_1R"]})

    # Sort by expectancy
    ranking = pd.DataFrame(all_conditions).sort_values("exp_1R", ascending=False)
    return ranking


def main():
    print("=" * 90)
    print("  CONFIDENCE CONDITIONING LAYER — LONG SETUPS (BTCUSDT)")
    print("  Mapping where the edge lives")
    print("=" * 90)

    longs = load_long_setups()
    print(f"\n  Total LONG setups: {len(longs)}")
    print(f"  With HTF RSI: {longs['h4_rsi_entry'].notna().sum()}")

    # ═══════════════════════════════════════════════════════
    # INDIVIDUAL CONDITION TABLES
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  INDIVIDUAL CONDITION TABLES")
    print(f"{'═' * 90}")

    # H4 RSI
    print_table("H4 RSI at entry",
                bin_and_measure(longs, "h4_rsi_entry",
                                [0, 40, 50, 55, 60, 70, 100],
                                ["<40", "40-50", "50-55", "55-60", "60-70", ">70"]))

    # H6 RSI
    print_table("H6 RSI at entry",
                bin_and_measure(longs, "h6_rsi_entry",
                                [0, 40, 50, 55, 60, 70, 100],
                                ["<40", "40-50", "50-55", "55-60", "60-70", ">70"]))

    # H12 RSI
    print_table("H12 RSI at entry",
                bin_and_measure(longs, "h12_rsi_entry",
                                [0, 40, 50, 55, 60, 70, 100],
                                ["<40", "40-50", "50-55", "55-60", "60-70", ">70"]))

    # H4 RSI slope
    if "h4_rsi_slope_1" in longs.columns:
        print_table("H4 RSI slope (1-bar change)",
                    bin_and_measure(longs, "h4_rsi_slope_1",
                                    [-100, -2, -0.5, 0, 0.5, 2, 100],
                                    ["<-2", "-2 to -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 2", ">2"]))

    # EMA distance
    if "ema_dist_pct" in longs.columns:
        print_table("Distance from EMA20",
                    bin_and_measure(longs, "ema_dist_pct",
                                    [-0.1, -0.01, -0.003, 0, 0.003, 0.01, 0.1],
                                    ["<-1%", "-1% to -0.3%", "-0.3% to 0%", "0% to 0.3%", "0.3% to 1%", ">1%"]))

    # Volatility
    if "rv_6" in longs.columns:
        print_table("Realized Volatility (6-bar)",
                    bin_and_measure(longs, "rv_6",
                                    [0, 0.001, 0.002, 0.003, 0.005, 0.1],
                                    ["Low (<0.1%)", "0.1-0.2%", "0.2-0.3%", "0.3-0.5%", ">0.5%"]))

    # ADX
    if "h4_adx" in longs.columns:
        print_table("ADX (trend strength)",
                    bin_and_measure(longs, "h4_adx",
                                    [0, 15, 20, 25, 30, 100],
                                    ["<15 (weak)", "15-20", "20-25", "25-30", ">30 (strong)"]))

    # M15 RSI
    if "m15_rsi" in longs.columns:
        print_table("M15 RSI at entry",
                    bin_and_measure(longs, "m15_rsi",
                                    [0, 40, 50, 55, 60, 70, 100],
                                    ["<40", "40-50", "50-55", "55-60", "60-70", ">70"]))

    # Recent return
    if "ret_3" in longs.columns:
        print_table("Recent 3-bar return",
                    bin_and_measure(longs, "ret_3",
                                    [-0.1, -0.005, -0.001, 0, 0.001, 0.005, 0.1],
                                    ["<-0.5%", "-0.5% to -0.1%", "-0.1% to 0%", "0% to 0.1%", "0.1% to 0.5%", ">0.5%"]))

    # ═══════════════════════════════════════════════════════
    # COMBINED CONDITIONS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  COMBINED CONDITIONS (H4 RSI + Slope)")
    print(f"{'═' * 90}")

    if "h4_rsi_slope_1" in longs.columns:
        longs["h4_rsi_bin"] = pd.cut(longs["h4_rsi_entry"],
                                      bins=[0, 50, 55, 60, 100],
                                      labels=["<50", "50-55", "55-60", ">60"])
        longs["h4_slope_bin"] = pd.cut(longs["h4_rsi_slope_1"],
                                        bins=[-100, -0.5, 0, 0.5, 100],
                                        labels=["falling", "flat-down", "flat-up", "rising"])

        print(f"\n  {'H4 RSI + Slope':<35s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'MedMFE':>7s}")
        print(f"  {'─'*35} {'─'*4} {'─'*7} {'─'*8} {'─'*7}")
        for rsi_bin in ["<50", "50-55", "55-60", ">60"]:
            for slope_bin in ["falling", "flat-down", "flat-up", "rising"]:
                sub = longs[(longs["h4_rsi_bin"] == rsi_bin) & (longs["h4_slope_bin"] == slope_bin)]
                m = compute_group_metrics(sub)
                if m and m["count"] >= 2:
                    print(f"  H4_RSI={rsi_bin:<6s} + {slope_bin:<10s} {m['count']:>4d} {m['pct_hit_1R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R")

    # ═══════════════════════════════════════════════════════
    # COMBINED: H4 RSI + EMA distance
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  COMBINED CONDITIONS (H4 RSI + EMA distance)")
    print(f"{'═' * 90}")

    if "ema_dist_pct" in longs.columns:
        longs["ema_bin"] = pd.cut(longs["ema_dist_pct"],
                                   bins=[-0.1, -0.003, 0, 0.003, 0.1],
                                   labels=["below-EMA", "at-EMA-down", "at-EMA-up", "above-EMA"])

        print(f"\n  {'H4 RSI + EMA dist':<35s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'MedMFE':>7s}")
        print(f"  {'─'*35} {'─'*4} {'─'*7} {'─'*8} {'─'*7}")
        for rsi_bin in ["<50", "50-55", "55-60", ">60"]:
            for ema_bin in ["below-EMA", "at-EMA-down", "at-EMA-up", "above-EMA"]:
                sub = longs[(longs["h4_rsi_bin"] == rsi_bin) & (longs["ema_bin"] == ema_bin)]
                m = compute_group_metrics(sub)
                if m and m["count"] >= 2:
                    print(f"  H4_RSI={rsi_bin:<6s} + {ema_bin:<12s} {m['count']:>4d} {m['pct_hit_1R']:>6.1%} {m['expectancy_1R']:>+7.3f}R {m['median_MFE_R']:>6.2f}R")

    # ═══════════════════════════════════════════════════════
    # FULL RANKING
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  CONDITION RANKING BY EXPECTANCY (top 30)")
    print(f"{'═' * 90}")

    ranking = condition_ranking(longs)
    print(f"\n  {'Rank':<5s} {'Condition':<35s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s}")
    print(f"  {'─'*5} {'─'*35} {'─'*4} {'─'*7} {'─'*8}")
    for i, (_, row) in enumerate(ranking.head(30).iterrows()):
        marker = "✅" if row["exp_1R"] > 0.1 and row["n"] >= 5 else "⚠️" if row["exp_1R"] > 0 else "❌"
        print(f"  {i+1:>4d}  {row['condition']:<35s} {row['n']:>4d} {row['hit_1R']:>6.1%} {row['exp_1R']:>+7.3f}R {marker}")

    # Bottom 10
    print(f"\n  WORST CONDITIONS:")
    print(f"  {'Rank':<5s} {'Condition':<35s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s}")
    print(f"  {'─'*5} {'─'*35} {'─'*4} {'─'*7} {'─'*8}")
    for i, (_, row) in enumerate(ranking.tail(10).iterrows()):
        print(f"  {len(ranking)-9+i:>4d}  {row['condition']:<35s} {row['n']:>4d} {row['hit_1R']:>6.1%} {row['exp_1R']:>+7.3f}R ❌")

    # Summary
    print(f"\n{'═' * 90}")
    print(f"  SUMMARY")
    print(f"{'═' * 90}")
    profitable = ranking[ranking["exp_1R"] > 0]
    losing = ranking[ranking["exp_1R"] < 0]
    strong = ranking[(ranking["exp_1R"] > 0.1) & (ranking["n"] >= 5)]
    print(f"  Total conditions tested: {len(ranking)}")
    print(f"  Profitable (Exp1R > 0): {len(profitable)}")
    print(f"  Losing (Exp1R < 0): {len(losing)}")
    print(f"  Strong (Exp1R > 0.1, N>=5): {len(strong)}")
    if len(strong) > 0:
        print(f"\n  STRONGEST CONDITIONS:")
        for _, row in strong.head(10).iterrows():
            print(f"    {row['condition']:<35s} | N={row['n']:>3d} | Exp1R={row['exp_1R']:+.3f}R | Hit1R={row['hit_1R']:.1%}")


if __name__ == "__main__":
    main()
