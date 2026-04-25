"""
VOLATILITY EXPANSION TEST
=========================
Ignores direction. Measures whether setups predict absolute move magnitude.
If setup timestamps predict bigger moves than random → timing edge is real.
If not → the entire setup detection adds nothing.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def load_and_build(symbol="BTCUSDT"):
    from setup_validation_engine import build_master_dataset, run_validation, TRAIN_CUTOFF
    data_path = f"data/features/{symbol.lower()}_1m.csv"
    df = build_master_dataset(data_path)
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)
    test_results = run_validation(df_test, symbol)
    test_results["period"] = "TEST"
    return df_test, test_results


def measure_absolute_moves(df, indices, horizons=[5, 10, 20, 50, 100]):
    """
    For each index, measure |return| over next N bars.
    Returns dict of horizon → list of absolute returns.
    """
    results = {h: [] for h in horizons}
    entry_prices = []
    directions = []

    for idx in indices:
        if idx + 1 >= len(df):
            continue
        entry_price = float(df.iloc[idx + 1]["open"])  # next candle open
        entry_prices.append(entry_price)

        for h in horizons:
            end_idx = min(idx + 1 + h, len(df) - 1)
            # Measure from entry open to close of horizon bar
            end_price = float(df.iloc[end_idx]["close"])
            abs_ret = abs(end_price - entry_price) / entry_price
            results[h].append(abs_ret)

    return results, entry_prices


def measure_signed_moves(df, indices, horizons=[5, 10, 20, 50, 100]):
    """
    For each index, measure signed return over next N bars.
    Returns dict of horizon → list of signed returns.
    """
    results = {h: [] for h in horizons}

    for idx in indices:
        if idx + 1 >= len(df):
            continue
        entry_price = float(df.iloc[idx + 1]["open"])

        for h in horizons:
            end_idx = min(idx + 1 + h, len(df) - 1)
            end_price = float(df.iloc[end_idx]["close"])
            signed_ret = (end_price - entry_price) / entry_price
            results[h].append(signed_ret)

    return results


def compute_R_distance(df, indices):
    """
    For each setup, compute R_abs = |entry - stop| as a fraction of entry.
    Then measure how often the absolute move exceeds 1R, 2R, 3R.
    """
    from setup_validation_engine import compute_structural_stop

    r_fractions = []
    for idx in indices:
        if idx + 1 >= len(df):
            continue
        entry_price = float(df.iloc[idx + 1]["open"])
        row = df.iloc[idx]

        # Get stop for the ACTUAL direction
        direction = int(row["direction"]) if "direction" in df.columns else 1
        stop_info = compute_structural_stop(df, idx, direction, str(row.get("setup_type", "RSI_SCALP")))
        if not stop_info["is_stop_valid"]:
            continue
        r_abs = abs(entry_price - stop_info["stop_price"])
        r_frac = r_abs / entry_price
        r_fractions.append(r_frac)

    return r_fractions


def run_test(df, results, horizons=[5, 10, 20, 50, 100]):
    """Run the full volatility expansion test."""

    # Setup indices
    setup_mask = (df["setup_type"] != "none") & (df["take_trade"] if "take_trade" in df.columns else True)
    setup_indices = df[setup_mask].index.tolist()

    # Validated setup indices (those that passed all filters)
    validated_times = set(results["signal_time"].tolist())
    validated_indices = df[df["timestamp"].isin(validated_times)].index.tolist()

    # Random baseline: same number of timestamps, from all valid bars
    rng = np.random.RandomState(42)
    min_start = 500
    valid_bars = list(range(min_start, len(df) - max(horizons) - 1))
    n_validated = len(validated_indices)
    random_indices = sorted(rng.choice(valid_bars, size=min(n_validated * 5, len(valid_bars)), replace=False))

    # Also random from setup bars (restricted universe)
    setup_only_indices = sorted(rng.choice(
        [i for i in setup_indices if i >= min_start and i < len(df) - max(horizons) - 1],
        size=min(n_validated * 3, len(setup_indices)),
        replace=False
    ))

    print(f"\n  Sample sizes:")
    print(f"    Validated setups:    {len(validated_indices)}")
    print(f"    All setup bars:      {len(setup_indices)}")
    print(f"    Random (full data):  {len(random_indices)}")
    print(f"    Random (setup bars): {len(setup_only_indices)}")

    # Measure absolute moves
    val_abs, _ = measure_absolute_moves(df, validated_indices, horizons)
    all_abs, _ = measure_absolute_moves(df, setup_indices[:len(validated_indices)*3], horizons)
    rand_abs, _ = measure_absolute_moves(df, random_indices, horizons)
    setup_rand_abs, _ = measure_absolute_moves(df, setup_only_indices, horizons)

    # Measure signed moves
    val_signed = measure_signed_moves(df, validated_indices, horizons)
    rand_signed = measure_signed_moves(df, random_indices, horizons)

    # R distances for validated setups
    r_fracs = compute_R_distance(df, validated_indices)

    # ─────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────
    print(f"\n\n{'═' * 100}")
    print(f"  VOLATILITY EXPANSION TEST — BTCUSDT")
    print(f"  Do setups predict bigger absolute moves than random?")
    print(f"{'═' * 100}")

    # Table: mean |return| by horizon
    print(f"\n  MEAN ABSOLUTE RETURN BY HORIZON (5m bars after entry):")
    print(f"  {'Horizon':<12s} {'Validated':>10s} {'AllSetup':>10s} {'Random':>10s} {'RandSetup':>10s} {'Edge':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for h in horizons:
        v = np.mean(val_abs[h]) if val_abs[h] else 0
        a = np.mean(all_abs[h]) if all_abs[h] else 0
        r = np.mean(rand_abs[h]) if rand_abs[h] else 0
        s = np.mean(setup_rand_abs[h]) if setup_rand_abs[h] else 0
        edge = v - r
        print(f"  {h:>3d} bars     {v:>9.4f}  {a:>9.4f}  {r:>9.4f}  {s:>9.4f}  {edge:>+9.4f}")

    # Table: median |return| by horizon
    print(f"\n  MEDIAN ABSOLUTE RETURN BY HORIZON:")
    print(f"  {'Horizon':<12s} {'Validated':>10s} {'Random':>10s} {'Edge':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for h in horizons:
        v = np.median(val_abs[h]) if val_abs[h] else 0
        r = np.median(rand_abs[h]) if rand_abs[h] else 0
        edge = v - r
        print(f"  {h:>3d} bars     {v:>9.4f}  {r:>9.4f}  {edge:>+9.4f}")

    # Probability of expansion > threshold
    if r_fracs:
        median_r_frac = np.median(r_fracs)
        mean_r_frac = np.mean(r_fracs)
        print(f"\n  R-DISTANCE (structural stop as % of entry):")
        print(f"    Mean:   {mean_r_frac*100:.3f}%")
        print(f"    Median: {median_r_frac*100:.3f}%")

        # For each horizon, what % of moves exceed 1R, 2R, 3R?
        print(f"\n  PROBABILITY OF MOVE > N*R (absolute, validated setups):")
        print(f"  {'Horizon':<12s} {'P(>0.5R)':>10s} {'P(>1R)':>10s} {'P(>2R)':>10s} {'P(>3R)':>10s}")
        print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for h in horizons:
            if not val_abs[h]:
                continue
            abs_rets = np.array(val_abs[h])
            p_half = np.mean(abs_rets > median_r_frac * 0.5)
            p_1r = np.mean(abs_rets > median_r_frac)
            p_2r = np.mean(abs_rets > median_r_frac * 2)
            p_3r = np.mean(abs_rets > median_r_frac * 3)
            print(f"  {h:>3d} bars     {p_half:>9.1%}  {p_1r:>9.1%}  {p_2r:>9.1%}  {p_3r:>9.1%}")

        # Same for random
        print(f"\n  PROBABILITY OF MOVE > N*R (absolute, random timestamps):")
        print(f"  {'Horizon':<12s} {'P(>0.5R)':>10s} {'P(>1R)':>10s} {'P(>2R)':>10s} {'P(>3R)':>10s}")
        print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for h in horizons:
            if not rand_abs[h]:
                continue
            abs_rets = np.array(rand_abs[h])
            p_half = np.mean(abs_rets > median_r_frac * 0.5)
            p_1r = np.mean(abs_rets > median_r_frac)
            p_2r = np.mean(abs_rets > median_r_frac * 2)
            p_3r = np.mean(abs_rets > median_r_frac * 3)
            print(f"  {h:>3d} bars     {p_half:>9.1%}  {p_1r:>9.1%}  {p_2r:>9.1%}  {p_3r:>9.1%}")

    # Expansion vs contraction
    print(f"\n  EXPANSION vs CONTRACTION (% of moves where |move| > median |move| at same horizon):")
    print(f"  Using median |return| at 20 bars as expansion threshold")
    if val_abs[20]:
        median_move_20 = np.median(val_abs[20])
        rand_median_20 = np.median(rand_abs[20]) if rand_abs[20] else median_move_20

        for h in horizons:
            if not val_abs[h] or not rand_abs[h]:
                continue
            val_exp = np.mean(np.array(val_abs[h]) > median_move_20)
            rand_exp = np.mean(np.array(rand_abs[h]) > median_move_20)
            print(f"    {h:>3d} bars: Setup expansion rate={val_exp:.1%}, Random={rand_exp:.1%}, Delta={val_exp-rand_exp:+.1%}")

    # Direction accuracy (signed)
    print(f"\n  DIRECTION ACCURACY (signed returns, validated setups):")
    print(f"  {'Horizon':<12s} {'Mean signed':>12s} {'P(positive)':>12s} {'P(negative)':>12s}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for h in horizons:
        if not val_signed[h]:
            continue
        signed = np.array(val_signed[h])
        mean_s = np.mean(signed)
        p_pos = np.mean(signed > 0)
        p_neg = np.mean(signed < 0)
        print(f"  {h:>3d} bars     {mean_s:>+11.4f}  {p_pos:>11.1%}  {p_neg:>11.1%}")

    print(f"\n  DIRECTION ACCURACY (random):")
    for h in horizons:
        if not rand_signed[h]:
            continue
        signed = np.array(rand_signed[h])
        p_pos = np.mean(signed > 0)
        p_neg = np.mean(signed < 0)
        print(f"  {h:>3d} bars: P(positive)={p_pos:.1%}, P(negative)={p_neg:.1%}")

    # Direction accuracy by setup direction
    print(f"\n  DIRECTION ACCURACY BY SETUP DIRECTION:")
    for direction_label in ["LONG", "SHORT"]:
        dir_results = results[results["dir_label"] == direction_label]
        dir_times = set(dir_results["signal_time"].tolist())
        dir_indices = df[df["timestamp"].isin(dir_times)].index.tolist()
        dir_signed = measure_signed_moves(df, dir_indices, [5, 10, 20, 50])

        print(f"\n    {direction_label} ({len(dir_indices)} setups):")
        for h in [5, 10, 20, 50]:
            if not dir_signed[h]:
                continue
            signed = np.array(dir_signed[h])
            if direction_label == "LONG":
                correct = np.mean(signed > 0)
            else:
                correct = np.mean(signed < 0)
            print(f"      {h:>3d} bars: mean={np.mean(signed):+.4f}, correct direction={correct:.1%}")

    # Verdict
    print(f"\n{'═' * 100}")
    print(f"  VERDICT")
    print(f"{'═' * 100}")

    if val_abs[20] and rand_abs[20]:
        val_mean = np.mean(val_abs[20])
        rand_mean = np.mean(rand_abs[20])
        ratio = val_mean / rand_mean if rand_mean > 0 else 0

        print(f"\n  Mean |return| at 20 bars:")
        print(f"    Validated setups: {val_mean:.4f}")
        print(f"    Random:           {rand_mean:.4f}")
        print(f"    Ratio:            {ratio:.3f}x")

        if ratio > 1.1:
            print(f"\n  ✅ TIMING EDGE CONFIRMED: Setups predict {ratio:.1f}x larger moves than random.")
            print(f"     The setup detection captures volatility expansion events.")
            print(f"     Direction logic is the weak link — timing is real, calls are wrong.")
        elif ratio > 1.0:
            print(f"\n  ⚠️  MARGINAL TIMING EDGE: Setups predict {ratio:.2f}x moves vs random.")
            print(f"     Weak evidence of timing advantage.")
        else:
            print(f"\n  ❌ NO TIMING EDGE: Setups predict {ratio:.2f}x moves vs random.")
            print(f"     Setup timestamps do not predict volatility expansion.")
            print(f"     The setup detection itself adds no value.")


if __name__ == "__main__":
    df, results = load_and_build("BTCUSDT")
    run_test(df, results)
