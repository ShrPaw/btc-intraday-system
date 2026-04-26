"""
REGIME DETECTION LAYER — BTCUSDT LONG
======================================
Determines WHEN the market allows the LONG setup to produce positive R.

We do NOT improve the setup. We only detect regime state.
All features are past-available at signal time (no future leakage).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF
)


# ═══════════════════════════════════════════════════════════
# STEP 1 — LABEL MARKET BEHAVIOR (post-hoc, for training)
# ═══════════════════════════════════════════════════════════

def label_regime(results):
    """
    Label each setup as 'trend' or 'chop' based on outcome.
    This is POST-HOC labeling — used only to identify what good
    regimes look like, NOT for live detection.
    """
    results = results.copy()

    def classify(row):
        mfe = row.get("max_favorable_excursion_R", 0)
        mae = row.get("max_adverse_excursion_R", 0)

        if mfe >= 1.5 and mfe > mae:
            return "trend"
        elif mfe < 1.0 and mae >= mfe:
            return "chop"
        elif mfe >= 1.0 and mfe > mae:
            return "mild_trend"
        else:
            return "neutral"

    results["regime_label"] = results.apply(classify, axis=1)
    return results


# ═══════════════════════════════════════════════════════════
# STEP 2 — BUILD REGIME FEATURES (past-only)
# ═══════════════════════════════════════════════════════════

def compute_regime_features(df):
    """
    Compute regime detection features on the 5m dataframe.
    ALL features use only past data (rolling windows, no lookahead).
    """
    df = df.copy()

    # ─── Volatility ───
    ret_1 = df["close"].pct_change(1)

    # Realized vol: short (20 bars) vs long (100 bars)
    rv_short = ret_1.rolling(20, min_periods=10).std()
    rv_long = ret_1.rolling(100, min_periods=50).std()
    df["rv_short"] = rv_short
    df["rv_long"] = rv_long
    df["vol_expansion"] = (rv_short / rv_long.replace(0, np.nan)).clip(0, 5)

    # Volatility percentile (rolling)
    df["vol_pctile"] = rv_short.rolling(500, min_periods=100).rank(pct=True).shift(1)

    # ─── Trend: multi-timeframe returns ───
    # 5m return (20 bars ≈ 100 min)
    df["ret_5m_20"] = df["close"].pct_change(20)
    # 15m return equivalent (60 bars of 5m ≈ 5 hours)
    df["ret_15m"] = df["close"].pct_change(60)
    # 1h return equivalent (120 bars of 5m ≈ 10 hours)
    df["ret_1h"] = df["close"].pct_change(120)

    # ─── Trend: price slope ───
    # Rolling linear regression slope over 20 bars
    def rolling_slope(series, window=20):
        """Simple rolling slope via least squares."""
        slopes = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values
            x = np.arange(window, dtype=float)
            if np.isnan(y).any():
                continue
            slope = np.polyfit(x, y, 1)[0]
            slopes.iloc[i] = slope
        return slopes

    # Use a vectorized approximation instead of loop
    # slope ≈ (close - close.shift(window)) / window normalized by price
    df["price_slope_20"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20) / 20
    df["price_slope_60"] = (df["close"] - df["close"].shift(60)) / df["close"].shift(60) / 60

    # ─── Trend: EMA distances ───
    if "ema20" not in df.columns:
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    if "ema50" not in df.columns:
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    # EMA200
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["dist_ema50"] = (df["close"] - df["ema50"]) / df["ema50"]
    df["dist_ema200"] = (df["close"] - df["ema200"]) / df["ema200"]

    # EMA alignment: are EMAs stacked bullish? (20 > 50 > 200)
    df["ema_bullish_stack"] = (
        (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    ).astype(float)

    # ─── Persistence: directional consistency ───
    # % of last 20 bars where close > open
    df["bull_candle_pct_20"] = (df["close"] > df["open"]).rolling(20, min_periods=10).mean()
    df["bull_candle_pct_60"] = (df["close"] > df["open"]).rolling(60, min_periods=30).mean()

    # Consecutive direction: how many of last 5 bars moved in same direction
    ret_sign = (df["close"].diff() > 0).astype(float)
    df["consec_bull_5"] = ret_sign.rolling(5, min_periods=3).sum() / 5

    # ─── Structure: range and breakout ───
    # Range compression: current 20-bar range vs 100-bar range
    range_20 = df["high"].rolling(20).max() - df["low"].rolling(20).min()
    range_100 = df["high"].rolling(100).max() - df["low"].rolling(100).min()
    df["range_compression"] = (range_20 / range_100.replace(0, np.nan)).clip(0, 2)

    # Breakout frequency: how many new highs in last 20 bars
    rolling_high = df["high"].rolling(20).max()
    df["new_high_count"] = (df["high"] >= rolling_high.shift(1)).rolling(20, min_periods=10).sum()
    rolling_low = df["low"].rolling(20).min()
    df["new_low_count"] = (df["low"] <= rolling_low.shift(1)).rolling(20, min_periods=10).sum()
    df["breakout_balance"] = df["new_high_count"] - df["new_low_count"]

    # ─── H4 RSI slope (already in dataset but let's compute fresh) ───
    # Already have h4_rsi_slope_1 from engine

    return df


# ═══════════════════════════════════════════════════════════
# STEP 3 — LINK REGIME TO PERFORMANCE
# ═══════════════════════════════════════════════════════════

def analyze_feature_predictiveness(results, df, feature_name, bins, labels):
    """
    For a given feature, bin the setups and compute performance per bin.
    Uses the feature value at signal time (already on df, merged to results).
    """
    # Merge feature from df to results via signal_time
    merged = results.copy()
    if feature_name not in merged.columns:
        # Need to merge from df
        feature_lookup = df.set_index("timestamp")[feature_name]
        merged[feature_name] = merged["signal_time"].map(feature_lookup)

    merged = merged[merged[feature_name].notna()].copy()
    if len(merged) == 0:
        return None

    merged["feature_bin"] = pd.cut(merged[feature_name], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for label in labels:
        subset = merged[merged["feature_bin"] == label]
        if len(subset) == 0:
            continue

        mfe = subset["max_favorable_excursion_R"]
        mae = subset["max_adverse_excursion_R"]

        # R per trade
        r_list = []
        for _, row in subset.iterrows():
            if row["sl_hit"] and not row["hit_1R"]:
                r = -1.0
            elif row["hit_1R"] and row["sl_hit"]:
                r = 0.0
            elif row["hit_1R"] and not row["sl_hit"]:
                r = min(4.0, 1.0 + sum([row.get(f"hit_{i}R", False) for i in [2,3,4]]))
            else:
                r = 0.0
            r_list.append(r)
        r_arr = np.array(r_list)

        rows.append({
            "bin": label,
            "N": len(subset),
            "Hit1R": subset["hit_1R"].astype(bool).mean(),
            "Exp1R": r_arr.mean(),
            "TotalR": r_arr.sum(),
            "MedMFE": mfe.median(),
            "MedMAE": mae.median(),
            "MFE_gt_MAE": (mfe > mae).mean(),
        })

    return pd.DataFrame(rows)


def print_feature_table(feature_name, df_results):
    """Print feature analysis table."""
    if df_results is None or len(df_results) == 0:
        print(f"    {feature_name}: no data")
        return
    print(f"\n    {feature_name}:")
    print(f"    {'Bin':<18s} {'N':>5s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s} {'MedMFE':>7s} {'MedMAE':>7s} {'MFE>MAE':>8s}")
    print(f"    {'─'*18} {'─'*5} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*8}")
    for _, row in df_results.iterrows():
        print(f"    {str(row['bin']):<18s} {row['N']:>5.0f} {row['Hit1R']:>6.1%} {row['Exp1R']:>+7.3f}R {row['TotalR']:>+7.1f}R {row['MedMFE']:>6.2f}R {row['MedMAE']:>6.2f}R {row['MFE_gt_MAE']:>7.1%}")


# ═══════════════════════════════════════════════════════════
# STEP 4 — BUILD REGIME FILTER
# ═══════════════════════════════════════════════════════════

def build_regime_filter(results_train, df_train):
    """
    Build a simple rule-based regime filter from training data.
    Finds which feature combinations correlate with positive Exp1R.
    """
    print(f"\n    Building regime filter from train data...")

    # Merge features
    merged = results_train.copy()
    for feat in ["vol_expansion", "vol_pctile", "ret_5m_20", "ret_1h",
                 "price_slope_20", "dist_ema20", "dist_ema50",
                 "bull_candle_pct_20", "ema_bullish_stack",
                 "range_compression", "breakout_balance",
                 "h4_rsi_slope_1"]:
        if feat in df_train.columns:
            feature_lookup = df_train.set_index("timestamp")[feat]
            merged[feat] = merged["signal_time"].map(feature_lookup)

    longs = merged[merged["direction"] == 1].copy()

    if len(longs) == 0:
        return None

    # Test simple rules on train data
    rules = []

    # Rule 1: vol_expansion > median
    vol_med = longs["vol_expansion"].median()
    if pd.notna(vol_med):
        mask = longs["vol_expansion"] > vol_med
        m = compute_r_metrics(longs[mask])
        rules.append(("vol_expansion > median", mask, m))

    # Rule 2: bull_candle_pct > 0.55
    if "bull_candle_pct_20" in longs.columns:
        mask = longs["bull_candle_pct_20"] > 0.55
        m = compute_r_metrics(longs[mask])
        rules.append(("bull_pct_20 > 55%", mask, m))

    # Rule 3: ema_bullish_stack == 1
    if "ema_bullish_stack" in longs.columns:
        mask = longs["ema_bullish_stack"] > 0.5
        m = compute_r_metrics(longs[mask])
        rules.append(("EMA bullish stack", mask, m))

    # Rule 4: dist_ema20 > 0 (price above EMA20)
    if "dist_ema20" in longs.columns:
        mask = longs["dist_ema20"] > 0
        m = compute_r_metrics(longs[mask])
        rules.append(("price > EMA20", mask, m))

    # Rule 5: breakout_balance > 0 (more new highs than lows)
    if "breakout_balance" in longs.columns:
        mask = longs["breakout_balance"] > 0
        m = compute_r_metrics(longs[mask])
        rules.append(("breakout_balance > 0", mask, m))

    # Rule 6: ret_1h > 0 (positive 1h return)
    if "ret_1h" in longs.columns:
        mask = longs["ret_1h"] > 0
        m = compute_r_metrics(longs[mask])
        rules.append(("ret_1h > 0", mask, m))

    # Rule 7: h4_rsi_slope > 0
    if "h4_rsi_slope_1" in longs.columns:
        mask = longs["h4_rsi_slope_1"] > 0
        m = compute_r_metrics(longs[mask])
        rules.append(("H4 RSI slope > 0", mask, m))

    # Rule 8: vol_expansion > 1.0 (expanding vol)
    if "vol_expansion" in longs.columns:
        mask = longs["vol_expansion"] > 1.0
        m = compute_r_metrics(longs[mask])
        rules.append(("vol_expansion > 1.0", mask, m))

    # Print all rules
    print(f"\n    {'Rule':<30s} {'N':>5s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"    {'─'*30} {'─'*5} {'─'*7} {'─'*8} {'─'*8}")
    print(f"    {'ALL LONGS':<30s} {len(longs):>5d} {longs['hit_1R'].astype(bool).mean():>6.1%} ", end="")
    r_all = compute_r_arr(longs)
    print(f"{r_all.mean():>+7.3f}R {r_all.sum():>+7.1f}R")

    best_rule = None
    best_exp = -999
    for name, mask, m in rules:
        if m:
            print(f"    {name:<30s} {m['N']:>5d} {m['Hit1R']:>6.1%} {m['Exp1R']:>+7.3f}R {m['TotalR']:>+7.1f}R")
            if m["Exp1R"] > best_exp and m["N"] >= 3:
                best_exp = m["Exp1R"]
                best_rule = name

    # Try combining the best 2 rules
    print(f"\n    COMBINED RULES:")
    for name1, mask1, m1 in rules:
        if m1 and m1["Exp1R"] > 0:
            for name2, mask2, m2 in rules:
                if m2 and m2["Exp1R"] > 0 and name1 != name2:
                    combined = mask1 & mask2
                    m_comb = compute_r_metrics(longs[combined])
                    if m_comb and m_comb["N"] >= 3:
                        print(f"    {name1} + {name2}: N={m_comb['N']}, Hit1R={m_comb['Hit1R']:.0%}, Exp1R={m_comb['Exp1R']:+.3f}, TotalR={m_comb['TotalR']:+.1f}R")

    return best_rule


def compute_r_metrics(subset):
    """Compute R metrics for a subset."""
    if len(subset) == 0:
        return None
    r_arr = compute_r_arr(subset)
    return {
        "N": len(subset),
        "Hit1R": subset["hit_1R"].astype(bool).mean(),
        "Exp1R": r_arr.mean(),
        "TotalR": r_arr.sum(),
    }


def compute_r_arr(subset):
    """Compute R per trade array."""
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


# ═══════════════════════════════════════════════════════════
# STEP 5 — VALIDATE FILTER
# ═══════════════════════════════════════════════════════════

def validate_filter(results, df, filter_fn, label):
    """Run validation with and without filter."""
    longs = results[results["direction"] == 1].copy()

    # Merge features
    for feat in ["vol_expansion", "vol_pctile", "ret_5m_20", "ret_1h",
                 "price_slope_20", "dist_ema20", "dist_ema50",
                 "bull_candle_pct_20", "ema_bullish_stack",
                 "range_compression", "breakout_balance",
                 "h4_rsi_slope_1"]:
        if feat in df.columns:
            feature_lookup = df.set_index("timestamp")[feat]
            longs[feat] = longs["signal_time"].map(feature_lookup)

    # Apply filter
    mask = longs.apply(filter_fn, axis=1)
    filtered = longs[mask]

    print(f"\n    {label}:")
    print(f"    {'Case':<15s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'TotalR':>8s} {'MaxDD':>7s} {'MaxCL':>6s}")
    print(f"    {'─'*15} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*6}")

    for case_label, case_df in [("ALL LONG", longs), ("FILTERED", filtered)]:
        if len(case_df) == 0:
            print(f"    {case_label:<15s} —")
            continue
        m = compute_full_metrics(case_df)
        if m:
            print(f"    {case_label:<15s} {m['N']:>5d} {m['Hit1R']:>6.1%} {m['Hit2R']:>6.1%} {m['Exp1R']:>+7.3f}R {m['TotalR']:>+7.1f}R {m['MaxDD']:>6.1f}R {m['MaxConsec']:>5d}")

    return longs, filtered


def compute_full_metrics(subset):
    """Compute full metrics including drawdown and losing streak."""
    if len(subset) == 0:
        return None
    r_arr = compute_r_arr(subset)
    cum_r = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cum_r)
    dd = cum_r - peak

    max_consec = 0
    cur = 0
    for r in r_arr:
        if r < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    mfe = subset["max_favorable_excursion_R"]
    mae = subset["max_adverse_excursion_R"]

    return {
        "N": len(subset),
        "Hit1R": subset["hit_1R"].astype(bool).mean(),
        "Hit2R": subset.get("hit_2R", pd.Series(False, index=subset.index)).astype(bool).mean(),
        "Exp1R": r_arr.mean(),
        "TotalR": r_arr.sum(),
        "MaxDD": dd.min(),
        "MaxConsec": max_consec,
        "MedMFE": mfe.median(),
        "MedMAE": mae.median(),
    }


# ═══════════════════════════════════════════════════════════
# STEP 6 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════

def walk_forward_filter(all_results, df_all):
    """Build filter on train, apply to test, check if it generalizes."""
    print(f"\n    WALK-FORWARD FILTER VALIDATION")
    print(f"    {'─' * 60}")

    train = all_results[all_results["period"] == "TRAIN"]
    test = all_results[all_results["period"] == "TEST"]

    # Merge features for both periods
    for period_label, period_results, period_df in [("TRAIN", train, df_all), ("TEST", test, df_all)]:
        for feat in ["vol_expansion", "bull_candle_pct_20", "ema_bullish_stack",
                     "dist_ema20", "breakout_balance", "ret_1h", "h4_rsi_slope_1"]:
            if feat in period_df.columns:
                feature_lookup = period_df.set_index("timestamp")[feat]
                period_results[feat] = period_results["signal_time"].map(feature_lookup)

    # Find best simple rule on train
    train_longs = train[train["direction"] == 1].copy()
    test_longs = test[test["direction"] == 1].copy()

    if len(train_longs) == 0 or len(test_longs) == 0:
        print(f"    Insufficient data for walk-forward")
        return

    # Test each feature threshold on train, apply to test
    print(f"\n    {'Rule (built on train)':<35s} {'Train_N':>7s} {'Train_Exp':>10s} {'Test_N':>7s} {'Test_Exp':>10s} {'Verdict'}")
    print(f"    {'─'*35} {'─'*7} {'─'*10} {'─'*7} {'─'*10} {'─'*15}")

    # Baseline
    r_train_all = compute_r_arr(train_longs)
    r_test_all = compute_r_arr(test_longs)
    print(f"    {'ALL LONG (no filter)':<35s} {len(train_longs):>7d} {r_train_all.mean():>+9.3f}R {len(test_longs):>7d} {r_test_all.mean():>+9.3f}R")

    best_filters = []

    for feat in ["vol_expansion", "bull_candle_pct_20", "ema_bullish_stack",
                 "dist_ema20", "breakout_balance", "ret_1h"]:
        if feat not in train_longs.columns:
            continue

        train_feat = train_longs[feat].dropna()
        if len(train_feat) < 5:
            continue

        # Try different thresholds
        for pct in [0.25, 0.50, 0.75]:
            threshold = train_feat.quantile(pct)

            train_mask = train_longs[feat] > threshold
            test_mask = test_longs[feat] > threshold

            r_train = compute_r_arr(train_longs[train_mask])
            r_test = compute_r_arr(test_longs[test_mask])

            if len(r_train) >= 3 and len(r_test) >= 2:
                train_exp = r_train.mean()
                test_exp = r_test.mean()

                # Check if it improves in both
                improves_train = train_exp > r_train_all.mean()
                improves_test = test_exp > r_test_all.mean()

                if improves_train:
                    verdict = "✅" if improves_test else "⚠️ train only"
                    print(f"    {feat}>{threshold:.3f} (p{int(pct*100)}) {' '*max(0,20-len(feat))}{len(r_train):>7d} {train_exp:>+9.3f}R {len(r_test):>7d} {test_exp:>+9.3f}R {verdict}")

                    if improves_test and len(r_test) >= 3:
                        best_filters.append((feat, threshold, train_exp, test_exp, len(r_train), len(r_test)))

    # Try the best combined rule from train
    print(f"\n    COMBINED RULES (built on train, applied to test):")
    # Test promising combinations
    combos = [
        ("vol_expansion > 1.0", lambda r: r.get("vol_expansion", 0) > 1.0),
        ("bull_pct > 0.55 & vol_exp > med", lambda r: r.get("bull_candle_pct_20", 0) > 0.55 and r.get("vol_expansion", 0) > train_longs["vol_expansion"].median()),
        ("EMA stack & dist_ema20 > 0", lambda r: r.get("ema_bullish_stack", 0) > 0.5 and r.get("dist_ema20", 0) > 0),
        ("ret_1h > 0 & vol_exp > 1", lambda r: r.get("ret_1h", 0) > 0 and r.get("vol_expansion", 0) > 1.0),
        ("breakout > 0 & bull_pct > 0.5", lambda r: r.get("breakout_balance", 0) > 0 and r.get("bull_candle_pct_20", 0) > 0.5),
    ]

    for combo_name, combo_fn in combos:
        train_mask = train_longs.apply(combo_fn, axis=1)
        test_mask = test_longs.apply(combo_fn, axis=1)

        r_train = compute_r_arr(train_longs[train_mask])
        r_test = compute_r_arr(test_longs[test_mask])

        if len(r_train) >= 3:
            improves_test = len(r_test) >= 2 and r_test.mean() > r_test_all.mean()
            verdict = "✅" if improves_test else "❌"
            test_str = f"{r_test.mean():>+9.3f}R" if len(r_test) >= 2 else "   —"
            print(f"    {combo_name:<45s} Train: {len(r_train):>3d}, {r_train.mean():>+.3f}R  Test: {len(r_test):>3d}, {test_str} {verdict}")

    return best_filters


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print("=" * 80)
    print("  REGIME DETECTION LAYER — BTCUSDT LONG")
    print("  Can we detect WHEN the LONG setup works?")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Build dataset
    print("\n  Building master dataset...")
    df = build_master_dataset("data/features/btcusdt_1m.csv")

    # Compute regime features
    print("  Computing regime features...")
    df = compute_regime_features(df)

    df_train = df[df["timestamp"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
    df_test = df[df["timestamp"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

    # Run validation
    print("  Running validation...")
    train_results = run_validation(df_train, "BTCUSDT")
    test_results = run_validation(df_test, "BTCUSDT")
    train_results["period"] = "TRAIN"
    test_results["period"] = "TEST"
    all_results = pd.concat([train_results, test_results], ignore_index=True)

    # Long only
    longs = all_results[all_results["direction"] == 1]
    print(f"  LONG setups: {(longs['period']=='TRAIN').sum()} train + {(longs['period']=='TEST').sum()} test = {len(longs)} total")

    # ═══════════════════════════════════════════════════════
    # STEP 1 — LABEL MARKET BEHAVIOR
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 1 — POST-HOC REGIME LABELS")
    print(f"{'═' * 80}")

    labeled = label_regime(longs)
    print(f"\n  Regime distribution:")
    for regime in ["trend", "mild_trend", "neutral", "chop"]:
        n = (labeled["regime_label"] == regime).sum()
        subset = labeled[labeled["regime_label"] == regime]
        r_arr = compute_r_arr(subset)
        if n > 0:
            print(f"    {regime:<15s}: N={n:>3d}, Exp1R={r_arr.mean():>+.3f}R, Hit1R={subset['hit_1R'].astype(bool).mean():.0%}, MedMFE={subset['max_favorable_excursion_R'].median():.2f}R")

    # Regime by period
    for period in ["TRAIN", "TEST"]:
        print(f"\n  {period}:")
        period_labeled = labeled[labeled["period"] == period]
        for regime in ["trend", "mild_trend", "neutral", "chop"]:
            subset = period_labeled[period_labeled["regime_label"] == regime]
            if len(subset) > 0:
                r_arr = compute_r_arr(subset)
                print(f"    {regime:<15s}: N={len(subset):>3d}, Exp1R={r_arr.mean():>+.3f}R, Hit1R={subset['hit_1R'].astype(bool).mean():.0%}")

    # ═══════════════════════════════════════════════════════
    # STEP 3 — FEATURE ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 3 — FEATURE PREDICTIVENESS (ALL LONGS)")
    print(f"{'═' * 80}")

    features_to_test = [
        ("vol_expansion", [0, 0.5, 0.8, 1.0, 1.3, 5.0], ["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.3", ">1.3"]),
        ("vol_pctile", [0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]),
        ("bull_candle_pct_20", [0, 0.40, 0.48, 0.52, 0.56, 1.0], ["<40%", "40-48%", "48-52%", "52-56%", ">56%"]),
        ("ret_5m_20", [-0.1, -0.02, -0.005, 0.005, 0.02, 0.1], ["<-2%", "-2%~-0.5%", "-0.5%~0.5%", "0.5%~2%", ">2%"]),
        ("ret_1h", [-0.1, -0.03, -0.01, 0.01, 0.03, 0.1], ["<-3%", "-3%~-1%", "-1%~1%", "1%~3%", ">3%"]),
        ("dist_ema20", [-0.05, -0.01, -0.002, 0.002, 0.01, 0.05], ["<-1%", "-1%~-0.2%", "-0.2%~0.2%", "0.2%~1%", ">1%"]),
        ("dist_ema50", [-0.1, -0.02, -0.005, 0.005, 0.02, 0.1], ["<-2%", "-2%~-0.5%", "-0.5%~0.5%", "0.5%~2%", ">2%"]),
        ("ema_bullish_stack", [-0.1, 0.5, 1.1], ["Not stacked", "Stacked"]),
        ("range_compression", [0, 0.3, 0.5, 0.7, 2.0], ["<30%", "30-50%", "50-70%", ">70%"]),
        ("breakout_balance", [-10, -1, 0, 1, 10], ["<-1", "-1~0", "0~1", ">1"]),
        ("price_slope_20", [-0.01, -0.001, 0, 0.001, 0.01], ["<-0.1%", "-0.1%~0", "0~0.1%", ">0.1%"]),
        ("h4_rsi_slope_1", [-5, -1, 0, 1, 5], ["<-1", "-1~0", "0~1", ">1"]),
    ]

    for feat_name, bins, labels in features_to_test:
        result = analyze_feature_predictiveness(longs, df, feat_name, bins, labels)
        print_feature_table(feat_name, result)

    # ═══════════════════════════════════════════════════════
    # STEP 4 — BUILD REGIME FILTER
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 4 — REGIME FILTER CANDIDATES")
    print(f"{'═' * 80}")

    best_rule = build_regime_filter(train_results, df_train)

    # ═══════════════════════════════════════════════════════
    # STEP 5 — VALIDATE BEST FILTER
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 5 — FILTER VALIDATION")
    print(f"{'═' * 80}")

    # Test the most promising filters
    filter_candidates = [
        ("vol_expansion > 1.0", lambda r: r.get("vol_expansion", 0) > 1.0 if pd.notna(r.get("vol_expansion")) else False),
        ("bull_pct > 0.52", lambda r: r.get("bull_candle_pct_20", 0) > 0.52 if pd.notna(r.get("bull_candle_pct_20")) else False),
        ("EMA stack + bull", lambda r: r.get("ema_bullish_stack", 0) > 0.5 and r.get("bull_candle_pct_20", 0) > 0.50 if pd.notna(r.get("ema_bullish_stack")) else False),
        ("ret_1h > 0", lambda r: r.get("ret_1h", 0) > 0 if pd.notna(r.get("ret_1h")) else False),
        ("breakout > 0 + vol", lambda r: r.get("breakout_balance", 0) > 0 and r.get("vol_expansion", 0) > 0.8 if pd.notna(r.get("breakout_balance")) else False),
    ]

    for filter_name, filter_fn in filter_candidates:
        print(f"\n  Filter: {filter_name}")

        # Train
        train_longs = train_results[train_results["direction"] == 1].copy()
        for feat in ["vol_expansion", "bull_candle_pct_20", "ema_bullish_stack",
                     "dist_ema20", "breakout_balance", "ret_1h", "h4_rsi_slope_1",
                     "vol_pctile", "ret_5m_20", "range_compression"]:
            if feat in df_train.columns:
                feature_lookup = df_train.set_index("timestamp")[feat]
                train_longs[feat] = train_longs["signal_time"].map(feature_lookup)

        train_mask = train_longs.apply(filter_fn, axis=1)
        train_filtered = train_longs[train_mask]

        # Test
        test_longs = test_results[test_results["direction"] == 1].copy()
        for feat in ["vol_expansion", "bull_candle_pct_20", "ema_bullish_stack",
                     "dist_ema20", "breakout_balance", "ret_1h", "h4_rsi_slope_1",
                     "vol_pctile", "ret_5m_20", "range_compression"]:
            if feat in df_test.columns:
                feature_lookup = df_test.set_index("timestamp")[feat]
                test_longs[feat] = test_longs["signal_time"].map(feature_lookup)

        test_mask = test_longs.apply(filter_fn, axis=1)
        test_filtered = test_longs[test_mask]

        print(f"    {'Case':<15s} {'N':>5s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s} {'MaxDD':>7s} {'MaxCL':>6s}")
        print(f"    {'─'*15} {'─'*5} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*6}")

        for case_label, case_df in [
            ("TRAIN all", train_longs), ("TRAIN filtered", train_filtered),
            ("TEST all", test_longs), ("TEST filtered", test_filtered)
        ]:
            if len(case_df) == 0:
                print(f"    {case_label:<15s} —")
                continue
            m = compute_full_metrics(case_df)
            if m:
                print(f"    {case_label:<15s} {m['N']:>5d} {m['Hit1R']:>6.1%} {m['Exp1R']:>+7.3f}R {m['TotalR']:>+7.1f}R {m['MaxDD']:>6.1f}R {m['MaxConsec']:>5d}")

    # ═══════════════════════════════════════════════════════
    # STEP 6 — WALK-FORWARD
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 6 — WALK-FORWARD FILTER")
    print(f"{'═' * 80}")

    best_filters = walk_forward_filter(all_results, df)

    # ═══════════════════════════════════════════════════════
    # FINAL CONCLUSION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  FINAL CONCLUSION")
    print(f"{'═' * 80}")

    # Summary stats
    train_longs = all_results[(all_results["direction"] == 1) & (all_results["period"] == "TRAIN")]
    test_longs = all_results[(all_results["direction"] == 1) & (all_results["period"] == "TEST")]

    r_train = compute_r_arr(train_longs)
    r_test = compute_r_arr(test_longs)

    print(f"\n  BASELINE:")
    print(f"    Train: N={len(train_longs)}, Exp1R={r_train.mean():+.3f}R, TotalR={r_train.sum():+.1f}R")
    print(f"    Test:  N={len(test_longs)}, Exp1R={r_test.mean():+.3f}R, TotalR={r_test.sum():+.1f}R")

    # Post-hoc regime analysis
    train_labeled = label_regime(train_longs)
    test_labeled = label_regime(test_longs)

    for regime in ["trend", "chop"]:
        tr = train_labeled[train_labeled["regime_label"] == regime]
        te = test_labeled[test_labeled["regime_label"] == regime]
        if len(tr) > 0 and len(te) > 0:
            r_tr = compute_r_arr(tr)
            r_te = compute_r_arr(te)
            print(f"\n  POST-HOC '{regime.upper()}' REGIME:")
            print(f"    Train: N={len(tr)}, Exp1R={r_tr.mean():+.3f}R")
            print(f"    Test:  N={len(te)}, Exp1R={r_te.mean():+.3f}R")

    print(f"\n  ANSWER: Can we detect, in real time, when the system should be ON vs OFF?")
    print(f"  See walk-forward results above.")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
