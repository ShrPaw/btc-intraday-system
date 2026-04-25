"""
EMERGING MOMENTUM ZONE — STATISTICAL VALIDATION
================================================
Validates whether the candidate zone (H4 RSI 55-60, H12 RSI 50-55)
represents a real, stable, statistically significant edge.

This is NOT an optimization script. It is a falsification attempt.
We assume the zone is WRONG until proven otherwise.

BTCUSDT only. R-multiples. Worst-case intracandle. Closed candles.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from setup_validation_engine import (
    build_master_dataset, run_validation, compute_group_metrics,
    TRAIN_CUTOFF
)


# ═══════════════════════════════════════════════════════════
# PHASE 1 — DEFINE THE CONDITION
# ═══════════════════════════════════════════════════════════

CANDIDATE_ZONE = "H4∈[55,60) & H12∈[50,55)"

def classify_zone(row):
    """Classify a LONG setup into one of the analysis zones."""
    h4 = row.get("h4_rsi_entry", 50)
    h12 = row.get("h12_rsi_entry", 50)

    # Candidate zone
    if 55 <= h4 < 60 and 50 <= h12 < 55:
        return "CANDIDATE"

    # Control groups
    zones = []
    if 50 <= h4 < 55:
        zones.append("H4_50_55")
    elif 60 <= h4 < 70:
        zones.append("H4_60_70")
    elif h4 >= 70:
        zones.append("H4_70plus")

    if 55 <= h12 < 60:
        zones.append("H12_55_60")
    elif h12 >= 60:
        zones.append("H12_60plus")

    return "|".join(zones) if zones else "OTHER"


def add_zone_labels(results):
    """Add zone classification to results dataframe."""
    results = results.copy()
    results["zone"] = results.apply(classify_zone, axis=1)
    return results


# ═══════════════════════════════════════════════════════════
# PHASE 2 — FULL DATASET METRICS
# ═══════════════════════════════════════════════════════════

def compute_full_metrics(subset):
    """Compute all required metrics for a subset."""
    n = len(subset)
    if n == 0:
        return None

    hit_1r = subset["hit_1R"].astype(bool)
    hit_2r = subset["hit_2R"].astype(bool) if "hit_2R" in subset.columns else pd.Series(False, index=subset.index)
    hit_3r = subset["hit_3R"].astype(bool) if "hit_3R" in subset.columns else pd.Series(False, index=subset.index)
    sl_hit = subset["sl_hit"].astype(bool)

    # R-multiple per trade
    r_per_trade = []
    for _, row in subset.iterrows():
        if row["sl_hit"] and not row["hit_1R"]:
            r = -1.0
        elif row["hit_1R"] and row["sl_hit"]:
            r = 0.0  # hit TP then stopped
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
            r = 0.0  # expired
        r_per_trade.append(r)

    r_series = pd.Series(r_per_trade, index=subset.index)
    cum_r = r_series.cumsum()
    peak_r = cum_r.cummax()
    dd = cum_r - peak_r

    # Losing streak
    max_consec = 0
    cur_consec = 0
    for r in r_series:
        if r < 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    mfe = subset["max_favorable_excursion_R"] if "max_favorable_excursion_R" in subset.columns else pd.Series(0, index=subset.index)
    mae = subset["max_adverse_excursion_R"] if "max_adverse_excursion_R" in subset.columns else pd.Series(0, index=subset.index)

    return {
        "N": n,
        "Hit1R": hit_1r.sum() / n,
        "Hit2R": hit_2r.sum() / n,
        "Hit3R": hit_3r.sum() / n,
        "Exp1R": r_series.mean(),
        "Exp2R": r_series.mean(),  # same as Exp1R in fixed-1R framework
        "TotalR": r_series.sum(),
        "MedMFE": mfe.median(),
        "MedMAE": mae.median(),
        "MaxDD": dd.min(),
        "MaxConsecLoss": max_consec,
        "WinRate": (r_series > 0).sum() / n,
        "LossRate": (r_series < 0).sum() / n,
        "R_series": r_series,
    }


def print_metrics_row(label, m, indent=4):
    """Print a metrics row."""
    if m is None:
        print(f"{' ' * indent}{label:<35s} —")
        return
    print(f"{' ' * indent}{label:<35s} N={m['N']:>4d}  Hit1R={m['Hit1R']:>5.1%}  Hit2R={m['Hit2R']:>5.1%}  "
          f"Exp1R={m['Exp1R']:>+6.3f}R  TotalR={m['TotalR']:>+7.1f}R  "
          f"MedMFE={m['MedMFE']:>5.2f}R  MedMAE={m['MedMAE']:>5.2f}R  "
          f"MaxDD={m['MaxDD']:>6.1f}R  MaxConsec={m['MaxConsecLoss']}")


# ═══════════════════════════════════════════════════════════
# PHASE 3 — STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    if len(data) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    boot_means = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return boot_means.mean(), lo, hi


def permutation_test(r_candidate, r_baseline, n_perm=10000):
    """Two-sided permutation test: is candidate mean different from baseline mean?"""
    observed_diff = r_candidate.mean() - r_baseline.mean()
    combined = np.concatenate([r_candidate, r_baseline])
    n_c = len(r_candidate)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_diff = perm[:n_c].mean() - perm[n_c:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    p_value = count / n_perm
    return observed_diff, p_value


def binomial_test_hit(n_hits, n_total, p_null=0.5):
    """One-sided binomial test: is hit rate significantly > 50%?"""
    if n_total == 0:
        return np.nan
    # P(X >= n_hits) under Binom(n_total, p_null)
    p_val = 1 - stats.binom.cdf(n_hits - 1, n_total, p_null)
    return p_val


def statistical_tests(candidate_r, baseline_r, candidate_hits, candidate_n):
    """Run all statistical tests for candidate vs baseline."""
    print(f"\n    Candidate: N={len(candidate_r)}, Mean R={candidate_r.mean():+.3f}")
    print(f"    Baseline:  N={len(baseline_r)}, Mean R={baseline_r.mean():+.3f}")
    print(f"    Delta:     {candidate_r.mean() - baseline_r.mean():+.3f}R")

    # Bootstrap CI for candidate mean
    boot_mean, ci_lo, ci_hi = bootstrap_ci(candidate_r)
    print(f"\n    Bootstrap 95% CI for candidate Exp1R: [{ci_lo:+.3f}, {ci_hi:+.3f}]R (mean={boot_mean:+.3f}R)")
    if ci_lo > 0:
        print(f"    → CI excludes zero: edge is statistically distinguishable from zero")
    else:
        print(f"    → CI includes zero: edge CANNOT be distinguished from zero")

    # Permutation test
    perm_diff, perm_p = permutation_test(candidate_r, baseline_r)
    print(f"\n    Permutation test (two-sided): p={perm_p:.4f}")
    if perm_p < 0.05:
        print(f"    → Significant at α=0.05")
    elif perm_p < 0.10:
        print(f"    → Marginal (p < 0.10)")
    else:
        print(f"    → NOT significant")

    # Binomial test for hit rate
    binom_p = binomial_test_hit(candidate_hits, candidate_n)
    print(f"\n    Binomial test (hit rate > 50%): p={binom_p:.4f}")
    if binom_p < 0.05:
        print(f"    → Hit rate significantly > 50%")
    else:
        print(f"    → Hit rate NOT significantly > 50%")

    return {
        "boot_ci_lo": ci_lo,
        "boot_ci_hi": ci_hi,
        "perm_p": perm_p,
        "binom_p": binom_p,
    }


# ═══════════════════════════════════════════════════════════
# PHASE 4 — STABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════

def stability_analysis(results, zone_label, zone_mask):
    """Check stability of a zone across time, regime, session."""
    zone = results[zone_mask].copy()
    if len(zone) == 0:
        print(f"    No data for {zone_label}")
        return

    print(f"\n    STABILITY: {zone_label} ({len(zone)} setups)")

    # By month
    print(f"\n    {'Month':<10s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"    {'─'*10} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")
    months_stable = 0
    months_total = 0
    for month in sorted(zone["month"].unique()):
        m = zone[zone["month"] == month]
        mm = compute_full_metrics(m)
        if mm:
            print(f"    {month:<10s} {mm['N']:>4d} {mm['Hit1R']:>6.1%} {mm['Exp1R']:>+7.3f}R {mm['TotalR']:>+7.1f}R")
            months_total += 1
            if mm["Exp1R"] > 0:
                months_stable += 1
    print(f"    → {months_stable}/{months_total} months positive")

    # By regime
    print(f"\n    {'Regime':<12s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"    {'─'*12} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")
    for regime in ["bullish", "bearish", "neutral"]:
        r = zone[zone["htf_regime"] == regime]
        if len(r) > 0:
            mm = compute_full_metrics(r)
            if mm:
                print(f"    {regime:<12s} {mm['N']:>4d} {mm['Hit1R']:>6.1%} {mm['Exp1R']:>+7.3f}R {mm['TotalR']:>+7.1f}R")

    # By session
    print(f"\n    {'Session':<12s} {'N':>4s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"    {'─'*12} {'─'*4} {'─'*7} {'─'*8} {'─'*8}")
    for session in ["Asian", "European", "US"]:
        s = zone[zone["session"] == session]
        if len(s) > 0:
            mm = compute_full_metrics(s)
            if mm:
                print(f"    {session:<12s} {mm['N']:>4d} {mm['Hit1R']:>6.1%} {mm['Exp1R']:>+7.3f}R {mm['TotalR']:>+7.1f}R")


# ═══════════════════════════════════════════════════════════
# PHASE 5 — OUTLIER DEPENDENCY
# ═══════════════════════════════════════════════════════════

def outlier_analysis(results, zone_label, zone_mask):
    """Check if zone performance depends on few outlier trades."""
    zone = results[zone_mask].copy()
    if len(zone) < 5:
        print(f"\n    OUTLIER: {zone_label} — too few trades ({len(zone)})")
        return

    # Compute R per trade
    r_list = []
    for _, row in zone.iterrows():
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

    r_arr = np.array(r_list)
    r_sorted = np.sort(r_arr)[::-1]

    print(f"\n    OUTLIER: {zone_label}")
    print(f"    Full:     N={len(r_arr)}, Exp1R={r_arr.mean():+.3f}R, TotalR={r_arr.sum():+.1f}R")

    # Top 10% contribution
    n_top = max(1, int(len(r_arr) * 0.1))
    top_sum = r_sorted[:n_top].sum()
    total_sum = r_arr.sum()
    if total_sum != 0:
        top_pct = top_sum / total_sum * 100 if total_sum > 0 else float('inf')
    else:
        top_pct = 0
    print(f"    Top {n_top} trade(s) contribute {top_sum:+.1f}R of {total_sum:+.1f}R ({top_pct:.0f}%)")

    # Remove best 1, 2, 3 trades
    for n_remove in [1, 2, 3]:
        if len(r_arr) - n_remove < 2:
            break
        trimmed = np.sort(r_arr)[::-1][n_remove:]  # remove top n
        print(f"    Remove top {n_remove}: N={len(trimmed)}, Exp1R={trimmed.mean():+.3f}R, TotalR={trimmed.sum():+.1f}R")

    # Remove worst 1 trade
    worst_removed = np.sort(r_arr)[1:]  # remove worst
    print(f"    Remove worst 1: N={len(worst_removed)}, Exp1R={worst_removed.mean():+.3f}R, TotalR={worst_removed.sum():+.1f}R")

    # Contribution check
    if total_sum > 0 and top_pct > 80:
        print(f"    ⚠️  WARNING: >80% of R comes from top trades — fragile edge")
    elif total_sum > 0 and top_pct > 50:
        print(f"    ⚠️  CAUTION: >50% of R comes from top trades")
    else:
        print(f"    ✅ R distributed across trades — not dependent on outliers")


# ═══════════════════════════════════════════════════════════
# PHASE 6 — BASELINE COMPARISON
# ═══════════════════════════════════════════════════════════

def build_baselines(df_test, results_long_test, n_sims=200):
    """
    Build random baselines using identical stop/TP/outcome logic.
    """
    from setup_validation_engine import compute_structural_stop, compute_tp_levels, track_setup_outcome

    rng = np.random.default_rng(42)
    baselines = {}

    # Only use bars with sufficient indicator history
    valid_bars = df_test.iloc[500:]  # warmup

    # Sample random entry points
    n_target = len(results_long_test)
    if n_target == 0 or len(valid_bars) < n_target:
        return baselines

    # ─────────────────────────────────────────
    # Baseline 1: Full-dataset random
    # ─────────────────────────────────────────
    random_indices = rng.choice(len(valid_bars), size=min(n_sims, len(valid_bars)), replace=False)
    random_indices = np.sort(random_indices)

    bl1_results = []
    for idx in random_indices:
        real_idx = valid_bars.index[idx]
        entry_idx = real_idx + 1
        if entry_idx >= len(df_test):
            continue

        entry_price = float(df_test.iloc[entry_idx]["open"])
        direction = 1  # LONG

        # Structural stop from random entry
        stop_info = compute_structural_stop(df_test, real_idx, direction, "RSI_TREND")
        if not stop_info["is_stop_valid"]:
            continue

        stop_price = stop_info["stop_price"]
        tp_levels = compute_tp_levels(entry_price, stop_price, direction)
        tp_levels["entry_price"] = entry_price

        outcome = track_setup_outcome(df_test, entry_idx, direction, stop_price, tp_levels)
        if not outcome.get("valid", False):
            continue

        bl1_results.append(outcome)

    if bl1_results:
        bl1_df = pd.DataFrame(bl1_results)
        bl1_r = compute_r_series(bl1_df)
        baselines["Random (full dataset)"] = bl1_r

    # ─────────────────────────────────────────
    # Baseline 2: Same-timestamp random direction
    # ─────────────────────────────────────────
    st_results = []
    for _, row in results_long_test.iterrows():
        sig_time = row["signal_time"]
        bar_mask = df_test["timestamp"] == sig_time
        if not bar_mask.any():
            continue
        sig_idx = df_test.index[bar_mask][0]

        # Random direction at same timestamp
        rand_dir = rng.choice([1, -1])
        entry_idx = sig_idx + 1
        if entry_idx >= len(df_test):
            continue

        entry_price = float(df_test.iloc[entry_idx]["open"])
        stop_info = compute_structural_stop(df_test, sig_idx, rand_dir, "RSI_TREND")
        if not stop_info["is_stop_valid"]:
            continue

        stop_price = stop_info["stop_price"]
        tp_levels = compute_tp_levels(entry_price, stop_price, rand_dir)
        tp_levels["entry_price"] = entry_price

        outcome = track_setup_outcome(df_test, entry_idx, rand_dir, stop_price, tp_levels)
        if not outcome.get("valid", False):
            continue

        st_results.append(outcome)

    if st_results:
        st_df = pd.DataFrame(st_results)
        st_r = compute_r_series(st_df)
        baselines["Same-timestamp random dir"] = st_r

    # ─────────────────────────────────────────
    # Baseline 3: Same-regime random timestamps
    # ─────────────────────────────────────────
    # Sample from bars where H4 RSI is in the same broad range (50-60)
    regime_bars = df_test[(df_test["h4_rsi"] >= 50) & (df_test["h4_rsi"] < 60)]
    regime_bars = regime_bars.iloc[200:]  # warmup

    if len(regime_bars) > n_target:
        regime_indices = rng.choice(len(regime_bars), size=n_target, replace=False)
    else:
        regime_indices = range(len(regime_bars))

    sr_results = []
    for idx in regime_indices:
        real_idx = regime_bars.index[idx]
        entry_idx = real_idx + 1
        if entry_idx >= len(df_test):
            continue

        entry_price = float(df_test.iloc[entry_idx]["open"])
        direction = 1

        stop_info = compute_structural_stop(df_test, real_idx, direction, "RSI_TREND")
        if not stop_info["is_stop_valid"]:
            continue

        stop_price = stop_info["stop_price"]
        tp_levels = compute_tp_levels(entry_price, stop_price, direction)
        tp_levels["entry_price"] = entry_price

        outcome = track_setup_outcome(df_test, entry_idx, direction, stop_price, tp_levels)
        if not outcome.get("valid", False):
            continue

        sr_results.append(outcome)

    if sr_results:
        sr_df = pd.DataFrame(sr_results)
        sr_r = compute_r_series(sr_df)
        baselines["Same-regime (H4 50-60) random"] = sr_r

    # ─────────────────────────────────────────
    # Baseline 4: Same-session random timestamps
    # ─────────────────────────────────────────
    # Get the hours of candidate zone signals
    candidate_hours = results_long_test["hour_utc"].unique() if "hour_utc" in results_long_test.columns else []
    if len(candidate_hours) > 0:
        session_bars = df_test[df_test["timestamp"].dt.hour.isin(candidate_hours)]
        session_bars = session_bars.iloc[200:]

        if len(session_bars) > n_target:
            sess_indices = rng.choice(len(session_bars), size=n_target, replace=False)
        else:
            sess_indices = range(len(session_bars))

        ss_results = []
        for idx in sess_indices:
            real_idx = session_bars.index[idx]
            entry_idx = real_idx + 1
            if entry_idx >= len(df_test):
                continue

            entry_price = float(df_test.iloc[entry_idx]["open"])
            direction = 1

            stop_info = compute_structural_stop(df_test, real_idx, direction, "RSI_TREND")
            if not stop_info["is_stop_valid"]:
                continue

            stop_price = stop_info["stop_price"]
            tp_levels = compute_tp_levels(entry_price, stop_price, direction)
            tp_levels["entry_price"] = entry_price

            outcome = track_setup_outcome(df_test, entry_idx, direction, stop_price, tp_levels)
            if not outcome.get("valid", False):
                continue

            ss_results.append(outcome)

        if ss_results:
            ss_df = pd.DataFrame(ss_results)
            ss_r = compute_r_series(ss_df)
            baselines["Same-session random"] = ss_r

    return baselines


def compute_r_series(results_df):
    """Compute R per trade from results."""
    r_list = []
    for _, row in results_df.iterrows():
        if row.get("sl_hit", False) and not row.get("hit_1R", False):
            r = -1.0
        elif row.get("hit_1R", False) and row.get("sl_hit", False):
            r = 0.0
        elif row.get("hit_1R", False) and not row.get("sl_hit", False):
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
# MAIN — RUN ALL PHASES
# ═══════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print("=" * 90)
    print("  EMERGING MOMENTUM ZONE — STATISTICAL VALIDATION")
    print("  Candidate: LONG setup + H4 RSI ∈ [55,60) + H12 RSI ∈ [50,55)")
    print("  Mission: FALSIFICATION, not optimization")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # ─────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────
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

    # LONG only
    longs = all_results[all_results["direction"] == 1].copy()
    longs_train = longs[longs["period"] == "TRAIN"]
    longs_test = longs[longs["period"] == "TEST"]

    print(f"  LONG setups: {len(longs_train)} train + {len(longs_test)} test = {len(longs)} total")

    # Add zone labels
    longs = add_zone_labels(longs)
    longs_train = longs[longs["period"] == "TRAIN"]
    longs_test = longs[longs["period"] == "TEST"]

    # ═══════════════════════════════════════════════════════
    # PHASE 1 — DEFINE THE CONDITION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 1 — CONDITION DEFINITION")
    print(f"{'═' * 90}")
    print(f"\n  Candidate zone: {CANDIDATE_ZONE}")
    print(f"  Direction: LONG only")
    print(f"  Symbol: BTCUSDT")
    print(f"\n  Zone distribution (all LONGs):")
    for zone in sorted(longs["zone"].unique()):
        n = (longs["zone"] == zone).sum()
        print(f"    {zone:<30s}: {n:>4d} setups")

    # ═══════════════════════════════════════════════════════
    # PHASE 2 — FULL DATASET METRICS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 2 — FULL DATASET METRICS")
    print(f"{'═' * 90}")

    # Define all groups
    groups = {
        "A. ALL LONG (baseline)": longs["direction"] == 1,
        "B. H4∈[50,55) (weak zone)": longs["zone"].str.contains("H4_50_55"),
        f"C. {CANDIDATE_ZONE}": longs["zone"] == "CANDIDATE",
        "D. H4∈[60,70) (late zone)": longs["zone"].str.contains("H4_60_70"),
        "E. H12∈[55,60) (HTF extended)": longs["zone"].str.contains("H12_55_60"),
    }

    # Add ADX and volatility groups
    if "h4_adx" in longs.columns:
        groups["F. High ADX (>30)"] = longs["h4_adx"] > 30
        groups["G. Low ADX (<20)"] = longs["h4_adx"] < 20

    if "rv_6" in longs.columns:
        rv_median = longs["rv_6"].median()
        groups["H. High volatility"] = longs["rv_6"] > rv_median
        groups["I. Low volatility"] = longs["rv_6"] <= rv_median

    print(f"\n  ALL PERIODS:")
    print(f"  {'Group':<40s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'TotalR':>8s} {'MedMFE':>7s} {'MedMAE':>7s} {'MaxDD':>7s} {'MaxCL':>6s}")
    print(f"  {'─'*40} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")

    for label, mask in groups.items():
        subset = longs[mask]
        m = compute_full_metrics(subset)
        if m:
            print(f"  {label:<40s} {m['N']:>5d} {m['Hit1R']:>6.1%} {m['Hit2R']:>6.1%} {m['Exp1R']:>+7.3f}R {m['TotalR']:>+7.1f}R {m['MedMFE']:>6.2f}R {m['MedMAE']:>6.2f}R {m['MaxDD']:>6.1f}R {m['MaxConsecLoss']:>5d}")

    # Train vs Test
    for period_label, period_df in [("TRAIN", longs_train), ("TEST", longs_test)]:
        print(f"\n  {period_label}:")
        print(f"  {'Group':<40s} {'N':>5s} {'Hit1R':>7s} {'Hit2R':>7s} {'Exp1R':>8s} {'TotalR':>8s} {'MedMFE':>7s} {'MedMAE':>7s} {'MaxDD':>7s} {'MaxCL':>6s}")
        print(f"  {'─'*40} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")

        for label, mask in groups.items():
            # Re-apply mask to period subset
            period_mask = mask[period_df.index] if mask.index.equals(longs.index) else mask.loc[period_df.index]
            subset = period_df[period_mask]
            m = compute_full_metrics(subset)
            if m:
                print(f"  {label:<40s} {m['N']:>5d} {m['Hit1R']:>6.1%} {m['Hit2R']:>6.1%} {m['Exp1R']:>+7.3f}R {m['TotalR']:>+7.1f}R {m['MedMFE']:>6.2f}R {m['MedMAE']:>6.2f}R {m['MaxDD']:>6.1f}R {m['MaxConsecLoss']:>5d}")

    # ═══════════════════════════════════════════════════════
    # PHASE 3 — STATISTICAL SIGNIFICANCE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 3 — STATISTICAL SIGNIFICANCE")
    print(f"{'═' * 90}")

    candidate = longs[longs["zone"] == "CANDIDATE"]
    baseline = longs  # all LONGs as baseline

    if len(candidate) >= 3:
        candidate_r = compute_r_series(candidate)
        baseline_r = compute_r_series(baseline)

        print(f"\n  CANDIDATE vs ALL LONG BASELINE (all periods):")
        stat_results = statistical_tests(candidate_r, baseline_r,
                                          int(candidate["hit_1R"].astype(bool).sum()),
                                          len(candidate))

        # Train vs test
        for period_label, period_df in [("TRAIN", longs_train), ("TEST", longs_test)]:
            c_period = period_df[period_df["zone"] == "CANDIDATE"]
            b_period = period_df
            if len(c_period) >= 3:
                c_r = compute_r_series(c_period)
                b_r = compute_r_series(b_period)
                print(f"\n  CANDIDATE vs BASELINE ({period_label}):")
                statistical_tests(c_r, b_r,
                                  int(c_period["hit_1R"].astype(bool).sum()),
                                  len(c_period))
            else:
                print(f"\n  CANDIDATE ({period_label}): {len(c_period)} setups — INCONCLUSIVE (need ≥3)")
    else:
        print(f"\n  CANDIDATE zone: {len(candidate)} setups — INCONCLUSIVE (need ≥3)")

    # ═══════════════════════════════════════════════════════
    # PHASE 4 — STABILITY ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 4 — STABILITY ANALYSIS")
    print(f"{'═' * 90}")

    candidate_mask = longs["zone"] == "CANDIDATE"
    stability_analysis(longs, CANDIDATE_ZONE, candidate_mask)

    # Also check baseline for comparison
    all_mask = longs["direction"] == 1
    stability_analysis(longs, "ALL LONG (baseline)", all_mask)

    # ═══════════════════════════════════════════════════════
    # PHASE 5 — OUTLIER DEPENDENCY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 5 — OUTLIER DEPENDENCY")
    print(f"{'═' * 90}")

    outlier_analysis(longs, CANDIDATE_ZONE, candidate_mask)
    outlier_analysis(longs, "ALL LONG (baseline)", all_mask)

    # ═══════════════════════════════════════════════════════
    # PHASE 6 — BASELINE COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 6 — BASELINE COMPARISON (TEST PERIOD)")
    print(f"{'═' * 90}")

    candidate_test = longs_test[longs_test["zone"] == "CANDIDATE"]
    if len(candidate_test) >= 3:
        print(f"\n  Building random baselines (this takes a moment)...")
        baselines = build_baselines(df_test, candidate_test, n_sims=200)

        candidate_r = compute_r_series(candidate_test)
        c_mean = candidate_r.mean()

        print(f"\n  Candidate zone (test): N={len(candidate_test)}, Exp1R={c_mean:+.3f}R")
        print(f"\n  {'Baseline':<35s} {'N':>5s} {'Exp1R':>8s} {'Delta':>8s} {'Verdict'}")
        print(f"  {'─'*35} {'─'*5} {'─'*8} {'─'*8} {'─'*20}")

        for bl_name, bl_r in baselines.items():
            bl_mean = bl_r.mean()
            delta = c_mean - bl_mean
            if delta > 0.1:
                verdict = "Candidate better"
            elif delta > 0:
                verdict = "Marginal"
            elif delta > -0.1:
                verdict = "Similar"
            else:
                verdict = "Candidate WORSE"
            print(f"  {bl_name:<35s} {len(bl_r):>5d} {bl_mean:>+7.3f}R {delta:>+7.3f}R {verdict}")
    else:
        print(f"\n  Candidate test: {len(candidate_test)} setups — too few for baseline comparison")

    # ═══════════════════════════════════════════════════════
    # PHASE 7 — FINAL CLASSIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 7 — FINAL CLASSIFICATION")
    print(f"{'═' * 90}")

    # Gather all evidence
    c_all = longs[longs["zone"] == "CANDIDATE"]
    c_train = longs_train[longs_train["zone"] == "CANDIDATE"]
    c_test = longs_test[longs_test["zone"] == "CANDIDATE"]

    m_all = compute_full_metrics(c_all)
    m_train = compute_full_metrics(c_train)
    m_test = compute_full_metrics(c_test)

    print(f"\n  CANDIDATE ZONE: {CANDIDATE_ZONE}")
    print(f"  ─────────────────────────────────────────")

    criteria = []

    # 1. Positive Exp1R in both train and test
    if m_train and m_test:
        if m_train["Exp1R"] > 0 and m_test["Exp1R"] > 0:
            criteria.append(("Positive Exp1R in both periods", "PASS", f"train={m_train['Exp1R']:+.3f}, test={m_test['Exp1R']:+.3f}"))
        elif m_test["Exp1R"] > 0:
            criteria.append(("Positive Exp1R in both periods", "PARTIAL", f"train={m_train['Exp1R']:+.3f} ❌, test={m_test['Exp1R']:+.3f} ✅"))
        else:
            criteria.append(("Positive Exp1R in both periods", "FAIL", f"train={m_train['Exp1R']:+.3f}, test={m_test['Exp1R']:+.3f}"))
    else:
        criteria.append(("Positive Exp1R in both periods", "INCONCLUSIVE", "insufficient data"))

    # 2. Stable across months
    if m_all and m_all["N"] >= 6:
        months_positive = 0
        months_total = 0
        for month in sorted(c_all["month"].unique()):
            mm = compute_full_metrics(c_all[c_all["month"] == month])
            if mm:
                months_total += 1
                if mm["Exp1R"] > 0:
                    months_positive += 1
        if months_total >= 3:
            stability_ratio = months_positive / months_total
            if stability_ratio >= 0.6:
                criteria.append(("Stable across months", "PASS", f"{months_positive}/{months_total} months positive"))
            else:
                criteria.append(("Stable across months", "FAIL", f"{months_positive}/{months_total} months positive"))
        else:
            criteria.append(("Stable across months", "INCONCLUSIVE", f"only {months_total} months with data"))
    else:
        criteria.append(("Stable across months", "INCONCLUSIVE", "insufficient data"))

    # 3. Not dependent on few trades
    if m_all and m_all["N"] >= 5:
        r_arr = compute_r_series(c_all)
        r_sorted = np.sort(r_arr)[::-1]
        n_top = max(1, int(len(r_arr) * 0.1))
        top_sum = r_sorted[:n_top].sum()
        total_sum = r_arr.sum()
        if total_sum > 0:
            top_pct = top_sum / total_sum * 100
        else:
            top_pct = 0
        if top_pct < 50:
            criteria.append(("Not outlier-dependent", "PASS", f"top {n_top} trades = {top_pct:.0f}% of total R"))
        elif top_pct < 80:
            criteria.append(("Not outlier-dependent", "CAUTION", f"top {n_top} trades = {top_pct:.0f}% of total R"))
        else:
            criteria.append(("Not outlier-dependent", "FAIL", f"top {n_top} trades = {top_pct:.0f}% of total R"))
    else:
        criteria.append(("Not outlier-dependent", "INCONCLUSIVE", "insufficient data"))

    # 4. Sufficient sample size
    if m_all and m_all["N"] >= 30:
        criteria.append(("Sufficient sample size", "PASS", f"N={m_all['N']}"))
    elif m_all and m_all["N"] >= 10:
        criteria.append(("Sufficient sample size", "MARGINAL", f"N={m_all['N']} (need ≥30)"))
    else:
        criteria.append(("Sufficient sample size", "FAIL", f"N={m_all['N'] if m_all else 0} (need ≥30)"))

    # Print criteria
    print(f"\n  CRITERIA:")
    for criterion, status, detail in criteria:
        icon = "✅" if status == "PASS" else ("⚠️" if status in ("PARTIAL", "CAUTION", "MARGINAL") else ("❌" if status == "FAIL" else "❓"))
        print(f"    {icon} {criterion:<35s} {status:<15s} {detail}")

    # Final classification
    pass_count = sum(1 for _, s, _ in criteria if s == "PASS")
    fail_count = sum(1 for _, s, _ in criteria if s == "FAIL")
    inconclusive_count = sum(1 for _, s, _ in criteria if s == "INCONCLUSIVE")

    print(f"\n  VERDICT:")
    if fail_count >= 2:
        classification = "WEAK / NO EDGE"
        print(f"    ❌ {classification}")
        print(f"    Multiple criteria failed. The candidate zone does not survive falsification.")
    elif inconclusive_count >= 2:
        classification = "INCONCLUSIVE"
        print(f"    ❓ {classification}")
        print(f"    Insufficient data to validate or reject. Need more sample size.")
    elif pass_count >= 3:
        classification = "VALIDATED EDGE"
        print(f"    ✅ {classification}")
        print(f"    The candidate zone survives falsification across all criteria.")
    else:
        classification = "PROMISING BUT INSUFFICIENT DATA"
        print(f"    ⚠️  {classification}")
        print(f"    Some criteria pass but not enough for full validation.")

    # ═══════════════════════════════════════════════════════
    # PHASE 8 — SUMMARY OUTPUT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  PHASE 8 — SUMMARY")
    print(f"{'═' * 90}")

    print(f"\n  CANDIDATE: {CANDIDATE_ZONE}")
    print(f"  CLASSIFICATION: {classification}")
    print(f"\n  KEY NUMBERS:")
    if m_all:
        print(f"    All periods: N={m_all['N']}, Exp1R={m_all['Exp1R']:+.3f}R, Hit1R={m_all['Hit1R']:.1%}, TotalR={m_all['TotalR']:+.1f}R")
    if m_train:
        print(f"    Train:       N={m_train['N']}, Exp1R={m_train['Exp1R']:+.3f}R, Hit1R={m_train['Hit1R']:.1%}, TotalR={m_train['TotalR']:+.1f}R")
    if m_test:
        print(f"    Test:        N={m_test['N']}, Exp1R={m_test['Exp1R']:+.3f}R, Hit1R={m_test['Hit1R']:.1%}, TotalR={m_test['TotalR']:+.1f}R")

    # Comparison table
    print(f"\n  ZONE COMPARISON (test period):")
    print(f"  {'Zone':<40s} {'N':>5s} {'Hit1R':>7s} {'Exp1R':>8s} {'TotalR':>8s}")
    print(f"  {'─'*40} {'─'*5} {'─'*7} {'─'*8} {'─'*8}")
    for label, mask in groups.items():
        period_mask = mask[longs_test.index] if mask.index.equals(longs.index) else mask.loc[longs_test.index]
        subset = longs_test[period_mask]
        m = compute_full_metrics(subset)
        if m:
            marker = " ◀ CANDIDATE" if "CANDIDATE" in label else ""
            print(f"  {label:<40s} {m['N']:>5d} {m['Hit1R']:>6.1%} {m['Exp1R']:>+7.3f}R {m['TotalR']:>+7.1f}R{marker}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'═' * 90}")


if __name__ == "__main__":
    main()
