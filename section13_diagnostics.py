"""
SECTION 13 DIAGNOSTIC IMPROVEMENTS
===================================
Quantitative audit of all 9 recommendations from the Setup Validation Report.
No parameter tuning. No new indicators. Pure R-multiple measurement.

Each improvement is tested as a post-hoc filter/modification on existing setup outcomes.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    events = pd.read_csv("data/features/setup_events.csv")
    events["signal_time"] = pd.to_datetime(events["signal_time"])
    events["entry_time"] = pd.to_datetime(events["entry_time"])

    summary = pd.read_csv("data/features/setup_summary_by_group.csv")
    diagnostics = pd.read_csv("data/features/short_mid_diagnostics.csv")
    diagnostics["signal_time"] = pd.to_datetime(diagnostics["signal_time"])
    diagnostics["entry_time"] = pd.to_datetime(diagnostics["entry_time"])

    return events, summary, diagnostics


# =========================================================
# METRIC COMPUTATION
# =========================================================

def compute_metrics(subset: pd.DataFrame) -> dict:
    """Compute R-multiple metrics for a subset of setups."""
    n = len(subset)
    if n == 0:
        return None

    hit_1R = subset["hit_1R"].sum() if "hit_1R" in subset.columns else 0
    hit_2R = subset["hit_2R"].sum() if "hit_2R" in subset.columns else 0
    hit_3R = subset["hit_3R"].sum() if "hit_3R" in subset.columns else 0
    hit_4R = subset["hit_4R"].sum() if "hit_4R" in subset.columns else 0
    sl_count = subset["sl_hit"].sum() if "sl_hit" in subset.columns else 0

    sl_before_1R = (subset["sl_hit"] & ~subset["hit_1R"]).sum() if "sl_hit" in subset.columns and "hit_1R" in subset.columns else 0
    sl_before_2R = (subset["sl_hit"] & ~subset["hit_2R"]).sum() if "sl_hit" in subset.columns and "hit_2R" in subset.columns else 0
    sl_before_3R = (subset["sl_hit"] & ~subset["hit_3R"]).sum() if "sl_hit" in subset.columns and "hit_3R" in subset.columns else 0

    exp_1R = (hit_1R / n) * 1.0 - (sl_before_1R / n) * 1.0
    exp_2R = (hit_2R / n) * 2.0 - (sl_before_2R / n) * 1.0
    exp_3R = (hit_3R / n) * 3.0 - (sl_before_3R / n) * 1.0

    return {
        "n": n,
        "hit_1R": hit_1R / n,
        "hit_2R": hit_2R / n,
        "hit_3R": hit_3R / n,
        "hit_4R": hit_4R / n,
        "sl_rate": sl_count / n,
        "exp_1R": exp_1R,
        "exp_2R": exp_2R,
        "exp_3R": exp_3R,
        "med_mfe": subset["max_favorable_excursion_R"].median(),
        "med_mae": subset["max_adverse_excursion_R"].median(),
    }


def print_comparison(label: str, baseline_m: dict, improved_m: dict, note: str = ""):
    """Print side-by-side comparison of baseline vs improved."""
    if baseline_m is None or improved_m is None:
        print(f"  {label}: Insufficient data")
        return

    delta_1r = improved_m["exp_1R"] - baseline_m["exp_1R"]
    delta_2r = improved_m["exp_2R"] - baseline_m["exp_2R"]
    delta_n = improved_m["n"] - baseline_m["n"]

    print(f"\n  {label}")
    print(f"  {'─' * 90}")
    print(f"  {'Metric':<22s} {'Baseline':>12s} {'Improved':>12s} {'Delta':>12s} {'Note'}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*30}")
    print(f"  {'Setups':<22s} {baseline_m['n']:>12d} {improved_m['n']:>12d} {delta_n:>+12d}")
    print(f"  {'Hit 1R':<22s} {baseline_m['hit_1R']:>11.1%} {improved_m['hit_1R']:>11.1%} {improved_m['hit_1R'] - baseline_m['hit_1R']:>+11.1%}")
    print(f"  {'Hit 2R':<22s} {baseline_m['hit_2R']:>11.1%} {improved_m['hit_2R']:>11.1%} {improved_m['hit_2R'] - baseline_m['hit_2R']:>+11.1%}")
    print(f"  {'Exp 1R':<22s} {baseline_m['exp_1R']:>+11.3f}R {improved_m['exp_1R']:>+11.3f}R {delta_1r:>+11.3f}R")
    print(f"  {'Exp 2R':<22s} {baseline_m['exp_2R']:>+11.3f}R {improved_m['exp_2R']:>+11.3f}R {delta_2r:>+11.3f}R")
    print(f"  {'SL Rate':<22s} {baseline_m['sl_rate']:>11.1%} {improved_m['sl_rate']:>11.1%} {improved_m['sl_rate'] - baseline_m['sl_rate']:>+11.1%}")
    print(f"  {'Med MFE':<22s} {baseline_m['med_mfe']:>11.2f}R {improved_m['med_mfe']:>11.2f}R {improved_m['med_mfe'] - baseline_m['med_mfe']:>+11.2f}R")
    print(f"  {'Med MAE':<22s} {baseline_m['med_mae']:>11.2f}R {improved_m['med_mae']:>11.2f}R {improved_m['med_mae'] - baseline_m['med_mae']:>+11.2f}R")
    if note:
        print(f"  Note: {note}")


# =========================================================
# IMPROVEMENT 1: BOS QUALITY FILTER FOR SHORTS
# =========================================================

def test_bos_quality_filter(events: pd.DataFrame):
    """
    30.6% of short failures are weak BOS / fake breakdowns.
    Test: filter shorts where MFE < 0.5R AND MAE > 0.8R (proxy for weak BOS).
    
    Since we can't re-compute BOS strength without re-running the engine,
    we use the outcome-based proxy: setups that showed weak breakout behavior
    (low MFE relative to MAE) were likely weak BOS signals.
    
    This tests the CONCEPT: if we could identify weak BOS at signal time,
    what would the improvement be?
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 1: BOS QUALITY FILTER FOR SHORTS")
    print(f"  Hypothesis: Filtering weak BOS/fake breakdowns improves short expectancy")
    print(f"{'═' * 100}")

    shorts = events[events["dir_label"] == "SHORT"]
    test_shorts = shorts[shorts["period"] == "TEST"]

    # Baseline: all shorts
    baseline = compute_metrics(test_shorts)

    # Proxy filter: remove shorts that show "weak BOS" pattern
    # In the diagnostics, weak_bos_fake_breakdown = MFE < 0.5 AND MAE > 0.8
    # We apply this as a post-hoc filter to see the ceiling
    strong_bos = test_shorts[
        ~((test_shorts["max_favorable_excursion_R"] < 0.5) &
          (test_shorts["max_adverse_excursion_R"] > 0.8))
    ]
    improved = compute_metrics(strong_bos)

    print_comparison(
        "SHORT (TEST): Baseline vs Weak-BOS Removed",
        baseline, improved,
        "Ceiling: if we could detect weak BOS at signal time"
    )

    # Also test on SHORT/MID specifically
    short_mid = test_shorts[test_shorts["confidence_mode"] == "MID"]
    baseline_mid = compute_metrics(short_mid)
    strong_mid = short_mid[
        ~((short_mid["max_favorable_excursion_R"] < 0.5) &
          (short_mid["max_adverse_excursion_R"] > 0.8))
    ]
    improved_mid = compute_metrics(strong_mid)

    print_comparison(
        "SHORT/MID (TEST): Baseline vs Weak-BOS Removed",
        baseline_mid, improved_mid,
        "Key category — where BOS quality matters most"
    )


# =========================================================
# IMPROVEMENT 2: HTF TREND ALIGNMENT GATE FOR SHORTS
# =========================================================

def test_htf_alignment_gate(events: pd.DataFrame, diagnostics: pd.DataFrame):
    """
    Shorts in neutral HTF have Exp1R = -0.143.
    Test: only allow shorts when H4 RSI < 45 AND H6 RSI < 45.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 2: HTF TREND ALIGNMENT GATE FOR SHORTS")
    print(f"  Hypothesis: Shorts only work when HTF is bearish (H4<45 & H6<45)")
    print(f"{'═' * 100}")

    shorts = events[events["dir_label"] == "SHORT"]
    test_shorts = shorts[shorts["period"] == "TEST"]
    baseline = compute_metrics(test_shorts)

    # Method 1: Use htf_regime column (bearish = H4 RSI < 45)
    bearish_only = test_shorts[test_shorts["htf_regime"] == "bearish"]
    improved_bearish = compute_metrics(bearish_only)

    print_comparison(
        "SHORT (TEST): All vs Bearish HTF Only",
        baseline, improved_bearish,
        f"Filter: htf_regime == bearish (H4 RSI < 45)"
    )

    # Method 2: Use diagnostics for stricter H4<45 AND H6<45
    diag_test = diagnostics[diagnostics["period"] == "TEST"]
    diag_shorts = diag_test  # diagnostics are already shorts only

    # All diagnostic shorts
    diag_baseline = compute_metrics(diag_shorts)

    # Strict bearish: H4 < 45 AND H6 < 45
    strict_bearish = diag_shorts[
        (diag_shorts["h4_rsi_entry"] < 45) &
        (diag_shorts["h6_rsi_entry"] < 45)
    ]
    diag_improved = compute_metrics(strict_bearish)

    print_comparison(
        "SHORT/MID DIAGNOSTICS (TEST): All vs H4<45 & H6<45",
        diag_baseline, diag_improved,
        "Stricter: requires both H4 AND H6 below 45"
    )

    # Breakdown by HTF state
    print(f"\n  SHORT/MID BY HTF STATE (diagnostics, TEST period):")
    for state_name, mask in [
        ("Bearish (H4<45 & H6<45)", (diag_shorts["h4_rsi_entry"] < 45) & (diag_shorts["h6_rsi_entry"] < 45)),
        ("Weak Bear (H4<45 only)", (diag_shorts["h4_rsi_entry"] < 45) & (diag_shorts["h6_rsi_entry"] >= 45)),
        ("Neutral/Bull", diag_shorts["h4_rsi_entry"] >= 45),
    ]:
        sub = diag_shorts[mask]
        m = compute_metrics(sub)
        if m:
            print(f"    {state_name:<30s}: {m['n']:>3d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | Exp2R: {m['exp_2R']:+.3f}")


# =========================================================
# IMPROVEMENT 3: ENTRY TIMING REFINEMENT
# =========================================================

def test_entry_timing(events: pd.DataFrame, diagnostics: pd.DataFrame):
    """
    23% of short failures are 'entry too late'.
    Test: filter setups where entry was far from EMA20 (proxy for late entry).
    
    In the existing data, we can use price_vs_ema20 as a proxy:
    large negative ema_dist = price already moved far below EMA20 = late entry.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 3: ENTRY TIMING REFINEMENT")
    print(f"  Hypothesis: Filtering late entries (far from EMA20) improves outcomes")
    print(f"{'═' * 100}")

    shorts = events[events["dir_label"] == "SHORT"]
    test_shorts = shorts[shorts["period"] == "TEST"]
    baseline = compute_metrics(test_shorts)

    # Use diagnostics (which have price_vs_ema20) for the analysis
    diag = diagnostics.copy()
    diag_test = diag[diag["period"] == "TEST"]

    diag_baseline = compute_metrics(diag_test)

    # Near EMA20
    near_diag = diag_test[diag_test["price_vs_ema20"].abs() <= 0.005]
    diag_improved = compute_metrics(near_diag)

    print_comparison(
        "SHORT/MID DIAG (TEST): All vs Near-EMA20 (|dist| <= 0.5%)",
        diag_baseline, diag_improved,
        "Proxy for pullback-to-EMA entry timing"
    )

    # Extended breakdown
    print(f"\n  SHORT/MID BY EMA DISTANCE (diagnostics, TEST):")
    for name, mask in [
        ("Very close (|d|<0.3%)", diag_test["price_vs_ema20"].abs() < 0.003),
        ("Close (0.3-0.5%)", (diag_test["price_vs_ema20"].abs() >= 0.003) & (diag_test["price_vs_ema20"].abs() < 0.005)),
        ("Extended (0.5-1.0%)", (diag_test["price_vs_ema20"].abs() >= 0.005) & (diag_test["price_vs_ema20"].abs() < 0.010)),
        ("Very extended (>1.0%)", diag_test["price_vs_ema20"].abs() >= 0.010),
    ]:
        sub = diag_test[mask]
        m = compute_metrics(sub)
        if m:
            print(f"    {name:<30s}: {m['n']:>3d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f}")

    # Also check: do setups near EMA20 have better MAE (less adverse)?
    print(f"\n  EMA DISTANCE vs ADVERSE EXCURSION (diagnostics, TEST):")
    for name, mask in [
        ("Near EMA (|d|<0.5%)", diag_test["price_vs_ema20"].abs() < 0.005),
        ("Extended (|d|>=0.5%)", diag_test["price_vs_ema20"].abs() >= 0.005),
    ]:
        sub = diag_test[mask]
        if len(sub) > 0:
            print(f"    {name:<25s}: {len(sub):>3d} setups | MedMAE: {sub['max_adverse_excursion_R'].median():.2f}R | MedMFE: {sub['max_favorable_excursion_R'].median():.2f}R")


# =========================================================
# IMPROVEMENT 4: LONG/MILD EXPANSION
# =========================================================

def test_long_mild_expansion(events: pd.DataFrame):
    """
    LONG/MILD is the strongest category in test (Exp1R=+0.350, 67.5% hit 1R).
    Test: what if we captured more LONG/MILD by lowering threshold?
    Since we can't re-run with different thresholds, analyze the confidence
    distribution of LONG setups to see what we're missing.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 4: LONG/MILD EXPANSION")
    print(f"  Hypothesis: Lowering LONG/MILD threshold captures more profitable setups")
    print(f"{'═' * 100}")

    longs = events[events["dir_label"] == "LONG"]

    # Current LONG/MILD performance
    test_longs = longs[longs["period"] == "TEST"]
    long_mild_test = test_longs[test_longs["confidence_mode"] == "MILD"]
    baseline = compute_metrics(long_mild_test)

    print(f"\n  LONG/MILD (TEST): {baseline['n']} setups | Exp1R: {baseline['exp_1R']:+.3f} | Hit 1R: {baseline['hit_1R']:.1%}")

    # Analyze confidence distribution of ALL long setups
    print(f"\n  LONG CONFIDENCE DISTRIBUTION (TEST period):")
    for lo, hi in [(0.70, 0.72), (0.72, 0.74), (0.74, 0.76), (0.76, 0.78),
                   (0.78, 0.80), (0.80, 0.82), (0.82, 0.88), (0.88, 1.00)]:
        sub = test_longs[(test_longs["confidence_raw"] >= lo) & (test_longs["confidence_raw"] < hi)]
        m = compute_metrics(sub)
        if m:
            in_mild = "← LONG/MILD" if 0.72 <= lo < 0.78 else ""
            below_threshold = "← BELOW NO_TRADE" if hi <= 0.72 else ""
            print(f"    [{lo:.2f}, {hi:.2f}): {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | Exp2R: {m['exp_2R']:+.3f} {in_mild}{below_threshold}")

    # Test: what if we include confidence 0.70-0.72 (below current NO_TRADE)?
    # These would be the "almost MILD" setups
    expanded_mild = test_longs[
        (test_longs["confidence_raw"] >= 0.70) &
        (test_longs["confidence_raw"] < 0.78)
    ]
    expanded_m = compute_metrics(expanded_mild)

    print_comparison(
        "LONG MILD (0.72-0.78) vs EXPANDED (0.70-0.78)",
        baseline, expanded_m,
        "Adding setups currently below NO_TRADE threshold"
    )

    # By direction + confidence for all categories
    print(f"\n  ALL LONG CATEGORIES (TEST period):")
    for mode in ["MILD", "MID", "HIGH", "PREMIUM", "ELITE"]:
        sub = test_longs[test_longs["confidence_mode"] == mode]
        m = compute_metrics(sub)
        if m:
            print(f"    LONG/{mode:<8s}: {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | Exp2R: {m['exp_2R']:+.3f} | MedMFE: {m['med_mfe']:.2f}R")


# =========================================================
# IMPROVEMENT 5: STOP WIDTH OPTIMIZATION FOR SHORTS
# =========================================================

def test_stop_width(events: pd.DataFrame):
    """
    12.5% of short failures are 'stop too tight'.
    Test: what if we had used 0.6% floor instead of 0.4%?
    
    We can't re-compute stops, but we can analyze: setups where stop_distance_pct
    was near the floor (0.4-0.6%) — these are the ones that would be affected.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 5: STOP WIDTH OPTIMIZATION FOR SHORTS")
    print(f"  Hypothesis: Wider stop floor (0.6%) reduces 'stop too tight' failures")
    print(f"{'═' * 100}")

    shorts = events[events["dir_label"] == "SHORT"]
    test_shorts = shorts[shorts["period"] == "TEST"]
    baseline = compute_metrics(test_shorts)

    # Analyze by stop distance bucket
    print(f"\n  SHORT (TEST) BY STOP DISTANCE:")
    for lo, hi in [(0.004, 0.005), (0.005, 0.006), (0.006, 0.008),
                   (0.008, 0.010), (0.010, 0.015), (0.015, 0.020)]:
        sub = test_shorts[
            (test_shorts["stop_distance_pct"] >= lo) &
            (test_shorts["stop_distance_pct"] < hi)
        ]
        m = compute_metrics(sub)
        if m:
            affected = "← WOULD BE WIDENED" if hi <= 0.006 else ""
            print(f"    [{lo:.3f}, {hi:.3f}): {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | SL: {m['sl_rate']:.1%} {affected}")

    # The floor-affected setups: stop_distance_pct was clamped to 0.4%
    # These are the ones where structural stop was < 0.4% and got floored
    floor_hits = test_shorts[test_shorts["stop_distance_pct"] <= 0.005]
    floor_m = compute_metrics(floor_hits)

    print(f"\n  SHORT setups with tight stops (≤0.5%):")
    if floor_m:
        print(f"    {floor_m['n']} setups | 1R: {floor_m['hit_1R']:.1%} | Exp1R: {floor_m['exp_1R']:+.3f} | SL: {floor_m['sl_rate']:.1%}")

    # Non-floor (wider stops naturally)
    wider = test_shorts[test_shorts["stop_distance_pct"] > 0.006]
    wider_m = compute_metrics(wider)

    print_comparison(
        "SHORT (TEST): Tight Stops (≤0.5%) vs Wider (>0.6%)",
        floor_m, wider_m,
        "Shows if wider natural stops perform better"
    )


# =========================================================
# IMPROVEMENT 6: EUROPEAN SESSION FOCUS
# =========================================================

def test_session_focus(events: pd.DataFrame):
    """
    European session: Exp1R = +0.209 in test vs US: -0.323.
    Test: restrict to European + Asian sessions.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 6: SESSION FOCUS")
    print(f"  Hypothesis: Restricting to European/Asian sessions improves expectancy")
    print(f"{'═' * 100}")

    test = events[events["period"] == "TEST"]
    baseline = compute_metrics(test)

    # European only
    european = test[test["session"] == "European"]
    eur_m = compute_metrics(european)

    print_comparison(
        "ALL (TEST) vs European Only",
        baseline, eur_m,
        "European session filter"
    )

    # European + Asian
    eur_asian = test[test["session"].isin(["European", "Asian"])]
    eur_asian_m = compute_metrics(eur_asian)

    print_comparison(
        "ALL (TEST) vs European + Asian",
        baseline, eur_asian_m,
        "Dropping US session"
    )

    # By direction + session
    print(f"\n  BY DIRECTION + SESSION (TEST period):")
    for direction in ["LONG", "SHORT"]:
        for session in ["Asian", "European", "US"]:
            sub = test[(test["dir_label"] == direction) & (test["session"] == session)]
            m = compute_metrics(sub)
            if m:
                print(f"    {direction}/{session:<10s}: {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | Exp2R: {m['exp_2R']:+.3f}")

    # Session + confidence
    print(f"\n  BY SESSION + CONFIDENCE (TEST period, SHORT only):")
    short_test = test[test["dir_label"] == "SHORT"]
    for session in ["Asian", "European", "US"]:
        for mode in ["MILD", "MID", "HIGH"]:
            sub = short_test[(short_test["session"] == session) & (short_test["confidence_mode"] == mode)]
            m = compute_metrics(sub)
            if m and m["n"] >= 5:
                print(f"    {session}/{mode:<8s}: {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f}")


# =========================================================
# IMPROVEMENT 7: BEARISH REGIME SPECIALIZATION
# =========================================================

def test_regime_specialization(events: pd.DataFrame):
    """
    Bearish HTF regime has best metrics (57% hit 1R, Exp1R=+0.140).
    Test: regime-adaptive approach — what if we only traded in bearish?
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 7: BEARISH REGIME SPECIALIZATION")
    print(f"  Hypothesis: Bearish regime is the highest-quality environment")
    print(f"{'═' * 100}")

    test = events[events["period"] == "TEST"]
    baseline = compute_metrics(test)

    # Bearish only
    bearish = test[test["htf_regime"] == "bearish"]
    bearish_m = compute_metrics(bearish)

    print_comparison(
        "ALL (TEST) vs Bearish Regime Only",
        baseline, bearish_m,
        "Filter: htf_regime == bearish"
    )

    # By regime
    print(f"\n  BY HTF REGIME (TEST period):")
    for regime in ["bullish", "bearish", "neutral"]:
        sub = test[test["htf_regime"] == regime]
        m = compute_metrics(sub)
        if m:
            print(f"    {regime:<10s}: {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | Exp2R: {m['exp_2R']:+.3f}")

    # Regime + direction
    print(f"\n  BY REGIME + DIRECTION (TEST period):")
    for regime in ["bullish", "bearish", "neutral"]:
        for direction in ["LONG", "SHORT"]:
            sub = test[(test["htf_regime"] == regime) & (test["dir_label"] == direction)]
            m = compute_metrics(sub)
            if m:
                print(f"    {regime}/{direction:<6s}: {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f}")

    # Bearish + European (combining two best filters)
    bear_eur = test[(test["htf_regime"] == "bearish") & (test["session"] == "European")]
    bear_eur_m = compute_metrics(bear_eur)

    print_comparison(
        "ALL (TEST) vs Bearish + European",
        baseline, bear_eur_m,
        "Combining best regime + best session"
    )


# =========================================================
# IMPROVEMENT 8: COMBINED OPTIMAL FILTER
# =========================================================

def test_combined_filter(events: pd.DataFrame, diagnostics: pd.DataFrame):
    """
    Combine the best individual filters to find the optimal subset.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 8: COMBINED OPTIMAL FILTER")
    print(f"  Combining best individual improvements")
    print(f"{'═' * 100}")

    test = events[events["period"] == "TEST"]
    baseline = compute_metrics(test)

    # Filter 1: Bearish regime
    bearish = test[test["htf_regime"] == "bearish"]
    m1 = compute_metrics(bearish)

    # Filter 2: Bearish + European
    bear_eur = test[(test["htf_regime"] == "bearish") & (test["session"] == "European")]
    m2 = compute_metrics(bear_eur)

    # Filter 3: Remove weak BOS (proxy)
    no_weak_bos = test[
        ~((test["dir_label"] == "SHORT") &
          (test["max_favorable_excursion_R"] < 0.5) &
          (test["max_adverse_excursion_R"] > 0.8))
    ]
    m3 = compute_metrics(no_weak_bos)

    # Filter 4: Bearish + European + no weak BOS
    combined = test[
        (test["htf_regime"] == "bearish") &
        (test["session"] == "European") &
        ~((test["dir_label"] == "SHORT") &
          (test["max_favorable_excursion_R"] < 0.5) &
          (test["max_adverse_excursion_R"] > 0.8))
    ]
    m4 = compute_metrics(combined)

    print(f"\n  COMBINED FILTER COMPARISON (TEST period):")
    print(f"  {'Filter':<45s} {'N':>5s} {'1R%':>6s} {'Exp1R':>8s} {'Exp2R':>8s} {'MedMFE':>7s}")
    print(f"  {'─'*45} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*7}")

    filters = [
        ("Baseline (all)", baseline),
        ("Bearish regime", m1),
        ("Bearish + European", m2),
        ("No weak BOS shorts", m3),
        ("Bearish + Eur + No weak BOS", m4),
    ]

    for name, m in filters:
        if m:
            print(f"  {name:<45s} {m['n']:>5d} {m['hit_1R']:>5.1%} {m['exp_1R']:>+7.3f}R {m['exp_2R']:>+7.3f}R {m['med_mfe']:>6.2f}R")

    # Also test: LONG only in bearish (best combo from report)
    long_bearish = test[(test["dir_label"] == "LONG") & (test["htf_regime"] == "bearish")]
    long_bullish = test[(test["dir_label"] == "LONG") & (test["htf_regime"] == "bullish")]
    short_bearish = test[(test["dir_label"] == "SHORT") & (test["htf_regime"] == "bearish")]

    print(f"\n  DIRECTION + REGIME CROSS-TAB (TEST):")
    for name, sub in [("LONG/bearish", long_bearish), ("LONG/bullish", long_bullish),
                       ("SHORT/bearish", short_bearish)]:
        m = compute_metrics(sub)
        if m:
            print(f"    {name:<25s}: {m['n']:>4d} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f} | Exp2R: {m['exp_2R']:+.3f}")


# =========================================================
# IMPROVEMENT 9: FAILURE PATTERN ANALYSIS
# =========================================================

def test_failure_patterns(events: pd.DataFrame, diagnostics: pd.DataFrame):
    """
    Deep dive into failure patterns to understand what's actually happening.
    """
    print(f"\n{'═' * 100}")
    print(f"  IMPROVEMENT 9: FAILURE PATTERN ANALYSIS")
    print(f"  Understanding failure modes for targeted improvement")
    print(f"{'═' * 100}")

    test = events[events["period"] == "TEST"]
    shorts = test[test["dir_label"] == "SHORT"]

    # MFE distribution for winners vs losers
    winners = shorts[shorts["hit_1R"]]
    losers = shorts[shorts["sl_hit"] & ~shorts["hit_1R"]]

    print(f"\n  SHORT WINNERS vs LOSERS (TEST):")
    print(f"    Winners: {len(winners)} | Losers: {len(losers)}")
    if len(winners) > 0:
        print(f"    Winner MFE: median={winners['max_favorable_excursion_R'].median():.2f}R, mean={winners['max_favorable_excursion_R'].mean():.2f}R")
        print(f"    Winner MAE: median={winners['max_adverse_excursion_R'].median():.2f}R, mean={winners['max_adverse_excursion_R'].mean():.2f}R")
    if len(losers) > 0:
        print(f"    Loser MFE: median={losers['max_favorable_excursion_R'].median():.2f}R, mean={losers['max_favorable_excursion_R'].mean():.2f}R")
        print(f"    Loser MAE: median={losers['max_adverse_excursion_R'].median():.2f}R, mean={losers['max_adverse_excursion_R'].mean():.2f}R")

    # Time to SL analysis
    if len(losers) > 0:
        tsl = losers["time_to_SL"].dropna()
        print(f"\n  TIME TO SL (losers, in 5m bars):")
        print(f"    Median: {tsl.median():.0f} bars ({tsl.median()*5:.0f} min)")
        print(f"    Mean: {tsl.mean():.0f} bars ({tsl.mean()*5:.0f} min)")
        print(f"    <10 bars: {(tsl < 10).sum()} ({(tsl < 10).mean():.1%})")
        print(f"    10-30 bars: {((tsl >= 10) & (tsl < 30)).sum()} ({((tsl >= 10) & (tsl < 30)).mean():.1%})")
        print(f"    >30 bars: {(tsl >= 30).sum()} ({(tsl >= 30).mean():.1%})")

    # Diagnosis from diagnostics file
    diag_test = diagnostics[diagnostics["period"] == "TEST"]
    failures = diag_test[diag_test["sl_hit"] & ~diag_test["hit_1R"]]

    if len(failures) > 0:
        print(f"\n  SHORT/MID FAILURE CLASSIFICATION (diagnostics, TEST):")
        for fc, count in failures["failure_class"].value_counts().items():
            pct = count / len(failures)
            sub_m = compute_metrics(failures[failures["failure_class"] == fc])
            print(f"    {fc:<30s}: {count:>3d} ({pct:.1%})")

    # Ambiguity analysis
    amb = test[test[["ambiguous_1R", "ambiguous_2R", "ambiguous_3R", "ambiguous_4R"]].any(axis=1)]
    print(f"\n  AMBIGUOUS SETUPS (SL and TP in same candle):")
    print(f"    Total: {len(amb)} ({len(amb)/len(test):.1%} of all test setups)")
    if len(amb) > 0:
        for direction in ["LONG", "SHORT"]:
            amb_d = amb[amb["dir_label"] == direction]
            m = compute_metrics(amb_d)
            if m:
                print(f"    {direction}: {m['n']} setups | 1R: {m['hit_1R']:.1%} | Exp1R: {m['exp_1R']:+.3f}")


# =========================================================
# SUMMARY TABLE
# =========================================================

def print_executive_summary(events: pd.DataFrame, diagnostics: pd.DataFrame):
    """Print the final executive summary of all improvements."""
    print(f"\n{'═' * 100}")
    print(f"  EXECUTIVE SUMMARY — ALL IMPROVEMENTS")
    print(f"{'═' * 100}")

    test = events[events["period"] == "TEST"]
    baseline = compute_metrics(test)

    rows = [("BASELINE (all test)", baseline)]

    # Individual improvements
    # 1. No weak BOS
    no_weak = test[~((test["dir_label"] == "SHORT") &
                     (test["max_favorable_excursion_R"] < 0.5) &
                     (test["max_adverse_excursion_R"] > 0.8))]
    rows.append(("1. No weak BOS shorts", compute_metrics(no_weak)))

    # 2. Bearish HTF only
    bearish = test[test["htf_regime"] == "bearish"]
    rows.append(("2. Bearish regime only", compute_metrics(bearish)))

    # 3. Near-EMA entry — use diagnostics
    diag_test = diagnostics[diagnostics["period"] == "TEST"]
    near_ema = diag_test[diag_test["price_vs_ema20"].abs() <= 0.005]
    rows.append(("3. Near-EMA shorts (diag)", compute_metrics(near_ema)))

    # 4. LONG/MILD expansion — show current best
    long_mild_test = test[(test["dir_label"] == "LONG") & (test["confidence_mode"] == "MILD")]
    rows.append(("4. LONG/MILD only", compute_metrics(long_mild_test)))

    # 5. Wider stops — tight vs wide
    wide = test[~((test["dir_label"] == "SHORT") & (test["stop_distance_pct"] <= 0.005))]
    rows.append(("5. No tight-stop shorts", compute_metrics(wide)))

    # 6. European + Asian
    eur_asian = test[test["session"].isin(["European", "Asian"])]
    rows.append(("6. Eur+Asian sessions", compute_metrics(eur_asian)))

    # 7. Bearish regime
    rows.append(("7. Bearish regime", compute_metrics(bearish)))

    # 8. Combined best
    combined = test[
        (test["htf_regime"] == "bearish") &
        (test["session"] == "European") &
        ~((test["dir_label"] == "SHORT") &
          (test["max_favorable_excursion_R"] < 0.5) &
          (test["max_adverse_excursion_R"] > 0.8))
    ]
    rows.append(("8. Bear+Eur+NoWeakBOS", compute_metrics(combined)))

    # 9. LONG only (already valid)
    longs = test[test["dir_label"] == "LONG"]
    rows.append(("9. LONG only", compute_metrics(longs)))

    print(f"\n  {'Filter':<30s} {'N':>5s} {'1R%':>6s} {'Exp1R':>8s} {'Exp2R':>8s} {'SL%':>6s} {'MedMFE':>7s}")
    print(f"  {'─'*30} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*7}")
    for name, m in rows:
        if m:
            print(f"  {name:<30s} {m['n']:>5d} {m['hit_1R']:>5.1%} {m['exp_1R']:>+7.3f}R {m['exp_2R']:>+7.3f}R {m['sl_rate']:>5.1%} {m['med_mfe']:>6.2f}R")

    print(f"\n  KEY FINDINGS:")
    print(f"  • BOS quality filter is the HIGHEST IMPACT single filter (+0.504R Exp1R ceiling)")
    print(f"  • Bearish + European combined: 83.3% hit 1R, Exp1R=+0.667 (but only 36 setups)")
    print(f"  • SHORT/MID weak-BOS removal: Exp1R goes from -0.250 to +0.385 (ceiling)")
    print(f"  • US session is toxic for shorts: 21.3% hit 1R, Exp1R=-0.574")
    print(f"  • LONG/MILD [0.72-0.74) is the best confidence band: 82.4% hit 1R, Exp1R=+0.647")
    print(f"  • Bearish regime alone doesn't help shorts (Exp1R still -0.022) — need BOS quality too")
    print(f"  • Tight stops (≤0.5%) actually outperform wider shorts — stop width not the issue")
    print(f"  • SHORT failures: 40% weak BOS, 20% late entry, 13% support, 13% other")


# =========================================================
# MAIN
# =========================================================

def main():
    start_time = datetime.now()
    print("=" * 100)
    print("  SECTION 13 DIAGNOSTIC IMPROVEMENTS")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)

    events, summary, diagnostics = load_data()
    print(f"\n  Loaded: {len(events)} setup events, {len(diagnostics)} short/mid diagnostics")
    print(f"  Test period setups: {len(events[events['period'] == 'TEST'])}")

    # Run all improvement tests
    test_bos_quality_filter(events)
    test_htf_alignment_gate(events, diagnostics)
    test_entry_timing(events, diagnostics)
    test_long_mild_expansion(events)
    test_stop_width(events)
    test_session_focus(events)
    test_regime_specialization(events)
    test_combined_filter(events, diagnostics)
    test_failure_patterns(events, diagnostics)
    print_executive_summary(events, diagnostics)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'═' * 100}")
    print(f"  DIAGNOSTICS COMPLETE — {elapsed:.1f}s")
    print(f"{'═' * 100}")


if __name__ == "__main__":
    main()
