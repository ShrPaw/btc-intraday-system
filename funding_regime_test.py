"""
FUNDING × REGIME — CONDITIONED SIGNAL TEST
===========================================
Tests: High funding predicts negative returns ONLY in trending regimes.

Rules:
  - BTCUSDT only
  - Funding data only
  - Non-overlapping events (24h cooldown)
  - Regime defined from past data only (no lookahead)
  - Compare against same-regime random baselines
  - Do NOT call it alpha unless it beats same-regime random
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data/features"
COOLDOWN = 24  # bars (hours) between events


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════

def load_data():
    """Load price and funding, merge onto 1h timeline."""
    price_1m = pd.read_csv(f"{DATA_DIR}/btcusdt_1m.csv")
    price_1m["timestamp"] = pd.to_datetime(price_1m["timestamp"])
    price_1h = price_1m.set_index("timestamp").resample("1h").agg({
        "price": ["first", "max", "min", "last"],
        "volume": "sum"
    }).dropna()
    price_1h.columns = ["open", "high", "low", "close", "volume"]
    price_1h = price_1h.reset_index()

    funding = pd.read_csv(f"{DATA_DIR}/btcusdt_funding.csv")
    funding["timestamp"] = pd.to_datetime(funding["timestamp"])

    df = price_1h.set_index("timestamp")
    fr = funding.set_index("timestamp")["fundingRate"]
    df["fundingRate"] = fr.reindex(df.index, method="ffill")
    df = df.reset_index()

    # Forward returns
    for h in [4, 8, 24]:
        df[f"fwd_ret_{h}h"] = df["close"].shift(-h) / df["close"] - 1

    # 7-day return for regime (past only)
    df["ret_7d"] = df["close"].pct_change(168)

    # Funding stats (past only)
    df["funding_cum_24h"] = df["fundingRate"].rolling(3, min_periods=1).sum()

    return df[df["fundingRate"].notna()].copy()


# ═══════════════════════════════════════════════════════════
# REGIME — PAST DATA ONLY
# ═══════════════════════════════════════════════════════════

def classify_regime(df, lookback=168):
    """
    Regime from past N bars of returns.
    Uses only past data (no lookahead).
    """
    ret = df["close"].pct_change(lookback)
    df["regime"] = "neutral"
    df.loc[ret > 0.05, "regime"] = "bullish"
    df.loc[ret < -0.05, "regime"] = "bearish"
    return df


# ═══════════════════════════════════════════════════════════
# NON-OVERLAPPING EVENTS
# ═══════════════════════════════════════════════════════════

def non_overlapping_indices(df, mask, cooldown=COOLDOWN):
    """Return indices of events with minimum cooldown between them."""
    indices = df.index[mask].tolist()
    selected = []
    last = -cooldown - 1
    for idx in indices:
        if idx - last >= cooldown:
            selected.append(idx)
            last = idx
    return selected


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(rets):
    """Standard metrics for a return series."""
    if len(rets) < 3:
        return None
    n = len(rets)
    mean = rets.mean()
    median = rets.median()
    std = rets.std()
    wr = (rets > 0).mean()
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return {"N": n, "mean": mean, "median": median, "std": std,
            "winrate": wr, "t_stat": t_stat, "p_val": p_val}


def bootstrap_ci(rets, n_boot=10000):
    """95% bootstrap CI for mean."""
    if len(rets) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    boot = rng.choice(rets, size=(n_boot, len(rets)), replace=True).mean(axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def random_baseline(df, mask, horizon, n_sims=1000, cooldown=COOLDOWN):
    """
    Same-regime random baseline.
    For each event timestamp, sample a random bar from the same regime
    with the same cooldown constraint.
    """
    rng = np.random.default_rng(42)
    event_indices = df.index[mask].tolist()

    # Get regime for each event
    event_regimes = df.loc[event_indices, "regime"].values

    # Pool of valid bars per regime
    pools = {}
    for regime in ["bullish", "bearish", "neutral"]:
        regime_mask = df["regime"] == regime
        valid = df[regime_mask & df[f"fwd_ret_{horizon}h"].notna()].index.tolist()
        pools[regime] = valid

    col = f"fwd_ret_{horizon}h"
    random_means = []

    for _ in range(n_sims):
        sampled = []
        last_idx = -cooldown - 1
        for i, event_idx in enumerate(event_indices):
            regime = event_regimes[i]
            pool = pools.get(regime, [])
            if not pool:
                continue
            # Sample with cooldown
            attempts = 0
            while attempts < 50:
                candidate = rng.choice(pool)
                if abs(candidate - last_idx) >= cooldown:
                    sampled.append(candidate)
                    last_idx = candidate
                    break
                attempts += 1

        if sampled:
            rets = df.loc[sampled, col].dropna()
            if len(rets) > 0:
                random_means.append(rets.mean())

    return np.array(random_means) if random_means else np.array([])


# ═══════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════

def test_regime_conditioned(df, regime, threshold_name, mask):
    """Test high funding in a specific regime."""
    regime_mask = df["regime"] == regime
    combined = mask & regime_mask

    # Non-overlapping
    indices = non_overlapping_indices(df, combined)
    if len(indices) < 3:
        return None

    results = {"regime": regime, "threshold": threshold_name, "N": len(indices)}

    for h in [4, 8, 24]:
        col = f"fwd_ret_{h}h"
        rets = df.loc[indices, col].dropna()
        m = compute_metrics(rets)
        if m:
            ci_lo, ci_hi = bootstrap_ci(rets.values)
            results[f"{h}h_mean"] = m["mean"]
            results[f"{h}h_median"] = m["median"]
            results[f"{h}h_wr"] = m["winrate"]
            results[f"{h}h_p"] = m["p_val"]
            results[f"{h}h_ci_lo"] = ci_lo
            results[f"{h}h_ci_hi"] = ci_hi

            # Random baseline
            rand = random_baseline(df, combined, h, n_sims=1000)
            if len(rand) > 0:
                results[f"{h}h_rand_mean"] = rand.mean()
                results[f"{h}h_delta"] = m["mean"] - rand.mean()
                # P(event < random) — one-sided
                results[f"{h}h_p_vs_rand"] = (rand <= m["mean"]).mean()

    return results


def print_result(r, h):
    """Print one result row."""
    mean = r.get(f"{h}h_mean", np.nan)
    median = r.get(f"{h}h_median", np.nan)
    wr = r.get(f"{h}h_wr", np.nan)
    p = r.get(f"{h}h_p", np.nan)
    ci_lo = r.get(f"{h}h_ci_lo", np.nan)
    ci_hi = r.get(f"{h}h_ci_hi", np.nan)
    rand_mean = r.get(f"{h}h_rand_mean", np.nan)
    delta = r.get(f"{h}h_delta", np.nan)
    p_rand = r.get(f"{h}h_p_vs_rand", np.nan)

    sig = "✓" if p < 0.05 else ("~" if p < 0.10 else " ")
    rand_sig = "✓" if p_rand < 0.05 else ("~" if p_rand < 0.10 else " ") if not np.isnan(p_rand) else " "

    mean_s = f"{mean*100:+.3f}%" if not np.isnan(mean) else "     —"
    median_s = f"{median*100:+.3f}%" if not np.isnan(median) else "     —"
    wr_s = f"{wr:.0%}" if not np.isnan(wr) else "  —"
    p_s = f"{p:.3f}{sig}" if not np.isnan(p) else "    —"
    ci_s = f"[{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%]" if not np.isnan(ci_lo) else "       —"
    rand_s = f"{rand_mean*100:+.3f}%" if not np.isnan(rand_mean) else "     —"
    delta_s = f"{delta*100:+.3f}%" if not np.isnan(delta) else "     —"
    p_rand_s = f"{p_rand:.3f}{rand_sig}" if not np.isnan(p_rand) else "    —"

    return mean_s, median_s, wr_s, p_s, ci_s, rand_s, delta_s, p_rand_s


def main():
    start_time = datetime.now()
    print("=" * 90)
    print("  FUNDING × REGIME — CONDITIONED SIGNAL TEST")
    print("  Hypothesis: High funding → neg returns ONLY in trending regimes")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Load
    print("\n  Loading data...")
    global df
    df = load_data()
    df = classify_regime(df)
    print(f"  {len(df)} bars ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    print(f"  Regime distribution:")
    for r in ["bullish", "bearish", "neutral"]:
        n = (df["regime"] == r).sum()
        print(f"    {r}: {n} bars ({n/len(df)*100:.1f}%)")

    # Unconditional baseline
    print(f"\n  UNCONDITIONAL BASELINE (all bars):")
    for h in [4, 8, 24]:
        col = f"fwd_ret_{h}h"
        rets = df[col].dropna()
        m = compute_metrics(rets)
        if m:
            print(f"    {h}h: mean={m['mean']*100:+.4f}%, median={m['median']*100:+.4f}%, wr={m['winrate']:.1%}, N={m['N']}")

    # Funding thresholds
    fr = df["fundingRate"]
    p80 = fr.quantile(0.80)
    p90 = fr.quantile(0.90)
    p95 = fr.quantile(0.95)

    thresholds = {
        "funding > p80": fr > p80,
        "funding > p90": fr > p90,
        "funding > p95": fr > p95,
    }

    regimes = ["bullish", "bearish", "neutral"]

    # ═══════════════════════════════════════════════════════
    # MAIN RESULTS TABLE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  NON-OVERLAPPING EVENTS (24h cooldown) — BY REGIME")
    print(f"{'═' * 90}")

    all_results = []
    for threshold_name, threshold_mask in thresholds.items():
        for regime in regimes:
            r = test_regime_conditioned(df, regime, threshold_name, threshold_mask)
            if r:
                all_results.append(r)

    # Print by horizon
    for h in [4, 8, 24]:
        print(f"\n  {h}h FORWARD RETURN:")
        print(f"  {'Threshold':<18s} {'Regime':<10s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'WR':>5s} {'p':>6s} {'CI':>24s} {'Rand':>8s} {'Delta':>8s} {'p_rand':>7s}")
        print(f"  {'─'*18} {'─'*10} {'─'*5} {'─'*8} {'─'*8} {'─'*5} {'─'*6} {'─'*24} {'─'*8} {'─'*8} {'─'*7}")

        for r in all_results:
            mean_s, median_s, wr_s, p_s, ci_s, rand_s, delta_s, p_rand_s = print_result(r, h)
            print(f"  {r['threshold']:<18s} {r['regime']:<10s} {r['N']:>5d} {mean_s} {median_s} {wr_s} {p_s} {ci_s} {rand_s} {delta_s} {p_rand_s}")

    # ═══════════════════════════════════════════════════════
    # ROLLING WINDOW STABILITY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  ROLLING 30-DAY STABILITY (p90, non-overlapping, 24h cooldown)")
    print(f"{'═' * 90}")

    window_size = 30 * 24
    step = 15 * 24
    p90_mask = df["fundingRate"] > p90

    print(f"\n  {'Window':<15s} {'Regime':<10s} {'N':>5s} {'4h_mean':>8s} {'4h_wr':>5s} {'8h_mean':>8s} {'24h_mean':>9s}")
    print(f"  {'─'*15} {'─'*10} {'─'*5} {'─'*8} {'─'*5} {'─'*8} {'─'*9}")

    for start in range(0, len(df) - window_size, step):
        window = df.iloc[start:start + window_size]
        window_start = window["timestamp"].min().strftime("%Y-%m-%d")

        for regime in ["bearish", "bullish", "neutral"]:
            mask = (window["fundingRate"] > p90) & (window["regime"] == regime)
            indices = non_overlapping_indices(window, mask)
            if len(indices) < 3:
                continue

            vals = {}
            for h in [4, 8, 24]:
                rets = window.loc[indices, f"fwd_ret_{h}h"].dropna()
                if len(rets) > 0:
                    vals[f"{h}h"] = rets.mean()

            if vals:
                v4 = vals.get('4h', np.nan)
                v8 = vals.get('8h', np.nan)
                v24 = vals.get('24h', np.nan)
                v4s = f"{v4*100:>+7.3f}%" if not np.isnan(v4) else "     — "
                v8s = f"{v8*100:>+7.3f}%" if not np.isnan(v8) else "     — "
                v24s = f"{v24*100:>+8.3f}%" if not np.isnan(v24) else "      — "
                print(f"  {window_start:<15s} {regime:<10s} {len(indices):>5d} {v4s} {v8s} {v24s}")

    # ═══════════════════════════════════════════════════════
    # MONOTONICITY BY REGIME
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  MONOTONICITY BY REGIME (funding quintiles → 24h return, non-overlapping)")
    print(f"{'═' * 90}")

    for regime in regimes:
        regime_df = df[df["regime"] == regime].copy()
        if len(regime_df) < 50:
            continue

        quintiles = pd.qcut(regime_df["fundingRate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        print(f"\n  {regime.upper()}:")
        prev_mean = None
        monotonic = True
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = quintiles == q
            indices = non_overlapping_indices(regime_df, mask)
            rets = regime_df.loc[indices, "fwd_ret_24h"].dropna() if indices else pd.Series(dtype=float)
            if len(rets) > 0:
                mean = rets.mean()
                if prev_mean is not None and mean > prev_mean:
                    monotonic = False
                prev_mean = mean
                print(f"    {q}: N={len(rets):>3d}, mean={mean*100:+.3f}%")
            else:
                print(f"    {q}: —")
        print(f"    Monotonic: {'YES' if monotonic else 'NO'}")

    # ═══════════════════════════════════════════════════════
    # FINAL CLASSIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  FINAL CLASSIFICATION")
    print(f"{'═' * 90}")

    # Gather evidence for each regime
    for regime in regimes:
        r = [x for x in all_results if x["regime"] == regime and x["threshold"] == "funding > p90"]
        if not r:
            continue
        r = r[0]
        n = r["N"]
        mean_4h = r.get("4h_mean", np.nan)
        p_4h = r.get("4h_p", np.nan)
        p_rand_4h = r.get("4h_p_vs_rand", np.nan)
        delta_4h = r.get("4h_delta", np.nan)
        mean_24h = r.get("24h_mean", np.nan)
        p_24h = r.get("24h_p", np.nan)

        sig_4h = not np.isnan(p_4h) and p_4h < 0.05
        beats_rand = not np.isnan(p_rand_4h) and p_rand_4h < 0.05
        neg_24h = not np.isnan(mean_24h) and mean_24h < 0

        if sig_4h and beats_rand and neg_24h:
            verdict = "✅ VALIDATED"
        elif sig_4h or (beats_rand and neg_24h):
            verdict = "⚠️ PROMISING"
        else:
            verdict = "❌ NO EDGE"

        print(f"\n  {regime.upper()} (p90, non-overlapping):")
        print(f"    N={n}, 4h={mean_4h*100:+.3f}% (p={p_4h:.3f}), 24h={mean_24h*100:+.3f}%")
        print(f"    vs random: delta={delta_4h*100:+.3f}%, p={p_rand_4h:.3f}")
        print(f"    {verdict}")

    # Overall
    print(f"\n  OVERALL ASSESSMENT:")
    bearish_r = [x for x in all_results if x["regime"] == "bearish" and x["threshold"] == "funding > p90"]
    if bearish_r:
        br = bearish_r[0]
        p4 = br.get("4h_p", 1)
        pr = br.get("4h_p_vs_rand", 1)
        if p4 < 0.05 and pr < 0.05:
            print(f"    High funding in BEARISH regime is a validated short signal.")
            print(f"    It beats same-regime random with statistical significance.")
        elif p4 < 0.10:
            print(f"    High funding in BEARISH regime is PROMISING but not fully validated.")
        else:
            print(f"    High funding in BEARISH regime does NOT show significant edge.")

    neutral_r = [x for x in all_results if x["regime"] == "neutral" and x["threshold"] == "funding > p90"]
    if neutral_r:
        nr = neutral_r[0]
        p4 = nr.get("4h_p", 1)
        if p4 > 0.10:
            print(f"    High funding in NEUTRAL regime has NO edge. Confirms regime-dependence.")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'═' * 90}")


if __name__ == "__main__":
    main()
