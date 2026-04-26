"""
FUNDING EDGE RE-AUDIT — BTCUSDT Perpetual Futures
==================================================
Rigorous re-examination of the high-funding → negative-returns edge.
Addresses autocorrelation, regime confound, overlapping returns, and
baseline comparison.

DO NOT claim VALID EDGE until all tests pass.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data/features"
TRAIN_CUTOFF = pd.Timestamp("2026-01-26")  # ~90 day split


def load_price_1h():
    """Load 1h price from 1m data."""
    price_1m = pd.read_csv(f"{DATA_DIR}/btcusdt_1m.csv")
    price_1m["timestamp"] = pd.to_datetime(price_1m["timestamp"])
    price_1h = price_1m.set_index("timestamp").resample("1h").agg({
        "price": ["first", "max", "min", "last"],
        "volume": "sum"
    }).dropna()
    price_1h.columns = ["open", "high", "low", "close", "volume"]
    return price_1h.reset_index()


def load_funding():
    """Load funding data."""
    funding = pd.read_csv(f"{DATA_DIR}/btcusdt_funding.csv")
    funding["timestamp"] = pd.to_datetime(funding["timestamp"])
    return funding


def merge(price_1h, funding):
    """Merge funding onto 1h price timeline with forward-fill."""
    df = price_1h.set_index("timestamp").copy()
    fr = funding.set_index("timestamp")["fundingRate"]
    df["fundingRate"] = fr.reindex(df.index, method="ffill")
    df = df.reset_index()

    # Derived features
    df["ret_1h"] = df["close"].pct_change(1)
    for bars in [1, 4, 24]:
        df[f"fwd_ret_{bars}h"] = df["close"].shift(-bars) / df["close"] - 1

    # Funding features
    df["funding_cum_24h"] = df["fundingRate"].rolling(3, min_periods=1).sum()
    df["funding_cum_72h"] = df["fundingRate"].rolling(9, min_periods=3).sum()

    # Z-score of funding (rolling 30-day = 90 8h periods)
    fr_mean = df["fundingRate"].rolling(270, min_periods=30).mean()
    fr_std = df["fundingRate"].rolling(270, min_periods=30).std()
    df["funding_zscore"] = (df["fundingRate"] - fr_mean) / fr_std.replace(0, np.nan)

    # Regime: 7-day return
    df["ret_7d"] = df["close"].pct_change(168)  # 7 days * 24h
    df["ret_30d"] = df["close"].pct_change(720)  # 30 days * 24h

    return df


def classify_regime(df):
    """Classify BTC regime based on 7-day return."""
    ret_7d = df["ret_7d"]
    df["regime"] = "neutral"
    df.loc[ret_7d > 0.05, "regime"] = "bullish"
    df.loc[ret_7d < -0.05, "regime"] = "bearish"
    return df


def compute_r_metrics(rets):
    """Compute standard metrics for a return series."""
    if len(rets) < 3:
        return None
    n = len(rets)
    mean = rets.mean()
    median = rets.median()
    std = rets.std()
    wr = (rets > 0).mean()
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return {
        "N": n, "mean": mean, "median": median, "std": std,
        "winrate": wr, "t_stat": t_stat, "p_val": p_val,
    }


def bootstrap_ci(rets, n_boot=10000, ci=0.95):
    """Bootstrap CI for mean."""
    if len(rets) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    boot = rng.choice(rets, size=(n_boot, len(rets)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return np.percentile(boot, alpha * 100), np.percentile(boot, (1 - alpha) * 100)


# ═══════════════════════════════════════════════════════════
# TEST 1 — FUNDING THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════

def threshold_sweep(df):
    """Test multiple funding thresholds."""
    print(f"\n{'═' * 80}")
    print(f"  TEST 1 — FUNDING THRESHOLD SWEEP")
    print(f"{'═' * 80}")

    fr = df["fundingRate"].dropna()
    fz = df["funding_zscore"].dropna()

    thresholds = {
        "funding > 0": df["fundingRate"] > 0,
        "funding > p50": df["fundingRate"] > fr.median(),
        "funding > p70": df["fundingRate"] > fr.quantile(0.70),
        "funding > p80": df["fundingRate"] > fr.quantile(0.80),
        "funding > p90": df["fundingRate"] > fr.quantile(0.90),
        "funding > p95": df["fundingRate"] > fr.quantile(0.95),
        "z-score > 1": df["funding_zscore"] > 1.0,
        "z-score > 1.5": df["funding_zscore"] > 1.5,
    }

    # Unconditional baseline
    print(f"\n  UNCONDITIONAL BASELINE:")
    for h in [1, 4, 24]:
        col = f"fwd_ret_{h}h"
        rets = df[col].dropna()
        m = compute_r_metrics(rets)
        if m:
            print(f"    {h}h: mean={m['mean']*100:+.4f}%, median={m['median']*100:+.4f}%, wr={m['winrate']:.1%}, N={m['N']}")

    print(f"\n  {'Threshold':<20s} {'N':>5s} │ {'1h_mean':>8s} {'1h_wr':>5s} {'1h_p':>6s} │ {'4h_mean':>8s} {'4h_wr':>5s} {'4h_p':>6s} │ {'24h_mean':>9s} {'24h_wr':>5s} {'24h_p':>6s} │ {'4h_CI_lo':>9s} {'4h_CI_hi':>9s}")
    print(f"  {'─'*20} {'─'*5} {'─'*8} {'─'*5} {'─'*6} {'─'*8} {'─'*5} {'─'*6} {'─'*9} {'─'*5} {'─'*6} {'─'*9} {'─'*9}")

    for name, mask in thresholds.items():
        event_df = df[mask]
        row = {"name": name, "N": mask.sum()}

        for h in [1, 4, 24]:
            col = f"fwd_ret_{h}h"
            rets = event_df[col].dropna()
            m = compute_r_metrics(rets)
            if m:
                row[f"{h}h_mean"] = m["mean"]
                row[f"{h}h_wr"] = m["winrate"]
                row[f"{h}h_p"] = m["p_val"]
                if h == 4:
                    ci_lo, ci_hi = bootstrap_ci(rets.values)
                    row["4h_ci_lo"] = ci_lo
                    row["4h_ci_hi"] = ci_hi

        def fmt_val(key, pct=True, digs=3):
            v = row.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "   —   "
            if pct:
                return f"{v*100:>+{digs+3}.{digs}f}%"
            return f"{v:.3f}"

        def fmt_p(key):
            v = row.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "  —  "
            sig = "✓" if v < 0.05 else ("~" if v < 0.10 else " ")
            return f"{v:.3f}{sig}"

        ci_lo = row.get("4h_ci_lo", np.nan)
        ci_hi = row.get("4h_ci_hi", np.nan)
        ci_lo_s = f"{ci_lo*100:+.3f}%" if not np.isnan(ci_lo) else "   —   "
        ci_hi_s = f"{ci_hi*100:+.3f}%" if not np.isnan(ci_hi) else "   —   "

        print(f"  {name:<20s} {row['N']:>5d} │ "
              f"{fmt_val('1h_mean')} {fmt_val('1h_wr', False, 0)} {fmt_p('1h_p')} │ "
              f"{fmt_val('4h_mean')} {fmt_val('4h_wr', False, 0)} {fmt_p('4h_p')} │ "
              f"{fmt_val('24h_mean')} {fmt_val('24h_wr', False, 0)} {fmt_p('24h_p')} │ "
              f"{ci_lo_s} {ci_hi_s}")

    # Monotonicity check
    print(f"\n  MONOTONICITY CHECK (funding quintiles → 24h return):")
    quintiles = pd.qcut(df["fundingRate"], 5, labels=["Q1(lo)", "Q2", "Q3", "Q4", "Q5(hi)"], duplicates="drop")
    prev_mean = None
    monotonic = True
    for q in ["Q1(lo)", "Q2", "Q3", "Q4", "Q5(hi)"]:
        mask = quintiles == q
        rets = df.loc[mask, "fwd_ret_24h"].dropna()
        if len(rets) > 0:
            mean = rets.mean()
            if prev_mean is not None and mean > prev_mean:
                monotonic = False
            prev_mean = mean
            print(f"    {q}: mean={mean*100:+.3f}%, N={len(rets)}")
    print(f"    Monotonic: {'YES' if monotonic else 'NO'}")


# ═══════════════════════════════════════════════════════════
# TEST 2 — NON-OVERLAPPING VALIDATION
# ═══════════════════════════════════════════════════════════

def non_overlapping_test(df):
    """Impose 24h cooldown between events to remove autocorrelation."""
    print(f"\n{'═' * 80}")
    print(f"  TEST 2 — NON-OVERLAPPING EVENTS (24h cooldown)")
    print(f"{'═' * 80}")

    thresholds = {
        "funding > p80": df["fundingRate"] > df["fundingRate"].quantile(0.80),
        "funding > p90": df["fundingRate"] > df["fundingRate"].quantile(0.90),
        "funding > p95": df["fundingRate"] > df["fundingRate"].quantile(0.95),
        "z-score > 1": df["funding_zscore"] > 1.0,
        "z-score > 1.5": df["funding_zscore"] > 1.5,
    }

    COOLDOWN = 24  # bars (hours)

    print(f"\n  {'Threshold':<20s} │ {'Overlapping N':>13s} {'4h_mean':>8s} {'4h_p':>6s} │ {'NonOverlap N':>12s} {'4h_mean':>8s} {'4h_p':>6s} │ {'Delta':>8s}")
    print(f"  {'─'*20} {'─'*13} {'─'*8} {'─'*6} {'─'*12} {'─'*8} {'─'*6} {'─'*8}")

    for name, mask in thresholds.items():
        # Overlapping
        event_df = df[mask]
        rets_ov = event_df["fwd_ret_4h"].dropna()
        m_ov = compute_r_metrics(rets_ov)

        # Non-overlapping: enforce 24h gap
        indices = df.index[mask].tolist()
        non_overlap_idx = []
        last_idx = -COOLDOWN - 1
        for idx in indices:
            if idx - last_idx >= COOLDOWN:
                non_overlap_idx.append(idx)
                last_idx = idx

        rets_no = df.loc[non_overlap_idx, "fwd_ret_4h"].dropna()
        m_no = compute_r_metrics(rets_no)

        if m_ov and m_no:
            delta = m_no["mean"] - m_ov["mean"]
            ov_p_s = f"{m_ov['p_val']:.3f}{'✓' if m_ov['p_val']<0.05 else '~' if m_ov['p_val']<0.10 else ' '}"
            no_p_s = f"{m_no['p_val']:.3f}{'✓' if m_no['p_val']<0.05 else '~' if m_no['p_val']<0.10 else ' '}"
            print(f"  {name:<20s} │ {m_ov['N']:>13d} {m_ov['mean']*100:>+7.3f}% {ov_p_s} │ {m_no['N']:>12d} {m_no['mean']*100:>+7.3f}% {no_p_s} │ {delta*100:>+7.3f}%")
        elif m_ov:
            print(f"  {name:<20s} │ {m_ov['N']:>13d} {m_ov['mean']*100:>+7.3f}% │ {'—':>12s} {'—':>8s} │")

    # Also test non-overlapping for 24h horizon
    print(f"\n  NON-OVERLAPPING AT 24h HORIZON (24h cooldown):")
    for name, mask in thresholds.items():
        indices = df.index[mask].tolist()
        non_overlap_idx = []
        last_idx = -COOLDOWN - 1
        for idx in indices:
            if idx - last_idx >= COOLDOWN:
                non_overlap_idx.append(idx)
                last_idx = idx

        rets_no = df.loc[non_overlap_idx, "fwd_ret_24h"].dropna()
        m_no = compute_r_metrics(rets_no)
        if m_no:
            ci_lo, ci_hi = bootstrap_ci(rets_no.values)
            sig = "✓" if m_no["p_val"] < 0.05 else ("~" if m_no["p_val"] < 0.10 else " ")
            print(f"    {name:<20s}: N={m_no['N']:>4d}, mean={m_no['mean']*100:+.3f}%, wr={m_no['winrate']:.0%}, p={m_no['p_val']:.3f}{sig}, CI=[{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%]")


# ═══════════════════════════════════════════════════════════
# TEST 3 — BASELINE COMPARISON
# ═══════════════════════════════════════════════════════════

def baseline_comparison(df):
    """Compare high-funding events to random baselines."""
    print(f"\n{'═' * 80}")
    print(f"  TEST 3 — BASELINE COMPARISON")
    print(f"{'═' * 80}")

    rng = np.random.default_rng(42)

    # High funding events
    p90_mask = df["fundingRate"] > df["fundingRate"].quantile(0.90)
    event_rets = df.loc[p90_mask, "fwd_ret_4h"].dropna()
    event_mean = event_rets.mean()
    event_n = len(event_rets)

    print(f"\n  High funding (p90): N={event_n}, mean 4h={event_mean*100:+.3f}%")

    # Baseline 1: random timestamps (same N)
    print(f"\n  Baseline 1: Random timestamps (N={event_n}, 1000 simulations)")
    valid_idx = df[df["fwd_ret_4h"].notna()].index.tolist()
    random_means = []
    for _ in range(1000):
        rand_idx = rng.choice(valid_idx, size=event_n, replace=False)
        random_means.append(df.loc[rand_idx, "fwd_ret_4h"].mean())
    random_means = np.array(random_means)

    p_random = (random_means <= event_mean).mean()  # one-sided: is event more negative?
    print(f"    Random mean: {random_means.mean()*100:+.4f}%")
    print(f"    Event mean:  {event_mean*100:+.4f}%")
    print(f"    P(event < random): {p_random:.3f}")
    print(f"    {'SIGNIFICANT' if p_random < 0.05 else 'NOT significant'}")

    # Baseline 2: same-month random
    print(f"\n  Baseline 2: Same-month random timestamps")
    df_dated = df.copy()
    df_dated["month"] = df_dated["timestamp"].dt.to_period("M")

    month_diffs = []
    for month in df_dated["month"].unique():
        month_mask = df_dated["month"] == month
        month_event = p90_mask & month_mask
        month_n = month_event.sum()
        if month_n == 0:
            continue

        month_valid = df_dated[month_mask & df_dated["fwd_ret_4h"].notna()].index.tolist()
        if len(month_valid) < month_n:
            continue

        event_month_mean = df_dated.loc[month_event, "fwd_ret_4h"].mean()

        random_month_means = []
        for _ in range(500):
            rand_idx = rng.choice(month_valid, size=month_n, replace=False)
            random_month_means.append(df_dated.loc[rand_idx, "fwd_ret_4h"].mean())
        random_month_means = np.array(random_month_means)

        p = (random_month_means <= event_month_mean).mean()
        month_diffs.append({"month": str(month), "N": month_n, "event_mean": event_month_mean,
                           "random_mean": random_month_means.mean(), "p": p})

    if month_diffs:
        print(f"    {'Month':<10s} {'N':>5s} {'Event':>8s} {'Random':>8s} {'p':>6s}")
        for md in month_diffs:
            sig = "✓" if md["p"] < 0.05 else ("~" if md["p"] < 0.10 else " ")
            print(f"    {md['month']:<10s} {md['N']:>5d} {md['event_mean']*100:>+7.3f}% {md['random_mean']*100:>+7.3f}% {md['p']:.3f}{sig}")

    # Baseline 3: same-regime random
    print(f"\n  Baseline 3: Same-regime random timestamps")
    for regime in ["bullish", "bearish", "neutral"]:
        regime_mask = df["regime"] == regime
        regime_event = p90_mask & regime_mask
        regime_n = regime_event.sum()
        if regime_n < 5:
            continue

        regime_valid = df[regime_mask & df["fwd_ret_4h"].notna()].index.tolist()
        if len(regime_valid) < regime_n:
            continue

        event_regime_mean = df.loc[regime_event, "fwd_ret_4h"].mean()

        random_regime_means = []
        for _ in range(500):
            rand_idx = rng.choice(regime_valid, size=regime_n, replace=False)
            random_regime_means.append(df.loc[rand_idx, "fwd_ret_4h"].mean())
        random_regime_means = np.array(random_regime_means)

        p = (random_regime_means <= event_regime_mean).mean()
        sig = "✓" if p < 0.05 else ("~" if p < 0.10 else " ")
        print(f"    {regime:<10s}: N={regime_n}, event={event_regime_mean*100:+.3f}%, random={random_regime_means.mean()*100:+.3f}%, p={p:.3f}{sig}")


# ═══════════════════════════════════════════════════════════
# TEST 4 — REGIME CONTROL
# ═══════════════════════════════════════════════════════════

def regime_control(df):
    """Check if high funding predicts neg returns inside each regime."""
    print(f"\n{'═' * 80}")
    print(f"  TEST 4 — REGIME CONTROL")
    print(f"{'═' * 80}")

    # Overall regime distribution
    print(f"\n  Regime distribution:")
    for regime in ["bullish", "bearish", "neutral"]:
        n = (df["regime"] == regime).sum()
        print(f"    {regime}: {n} bars ({n/len(df)*100:.1f}%)")

    # High funding events per regime
    p90_mask = df["fundingRate"] > df["fundingRate"].quantile(0.90)
    p80_mask = df["fundingRate"] > df["fundingRate"].quantile(0.80)

    print(f"\n  High funding (p90) per regime — 4h forward return:")
    print(f"  {'Regime':<12s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'WR':>5s} {'p':>6s} {'CI_lo':>8s} {'CI_hi':>8s}")
    print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*5} {'─'*6} {'─'*8} {'─'*8}")

    for regime in ["bullish", "bearish", "neutral"]:
        mask = p90_mask & (df["regime"] == regime)
        rets = df.loc[mask, "fwd_ret_4h"].dropna()
        m = compute_r_metrics(rets)
        if m:
            ci_lo, ci_hi = bootstrap_ci(rets.values)
            sig = "✓" if m["p_val"] < 0.05 else ("~" if m["p_val"] < 0.10 else " ")
            print(f"  {regime:<12s} {m['N']:>5d} {m['mean']*100:>+7.3f}% {m['median']*100:>+7.3f}% {m['winrate']:>4.0%} {m['p_val']:.3f}{sig} {ci_lo*100:>+7.3f}% {ci_hi*100:>+7.3f}%")

    # Also p80 for more events per regime
    print(f"\n  High funding (p80) per regime — 4h forward return:")
    print(f"  {'Regime':<12s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'WR':>5s} {'p':>6s} {'CI_lo':>8s} {'CI_hi':>8s}")
    print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*5} {'─'*6} {'─'*8} {'─'*8}")

    for regime in ["bullish", "bearish", "neutral"]:
        mask = p80_mask & (df["regime"] == regime)
        rets = df.loc[mask, "fwd_ret_4h"].dropna()
        m = compute_r_metrics(rets)
        if m:
            ci_lo, ci_hi = bootstrap_ci(rets.values)
            sig = "✓" if m["p_val"] < 0.05 else ("~" if m["p_val"] < 0.10 else " ")
            print(f"  {regime:<12s} {m['N']:>5d} {m['mean']*100:>+7.3f}% {m['median']*100:>+7.3f}% {m['winrate']:>4.0%} {m['p_val']:.3f}{sig} {ci_lo*100:>+7.3f}% {ci_hi*100:>+7.3f}%")

    # Conditional: does funding ADD information beyond regime?
    print(f"\n  INCREMENTAL VALUE: Does funding add info beyond regime?")
    for regime in ["bullish", "bearish", "neutral"]:
        regime_df = df[df["regime"] == regime]
        if len(regime_df) < 50:
            continue

        # All bars in this regime
        all_rets = regime_df["fwd_ret_4h"].dropna()
        m_all = compute_r_metrics(all_rets)

        # High funding in this regime
        hf_mask = regime_df["fundingRate"] > regime_df["fundingRate"].quantile(0.80)
        hf_rets = regime_df.loc[hf_mask, "fwd_ret_4h"].dropna()
        m_hf = compute_r_metrics(hf_rets)

        # Low funding in this regime
        lf_mask = regime_df["fundingRate"] < regime_df["fundingRate"].quantile(0.20)
        lf_rets = regime_df.loc[lf_mask, "fwd_ret_4h"].dropna()
        m_lf = compute_r_metrics(lf_rets)

        if m_all and m_hf and m_lf:
            print(f"\n    {regime.upper()}:")
            print(f"      All bars:     N={m_all['N']:>5d}, mean={m_all['mean']*100:+.3f}%, wr={m_all['winrate']:.0%}")
            print(f"      High funding: N={m_hf['N']:>5d}, mean={m_hf['mean']*100:+.3f}%, wr={m_hf['winrate']:.0%}")
            print(f"      Low funding:  N={m_lf['N']:>5d}, mean={m_lf['mean']*100:+.3f}%, wr={m_lf['winrate']:.0%}")
            delta = m_hf["mean"] - m_all["mean"]
            print(f"      Delta (high vs all): {delta*100:+.3f}%")


# ═══════════════════════════════════════════════════════════
# TEST 5 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════

def walk_forward(df):
    """Split into time periods and check consistency."""
    print(f"\n{'═' * 80}")
    print(f"  TEST 5 — WALK-FORWARD")
    print(f"{'═' * 80}")

    p90 = df["fundingRate"].quantile(0.90)
    p80 = df["fundingRate"].quantile(0.80)

    # Split into halves
    n = len(df)
    mid = n // 2

    # Split into thirds
    third = n // 3

    splits = [
        ("1st half", df.iloc[:mid]),
        ("2nd half", df.iloc[mid:]),
        ("1st third", df.iloc[:third]),
        ("2nd third", df.iloc[third:2*third]),
        ("3rd third", df.iloc[2*third:]),
    ]

    print(f"\n  High funding (p90) — 4h return across splits:")
    print(f"  {'Split':<15s} {'Period':<30s} {'N':>5s} {'Mean':>8s} {'WR':>5s} {'p':>6s}")
    print(f"  {'─'*15} {'─'*30} {'─'*5} {'─'*8} {'─'*5} {'─'*6}")

    for label, split_df in splits:
        mask = split_df["fundingRate"] > p90
        rets = split_df.loc[mask, "fwd_ret_4h"].dropna()
        m = compute_r_metrics(rets)
        if m:
            period = f"{split_df['timestamp'].min().date()} → {split_df['timestamp'].max().date()}"
            sig = "✓" if m["p_val"] < 0.05 else ("~" if m["p_val"] < 0.10 else " ")
            print(f"  {label:<15s} {period:<30s} {m['N']:>5d} {m['mean']*100:>+7.3f}% {m['winrate']:>4.0%} {m['p_val']:.3f}{sig}")

    # Rolling 30-day windows
    print(f"\n  Rolling 30-day windows (p90, 4h return):")
    print(f"  {'Window start':<15s} {'N':>5s} {'Mean':>8s} {'WR':>5s}")
    print(f"  {'─'*15} {'─'*5} {'─'*8} {'─'*5}")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    window_size = 30 * 24  # 30 days in hours
    step = 15 * 24  # 15 day step

    for start in range(0, len(df_sorted) - window_size, step):
        window = df_sorted.iloc[start:start + window_size]
        mask = window["fundingRate"] > p90
        rets = window.loc[mask, "fwd_ret_4h"].dropna()
        m = compute_r_metrics(rets)
        if m and m["N"] >= 5:
            window_start = window["timestamp"].min().strftime("%Y-%m-%d")
            print(f"  {window_start:<15s} {m['N']:>5d} {m['mean']*100:>+7.3f}% {m['winrate']:>4.0%}")

    # Train vs test
    print(f"\n  Train vs Test (first 90d / last 90d):")
    train = df[df["timestamp"] < TRAIN_CUTOFF]
    test = df[df["timestamp"] >= TRAIN_CUTOFF]

    for label, split_df in [("Train (first 90d)", train), ("Test (last 90d)", test)]:
        mask = split_df["fundingRate"] > p90
        rets = split_df.loc[mask, "fwd_ret_4h"].dropna()
        m = compute_r_metrics(rets)
        if m:
            ci_lo, ci_hi = bootstrap_ci(rets.values)
            sig = "✓" if m["p_val"] < 0.05 else ("~" if m["p_val"] < 0.10 else " ")
            print(f"    {label:<25s}: N={m['N']:>4d}, mean={m['mean']*100:+.3f}%, wr={m['winrate']:.0%}, p={m['p_val']:.3f}{sig}, CI=[{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%]")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print("=" * 80)
    print("  FUNDING EDGE RE-AUDIT — BTCUSDT Perpetual Futures")
    print("  Rigorous re-examination. DO NOT claim VALID EDGE yet.")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Load and merge
    print("\n  Loading data...")
    price_1h = load_price_1h()
    funding = load_funding()
    df = merge(price_1h, funding)
    df = classify_regime(df)

    # Filter to funding-available period
    df = df[df["fundingRate"].notna()].copy()
    print(f"  Merged: {len(df)} bars ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    print(f"  Funding range: [{df['fundingRate'].min()*100:.4f}%, {df['fundingRate'].max()*100:.4f}%]")
    print(f"  Mean funding: {df['fundingRate'].mean()*100:.4f}%")

    # Run all tests
    threshold_sweep(df)
    non_overlapping_test(df)
    baseline_comparison(df)
    regime_control(df)
    walk_forward(df)

    # ═══════════════════════════════════════════════════════
    # FINAL CLASSIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  FINAL CLASSIFICATION")
    print(f"{'═' * 80}")

    # Gather evidence
    p90_mask = df["fundingRate"] > df["fundingRate"].quantile(0.90)
    rets_all = df.loc[p90_mask, "fwd_ret_4h"].dropna()
    m_all = compute_r_metrics(rets_all)

    # Non-overlapping
    indices = df.index[p90_mask].tolist()
    non_overlap_idx = []
    last_idx = -25
    for idx in indices:
        if idx - last_idx >= 24:
            non_overlap_idx.append(idx)
            last_idx = idx
    rets_no = df.loc[non_overlap_idx, "fwd_ret_4h"].dropna()
    m_no = compute_r_metrics(rets_no)

    # Train/test
    train = df[df["timestamp"] < TRAIN_CUTOFF]
    test = df[df["timestamp"] >= TRAIN_CUTOFF]
    rets_train = train.loc[train["fundingRate"] > df["fundingRate"].quantile(0.90), "fwd_ret_4h"].dropna()
    rets_test = test.loc[test["fundingRate"] > df["fundingRate"].quantile(0.90), "fwd_ret_4h"].dropna()
    m_train = compute_r_metrics(rets_train)
    m_test = compute_r_metrics(rets_test)

    print(f"\n  EVIDENCE SUMMARY (high funding p90, 4h return):")
    if m_all:
        print(f"    All events:      N={m_all['N']}, mean={m_all['mean']*100:+.3f}%, p={m_all['p_val']:.3f}")
    if m_no:
        print(f"    Non-overlapping: N={m_no['N']}, mean={m_no['mean']*100:+.3f}%, p={m_no['p_val']:.3f}")
    if m_train:
        print(f"    Train:           N={m_train['N']}, mean={m_train['mean']*100:+.3f}%, p={m_train['p_val']:.3f}")
    if m_test:
        print(f"    Test:            N={m_test['N']}, mean={m_test['mean']*100:+.3f}%, p={m_test['p_val']:.3f}")

    # Classification criteria
    criteria = []

    # 1. Statistically significant
    if m_all and m_all["p_val"] < 0.05:
        criteria.append(("Significant (all events)", "PASS", f"p={m_all['p_val']:.3f}"))
    elif m_all and m_all["p_val"] < 0.10:
        criteria.append(("Significant (all events)", "MARGINAL", f"p={m_all['p_val']:.3f}"))
    else:
        criteria.append(("Significant (all events)", "FAIL", f"p={m_all['p_val']:.3f}" if m_all else "no data"))

    # 2. Survives non-overlapping
    if m_no and m_no["p_val"] < 0.05:
        criteria.append(("Survives non-overlapping", "PASS", f"p={m_no['p_val']:.3f}"))
    elif m_no and m_no["p_val"] < 0.10:
        criteria.append(("Survives non-overlapping", "MARGINAL", f"p={m_no['p_val']:.3f}"))
    else:
        criteria.append(("Survives non-overlapping", "FAIL", f"p={m_no['p_val']:.3f}" if m_no else "no data"))

    # 3. Monotonic
    quintiles = pd.qcut(df["fundingRate"], 5, labels=False, duplicates="drop")
    q_means = []
    for q in range(5):
        rets = df.loc[quintiles == q, "fwd_ret_24h"].dropna()
        if len(rets) > 0:
            q_means.append(rets.mean())
    monotonic = all(q_means[i] >= q_means[i+1] for i in range(len(q_means)-1))
    criteria.append(("Monotonic (quintiles)", "PASS" if monotonic else "FAIL", f"Q1→Q5 {'monotonic' if monotonic else 'NOT monotonic'}"))

    # 4. Consistent across time
    if m_train and m_test:
        same_sign = (m_train["mean"] < 0) == (m_test["mean"] < 0)
        criteria.append(("Consistent train/test", "PASS" if same_sign else "FAIL",
                        f"train={m_train['mean']*100:+.3f}%, test={m_test['mean']*100:+.3f}%"))

    # 5. Not pure regime artifact
    regime_consistent = 0
    regime_total = 0
    for regime in ["bullish", "bearish", "neutral"]:
        mask = p90_mask & (df["regime"] == regime)
        rets = df.loc[mask, "fwd_ret_4h"].dropna()
        if len(rets) >= 5:
            regime_total += 1
            m = compute_r_metrics(rets)
            if m and m["mean"] < 0:
                regime_consistent += 1
    if regime_total >= 2:
        criteria.append(("Negative in all regimes", "PASS" if regime_consistent == regime_total else "PARTIAL",
                        f"{regime_consistent}/{regime_total} regimes negative"))

    # Print criteria
    print(f"\n  CRITERIA:")
    for criterion, status, detail in criteria:
        icon = "✅" if status == "PASS" else ("⚠️" if status in ("MARGINAL", "PARTIAL") else "❌")
        print(f"    {icon} {criterion:<35s} {status:<12s} {detail}")

    # Final verdict
    pass_count = sum(1 for _, s, _ in criteria if s == "PASS")
    fail_count = sum(1 for _, s, _ in criteria if s == "FAIL")

    print(f"\n  VERDICT:")
    if fail_count >= 2:
        print(f"    ❌ NO EDGE — Multiple criteria failed")
    elif pass_count >= 4:
        print(f"    ✅ VALIDATED — Survives all tests")
    elif pass_count >= 2:
        print(f"    ⚠️  PROMISING BUT NOT VALIDATED — Some criteria pass, need more evidence")
    else:
        print(f"    ❌ NO EDGE — Insufficient evidence")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
