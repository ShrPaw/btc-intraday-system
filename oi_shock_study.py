"""
OI SHOCK EVENT STUDY — Do OI shocks create exploitable price behavior?
======================================================================
Data: BTCUSDT OI (1h), price (1m), taker buy/sell, global LS ratio
Limitation: OI data covers only Mar 31 – Apr 25 2026 (26 days, 625 bars)
No liquidation data available.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "features")

# =========================================================
# CONFIG
# =========================================================

# OI spike: top 10% of hourly OI % change
OI_SPIKE_PCTL = 90
# OI collapse: bottom 10% of hourly OI % change
OI_COLLAPSE_PCTL = 10

# Taker extremes
TAKER_SPIKE_PCTL = 90
TAKER_COLLAPSE_PCTL = 10

# Forward tracking
FORWARD_HORIZONS = [5, 15, 30, 60]  # minutes (using 1m price data)
MAE_MFE_HORIZON = 60  # 1m bars

# =========================================================
# DATA
# =========================================================

def load_data():
    price = pd.read_csv(os.path.join(DATA_DIR, "btcusdt_1m.csv"), parse_dates=["timestamp"])
    price = price.sort_values("timestamp").reset_index(drop=True)

    oi = pd.read_csv(os.path.join(DATA_DIR, "btcusdt_oi_1h.csv"), parse_dates=["timestamp"])
    oi = oi.sort_values("timestamp").reset_index(drop=True)

    taker = pd.read_csv(os.path.join(DATA_DIR, "btcusdt_taker_1h.csv"), parse_dates=["timestamp"])
    taker = taker.sort_values("timestamp").reset_index(drop=True)

    ls = pd.read_csv(os.path.join(DATA_DIR, "btcusdt_global_ls_1h.csv"), parse_dates=["timestamp"])
    ls = ls.sort_values("timestamp").reset_index(drop=True)

    return price, oi, taker, ls


# =========================================================
# STEP 1 — DEFINE EVENTS
# =========================================================

def define_events(oi, taker, ls):
    """
    OI spike:   top 10% hourly OI % change
    OI collapse: bottom 10% hourly OI % change
    Taker shock: top/bottom 10% taker buy/sell ratio changes
    """
    df = oi.copy()

    # Merge taker and LS data
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        taker[["timestamp", "buySellRatio", "buyVol", "sellVol"]].sort_values("timestamp"),
        on="timestamp", direction="backward"
    )
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        ls[["timestamp", "globalLongRatio", "globalLSRatio"]].sort_values("timestamp"),
        on="timestamp", direction="backward"
    )

    # OI % change (hourly)
    df["oi_pct_change"] = df["sumOpenInterest"].pct_change()
    df["oi_value_pct_change"] = df["sumOpenInterestValue"].pct_change()

    # Taker buy/sell ratio change
    df["taker_ratio_change"] = df["buySellRatio"].pct_change()

    # Long/short ratio change
    df["ls_ratio_change"] = df["globalLSRatio"].pct_change()

    # Rolling thresholds (past-only, window=48 = 2 days)
    window = 48
    df["oi_spike_thresh"] = df["oi_pct_change"].rolling(window, min_periods=12).quantile(OI_SPIKE_PCTL / 100).shift(1)
    df["oi_collapse_thresh"] = df["oi_pct_change"].rolling(window, min_periods=12).quantile(OI_COLLAPSE_PCTL / 100).shift(1)
    df["taker_spike_thresh"] = df["taker_ratio_change"].rolling(window, min_periods=12).quantile(TAKER_SPIKE_PCTL / 100).shift(1)
    df["taker_collapse_thresh"] = df["taker_ratio_change"].rolling(window, min_periods=12).quantile(TAKER_COLLAPSE_PCTL / 100).shift(1)

    # Event flags
    df["is_oi_spike"] = df["oi_pct_change"] > df["oi_spike_thresh"]
    df["is_oi_collapse"] = df["oi_pct_change"] < df["oi_collapse_thresh"]
    df["is_taker_spike"] = df["taker_ratio_change"] > df["taker_spike_thresh"]
    df["is_taker_collapse"] = df["taker_ratio_change"] < df["taker_collapse_thresh"]

    # Combined OI shock (any extreme)
    df["is_oi_shock"] = df["is_oi_spike"] | df["is_oi_collapse"]

    # Clean
    df = df.dropna(subset=["oi_pct_change"]).reset_index(drop=True)

    return df


# =========================================================
# STEP 2 — FORWARD ANALYSIS
# =========================================================

def track_paths(events_df, price, event_col, event_label):
    """For each event where event_col is True, track forward price path."""
    price_values = price["price"].values
    n_bars = len(price_values)
    results = []

    event_rows = events_df[events_df[event_col]]
    if len(event_rows) == 0:
        return pd.DataFrame()

    for _, event in event_rows.iterrows():
        event_time = event["timestamp"]

        # Find nearest 1m bar at or after event
        mask = price["timestamp"] >= event_time
        if mask.sum() == 0:
            continue
        idx = price.index[mask][0]

        event_price = price_values[idx]

        # Forward returns
        forward_rets = {}
        for h in FORWARD_HORIZONS:
            target_idx = idx + h
            if target_idx < n_bars:
                forward_rets[f"ret_{h}m"] = (price_values[target_idx] - event_price) / event_price
            else:
                forward_rets[f"ret_{h}m"] = np.nan

        # MAE/MFE
        end_idx = min(idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[idx:end_idx]
        if len(path) < 2:
            continue
        path_pct = (path - event_price) / event_price

        # For OI spikes: direction unknown, measure absolute excursion
        # For directional events: use taker ratio as direction hint
        mfe_up = np.max(path_pct)
        mae_up = np.max(-path_pct)
        mfe_down = np.max(-path_pct)
        mae_down = np.max(path_pct)

        # "Reversal" = price goes back toward event level
        # Track if initial move (first 5 bars) is reversed by bar 60
        if len(path) >= 10:
            initial_move = path_pct[4]  # 5-bar move
            final_move = path_pct[-1]
            # Reversal: initial and final have opposite signs
            reversal = (initial_move * final_move) < 0
            # Partial reversion: final move is smaller magnitude than initial
            partial_reversion = abs(final_move) < abs(initial_move) * 0.5
        else:
            reversal = None
            partial_reversion = None

        results.append({
            "timestamp": event["timestamp"],
            "event_type": event_label,
            "oi_pct_change": event.get("oi_pct_change"),
            "sumOpenInterest": event.get("sumOpenInterest"),
            "buySellRatio": event.get("buySellRatio"),
            "globalLSRatio": event.get("globalLSRatio"),
            "event_price": event_price,
            **forward_rets,
            "mfe_up": mfe_up,
            "mae_up": mae_up,
            "mfe_down": mfe_down,
            "mae_down": mae_down,
            "reversal": reversal,
            "partial_reversion": partial_reversion,
        })

    return pd.DataFrame(results)


# =========================================================
# STEP 3 — BASELINE
# =========================================================

def build_baselines(events_df, price, n_events):
    """Random and same-time baselines."""
    price_values = price["price"].values
    n_bars = len(price_values)
    valid_range = n_bars - MAE_MFE_HORIZON - 10
    rng = np.random.default_rng(42)

    price_hours = price["timestamp"].dt.hour.values
    hour_indices = {}
    for h in range(24):
        mask = (price_hours == h) & (np.arange(n_bars) < valid_range)
        hour_indices[h] = np.where(mask)[0]

    results = {"random": [], "same_time": []}

    # Random baseline
    rand_indices = rng.choice(valid_range, size=n_events, replace=False)
    for idx in rand_indices:
        event_price = price_values[idx]
        if event_price == 0:
            continue
        forward_rets = {}
        for h in FORWARD_HORIZONS:
            target_idx = idx + h
            if target_idx < n_bars:
                forward_rets[f"ret_{h}m"] = (price_values[target_idx] - event_price) / event_price
            else:
                forward_rets[f"ret_{h}m"] = np.nan
        end_idx = min(idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[idx:end_idx]
        if len(path) < 2:
            continue
        path_pct = (path - event_price) / event_price
        results["random"].append({**forward_rets, "mfe_up": np.max(path_pct), "mae_up": np.max(-path_pct)})

    # Same-time baseline
    for _, event in events_df.head(n_events).iterrows():
        hour = event["timestamp"].hour
        candidates = hour_indices.get(hour, np.array([]))
        if len(candidates) < 5:
            continue
        pick_idx = rng.choice(candidates)
        event_price = price_values[pick_idx]
        if event_price == 0:
            continue
        forward_rets = {}
        for h in FORWARD_HORIZONS:
            target_idx = pick_idx + h
            if target_idx < n_bars:
                forward_rets[f"ret_{h}m"] = (price_values[target_idx] - event_price) / event_price
            else:
                forward_rets[f"ret_{h}m"] = np.nan
        end_idx = min(pick_idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[pick_idx:end_idx]
        if len(path) < 2:
            continue
        path_pct = (path - event_price) / event_price
        results["same_time"].append({**forward_rets, "mfe_up": np.max(path_pct), "mae_up": np.max(-path_pct)})

    return pd.DataFrame(results["random"]), pd.DataFrame(results["same_time"])


# =========================================================
# ANALYSIS
# =========================================================

def analyze_event_set(results, label):
    """Print analysis for one event type."""
    n = len(results)
    if n < 3:
        print(f"\n  {label}: n={n} — TOO FEW")
        return

    print(f"\n  {label}  |  n = {n}")
    print(f"  {'─'*55}")

    for h in FORWARD_HORIZONS:
        col = f"ret_{h}m"
        vals = results[col].dropna()
        if len(vals) < 3:
            continue
        wr = (vals > 0).sum() / len(vals) * 100
        print(f"    {h:>3}m return:  mean {vals.mean()*100:+.4f}%  median {vals.median()*100:+.4f}%  win {wr:.1f}%")

    mfe = results["mfe_up"].dropna()
    mae = results["mae_up"].dropna()
    print(f"    MFE 60m:   median {mfe.median()*100:.3f}%  mean {mfe.mean()*100:.3f}%")
    print(f"    MAE 60m:   median {mae.median()*100:.3f}%  mean {mae.mean()*100:.3f}%")

    if "reversal" in results.columns:
        rev = results["reversal"].dropna()
        if len(rev) > 0:
            print(f"    Reversal:  {rev.sum()}/{len(rev)} = {rev.mean()*100:.1f}%")
        partial = results["partial_reversion"].dropna()
        if len(partial) > 0:
            print(f"    Part.rev:  {partial.sum()}/{len(partial)} = {partial.mean()*100:.1f}%")


def print_comparison_table(all_results, baselines):
    """Print side-by-side comparison."""
    print(f"\n  {'Event':<25} {'N':>5} {'5m ret':>10} {'15m ret':>10} {'60m ret':>10} {'MFE/MAE':>10} {'Rev%':>8}")
    print(f"  {'─'*78}")

    for label, df in all_results.items():
        if len(df) < 3:
            continue
        r5 = df["ret_5m"].dropna()
        r15 = df["ret_15m"].dropna()
        r60 = df["ret_60m"].dropna()
        mfe = df["mfe_up"].dropna()
        mae = df["mae_up"].dropna()
        ratio = (mfe / mae).replace([np.inf, -np.inf], np.nan).dropna()
        rev = df["reversal"].dropna()
        rev_pct = rev.mean() * 100 if len(rev) > 0 else np.nan
        print(f"  {label:<25} {len(df):>5} {r5.mean()*100:>+10.4f} {r15.mean()*100:>+10.4f} {r60.mean()*100:>+10.4f} {ratio.median():>10.3f} {rev_pct:>7.1f}%")

    for label, df in baselines.items():
        if len(df) < 3:
            continue
        r5 = df["ret_5m"].dropna()
        r15 = df["ret_15m"].dropna()
        r60 = df["ret_60m"].dropna()
        mfe = df["mfe_up"].dropna()
        mae = df["mae_up"].dropna()
        ratio = (mfe / mae).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {'BASELINE/' + label:<25} {len(df):>5} {r5.mean()*100:>+10.4f} {r15.mean()*100:>+10.4f} {r60.mean()*100:>+10.4f} {ratio.median():>10.3f} {'—':>8}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  OI SHOCK EVENT STUDY")
    print("  ⚠ Data: OI/taker/LS covers only Mar 31 – Apr 25 2026 (26 days)")
    print("=" * 70)

    print("\nLoading data...")
    price, oi, taker, ls = load_data()
    print(f"  1m candles: {len(price):,}")
    print(f"  OI (1h):    {len(oi):,} ({oi['timestamp'].min().date()} → {oi['timestamp'].max().date()})")
    print(f"  Taker (1h): {len(taker):,}")
    print(f"  LS ratio:   {len(ls):,}")

    print("\nSTEP 1 — Defining events...")
    events = define_events(oi, taker, ls)
    print(f"  Total OI bars: {len(events):,}")
    print(f"  OI spikes:     {events['is_oi_spike'].sum()}")
    print(f"  OI collapses:  {events['is_oi_collapse'].sum()}")
    print(f"  Taker spikes:  {events['is_taker_spike'].sum()}")
    print(f"  Taker collapses: {events['is_taker_collapse'].sum()}")

    print("\nSTEP 2 — Tracking forward paths...")
    all_results = {}

    for event_col, event_label in [
        ("is_oi_spike", "OI_SPIKE"),
        ("is_oi_collapse", "OI_COLLAPSE"),
        ("is_taker_spike", "TAKER_SPIKE"),
        ("is_taker_collapse", "TAKER_COLLAPSE"),
    ]:
        results = track_paths(events, price, event_col, event_label)
        all_results[event_label] = results
        if len(results) > 0:
            print(f"  {event_label}: {len(results)} events tracked")

    # Combined OI shock
    oi_shock = pd.concat([all_results.get("OI_SPIKE", pd.DataFrame()), all_results.get("OI_COLLAPSE", pd.DataFrame())], ignore_index=True)
    all_results["OI_SHOCK_COMBINED"] = oi_shock

    print("\nSTEP 3 — Baselines...")
    n_events = max(len(df) for df in all_results.values()) if all_results else 10
    random_base, same_time_base = build_baselines(events, price, n_events)
    print(f"  Random baseline: {len(random_base)}")
    print(f"  Same-time baseline: {len(same_time_base)}")

    print(f"\n{'='*70}")
    print(f"  DETAILED RESULTS")
    print(f"{'='*70}")
    for label, df in all_results.items():
        analyze_event_set(df, label)

    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print_comparison_table(all_results, {"random": random_base, "same_time": same_time_base})

    # OI spike magnitude analysis
    print(f"\n{'='*70}")
    print(f"  OI SPIKE MAGNITUDE BREAKDOWN")
    print(f"{'='*70}")
    if len(all_results.get("OI_SPIKE", pd.DataFrame())) > 5:
        spikes = all_results["OI_SPIKE"].copy()
        spikes["oi_magnitude"] = spikes["oi_pct_change"].abs()
        q33 = spikes["oi_magnitude"].quantile(0.33)
        q67 = spikes["oi_magnitude"].quantile(0.67)
        spikes["magnitude_group"] = np.where(
            spikes["oi_magnitude"] >= q67, "LARGE",
            np.where(spikes["oi_magnitude"] >= q33, "MEDIUM", "SMALL")
        )
        for group, gdf in spikes.groupby("magnitude_group"):
            if len(gdf) < 3:
                continue
            r60 = gdf["ret_60m"].dropna()
            rev = gdf["reversal"].dropna()
            print(f"    {group:>8}: n={len(gdf):>3}  60m {r60.mean()*100:+.4f}%  reversal {rev.mean()*100:.1f}%")

    # OI collapse magnitude breakdown
    if len(all_results.get("OI_COLLAPSE", pd.DataFrame())) > 5:
        collapses = all_results["OI_COLLAPSE"].copy()
        collapses["oi_magnitude"] = collapses["oi_pct_change"].abs()
        q33 = collapses["oi_magnitude"].quantile(0.33)
        q67 = collapses["oi_magnitude"].quantile(0.67)
        collapses["magnitude_group"] = np.where(
            collapses["oi_magnitude"] >= q67, "LARGE",
            np.where(collapses["oi_magnitude"] >= q33, "MEDIUM", "SMALL")
        )
        for group, gdf in collapses.groupby("magnitude_group"):
            if len(gdf) < 3:
                continue
            r60 = gdf["ret_60m"].dropna()
            rev = gdf["reversal"].dropna()
            print(f"    {group:>8}: n={len(gdf):>3}  60m {r60.mean()*100:+.4f}%  reversal {rev.mean()*100:.1f}%")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    for label in ["OI_SPIKE", "OI_COLLAPSE", "TAKER_SPIKE", "TAKER_COLLAPSE"]:
        df = all_results.get(label, pd.DataFrame())
        if len(df) < 3:
            print(f"  {label}: INCONCLUSIVE (n={len(df)}, too few)")
            continue

        r60 = df["ret_60m"].dropna()
        rand_r60 = random_base["ret_60m"].dropna()
        delta = r60.mean() - rand_r60.mean()

        # Simple significance
        pooled_std = np.sqrt((r60.std()**2 + rand_r60.std()**2) / 2)
        se = pooled_std * np.sqrt(1/len(r60) + 1/len(rand_r60))
        t_stat = delta / se if se > 0 else 0

        if abs(t_stat) > 2.58 and abs(delta) > 0.001:
            verdict = "VALID EDGE" if delta > 0 else "VALID EDGE (negative)"
        elif abs(t_stat) > 1.96:
            verdict = "WEAK"
        elif abs(t_stat) > 1.65:
            verdict = "WEAK (marginal)"
        else:
            verdict = "NO EDGE"

        print(f"  {label}: {verdict}")
        print(f"    60m: {r60.mean()*100:+.4f}% vs random {rand_r60.mean()*100:+.4f}%  delta={delta*100:+.4f}%  t={t_stat:+.2f}")

    print(f"\n  ⚠ CAVEAT: Only 26 days of data. Results are NOT robust.")
    print(f"  ⚠ Train/test split impossible. No temporal validation.")
    print(f"  ⚠ Treat as preliminary hypothesis, not confirmed edge.")


if __name__ == "__main__":
    main()
