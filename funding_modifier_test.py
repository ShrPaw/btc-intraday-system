"""
FUNDING AS MODIFIER — Does funding change the outcome of a large move?
======================================================================
BTCUSDT only. Single event definition. No optimization.

Event: top 10% of absolute 5-minute returns.
Question: does funding state at event time change what happens next?
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "features")

# =========================================================
# CONFIG — deliberately minimal, no tuning
# =========================================================

MOVE_PERCENTILE = 90          # top 10% absolute 5m moves
FUNDING_HIGH_PCTL = 80        # top 20% funding = HIGH
FUNDING_LOW_PCTL = 20         # bottom 20% funding = LOW

FORWARD_HORIZONS = [5, 15, 30, 60]  # minutes
MAE_MFE_HORIZON = 60                 # bars to track MAE/MFE (1m bars)

TRAIN_CUTOFF = pd.Timestamp("2025-12-01")

# =========================================================
# DATA
# =========================================================

def load_data():
    price = pd.read_csv(os.path.join(DATA_DIR, "btcusdt_1m.csv"), parse_dates=["timestamp"])
    price = price.sort_values("timestamp").reset_index(drop=True)

    funding = pd.read_csv(os.path.join(DATA_DIR, "btcusdt_funding.csv"), parse_dates=["timestamp"])
    funding = funding.sort_values("timestamp").reset_index(drop=True)

    return price, funding


# =========================================================
# STEP 1 — DEFINE BASE EVENT
# =========================================================

def define_events(price):
    """Large move bar: absolute return over 5 minutes > top 10%.
    Uses rolling 5000-bar window to get stable threshold."""
    price = price.copy()
    price["ret_5m"] = price["price"].pct_change(5).abs()
    price["ret_5m_signed"] = price["price"].pct_change(5)

    # Rolling threshold (past-only, 5000 bar window ≈ 3.5 days)
    threshold = price["ret_5m"].rolling(5000, min_periods=500).quantile(MOVE_PERCENTILE / 100).shift(1)
    price["is_event"] = price["ret_5m"] > threshold
    price["move_direction"] = np.sign(price["ret_5m_signed"])

    events = price[price["is_event"]][["timestamp", "price", "ret_5m", "ret_5m_signed", "move_direction"]].copy()
    events = events.rename(columns={"price": "event_price"}).reset_index(drop=True)

    return events


# =========================================================
# STEP 2 — LABEL FUNDING
# =========================================================

def label_funding(events, funding):
    """Assign funding state at event time: HIGH / NORMAL / LOW."""
    events = events.copy()

    # Merge funding rate at each event timestamp (backward = most recent funding)
    events = pd.merge_asof(
        events.sort_values("timestamp"),
        funding[["timestamp", "fundingRate"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    # Rolling percentile thresholds (past-only)
    funding_sorted = funding.sort_values("timestamp").copy()
    funding_sorted["p80"] = funding_sorted["fundingRate"].rolling(30, min_periods=10).quantile(0.80).shift(1)
    funding_sorted["p20"] = funding_sorted["fundingRate"].rolling(30, min_periods=10).quantile(0.20).shift(1)

    events = pd.merge_asof(
        events.sort_values("timestamp"),
        funding_sorted[["timestamp", "p80", "p20"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    events["funding_state"] = "NORMAL"
    events.loc[events["fundingRate"] > events["p80"], "funding_state"] = "HIGH"
    events.loc[events["fundingRate"] < events["p20"], "funding_state"] = "LOW"

    return events


# =========================================================
# STEP 3 — FORWARD OUTCOMES (vectorized)
# =========================================================

def compute_outcomes(events, price):
    """For each event, compute forward returns and MAE/MFE."""
    price_values = price["price"].values
    n_bars = len(price_values)
    results = []

    for _, event in events.iterrows():
        event_time = event["timestamp"]
        event_price = event["event_price"]
        direction = event["move_direction"]

        # Find event bar index
        mask = price["timestamp"] >= event_time
        if mask.sum() == 0:
            continue
        idx = price.index[mask][0]

        # Forward returns at each horizon
        forward_rets = {}
        for h in FORWARD_HORIZONS:
            target_idx = idx + h
            if target_idx < n_bars:
                forward_rets[f"ret_{h}m"] = (price_values[target_idx] - event_price) / event_price
            else:
                forward_rets[f"ret_{h}m"] = np.nan

        # MAE/MFE over MAE_MFE_HORIZON 1m bars
        end_idx = min(idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[idx:end_idx]
        if len(path) < 2:
            continue

        path_pct = (path - event_price) / event_price

        if direction >= 0:  # up move
            favorable = path_pct
            adverse = -path_pct
        else:  # down move
            favorable = -path_pct
            adverse = path_pct

        mfe = np.max(favorable)
        mae = np.max(adverse)

        results.append({
            "timestamp": event["timestamp"],
            "event_price": event_price,
            "move_direction": direction,
            "ret_5m_signed": event["ret_5m_signed"],
            "funding_rate": event["fundingRate"],
            "funding_state": event["funding_state"],
            **forward_rets,
            "mfe_60m": mfe,
            "mae_60m": mae,
            "mfe_mae_ratio": mfe / mae if mae > 0 else np.nan,
        })

    return pd.DataFrame(results)


# =========================================================
# STEP 4 — GROUP COMPARISON
# =========================================================

def group_comparison(df, label="ALL"):
    print(f"\n{'='*70}")
    print(f"  GROUP COMPARISON — {label}")
    print(f"{'='*70}")

    for state in ["HIGH", "NORMAL", "LOW"]:
        sub = df[df["funding_state"] == state]
        n = len(sub)
        if n < 5:
            print(f"\n  {state:>8} funding (n={n}) — TOO FEW")
            continue

        print(f"\n  {state:>8} funding  |  n = {n}")
        print(f"  {'─'*50}")

        for h in FORWARD_HORIZONS:
            col = f"ret_{h}m"
            vals = sub[col].dropna()
            if len(vals) < 3:
                continue
            win_rate = (vals > 0).sum() / len(vals) * 100
            print(f"    {h:>3}m return:  mean {vals.mean()*100:+.4f}%  median {vals.median()*100:+.4f}%  win {win_rate:.1f}%")

        mfe = sub["mfe_60m"].dropna()
        mae = sub["mae_60m"].dropna()
        ratio = sub["mfe_mae_ratio"].dropna()
        if len(mfe) > 0:
            print(f"    MFE 60m:   median {mfe.median()*100:.3f}%  mean {mfe.mean()*100:.3f}%")
            print(f"    MAE 60m:   median {mae.median()*100:.3f}%  mean {mae.mean()*100:.3f}%")
            print(f"    MFE/MAE:   median {ratio.median():.3f}  mean {ratio.mean():.3f}")


# =========================================================
# STEP 5 — BASELINE
# =========================================================

def baseline_comparison(events_df, price):
    print(f"\n{'='*70}")
    print(f"  BASELINE COMPARISON")
    print(f"{'='*70}")

    n_events = len(events_df)
    price_values = price["price"].values
    n_bars = len(price_values)
    valid_range = n_bars - MAE_MFE_HORIZON - 10

    rng = np.random.default_rng(42)

    # --- Baseline 1: Random timestamps ---
    random_indices = rng.choice(valid_range, size=n_events, replace=False)
    random_indices = np.sort(random_indices)

    random_results = []
    for idx in random_indices:
        event_price = price_values[idx]
        if event_price == 0:
            continue
        direction = rng.choice([-1, 1])

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

        if direction >= 0:
            mfe, mae = np.max(path_pct), np.max(-path_pct)
        else:
            mfe, mae = np.max(-path_pct), np.max(path_pct)

        random_results.append({
            "source": "random", "direction": direction, **forward_rets,
            "mfe_60m": mfe, "mae_60m": mae,
            "mfe_mae_ratio": mfe / mae if mae > 0 else np.nan,
        })

    random_df = pd.DataFrame(random_results)

    # --- Baseline 2: Same-time-of-day random ---
    price_hours = price["timestamp"].dt.hour.values
    hour_indices = {}
    for h in range(24):
        mask = (price_hours == h) & (np.arange(n_bars) < valid_range)
        hour_indices[h] = np.where(mask)[0]

    same_time_results = []
    for _, event in events_df.iterrows():
        hour = event["timestamp"].hour
        candidates = hour_indices.get(hour, np.array([]))
        if len(candidates) < 5:
            continue
        pick_idx = rng.choice(candidates)
        event_price = price_values[pick_idx]
        if event_price == 0:
            continue
        direction = rng.choice([-1, 1])

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

        if direction >= 0:
            mfe, mae = np.max(path_pct), np.max(-path_pct)
        else:
            mfe, mae = np.max(-path_pct), np.max(path_pct)

        same_time_results.append({
            "source": "same_time", "direction": direction, **forward_rets,
            "mfe_60m": mfe, "mae_60m": mae,
            "mfe_mae_ratio": mfe / mae if mae > 0 else np.nan,
        })

    same_time_df = pd.DataFrame(same_time_results)

    # --- Print ---
    print(f"\n  {'Group':<25} {'N':>6} {'5m ret':>10} {'15m ret':>10} {'30m ret':>10} {'60m ret':>10} {'MFE/MAE':>10}")
    print(f"  {'─'*85}")

    for state in ["HIGH", "NORMAL", "LOW"]:
        sub = events_df[events_df["funding_state"] == state]
        if len(sub) < 3:
            continue
        r5 = sub["ret_5m"].dropna()
        r15 = sub["ret_15m"].dropna()
        r30 = sub["ret_30m"].dropna()
        r60 = sub["ret_60m"].dropna()
        ratio = sub["mfe_mae_ratio"].dropna()
        print(f"  {'EVENT/' + state:<25} {len(sub):>6} {r5.mean()*100:>+10.4f} {r15.mean()*100:>+10.4f} {r30.mean()*100:>+10.4f} {r60.mean()*100:>+10.4f} {ratio.mean():>10.3f}")

    r5 = events_df["ret_5m"].dropna()
    r15 = events_df["ret_15m"].dropna()
    r30 = events_df["ret_30m"].dropna()
    r60 = events_df["ret_60m"].dropna()
    ratio = events_df["mfe_mae_ratio"].dropna()
    print(f"  {'EVENT/ALL':<25} {len(events_df):>6} {r5.mean()*100:>+10.4f} {r15.mean()*100:>+10.4f} {r30.mean()*100:>+10.4f} {r60.mean()*100:>+10.4f} {ratio.mean():>10.3f}")

    print(f"  {'─'*85}")

    if len(random_df) > 0:
        r5 = random_df["ret_5m"].dropna()
        r15 = random_df["ret_15m"].dropna()
        r30 = random_df["ret_30m"].dropna()
        r60 = random_df["ret_60m"].dropna()
        ratio = random_df["mfe_mae_ratio"].dropna()
        print(f"  {'BASELINE/random':<25} {len(random_df):>6} {r5.mean()*100:>+10.4f} {r15.mean()*100:>+10.4f} {r30.mean()*100:>+10.4f} {r60.mean()*100:>+10.4f} {ratio.mean():>10.3f}")

    if len(same_time_df) > 0:
        r5 = same_time_df["ret_5m"].dropna()
        r15 = same_time_df["ret_15m"].dropna()
        r30 = same_time_df["ret_30m"].dropna()
        r60 = same_time_df["ret_60m"].dropna()
        ratio = same_time_df["mfe_mae_ratio"].dropna()
        print(f"  {'BASELINE/same_time':<25} {len(same_time_df):>6} {r5.mean()*100:>+10.4f} {r15.mean()*100:>+10.4f} {r30.mean()*100:>+10.4f} {r60.mean()*100:>+10.4f} {ratio.mean():>10.3f}")

    return random_df, same_time_df


# =========================================================
# STEP 6 — ROBUSTNESS
# =========================================================

def robustness_check(df):
    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS — TRAIN/TEST SPLIT")
    print(f"  Train: before {TRAIN_CUTOFF.date()}")
    print(f"  Test:  from {TRAIN_CUTOFF.date()}")
    print(f"{'='*70}")

    train = df[df["timestamp"] < TRAIN_CUTOFF]
    test = df[df["timestamp"] >= TRAIN_CUTOFF]

    for split_name, split_df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n  --- {split_name} ---")
        for state in ["HIGH", "NORMAL", "LOW"]:
            sub = split_df[split_df["funding_state"] == state]
            n = len(sub)
            if n < 3:
                print(f"    {state:>8}: n={n} (too few)")
                continue

            vals_5 = sub["ret_5m"].dropna()
            vals_60 = sub["ret_60m"].dropna()
            ratio = sub["mfe_mae_ratio"].dropna()

            win_5 = (vals_5 > 0).sum() / len(vals_5) * 100 if len(vals_5) > 0 else 0
            win_60 = (vals_60 > 0).sum() / len(vals_60) * 100 if len(vals_60) > 0 else 0

            print(f"    {state:>8}: n={n:>4}  5m {vals_5.mean()*100:+.4f}% (win {win_5:.0f}%)  60m {vals_60.mean()*100:+.4f}% (win {win_60:.0f}%)  MFE/MAE {ratio.mean():.3f}")

    print(f"\n  --- CONSISTENCY CHECK ---")
    for state in ["HIGH", "LOW"]:
        for period, period_label in [(train, "TRAIN"), (test, "TEST")]:
            state_df = period[period["funding_state"] == state]
            normal_df = period[period["funding_state"] == "NORMAL"]
            if len(state_df) < 3 or len(normal_df) < 3:
                print(f"    {state} vs NORMAL ({period_label}): insufficient data")
                continue
            s_60 = state_df["ret_60m"].dropna().mean()
            n_60 = normal_df["ret_60m"].dropna().mean()
            delta = s_60 - n_60
            print(f"    {state} vs NORMAL ({period_label}): {state}={s_60*100:+.4f}% vs NORMAL={n_60*100:+.4f}%  delta={delta*100:+.4f}%")

    train = train  # silence unused
    test = test


# =========================================================
# STEP 7 — DECISION
# =========================================================

def render_decision(events_df, random_df, same_time_df):
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    normal = events_df[events_df["funding_state"] == "NORMAL"]

    for state in ["HIGH", "LOW"]:
        sub = events_df[events_df["funding_state"] == state]
        if len(sub) < 5 or len(normal) < 5:
            print(f"\n  {state}: INCONCLUSIVE (too few events)")
            continue

        state_60 = sub["ret_60m"].dropna()
        normal_60 = normal["ret_60m"].dropna()
        diff = state_60.mean() - normal_60.mean()

        state_ratio = sub["mfe_mae_ratio"].dropna()
        normal_ratio = normal["mfe_mae_ratio"].dropna()
        ratio_diff = state_ratio.mean() - normal_ratio.mean()

        state_wr = (state_60 > 0).sum() / len(state_60) * 100
        normal_wr = (normal_60 > 0).sum() / len(normal_60) * 100
        wr_diff = state_wr - normal_wr

        pooled_std = np.sqrt((state_60.std()**2 + normal_60.std()**2) / 2)
        se = pooled_std * np.sqrt(1/len(state_60) + 1/len(normal_60))
        t_stat = diff / se if se > 0 else 0

        print(f"\n  {state} funding vs NORMAL:")
        print(f"    60m return:  {state_60.mean()*100:+.4f}% vs {normal_60.mean()*100:+.4f}%  (diff {diff*100:+.4f}%)")
        print(f"    Win rate:    {state_wr:.1f}% vs {normal_wr:.1f}%  (diff {wr_diff:+.1f}%)")
        print(f"    MFE/MAE:     {state_ratio.mean():.3f} vs {normal_ratio.mean():.3f}  (diff {ratio_diff:+.3f})")
        print(f"    t-stat:      {t_stat:+.2f}  {'***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.65 else 'ns'}")
        print(f"    N:           {len(state_60)} vs {len(normal_60)}")

    print(f"\n  Overall system (all events):")
    all_60 = events_df["ret_60m"].dropna()
    rand_60 = random_df["ret_60m"].dropna() if len(random_df) > 0 else pd.Series([0])
    st_60 = same_time_df["ret_60m"].dropna() if len(same_time_df) > 0 else pd.Series([0])

    print(f"    Event 60m mean:       {all_60.mean()*100:+.4f}%")
    print(f"    Random 60m mean:      {rand_60.mean()*100:+.4f}%")
    print(f"    Same-time 60m mean:   {st_60.mean()*100:+.4f}%")
    print(f"    Event vs Random:      {(all_60.mean() - rand_60.mean())*100:+.4f}%")
    print(f"    Event vs Same-time:   {(all_60.mean() - st_60.mean())*100:+.4f}%")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  FUNDING AS MODIFIER — Does funding change the outcome of a large move?")
    print("=" * 70)

    print("\nLoading BTCUSDT data...")
    price, funding = load_data()
    print(f"  1m candles: {len(price):,} ({price['timestamp'].min().date()} → {price['timestamp'].max().date()})")
    print(f"  Funding:    {len(funding):,} ({funding['timestamp'].min().date()} → {funding['timestamp'].max().date()})")

    # STEP 1
    print(f"\nSTEP 1 — Defining events (top {100-MOVE_PERCENTILE}% absolute 5m moves, rolling 5000-bar window)...")
    events = define_events(price)
    print(f"  Events found: {len(events)}")
    print(f"  5m return threshold range: {events['ret_5m'].min()*100:.3f}% – {events['ret_5m'].max()*100:.3f}%")
    print(f"  Mean event 5m move: {events['ret_5m'].mean()*100:.3f}%")

    # STEP 2
    print(f"\nSTEP 2 — Labeling funding state...")
    events = label_funding(events, funding)
    print(f"  HIGH:   {(events['funding_state'] == 'HIGH').sum()}")
    print(f"  NORMAL: {(events['funding_state'] == 'NORMAL').sum()}")
    print(f"  LOW:    {(events['funding_state'] == 'LOW').sum()}")

    # STEP 3
    print(f"\nSTEP 3 — Computing forward outcomes...")
    outcomes = compute_outcomes(events, price)
    print(f"  Events with outcomes: {len(outcomes)}")

    # STEP 4
    group_comparison(outcomes, "ALL PERIODS")

    # STEP 5
    random_df, same_time_df = baseline_comparison(outcomes, price)

    # STEP 6
    robustness_check(outcomes)

    # STEP 7
    render_decision(outcomes, random_df, same_time_df)

    # Save
    out_path = os.path.join(DATA_DIR, "funding_modifier_results.csv")
    outcomes.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
