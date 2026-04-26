"""
OI/TAKER EXPLORATORY STUDY — Hypothesis generation only
========================================================
⚠ EXPLORATORY ONLY — NOT a validated edge ⚠

Rules:
- No train/test conclusions
- No deployable signal claims
- N < 50 → INCONCLUSIVE regardless of performance
- Use only to decide if the idea deserves more data

Events: OI spike, OI collapse, taker imbalance, LS ratio extremes
Horizons: 1h, 4h, 24h
Data: 26 days (Mar 31 – Apr 25 2026)
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

EXTREME_PCTL = 90   # top/bottom 10%

FORWARD_HORIZONS = [60, 240, 1440]  # 1h, 4h, 24h in minutes
MAE_MFE_HORIZON = 1440              # 24h in 1m bars

MIN_N = 50  # hard floor

TRAIN_CUTOFF = pd.Timestamp("2025-12-01")  # unusable with 26 days, but defined

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
# EVENT DEFINITIONS
# =========================================================

def build_event_frame(oi, taker, ls):
    """Merge all hourly data, compute changes, define events."""
    df = oi.copy()

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

    # Changes
    df["oi_pct"] = df["sumOpenInterest"].pct_change()
    df["taker_ratio"] = df["buySellRatio"]
    df["taker_ratio_pct"] = df["buySellRatio"].pct_change()
    df["ls_ratio"] = df["globalLSRatio"]
    df["ls_ratio_pct"] = df["globalLSRatio"].pct_change()

    # Rolling thresholds (past-only, window=48h = 2 days)
    w = 48
    for col in ["oi_pct", "taker_ratio_pct", "ls_ratio_pct"]:
        df[f"{col}_p90"] = df[col].rolling(w, min_periods=12).quantile(0.90).shift(1)
        df[f"{col}_p10"] = df[col].rolling(w, min_periods=12).quantile(0.10).shift(1)

    # Events
    events = {}

    # 1. OI spike
    mask = df["oi_pct"] > df["oi_pct_p90"]
    events["OI_SPIKE"] = df[mask][["timestamp", "sumOpenInterest", "oi_pct"]].copy()
    events["OI_SPIKE"]["event_value"] = events["OI_SPIKE"]["oi_pct"]

    # 2. OI collapse
    mask = df["oi_pct"] < df["oi_pct_p10"]
    events["OI_COLLAPSE"] = df[mask][["timestamp", "sumOpenInterest", "oi_pct"]].copy()
    events["OI_COLLAPSE"]["event_value"] = events["OI_COLLAPSE"]["oi_pct"]

    # 3. Taker buy/sell imbalance (spike = buyers flood in)
    mask = df["taker_ratio_pct"] > df["taker_ratio_pct_p90"]
    events["TAKER_SPIKE"] = df[mask][["timestamp", "buySellRatio", "taker_ratio_pct"]].copy()
    events["TAKER_SPIKE"]["event_value"] = events["TAKER_SPIKE"]["taker_ratio_pct"]

    # 4. Taker collapse (buyers evaporate)
    mask = df["taker_ratio_pct"] < df["taker_ratio_pct_p10"]
    events["TAKER_COLLAPSE"] = df[mask][["timestamp", "buySellRatio", "taker_ratio_pct"]].copy()
    events["TAKER_COLLAPSE"]["event_value"] = events["TAKER_COLLAPSE"]["taker_ratio_pct"]

    # 5. LS ratio extreme high (crowd is long)
    mask = df["ls_ratio_pct"] > df["ls_ratio_pct_p90"]
    events["LS_HIGH"] = df[mask][["timestamp", "globalLSRatio", "ls_ratio_pct"]].copy()
    events["LS_HIGH"]["event_value"] = events["LS_HIGH"]["ls_ratio_pct"]

    # 6. LS ratio extreme low (crowd is short)
    mask = df["ls_ratio_pct"] < df["ls_ratio_pct_p10"]
    events["LS_LOW"] = df[mask][["timestamp", "globalLSRatio", "ls_ratio_pct"]].copy()
    events["LS_LOW"]["event_value"] = events["LS_LOW"]["ls_ratio_pct"]

    return events


# =========================================================
# FORWARD TRACKING
# =========================================================

def track_event(event_df, price, event_label):
    """Track forward price path for each event."""
    price_values = price["price"].values
    price_ts = price["timestamp"].values
    n_bars = len(price_values)
    results = []

    for _, event in event_df.iterrows():
        event_time = event["timestamp"]

        # Find nearest 1m bar at or after event
        mask = price["timestamp"] >= event_time
        if mask.sum() == 0:
            continue
        idx = price.index[mask][0]

        event_price = price_values[idx]

        # Forward returns at each horizon
        forward = {}
        for h in FORWARD_HORIZONS:
            target = idx + h
            if target < n_bars:
                forward[f"ret_{h}m"] = (price_values[target] - event_price) / event_price
            else:
                forward[f"ret_{h}m"] = np.nan

        # MAE/MFE over MAE_MFE_HORIZON
        end = min(idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[idx:end]
        if len(path) < 10:
            continue
        path_pct = (path - event_price) / event_price

        mfe = np.max(path_pct)
        mae = np.max(-path_pct)

        # Reversal: does initial 1h move reverse by 24h?
        if len(path) >= 60:
            move_1h = path_pct[59]
            move_24h = path_pct[-1]
            reversal = (move_1h * move_24h) < 0
            continuation = abs(move_24h) > abs(move_1h) and (move_1h * move_24h) > 0
        else:
            reversal = None
            continuation = None

        results.append({
            "timestamp": event["timestamp"],
            "event_label": event_label,
            "event_value": event.get("event_value"),
            "event_price": event_price,
            **forward,
            "mfe_24h": mfe,
            "mae_24h": mae,
            "mfe_mae_ratio": mfe / mae if mae > 0 else np.nan,
            "reversal": reversal,
            "continuation": continuation,
        })

    return pd.DataFrame(results)


# =========================================================
# BASELINES
# =========================================================

def build_baselines(price, n_events):
    """Random and same-time baselines."""
    price_values = price["price"].values
    n_bars = len(price_values)
    valid = n_bars - MAE_MFE_HORIZON - 10
    rng = np.random.default_rng(42)

    price_hours = price["timestamp"].dt.hour.values
    hour_idx = {}
    for h in range(24):
        mask = (price_hours == h) & (np.arange(n_bars) < valid)
        hour_idx[h] = np.where(mask)[0]

    results = {"random": [], "same_time": []}

    # Random
    rand_idx = rng.choice(valid, size=n_events, replace=False)
    for idx in rand_idx:
        ep = price_values[idx]
        if ep == 0:
            continue
        fwd = {}
        for h in FORWARD_HORIZONS:
            t = idx + h
            if t < n_bars:
                fwd[f"ret_{h}m"] = (price_values[t] - ep) / ep
            else:
                fwd[f"ret_{h}m"] = np.nan
        end = min(idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[idx:end]
        if len(path) < 10:
            continue
        pp = (path - ep) / ep
        results["random"].append({**fwd, "mfe_24h": np.max(pp), "mae_24h": np.max(-pp)})

    # Same-time
    hours = price["timestamp"].dt.hour.values
    for _ in range(min(n_events, 200)):
        pick = rng.choice(valid)
        hour = hours[pick]
        cands = hour_idx.get(hour, np.array([]))
        if len(cands) < 5:
            continue
        idx = rng.choice(cands)
        ep = price_values[idx]
        if ep == 0:
            continue
        fwd = {}
        for h in FORWARD_HORIZONS:
            t = idx + h
            if t < n_bars:
                fwd[f"ret_{h}m"] = (price_values[t] - ep) / ep
            else:
                fwd[f"ret_{h}m"] = np.nan
        end = min(idx + MAE_MFE_HORIZON, n_bars)
        path = price_values[idx:end]
        if len(path) < 10:
            continue
        pp = (path - ep) / ep
        results["same_time"].append({**fwd, "mfe_24h": np.max(pp), "mae_24h": np.max(-pp)})

    return pd.DataFrame(results["random"]), pd.DataFrame(results["same_time"])


# =========================================================
# ANALYSIS
# =========================================================

def analyze_event(results, label, rand_base, same_time_base):
    """Full analysis for one event type. Returns classification."""
    n = len(results)

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")

    # Hard floor
    if n < MIN_N:
        print(f"  N = {n}  (< {MIN_N})")
        print(f"  Classification: INCONCLUSIVE")
        return {"label": label, "n": n, "classification": "INCONCLUSIVE"}

    print(f"  N = {n}")

    # Forward returns
    print(f"\n  Forward returns:")
    for h in FORWARD_HORIZONS:
        col = f"ret_{h}m"
        vals = results[col].dropna()
        if len(vals) < 3:
            continue
        wr = (vals > 0).sum() / len(vals) * 100
        h_label = f"{h//60}h" if h >= 60 else f"{h}m"
        print(f"    {h_label:>4}: mean {vals.mean()*100:+.4f}%  median {vals.median()*100:+.4f}%  win {wr:.1f}%  (n={len(vals)})")

    # MFE/MAE
    mfe = results["mfe_24h"].dropna()
    mae = results["mae_24h"].dropna()
    ratio = results["mfe_mae_ratio"].dropna()
    print(f"\n  Excursion (24h):")
    print(f"    MFE:  median {mfe.median()*100:.3f}%  mean {mfe.mean()*100:.3f}%")
    print(f"    MAE:  median {mae.median()*100:.3f}%  mean {mae.mean()*100:.3f}%")
    print(f"    MFE/MAE:  median {ratio.median():.3f}  mean {ratio.mean():.3f}")

    # Reversal / continuation
    rev = results["reversal"].dropna()
    cont = results["continuation"].dropna()
    if len(rev) > 0:
        print(f"\n  Path behavior:")
        print(f"    Reversal (1h→24h):    {rev.sum()}/{len(rev)} = {rev.mean()*100:.1f}%")
        print(f"    Continuation (1h→24h): {cont.sum()}/{len(cont)} = {cont.mean()*100:.1f}%")

    # Baselines
    print(f"\n  vs Baselines (1h return):")
    event_1h = results["ret_60m"].dropna()
    rand_1h = rand_base["ret_60m"].dropna()
    st_1h = same_time_base["ret_60m"].dropna()

    if len(rand_1h) > 0 and len(event_1h) > 0:
        delta_rand = event_1h.mean() - rand_1h.mean()
        pooled = np.sqrt((event_1h.std()**2 + rand_1h.std()**2) / 2)
        se = pooled * np.sqrt(1/len(event_1h) + 1/len(rand_1h))
        t_rand = delta_rand / se if se > 0 else 0
        print(f"    vs random:    delta {delta_rand*100:+.4f}%  t={t_rand:+.2f}")

    if len(st_1h) > 0 and len(event_1h) > 0:
        delta_st = event_1h.mean() - st_1h.mean()
        pooled = np.sqrt((event_1h.std()**2 + st_1h.std()**2) / 2)
        se = pooled * np.sqrt(1/len(event_1h) + 1/len(st_1h))
        t_st = delta_st / se if se > 0 else 0
        print(f"    vs same-time: delta {delta_st*100:+.4f}%  t={t_st:+.2f}")

    # Classification
    # Check 1h, 4h, 24h consistency
    rets = {}
    for h in FORWARD_HORIZONS:
        col = f"ret_{h}m"
        vals = results[col].dropna()
        rets[h] = vals.mean() if len(vals) > 0 else 0

    # All positive or all negative across horizons?
    signs = [np.sign(rets[h]) for h in FORWARD_HORIZONS if rets[h] != 0]
    consistent = len(set(signs)) == 1 if len(signs) >= 2 else False

    # vs baselines
    beats_rand = abs(t_rand) > 1.96 if len(rand_1h) > 0 else False
    beats_st = abs(t_st) > 1.96 if len(st_1h) > 0 else False

    if consistent and (beats_rand or beats_st):
        classification = "WORTH COLLECTING MORE DATA"
    elif not consistent and not beats_rand and not beats_st:
        classification = "NOT WORTH CONTINUING"
    else:
        classification = "INCONCLUSIVE"

    print(f"\n  Classification: {classification}")
    return {"label": label, "n": n, "classification": classification}


# =========================================================
# SUMMARY TABLE
# =========================================================

def print_summary(all_results, rand_base, st_base):
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")

    print(f"\n  {'Event':<20} {'N':>5} {'1h ret':>10} {'4h ret':>10} {'24h ret':>10} {'MFE/MAE':>10} {'Rev%':>8} {'Class':<25}")
    print(f"  {'─'*100}")

    for label, df in all_results.items():
        n = len(df)
        if n < 3:
            print(f"  {label:<20} {n:>5} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>8} {'INCONCLUSIVE':<25}")
            continue

        r1 = df["ret_60m"].dropna().mean() * 100 if "ret_60m" in df else 0
        r4 = df["ret_240m"].dropna().mean() * 100 if "ret_240m" in df else 0
        r24 = df["ret_1440m"].dropna().mean() * 100 if "ret_1440m" in df else 0
        ratio = df["mfe_mae_ratio"].dropna().median() if "mfe_mae_ratio" in df else 0
        rev = df["reversal"].dropna().mean() * 100 if "reversal" in df else 0

        if n < MIN_N:
            cls = "INCONCLUSIVE (n<50)"
        else:
            cls = ""  # filled by analyze

        print(f"  {label:<20} {n:>5} {r1:>+9.4f}% {r4:>+9.4f}% {r24:>+9.4f}% {ratio:>10.3f} {rev:>7.1f}% {cls}")

    # Baselines
    for name, df in [("random", rand_base), ("same_time", st_base)]:
        n = len(df)
        r1 = df["ret_60m"].dropna().mean() * 100 if "ret_60m" in df else 0
        r4 = df["ret_240m"].dropna().mean() * 100 if "ret_240m" in df else 0
        r24 = df["ret_1440m"].dropna().mean() * 100 if "ret_1440m" in df else 0
        ratio = df["mfe_mae_ratio"].dropna().median() if "mfe_mae_ratio" in df else 0
        print(f"  {'BASE/' + name:<20} {n:>5} {r1:>+9.4f}% {r4:>+9.4f}% {r24:>+9.4f}% {ratio:>10.3f} {'—':>8}")


# =========================================================
# FINAL VERDICT
# =========================================================

def final_verdict(classifications):
    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT — EXPLORATORY ONLY")
    print(f"{'='*70}")

    for c in classifications:
        icon = "🟢" if "WORTH" in c["classification"] else ("🔴" if "NOT" in c["classification"] else "🟡")
        print(f"  {icon} {c['label']:<20} N={c['n']:>4}  →  {c['classification']}")

    print(f"\n  ⚠ Data period: 26 days only (Mar 31 – Apr 25 2026)")
    print(f"  ⚠ No train/test validation possible")
    print(f"  ⚠ No liquidation data available")
    print(f"  ⚠ All results are HYPOTHESIS-GRADE only")
    print(f"  ⚠ Do NOT deploy any signal based on this analysis")

    worth = [c for c in classifications if "WORTH" in c["classification"]]
    if worth:
        print(f"\n  Recommendation: Collect extended data for {', '.join(c['label'] for c in worth)}")
    else:
        print(f"\n  Recommendation: None of these ideas warrant extended data collection")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  OI/TAKER EXPLORATORY STUDY")
    print("  ⚠ EXPLORATORY ONLY — NOT a validated edge ⚠")
    print("  ⚠ 26 days of data. No train/test. No deployment.")
    print("=" * 70)

    print("\nLoading data...")
    price, oi, taker, ls = load_data()
    print(f"  1m candles: {len(price):,}")
    print(f"  OI/taker/LS: {len(oi):,} hourly bars ({oi['timestamp'].min().date()} → {oi['timestamp'].max().date()})")

    print("\nDefining events...")
    event_frames = build_event_frame(oi, taker, ls)
    for label, ef in event_frames.items():
        print(f"  {label}: {len(ef)}")

    print("\nTracking forward paths...")
    all_results = {}
    for label, ef in event_frames.items():
        results = track_event(ef, price, label)
        all_results[label] = results
        print(f"  {label}: {len(results)} tracked")

    n_max = max(len(df) for df in all_results.values()) if all_results else 50
    print(f"\nBuilding baselines (n={n_max})...")
    rand_base, st_base = build_baselines(price, n_max)
    print(f"  Random: {len(rand_base)}")
    print(f"  Same-time: {len(st_base)}")

    print(f"\n{'='*70}")
    print(f"  EVENT ANALYSIS")
    print(f"{'='*70}")

    classifications = []
    for label, df in all_results.items():
        c = analyze_event(df, label, rand_base, st_base)
        classifications.append(c)

    print_summary(all_results, rand_base, st_base)
    final_verdict(classifications)

    # Save
    out = pd.concat(all_results.values(), ignore_index=True)
    out_path = os.path.join(DATA_DIR, "oi_taker_exploratory_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
