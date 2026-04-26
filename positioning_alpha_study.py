"""
POSITIONING ALPHA STUDY — BTCUSDT Perpetual Futures
====================================================
Tests whether positioning data (funding, OI, LS ratio) contains
predictive power for future BTC returns.

This is a MEASUREMENT study, not a strategy.
No entries, no stops, no optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data/features"


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_all_data():
    """Load all positioning data and price data."""
    data = {}

    # Price (1h from existing 1m data)
    print("  Loading price data...")
    price_1m = pd.read_csv(f"{DATA_DIR}/btcusdt_1m.csv")
    price_1m["timestamp"] = pd.to_datetime(price_1m["timestamp"])
    price_1h = price_1m.set_index("timestamp").resample("1h").agg({
        "price": ["first", "max", "min", "last"],
        "volume": "sum"
    }).dropna()
    price_1h.columns = ["open", "high", "low", "close", "volume"]
    price_1h = price_1h.reset_index()
    data["price_1h"] = price_1h

    # Funding (8h)
    if pd.io.common.file_exists(f"{DATA_DIR}/btcusdt_funding.csv"):
        funding = pd.read_csv(f"{DATA_DIR}/btcusdt_funding.csv")
        funding["timestamp"] = pd.to_datetime(funding["timestamp"])
        data["funding"] = funding
        print(f"  Funding: {len(funding)} records")

    # OI (1h)
    if pd.io.common.file_exists(f"{DATA_DIR}/btcusdt_oi_1h.csv"):
        oi = pd.read_csv(f"{DATA_DIR}/btcusdt_oi_1h.csv")
        oi["timestamp"] = pd.to_datetime(oi["timestamp"])
        data["oi"] = oi
        print(f"  OI: {len(oi)} records")

    # LS ratio (1h)
    if pd.io.common.file_exists(f"{DATA_DIR}/btcusdt_ls_ratio_1h.csv"):
        ls = pd.read_csv(f"{DATA_DIR}/btcusdt_ls_ratio_1h.csv")
        ls["timestamp"] = pd.to_datetime(ls["timestamp"])
        data["ls"] = ls
        print(f"  LS ratio: {len(ls)} records")

    # Global LS (1h)
    if pd.io.common.file_exists(f"{DATA_DIR}/btcusdt_global_ls_1h.csv"):
        gls = pd.read_csv(f"{DATA_DIR}/btcusdt_global_ls_1h.csv")
        gls["timestamp"] = pd.to_datetime(gls["timestamp"])
        data["gls"] = gls
        print(f"  Global LS: {len(gls)} records")

    # Taker volume (1h)
    if pd.io.common.file_exists(f"{DATA_DIR}/btcusdt_taker_1h.csv"):
        taker = pd.read_csv(f"{DATA_DIR}/btcusdt_taker_1h.csv")
        taker["timestamp"] = pd.to_datetime(taker["timestamp"])
        data["taker"] = taker
        print(f"  Taker: {len(taker)} records")

    return data


def merge_positioning_data(data):
    """Merge all positioning data onto a 1h price timeline."""
    df = data["price_1h"].copy()

    # Funding: forward-fill from 8h to 1h
    if "funding" in data:
        funding = data["funding"].set_index("timestamp")["fundingRate"]
        df = df.set_index("timestamp")
        df["fundingRate"] = funding.reindex(df.index, method="ffill")
        df = df.reset_index()

    # OI
    if "oi" in data:
        oi = data["oi"].set_index("timestamp")[["sumOpenInterest", "sumOpenInterestValue"]]
        df = df.set_index("timestamp")
        df = df.join(oi, how="left")
        df = df.reset_index()

    # LS ratio
    if "ls" in data:
        ls = data["ls"].set_index("timestamp")[["longShortRatio", "longAccount", "shortAccount"]]
        df = df.set_index("timestamp")
        df = df.join(ls, how="left")
        df = df.reset_index()

    # Global LS
    if "gls" in data:
        gls = data["gls"].set_index("timestamp")[["globalLongRatio", "globalLSRatio"]]
        df = df.set_index("timestamp")
        df = df.join(gls, how="left")
        df = df.reset_index()

    # Taker
    if "taker" in data:
        taker = data["taker"].set_index("timestamp")[["buySellRatio", "buyVol", "sellVol"]]
        df = df.set_index("timestamp")
        df = df.join(taker, how="left")
        df = df.reset_index()

    # Compute derived features
    df["ret_1h"] = df["close"].pct_change(1)

    if "sumOpenInterest" in df.columns:
        df["oi_pct_change_1h"] = df["sumOpenInterest"].pct_change(1)
        df["oi_pct_change_4h"] = df["sumOpenInterest"].pct_change(4)
        df["oi_pct_change_24h"] = df["sumOpenInterest"].pct_change(24)

    if "longShortRatio" in df.columns:
        df["ls_ratio_pctile"] = df["longShortRatio"].rolling(168, min_periods=24).rank(pct=True).shift(1)

    if "globalLongRatio" in df.columns:
        df["gls_pctile"] = df["globalLongRatio"].rolling(168, min_periods=24).rank(pct=True).shift(1)

    if "buySellRatio" in df.columns:
        df["taker_pctile"] = df["buySellRatio"].rolling(168, min_periods=24).rank(pct=True).shift(1)

    if "fundingRate" in df.columns:
        # Funding percentile (rolling)
        df["funding_pctile"] = df["fundingRate"].rolling(168, min_periods=24).rank(pct=True).shift(1)
        # Cumulative funding (sum of last 3 periods = 24h)
        df["funding_cum_24h"] = df["fundingRate"].rolling(3, min_periods=1).sum()
        df["funding_cum_72h"] = df["fundingRate"].rolling(9, min_periods=3).sum()

    # Forward returns (for analysis)
    for bars in [1, 4, 24]:
        df[f"fwd_ret_{bars}h"] = df["close"].shift(-bars) / df["close"] - 1

    return df


# ═══════════════════════════════════════════════════════════
# STEP 1 — DEFINE EVENTS
# ═══════════════════════════════════════════════════════════

def define_events(df):
    """Define event triggers based on positioning extremes."""
    events = {}

    # 1. Funding extremes
    if "fundingRate" in df.columns:
        fr = df["fundingRate"].dropna()
        p90 = fr.quantile(0.90)
        p10 = fr.quantile(0.10)
        p95 = fr.quantile(0.95)
        p05 = fr.quantile(0.05)

        events["high_funding_p90"] = df["fundingRate"] > p90
        events["low_funding_p10"] = df["fundingRate"] < p10
        events["extreme_high_funding_p95"] = df["fundingRate"] > p95
        events["extreme_low_funding_p05"] = df["fundingRate"] < p05

        # Cumulative funding
        if "funding_cum_24h" in df.columns:
            fc = df["funding_cum_24h"].dropna()
            events["high_cum_funding_24h"] = df["funding_cum_24h"] > fc.quantile(0.90)
            events["low_cum_funding_24h"] = df["funding_cum_24h"] < fc.quantile(0.10)

    # 2. OI changes
    if "oi_pct_change_1h" in df.columns:
        oi_1h = df["oi_pct_change_1h"].dropna()
        events["oi_spike_1h"] = df["oi_pct_change_1h"] > oi_1h.quantile(0.90)
        events["oi_collapse_1h"] = df["oi_pct_change_1h"] < oi_1h.quantile(0.10)

    if "oi_pct_change_4h" in df.columns:
        oi_4h = df["oi_pct_change_4h"].dropna()
        events["oi_spike_4h"] = df["oi_pct_change_4h"] > oi_4h.quantile(0.90)
        events["oi_collapse_4h"] = df["oi_pct_change_4h"] < oi_4h.quantile(0.10)

    # 3. LS ratio extremes
    if "longShortRatio" in df.columns:
        ls = df["longShortRatio"].dropna()
        events["long_heavy"] = df["longShortRatio"] > ls.quantile(0.90)
        events["short_heavy"] = df["longShortRatio"] < ls.quantile(0.10)

    if "globalLongRatio" in df.columns:
        gl = df["globalLongRatio"].dropna()
        events["crowded_longs"] = df["globalLongRatio"] > gl.quantile(0.90)
        events["crowded_shorts"] = df["globalLongRatio"] < gl.quantile(0.10)

    # 4. Taker volume
    if "buySellRatio" in df.columns:
        tk = df["buySellRatio"].dropna()
        events["taker_buy_heavy"] = df["buySellRatio"] > tk.quantile(0.90)
        events["taker_sell_heavy"] = df["buySellRatio"] < tk.quantile(0.10)

    # 5. Combined events
    if "fundingRate" in df.columns and "oi_pct_change_1h" in df.columns:
        events["high_fund_oi_spike"] = events.get("high_funding_p90", pd.Series(False, index=df.index)) & events.get("oi_spike_1h", pd.Series(False, index=df.index))
        events["low_fund_oi_spike"] = events.get("low_funding_p10", pd.Series(False, index=df.index)) & events.get("oi_spike_1h", pd.Series(False, index=df.index))

    if "fundingRate" in df.columns and "longShortRatio" in df.columns:
        events["high_fund_long_heavy"] = events.get("high_funding_p90", pd.Series(False, index=df.index)) & events.get("long_heavy", pd.Series(False, index=df.index))
        events["low_fund_short_heavy"] = events.get("low_funding_p10", pd.Series(False, index=df.index)) & events.get("short_heavy", pd.Series(False, index=df.index))

    return events


# ═══════════════════════════════════════════════════════════
# STEP 2 — FORWARD ANALYSIS
# ═══════════════════════════════════════════════════════════

def forward_analysis(df, events, event_name, mask):
    """Compute forward returns for an event."""
    event_bars = df[mask].copy()
    n = len(event_bars)
    if n < 3:
        return None

    results = {"event": event_name, "N": n}

    for horizon in [1, 4, 24]:
        col = f"fwd_ret_{horizon}h"
        if col not in event_bars.columns:
            continue
        rets = event_bars[col].dropna()
        if len(rets) < 3:
            continue

        results[f"mean_{horizon}h"] = rets.mean()
        results[f"median_{horizon}h"] = rets.median()
        results[f"std_{horizon}h"] = rets.std()
        results[f"winrate_{horizon}h"] = (rets > 0).mean()
        results[f"skew_{horizon}h"] = rets.skew() if len(rets) > 5 else np.nan

        # T-stat for mean != 0
        if rets.std() > 0:
            t_stat = rets.mean() / (rets.std() / np.sqrt(len(rets)))
            results[f"tstat_{horizon}h"] = t_stat
            results[f"pval_{horizon}h"] = 2 * (1 - stats.t.cdf(abs(t_stat), len(rets) - 1))
        else:
            results[f"tstat_{horizon}h"] = 0
            results[f"pval_{horizon}h"] = 1.0

    return results


# ═══════════════════════════════════════════════════════════
# STEP 3 — DIRECTIONAL TEST
# ═══════════════════════════════════════════════════════════

def directional_test(df, events):
    """Test specific directional hypotheses."""
    print(f"\n  DIRECTIONAL HYPOTHESIS TESTS")
    print(f"  {'─' * 60}")

    hypotheses = [
        ("H1: High funding → neg returns", "high_funding_p90", "negative", [1, 4, 24]),
        ("H2: Low funding → pos returns", "low_funding_p10", "positive", [1, 4, 24]),
        ("H3: OI spike + price up → exhaustion", None, "negative", [4, 24]),
        ("H4: OI spike + price down → squeeze", None, "positive", [4, 24]),
        ("H5: Crowded longs → neg returns", "crowded_longs", "negative", [4, 24]),
        ("H6: Crowded shorts → pos returns", "crowded_shorts", "positive", [4, 24]),
    ]

    for name, event_key, expected_dir, horizons in hypotheses:
        if event_key and event_key in events:
            mask = events[event_key]
        elif "H3" in name and "oi_spike_1h" in events:
            # OI spike + price up in last 4h
            price_up = df["ret_1h"].rolling(4).sum() > 0.01
            mask = events["oi_spike_1h"] & price_up
        elif "H4" in name and "oi_spike_1h" in events:
            price_down = df["ret_1h"].rolling(4).sum() < -0.01
            mask = events["oi_spike_1h"] & price_down
        else:
            continue

        result = forward_analysis(df, events, name, mask)
        if result is None:
            print(f"\n    {name}: insufficient data")
            continue

        print(f"\n    {name}:")
        print(f"      N={result['N']}")
        for h in horizons:
            mean_key = f"mean_{h}h"
            pval_key = f"pval_{h}h"
            wr_key = f"winrate_{h}h"
            if mean_key in result:
                mean_ret = result[mean_key]
                pval = result[pval_key]
                wr = result[wr_key]
                direction_ok = (expected_dir == "positive" and mean_ret > 0) or (expected_dir == "negative" and mean_ret < 0)
                sig = "✓" if pval < 0.05 else ("~" if pval < 0.10 else "✗")
                dir_icon = "✅" if direction_ok else "❌"
                print(f"      {h}h: mean={mean_ret*100:+.3f}%, wr={wr:.0%}, p={pval:.3f} {sig} {dir_icon}")


# ═══════════════════════════════════════════════════════════
# STEP 4 — FILTER QUALITY (intensity bins)
# ═══════════════════════════════════════════════════════════

def intensity_analysis(df, feature_name, event_label):
    """Check if stronger signal = stronger edge."""
    if feature_name not in df.columns:
        return

    col = df[feature_name].dropna()
    if len(col) < 20:
        return

    print(f"\n  INTENSITY: {event_label} ({feature_name})")

    # Split into quintiles
    quintiles = pd.qcut(df[feature_name], 5, labels=["Q1(lo)", "Q2", "Q3", "Q4", "Q5(hi)"], duplicates="drop")

    print(f"  {'Quintile':<10s} {'N':>5s} {'1h_ret':>8s} {'4h_ret':>8s} {'24h_ret':>8s} {'1h_wr':>6s} {'4h_wr':>6s}")
    print(f"  {'─'*10} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    for q in ["Q1(lo)", "Q2", "Q3", "Q4", "Q5(hi)"]:
        mask = quintiles == q
        n = mask.sum()
        if n < 3:
            continue
        row = {"N": n}
        for h in [1, 4, 24]:
            col_name = f"fwd_ret_{h}h"
            if col_name in df.columns:
                rets = df.loc[mask, col_name].dropna()
                if len(rets) > 0:
                    row[f"{h}h_ret"] = rets.mean()
                    row[f"{h}h_wr"] = (rets > 0).mean()

        print(f"  {q:<10s} {n:>5d} {row.get('1h_ret',0)*100:>+7.3f}% {row.get('4h_ret',0)*100:>+7.3f}% {row.get('24h_ret',0)*100:>+7.3f}% {row.get('1h_wr',0):>5.0%} {row.get('4h_wr',0):>5.0%}")


# ═══════════════════════════════════════════════════════════
# STEP 5 — ROBUSTNESS
# ═══════════════════════════════════════════════════════════

def robustness_check(df, events):
    """Split into halves and check consistency."""
    print(f"\n  ROBUSTNESS CHECK")
    print(f"  {'─' * 60}")

    n = len(df)
    mid = n // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]

    print(f"  First half: {first_half['timestamp'].min().date()} → {first_half['timestamp'].max().date()} ({len(first_half)} bars)")
    print(f"  Second half: {second_half['timestamp'].min().date()} → {second_half['timestamp'].max().date()} ({len(second_half)} bars)")

    key_events = ["high_funding_p90", "low_funding_p10", "oi_spike_1h", "long_heavy", "short_heavy"]

    print(f"\n  {'Event':<25s} {'Half':<8s} {'N':>5s} {'4h_ret':>8s} {'4h_wr':>6s}")
    print(f"  {'─'*25} {'─'*8} {'─'*5} {'─'*8} {'─'*6}")

    for event_name in key_events:
        if event_name not in events:
            continue

        for half_label, half_df in [("1st", first_half), ("2nd", second_half)]:
            mask = events[event_name].loc[half_df.index]
            n_events = mask.sum()
            if n_events < 2:
                continue

            fwd = half_df.loc[mask, "fwd_ret_4h"].dropna()
            if len(fwd) > 0:
                print(f"  {event_name:<25s} {half_label:<8s} {n_events:>5d} {fwd.mean()*100:>+7.3f}% {(fwd>0).mean():>5.0%}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print("=" * 80)
    print("  POSITIONING ALPHA STUDY — BTCUSDT Perpetual Futures")
    print("  Does positioning data contain predictive power?")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Load data
    print("\n  Loading data...")
    data = load_all_data()

    # Merge onto 1h timeline
    print("\n  Merging positioning data...")
    df = merge_positioning_data(data)
    print(f"  Merged: {len(df)} bars ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

    # Available columns
    pos_cols = [c for c in ["fundingRate", "sumOpenInterest", "longShortRatio",
                            "globalLongRatio", "buySellRatio"] if c in df.columns]
    print(f"  Positioning columns: {pos_cols}")

    # ═══════════════════════════════════════════════════════
    # STEP 1 — DEFINE EVENTS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 1 — EVENT DEFINITION")
    print(f"{'═' * 80}")

    events = define_events(df)
    print(f"\n  Defined {len(events)} event types:")
    for name, mask in events.items():
        n = mask.sum()
        print(f"    {name:<35s}: {n:>4d} events")

    # ═══════════════════════════════════════════════════════
    # STEP 2 — FORWARD ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 2 — FORWARD ANALYSIS")
    print(f"{'═' * 80}")

    all_results = []
    for event_name, mask in events.items():
        result = forward_analysis(df, events, event_name, mask)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)

        print(f"\n  {'Event':<35s} {'N':>5s} {'1h_mean':>8s} {'1h_wr':>6s} {'4h_mean':>8s} {'4h_wr':>6s} {'24h_mean':>9s} {'24h_wr':>6s} {'4h_p':>6s}")
        print(f"  {'─'*35} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*6} {'─'*9} {'─'*6} {'─'*6}")

        for _, r in results_df.iterrows():
            def fmt(key, pct=True):
                v = r.get(key, np.nan)
                if pd.isna(v):
                    return "   —   "
                return f"{v*100:>+7.3f}%" if pct else f"{v:>5.0%}"

            p4 = r.get("pval_4h", 1.0)
            sig = "✓" if p4 < 0.05 else ("~" if p4 < 0.10 else " ")

            print(f"  {r['event']:<35s} {r['N']:>5.0f} "
                  f"{fmt('mean_1h')} {fmt('winrate_1h', False)} "
                  f"{fmt('mean_4h')} {fmt('winrate_4h', False)} "
                  f"{fmt('mean_24h')} {fmt('winrate_24h', False)} "
                  f"{p4:>5.3f}{sig}")

    # ═══════════════════════════════════════════════════════
    # STEP 3 — DIRECTIONAL TEST
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 3 — DIRECTIONAL HYPOTHESIS TESTS")
    print(f"{'═' * 80}")

    directional_test(df, events)

    # ═══════════════════════════════════════════════════════
    # STEP 4 — INTENSITY ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 4 — INTENSITY ANALYSIS")
    print(f"{'═' * 80}")

    intensity_analysis(df, "fundingRate", "Funding Rate")
    if "funding_cum_24h" in df.columns:
        intensity_analysis(df, "funding_cum_24h", "Cumulative Funding (24h)")
    if "oi_pct_change_1h" in df.columns:
        intensity_analysis(df, "oi_pct_change_1h", "OI Change (1h)")
    if "longShortRatio" in df.columns:
        intensity_analysis(df, "longShortRatio", "Top Trader LS Ratio")
    if "buySellRatio" in df.columns:
        intensity_analysis(df, "buySellRatio", "Taker Buy/Sell Ratio")

    # ═══════════════════════════════════════════════════════
    # STEP 5 — ROBUSTNESS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  STEP 5 — ROBUSTNESS CHECK")
    print(f"{'═' * 80}")

    robustness_check(df, events)

    # ═══════════════════════════════════════════════════════
    # BASELINE COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  BASELINE — UNCONDITIONAL FORWARD RETURNS")
    print(f"{'═' * 80}")

    for h in [1, 4, 24]:
        col = f"fwd_ret_{h}h"
        if col in df.columns:
            rets = df[col].dropna()
            print(f"    {h}h: mean={rets.mean()*100:+.4f}%, std={rets.std()*100:.3f}%, winrate={(rets>0).mean():.1%}, N={len(rets)}")

    # ═══════════════════════════════════════════════════════
    # FINAL CLASSIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  FINAL CLASSIFICATION")
    print(f"{'═' * 80}")

    # Count significant events
    sig_events = 0
    marginal_events = 0
    for r in all_results:
        p4 = r.get("pval_4h", 1.0)
        if p4 < 0.05:
            sig_events += 1
        elif p4 < 0.10:
            marginal_events += 1

    print(f"\n  Events with p<0.05 (4h): {sig_events}/{len(all_results)}")
    print(f"  Events with p<0.10 (4h): {marginal_events}/{len(all_results)}")

    # Check directional hypotheses
    h1 = any(r["event"].startswith("H1") and r.get("mean_4h", 0) < 0 and r.get("pval_4h", 1) < 0.10 for r in all_results)
    h2 = any(r["event"].startswith("H2") and r.get("mean_4h", 0) > 0 and r.get("pval_4h", 1) < 0.10 for r in all_results)

    print(f"\n  H1 (high funding → neg): {'Supported' if h1 else 'Not supported'}")
    print(f"  H2 (low funding → pos): {'Supported' if h2 else 'Not supported'}")

    if sig_events >= 3:
        classification = "VALID EDGE"
        print(f"\n  ✅ {classification}")
        print(f"  Multiple events show statistically significant predictive power.")
    elif sig_events >= 1 or marginal_events >= 3:
        classification = "WEAK EDGE"
        print(f"\n  ⚠️  {classification}")
        print(f"  Some events show marginal significance. More data needed.")
    elif marginal_events >= 1:
        classification = "INCONCLUSIVE"
        print(f"\n  ❓ {classification}")
        print(f"  Marginal signals detected but not statistically robust.")
    else:
        classification = "NO EDGE"
        print(f"\n  ❌ {classification}")
        print(f"  No events show significant predictive power.")

    # Data limitations
    oi_days = len(data.get("oi", pd.DataFrame())) / 24 if "oi" in data else 0
    funding_days = len(data.get("funding", pd.DataFrame())) / 3 if "funding" in data else 0
    print(f"\n  DATA LIMITATIONS:")
    print(f"    Funding: {funding_days:.0f} days ({len(data.get('funding', []))} records)")
    print(f"    OI/LS/Taker: {oi_days:.0f} days ({len(data.get('oi', []))} records)")
    if oi_days < 60:
        print(f"    ⚠️  OI/LS data < 60 days — robustness check is unreliable")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
