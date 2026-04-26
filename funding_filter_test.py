"""
FUNDING FILTER TEST — Does removing HIGH funding trades improve decisions?
==========================================================================
Base event: large move bar (top 10% absolute 5m returns)
Naive strategy: continuation (follow the move direction), hold 60m
Filter: skip trade if funding is HIGH
Question: does the filter improve the strategy?
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "features")

# =========================================================
# CONFIG — same as funding_modifier_test, no tuning
# =========================================================

MOVE_PERCENTILE = 90
FUNDING_HIGH_PCTL = 80

HOLD_BARS = 60  # minutes

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
# STEP 1 — BASELINE DECISION
# =========================================================

def define_events(price):
    """Same event definition: top 10% absolute 5m moves."""
    price = price.copy()
    price["ret_5m"] = price["price"].pct_change(5).abs()
    price["ret_5m_signed"] = price["price"].pct_change(5)
    threshold = price["ret_5m"].rolling(5000, min_periods=500).quantile(MOVE_PERCENTILE / 100).shift(1)
    price["is_event"] = price["ret_5m"] > threshold
    price["move_direction"] = np.sign(price["ret_5m_signed"])

    events = price[price["is_event"]][["timestamp", "price", "ret_5m", "ret_5m_signed", "move_direction"]].copy()
    events = events.rename(columns={"price": "event_price"}).reset_index(drop=True)
    return events


def label_funding(events, funding):
    events = events.copy()
    events = pd.merge_asof(
        events.sort_values("timestamp"),
        funding[["timestamp", "fundingRate"]].sort_values("timestamp"),
        on="timestamp", direction="backward"
    )
    funding_sorted = funding.sort_values("timestamp").copy()
    funding_sorted["p80"] = funding_sorted["fundingRate"].rolling(30, min_periods=10).quantile(0.80).shift(1)
    events = pd.merge_asof(
        events.sort_values("timestamp"),
        funding_sorted[["timestamp", "p80"]].sort_values("timestamp"),
        on="timestamp", direction="backward"
    )
    events["is_high_funding"] = events["fundingRate"] > events["p80"]
    return events


def compute_trade_outcomes(events, price):
    """
    Naive continuation strategy:
    - direction = move direction (follow the large move)
    - entry = next bar open (event bar + 1)
    - exit = event bar + 1 + HOLD_BARS
    - return = direction * (exit_price - entry_price) / entry_price
    """
    price_values = price["price"].values
    n_bars = len(price_values)
    results = []

    for _, event in events.iterrows():
        event_time = event["timestamp"]
        direction = event["move_direction"]

        # Find event bar index
        mask = price["timestamp"] >= event_time
        if mask.sum() == 0:
            continue
        idx = price.index[mask][0]

        entry_idx = idx + 1  # next bar open
        exit_idx = entry_idx + HOLD_BARS

        if exit_idx >= n_bars or entry_idx >= n_bars:
            continue

        entry_price = price_values[entry_idx]
        exit_price = price_values[exit_idx]

        if entry_price == 0:
            continue

        # Continuation trade: follow the move direction
        trade_return = direction * (exit_price - entry_price) / entry_price

        # MFE/MAE during hold period
        path = price_values[entry_idx:exit_idx + 1]
        path_pct = (path - entry_price) / entry_price

        if direction >= 0:
            mfe = np.max(path_pct)
            mae = np.max(-path_pct)
        else:
            mfe = np.max(-path_pct)
            mae = np.max(path_pct)

        # Running P&L for drawdown calculation (need full path)
        # For simplicity, use trade_return and mae as proxy
        # Max drawdown during trade = max adverse from peak favorable
        if direction >= 0:
            running = path_pct  # cumulative return path
        else:
            running = -path_pct

        # Drawdown from peak
        peak = np.maximum.accumulate(running)
        drawdown_from_peak = running - peak
        max_drawdown = np.min(drawdown_from_peak)

        results.append({
            "timestamp": event["timestamp"],
            "direction": direction,
            "is_high_funding": event["is_high_funding"],
            "funding_rate": event["fundingRate"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "trade_return": trade_return,
            "mfe": mfe,
            "mae": mae,
            "max_drawdown": max_drawdown,
            "won": trade_return > 0,
        })

    return pd.DataFrame(results)


# =========================================================
# STEP 2 & 3 — FILTER AND COMPARE
# =========================================================

def compare_baseline_vs_filtered(df, label="ALL PERIODS"):
    baseline = df.copy()
    filtered = df[~df["is_high_funding"]].copy()

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30} {'BASELINE':>15} {'FILTERED':>15} {'DELTA':>15}")
    print(f"  {'─'*75}")

    n_base = len(baseline)
    n_filt = len(filtered)
    n_removed = n_base - n_filt
    print(f"  {'Trades':<30} {n_base:>15,} {n_filt:>15,} {n_removed:>+15,} ({n_removed/n_base*100:.1f}%)")

    # Mean return
    base_ret = baseline["trade_return"]
    filt_ret = filtered["trade_return"]
    print(f"  {'Mean return':<30} {base_ret.mean()*100:>+14.4f}% {filt_ret.mean()*100:>+14.4f}% {(filt_ret.mean()-base_ret.mean())*100:>+14.4f}%")

    # Median return
    print(f"  {'Median return':<30} {base_ret.median()*100:>+14.4f}% {filt_ret.median()*100:>+14.4f}% {(filt_ret.median()-base_ret.median())*100:>+14.4f}%")

    # Win rate
    base_wr = baseline["won"].mean() * 100
    filt_wr = filtered["won"].mean() * 100
    print(f"  {'Win rate':<30} {base_wr:>14.1f}% {filt_wr:>14.1f}% {filt_wr-base_wr:>+14.1f}%")

    # Mean win / mean loss
    base_wins = baseline[baseline["won"]]["trade_return"]
    base_losses = baseline[~baseline["won"]]["trade_return"]
    filt_wins = filtered[filtered["won"]]["trade_return"]
    filt_losses = filtered[~filtered["won"]]["trade_return"]
    base_pw = base_wins.mean() if len(base_wins) > 0 else 0
    base_pl = base_losses.mean() if len(base_losses) > 0 else 0
    filt_pw = filt_wins.mean() if len(filt_wins) > 0 else 0
    filt_pl = filt_losses.mean() if len(filt_losses) > 0 else 0
    print(f"  {'Mean win':<30} {base_pw*100:>+14.4f}% {filt_pw*100:>+14.4f}% {(filt_pw-base_pw)*100:>+14.4f}%")
    print(f"  {'Mean loss':<30} {base_pl*100:>+14.4f}% {filt_pl*100:>+14.4f}% {(filt_pl-base_pl)*100:>+14.4f}%")

    # Profit factor
    base_pf = abs(base_wins.sum() / base_losses.sum()) if base_losses.sum() != 0 else np.nan
    filt_pf = abs(filt_wins.sum() / filt_losses.sum()) if filt_losses.sum() != 0 else np.nan
    print(f"  {'Profit factor':<30} {base_pf:>15.3f} {filt_pf:>15.3f} {filt_pf-base_pf:>+15.3f}")

    # Expectancy per trade
    base_exp = base_ret.mean()
    filt_exp = filt_ret.mean()
    print(f"  {'Expectancy / trade':<30} {base_exp*100:>+14.4f}% {filt_exp*100:>+14.4f}% {(filt_exp-base_exp)*100:>+14.4f}%")

    # Cumulative return
    base_cum = base_ret.sum() * 100
    filt_cum = filt_ret.sum() * 100
    print(f"  {'Cumulative return':<30} {base_cum:>+14.2f}% {filt_cum:>+14.2f}% {filt_cum-base_cum:>+14.2f}%")

    # Max drawdown (worst single trade)
    base_dd = baseline["max_drawdown"].min() * 100
    filt_dd = filtered["max_drawdown"].min() * 100
    print(f"  {'Max intratrade DD':<30} {base_dd:>14.2f}% {filt_dd:>14.2f}% {filt_dd-base_dd:>+14.2f}%")

    # Mean MAE
    base_mae = baseline["mae"].mean() * 100
    filt_mae = filtered["mae"].mean() * 100
    print(f"  {'Mean MAE':<30} {base_mae:>14.3f}% {filt_mae:>14.3f}% {filt_mae-base_mae:>+14.3f}%")

    # Sharpe-like ratio (mean / std)
    base_sharpe = base_ret.mean() / base_ret.std() if base_ret.std() > 0 else 0
    filt_sharpe = filt_ret.mean() / filt_ret.std() if filt_ret.std() > 0 else 0
    print(f"  {'Sharpe (mean/std)':<30} {base_sharpe:>15.4f} {filt_sharpe:>15.4f} {filt_sharpe-base_sharpe:>+15.4f}")

    # Return distribution
    print(f"\n  Return distribution:")
    for pct_label, lo, hi in [
        ("  < -1%", -np.inf, -0.01),
        ("  -1% to -0.5%", -0.01, -0.005),
        ("  -0.5% to 0%", -0.005, 0),
        ("  0% to +0.5%", 0, 0.005),
        ("  +0.5% to +1%", 0.005, 0.01),
        ("  > +1%", 0.01, np.inf),
    ]:
        base_n = ((base_ret >= lo) & (base_ret < hi)).sum()
        filt_n = ((filt_ret >= lo) & (filt_ret < hi)).sum()
        base_pct = base_n / n_base * 100
        filt_pct = filt_n / n_filt * 100
        print(f"    {pct_label:<20}  base {base_n:>5} ({base_pct:>5.1f}%)  filt {filt_n:>5} ({filt_pct:>5.1f}%)")

    return {
        "n_base": n_base, "n_filt": n_filt, "n_removed": n_removed,
        "base_mean": base_ret.mean(), "filt_mean": filt_ret.mean(),
        "base_wr": base_wr, "filt_wr": filt_wr,
        "base_pf": base_pf, "filt_pf": filt_pf,
    }


# =========================================================
# STEP 4 — ROBUSTNESS
# =========================================================

def robustness(df):
    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS — TRAIN/TEST SPLIT")
    print(f"{'='*70}")

    train = df[df["timestamp"] < TRAIN_CUTOFF]
    test = df[df["timestamp"] >= TRAIN_CUTOFF]

    print(f"\n  {'Period':<10} {'':>8} {'Trades':>8} {'Mean ret':>10} {'Win rate':>10} {'PF':>8} {'Sharpe':>8}")
    print(f"  {'─'*62}")

    for period_name, period_df in [("TRAIN", train), ("TEST", test)]:
        base = period_df
        filt = period_df[~period_df["is_high_funding"]]

        for label, sub in [("base", base), ("filt", filt)]:
            n = len(sub)
            ret = sub["trade_return"]
            wr = sub["won"].mean() * 100
            wins = sub[sub["won"]]["trade_return"]
            losses = sub[~sub["won"]]["trade_return"]
            pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0
            sharpe = ret.mean() / ret.std() if ret.std() > 0 else 0
            print(f"  {period_name:<10} {label:>8} {n:>8,} {ret.mean()*100:>+9.4f}% {wr:>9.1f}% {pf:>8.3f} {sharpe:>8.4f}")

        # Delta
        base_ret = base["trade_return"]
        filt_ret = filt["trade_return"]
        base_wr = base["won"].mean() * 100
        filt_wr = filt["won"].mean() * 100
        base_wins = base[base["won"]]["trade_return"]
        base_losses = base[~base["won"]]["trade_return"]
        filt_wins = filt[filt["won"]]["trade_return"]
        filt_losses = filt[~filt["won"]]["trade_return"]
        base_pf = abs(base_wins.sum() / base_losses.sum()) if base_losses.sum() != 0 else 0
        filt_pf = abs(filt_wins.sum() / filt_losses.sum()) if filt_losses.sum() != 0 else 0
        base_sharpe = base_ret.mean() / base_ret.std() if base_ret.std() > 0 else 0
        filt_sharpe = filt_ret.mean() / filt_ret.std() if filt_ret.std() > 0 else 0

        d_mean = (filt_ret.mean() - base_ret.mean()) * 100
        d_wr = filt_wr - base_wr
        d_pf = filt_pf - base_pf
        d_sharpe = filt_sharpe - base_sharpe
        print(f"  {period_name:<10} {'delta':>8} {'':>8} {d_mean:>+9.4f}% {d_wr:>+9.1f}% {d_pf:>+8.3f} {d_sharpe:>+8.4f}")
        print()


# =========================================================
# STEP 5 — DECISION
# =========================================================

def render_decision(df):
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    base = df
    filt = df[~df["is_high_funding"]]

    base_mean = base["trade_return"].mean()
    filt_mean = filt["trade_return"].mean()
    improvement = filt_mean - base_mean

    base_wr = base["won"].mean() * 100
    filt_wr = filt["won"].mean() * 100

    # Train/test consistency
    train = df[df["timestamp"] < TRAIN_CUTOFF]
    test = df[df["timestamp"] >= TRAIN_CUTOFF]

    train_base = train["trade_return"].mean()
    train_filt = train[~train["is_high_funding"]]["trade_return"].mean()
    test_base = test["trade_return"].mean()
    test_filt = test[~test["is_high_funding"]]["trade_return"].mean()

    train_delta = train_filt - train_base
    test_delta = test_filt - test_base

    print(f"\n  Does removing HIGH funding trades improve the strategy?")
    print(f"\n  Overall improvement:   {improvement*100:+.4f}% per trade")
    print(f"  Win rate change:       {filt_wr - base_wr:+.1f}%")
    print(f"  Trades removed:        {base['is_high_funding'].sum()} / {len(base)} ({base['is_high_funding'].mean()*100:.1f}%)")

    print(f"\n  Train delta:  {train_delta*100:+.4f}%")
    print(f"  Test delta:   {test_delta*100:+.4f}%")

    # Simple significance check
    base_ret = base["trade_return"]
    filt_ret = filt["trade_return"]
    pooled_std = np.sqrt((base_ret.std()**2 + filt_ret.std()**2) / 2)
    se = pooled_std * np.sqrt(1/len(base_ret) + 1/len(filt_ret))
    t_stat = improvement / se if se > 0 else 0

    print(f"  t-stat:       {t_stat:+.2f}  {'***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.65 else 'ns'}")

    # Final answer
    consistent = (train_delta > 0 and test_delta > 0) or (train_delta < 0 and test_delta < 0)
    significant = abs(t_stat) > 1.96

    if significant and consistent and improvement > 0:
        answer = "YES"
        reason = "Filter improves returns, is statistically significant, and consistent across train/test."
    elif significant and improvement > 0 and not consistent:
        answer = "INCONCLUSIVE"
        reason = "Filter improves returns overall but effect is not consistent across train/test."
    elif significant and improvement < 0:
        answer = "NO"
        reason = "Filter WORSENS returns. HIGH funding trades are not worse than average."
    else:
        answer = "NO"
        reason = "No significant difference. Filter does not meaningfully change outcomes."

    print(f"\n  ANSWER: {answer}")
    print(f"  {reason}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  FUNDING FILTER TEST — Does removing HIGH funding trades improve decisions?")
    print("=" * 70)

    print("\nLoading data...")
    price, funding = load_data()
    print(f"  1m candles: {len(price):,}")
    print(f"  Funding:    {len(funding):,}")

    print("\nSTEP 1 — Defining events + naive continuation trades...")
    events = define_events(price)
    print(f"  Events: {len(events):,}")

    print("\nSTEP 2 — Labeling funding state...")
    events = label_funding(events, funding)
    n_high = events["is_high_funding"].sum()
    print(f"  HIGH funding events: {n_high} ({n_high/len(events)*100:.1f}%)")

    print("\nSTEP 3 — Computing trade outcomes...")
    trades = compute_trade_outcomes(events, price)
    print(f"  Trades computed: {len(trades):,}")

    # Step 2 & 3 — compare
    all_stats = compare_baseline_vs_filtered(trades, "ALL PERIODS")

    # Step 4 — robustness
    robustness(trades)

    # Step 5 — verdict
    render_decision(trades)

    # Save
    out_path = os.path.join(DATA_DIR, "funding_filter_test_results.csv")
    trades.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
