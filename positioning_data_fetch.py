"""
POSITIONING DATA FETCHER — BTCUSDT Perpetual Futures
=====================================================
Fetches funding rates, open interest, LS ratios, and taker volume from Binance fapi.
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta


DATA_DIR = "data/features"
os.makedirs(DATA_DIR, exist_ok=True)
BASE = "https://fapi.binance.com"


def fetch_paginated(url, params, key_ts="timestamp", limit_field="limit", max_pages=200):
    """Generic paginated fetch."""
    all_data = []
    current = params.get("startTime", 0)
    end_time = params["endTime"]

    for _ in range(max_pages):
        p = {**params, "startTime": current, limit_field: params.get(limit_field, 1000)}
        try:
            resp = requests.get(url, params=p, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"      Error: {e}")
            break

        if not data:
            break

        all_data.extend(data)
        last_ts = int(data[-1][key_ts])
        if last_ts >= end_time:
            break
        current = last_ts + 1
        time.sleep(0.2)

    return all_data


def fetch_funding(symbol="BTCUSDT", start_ms=None, end_ms=None):
    """Fetch funding rate history (every 8h)."""
    if start_ms is None:
        start_ms = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    if end_ms is None:
        end_ms = int(datetime.now().timestamp() * 1000)

    url = f"{BASE}/fapi/v1/fundingRate"
    data = fetch_paginated(url, {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}, key_ts="fundingTime")

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "fundingRate"]].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def fetch_oi(symbol="BTCUSDT", period="1h", start_ms=None, end_ms=None):
    """Fetch open interest history."""
    if start_ms is None:
        start_ms = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    if end_ms is None:
        end_ms = int(datetime.now().timestamp() * 1000)

    url = f"{BASE}/futures/data/openInterestHist"
    data = fetch_paginated(url, {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms, "limit": 500})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    df = df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def fetch_ls_ratio(symbol="BTCUSDT", period="1h", start_ms=None, end_ms=None):
    """Fetch top trader long/short ratio."""
    if start_ms is None:
        start_ms = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    if end_ms is None:
        end_ms = int(datetime.now().timestamp() * 1000)

    url = f"{BASE}/futures/data/topLongShortPositionRatio"
    data = fetch_paginated(url, {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms, "limit": 500})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df["longAccount"] = df["longAccount"].astype(float)
    df["shortAccount"] = df["shortAccount"].astype(float)
    df = df[["timestamp", "longShortRatio", "longAccount", "shortAccount"]].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def fetch_global_ls(symbol="BTCUSDT", period="1h", start_ms=None, end_ms=None):
    """Fetch global long/short account ratio."""
    if start_ms is None:
        start_ms = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    if end_ms is None:
        end_ms = int(datetime.now().timestamp() * 1000)

    url = f"{BASE}/futures/data/globalLongShortAccountRatio"
    data = fetch_paginated(url, {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms, "limit": 500})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["globalLongRatio"] = df["longAccount"].astype(float)
    df["globalLSRatio"] = df["longShortRatio"].astype(float)
    df = df[["timestamp", "globalLongRatio", "globalLSRatio"]].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def fetch_taker_volume(symbol="BTCUSDT", period="1h", start_ms=None, end_ms=None):
    """Fetch taker buy/sell volume ratio."""
    if start_ms is None:
        start_ms = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    if end_ms is None:
        end_ms = int(datetime.now().timestamp() * 1000)

    url = f"{BASE}/futures/data/takerlongshortRatio"
    data = fetch_paginated(url, {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms, "limit": 500})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df["buySellRatio"] = df["buySellRatio"].astype(float)
    df["buyVol"] = df["buyVol"].astype(float)
    df["sellVol"] = df["sellVol"].astype(float)
    df = df[["timestamp", "buySellRatio", "buyVol", "sellVol"]].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def main():
    print("=" * 70)
    print("  POSITIONING DATA FETCHER — BTCUSDT Perpetual Futures")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    end_ms = int(datetime.now().timestamp() * 1000)
    start_180d = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)

    # 1. Funding rates (8h)
    print("\n  [1/5] Funding rates (8h, 180d)...")
    funding = fetch_funding("BTCUSDT", start_180d, end_ms)
    if len(funding) > 0:
        funding.to_csv(f"{DATA_DIR}/btcusdt_funding.csv", index=False)
        fr = funding["fundingRate"]
        print(f"    ✓ {len(funding)} records ({funding['timestamp'].min().date()} → {funding['timestamp'].max().date()})")
        print(f"    Mean={fr.mean()*100:.4f}%, Std={fr.std()*100:.4f}%, Range=[{fr.min()*100:.4f}%, {fr.max()*100:.4f}%]")

    # 2. Open interest (1h, 180d)
    print("\n  [2/5] Open interest (1h, 180d)...")
    oi = fetch_oi("BTCUSDT", "1h", start_180d, end_ms)
    if len(oi) > 0:
        oi.to_csv(f"{DATA_DIR}/btcusdt_oi_1h.csv", index=False)
        print(f"    ✓ {len(oi)} records ({oi['timestamp'].min().date()} → {oi['timestamp'].max().date()})")
        print(f"    Mean OI: {oi['sumOpenInterest'].mean():,.0f} BTC")

    # 3. Top trader LS ratio (1h, 180d)
    print("\n  [3/5] Top trader LS ratio (1h, 180d)...")
    ls = fetch_ls_ratio("BTCUSDT", "1h", start_180d, end_ms)
    if len(ls) > 0:
        ls.to_csv(f"{DATA_DIR}/btcusdt_ls_ratio_1h.csv", index=False)
        print(f"    ✓ {len(ls)} records ({ls['timestamp'].min().date()} → {ls['timestamp'].max().date()})")
        print(f"    Mean LS ratio: {ls['longShortRatio'].mean():.3f}")

    # 4. Global LS ratio (1h, 180d)
    print("\n  [4/5] Global LS ratio (1h, 180d)...")
    gls = fetch_global_ls("BTCUSDT", "1h", start_180d, end_ms)
    if len(gls) > 0:
        gls.to_csv(f"{DATA_DIR}/btcusdt_global_ls_1h.csv", index=False)
        print(f"    ✓ {len(gls)} records ({gls['timestamp'].min().date()} → {gls['timestamp'].max().date()})")
        print(f"    Mean global long%: {gls['globalLongRatio'].mean():.1%}")

    # 5. Taker buy/sell volume (1h, 180d)
    print("\n  [5/5] Taker buy/sell volume (1h, 180d)...")
    taker = fetch_taker_volume("BTCUSDT", "1h", start_180d, end_ms)
    if len(taker) > 0:
        taker.to_csv(f"{DATA_DIR}/btcusdt_taker_1h.csv", index=False)
        print(f"    ✓ {len(taker)} records ({taker['timestamp'].min().date()} → {taker['timestamp'].max().date()})")
        print(f"    Mean buy/sell ratio: {taker['buySellRatio'].mean():.3f}")

    # Summary
    print(f"\n  {'='*70}")
    print(f"  DATA FETCH COMPLETE")
    for f in ["btcusdt_funding.csv", "btcusdt_oi_1h.csv", "btcusdt_ls_ratio_1h.csv",
              "btcusdt_global_ls_1h.csv", "btcusdt_taker_1h.csv"]:
        path = f"{DATA_DIR}/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            lines = sum(1 for _ in open(path)) - 1
            print(f"    {f}: {lines:,} rows, {size/1024:.0f} KB")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
