"""
Fetch ALL 2026 1m data from Binance for multiple symbols.
Jan 1 2026 → present.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "1000PEPEUSDT"]
INTERVAL = "1m"
LIMIT = 1000
OUTPUT_DIR = "data/features"

def fetch_binance_klines(symbol, interval, start_ms, end_ms, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_symbol(symbol):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_1m.csv")

    start_ms = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    total_days = (now_ms - start_ms) / (24 * 60 * 60 * 1000)
    print(f"\nFetching {symbol} {INTERVAL}: {total_days:.0f} days (Jan 1 2026 → now)")

    all_rows = []
    cursor = start_ms

    while cursor < now_ms:
        batch = fetch_binance_klines(symbol, INTERVAL, cursor, now_ms, LIMIT)
        if not batch:
            break

        all_rows.extend(batch)
        cursor = batch[-1][0] + 60_000

        elapsed = len(all_rows)
        print(f"  {elapsed} candles... ({elapsed / (total_days * 1440) * 100:.0f}%)", end="\r")
        time.sleep(0.08)

    print(f"\nTotal candles: {len(all_rows)}")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df["price"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["trade_count"] = df["trades"].astype(float)
    df["delta"] = df["taker_buy_base"].astype(float) - (df["volume"].astype(float) - df["taker_buy_base"].astype(float))

    out = df[["timestamp", "price", "volume", "delta", "trade_count"]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    out.to_csv(output_file, index=False)
    print(f"[SAVED] {output_file} ({len(out)} rows)")
    print(f"  Range: {out['timestamp'].iloc[0]} to {out['timestamp'].iloc[-1]}")
    print(f"  Price range: ${out['price'].min():.2f} - ${out['price'].max():.2f}")
    return output_file

def main():
    for symbol in SYMBOLS:
        fetch_symbol(symbol)

if __name__ == "__main__":
    main()
