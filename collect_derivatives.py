#!/usr/bin/env python3
"""
BTCUSDT Derivatives Data Collector
===================================
Append-only hourly collection of:
  - OHLCV (1m candles, resampled to 1h)
  - Open Interest
  - Taker Buy/Sell Ratio
  - Global Long/Short Ratio
  - Funding Rate

Storage: CSV, append-only, UTC timestamps
Run: hourly via cron or systemd timer
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

SYMBOL = "BTCUSDT"
BASE_URL = "https://api.binance.com"
FUTURES_URL = "https://fapi.binance.com"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "collected")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "btcusdt_hourly_derivatives.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "collector.log")

RETRY_MAX = 3
RETRY_DELAY = 5  # seconds

# =========================================================
# LOGGING
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("collector")

# =========================================================
# API CALLS
# =========================================================

def api_get(url, params=None, retries=RETRY_MAX):
    """GET with retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"API error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    log.error(f"API failed after {retries} attempts: {url}")
    return None


def fetch_klines_1m(start_ms, end_ms):
    """Fetch 1m klines. Returns list of raw kline data."""
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": SYMBOL,
        "interval": "1m",
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": 1000,
    }
    data = api_get(url, params)
    return data if data else []


def fetch_open_interest(period="1h", limit=1, start_ms=None, end_ms=None):
    """Fetch open interest history."""
    url = f"{FUTURES_URL}/futures/data/openInterestHist"
    params = {
        "symbol": SYMBOL,
        "period": period,
        "limit": limit,
    }
    if start_ms:
        params["startTime"] = int(start_ms)
    if end_ms:
        params["endTime"] = int(end_ms)
    data = api_get(url, params)
    return data if data else []


def fetch_taker_ratio(period="1h", limit=1, start_ms=None, end_ms=None):
    """Fetch taker buy/sell ratio."""
    url = f"{FUTURES_URL}/futures/data/takerlongshortRatio"
    params = {
        "symbol": SYMBOL,
        "period": period,
        "limit": limit,
    }
    if start_ms:
        params["startTime"] = int(start_ms)
    if end_ms:
        params["endTime"] = int(end_ms)
    data = api_get(url, params)
    return data if data else []


def fetch_ls_ratio(period="1h", limit=1, start_ms=None, end_ms=None):
    """Fetch global long/short ratio."""
    url = f"{FUTURES_URL}/futures/data/globalLongShortAccountRatio"
    params = {
        "symbol": SYMBOL,
        "period": period,
        "limit": limit,
    }
    if start_ms:
        params["startTime"] = int(start_ms)
    if end_ms:
        params["endTime"] = int(end_ms)
    data = api_get(url, params)
    return data if data else []


def fetch_funding_rate(limit=1, start_ms=None, end_ms=None):
    """Fetch funding rate history."""
    url = f"{FUTURES_URL}/fapi/v1/fundingRate"
    params = {
        "symbol": SYMBOL,
        "limit": limit,
    }
    if start_ms:
        params["startTime"] = int(start_ms)
    if end_ms:
        params["endTime"] = int(end_ms)
    data = api_get(url, params)
    return data if data else []


# =========================================================
# COLLECTION
# =========================================================

def collect_hour(timestamp_hour):
    """
    Collect all data for a specific hour.
    timestamp_hour: datetime in UTC, floored to hour.

    Returns dict with all fields, or None on failure.
    """
    ts_ms = int(timestamp_hour.timestamp() * 1000)
    ts_end_ms = ts_ms + 3600_000 - 1  # end of hour

    result = {
        "timestamp": timestamp_hour.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. OHLCV from 1m klines → resample to 1h
    klines = fetch_klines_1m(ts_ms, ts_end_ms)
    if klines and len(klines) > 0:
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        result["open"] = opens[0]
        result["high"] = max(highs)
        result["low"] = min(lows)
        result["close"] = closes[-1]
        result["volume"] = sum(volumes)
        result["num_candles"] = len(klines)
    else:
        log.warning(f"No kline data for {result['timestamp']}")
        result["open"] = np.nan
        result["high"] = np.nan
        result["low"] = np.nan
        result["close"] = np.nan
        result["volume"] = np.nan
        result["num_candles"] = 0

    # 2. Open Interest (fetch with time range)
    oi_data = fetch_open_interest("1h", 1, start_ms=ts_ms, end_ms=ts_end_ms)
    if oi_data and len(oi_data) > 0:
        latest = oi_data[-1]
        result["open_interest"] = float(latest.get("sumOpenInterest", 0))
        result["oi_value"] = float(latest.get("sumOpenInterestValue", 0))
    else:
        log.warning(f"No OI data for {result['timestamp']}")
        result["open_interest"] = np.nan
        result["oi_value"] = np.nan

    # 3. Taker Buy/Sell Ratio (fetch with time range)
    taker_data = fetch_taker_ratio("1h", 1, start_ms=ts_ms, end_ms=ts_end_ms)
    if taker_data and len(taker_data) > 0:
        latest = taker_data[-1]
        result["taker_buy_vol"] = float(latest.get("buyVol", 0))
        result["taker_sell_vol"] = float(latest.get("sellVol", 0))
        result["taker_ratio"] = float(latest.get("buySellRatio", 0))
    else:
        log.warning(f"No taker data for {result['timestamp']}")
        result["taker_buy_vol"] = np.nan
        result["taker_sell_vol"] = np.nan
        result["taker_ratio"] = np.nan

    # 4. Long/Short Ratio (fetch with time range)
    ls_data = fetch_ls_ratio("1h", 1, start_ms=ts_ms, end_ms=ts_end_ms)
    if ls_data and len(ls_data) > 0:
        latest = ls_data[-1]
        result["ls_long_ratio"] = float(latest.get("longAccount", 0))
        result["ls_short_ratio"] = float(latest.get("shortAccount", 0))
        result["ls_ratio"] = float(latest.get("longShortRatio", 0))
    else:
        log.warning(f"No LS ratio data for {result['timestamp']}")
        result["ls_long_ratio"] = np.nan
        result["ls_short_ratio"] = np.nan
        result["ls_ratio"] = np.nan

    # 5. Funding Rate (get most recent funding before this hour)
    #    Funding is every 8h (00:00, 08:00, 16:00 UTC)
    #    Fetch last 3 and pick the one closest to but before this hour
    funding_data = fetch_funding_rate(3, end_ms=ts_end_ms)
    if funding_data and len(funding_data) > 0:
        # Find most recent funding before or at this hour
        valid = [f for f in funding_data if int(f.get("fundingTime", 0)) <= ts_end_ms]
        if valid:
            latest = valid[-1]
            result["funding_rate"] = float(latest.get("fundingRate", 0))
            result["funding_time"] = latest.get("fundingTime", 0)
        else:
            result["funding_rate"] = np.nan
            result["funding_time"] = np.nan
    else:
        log.warning(f"No funding data for {result['timestamp']}")
        result["funding_rate"] = np.nan
        result["funding_time"] = np.nan

    return result


# =========================================================
# STORAGE
# =========================================================

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "num_candles",
    "open_interest", "oi_value",
    "taker_buy_vol", "taker_sell_vol", "taker_ratio",
    "ls_long_ratio", "ls_short_ratio", "ls_ratio",
    "funding_rate", "funding_time",
]


def validate_row(row):
    """
    Validate a row before appending.
    Returns (is_valid, error_message).
    """
    ts = row.get("timestamp", "")
    if not ts or ts == "NaT":
        return False, "Missing or invalid timestamp"

    # Must have price data
    if pd.isna(row.get("close")):
        return False, f"{ts}: close price is NaN"

    # Sanity: close must be positive
    close = row.get("close", 0)
    if close <= 0:
        return False, f"{ts}: close={close} is non-positive"

    # Timestamp must be aligned to exact hour
    try:
        dt = pd.Timestamp(ts)
        if dt.minute != 0 or dt.second != 0:
            return False, f"{ts}: not aligned to hour"
    except Exception:
        return False, f"{ts}: unparseable timestamp"

    return True, ""


def load_existing():
    """Load existing CSV, return set of timestamps already collected."""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, usecols=["timestamp"])
        return set(df["timestamp"].tolist())
    return set()


def append_row(row):
    """Append a single row to CSV. Create file with headers if needed."""
    # Validate before writing
    is_valid, error = validate_row(row)
    if not is_valid:
        log.error(f"VALIDATION FAILED — not appending: {error}")
        return False

    file_exists = os.path.exists(OUTPUT_FILE)
    df = pd.DataFrame([row], columns=COLUMNS)

    if not file_exists:
        df.to_csv(OUTPUT_FILE, index=False)
        log.info(f"Created {OUTPUT_FILE}")
    else:
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    return True


# =========================================================
# MAIN
# =========================================================

def run_collection(target_hour=None):
    """
    Collect data for target_hour (UTC, floored to hour).
    If None, collects the previous completed hour.
    """
    if target_hour is None:
        # Previous completed hour
        now = datetime.now(timezone.utc)
        target_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

    ts_str = target_hour.strftime("%Y-%m-%d %H:%M:%S")

    # Check if already collected
    existing = load_existing()
    if ts_str in existing:
        log.info(f"Already collected: {ts_str}")
        return True

    log.info(f"Collecting: {ts_str}")
    row = collect_hour(target_hour)

    # Validate: at least price data must exist
    if pd.isna(row.get("close")):
        log.error(f"No price data for {ts_str} — skipping (API failure)")
        return False

    if not append_row(row):
        log.error(f"Validation failed for {ts_str} — row not saved")
        return False

    log.info(f"Saved: {ts_str} | OI={row.get('open_interest', 'N/A')} | taker={row.get('taker_ratio', 'N/A')} | funding={row.get('funding_rate', 'N/A')}")
    return True


def backfill(start_date, end_date=None):
    """
    Backfill historical data hour by hour.
    start_date: datetime UTC
    end_date: datetime UTC (default = now)
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)

    end_date = end_date.replace(minute=0, second=0, microsecond=0)
    current = start_date.replace(minute=0, second=0, microsecond=0)

    existing = load_existing()
    total_hours = int((end_date - current).total_seconds() / 3600)
    collected = 0
    skipped = 0
    failed = 0

    log.info(f"Backfill: {current} → {end_date} ({total_hours} hours)")

    while current < end_date:
        ts_str = current.strftime("%Y-%m-%d %H:%M:%S")

        if ts_str in existing:
            skipped += 1
            current += timedelta(hours=1)
            continue

        success = run_collection(current)
        if success:
            collected += 1
        else:
            failed += 1
            log.warning(f"GAP DETECTED: {ts_str} — collection failed, will retry on next backfill")

        current += timedelta(hours=1)

        # Rate limit: ~5 API calls per hour collected, be gentle
        time.sleep(0.5)

        if (collected + failed) % 24 == 0:
            log.info(f"Progress: {collected+failed+skipped}/{total_hours} ({collected} new, {skipped} existing, {failed} failed)")

    log.info(f"Backfill complete: {collected} collected, {skipped} existing, {failed} failed")
    if failed > 0:
        log.warning(f"WARNING: {failed} hours failed. Run backfill again to fill gaps.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTCUSDT Derivatives Data Collector")
    parser.add_argument("--backfill", action="store_true", help="Backfill from Mar 31 2026")
    parser.add_argument("--backfill-start", type=str, help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--hour", type=str, help="Collect specific hour (YYYY-MM-DD HH:00:00)")
    args = parser.parse_args()

    if args.backfill or args.backfill_start:
        start = datetime(2026, 3, 31, tzinfo=timezone.utc)
        if args.backfill_start:
            parts = args.backfill_start.split("-")
            start = datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=timezone.utc)
        backfill(start)

    elif args.hour:
        target = datetime.strptime(args.hour, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        run_collection(target)

    else:
        # Default: collect previous hour
        run_collection()


if __name__ == "__main__":
    main()
