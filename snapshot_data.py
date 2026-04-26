#!/usr/bin/env python3
"""
BTCUSDT Derivatives Data Snapshot
===================================
Creates daily snapshot copies of the data file.

Storage: data/snapshots/btcusdt_hourly_YYYYMMDD.csv
Retention: keeps last 30 snapshots (configurable)

Purpose: prevent data corruption loss, allow rollback.

Usage:
  python3 snapshot_data.py              # create today's snapshot
  python3 snapshot_data.py --clean      # also remove old snapshots
  python3 snapshot_data.py --retention 60  # keep 60 days
"""

import os
import sys
import shutil
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "data", "collected", "btcusdt_hourly_derivatives.csv")
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "data", "snapshots")
DEFAULT_RETENTION = 30  # days


def create_snapshot():
    """Create a timestamped snapshot of the data file."""
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return False

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"btcusdt_hourly_{today}.csv")

    # Don't overwrite if today's snapshot already exists
    if os.path.exists(snapshot_path):
        print(f"Snapshot already exists for today: {snapshot_path}")
        return True

    try:
        shutil.copy2(DATA_FILE, snapshot_path)
        size = os.path.getsize(snapshot_path)
        print(f"Snapshot created: {snapshot_path} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"ERROR creating snapshot: {e}")
        return False


def clean_old_snapshots(retention_days=DEFAULT_RETENTION):
    """Remove snapshots older than retention_days."""
    if not os.path.exists(SNAPSHOT_DIR):
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed = 0

    for f in sorted(os.listdir(SNAPSHOT_DIR)):
        if not f.startswith("btcusdt_hourly_") or not f.endswith(".csv"):
            continue

        # Extract date from filename: btcusdt_hourly_YYYYMMDD.csv
        try:
            date_str = f.replace("btcusdt_hourly_", "").replace(".csv", "")
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if file_date < cutoff:
            filepath = os.path.join(SNAPSHOT_DIR, f)
            os.remove(filepath)
            print(f"Removed old snapshot: {f}")
            removed += 1

    return removed


def list_snapshots():
    """List all existing snapshots."""
    if not os.path.exists(SNAPSHOT_DIR):
        print("No snapshots directory found.")
        return

    snapshots = sorted([
        f for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("btcusdt_hourly_") and f.endswith(".csv")
    ])

    if not snapshots:
        print("No snapshots found.")
        return

    print(f"\nSnapshots ({len(snapshots)}):")
    for s in snapshots:
        path = os.path.join(SNAPSHOT_DIR, s)
        size = os.path.getsize(path)
        print(f"  {s}  ({size:,} bytes)")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTCUSDT Derivatives Data Snapshot")
    parser.add_argument("--clean", action="store_true", help="Remove old snapshots")
    parser.add_argument("--retention", type=int, default=DEFAULT_RETENTION, help="Days to keep")
    parser.add_argument("--list", action="store_true", help="List existing snapshots")
    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return

    success = create_snapshot()

    if args.clean:
        removed = clean_old_snapshots(args.retention)
        if removed > 0:
            print(f"Cleaned {removed} old snapshot(s) (retention: {args.retention} days)")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
