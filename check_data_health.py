#!/usr/bin/env python3
"""
BTCUSDT Derivatives Data Health Check
======================================
Validates data pipeline integrity for future research reliability.

Checks:
1. Dataset integrity (row count, timestamps)
2. Time continuity (gaps)
3. Duplicates
4. Data quality (NaN, unrealistic values)
5. Freshness (last timestamp vs now)
6. Column validation
7. Summary report

Classification:
  OK      — all checks pass
  WARNING — non-critical issues (minor gaps, staleness)
  ERROR   — critical issues (major gaps, missing columns, corruption)

Usage:
  python3 check_data_health.py              # print report
  python3 check_data_health.py --log        # also append to health.log
  python3 check_data_health.py --json       # output JSON
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "data", "collected", "btcusdt_hourly_derivatives.csv")
HEALTH_LOG = os.path.join(SCRIPT_DIR, "data", "collected", "health.log")
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "data", "snapshots")

STALENESS_THRESHOLD_HOURS = 2  # warn if last timestamp older than this

REQUIRED_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "open_interest", "taker_buy_vol", "taker_sell_vol",
    "ls_ratio", "funding_rate",
]

REALISTIC_RANGES = {
    "close":           (1000, 500000),    # BTC price
    "open_interest":   (10000, 500000),   # OI contracts
    "taker_ratio":     (0.1, 10.0),       # buy/sell ratio
    "ls_ratio":        (0.1, 10.0),       # long/short ratio
    "funding_rate":    (-0.05, 0.05),     # funding rate
    "volume":          (0, 100000),        # hourly volume
}


# =========================================================
# CHECKS
# =========================================================

class HealthReport:
    def __init__(self):
        self.status = "OK"
        self.checks = []
        self.issues = []

    def add_check(self, name, status, detail=""):
        self.checks.append({"name": name, "status": status, "detail": detail})
        if status == "ERROR":
            self.status = "ERROR"
        elif status == "WARNING" and self.status != "ERROR":
            self.status = "WARNING"

    def add_issue(self, severity, message):
        self.issues.append({"severity": severity, "message": message})
        if severity == "ERROR":
            self.status = "ERROR"
        elif severity == "WARNING" and self.status != "ERROR":
            self.status = "WARNING"

    def to_dict(self):
        return {
            "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": self.checks,
            "issues": self.issues,
        }


def check_file_exists(report):
    """Check if data file exists and is non-empty."""
    if not os.path.exists(DATA_FILE):
        report.add_check("file_exists", "ERROR", f"File not found: {DATA_FILE}")
        return None

    size = os.path.getsize(DATA_FILE)
    if size == 0:
        report.add_check("file_exists", "ERROR", "File exists but is empty")
        return None

    report.add_check("file_exists", "OK", f"Size: {size:,} bytes")
    return DATA_FILE


def check_columns(report, df):
    """Validate all required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add_check("columns", "ERROR", f"Missing columns: {missing}")
    else:
        report.add_check("columns", "OK", f"All {len(REQUIRED_COLUMNS)} required columns present")


def check_integrity(report, df):
    """Row count, first/last timestamp, expected vs actual."""
    n = len(df)
    first = df["timestamp"].min()
    last = df["timestamp"].max()

    expected_hours = int((last - first).total_seconds() / 3600) + 1
    coverage = n / expected_hours * 100 if expected_hours > 0 else 0

    detail = (
        f"Rows: {n} | "
        f"Range: {first} → {last} | "
        f"Expected: {expected_hours} | "
        f"Coverage: {coverage:.1f}%"
    )

    if coverage < 95:
        report.add_check("integrity", "ERROR", detail)
    elif coverage < 99:
        report.add_check("integrity", "WARNING", detail)
    else:
        report.add_check("integrity", "OK", detail)

    return expected_hours


def check_continuity(report, df, expected_hours):
    """Detect missing hourly gaps."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    full_range = pd.date_range(
        df_sorted["timestamp"].iloc[0],
        df_sorted["timestamp"].iloc[-1],
        freq="h",
    )
    missing = full_range.difference(df_sorted["timestamp"])

    if len(missing) == 0:
        report.add_check("continuity", "OK", "No gaps detected")
    elif len(missing) <= 5:
        missing_str = ", ".join(str(m) for m in missing)
        report.add_check("continuity", "WARNING", f"{len(missing)} gap(s): {missing_str}")
    else:
        report.add_check("continuity", "ERROR", f"{len(missing)} missing hours (too many to list)")
        # Log first 10
        for m in missing[:10]:
            report.add_issue("ERROR", f"Missing hour: {m}")


def check_duplicates(report, df):
    """Detect duplicate timestamps."""
    dupes = df["timestamp"].duplicated().sum()
    if dupes == 0:
        report.add_check("duplicates", "OK", "No duplicate timestamps")
    else:
        report.add_check("duplicates", "ERROR", f"{dupes} duplicate timestamps found")


def check_nan_values(report, df):
    """Check for NaN values per column."""
    nan_counts = df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    if len(cols_with_nan) == 0:
        report.add_check("nan_values", "OK", "No NaN values in any column")
    else:
        detail = ", ".join(f"{col}={count}" for col, count in cols_with_nan.items())
        total = cols_with_nan.sum()
        if total > len(df) * 0.1:  # more than 10% NaN
            report.add_check("nan_values", "ERROR", f"High NaN count: {detail}")
        else:
            report.add_check("nan_values", "WARNING", f"NaN values: {detail}")


def check_realistic_values(report, df):
    """Flag values outside realistic ranges."""
    issues = []
    for col, (lo, hi) in REALISTIC_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if len(out_of_range) > 0:
            issues.append(f"{col}: {len(out_of_range)} rows outside [{lo}, {hi}]")

    if len(issues) == 0:
        report.add_check("realistic_values", "OK", "All values within expected ranges")
    else:
        detail = "; ".join(issues)
        if any("funding" in i or "close" in i for i in issues):
            report.add_check("realistic_values", "WARNING", detail)
        else:
            report.add_check("realistic_values", "WARNING", detail)


def check_freshness(report, df):
    """Last timestamp must be within STALENESS_THRESHOLD_HOURS of now."""
    now = datetime.now(timezone.utc)
    last_ts = df["timestamp"].max()

    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    age_hours = (now - last_ts).total_seconds() / 3600

    if age_hours <= STALENESS_THRESHOLD_HOURS:
        report.add_check("freshness", "OK", f"Last timestamp is {age_hours:.1f}h old")
    elif age_hours <= 6:
        report.add_check("freshness", "WARNING", f"Last timestamp is {age_hours:.1f}h old (threshold: {STALENESS_THRESHOLD_HOURS}h)")
    else:
        report.add_check("freshness", "ERROR", f"Last timestamp is {age_hours:.1f}h old — data may be stale!")


# =========================================================
# REPORTING
# =========================================================

STATUS_ICONS = {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌"}


def print_report(report):
    """Print human-readable report."""
    icon = STATUS_ICONS.get(report.status, "❓")
    print(f"\n{'='*60}")
    print(f"  BTCUSDT Derivatives Data Health Report")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")
    print(f"\n  Overall Status: {icon} {report.status}\n")

    for check in report.checks:
        ci = STATUS_ICONS.get(check["status"], "❓")
        print(f"  {ci} {check['name']:<20s} {check['detail']}")

    if report.issues:
        print(f"\n  --- Issues ---")
        for issue in report.issues:
            si = STATUS_ICONS.get(issue["severity"], "❓")
            print(f"  {si} [{issue['severity']}] {issue['message']}")

    print(f"\n{'='*60}\n")


def log_report(report):
    """Append report to health.log."""
    os.makedirs(os.path.dirname(HEALTH_LOG), exist_ok=True)

    lines = [
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Status: {report.status}",
    ]
    for check in report.checks:
        lines.append(f"  {check['status']:<8s} {check['name']:<20s} {check['detail']}")
    for issue in report.issues:
        lines.append(f"  {issue['severity']:<8s} {issue['message']}")
    lines.append("")

    with open(HEALTH_LOG, "a") as f:
        f.write("\n".join(lines))


# =========================================================
# MAIN
# =========================================================

def run_health_check(log=False, json_output=False):
    """Run full health check. Returns HealthReport."""
    report = HealthReport()

    # 1. File exists
    filepath = check_file_exists(report, )
    if filepath is None:
        if json_output:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print_report(report)
        return report

    # Load data
    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        report.add_check("load", "ERROR", f"Failed to load CSV: {e}")
        if json_output:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print_report(report)
        return report

    # 2. Columns
    check_columns(report, df)

    # 3. Integrity
    expected_hours = check_integrity(report, df)

    # 4. Continuity
    check_continuity(report, df, expected_hours)

    # 5. Duplicates
    check_duplicates(report, df)

    # 6. NaN values
    check_nan_values(report, df)

    # 7. Realistic values
    check_realistic_values(report, df)

    # 8. Freshness
    check_freshness(report, df)

    # Output
    if json_output:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print_report(report)

    if log:
        log_report(report)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTCUSDT Derivatives Data Health Check")
    parser.add_argument("--log", action="store_true", help="Append results to health.log")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    report = run_health_check(log=args.log, json_output=args.json)
    sys.exit(0 if report.status == "OK" else 1)


if __name__ == "__main__":
    main()
