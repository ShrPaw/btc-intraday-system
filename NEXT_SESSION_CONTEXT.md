NEXT SESSION CONTEXT — btc-intraday-system
===========================================
Repo: https://github.com/ShrPaw/btc-intraday-system
Read: PROJECT_CONTEXT.md, SESSION_CONTEXT.md, SESSION_CONTEXT_20260426_TESTS.md

============================================================
STATUS: DATA ACCUMULATION PHASE — RESEARCH FROZEN
============================================================

We are NOT doing research. We are NOT running hypotheses.
We are ONLY maintaining data integrity until ~June 29, 2026.

Resumption criteria:
  - Dataset reaches ~2160 hours (~3 months)
  - Each event class can reach N >= 50

Until then:
  - Do NOT modify event definitions
  - Do NOT run analysis
  - Do NOT interpret patterns
  - Do NOT suggest new signals

Your ONLY job is maintaining the data pipeline.

============================================================
WHAT HAPPENED TODAY (2026-04-26)
============================================================

1. FUNDING RATE RESEARCH — completed 3 tests (BEFORE FREEZE):
 - funding_event_study.py: funding extremes as discrete events. ~127 events.
   Behavioral finding: reversal ~50%. Not standalone signal.
 - funding_modifier_test.py: does funding modify large move outcomes?
   YES for HIGH funding: -0.049% at 60m vs NORMAL (t=-5.16, stable).
   LOW funding: INCONCLUSIVE (train/test instability).
 - funding_filter_test.py: removing HIGH funding trades → WORSENS outcomes.
   HIGH funding trades are NOT worse in direction-adjusted terms.

2. OI/TAKER EXPLORATORY — all INCONCLUSIVE (N<50 for every event class):
 - oi_shock_study.py, oi_taker_exploratory.py
 - 26 days of data (Mar 31 – Apr 25 2026). Too thin.
 - Taker buy/sell ratio shows most promise but needs 3+ months.

3. DATA INFRASTRUCTURE — built and deployed:
 - collect_derivatives.py: hourly BTCUSDT derivatives collector
   (OHLCV, OI, taker buy/sell, LS ratio, funding)
 - Append-only CSV at data/collected/btcusdt_hourly_derivatives.csv
 - Cron job: btc-derivatives-collector runs at :05 every hour UTC
 - Backfilled 628 hours (Mar 31 → Apr 26). Zero gaps, zero NaN.
 - Added validate_row() pre-append validation (hardened)
 - Added explicit gap logging on failures

4. HEALTH CHECK SYSTEM:
 - check_data_health.py: 7-point validation
   (file, columns, integrity, continuity, duplicates, NaN, realism, freshness)
 - Cron: runs daily at 06:00 UTC → appends to health.log
 - Exit code 1 on ERROR, 0 on OK

5. SNAPSHOT SYSTEM:
 - snapshot_data.py: daily CSV backup to data/snapshots/
 - 30-day retention, auto-prunes old snapshots
 - Cron: runs daily at 23:55 UTC

6. EVENT STUDY FRAMEWORK (skeleton only):
 - event_study_framework.py: stubbed functions
   load_data(), define_events(), compute_forward_returns(),
   compute_MFE_MAE(), baseline_comparison()
 - NO logic implemented. Ready to fill when research resumes.

============================================================
CURRENT DATA STATUS
============================================================

File: data/collected/btcusdt_hourly_derivatives.csv
Rows: 629
Range: 2026-03-31 00:00 UTC → 2026-04-26 03:00 UTC
Gaps: 0
Duplicates: 0
NaN values: 0
Coverage: 100%

Target: ~2160 hours (3 months)
Remaining: ~1532 hours
ETA: ~64 days → June 29, 2026

============================================================
CRON JOBS (active)
============================================================

1. Collector: 5 * * * * (every hour at :05 UTC)
   Script: collect_derivatives.py
   Log: data/collected/cron.log

2. Health check: 0 6 * * * (daily at 06:00 UTC)
   Script: check_data_health.py --log
   Log: data/collected/health_cron.log

3. Snapshot: 55 23 * * * (daily at 23:55 UTC)
   Script: snapshot_data.py --clean
   Log: data/collected/snapshot_cron.log

============================================================
KNOWN BUGS (from strict re-audit, NOT fixed today)
============================================================

- P0: Intracandle ambiguity dead code (SL checked after TP)
  → Fixed in setup_validation_engine.py earlier
- P0: Baseline index bug (all random baselines produce 0 rows)
  → Fixed in setup_validation_engine.py earlier

============================================================
WHAT'S FROZEN (from SESSION_CONTEXT priorities)
============================================================

DO NOT work on these until data is ready:
1. Train/test stability investigation (BTC train -0.417R)
2. SOL deep-dive (only stable symbol)
3. BOS quality filter for SHORTs (proven +0.467R)
4. Live paper trading deployment
5. OI/taker exploratory re-run (need N>=50)

============================================================
MANUAL COMMANDS
============================================================

# Health check
python3 check_data_health.py --log

# Snapshot
python3 snapshot_data.py --clean

# Collect current hour
python3 collect_derivatives.py

# Backfill gaps
python3 collect_derivatives.py --backfill-start "2026-03-01"

# Check data
wc -l data/collected/btcusdt_hourly_derivatives.csv
tail -3 data/collected/btcusdt_hourly_derivatives.csv

# Check cron logs
tail -20 data/collected/cron.log

# Bot status (if redeployed)
systemctl status btc-signal-bot

============================================================
KEY FILES (added/modified today)
============================================================

- collect_derivatives.py (hardened: validate_row, gap logging)
- check_data_health.py (NEW: 7-point health check)
- snapshot_data.py (NEW: daily backup + retention)
- event_study_framework.py (NEW: skeleton for future research)
- data/collected/btcusdt_hourly_derivatives.csv (628 hours backfilled)
- data/snapshots/btcusdt_hourly_20260426.csv (first snapshot)

============================================================
RESEARCH RESULTS (for reference, DO NOT re-run until data ready)
============================================================

Funding:
- funding_event_study_results.csv: reversal ~50%, not standalone
- funding_modifier_results.csv: HIGH funding → -0.049% at 60m
- funding_filter_test_results.csv: filter WORSENS outcomes

OI/Taker:
- oi_taker_exploratory_results.csv: INCONCLUSIVE (N<50 all classes)

Prior (from SESSION_CONTEXT):
- Setup validation: 642 setups, timing edge CONFIRMED (1.63x random)
- LONG-only BTCUSDT test: +0.692R, 76.9% hit, but train -0.417R
- SOLUSDT: only stable symbol (train +0.222R, test +0.351R)
- BOS quality for SHORTs: +0.467R at ≥0.60 floor
- SHORT direction accuracy: 45.2% (worse than coin flip)

============================================================
END CONTEXT
============================================================
