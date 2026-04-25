# BTC Intraday System — Session Context
## Last Updated: 2026-04-26 07:52 GMT+8

---

## PROJECT IDENTITY

**Repository:** https://github.com/ShrPaw/btc-intraday-system
**What it is:** A BTC/crypto intraday setup detector evaluated in R-multiples.
**What it is NOT:** A live trading bot, equity engine, or execution system.
**Framework:** Setup quality validation — measures everything in R with structural stops.

---

## FILE STRUCTURE

```
btc-intraday-system/
├── setup_validation_engine.py      # Main engine (11 phases, ~1600 lines)
├── strict_audit.py                 # Strict audit script
├── live_signal_bot.py              # Live Telegram signal bot (NOT the validation engine)
├── btc_intraday_system.py          # Original backtest equity engine (deprecated)
├── fetch_btc_data.py               # Data fetcher from Binance
├── generate_report_files.py        # Report generator
│
├── volatility_expansion_test.py    # Volatility expansion test (directionless)
├── long_only_test.py               # LONG-only & directionless expansion comparison
├── long_only_final.py              # LONG-only final validation (fixed 1R risk)
├── regime_filter_test.py           # HTF regime filter test
├── confidence_conditioning.py      # Confidence conditioning layer
├── section13_diagnostics.py        # Section 13 diagnostic improvements
│
├── SETUP_AUDIT_REPORT.md           # Original audit report
├── STRICT_REAUDIT_REPORT.md        # Strict re-audit (found 2 P0 bugs)
├── SECTION13_DIAGNOSTIC_REPORT.md  # Section 13 diagnostic report
├── PROJECT_CONTEXT.md              # Original project context
│
├── data/features/
│   ├── btcusdt_1m.csv              # ~455K 1m candles (Jun 2025 – Apr 2026)
│   ├── ethusdt_1m.csv              # ~455K candles
│   ├── solusdt_1m.csv              # ~455K candles
│   ├── xrpusdt_1m.csv              # ~455K candles
│   ├── setup_validation_results.csv # 642 validated setups (all symbols)
│   ├── setup_events.csv             # Copy of validation results
│   ├── setup_summary_by_group.csv   # 31 aggregated groups
│   └── short_mid_diagnostics.csv    # 54 SHORT/MID diagnostics
│
├── .env.example
├── .gitignore
└── venv/
```

---

## ENGINE ARCHITECTURE (setup_validation_engine.py)

### Config
- SYMBOLS: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
- RSI_PERIOD=14, EMA_PERIOD=20, STRUCTURE_GATE_WINDOW=5
- STOP_FLOOR_PCT=0.4%, STOP_CAP_PCT=2.0%
- TP_R_MULTIPLES=[1,2,3,4]
- TRAIN_CUTOFF=2025-12-01, OUTCOME_HORIZON_BARS=200

### Phases
1. Data loading + resampling (1m→5m/15m/30m/4h/6h/12h)
2. Indicators (RSI, EMA20/50, BOS, ADX)
3. Setup engine (RSI_SCALP + RSI_TREND context)
4. Structure gate (EMA reclaim + BOS within 5 bars)
5. Confidence engine (rolling quantiles, ADX modifier)
6. Trade decision (volume filter, cooldown)
7. Structural stop (swing high/low + EMA20, floor/cap)
8. R-multiple tracking (1R/2R/3R/4R TP levels)
9. Outcome tracking (worst-case intracandle, SL-first)
10. Validation (per-setup, walk-forward)
11. Reporting (metrics, baselines, short focus)

### Data Leakage Fixes (9 total, all verified)
- F1/F9: Global volume quantile → expanding().quantile().shift(1)
- F2: Same-candle entry → next candle open
- F3: TP-before-SL → SL-first worst-case
- F4: Global quantile in confidence → rolling 5000-bar shift(1)
- F5: Global EMA dist → rolling 5000-bar shift(1)
- F6: Multi-TF future candle → shift(1) on merge_asof
- F7: Equity/leverage → pure R-multiples
- F8: Fixed % TP → TP at 1R/2R/3R/4R from structural stop

### P0 Bugs Found and Fixed (2026-04-26)
1. **Intracandle ambiguity dead code** — TP check ran before SL check. Ambiguity block never fired. Fixed: SL checked first, ambiguous TPs don't count as hits.
2. **Baseline index bug** — df_test had non-sequential index labels from original df. entry_idx >= len(df) always true. Fixed: df.reset_index(drop=True) at start.

### P1 Fix: Primary Validation Methodology
- Primary validation = BTCUSDT only
- Headline verdict based on BTCUSDT
- Multi-symbol = secondary robustness checks only
- Never aggregate into one headline metric

---

## KEY FINDINGS (as of 2026-04-26)

### 1. Timing Edge is REAL
- Setups predict 1.63x larger absolute moves than random (20 bars)
- Directionless max |move|: 1.75-1.95x vs random
- P(|move| > 1R) at 20 bars: setup=49.1%, random=24.4%

### 2. LONG-Only on BTCUSDT is VALIDATED (test period)
- 13 setups, Exp1R=+0.692R, Hit1R=76.9%
- Total R=+16.0R (fixed 1R risk), Max DD=-1.0R, Max consec loss=1
- **CAVEAT:** Train period deeply negative (-0.417R, -8.0R total)
- **CAVEAT:** April 2026 accounts for +14R of +16R total (4 trades)
- **CAVEAT:** 13 setups is thin sample

### 3. SHORT is the Problem
- SHORT test: Exp1R=+0.000R, Hit1R=50.0%, Total R=-5.0R
- SHORT direction accuracy: 45.2% correct at 20 bars (worse than coin flip)
- SHORT destroys value: 42 setups, -5R, 8 consecutive losses

### 4. Where LONG Edge Lives (Confidence Conditioning)
**Strongest conditions (Exp1R > 0.1, N >= 5):**
- H12 RSI 50-55: Exp1R=+0.600R, Hit1R=80%, N=10
- H4 RSI 55-60: Exp1R=+0.294R, Hit1R=65%, N=17
- H6 RSI 50-55: Exp1R=+0.111R, Hit1R=56%, N=9

**Worst conditions:**
- H4 RSI 50-55: Exp1R=-0.714R (death zone)
- H12 RSI 55-60: Exp1R=-1.000R (0% hit rate)
- H4 RSI 60-70: Exp1R=-0.273R (overbought)

**Pattern:** Edge lives in EMERGING momentum (RSI 50-60), not strong (>60) or flat (<50).

### 5. Baseline Comparison (BTCUSDT test)
- Beats full-dataset random by +0.460R
- Beats same-regime random by +0.021R
- LOSES to same-time random-dir by -0.170R
- LOSES to same-session random by -0.170R

### 6. Regime Filter: No Improvement
All 13 LONG test setups already in bullish HTF. Filter adds nothing.

### 7. Multi-Symbol Robustness
- BTC: +0.164R (test), SOL: +0.128R, ETH: -0.048R, XRP: -0.091R
- 2/4 positive. SOL confirms, ETH/XRP contradict.

---

## WHAT HAS BEEN TESTED

| Test | Result | File |
|------|--------|------|
| Worst-case intracandle | Fixed (P0→Fixed) | setup_validation_engine.py |
| Random baselines | Fixed (P0→Fixed) | setup_validation_engine.py |
| Next-open entry | Fixed | setup_validation_engine.py |
| Rolling thresholds | Fixed | setup_validation_engine.py |
| HTF closed-candle | Fixed | setup_validation_engine.py |
| Structural stop validity | Fixed | setup_validation_engine.py |
| BTC-only primary validation | Fixed (P1→Fixed) | setup_validation_engine.py |
| Section 13 diagnostics | Completed | section13_diagnostics.py |
| Volatility expansion | CONFIRMED 1.63x | volatility_expansion_test.py |
| LONG-only validation | +0.692R test | long_only_final.py |
| Directionless expansion | 1.75-1.95x | long_only_test.py |
| HTF regime filter | No improvement | regime_filter_test.py |
| Confidence conditioning | Mapped edge zone | confidence_conditioning.py |

---

## WHAT HAS BEEN TESTED (session 2 additions)

| Test | Result | File |
|------|--------|------|
| BOS quality filter for LONGs | NEGATIVE — doesn't improve LONGs | bos_quality_filter_test.py |
| BOS quality filter for SHORTs | POSITIVE — +0.467R at ≥0.60 floor | bos_quality_filter_test.py |
| LONG sample expansion (4 symbols) | 88 setups, +0.273R combined | long_sample_expansion.py |
| SOLUSDT LONG stability | POSITIVE — only stable symbol | long_sample_expansion.py |
| Entry timing (pullback to EMA20) | REJECTED — zero pullbacks in test winners | entry_timing_test.py |

---

## WHAT HAS NOT BEEN TESTED (from original Section 13)

1. ~~BOS quality filter~~ — DONE, helps SHORTs only (+0.467R at ≥0.60 floor)
2. ~~Entry timing refinement~~ — DONE, rejected for LONGs (winners don't pull back)
3. LONG/MILD expansion (lower confidence threshold) — partially mapped in Section 13
4. Stop width optimization (rejected — tight stops outperform)
5. European session focus (not re-tested after P0 fixes)
6. Live paper trading deployment
7. **Train/test stability investigation** — BTC train deeply negative
8. **SOL deep-dive** — only stable symbol, different RSI zone

---

## STRATEGY LOGIC (DO NOT CHANGE)

- RSI_SCALP: h4_fresh_long/short + m15 RSI > 50 + h6/h12 RSI > 50
- RSI_TREND: h4 RSI > 55 + slopes positive + m15 RSI > 55 + h12 RSI > 50
- Structure gate: EMA reclaim + BOS within 5 bars
- Confidence: weighted blend of alignment, freshness, momentum, structure, volatility
- Stop: structural (swing low/high, EMA20 for RSI_TREND), floor 0.4%, cap 2.0%

---

## NEXT SESSION PRIORITIES (revised 2026-04-26)

1. **Train/test stability investigation** — why is BTC train -0.417R? Regime? Market structure?
2. **SOL deep-dive** — only stable symbol (train +0.222R, test +0.351R). RSI zone 50-55.
3. **BOS quality for SHORTs** — implement as live filter (proven +0.467R)
4. **Live paper trading** — deploy signal output for real-time validation

---

## CRITICAL WARNINGS

- Do NOT aggregate multi-symbol results into headline metrics
- Do NOT claim edge from 13 setups alone — need more data
- Do NOT ignore train-period failure (-0.417R for BTC)
- Do NOT trade SHORTs without rebuilding direction logic from scratch
- Do NOT use equity/leverage/position sizing in validation
- Always use worst-case intracandle (SL-first)
- Always use next-candle open for entry
- Always use rolling/past-only thresholds
- **NEW:** BOS quality does NOT help LONGs — don't apply it
- **NEW:** Pullback-to-EMA20 does NOT help LONGs — momentum continuation is the edge
- **NEW:** SOL is the only symbol with train/test stability
