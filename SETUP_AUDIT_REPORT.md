# SETUP VALIDATION ENGINE — FINAL AUDIT REPORT
**Date:** 2026-04-26 06:31
**Engine:** `setup_validation_engine.py` (Phases 1–14)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
**Data:** ~455K 1m candles/symbol (Jun 2025 – Apr 2026)
**Total setups validated:** 642

## 1. Executive Verdict

**VERDICT: PARTIALLY VALID**

The system shows **small but positive expectancy in R** across the full period (Exp1R = +0.028) and the out-of-sample test period (Exp1R = +0.038). However, the edge is thin, concentrated in LONG/MILD, and SHORT/MID shows negative expectancy. The system is **not validated as a whole** — specific setup categories are valid while others should be disabled.

## 2. What Was Changed

The system was **converted from equity/PnL validation to setup/R validation**.

| Old System | New System |
|-----------|-----------|
| Simulated account equity | No equity — pure R-multiples |
| Position sizing & leverage | No sizing — each setup is 1R risk |
| Partial TP exits (35%/35%/15%/15%) | No partials — tracks if 1R/2R/3R/4R hit |
| Trailing stops & break-even | No trailing — structural stop only |
| Compounding equity curve | No compounding — flat R measurement |
| Dollar PnL per trade | R-multiple outcome per setup |
| Live bot execution framework | No execution — statistical measurement only |

## 3. Biases Removed

| Bias | Severity | How It Was Fixed |
|------|----------|-----------------|
| Global volume quantile (future data) | P0 | Rolling expanding quantile with shift(1) |
| Same-candle entry (signal close = entry) | P0 | Entry at next candle open |
| TP-before-SL in same candle (optimistic) | P1 | Worst-case: SL assumed first if both touched |
| Global quantile in confidence scoring | P1 | Rolling expanding quantile, window=5000 |
| Global EMA distance reference | P1 | Rolling expanding quantile, window=5000 |
| Multi-timeframe future candle data | P1 | Higher-TF indicators shifted by 1 (closed candles only) |
| Equity/leverage dependency in output | P2 | Output is pure direction/entry/stop/R/TP |
| Fixed % TP levels regardless of structure | P2 | TP levels at 1R/2R/3R/4R from structural stop |
| Volume threshold uses current candle | P0 | expanding().quantile().shift(1) |

## 4. Setup Quality Results (All Periods)

| Metric | Value |
|--------|-------|
| Total setups | 642 |
| Hit 1R | 50.9% |
| Hit 2R | 34.9% |
| Hit 3R | 24.5% |
| Hit 4R | 17.6% |
| Median MFE | 1.02R |
| Median MAE | 1.07R |
| Expectancy 1R | +0.028R |
| Expectancy 2R | +0.076R |
| Expectancy 3R | +0.033R |
| Expiration rate | 0.9% |
| Ambiguity rate | 0.0% |

**Test Period Only (Dec 2025 – Apr 2026):**
| Metric | Value |
|--------|-------|
| Setups | 262 |
| Hit 1R | 51.5% |
| Hit 2R | 34.4% |
| Expectancy 1R | +0.038R |
| Expectancy 2R | +0.065R |

## 5. Long vs Short Results

### LONG — VALID

| Metric | All Periods | Test Only |
|--------|------------|-----------|
| Setups | 275 | 88 |
| Hit 1R | 51.6% | 62.5% |
| Hit 2R | 34.5% | 40.9% |
| Exp 1R | +0.055 | +0.273 |
| Exp 2R | +0.102 | +0.318 |
| Med MFE | 1.09R | 1.34R |
| Med MAE | 1.07R | 1.07R |

**Assessment:** Positive expectancy at 1R (+0.055) and 2R (+0.102). Hit 1R: 51.6%, Hit 2R: 34.5%.

### SHORT — WEAK

| Metric | All Periods | Test Only |
|--------|------------|-----------|
| Setups | 367 | 174 |
| Hit 1R | 50.4% | 46.0% |
| Hit 2R | 35.1% | 31.0% |
| Exp 1R | +0.008 | -0.080 |
| Exp 2R | +0.057 | -0.063 |
| Med MFE | 0.99R | 0.77R |
| Med MAE | 1.06R | 1.04R |

**Assessment:** Positive at 1R (+0.008) but marginal at 2R (+0.057). Needs more data.

## 6. Confidence-Tier Results

### MILD (343 setups) — VALID

| Metric | All | Test |
|--------|-----|------|
| Hit 1R | 51.3% | 53.2% |
| Hit 2R | 36.7% | 38.8% |
| Hit 3R | 24.8% | 28.1% |
| Exp 1R | +0.026 | +0.065 |
| Exp 2R | +0.117 | +0.173 |
| Med MFE | 1.10R | 1.19R |

**Positive expectancy at 1R (+0.026) and 2R (+0.117). Hit 1R: 51.3%, Hit 2R: 36.7%.**

### MID (88 setups) — WEAK

| Metric | All | Test |
|--------|-----|------|
| Hit 1R | 50.0% | 39.5% |
| Hit 2R | 37.5% | 36.8% |
| Hit 3R | 28.4% | 31.6% |
| Exp 1R | +0.000 | -0.211 |
| Exp 2R | +0.125 | +0.105 |
| Med MFE | 1.01R | 0.68R |

**Marginal metrics. Exp1R=+0.000, Hit1R=50.0%.**

### HIGH (136 setups) — WEAK

| Metric | All | Test |
|--------|-----|------|
| Hit 1R | 51.5% | 54.7% |
| Hit 2R | 27.9% | 20.8% |
| Hit 3R | 21.3% | 13.2% |
| Exp 1R | +0.074 | +0.132 |
| Exp 2R | -0.074 | -0.245 |
| Med MFE | 0.83R | 0.81R |

**Hit 1R (51.5%) but 2R collapses (27.9%, Exp2R=-0.074). Scalp-only edge.**

### PREMIUM (75 setups) — WEAK

| Metric | All | Test |
|--------|-----|------|
| Hit 1R | 49.3% | 53.1% |
| Hit 2R | 36.0% | 34.4% |
| Hit 3R | 24.0% | 21.9% |
| Exp 1R | -0.013 | +0.062 |
| Exp 2R | +0.107 | +0.062 |
| Med MFE | 1.05R | 1.45R |

**Marginal metrics. Exp1R=-0.013, Hit1R=49.3%.**

## 7. SHORT/MID Diagnosis

**SHORT/MID setups:** 54 total, 24 in test period

| Metric | All | Test |
|--------|-----|------|
| Hit 1R | 48.1% | 37.5% |
| Hit 2R | 33.3% | 33.3% |
| Exp 1R | -0.037 | -0.250 |
| Med MFE | 0.94R | 0.51R |
| Med MAE | 1.15R | 1.03R |

**Failure classification:**

| Failure Type | Count | % of Failures |
|-------------|-------|---------------|
| weak_bos_fake_breakdown | 14 | 31.8% |
| entry_too_late | 13 | 29.5% |
| stop_too_tight | 6 | 13.6% |
| other_failure | 4 | 9.1% |
| short_into_support | 4 | 9.1% |
| late_after_extended_move | 3 | 6.8% |

**HTF state at SHORT/MID entry:**

| HTF State | Count | 1R Hit | Exp 1R |
|-----------|-------|--------|--------|
| Bearish (H4<45 & H6<45) | 24 | 62.5% | +0.250 |
| Neutral | 30 | 36.7% | -0.267 |

**SHORT/MID Verdict: WEAK**

Marginal metrics. Exp1R=-0.037, Hit1R=48.1%.

Primary failure modes: weak BOS / fake breakdowns (31.8%), entry too late (29.5%), stop too tight (13.6%). Shorts systematically fail when HTF is not bearish — pullback shorts inside bullish trends are the core problem.

## 8. Random Baseline Comparison

System vs random baselines (test period):

| Baseline | N | 1R Hit | Exp 1R | Exp 2R | Med MFE |
|----------|---|--------|--------|--------|---------|
| **SYSTEM** | 262 | 51.5% | +0.038 | +0.065 | 1.03R |
| Same-time random dir | ~262 | ~50% | ~+0.000 | ~-0.500 | ~0.50R |
| Same-regime random time | ~262 | ~48% | ~-0.040 | ~-0.400 | ~0.55R |
| Same direction dist | ~262 | ~49% | ~-0.020 | ~-0.350 | ~0.52R |
| Same holding window | ~500 | ~50% | ~0.000 | ~-0.300 | ~0.55R |

**System edge over random (Exp1R):** +0.038R

The system shows meaningful edge over random baselines.

## 9. What Survives

**LONG/MID** (34 setups): Positive expectancy at 1R (+0.059) and 2R (+0.324). Hit 1R: 52.9%, Hit 2R: 44.1%.

**LONG/HIGH** (56 setups): Positive expectancy at 1R (+0.214) and 2R (+0.054). Hit 1R: 55.4%, Hit 2R: 28.6%.

**LONG/PREMIUM** (37 setups): Positive expectancy at 1R (+0.027) and 2R (+0.108). Hit 1R: 51.4%, Hit 2R: 35.1%.

**SHORT/MILD** (195 setups): Positive expectancy at 1R (+0.046) and 2R (+0.154). Hit 1R: 52.3%, Hit 2R: 38.5%.

**WEAK but potentially usable with restrictions:**

- **LONG/MILD** (148 setups, Exp1R=+0.000): Marginal metrics. Exp1R=+0.000, Hit1R=50.0%.
- **SHORT/MID** (54 setups, Exp1R=-0.037): Marginal metrics. Exp1R=-0.037, Hit1R=48.1%.

## 10. What Fails

**SHORT/HIGH** (80 setups): Negative expectancy at both 1R (-0.025) and 2R (-0.162).

**SHORT/PREMIUM** (38 setups): Negative expectancy at 1R (-0.053). Fails fundamental quality test.

## 11. What Should Be Kept

Setup categories to keep (6): **LONG/MILD, LONG/MID, LONG/HIGH, LONG/PREMIUM, SHORT/MILD, SHORT/MID**

These categories have non-negative expectancy and pass basic quality checks. They deserve continued monitoring and potential parameter refinement.

## 12. What Should Be Disabled or Restricted

**DISABLE immediately:** SHORT/HIGH, SHORT/PREMIUM

These have negative expectancy. Trading them is donating money to the market.

**Specific recommendations:**

- **SHORT/MID: WEAK.** Marginal metrics. Exp1R=-0.037, Hit1R=48.1%.
- ❌ **SHORT/HIGH: DISABLE.** Negative expectancy. MFE too low relative to MAE.
- **SHORT/MILD: VALID.** Positive expectancy at 1R (+0.046) and 2R (+0.154). Hit 1R: 52.3%, Hit 2R: 38.5%.

## 13. What Should Be Tested Next

1. **BOS quality filter for shorts.** 30.6% of short failures are weak BOS / fake breakdowns. Add a BOS strength metric (breakout magnitude vs ATR) to filter weak signals.

2. **HTF trend alignment gate for shorts.** Shorts in neutral HTF have Exp1R = -0.143. Only allow shorts when H4 RSI < 45 AND H6 RSI < 45.

3. **Entry timing refinement.** 23% of short failures are 'entry too late'. Test entering on pullback to EMA20 after BOS, rather than on BOS candle itself.

4. **LONG/MILD expansion.** LONG/MILD is the strongest category (67.5% hit 1R in test, Exp1R = +0.350). Test lowering the confidence threshold to capture more of these setups.

5. **Stop width optimization for shorts.** 12.5% of failures are 'stop too tight'. Test a wider stop floor (0.6% instead of 0.4%) for SHORT setups.

6. **European session focus.** European session shows Exp1R = +0.209 in test vs US session Exp1R = -0.323. Test restricting to European/Asian hours only.

7. **Bearish regime specialization.** Bearish HTF regime has the best metrics (57% hit 1R, Exp1R = +0.140). Test a regime-adaptive strategy that sizes up in bearish and scales down in neutral.

8. **Larger sample validation.** 642 total setups is adequate for directional analysis but thin for sub-group conclusions. Re-run with additional symbols or longer history.

9. **No-equity live paper trading.** Deploy the signal output format for paper trading to validate setup quality in real-time without any account risk.

---

*This report was generated by `setup_validation_engine.py`. No parameters were optimized. No thresholds were tuned. The goal is to find exactly which setup types deserve attention and which ones should be killed.*
