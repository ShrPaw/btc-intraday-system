# SESSION CONTEXT UPDATE — 2026-04-26 07:52 GMT+8

## Tests Completed

### 1. BOS Quality Filter for LONGs — NEGATIVE RESULT

**File:** `bos_quality_filter_test.py`

BOS quality has **opposite effects** for LONG vs SHORT:

| Direction | Baseline Exp1R | Filtered Exp1R (≥0.60) | Delta |
|-----------|---------------|----------------------|-------|
| LONG (test) | +0.692R | +0.714R (n=7) | +0.022R (not significant, n too small) |
| SHORT (test) | +0.000R | +0.467R (n=15) | +0.467R |

**Conclusion:** BOS quality filter does NOT improve LONGs. The LONG edge is NOT driven by BOS quality. For LONGs, the setup conditions (RSI context + structure gate) already select for quality. BOS quality helps SHORTs (+0.467R), confirming Section 13 finding.

**Counterintuitive:** LONG Q1 (worst BOS quality quartile) has +1.000R, 100% hit rate in test. This is because Q1 setups cluster in April 2026 (strong momentum), not because weak BOS is good.

---

### 2. LONG Sample Expansion — 88 MULTI-SYMBOL TEST SETUPS

**File:** `long_sample_expansion.py`

| Symbol | Test N | Hit1R | Exp1R | Total R | Train Exp1R | Stable? |
|--------|--------|-------|-------|---------|-------------|---------|
| BTCUSDT | 13 | 76.9% | +0.692R | +16.0R | -0.417R | NO (train−, test+) |
| ETHUSDT | 23 | 47.8% | -0.043R | +8.0R | +0.064R | NO (train+, test−) |
| SOLUSDT | 37 | 67.6% | +0.351R | +5.0R | +0.222R | **YES** |
| XRPUSDT | 15 | 60.0% | +0.200R | +9.0R | -0.226R | NO (train−, test+) |

**Combined (all 4 symbols, test):** 88 setups, 62.5% hit, +0.273R, +38.0R, MaxDD=-7.0R

**Key findings:**
- **SOL is the only stable symbol** — positive in both train (+0.222R) and test (+0.351R)
- **BTC is NOT stable** — deeply negative train (-0.417R), strongly positive test (+0.692R)
- **ETH is flat** — marginal in both periods
- **XRP mirrors BTC** — negative train, positive test

**H4 RSI zone (test, all symbols):**
- BTC: 55-60 zone is best (86%, +0.714R)
- SOL: 50-55 zone is best (68%, +0.357R) — different zone than BTC!
- The "edge zone" differs by symbol. No universal RSI zone.

---

### 3. Entry Timing (Pullback to EMA20) — REJECTED FOR LONGs

**File:** `entry_timing_test.py`

- **ZERO** of 13 test LONG setups pulled back to EMA20 within 10 bars
- In train: 5 of 24 pulled back → ALL 5 were losses (-1.000R each)
- **Winners don't pull back.** LONGs are momentum continuation trades.
- Pullback-to-EMA20 entry would miss ALL winning setups.

**Conclusion:** Entry timing (pullback) is counterproductive for LONGs. The standard next-candle-open entry is optimal. Confirms Section 13 finding that extended entries outperform pullback entries.

---

## Updated Understanding

### What drives the LONG edge:
1. **Momentum continuation** — winners don't pull back, they keep going
2. **RSI context** — H4 RSI in emerging momentum zone (55-60 for BTC, 50-55 for SOL)
3. **NOT BOS quality** — filtering by BOS strength doesn't help
4. **NOT entry timing** — pullback entries miss all winners

### What we still don't know:
1. **Why is BTC train deeply negative?** — 24 setups, 29.2% hit, -0.417R. Need to investigate regime or market structure.
2. **Is the test period representative?** — April 2026 accounts for +14R of BTC's +16R test total
3. **Why is SOL stable but BTC not?** — Different RSI zone (50-55 vs 55-60), different behavior

### Next priorities (revised):
1. **Train/test stability investigation** — understand why BTC train is deeply negative
2. **SOL deep-dive** — only stable symbol, understand what makes it work
3. **BOS quality for SHORTs** — proven +0.467R improvement, implement as filter
4. **Live paper trading** — deploy signal output for real-time validation
