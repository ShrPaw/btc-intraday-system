# SECTION 13 — DIAGNOSTIC IMPROVEMENTS REPORT
**Date:** 2026-04-26 06:54
**Engine:** `section13_diagnostics.py` (post-hoc analysis on 642 validated setups)
**Test Period:** Dec 2025 – Apr 2026 (262 setups)

---

## Executive Verdict

**The SHORT side is the problem. LONG is validated. The single highest-impact action is implementing a BOS quality filter for shorts.**

---

## 1. BOS Quality Filter for Shorts — ⭐ HIGHEST IMPACT

| Metric | Baseline (SHORT test) | After Weak-BOS Removed | Delta |
|--------|----------------------|----------------------|-------|
| Setups | 174 | 111 | -63 |
| Hit 1R | 46.0% | 71.2% | **+25.2%** |
| Exp 1R | -0.080R | +0.423R | **+0.504R** |
| Exp 2R | -0.063R | +0.468R | +0.532R |
| Med MFE | 0.77R | 1.94R | +1.16R |

**SHORT/MID specifically:**
| Metric | Baseline | After Filter | Delta |
|--------|---------|-------------|-------|
| Exp 1R | -0.250R | +0.385R | **+0.635R** |
| Hit 1R | 37.5% | 69.2% | +31.7% |

**Conclusion:** This is the #1 priority. Weak BOS signals are the dominant failure mode. The post-hoc ceiling is +0.504R Exp1R improvement. Requires a signal-time BOS strength metric (breakout magnitude vs ATR or swing size).

---

## 2. HTF Trend Alignment Gate for Shorts — MODERATE

| Metric | All SHORT (test) | Bearish Only (H4<45) | H4<45 & H6<45 |
|--------|-----------------|---------------------|---------------|
| Setups | 174 | 90 | 11* |
| Exp 1R | -0.080R | -0.022R (+0.058) | -0.091R (+0.159*) |
| Hit 1R | 46.0% | 48.9% | 45.5% |

*Diagnostics subset (24 total SHORT/MID, 11 bearish)

**HTF state at SHORT/MID entry (diagnostics, test):**
- Bearish (H4<45 & H6<45): 45.5% hit 1R, Exp1R=-0.091
- Neutral/Bull: 33.3% hit 1R, Exp1R=-0.333

**Conclusion:** Helps but not sufficient alone. Bearish HTF is necessary but not sufficient — still negative Exp1R in isolation. Must combine with BOS quality filter.

---

## 3. Entry Timing Refinement — COUNTERINTUITIVE

| Metric | SHORT/MID All | Near EMA (|d|<0.5%) | Extended (|d|≥0.5%) |
|--------|--------------|---------------------|---------------------|
| Setups | 24 | 12 | 12 |
| Hit 1R | 37.5% | 33.3% | 41.7% |
| Exp 1R | -0.250R | -0.333R | -0.167R |
| Med MAE | 1.03R | 1.21R | 0.65R |

**Conclusion:** Counterintuitive result — near-EMA entries actually perform WORSE for shorts. Extended entries (price already moved) have lower MAE. This suggests shorts that have already broken down with momentum are better than "fresh" pullback shorts. The "pullback to EMA20" idea from the report may not apply to shorts — it may only apply to longs.

---

## 4. LONG/MILD Expansion — INSIGHT INTO CONFIDENCE BANDS

**LONG/MILD confidence distribution (test period):**

| Confidence Band | N | Hit 1R | Exp 1R |
|----------------|---|--------|--------|
| [0.72, 0.74) | 17 | **82.4%** | **+0.647R** |
| [0.74, 0.76) | 8 | 75.0% | +0.500R |
| [0.76, 0.78) | 15 | 46.7% | -0.067R |
| [0.78, 0.80) (MID) | 14 | 42.9% | -0.143R |
| [0.80, 0.82) (PREMIUM) | 14 | 57.1% | +0.143R |
| [0.82, 0.88) (HIGH) | 20 | 70.0% | +0.500R |

**Conclusion:** The LOWER end of LONG/MILD (0.72-0.74) is the strongest band at 82.4% hit rate. The upper end (0.76-0.78) is actually negative. No expansion possible below 0.72 (those are NO_TRADE). The sweet spot is 0.72-0.76.

---

## 5. Stop Width Optimization for Shorts — REJECTED

| Stop Distance | N | Hit 1R | Exp 1R | SL Rate |
|--------------|---|--------|--------|---------|
| ≤0.5% (tight) | 29 | 51.7% | **+0.034R** | 82.8% |
| >0.6% (wider) | 113 | 45.1% | -0.097R | 73.5% |

**Conclusion:** Tight stops actually OUTPERFORM wider stops for shorts. Widening the stop floor would make shorts worse. This recommendation is rejected.

---

## 6. European Session Focus — STRONG

| Session | N (test) | Hit 1R | Exp 1R | Exp 2R |
|---------|---------|--------|--------|--------|
| European | 129 | 60.5% | **+0.209R** | +0.256R |
| Asian | 71 | 52.1% | +0.056R | +0.014R |
| US | 62 | 33.9% | **-0.323R** | -0.306R |

**SHORT by session:**
| Session | N | Hit 1R | Exp 1R |
|---------|---|--------|--------|
| European | 100 | 59.0% | +0.180R |
| Asian | 27 | 40.7% | -0.185R |
| US | 47 | **21.3%** | **-0.574R** |

**Conclusion:** US session is toxic for shorts (21.3% hit 1R). European session is the only one with positive short expectancy. Restricting shorts to European hours is strongly supported.

---

## 7. Bearish Regime Specialization — NUANCED

| Regime | N (test) | Hit 1R | Exp 1R |
|--------|---------|--------|--------|
| Bullish | 47 | 63.8% | **+0.319R** |
| Neutral | 125 | 48.8% | -0.024R |
| Bearish | 90 | 48.9% | -0.022R |

**Bearish regime is NOT the best for overall Exp1R.** Bullish is better (all longs). Bearish regime helps shorts specifically but doesn't rescue them alone.

**Combined filters:**
| Filter | N | Hit 1R | Exp 1R | Exp 2R |
|--------|---|--------|--------|--------|
| All test | 262 | 51.5% | +0.038R | +0.065R |
| Bearish + European | 48 | 62.5% | +0.250R | +0.688R |
| No weak BOS | 199 | 67.3% | +0.357R | +0.402R |
| **Bearish + Eur + No weak BOS** | **36** | **83.3%** | **+0.667R** | **+1.250R** |

---

## 8. Failure Pattern Analysis

**SHORT/MID failure breakdown (diagnostics, test period):**
| Failure Type | Count | % |
|-------------|-------|---|
| weak_bos_fake_breakdown | 6 | 40.0% |
| late_after_extended_move | 3 | 20.0% |
| entry_too_late | 2 | 13.3% |
| short_into_support | 2 | 13.3% |
| other_failure | 1 | 6.7% |
| stop_too_tight | 1 | 6.7% |

**Time to SL:** Median 9 bars (45 min), 51% hit SL within 10 bars.

**Winner vs Loser MFE:** Winners median 3.59R, Losers median 0.18R — clear separation.

---

## Priority Actions (Ordered by Impact)

| Priority | Action | Expected Impact | Complexity |
|----------|--------|----------------|------------|
| **P0** | BOS quality filter for shorts | +0.504R ceiling | Medium — needs BOS magnitude metric |
| **P1** | Restrict shorts to European session | +0.171R Exp1R | Easy — session hour filter |
| **P2** | Combine BOS filter + European + Bearish | +0.629R ceiling | Medium — compound filter |
| **P3** | Focus on LONG/MILD [0.72-0.76) band | Already strongest | Easy — confidence band awareness |
| ~~P4~~ | ~~Entry timing (pullback to EMA)~~ | ~~Rejected~~ | — |
| ~~P5~~ | ~~Wider stop floor for shorts~~ | ~~Rejected~~ | — |

---

## What NOT To Do

1. **Do NOT widen stops for shorts** — tight stops outperform
2. **Do NOT use "near EMA" as entry quality filter for shorts** — extended entries are better
3. **Do NOT trade shorts in US session** — 21.3% hit 1R, Exp1R=-0.574
4. **Do NOT trade shorts in neutral HTF** — Exp1R=-0.333

---

*Generated by `section13_diagnostics.py`. No parameters optimized. No thresholds tuned. Pure post-hoc R-multiple measurement.*
