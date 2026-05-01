# MANTIS SPE — DECISION MEMO

## Date: 2026-05-01
## Symbol: BTCUSDT
## Classification: Final

---

## Summary

SPE (Structured Price Event) is an 8-layer evaluation framework tested across
315 days of BTCUSDT 5m data (Jun 2025 – Apr 2026). The framework is
structurally valid (no lookahead), but the edge is regime-conditional,
direction-conditional, and outlier-dependent.

Verdict from strict validation: **B — Promising but not fully validated.**

This memo classifies each module and defines the path forward.

---

## Module Classification

### 1. SPE_GENERAL — ❌ REJECTED

**Definition:** Full 8-layer SPE pass, all directions, all volume regimes.

**Reason:** Unstable. Outlier-dependent. Not symmetric.

Evidence:
- Removing top 5% of events flips edge from +2.0bps to -12.8bps
- Only 45% of weeks are positive
- First half: +4.9bps / Second half: -1.0bps (not stable across time)
- Monthly: 5 positive, 6 negative
- Edge does not survive symmetric treatment of directions

**Action:** No further development. No deployment.

---

### 2. SPE_LONG — ❌ REJECTED

**Definition:** Full 8-layer SPE pass, LONG direction only.

**Reason:** No edge in any regime.

Evidence:
- LONG overall: N=421, 30m=-2.0bps, PF=0.94, maker=-3.0bps
- LONG + HIGH_VOLUME: N=372, 30m=-0.9bps, PF=0.97, maker=-1.9bps
- LONG + HIGH_VOLUME + HIGH_VOLATILITY: insufficient sample, direction still negative
- No volume or volatility regime rescues LONG direction

**Action:** No further development. Do not trade LONG side of SPE.

---

### 3. SPE_SHORT_STRESS — ⚠️ CANDIDATE FOR FURTHER VALIDATION

**Definition:**
- Direction: SHORT only
- Volume regime: HIGH_VOLUME only
- Volatility regime: HIGH_VOLATILITY only
- Layer pass: Full L1-L8 all pass
- Execution: Maker only (0-1bps)
- Mode: Observation-only

**Why candidate, not approved:**
- Strong numbers (N=651 subset of SHORT+HV, 30m=+13.7bps, PF=1.39, maker=+12.7bps)
- But edge is outlier-dependent (top 5% removal flips sign)
- Only 45% of weeks positive
- Bearish data bias: Jun 2025 – Apr 2026 includes significant bearish regime
- Cannot distinguish genuine SHORT edge from bearish market luck

**Evidence (for reference, not approval):**

| Segment | N | 30m | 60m | MFE/MAE | PF | Maker |
|---------|---|-----|-----|---------|-----|-------|
| SHORT + HV | 752 | +11.8bps | +12.0bps | 1.26 | 1.30 | +10.8bps |
| SHORT + HV + HIGH_VOL | 651 | +13.7bps | +16.6bps | 1.25 | 1.39 | +12.7bps |

**Action:** Observe only. Log events in live market. No execution.

---

## Required Next Validation (before any upgrade from candidate)

SPE_SHORT_STRESS must pass ALL of the following before reconsideration:

1. **Separate walk-forward test**
   - Train on first 6 months, test on remaining
   - Train on first 9 months, test on remaining
   - Both test periods must show positive maker net

2. **Newer unseen data**
   - Accumulate data from Apr 2026 onward
   - Minimum 3 months of new data (~2,160 hours)
   - Re-run full audit on new data only

3. **Top 5% outlier removal**
   - Edge must survive removal of top 5% best events
   - Net return must remain positive after removal

4. **Same-regime random baseline**
   - Random SHORT entries in HIGH_VOLUME + HIGH_VOLATILITY only
   - SPE must beat this baseline by >2× at 30m horizon

5. **Missed-fill / adverse-selection stress**
   - 25% missed fill rate at 1bps maker cost must remain viable
   - Adverse selection penalty (worse fill on losing trades) must be modeled

---

## Hard Rules

| Rule | Status |
|------|--------|
| No live trading | ✅ Enforced |
| No execution enablement | ✅ Enforced |
| No threshold tuning | ✅ Enforced |
| No feature addition | ✅ Enforced |
| Observe + log only | ✅ Active |

---

## What This Means

SPE is not a fraud. The layer structure correctly identifies high-pressure
conditions where SHORT directional edge exists. But the edge is:

- Narrow (SHORT + HIGH_VOLUME + HIGH_VOLATILITY only)
- Fragile (outlier-dependent, time-unstable)
- Possibly regime-specific (bearish data period)

The honest path is observation. Log SPE_SHORT_STRESS events in the live market.
Accumulate unseen data. Re-validate when sufficient new data exists.

If the edge survives walk-forward + unseen data + outlier removal + same-regime
baseline, it can be reconsidered. Until then, it remains a candidate — nothing more.

---

*Report the truth.*
