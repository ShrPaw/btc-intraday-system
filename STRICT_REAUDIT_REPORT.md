# STRICT RE-AUDIT REPORT — setup_validation_engine.py
**Date:** 2026-04-26 07:05 GMT+8
**Auditor:** Quantitative Research + Backtest Integrity
**Engine:** `setup_validation_engine.py` (Phases 1–14)
**Data:** 642 setups, 4 symbols, Jun 2025 – Apr 2026

---

## 1. Executive Verdict

**PARTIALLY FIXED — TWO P0 REMAIN**

| Issue | Previous | Current | Verdict |
|-------|----------|---------|---------|
| Worst-case intracandle | P0 | **Still P0** (dead ambiguity code) | NOT FIXED |
| Random baselines | P1 | **P0** (all baselines empty) | NOT FIXED |
| Next-open entry | P0 | Fixed | FIXED |
| Rolling/past-only thresholds | P0 | Fixed | FIXED |
| HTF closed-candle | P1 | Fixed | FIXED |
| Structural stop validity | P0 | Fixed | FIXED |
| Outcome logic in R | P0 | Fixed | FIXED |

---

## 2. Critical Fixes Table

| Issue | Prev Severity | Status | Curr Severity | Evidence | Remaining Risk |
|-------|--------------|--------|---------------|----------|----------------|
| Worst-case intracandle | P0 | **NOT FIXED** | **P0** | Ambiguity block is dead code (see §3) | TP can count as win when SL touched in same candle |
| Random baselines | P1 | **NOT FIXED** | **P0** | All baselines produce 0 rows (see §4) | No baseline comparison exists; edge claim unvalidated |
| Next-open entry | P0 | Fixed | Fixed | `entry_price = df.iloc[entry_idx]["open"]` confirmed | None |
| Rolling thresholds | P0 | Fixed | Fixed | All quantiles use `.rolling().quantile().shift(1)` | None |
| HTF closed-candle | P1 | Fixed | Fixed | `rhs[c] = rhs[c].shift(1)` in merge_asof_feature | None |
| Structural stop | P0 | Fixed | Fixed | `window = df.iloc[start:signal_idx + 1]` — no future bars | None |
| Outcome in R | P0 | Fixed | Fixed | No equity/leverage in output; pure R-multiples | None |

---

## 3. P0 Status: Worst-Case Intracandle

### The Bug

The `track_setup_outcome()` function executes in this order per bar:

```
STEP 1: Check TP → if touched, set hit_XR = True
STEP 2: Check SL → if touched, set sl_hit = True
STEP 3: Ambiguity check → if SL AND TP and not hit_XR → set ambiguous, hit_XR = True
STEP 4: if sl_hit → break
```

**The problem:** Step 1 fires before Step 3. If TP1 and SL are both touched in the same candle (first time for TP1), Step 1 sets `hit_1R = True`. Then Step 3's condition `not hit_1R` is `False` — the ambiguity block **never fires**.

### Exact Code Evidence

```python
# STEP 1 — TP check runs FIRST
for n in TP_R_MULTIPLES:
    tp_p = tp_prices[n]
    if direction == 1:
        tp_touched = high >= tp_p
    else:
        tp_touched = low <= tp_p
    if tp_touched:
        if n == 1 and not hit_1R:
            hit_1R = True          # ← TP COUNTED AS WIN
            time_to_1R = j - entry_idx

# STEP 2 — SL check runs SECOND
if direction == 1:
    sl_touched = low <= stop_price
else:
    sl_touched = high >= stop_price
if sl_touched and not sl_hit:
    sl_hit = True
    time_to_SL = j - entry_idx

# STEP 3 — Ambiguity check runs THIRD (DEAD CODE for first-time TP)
if direction == 1:
    if low <= stop_price:
        if high >= tp_prices[1] and not hit_1R:  # ← never True: hit_1R already set in Step 1
            ambiguous_1R = True
            hit_1R = True          # ← would set TP as hit even in ambiguity block!
```

### Why It's Dead Code

For the ambiguity block to fire, `not hit_1R` must be `True`. But if `high >= tp_prices[1]` is `True` in Step 3, then Step 1's `high >= tp_p` is also `True` (same condition, same bar), so Step 1 already set `hit_1R = True`. The ambiguity block can never execute.

**Even if it could fire**, it would set `hit_1R = True` — counting TP as a win when SL was also touched. The correct behavior is `hit_1R = False`.

### Correct Implementation

```python
# Correct: check SL FIRST, then TP, then resolve ambiguity
sl_touched = (low <= stop_price) if direction == 1 else (high >= stop_price)

tp_touched_any = False
for n in TP_R_MULTIPLES:
    tp_touched = (high >= tp_prices[n]) if direction == 1 else (low <= tp_prices[n])
    if tp_touched and not hit_XR[n]:
        if sl_touched:
            # Same candle: ambiguous → count as LOSS
            ambiguous_XR[n] = True
            # Do NOT set hit_XR = True
        else:
            hit_XR[n] = True
            time_to_XR[n] = j - entry_idx

if sl_touched:
    sl_hit = True
    time_to_SL = j - entry_idx
    break
```

### Data Impact

In the current 642-setup dataset:
- **0 same-candle TP+SL events** detected (time_to_1R == time_to_SL = 0 matches)
- All 188 setups with both hit_1R=True and sl_hit=True have time_to_1R < time_to_SL (TP on earlier candle, SL on later candle — **correct**)

**Classification: Latent P0.** The bug exists in the code but is not triggered by this dataset. However, it will produce incorrect results if any future candle touches both TP and SL for the first time.

### Answers

- **Is worst-case TP/SL fixed?** NO. The ambiguity block is dead code.
- **Can TP still count when SL is touched in same candle?** YES. If both are first-time touches, TP is counted as win.
- **Can TP count after stop has already occurred?** No — the loop breaks after sl_hit. This is correct.

---

## 4. Baseline Status

### The Bug

`build_random_baselines()` receives a sliced DataFrame (`df_test`) with non-sequential index labels from the original full dataset.

```python
sig_idx = df.index[mask][0]     # Returns label from original df, e.g., 53377
entry_idx = sig_idx + 1          # 53378
if entry_idx >= len(df):         # 53378 >= 38272 → TRUE → SKIP
    continue
```

**Every single baseline entry is skipped** because index labels (50544–88815) exceed `len(df_test)` (38272).

### Proof

```
BTCUSDT test period:
  df_test len:          38,272
  df_test index range:  50,544 – 88,815
  Baseline 1 attempts:  55
  Baseline 1 skipped:   55 (100%)
  Baseline 1 valid:      0 (0%)
```

### Output Confirmation

Phase 9 output shows **only the SYSTEM row** — zero baseline rows:

```
Baseline                            N    1R%    2R%    3R%   Exp1R   Exp2R   Exp3R
--------------------------------------------------------------------------------------------------
SYSTEM (actual)                   262 51.5% 34.4% 24.8% +0.038 +0.065 +0.050
```

No random baselines are displayed because none were generated.

### Additional Baseline Issue

Baselines 2, 3, 4 filter with `df[df["setup_type"] != "none"]`, restricting sampling to the strategy's pre-filtered setup universe. This tests "random within the strategy's filtered universe" rather than "random market baseline."

### Answers

- **Is there a full-dataset random baseline?** NO. All baselines produce 0 rows.
- **Is same-timestamp random direction baseline implemented?** Code exists but produces 0 results.
- **Is same-regime baseline implemented without setup_type filtering?** NO — it filters on setup_type.
- **Is same-session baseline implemented?** NO — no session baseline exists.
- **Are all baselines using identical stop/TP/outcome logic?** Would be yes, but they produce no data.

### The SETUP_AUDIT_REPORT.md baseline numbers are INVALID

The report states:
```
| Same-time random dir | ~262 | ~50% | ~+0.000 | ~-0.500 | ~0.50R |
```

The `~` prefix and approximate values confirm these were **manually estimated**, not computed by the engine. The engine produces zero baseline data.

---

## 5. Metrics After Fixes

These are the VALIDATED metrics (from the engine output, which IS mechanically correct for the system setups themselves):

| Metric | All Periods | Test Period |
|--------|------------|-------------|
| Total setups | 642 | 262 |
| Hit 1R | 50.9% | 51.5% |
| Hit 2R | 34.9% | 34.4% |
| Hit 3R | 24.5% | 24.8% |
| Hit 4R | 17.6% | — |
| Exp 1R | +0.028R | +0.038R |
| Exp 2R | +0.076R | +0.065R |
| Exp 3R | +0.033R | +0.050R |
| Median MFE | 1.02R | 1.03R |
| Median MAE | 1.07R | 1.06R |
| Ambiguity rate | 0.0% | 0.0% |
| Expired rate | 0.9% | — |

**Note on ambiguity rate = 0%:** This does NOT mean the bug is fixed. It means no same-candle TP+SL events occurred in this dataset. The code would produce incorrect results if they did.

---

## 6. Group Breakdown

### By Direction

| | N | Hit 1R | Exp 1R | Exp 2R |
|---|---|--------|--------|--------|
| LONG | 275 | 51.6% | +0.055R | +0.102R |
| SHORT | 367 | 50.4% | +0.008R | +0.057R |

### By Confidence Tier

| | N | Hit 1R | Exp 1R | Exp 2R |
|---|---|--------|--------|--------|
| MILD | 343 | 51.3% | +0.026R | +0.117R |
| MID | 88 | 50.0% | +0.000R | +0.125R |
| HIGH | 136 | 51.5% | +0.074R | -0.074R |
| PREMIUM | 75 | 49.3% | -0.013R | +0.107R |

### By Direction + Confidence

| | N | Hit 1R | Exp 1R | Exp 2R |
|---|---|--------|--------|--------|
| LONG/MILD | 148 | 50.0% | +0.000R | +0.068R |
| LONG/MID | 34 | 52.9% | +0.059R | +0.324R |
| LONG/HIGH | 56 | 55.4% | +0.214R | +0.054R |
| LONG/PREMIUM | 37 | 51.4% | +0.027R | +0.108R |
| SHORT/MILD | 195 | 52.3% | +0.046R | +0.154R |
| SHORT/MID | 54 | 48.1% | -0.037R | +0.000R |
| SHORT/HIGH | 80 | 48.8% | -0.025R | -0.162R |
| SHORT/PREMIUM | 38 | 47.4% | -0.053R | +0.105R |

### By Month

| Month | N | Hit 1R | Exp 1R |
|-------|---|--------|--------|
| 2025-06 | 38 | 76.3% | +0.526R |
| 2025-07 | 88 | 55.7% | +0.136R |
| 2025-08 | 55 | 45.5% | -0.091R |
| 2025-09 | 49 | 49.0% | +0.020R |
| 2025-10 | 76 | 50.0% | +0.000R |
| 2025-11 | 74 | 36.5% | -0.270R |
| 2025-12 | 35 | 51.4% | +0.029R |
| 2026-01 | 54 | 57.4% | +0.148R |
| 2026-02 | 79 | 46.8% | -0.063R |
| 2026-03 | 78 | 53.8% | +0.103R |
| 2026-04 | 16 | 43.8% | -0.125R |

### By HTF Regime

| | N | Hit 1R | Exp 1R |
|---|---|--------|--------|
| Bullish | 169 | 53.3% | +0.101R |
| Bearish | 193 | 57.0% | +0.140R |
| Neutral | 280 | 45.4% | -0.093R |

### By Session

| | N | Hit 1R | Exp 1R |
|---|---|--------|--------|
| Asian | 169 | 52.1% | +0.077R |
| European | 281 | 50.5% | +0.011R |
| US | 192 | 50.5% | +0.010R |

---

## 7. Baseline Comparison

**NO VALID BASELINE COMPARISON CAN BE MADE.**

The engine produces zero baseline data. The SETUP_AUDIT_REPORT.md baseline numbers were manually estimated, not computed. The claim "system edge over random (Exp1R): +0.038R" is unsubstantiated by the engine.

To properly compare:
- Fix the index bug in `build_random_baselines()` (use `df.reset_index(drop=True)` or positional indexing)
- Add a full-dataset baseline (sample from all bars with sufficient indicator history, not just setup_type != none)
- Add a same-session baseline
- Remove setup_type filtering from baselines 2, 3, 4

---

## 8. Signal-to-Entry Gap (BTCUSDT)

| Metric | Value |
|--------|-------|
| Mean | -0.0119% |
| Median | -0.0076% |
| Std | 0.1311% |
| 5th percentile | -0.2483% |
| 95th percentile | +0.1938% |
| Max | +0.3556% |
| Min | -0.4900% |

LONG mean gap: +0.0470% (positive = entry above signal close, slight slippage against)
SHORT mean gap: -0.0402% (negative = entry below signal close, slight slippage against)

Gap is small and symmetrical. No systematic bias.

---

## 9. Structural Stop Diagnostics

| Metric | Value |
|--------|-------|
| Mean stop distance | 0.787% |
| Median stop distance | 0.657% |
| 5th percentile | 0.430% (near floor 0.400%) |
| 95th percentile | 1.564% |
| Min | 0.400% (floor) |
| Max | 1.986% (near cap 2.000%) |

Stops are derived exclusively from pre-signal data (swing high/low over 4-bar lookback + EMA20 for RSI_TREND). No future bars used.

---

## 10. Final Classification

### System-Level

**PROMISING BUT MECHANICALLY DEFECTIVE**

The setup detection logic appears sound. The R-multiple framework is correct. The rolling quantile fixes are verified. However, two P0 bugs prevent validation:

1. The intracandle ambiguity handling is dead code — TP can be counted as a win when SL is also touched in the same candle.
2. All random baselines produce zero results — no statistical edge validation is possible.

### Per-Category Classification

| Category | Classification | Reasoning |
|----------|---------------|-----------|
| LONG/MILD | **INCONCLUSIVE** | Exp1R=+0.000R (all), +0.065R (test). Near zero. Cannot validate without baselines. |
| LONG/MID | **PROMISING BUT FRAGILE** | Exp1R=+0.059R, Exp2R=+0.324R. Positive but n=34, thin sample. |
| LONG/HIGH | **PROMISING BUT FRAGILE** | Exp1R=+0.214R, n=56. Best LONG tier but small sample. |
| LONG/PREMIUM | **WEAK** | Exp1R=+0.027R, n=37. Marginal. |
| SHORT/MILD | **PROMISING BUT FRAGILE** | Exp1R=+0.046R, n=195. Largest SHORT group, slightly positive. |
| SHORT/MID | **WEAK** | Exp1R=-0.037R, n=54. Negative expectancy. |
| SHORT/HIGH | **DISABLE** | Exp1R=-0.025R, n=80. Negative at 2R (-0.162R). |
| SHORT/PREMIUM | **DISABLE** | Exp1R=-0.053R, n=38. Negative. |
| Overall System | **INCONCLUSIVE** | Exp1R=+0.028R. Thin edge. Cannot validate without working baselines. |

---

## 11. Required Fixes Before Re-Validation

### P0 — Must Fix

1. **Intracandle ambiguity:** Reorder checks so SL is evaluated before TP. If both are touched in same candle, count as SL (hit_XR = False, ambiguous_XR = True). Remove `hit_XR = True` from ambiguity block.

2. **Baseline index bug:** In `build_random_baselines()`, use `df.reset_index(drop=True)` on the input `df`, or use positional indexing (`df.index.get_loc(sig_idx)`) instead of label-as-position. Verify baselines produce >0 rows.

3. **Baseline scope:** Add a full-dataset random baseline (sample from all bars with indicator warmup, not just `setup_type != "none"`). Add a same-session baseline. Remove `setup_type` filtering from baselines 2, 3, 4.

### P1 — Should Fix

4. **Report ambiguity in expectancy:** If ambiguous flags exist, report them separately. Treat ambiguous as loss in expectancy calculation. Currently `ambiguous_rate = 0%` but this is because the bug prevents detection.

---

*No hype. No optimization. No parameter tuning. Code evidence above. The engine needs two P0 fixes before the statistical edge can be evaluated.*
