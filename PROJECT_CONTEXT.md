# BTC Intraday Trading System — Project Context
## Last Updated: 2026-04-13 05:00 GMT+8

---

## PROJECT OVERVIEW

We are reverse-engineering a crypto trading bot created by **Bryan Lee** (GitHub: bryanmylee).
The bot sends Telegram signals via @pokexrpbot (Poké Crypto group) and is a paper-trading
signal service. Our goal is to clone and improve the strategy logic.

**Repository:** https://github.com/ShrPaw/btc-intraday-system
**Original operator:** Bryan Lee (Singapore, ByteDance/TikTok R&D, ex-DSO National Labs cybersecurity)

---

## FILE STRUCTURE

```
btc-intraday-system/
├── btc_intraday_system.py    # Main backtester (1035+ lines)
├── fetch_btc_data.py          # Data fetcher from Binance (multi-asset)
├── data/features/
│   ├── btcusdt_1m.csv         # 146,700 candles (Jan 1 - Apr 12 2026)
│   ├── ethusdt_1m.csv         # 146,700 candles
│   ├── solusdt_1m.csv         # 146,701 candles
│   ├── xrpusdt_1m.csv         # 146,701 candles
│   ├── research_dataset.csv   # Legacy BTC-only dataset
│   ├── baseline_trades.csv    # Latest backtest output
│   └── baseline_setups.csv    # Latest setups output
└── PROJECT_CONTEXT.md         # THIS FILE
```

---

## TRADING STRATEGIES

### RSI_SCALP (Short-term Scalping)
Entry Logic:
- 4H RSI crosses above/below 50 midline (primary trigger)
- 15M RSI holds direction for 2+ candles (confirmation)
- 6H + 12H RSI must agree with direction (filter)
- "FRESH" = cross happened within last 4 bars
- Structure gate: EMA reclaim + BOS within 3-bar window

Confidence weights:
- 0.30 HTF alignment (4H/6H/12H RSI direction)
- 0.25 Freshness (bars since H4 cross)
- 0.20 Momentum (H4 slope + m15 slope + returns)
- 0.15 Realized vol rank
- 0.10 Structure (EMA reclaim + BOS)

Stage tracking (from live signals):
- Stage 1: bars_since_cross 0-1 (fresh cross, highest confidence)
- Stage 2: bars_since_cross 2-3 (still valid, decaying)
- Stage 3: bars_since_cross 4+ (aging, near expiry)

### RSI_TREND (Trend Following)
Entry Logic:
- 4H RSI shows sustained bullish/bearish momentum (>55 with positive slope for 2 bars)
- 5m EMA 20 reclaim (price crosses above 20 EMA)
- 5m BOS (Break of Structure - price breaks recent swing)
- 15M trend must align with direction
- 12H RSI must agree (>50 for long, <50 for short)

Confidence weights:
- 0.25 HTF alignment
- 0.20 Momentum
- 0.30 Structure
- 0.15 Realized vol rank
- 0.10 CFX (30M RSI alignment — hypothesis, needs validation)

CFX indicator (from live signals — "CFX 30M s=0/1"):
- Only shown on RSI_TREND, not always present
- Hypothesis: binary 30M RSI direction check
- s=1 = 30M RSI agrees with trade direction
- s=0 = 30M RSI disagrees
- Added as 10% soft weight in confidence
- NOT confirmed — needs more signal data to validate

---

## TAKE PROFIT STRUCTURE

### RSI_SCALP:
- TP1: +0.5% → Close 35%
- TP2: +0.7% → Close 35%
- TP3: +0.8% → Close 15%
- TP4: +9.0% → Close 15% (moon target, rarely hit)

### RSI_TREND:
- TP1: +0.5% → Close 35%
- TP2: +0.6% → Close 35% (DIFFERENT from SCALP — confirmed from live signals)
- TP3: +0.8% → Close 15%
- TP4: +9.0% → Close 15%

### Trailing Stop Loss:
- Activates after TP2 fills
- HIGH mode: 0.45% trail
- MID mode: 0.50% trail
- ELITE mode: 0.40% trail
- Ratchets up (LONG) / down (SHORT) only

### Break-Even:
- Armed after TP1 fills
- Offset: 0.02% above entry (LONG) / below entry (SHORT)

### Stop Loss:
- Structural: based on recent swing low/high (6-bar lookback)
- For RSI_TREND: also factors in EMA20
- Minimum floor: 0.5%

---

## CONFIDENCE TIERS

| Tier    | Threshold | Risk %  | Status |
|---------|-----------|---------|--------|
| NO_TRADE| < 0.72    | 0%      | Skip   |
| MID     | 0.72-0.80 | 3.00%   | DISABLED (WR 52.9%, PF 0.723 in backtest) |
| PREMIUM | 0.80-0.85 | 3.25%   | ACTIVE |
| HIGH    | 0.85-0.90 | 3.50%   | ACTIVE |
| ELITE   | 0.90+     | 4.00%   | ACTIVE |

MID was disabled because backtest showed it's a net loser. PREMIUM was added to fill the gap.

---

## BACKTEST RESULTS (as of 2026-04-13)

### Multi-Asset Walk-Forward (BTC, ETH, SOL, XRP)

| Period  | Trades | WR   | PF   | PnL       |
|---------|--------|------|------|-----------|
| FULL    | 73     | 89%  | 5.38 | +$14,613  |
| TRAIN   | 69     | 90%  | 5.97 | +$14,703  |
| TEST    | 4      | 75%  | 0.76 | -$90      |

### Walk-Forward Analysis:
- Train: Jan 1 - Feb 28, 2026 (strong trending market)
- Test: Mar 1 - Apr 12, 2026 (consolidation/chop)
- **OVERFIT RISK CONFIRMED**: System works in trending markets, fails in chop
- Test has too few trades (4) for statistical significance
- The strategy is regime-dependent

### By Symbol (FULL):
- BTC: 21 trades, avg +$244
- ETH: 15 trades, avg +$283
- SOL: 17 trades, avg +$131
- XRP: 20 trades, avg +$151

### By Strategy (FULL):
- RSI_SCALP: 23 trades, avg +$181
- RSI_TREND: 50 trades, avg +$209

### By Mode (FULL):
- ELITE: 1 trade, +$141
- HIGH: 39 trades, avg +$226
- PREMIUM: 33 trades, avg +$172

### By Exit (FULL):
- break_even_stop: 8 trades, avg +$39
- stop_loss: 8 trades, avg -$417
- trailing_stop: 57 trades, avg +$309

---

## OPERATOR PROFILE — BRYAN LEE

- GitHub: github.com/bryanmylee (84 repos, 94 followers)
- Website: bryanmylee.com
- Location: Singapore (UTC+08:00)
- Current: ByteDance (TikTok) R&D — LLM content understanding
- Ex: Meta, DSO National Laboratories (Singapore defense cybersecurity)
- Skills: TypeScript, Python, Go, Rust, React, Firebase/GCP

### His Bot Architecture (inferred from public repos):
- Firebase Cloud Functions (asia-east2 region)
- Telegram webhook → handleUpdate.ts (command router)
- Firestore for state (positions, trades, sessions)
- Cloud Tasks for TP/SL/TSL monitoring
- PubSub for market scanning
- TypeDI dependency injection pattern

### Key repos studied:
- meetwhen-telegram: Telegram bot on Firebase (architectural template)
- ts-rest-template: TypeDI DI container pattern
- wavefocus: Firebase monorepo with Cloud Tasks scheduling

---

## SIGNAL FORMAT (from Telegram screenshots)

### RSI_SCALP Signal:
```
#144 🟢 SOLUSDT | 📈 RSI Scalp | LONG
⏰ 2026-04-11 18:30 UTC | TF: 5m
📌 RSI LONG | FRESH 4H ↑
4H RSI: 61.4/56.1
Stage: 1
4H cross
15M RSI: 61.9/58.3 | holdCnt=2
6H RSI: 63.1/54.5
12H RSI: 59.
Entry: 85.1300
TP1: 85.5556 (+0.5%)
TP2: 85.6833 (+0.7%)
TP3: 85.7818 (+0.8%)
TP4: 92.7917 (+9.0%) 🎯
SL: 82.5761
R:R: 0.2R | Confidence: 80%
```

### RSI_TREND Signal:
```
#142 🟢 BTCUSDT | 📈 RSI Trend | LONG
⏰ 2026-04-11 12:40 UTC | TF: 5m
📌 RSI4H Trend LONG | 4H RSI 67.0/64.6 bullish sustained |
5m EMA reclaim + BOS | LTF trend 15M aligned |
CFX 30M s=0/1 | 12H RSI 77.
Entry: 72914.80
TP1: 73279.37 (+0.5%)
TP2: 73388.75 (+0.6%)  ← DIFFERENT FROM SCALP
TP3: 73498.12 (+0.8%)
TP4: 79477.13 (+9.0%) 🎯
SL: 70727.36
R:R: 0.2R | Confidence: 72%
```

### Live Stats (April 11 snapshot):
- RSI_SCALP: 93 trades, WR 86%, PnL +$1,481
- RSI_TREND: 51 trades, WR 82%, PnL +$55.71

### Exit Receipt Format:
```
TP #141 XRPUSDT [RSI_SCALP] SHORT
TP2 @ 1.341126 | PnL +71.63 | Rem 30% | SL 1.343151

CLOSE #141 XRPUSDT [RSI_SCALP] SHORT
TSL | Exit 1.343151 | Final PnL +55.10
```

---

## RED FLAGS (about the original bot)

1. Paper trading only — no real exchange
2. Self-reported stats — pinned message is manually editable
3. TP4 at +9% is rarely hit — most close via TSL at +1-2%
4. No verification link or third-party audit
5. VIP group is paid — classic signal group monetization
6. All groups are private/invite-only
7. "Confidence %" is self-assigned, not from verifiable model

---

## ORIGINAL 7 QUESTIONS — STATUS

1. TP2→TP3 gap (0.1%) — PARTIALLY FIXED (TP2 differentiated by strategy)
2. R:R asymmetry — RESOLVED (structural SL, not fixed 3%)
3. "RSI 61.4/56.1" second number — RESOLVED (previous candle RSI)
4. "STAGE 1" — RESOLVED (added to code as bars_since_cross binning)
5. RSI_TREND "sustained" — RESOLVED (concrete slope thresholds)
6. Early exit on 4H flip — NOT IMPLEMENTED (still open)
7. PnL inconsistency — RESOLVED (consistent dollar PnL)

---

## WHAT'S STILL NEEDED (PRIORITY ORDER)

### 1. ADX/Regime Filter (CRITICAL)
- Only trade when market is trending
- Add ADX(14) on 4H or daily timeframe
- Threshold: ADX > 25 = trending, ADX < 20 = skip
- This should fix the walk-forward test period failure
- Alternative: daily EMA slope (200 EMA rising = bull regime)

### 2. More Trades
- Current: 1-2 trades/week across 4 symbols
- Bryan's bot: 3-5 trades/day
- Options:
  - Lower PREMIUM threshold to 0.78
  - Add more symbols (CHZUSDT — trades frequently in his signals)
  - Reduce cooldown bars from 6 to 3
  - Add a "MILD" tier at 0.72-0.78 with smaller position size

### 3. More Symbols
- Currently: BTC, ETH, SOL, XRP
- Need: CHZUSDT (frequently traded in his signals)
- Consider: any trending altcoin with good volume

### 4. Realistic Slippage
- Current: 0.08% round-trip (too optimistic for alts)
- BTC: 0.05%, ETH: 0.08%, SOL: 0.12%, XRP: 0.15%
- Differentiate by symbol

### 5. Position Sizing Validation
- Current: $10K equity, 3.5% risk = $350 risk per trade
- On $70K BTC with 0.5% stop: $350 / 0.5% = $70K notional = 1 BTC
- Need to verify this matches Bryan's actual sizing

### 6. CFX Indicator Validation
- Need more signal screenshots showing CFX values
- Especially: s=1/1 (passing) vs s=0/1 (failing)
- Current hypothesis: 30M RSI alignment (unconfirmed)

### 7. Firebase/TypeScript Live Bot
- Clone architecture from meetwhen-telegram
- Telegram webhook on Firebase Cloud Functions
- Firestore for position/state management
- Cloud Tasks for TP/SL/TSL monitoring
- ccxt or direct Binance API for live data

### 8. Volume/Delta Features
- Currently computing delta_3, delta_6 but not using them
- Could add: volume spike detection, delta divergence
- High delta (aggressive buying) on entry could filter bad setups

### 9. Multi-Timeframe Confirmation Beyond RSI
- Currently: RSI only
- Consider: MACD divergence, VWAP, order book imbalance
- Bryan might be using ICT concepts (inner circle trader)

---

## CODE CHANGES LOG

### v1 — Original reverse-engineered code
- BTC-only backtester
- RSI_SCALP + RSI_TREND strategies
- Confidence engine with MID/HIGH/ELITE tiers
- 4-TP structure with trailing stop

### v2 — Signal analysis updates (2026-04-13)
- TP2 differentiated: SCALP +0.7%, TREND +0.6%
- Stage tracking added (1/2/3 for RSI_SCALP)
- CFX indicator: 30M RSI alignment as 10% soft weight
- Walk-forward validation framework

### v3 — Multi-asset + PREMIUM tier (2026-04-13)
- Multi-asset support (BTC, ETH, SOL, XRP)
- PREMIUM tier (0.80-0.85) added between MID and HIGH
- MID disabled (confirmed loser in backtest)
- Volume filter (skip bottom 30% dead candles)
- Hour filter framework (currently all hours)
- Multi-asset walk-forward validation

---

## GITHUB TOKEN NOTE
A GitHub token was used for pushing. **ROTATE THIS TOKEN** after session.
URL: https://github.com/settings/tokens

---

## NEXT SESSION STARTUP

1. Read this file first
2. Check latest backtest results
3. Start with ADX regime filter implementation
4. Then add CHZUSDT data
5. Then work on more trades (lower thresholds, reduce cooldown)
