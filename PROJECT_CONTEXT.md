# BTC Intraday Trading System — Project Context
## Last Updated: 2026-04-14 15:11 GMT+8

---

## PROJECT OVERVIEW

Confidence-based BTC intraday trading system. Multi-asset (BTC/ETH/SOL/XRP), multi-timeframe
RSI analysis with dynamic confidence scoring, ADX regime filtering, and tiered risk management.
Includes a live Telegram signal bot with Binance Futures execution.

**Repository:** https://github.com/ShrPaw/btc-intraday-system

---

## FILE STRUCTURE

```
btc-intraday-system/
├── btc_intraday_system.py    # Main backtester (1200+ lines)
├── fetch_btc_data.py          # Data fetcher from Binance (multi-asset)
├── live_signal_bot.py         # Live Telegram signal bot v2
├── .env                       # Credentials (gitignored)
├── .env.example               # Template for credentials
├── .gitignore                 # Ignores .env, logs, __pycache__
├── data/features/
│   ├── btcusdt_1m.csv         # ~455K candles (Jun 2025 - Apr 2026)
│   ├── ethusdt_1m.csv         # ~455K candles
│   ├── solusdt_1m.csv         # ~455K candles
│   ├── xrpusdt_1m.csv         # ~455K candles
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
- Structure gate: EMA reclaim + BOS within **5-bar window**

Confidence weights:
- 0.30 HTF alignment (4H/6H/12H RSI direction)
- 0.25 Freshness (bars since H4 cross)
- 0.20 Momentum (H4 slope + m15 slope + returns)
- 0.15 Realized vol rank
- 0.10 Structure (EMA reclaim + BOS)

### RSI_TREND (Trend Following)
Entry Logic:
- 4H RSI shows sustained bullish/bearish momentum (>55 with positive slope for 2 bars)
- 5m EMA 20 reclaim + 5m BOS
- 15M trend must align with direction
- 12H RSI must agree (>50 for long, <50 for short)

Confidence weights:
- 0.25 HTF alignment
- 0.20 Momentum
- 0.30 Structure
- 0.15 Realized vol rank
- 0.10 CFX (30M RSI alignment)

---

## CONFIDENCE TIERS

| Tier     | Threshold | Risk %  | Status |
|----------|-----------|---------|--------|
| NO_TRADE | < 0.72    | 0%      | Skip   |
| MILD     | 0.72-0.78 | 2.50%   | ACTIVE |
| MID      | 0.78-0.80 | 3.00%   | **ACTIVE** (W5+MID backtested: 80.4% WR, PF 3.08) |
| PREMIUM  | 0.80-0.82 | 3.25%   | ACTIVE |
| HIGH     | 0.82-0.88 | 3.50%   | ACTIVE |
| ELITE    | 0.88+     | 4.00%   | ACTIVE |

---

## ADX REGIME FILTER

ADX is used as a **confidence modifier**, NOT a hard gate:
- ADX > 25 (strong trend): +3% confidence boost
- ADX 15-25 (normal): no change
- ADX < 15 (weak/choppy): -3% confidence penalty
- Computed on 4H bars, merged to 5m via merge_asof_feature
- Config: ENABLE_ADX_FILTER, ADX_TRENDING_BONUS=25, ADX_CHOPPY_PENALTY=15

---

## STRUCTURE GATE

Requires BOTH within N bars of each other:
1. **EMA20 reclaim** — price crosses above/below EMA20 on 5m
2. **BOS** — price breaks 12-bar swing high/low

**STRUCTURE_GATE_WINDOW = 5 bars** (~25 min, configurable in live bot)
- Backtested: window=5 significantly outperforms window=3
- The wider window captures legitimate setups the tighter window misses

---

## TAKE PROFIT STRUCTURE

### RSI_SCALP:
- TP1: +0.5% → Close 35%
- TP2: +0.7% → Close 35%
- TP3: +0.8% → Close 15%
- TP4: +9.0% → Close 15% (moon target, rarely hit)

### RSI_TREND:
- TP1: +0.5% → Close 35%
- TP2: +0.6% → Close 35% (DIFFERENT from SCALP)
- TP3: +0.8% → Close 15%
- TP4: +9.0% → Close 15%

### Stop Loss:
- Structural: based on **4-bar lookback** min(low)/max(high)
- For RSI_TREND: also factors in EMA20 (min/max with structural stop)
- Minimum floor: 0.4%

### Trailing Stop:
- Activates after TP2 fills
- HIGH mode: 0.45% trail
- MID/MILD mode: 0.50% trail
- ELITE mode: 0.40% trail

### Break-Even:
- Armed after TP1 fills
- Offset: 0.02% above/below entry

---

## CONFIG (CURRENT)

### Backtester:
- SYMBOLS: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
- INITIAL_EQUITY: $10,000
- COOLDOWN_BARS_5M: 3
- STRUCTURE_GATE_WINDOW: 5
- ALLOWED_MODES: MILD, MID, PREMIUM, HIGH, ELITE
- Symbol-specific slippage: BTC 0.05%, ETH 0.08%, SOL 0.12%, XRP 0.15%
- MAX_NOTIONAL_FRACTION: 10.0 (position sizing cap)
- Volume filter: skip bottom 30% dead candles

### Live Bot:
- Bot: WRAITH (@ShrPawsPsudoBot)
- TRADING_SYMBOLS: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT (all 4)
- STRUCTURE_GATE_WINDOW: 5 (configurable)
- ALLOWED_MODES: MILD, MID, PREMIUM, HIGH, ELITE
- Rolling window: 10,080 1m candles (7 days)
- Check interval: 60s
- Default leverage: 5x
- Position monitor: every 15s
- Mode: LIVE (PAPER_MODE=false)
- Diagnostic logging: per-symbol setup/trigger/confidence/rejection reason

---

## BACKTEST RESULTS

### Walk-Forward: Train Jun-Nov 2025, Test Dec 2025-Apr 2026

#### Structure Gate Window + MID Comparison (all 4 symbols combined):

| Config | Trades | WR | PF | PnL | Red Months | Change |
|--------|--------|-----|-----|------|------------|--------|
| W3 +NO_MID (old) | 337 | 77.7% | 2.32 | +$49,987 | 0/5 | baseline |
| W5 +NO_MID | 471 | 80.7% | 3.01 | +$114,295 | 0/5 | +129% |
| W3 +MID | 378 | 78.8% | 2.68 | +$70,651 | 0/5 | +41% |
| **W5 +MID (CURRENT)** | **509** | **80.4%** | **3.08** | **+$136,962** | **0/5** | **+174%** |

**All configurations profitable. Zero red months across all configs.**

#### Monthly (W5 +MID — current version):
- Dec 2025: 83 trades, WR 75%, +$8,891
- Jan 2026: 115 trades, WR 76%, +$17,926
- Feb 2026: 126 trades, WR 83%, +$24,633
- Mar 2026: 147 trades, WR 82%, +$52,505
- Apr 2026 (12 days): 38 trades, WR 92%, +$33,006

#### By Symbol (W5 +MID):
- BTC: 135 trades, WR 81%, +$68,706
- ETH: 133 trades, WR 77%, +$24,730
- SOL: 122 trades, WR 84%, +$27,772
- XRP: 119 trades, WR 80%, +$15,754

---

## CODE CHANGE LOG

### v1 — Initial backtester
- BTC-only backtester, RSI_SCALP + RSI_TREND, MID/HIGH/ELITE tiers

### v2 — Signal analysis updates
- TP2 differentiated by strategy, Stage tracking, CFX indicator

### v3 — Multi-asset + PREMIUM tier
- Multi-asset (BTC/ETH/SOL/XRP), PREMIUM tier, MID disabled

### v4 — 5 critical fixes (2026-04-13)
- ADX regime filter (initially as hard gate)
- Wider confidence bands (MILD tier, lower thresholds, cooldown 6→3)
- Position sizing cap (MAX_NOTIONAL_FRACTION)
- Symbol-specific slippage

### v5 — ADX as confidence modifier (2026-04-13)
- ADX changed from hard gate to soft +/-3% modifier
- Extended dataset: Jun 2025 → Apr 2026 (~455K candles/symbol)
- Walk-forward validation: Train Jun-Nov, Test Dec-Apr
- Test: 331 trades, 79.8% WR, PF 2.51, +$47,571

### v6 — Tighter stops (2026-04-13)
- Structural stop lookback: 6 → 4 bars
- Minimum stop floor: 0.5% → 0.4%
- Test: 337 trades, 77.7% WR, PF 2.32, +$49,987

### v7 — WRAITH Live Signal Bot (2026-04-13)
- Telegram signal bot with inline trade buttons
- Binance WebSocket for live 1m candles (BTC/ETH/SOL/XRP)
- Multi-timeframe resampling + exact same backtester logic
- Telegram inline keyboard: execute trade with leverage picker (3x/5x/10x)
- Position monitor: auto TP1/TP2/TP3/TP4 alerts, break-even, trailing stop
- ccxt integration for live Binance Futures execution
- Paper mode by default (set PAPER_MODE=false for live)

### v8 — Live deployment on Alibaba Cloud (2026-04-14)
- Systemd service on Alibaba Cloud ECS
- Live trading mode enabled
- Server IP: 8.219.194.199

### v9 — Bug fixes + flexibility (2026-04-14)
- **FIX:** TRADING_SYMBOLS was ["XRPUSDT"] only → now all 4 symbols
- **FIX:** Structural stop used current bar only → now 4-bar lookback + EMA20 for RSI_TREND
- **ENHANCE:** Structure gate window 3→5 bars (+129% PnL in backtest)
- **ENHANCE:** MID tier enabled (+174% PnL combined with W5)
- **ENHANCE:** Added per-symbol diagnostic logging (setup, trigger, confidence, rejection reason)
- **ENHANCE:** STRUCTURE_GATE_WINDOW configurable in live bot
- Test: 509 trades, 80.4% WR, PF 3.08, +$136,962 (0 red months)

---

## LIVE SIGNAL BOT — FEATURES

- Signal fires → Telegram message with entry/stop/TPs/confidence
- Inline buttons: ⚡ GO LONG/SHORT, leverage picker (3x/5x/10x), ❌ Pass
- Position monitor checks every 15s for TP/SL hits
- Auto alerts: TP1, TP2, TP3, TP4, break-even, trailing stop, stop loss
- Close buttons on active positions: Close 50% / Close All
- Cooldown: 3 × 5m bars (15 min) between signals per symbol
- 7-day rolling window (10,080 1m candles) for indicator warmup
- Signal check every 60 seconds
- Default leverage: 5x
- Risk per trade: 2.5-4% based on confidence tier
- **Diagnostic logging:** each check logs setup type, direction, confidence, mode, trigger status, H4 RSI, ADX, and rejection reason

---

## DEPLOYMENT STATUS

### Infrastructure:
- **Server:** Alibaba Cloud ECS
- **Public IP:** 8.219.194.199
- **OS:** Linux 6.8.0-100-generic (x64), Ubuntu-based, Python 3.12
- **Dependencies:** websocket-client, pandas, numpy, requests, ccxt

### Bot Service:
- **Service name:** btc-signal-bot.service (systemd)
- **Status:** Running ✅, enabled at boot
- **Mode:** LIVE (PAPER_MODE=false)
- **Log path:** `/root/.openclaw/workspace/btc-intraday-system/live_bot.log`
- **Working dir:** `/root/.openclaw/workspace/btc-intraday-system`

### Commands:
- Status: `systemctl status btc-signal-bot`
- Logs: `tail -f /root/.openclaw/workspace/btc-intraday-system/live_bot.log`
- Restart: `systemctl restart btc-signal-bot`
- Stop: `systemctl stop btc-signal-bot`

---

## NEXT SESSION STARTUP

1. Read this file first
2. Check bot status: `systemctl status btc-signal-bot`
3. Check logs: `tail -50 /root/.openclaw/workspace/btc-intraday-system/live_bot.log`
4. If bot not running: `systemctl start btc-signal-bot`
