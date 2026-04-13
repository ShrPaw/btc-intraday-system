# BTC Intraday Trading System — Project Context
## Last Updated: 2026-04-13 06:47 GMT+8

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
├── btc_intraday_system.py    # Main backtester (1200+ lines)
├── fetch_btc_data.py          # Data fetcher from Binance (multi-asset)
├── live_signal_bot.py         # Live Telegram signal bot v2 (NEW)
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
- Structure gate: EMA reclaim + BOS within 3-bar window

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

## CONFIDENCE TIERS (UPDATED v5)

| Tier     | Threshold | Risk %  | Status |
|----------|-----------|---------|--------|
| NO_TRADE | < 0.72    | 0%      | Skip   |
| MILD     | 0.72-0.78 | 2.50%   | ACTIVE (new in v4) |
| MID      | 0.78-0.80 | 3.00%   | DISABLED (WR 52.9%, PF 0.723) |
| PREMIUM  | 0.80-0.82 | 3.25%   | ACTIVE (lowered from 0.85) |
| HIGH     | 0.82-0.88 | 3.50%   | ACTIVE (lowered from 0.90) |
| ELITE    | 0.88+     | 4.00%   | ACTIVE |

---

## ADX REGIME FILTER (UPDATED v5)

ADX is used as a **confidence modifier**, NOT a hard gate:
- ADX > 25 (strong trend): +3% confidence boost
- ADX 15-25 (normal): no change
- ADX < 15 (weak/choppy): -3% confidence penalty
- Computed on 4H bars, merged to 5m via merge_asof_feature
- Config: ENABLE_ADX_FILTER, ADX_TRENDING_BONUS=25, ADX_CHOPPY_PENALTY=15

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

### Stop Loss (UPDATED v6):
- Structural: based on recent swing low/high (**4-bar lookback**, was 6)
- For RSI_TREND: also factors in EMA20
- Minimum floor: **0.4%** (was 0.5%)

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
- COOLDOWN_BARS_5M: 3 (was 6)
- Symbol-specific slippage: BTC 0.05%, ETH 0.08%, SOL 0.12%, XRP 0.15%
- MAX_NOTIONAL_FRACTION: 10.0 (position sizing cap)
- Volume filter: skip bottom 30% dead candles

### Live Bot:
- Bot: @ShrPawsPsudoBot
- Rolling window: 10,080 1m candles (7 days)
- Check interval: 60s
- Default leverage: 5x
- Position monitor: every 15s
- Mode: PAPER (set PAPER_MODE=false for live)
- Server IP: <YOUR_SERVER_IP>

---

## BACKTEST RESULTS — FINAL (v6, 2026-04-13)

### Walk-Forward: Train Jun-Nov 2025, Test Dec 2025-Apr 2026

| Version | Trades | WR | PF | PnL | Avg Stop | TSL R:R | SL R:R |
|---------|--------|-----|-----|------|----------|---------|--------|
| v5 (6-bar, 0.5% min) | 331 | 79.8% | 2.51 | +$47,571 | 0.85% | 0.78R | -1.16R |
| **v5.5/v6 (4-bar, 0.4% min)** | **337** | **77.7%** | **2.32** | **+$49,987** | **0.78%** | **0.85R** | **-1.19R** |
| v6 extreme (3-bar, 0.3% min) | 341 | 76.0% | 2.29 | +$53,901 | 0.73% | 0.94R | -1.22R |

**Current version: v5.5 (4-bar lookback, 0.4% min stop)**

### TEST PERIOD MONTHLY (v5.5):
- Dec 2025: +$7,380
- Jan 2026: +$6,574
- Feb 2026: +$8,410
- Mar 2026: +$17,781
- Apr 2026 (12 days): +$7,425
- **All 5 months profitable, all 4 symbols profitable**

### By Symbol (Test):
- BTC: 83 trades, +$18,074
- ETH: 85 trades, +$10,358
- SOL: 85 trades, +$12,117
- XRP: 78 trades, +$7,022

---

## CODE CHANGE LOG

### v1 — Original reverse-engineered code
- BTC-only backtester, RSI_SCALP + RSI_TREND, MID/HIGH/ELITE tiers

### v2 — Signal analysis updates
- TP2 differentiated by strategy, Stage tracking, CFX indicator

### v3 — Multi-asset + PREMIUM tier
- Multi-asset (BTC/ETH/SOL/XRP), PREMIUM tier, MID disabled

### v4 — 5 critical fixes (2026-04-13)
- ADX regime filter (initially as hard gate)
- Wider confidence bands (MILD tier, lower thresholds, cooldown 6→3)
- CHZUSDT added (later removed)
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

### v7 — Live Signal Bot v2 (2026-04-13)
- Built `live_signal_bot.py` — Telegram signal bot with inline trade buttons
- Binance WebSocket for live 1m candles (BTC/ETH/SOL/XRP)
- Multi-timeframe resampling + exact same backtester logic
- Telegram inline keyboard: execute trade with leverage picker (3x/5x/10x)
- Position monitor: auto TP1/TP2/TP3/TP4 alerts, break-even, trailing stop
- ccxt integration for live Binance Futures execution
- Paper mode by default (set PAPER_MODE=false for live)
- .env for credentials (gitignored), .env.example template
- Bot Telegram: @ShrPawsPsudoBot

---

## LIVE SIGNAL BOT — SETUP STATUS

### ✅ Done:
- Telegram bot created: @ShrPawsPsudoBot
- Bot token + chat ID configured in .env
- Bot running in paper mode (signals + buttons working)
- Server IP: <YOUR_SERVER_IP>

### ✅ Deployed (2026-04-14):
- Bot running as systemd service on Alibaba Cloud ECS
- Server IP: 47.84.102.150
- Service: `btc-signal-bot.service` (auto-restart enabled)
- Mode: LIVE (PAPER_MODE=false)
- Logs: `/root/.openclaw/workspace/btc-intraday-system/live_bot.log`
- Commands: `systemctl status btc-signal-bot`, `tail -f live_bot.log`, `systemctl restart btc-signal-bot`

### ⏳ Remaining (to go live):
1. Activate Binance Futures account:
   - Complete KYC if not done
   - Go to Derivatives → USDⓈ-M Futures → pass quiz + accept ToS
2. Create Binance API key with these restrictions:
   - Enable Reading ✅
   - Enable Futures ✅
   - **IP Restriction: add <YOUR_SERVER_IP>** (required for Futures API)
   - Disable Spot Trading ❌
   - Disable Withdrawals ❌
   - Disable Universal Transfer ❌
3. Add keys to .env:
   ```
   BINANCE_API_KEY=<paste key>
   BINANCE_API_SECRET=<paste secret>
   PAPER_MODE=false
   ```
4. Restart bot: `pkill -f live_signal_bot && cd btc-intraday-system && nohup python3 -u live_signal_bot.py >> live_bot.log 2>&1 &`
5. Fund Binance Futures wallet with USDT

### Bot Features (v2):
- Signal fires → Telegram message with entry/stop/TPs/confidence
- Inline buttons: ⚡ GO LONG/SHORT, leverage picker (3x/5x/10x), ❌ Pass
- Position monitor checks every 15s for TP/SL hits
- Auto alerts: TP1, TP2, TP3, TP4, break-even, trailing stop, stop loss
- Close buttons on active positions: Close 50% / Close All
- Cooldown: 3 × 5m bars (15 min) between signals per symbol

### Bot Config:
- 7-day rolling window (10,080 1m candles) for indicator warmup
- Signal check every 60 seconds
- Default leverage: 5x
- Risk per trade: 2.5-4% based on confidence tier
- Paper equity: $10,000

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

## OPERATOR PROFILE — BRYAN LEE

- GitHub: github.com/bryanmylee (84 repos, 94 followers)
- Location: Singapore (UTC+08:00)
- Current: ByteDance (TikTok) R&D — LLM content understanding
- Ex: Meta, DSO National Laboratories (Singapore defense cybersecurity)

### His Bot Architecture (inferred):
- Firebase Cloud Functions (asia-east2 region)
- Telegram webhook → handleUpdate.ts (command router)
- Firestore for state (positions, trades, sessions)
- Cloud Tasks for TP/SL/TSL monitoring
- TypeDI dependency injection pattern

---

## DEPLOYMENT STATUS (2026-04-14)

### Infrastructure:
- **Server:** Alibaba Cloud ECS (iZt4n45v2ngpm5cz1ml38cZ)
- **Public IP:** 47.84.102.150
- **OS:** Linux 6.8.0-100-generic (x64), Ubuntu-based, Python 3.12
- **Dependencies installed:** websocket-client, pandas, numpy, requests, ccxt (system-wide pip)

### Bot Service:
- **Service name:** btc-signal-bot.service (systemd)
- **Status:** Running ✅, enabled at boot
- **Mode:** LIVE (PAPER_MODE=false)
- **Log path:** /root/.openclaw/workspace/btc-intraday-system/live_bot.log
- **Working dir:** /root/.openclaw/workspace/btc-intraday-system
- **Commands:**
  - Status: `systemctl status btc-signal-bot`
  - Logs: `tail -f /root/.openclaw/workspace/btc-intraday-system/live_bot.log`
  - Restart: `systemctl restart btc-signal-bot`
  - Stop: `systemctl stop btc-signal-bot`

### Known Issues:
- **SSH blocked:** Alibaba Cloud security group needs port 22 inbound rule added
- User is setting up Alibaba Cloud account (was in Chinese, now switching to English)
- Need to add security group rule: TCP 22/22 from 0.0.0.0/0
- Also recommend adding ports 80 and 443 for future use

### Pending:
1. ⏳ User adding SSH rule in Alibaba Cloud security group
2. ⏳ User connecting via PuTTY from PC (has PuTTY installed)
3. ✅ .env configured with live credentials
4. ✅ Bot running in live mode
5. ✅ Code pushed to GitHub

## NEXT SESSION STARTUP

1. Read this file first
2. Check bot status: `systemctl status btc-signal-bot`
3. Check logs: `tail -50 /root/.openclaw/workspace/btc-intraday-system/live_bot.log`
4. If bot not running: `systemctl start btc-signal-bot`
5. SSH issue: User needs to add port 22 rule in Alibaba Cloud security group (47.84.102.150)
6. User connects via PuTTY from their Windows PC
