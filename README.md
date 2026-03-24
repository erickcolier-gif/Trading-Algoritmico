# NAS100 AI Scalper — PropXP Trading System

An AI-powered NASDAQ scalping system with MetaTrader 5 integration, Claude AI analysis, real-time dashboard, and PropXP prop firm compliance.

---

## Features

- **NASDAQ (NAS100/US100) 4-minute scalping** — enforces 4-min minimum hold time
- **Claude AI (claude-opus-4-5)** for trade analysis with extended thinking
- **MetaTrader 5** integration via Python library (with mock fallback)
- **PropXP compliance** — daily loss limits, drawdown limits, position sizing
- **Real-time dashboard** with TradingView Lightweight Charts candlesticks
- **News sentiment** from Google News RSS feeds
- **FastAPI backend** with WebSocket for live price updates
- **Three PropXP accounts**: Funded $10k + 2x Challenge $3k

---

## Quick Start

### 1. Prerequisites

- Python 3.11 or higher
- Windows OS (required for MetaTrader 5)
- MetaTrader 5 terminal installed and running
- PropXP MT5 account credentials
- Anthropic API key (get one at console.anthropic.com)

### 2. Install MetaTrader 5

Download and install MT5 from your PropXP broker portal. Log into your account in the terminal before starting the system.

### 3. Clone / Download

Place all files in:
```
C:\Users\YourName\Desktop\AI Agent Trading\
```

### 4. Configure Environment

Copy the example env file and fill in your credentials:
```bash
cd "C:\Users\YourName\Desktop\AI Agent Trading"
copy .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
MT5_LOGIN=12345678
MT5_PASSWORD=your_mt5_password
MT5_SERVER=PropXP-Server
ACTIVE_ACCOUNT=funded_10k
```

**ACTIVE_ACCOUNT options:**
- `funded_10k` — Funded $10,000 account
- `challenge_3k_1` — Challenge $3,000 account #1
- `challenge_3k_2` — Challenge $3,000 account #2

### 5. Install Python Dependencies

```bash
cd "C:\Users\YourName\Desktop\AI Agent Trading\backend"
pip install -r requirements.txt
```

### 6. Run the Backend

```bash
cd "C:\Users\YourName\Desktop\AI Agent Trading\backend"
python start.py
```

The API will start at `http://localhost:8000`

You should see:
```
INFO: MT5 connection established.
INFO: Active symbol: NAS100
INFO: WebSocket broadcast loop started.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 7. Open the Dashboard

Open `index.html` directly in your browser:
```
C:\Users\YourName\Desktop\AI Agent Trading\index.html
```

Or double-click the file. No web server needed — it connects directly to the backend.

---

## PropXP Account Rules (Built-in)

| Account | Balance | Max Daily Loss | Max Drawdown | Profit Target |
|---------|---------|----------------|--------------|---------------|
| Funded $10k | $10,000 | $500 (5%) | $1,000 (10%) | None |
| Challenge $3k #1 | $3,000 | $150 (5%) | $300 (10%) | $300 (10%) |
| Challenge $3k #2 | $3,000 | $150 (5%) | $300 (10%) | $300 (10%) |

The system automatically:
- Halts trading when daily loss limit is reached
- Warns at 80% of limits
- Blocks trades that exceed risk thresholds
- Enforces minimum 4-minute hold time (PropXP rule)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/account` | Account info + risk status |
| GET | `/api/market` | Current price + 50 OHLCV candles |
| POST | `/api/analyze` | Run full AI analysis |
| GET | `/api/positions` | Open positions |
| POST | `/api/trade` | Place a trade |
| DELETE | `/api/trade/{ticket}` | Close a position |
| GET | `/api/news` | Recent NASDAQ news |
| GET | `/health` | Health check |
| WS | `/ws` | Real-time price updates |

---

## Demo / Mock Mode

If MT5 is not installed or credentials are wrong, the system runs in **mock mode**:
- Price data is simulated with realistic NAS100 movement
- Orders are tracked in memory
- All dashboard features work normally

If no Anthropic API key is set, AI analysis returns **mock recommendations** based on technical signals.

---

## Project Structure

```
AI Agent Trading/
├── index.html                    # Trading dashboard (open in browser)
├── .env.example                  # Environment template
├── .env                          # Your credentials (create this)
├── README.md
└── backend/
    ├── requirements.txt
    ├── start.py                  # Run this to start the server
    ├── main.py                   # FastAPI app
    ├── config.py                 # Configuration & PropXP rules
    └── modules/
        ├── __init__.py
        ├── mt5_connector.py      # MT5 connection & orders
        ├── market_analysis.py    # Technical indicators
        ├── news_analyzer.py      # News sentiment
        ├── ai_advisor.py         # Claude AI integration
        └── risk_manager.py       # PropXP risk management
```

---

## Troubleshooting

**MT5 not connecting:**
- Ensure MetaTrader 5 terminal is open and logged in
- Verify server name matches exactly (check in MT5 terminal)
- Run MT5 as administrator if needed

**No NAS100 symbol found:**
- The system tries: NAS100, US100, USTEC, NDX100
- In MT5, search for NASDAQ in the Market Watch and add the symbol
- Update `MT5_SERVER` in `.env` to match your broker

**AI giving mock responses:**
- Check your `ANTHROPIC_API_KEY` in `.env`
- Ensure the key has API access (not just Claude.ai)
- Check API credits at console.anthropic.com

**Dashboard shows "OFFLINE":**
- Make sure backend is running (`python start.py`)
- Check browser console for CORS errors
- Ensure port 8000 is not blocked by firewall

---

## Risk Disclaimer

This software is for educational purposes. Trading NAS100/NASDAQ involves significant financial risk. Always:
- Test thoroughly in demo accounts first
- Never risk more than you can afford to lose
- Follow your prop firm's rules carefully
- The AI recommendations are not financial advice
