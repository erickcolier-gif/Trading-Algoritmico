"""
Configuration for AI Agent Trading System.
Real-time NASDAQ 100 data via Finnhub + yfinance. AI analysis via OpenAI.
"""
import os
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=_env_path)

# === Symbols ===
SYMBOL = os.getenv("SYMBOL", "^NDX")   # NASDAQ-100 Index = NAS100 / US Tech 100 (Yahoo Finance)
VIX = "^VIX"                           # Volatility Index (Yahoo Finance)

# === API Keys ===
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Groq — FREE (primary AI). Register at groq.com, 30 req/min free
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI — paid (secondary fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# === MetaTrader 5 ===
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "NAS100")   # Symbol as it appears in your broker

# === Capital ===
CAPITAL = float(os.getenv("CAPITAL", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PCT", "1.0"))
MIN_RISK_REWARD = 1.5

# === Server ===
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# === Technical Indicators ===
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
STOCH_K = 14
STOCH_D = 3

# === Volume Analysis ===
VOLUME_SMA_PERIOD = 20
VOLUME_SPIKE_MULT = 2.0     # 2x average = institutional spike

# === SMC (Smart Money Concepts) ===
SWING_LOOKBACK = 10
OB_IMPULSE_MULT = 1.5       # Impulse must be 1.5x avg candle size
FVG_MIN_PCT = 0.001         # Min FVG size = 0.1%

# === Pattern Matching ===
PATTERN_WINDOW = 20
PATTERN_LOOKBACK = 1000
PATTERN_TOP_K = 10
PATTERN_FUTURE = 20

# === Update Intervals ===
WS_PRICE_INTERVAL = 2
WS_ANALYSIS_INTERVAL = 30
NEWS_INTERVAL = 300
DATA_CACHE_TTL = 30
