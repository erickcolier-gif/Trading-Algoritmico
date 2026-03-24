"""
AI Agent Trading System — FastAPI Backend
Real-time NASDAQ 100 analysis: MetaTrader 5 + Finnhub (news) + OpenAI + SMC + Pattern Matching
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=_env_path)

import config as cfg
from modules.data_provider import DataProvider
from modules.technical_analysis import full_analysis
from modules.pattern_matcher import pattern_matcher
from modules.news_analyzer import analyze_news_with_ai, _fallback_analysis
from modules.mt5_connector import mt5_connector
from modules.adaptive_learner import learner as adaptive_learner
from modules.signal_manager import signal_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serial(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _serial(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serial(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    return obj


# ── WebSocket Manager ─────────────────────────────────────────────────────────

class WsManager:
    def __init__(self):
        self.connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)

    async def broadcast(self, data: dict):
        if not self.connections:
            return
        msg = json.dumps(_serial(data))
        dead = set()
        for ws in list(self.connections):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections.discard(ws)


ws_manager = WsManager()

# ── App State ─────────────────────────────────────────────────────────────────

provider = DataProvider(
    symbol=cfg.SYMBOL,
    finnhub_api_key=cfg.FINNHUB_API_KEY,
    mt5_connector=mt5_connector,
    mt5_symbol=cfg.MT5_SYMBOL,
)

_state = {
    "price": {},
    "analysis": {},
    "news": [],
    "news_analysis": {},
    "patterns": {},
    "vix": 20.0,
    "last_analysis_ts": None,
    "last_news_ts": None,
    # Multi-TF signals (5m / 1h), updated every analysis cycle
    "multi_signals": {"5m": {}, "1h": {}},
    # OHLCV cache keyed by interval
    "candle_cache": {},
}

_broadcast_task: Optional[asyncio.Task] = None
_analysis_task: Optional[asyncio.Task] = None
_news_task: Optional[asyncio.Task] = None
_signal_update_task: Optional[asyncio.Task] = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _df_to_candles(df) -> list:
    """Convert OHLCV DataFrame to JSON-serialisable candle list."""
    candles = []
    for i in range(len(df)):
        t = df.index[i]
        try:
            ts = int(t.timestamp()) if hasattr(t, 'timestamp') else i
        except Exception:
            ts = i
        candles.append({
            'time': ts,
            'open': round(float(df['open'].iloc[i]), 2),
            'high': round(float(df['high'].iloc[i]), 2),
            'low': round(float(df['low'].iloc[i]), 2),
            'close': round(float(df['close'].iloc[i]), 2),
            'volume': int(df['volume'].iloc[i]),
        })
    return candles


# ── Background Tasks ──────────────────────────────────────────────────────────

async def _price_loop():
    """Broadcast real-time price every WS_PRICE_INTERVAL seconds."""
    while True:
        try:
            await asyncio.sleep(cfg.WS_PRICE_INTERVAL)
            price = await provider.get_current_price()
            _state["price"] = price
            await ws_manager.broadcast({"type": "price", "data": price})
            # Feed current price to adaptive learner so it can resolve pending signals
            try:
                resolved = adaptive_learner.update_price(float(price.get('price', 0) or 0))
                if resolved:
                    stats = adaptive_learner.get_stats()
                    await ws_manager.broadcast({"type": "learner_update", "data": stats})
            except Exception as le:
                logger.debug(f"Learner price update error: {le}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Price loop error: {e}")
            await asyncio.sleep(2)


async def _analysis_loop():
    """Run full technical analysis every WS_ANALYSIS_INTERVAL seconds."""
    while True:
        try:
            await asyncio.sleep(cfg.WS_ANALYSIS_INTERVAL)
            await _run_analysis()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Analysis loop error: {e}")
            await asyncio.sleep(5)


async def _news_loop():
    """Fetch and analyze news every NEWS_INTERVAL seconds."""
    while True:
        try:
            await asyncio.sleep(cfg.NEWS_INTERVAL)
            await _run_news()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"News loop error: {e}")
            await asyncio.sleep(30)


async def _signal_update_loop():
    """Update active signal probabilities every 10 s based on current price + regime."""
    while True:
        try:
            await asyncio.sleep(10)
            price_data = _state.get("price", {})
            current_price = float(price_data.get("price", 0) or 0)
            if current_price <= 0:
                continue

            analysis   = _state.get("analysis", {})
            regime_obj = analysis.get("regime", {})
            session_obj= analysis.get("session", {})
            regime  = regime_obj.get("regime", "")
            session = session_obj.get("session", "")

            updated = signal_manager.update_price(current_price, regime, session)
            stats   = signal_manager.get_stats()

            await ws_manager.broadcast({
                "type": "signal_update",
                "data": {
                    "signals": updated,
                    "stats":   stats,
                    "price":   current_price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Signal update loop error: {e}")
            await asyncio.sleep(5)


async def _run_analysis():
    """Fetch OHLCV, run full analysis, broadcast result, then compute multi-TF signals."""
    try:
        df_5m = await provider.get_ohlcv("5m", 300)
        if df_5m.empty:
            return

        analysis = full_analysis(df_5m)

        # Cache 5m candles
        _state["candle_cache"]["5m"] = {
            "candles": _df_to_candles(df_5m),
            "ts": datetime.now(),
        }

        # Pattern matching
        df_hist = await provider.get_historical_for_patterns("5m", 1000)
        if not df_hist.empty:
            pattern_matcher.load_data(df_hist)
            patterns = pattern_matcher.find_similar_patterns(df_5m['close'].tail(100))
        else:
            patterns = pattern_matcher._empty_result()

        _state["analysis"] = analysis
        _state["patterns"] = patterns
        _state["last_analysis_ts"] = datetime.now().isoformat()

        await ws_manager.broadcast({
            "type": "analysis",
            "data": {
                "analysis": analysis,
                "patterns": patterns,
                "timestamp": _state["last_analysis_ts"],
            },
        })
        logger.info(f"Analysis complete — trend: {analysis.get('trend')} signals: {len(analysis.get('signals', []))}")

        # Register fresh signals with adaptive learner for outcome tracking
        try:
            for sig in analysis.get('signals', []):
                adaptive_learner.register_signal(sig)
        except Exception as le:
            logger.debug(f"Learner register error: {le}")

        # Feed fresh signals into the 30-min window manager
        try:
            regime_obj  = analysis.get('regime',  {})
            session_obj = analysis.get('session', {})
            managed = signal_manager.add_signals(
                analysis.get('signals', []),
                regime  = regime_obj.get('regime', ''),
                session = session_obj.get('session', ''),
            )
            await ws_manager.broadcast({
                "type": "managed_signals",
                "data": {
                    "signals": managed,
                    "stats":   signal_manager.get_stats(),
                    "regime":  regime_obj,
                    "session": session_obj,
                    "timestamp": _state["last_analysis_ts"],
                },
            })
        except Exception as se:
            logger.error(f"Signal manager error: {se}")

        # Compute 1h signals + cache candles (5m already broadcast above)
        await _run_multi_signals()

    except Exception as e:
        logger.error(f"Analysis run error: {e}")


async def _run_multi_signals():
    """Fetch 1h candles, run analysis, store signals + candle cache."""
    multi = {}
    for tf, bars in [("1h", 200)]:
        try:
            await asyncio.sleep(0.5)  # small pause between MT5 requests
            df = await provider.get_ohlcv(tf, bars)
            if df.empty:
                multi[tf] = {"signals": [], "trend": "UNKNOWN", "vwap_position": "—"}
                continue
            _state["candle_cache"][tf] = {"candles": _df_to_candles(df), "ts": datetime.now()}
            a = full_analysis(df)
            multi[tf] = {
                "signals": a.get("signals", []),
                "trend": a.get("trend", "UNKNOWN"),
                "vwap_position": a.get("vwap", {}).get("position", "UNKNOWN"),
            }
        except Exception as e:
            logger.error(f"Multi-signals {tf}: {e}")
            multi[tf] = {"signals": [], "trend": "UNKNOWN", "vwap_position": "—"}

    # Copy 5m from main analysis
    analysis_5m = _state.get("analysis", {})
    multi["5m"] = {
        "signals": analysis_5m.get("signals", []),
        "trend": analysis_5m.get("trend", "UNKNOWN"),
        "vwap_position": analysis_5m.get("vwap", {}).get("position", "UNKNOWN"),
    }
    _state["multi_signals"] = multi

    await ws_manager.broadcast({
        "type": "multi_signals",
        "data": {"multi_signals": multi},
    })
    logger.info(f"Multi-signals: 5m={len(multi.get('5m',{}).get('signals',[]))} 1h={len(multi.get('1h',{}).get('signals',[]))}")


async def _run_news():
    """Fetch news and run AI analysis (Groq → OpenAI → keyword fallback)."""
    try:
        news = await provider.get_news()
        _state["news"] = news

        tech_ctx = _state.get("analysis") or None
        pattern_ctx = _state.get("patterns") or None

        ai_analysis = await analyze_news_with_ai(
            news_items=news,
            technical_context=tech_ctx,
            pattern_context=pattern_ctx,
            groq_api_key=cfg.GROQ_API_KEY,
            groq_model=cfg.GROQ_MODEL,
            openai_api_key=cfg.OPENAI_API_KEY,
            openai_model=cfg.OPENAI_MODEL,
        )

        _state["news_analysis"] = ai_analysis
        _state["last_news_ts"] = datetime.now().isoformat()

        await ws_manager.broadcast({
            "type": "news",
            "data": {
                "news": news[:12],
                "ai_analysis": ai_analysis,
                "timestamp": _state["last_news_ts"],
            },
        })
        logger.info(f"News analyzed — sentiment: {ai_analysis.get('sentiment_label')} direction: {ai_analysis.get('direction_bias')}")
    except Exception as e:
        logger.error(f"News run error: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _broadcast_task, _analysis_task, _news_task

    logger.info("Starting AI Agent Trading System (Real-time NASDAQ)...")

    # Connect MetaTrader 5 (if credentials are configured)
    if cfg.MT5_LOGIN and cfg.MT5_PASSWORD and cfg.MT5_SERVER:
        ok = mt5_connector.connect(
            login=cfg.MT5_LOGIN,
            password=cfg.MT5_PASSWORD,
            server=cfg.MT5_SERVER,
        )
        if ok:
            # Resolve the exact symbol name available on the broker
            resolved = mt5_connector.get_symbol(
                primary=cfg.MT5_SYMBOL,
                fallbacks=["US100", "USTEC", "NDX100", "NAS100.raw", "NAS100m"],
            )
            if resolved:
                provider.mt5_symbol = resolved
                logger.info(f"MT5 connected — using symbol: {resolved}")
            else:
                logger.warning(f"MT5: symbol '{cfg.MT5_SYMBOL}' not found on broker, trying with default")
        else:
            logger.warning("MT5 connection failed — market data unavailable (no Yahoo Finance fallback).")
    else:
        # No explicit credentials — try connecting to the already-running MT5 terminal
        ok = mt5_connector.connect(login=0, password="", server="")
        if ok and mt5_connector.is_connected:
            resolved = mt5_connector.get_symbol(
                primary=cfg.MT5_SYMBOL,
                fallbacks=["US100", "USTEC", "NDX100", "NAS100.raw", "NAS100m"],
            )
            if resolved:
                provider.mt5_symbol = resolved
                logger.info(f"MT5 terminal connected (no explicit credentials) — using symbol: {resolved}")
            else:
                logger.warning(f"MT5 terminal connected but symbol '{cfg.MT5_SYMBOL}' not found on broker.")
        else:
            logger.info("MT5 credentials not set and no terminal running — mock/demo mode.")

    # Initial data load
    try:
        logger.info("Loading initial market data...")
        price = await provider.get_current_price()
        _state["price"] = price
        logger.info(f"Current US100 price: {price.get('price')} (source: {price.get('source')})")

        await _run_analysis()
        await _run_news()
    except Exception as e:
        logger.warning(f"Initial data load: {e}")

    # Start background tasks
    _broadcast_task     = asyncio.create_task(_price_loop())
    _analysis_task      = asyncio.create_task(_analysis_loop())
    _news_task          = asyncio.create_task(_news_loop())
    _signal_update_task = asyncio.create_task(_signal_update_loop())
    logger.info("Background tasks started.")

    yield

    logger.info("Shutting down...")
    for task in [_broadcast_task, _analysis_task, _news_task, _signal_update_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    mt5_connector.disconnect()
    logger.info("Shutdown complete.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Agent Trading — NASDAQ 100",
    description="Real-time NAS100 analysis with SMC, institutional flow, OpenAI & pattern matching",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/price")
async def get_price():
    """Current real-time price."""
    try:
        price = await provider.get_current_price()
        _state["price"] = price
        vix = await provider.get_vix()
        _state["vix"] = vix
        return JSONResponse(content=_serial({**price, "vix": vix}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market")
async def get_market():
    """OHLCV candles with full technical analysis."""
    try:
        df_5m = await provider.get_ohlcv("5m", 300)
        if df_5m.empty:
            raise HTTPException(status_code=503, detail="Market data unavailable")

        analysis = full_analysis(df_5m)
        _state["analysis"] = analysis

        price = _state.get("price") or await provider.get_current_price()
        vix = await provider.get_vix()

        return JSONResponse(content=_serial({
            "symbol": cfg.SYMBOL,
            "price": price,
            "vix": vix,
            "analysis": analysis,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /api/market error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis")
async def get_analysis():
    """Return cached full technical analysis or trigger fresh one."""
    try:
        if not _state.get("analysis"):
            await _run_analysis()

        return JSONResponse(content=_serial({
            "analysis": _state.get("analysis", {}),
            "patterns": _state.get("patterns", {}),
            "timestamp": _state.get("last_analysis_ts"),
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def trigger_analysis():
    """Trigger fresh analysis (price + technical + patterns + news)."""
    try:
        await _run_analysis()
        await _run_news()
        return JSONResponse(content=_serial({
            "analysis": _state.get("analysis", {}),
            "patterns": _state.get("patterns", {}),
            "news_analysis": _state.get("news_analysis", {}),
            "news": _state.get("news", [])[:10],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
async def get_news():
    """Fetch and analyze latest NASDAQ news."""
    try:
        await _run_news()
        return JSONResponse(content=_serial({
            "news": _state.get("news", [])[:15],
            "ai_analysis": _state.get("news_analysis", {}),
            "fetched_at": _state.get("last_news_ts"),
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patterns")
async def get_patterns():
    """Return historical pattern matches for current price action."""
    try:
        if not _state.get("patterns"):
            df_5m = await provider.get_ohlcv("5m", 300)
            df_hist = await provider.get_historical_for_patterns("5m", 1000)
            if not df_hist.empty:
                pattern_matcher.load_data(df_hist)
                _state["patterns"] = pattern_matcher.find_similar_patterns(df_5m['close'].tail(100))

        return JSONResponse(content=_serial(_state.get("patterns", {})))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_signals():
    """Return current trading signals with entry/exit levels."""
    try:
        analysis = _state.get("analysis", {})
        signals = analysis.get("signals", [])
        news_analysis = _state.get("news_analysis", {})
        patterns = _state.get("patterns", {})

        # Combine all context into enriched signals
        enriched = []
        for sig in signals:
            enriched.append({
                **sig,
                "news_sentiment": news_analysis.get("sentiment_label", "NEUTRAL"),
                "news_direction": news_analysis.get("direction_bias", "SIDEWAYS"),
                "pattern_bias": patterns.get("statistics", {}).get("bias", "NEUTRAL"),
                "pattern_win_rate": patterns.get("statistics", {}).get("win_rate_long", 50),
                "ai_reasoning": news_analysis.get("reasoning", ""),
            })

        return JSONResponse(content=_serial({
            "signals": enriched,
            "count": len(enriched),
            "market_trend": analysis.get("trend", "UNKNOWN"),
            "vwap_position": analysis.get("vwap", {}).get("position", "UNKNOWN"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/candles")
async def get_candles(interval: str = "5m", bars: int = 150):
    """Get OHLCV candles for a given interval. Serves from cache when fresh."""
    allowed = {"1m", "5m", "15m", "1h", "1d"}
    if interval not in allowed:
        raise HTTPException(status_code=400, detail=f"Interval must be one of {allowed}")
    try:
        # TTL: 1m→20s, 5m→45s, others→90s
        ttl = {"1m": 20, "5m": 45}.get(interval, 90)
        cached = _state.get("candle_cache", {}).get(interval)
        if cached:
            age = (datetime.now() - cached["ts"]).total_seconds()
            # Only use cache if it's fresh AND has at least as many bars as requested
            if age < ttl and len(cached["candles"]) >= bars:
                return JSONResponse(content=_serial({
                    "interval": interval,
                    "candles": cached["candles"],
                    "count": len(cached["candles"]),
                    "cached": True,
                }))

        df = await provider.get_ohlcv(interval, min(bars, 500))
        candles = _df_to_candles(df)
        _state["candle_cache"][interval] = {"candles": candles, "ts": datetime.now()}
        return JSONResponse(content=_serial({"interval": interval, "candles": candles, "count": len(candles)}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals_multi")
async def get_signals_multi():
    """Return cached multi-TF signals (1m / 5m / 1h)."""
    try:
        multi = _state.get("multi_signals", {})
        if not multi or not any(multi.values()):
            await _run_multi_signals()
            multi = _state.get("multi_signals", {})
        news_analysis = _state.get("news_analysis", {})
        patterns = _state.get("patterns", {})
        # Enrich signals with news/pattern context
        result = {}
        for tf, data in multi.items():
            enriched = []
            for sig in data.get("signals", []):
                enriched.append({
                    **sig,
                    "tf": tf,
                    "news_sentiment": news_analysis.get("sentiment_label", "NEUTRAL"),
                    "pattern_bias": patterns.get("statistics", {}).get("bias", "NEUTRAL"),
                })
            result[tf] = {**data, "signals": enriched}
        return JSONResponse(content=_serial(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session")
async def get_session():
    """Current market session + regime info (no recalculation, uses cached analysis)."""
    try:
        analysis    = _state.get("analysis", {})
        session_obj = analysis.get("session", {})
        regime_obj  = analysis.get("regime",  {})

        # If no cached analysis yet, compute session live
        if not session_obj:
            from modules.technical_analysis import detect_market_session
            session_obj = detect_market_session()

        return JSONResponse(content=_serial({
            "session": session_obj,
            "regime":  regime_obj,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/active")
async def get_active_signals():
    """
    Current 30-minute signal window:
    - Max 3 signals, real-time confidence scores
    - Includes status (ACTIVE/TP_HIT/SL_HIT/EXPIRED/INVALIDATED)
    - Regime + session context
    """
    try:
        analysis    = _state.get("analysis", {})
        regime_obj  = analysis.get("regime",  {})
        session_obj = analysis.get("session", {})

        active = signal_manager.get_active()
        stats  = signal_manager.get_stats()

        return JSONResponse(content=_serial({
            "signals":  active,
            "count":    len(active),
            "stats":    stats,
            "regime":   regime_obj,
            "session":  session_obj,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learner")
async def get_learner_stats():
    """Adaptive learner stats: win rates per setup, recent P&L, total signals tracked."""
    try:
        stats = adaptive_learner.get_stats()
        recent = adaptive_learner.get_recent_signals(20)
        return JSONResponse(content=_serial({"stats": stats, "recent_signals": recent}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mt5/account")
async def get_mt5_account():
    """MT5 account info: balance, equity, profit, margin, leverage."""
    try:
        return JSONResponse(content=_serial(mt5_connector.get_account_info()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mt5/positions")
async def get_mt5_positions():
    """List of open MT5 positions."""
    try:
        return JSONResponse(content=_serial(mt5_connector.get_open_positions()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mt5/status")
async def get_mt5_status():
    """MT5 connection status."""
    from modules.mt5_connector import MT5_AVAILABLE
    return JSONResponse(content={
        "connected": mt5_connector.is_connected,
        "mt5_available": MT5_AVAILABLE,
        "mode": "live" if MT5_AVAILABLE and mt5_connector.is_connected else "mock",
        "symbol": provider.mt5_symbol,
        "data_source": "mt5" if (MT5_AVAILABLE and mt5_connector.is_connected) else "mock",
    })


@app.get("/health")
async def health():
    price = _state.get("price", {})
    return {
        "status": "ok",
        "symbol": cfg.SYMBOL,
        "price": price.get("price"),
        "data_source": price.get("source", "unknown"),
        "openai": bool(cfg.OPENAI_API_KEY),
        "finnhub": bool(cfg.FINNHUB_API_KEY),
        "mt5": mt5_connector.is_connected,
        "signals": len(_state.get("analysis", {}).get("signals", [])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        # Send full state on connect
        analysis = _state.get("analysis", {})
        await ws.send_text(json.dumps(_serial({
            "type": "connected",
            "data": {
                "price":           _state.get("price", {}),
                "analysis":        analysis,
                "patterns":        _state.get("patterns", {}),
                "news":            _state.get("news", [])[:10],
                "news_analysis":   _state.get("news_analysis", {}),
                "multi_signals":   _state.get("multi_signals", {}),
                "managed_signals": signal_manager.get_active(),
                "signal_stats":    signal_manager.get_stats(),
                "regime":          analysis.get("regime", {}),
                "session":         analysis.get("session", {}),
                "symbol":          cfg.SYMBOL,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })))

        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if msg == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif msg == "analyze":
                    await _run_analysis()
                    await ws.send_text(json.dumps(_serial({
                        "type": "analysis",
                        "data": {
                            "analysis": _state.get("analysis", {}),
                            "patterns": _state.get("patterns", {}),
                        },
                    })))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(ws)
