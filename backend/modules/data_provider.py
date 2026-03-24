"""
Market data provider — MetaTrader 5 only.
OHLCV + live price come exclusively from MT5.
Finnhub is used only for news. RSS is the news fallback.
"""
import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False
    logger.warning("finnhub-python not installed — news from RSS only")

# MT5 timeframe map — populated when MetaTrader5 lib is available
_MT5_TF_MAP: Dict[str, Any] = {}
try:
    import MetaTrader5 as _mt5lib
    _MT5_TF_MAP = {
        "1m":  _mt5lib.TIMEFRAME_M1,
        "5m":  _mt5lib.TIMEFRAME_M5,
        "15m": _mt5lib.TIMEFRAME_M15,
        "30m": _mt5lib.TIMEFRAME_M30,
        "1h":  _mt5lib.TIMEFRAME_H1,
        "4h":  _mt5lib.TIMEFRAME_H4,
        "1d":  _mt5lib.TIMEFRAME_D1,
    }
except ImportError:
    pass


class DataProvider:
    """
    Market data exclusively from MetaTrader 5.
    News from Finnhub (primary) or RSS (fallback).
    """

    def __init__(
        self,
        symbol: str = "QQQ",          # kept for news/compatibility, not used for OHLCV
        finnhub_api_key: str = "",
        mt5_connector=None,
        mt5_symbol: str = "NAS100",
    ):
        self.symbol = symbol
        self.mt5_connector = mt5_connector
        self.mt5_symbol = mt5_symbol
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache: Dict = {}
        self._cache_time: Dict = {}
        self.finnhub_client = None

        if HAS_FINNHUB and finnhub_api_key:
            try:
                self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
                logger.info("Finnhub client initialized (news only)")
            except Exception as e:
                logger.warning(f"Finnhub init failed: {e}")

        if mt5_connector and mt5_connector.is_connected:
            logger.info(f"DataProvider: MT5 is the sole market data source (symbol={mt5_symbol})")

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _is_cached(self, key: str, ttl: int = 30) -> bool:
        if key not in self._cache_time:
            return False
        return (datetime.now() - self._cache_time[key]).seconds < ttl

    def _set_cache(self, key: str, value):
        self._cache[key] = value
        self._cache_time[key] = datetime.now()

    def _get_cache(self, key: str):
        return self._cache.get(key)

    # ── MT5 helpers ───────────────────────────────────────────────────────

    def _mt5_available(self) -> bool:
        return (
            self.mt5_connector is not None
            and self.mt5_connector.is_connected
            and bool(_MT5_TF_MAP)
        )

    def _fetch_mt5_ohlcv(self, interval: str, count: int) -> pd.DataFrame:
        """Fetch OHLCV bars from MT5 (blocking — must run in executor)."""
        if not self._mt5_available():
            return pd.DataFrame()

        tf = _MT5_TF_MAP.get(interval)
        if tf is None:
            logger.warning(f"MT5: unsupported interval '{interval}'")
            return pd.DataFrame()

        try:
            df = self.mt5_connector.get_ohlcv(self.mt5_symbol, tf, count)
            if df is None or df.empty:
                return pd.DataFrame()
            # mt5_connector returns 'time' as a column — make it the index
            if 'time' in df.columns:
                df = df.set_index('time')
                df.index.name = None
            logger.info(f"MT5 OHLCV: {self.mt5_symbol} {interval} → {len(df)} bars")
            return df
        except Exception as e:
            logger.error(f"MT5 OHLCV error ({interval}): {e}")
            return pd.DataFrame()

    def _fetch_mt5_price(self) -> dict:
        """Fetch live tick (bid/ask) from MT5 (blocking — must run in executor)."""
        if not self._mt5_available():
            return {}
        try:
            tick = self.mt5_connector.get_current_price(self.mt5_symbol)
            bid = tick.get('bid', 0)
            ask = tick.get('ask', 0)
            if bid <= 0:
                return {}
            price = round((bid + ask) / 2, 2)
            return {
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 2),
                "source": "mt5",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"MT5 price error: {e}")
            return {}

    # ── Public async API ──────────────────────────────────────────────────

    async def get_ohlcv(self, interval: str = "5m", bars: int = 300) -> pd.DataFrame:
        """OHLCV candles from MT5. Falls back to mock if MT5 is unavailable."""
        cache_key = f"ohlcv_mt5_{self.mt5_symbol}_{interval}_{bars}"
        if self._is_cached(cache_key, 30):
            return self._get_cache(cache_key)

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            self._executor, self._fetch_mt5_ohlcv, interval, bars
        )

        if df.empty:
            logger.warning(f"MT5 returned no data for {interval} — using mock")
            df = self._mock_ohlcv(bars)
        else:
            df = df.tail(bars)

        self._set_cache(cache_key, df)
        return df

    async def get_current_price(self) -> dict:
        """Live price from MT5. Finnhub quote as fallback, mock as last resort."""
        loop = asyncio.get_event_loop()

        # Primary: MT5
        data = await loop.run_in_executor(self._executor, self._fetch_mt5_price)
        if data and data.get('price', 0) > 0:
            # Enrich with change vs previous daily close
            try:
                df_1d = await self.get_ohlcv("1d", 3)
                if len(df_1d) >= 2:
                    prev = float(df_1d['close'].iloc[-2])
                    change = data['price'] - prev
                    data.update({
                        "prev_close": round(prev, 2),
                        "change": round(change, 2),
                        "change_pct": round((change / prev * 100) if prev else 0, 2),
                    })
            except Exception:
                pass
            return data

        # Fallback: Finnhub (for non-index symbols)
        if self.finnhub_client and not self.symbol.startswith("^"):
            try:
                quote = await loop.run_in_executor(
                    self._executor, lambda: self.finnhub_client.quote(self.symbol)
                )
                if quote and quote.get('c') and float(quote['c']) > 0:
                    price = float(quote['c'])
                    prev = float(quote.get('pc', price))
                    change = price - prev
                    return {
                        "price": round(price, 2),
                        "bid": round(price - 0.01, 2),
                        "ask": round(price + 0.01, 2),
                        "prev_close": round(prev, 2),
                        "change": round(change, 2),
                        "change_pct": round((change / prev * 100) if prev else 0, 2),
                        "source": "finnhub",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Finnhub quote fallback error: {e}")

        logger.warning("All price sources failed — returning mock price")
        return self._mock_price()

    async def get_vix(self) -> float:
        """VIX placeholder — MT5 does not provide ^VIX. Returns last cached or 20.0."""
        return self._cache.get("vix", 20.0)

    async def get_historical_for_patterns(self, interval: str = "5m", bars: int = 2000) -> pd.DataFrame:
        """Extended OHLCV history for pattern matching — MT5 only."""
        cache_key = f"hist_mt5_{self.mt5_symbol}_{interval}"
        if self._is_cached(cache_key, 300):
            return self._get_cache(cache_key)

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            self._executor, self._fetch_mt5_ohlcv, interval, bars
        )

        if df.empty:
            logger.warning("MT5 pattern history empty — using mock")
            df = self._mock_ohlcv(bars)
        else:
            df = df.tail(bars)

        self._set_cache(cache_key, df)
        return df

    async def get_news(self) -> List[dict]:
        """News from Finnhub (primary) or RSS (fallback)."""
        news = []

        if self.finnhub_client:
            news_symbol = "QQQ" if self.symbol.startswith("^") else self.symbol
            # Company news
            try:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    self._executor,
                    lambda: self.finnhub_client.company_news(
                        news_symbol,
                        _from=(datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                        to=datetime.now().strftime("%Y-%m-%d"),
                    ),
                )
                for item in raw[:20]:
                    if item.get('headline'):
                        news.append({
                            "title": item.get('headline', ''),
                            "summary": item.get('summary', '')[:300],
                            "source": item.get('source', 'Finnhub'),
                            "url": item.get('url', ''),
                            "timestamp": (
                                datetime.fromtimestamp(item['datetime']).isoformat()
                                if item.get('datetime') else datetime.now().isoformat()
                            ),
                        })
                if news:
                    return news[:15]
            except Exception as e:
                logger.warning(f"Finnhub company news error: {e}")

            # General market news
            if not news:
                try:
                    loop = asyncio.get_event_loop()
                    raw = await loop.run_in_executor(
                        self._executor,
                        lambda: self.finnhub_client.general_news('general', min_id=0),
                    )
                    kw = {'nasdaq', 'qqq', 'tech', 'fed', 'interest rate', 'inflation', 'ndx'}
                    for item in raw[:40]:
                        if any(k in (item.get('headline', '') or '').lower() for k in kw):
                            news.append({
                                "title": item.get('headline', ''),
                                "summary": item.get('summary', '')[:300],
                                "source": item.get('source', 'Finnhub'),
                                "url": item.get('url', ''),
                                "timestamp": (
                                    datetime.fromtimestamp(item['datetime']).isoformat()
                                    if item.get('datetime') else datetime.now().isoformat()
                                ),
                            })
                    if news:
                        return news[:15]
                except Exception as e:
                    logger.warning(f"Finnhub general news error: {e}")

        return await self._fetch_rss_news()

    async def _fetch_rss_news(self) -> List[dict]:
        import httpx
        news = []
        feeds = [
            "https://news.google.com/rss/search?q=NASDAQ+100+QQQ+stocks&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Federal+Reserve+interest+rate+market&hl=en-US&gl=US&ceid=US:en",
            "https://feeds.reuters.com/reuters/businessNews",
        ]
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in feeds:
                try:
                    resp = await client.get(url, follow_redirects=True)
                    if resp.status_code != 200:
                        continue
                    content = resp.text
                    titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content) or \
                             re.findall(r'<title>(.*?)</title>', content)
                    descs  = re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>', content) or \
                             re.findall(r'<description>(.*?)</description>', content)
                    links  = re.findall(r'<link>(https?://[^<]+)</link>', content)
                    dates  = re.findall(r'<pubDate>(.*?)</pubDate>', content)
                    for i, title in enumerate(titles[1:16]):
                        if len(title) < 10:
                            continue
                        news.append({
                            "title": title.strip(),
                            "summary": re.sub(r'<[^>]+>', '', descs[i + 1]).strip()[:300] if i + 1 < len(descs) else "",
                            "source": "Google News" if "google" in url else "Reuters",
                            "url": links[i] if i < len(links) else "",
                            "timestamp": dates[i].strip() if i < len(dates) else datetime.now().isoformat(),
                        })
                    if news:
                        break
                except Exception as e:
                    logger.warning(f"RSS error: {e}")
        return news[:15]

    # ── Mock fallbacks (used only if MT5 is not connected) ────────────────

    def _mock_ohlcv(self, bars: int = 300) -> pd.DataFrame:
        base = 21000.0
        dates = pd.date_range(end=datetime.now(), periods=bars, freq="5min")
        prices = [base]
        for _ in range(bars - 1):
            prices.append(max(prices[-1] * (1 + np.random.normal(0, 0.005)), 1))
        rows = []
        for price in prices:
            spread = price * 0.002
            o = price + np.random.normal(0, spread * 0.5)
            h = max(o, price) + abs(np.random.normal(0, spread))
            l = min(o, price) - abs(np.random.normal(0, spread))
            rows.append({
                'open': round(o, 2), 'high': round(h, 2),
                'low': round(l, 2), 'close': round(price, 2),
                'volume': int(np.random.lognormal(13, 0.5)),
            })
        return pd.DataFrame(rows, index=dates)

    def _mock_price(self) -> dict:
        base = 21000.0
        price = base + np.random.normal(0, base * 0.001)
        return {
            "price": round(price, 2),
            "bid": round(price - 0.5, 2),
            "ask": round(price + 0.5, 2),
            "change": round(np.random.normal(0, base * 0.0003), 2),
            "change_pct": round(np.random.normal(0, 0.3), 2),
            "source": "mock",
            "timestamp": datetime.now().isoformat(),
        }
