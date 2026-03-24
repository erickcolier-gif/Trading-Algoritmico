"""
MetaTrader 5 connector module.
Handles all MT5 operations: connection, data fetching, order placement.
Falls back to mock data when MT5 is not available.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import MetaTrader5
MT5_AVAILABLE = False
mt5 = None
try:
    import MetaTrader5 as _mt5
    mt5 = _mt5
    MT5_AVAILABLE = True
    logger.info("MetaTrader5 library imported successfully.")
except ImportError:
    logger.warning("MetaTrader5 library not available. Running in mock/demo mode.")


class MT5Connector:
    """
    Singleton connector for MetaTrader 5.
    Provides a unified interface for all MT5 operations with mock fallback.
    """

    _instance: Optional["MT5Connector"] = None
    _initialized: bool = False

    def __new__(cls) -> "MT5Connector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._connected: bool = False
        self._symbol: Optional[str] = None
        self._mock_base_price: float = 21000.0
        self._mock_tick: int = 0
        self._mock_positions: List[Dict[str, Any]] = []
        self._mock_balance: float = 10000.0
        self._mock_equity: float = 10000.0
        self._mock_ticket_counter: int = 1000000

    def connect(self, login: int, password: str, server: str) -> bool:
        """
        Initialize connection to MT5 terminal.
        Returns True on success, False on failure.
        """
        if not MT5_AVAILABLE:
            logger.info("MT5 not available — running in mock mode.")
            self._connected = True
            return True

        try:
            if not mt5.initialize(login=login, password=password, server=server):
                error = mt5.last_error()
                logger.error(f"MT5 initialize failed: {error}")
                self._connected = False
                return False

            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info after initialization.")
                self._connected = False
                return False

            logger.info(
                f"MT5 connected: account={account_info.login}, "
                f"server={account_info.server}, balance={account_info.balance}"
            )
            self._connected = True
            return True

        except Exception as exc:
            logger.error(f"MT5 connect exception: {exc}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if MT5_AVAILABLE and self._connected:
            try:
                mt5.shutdown()
                logger.info("MT5 disconnected.")
            except Exception as exc:
                logger.error(f"MT5 disconnect error: {exc}")
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_symbol(self, primary: str = "NAS100", fallbacks: Optional[List[str]] = None) -> Optional[str]:
        """
        Find the correct NAS100 symbol available on the broker.
        Tries primary symbol first, then fallbacks.
        """
        if fallbacks is None:
            fallbacks = ["US100", "USTEC", "NDX100"]

        if not MT5_AVAILABLE:
            self._symbol = primary
            return primary

        candidates = [primary] + fallbacks
        for sym in candidates:
            try:
                info = mt5.symbol_info(sym)
                if info is not None and info.visible:
                    self._symbol = sym
                    logger.info(f"Using symbol: {sym}")
                    return sym
                # Try to select the symbol (make it visible)
                if info is not None:
                    mt5.symbol_select(sym, True)
                    import time; time.sleep(0.5)  # wait for tick data to start streaming
                    info = mt5.symbol_info(sym)
                    if info is not None:
                        self._symbol = sym
                        logger.info(f"Selected and using symbol: {sym}")
                        return sym
            except Exception as exc:
                logger.warning(f"Symbol check failed for {sym}: {exc}")
                continue

        logger.error(f"No valid symbol found from candidates: {candidates}")
        return None

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[Any] = None,
        count: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for the given symbol and timeframe.
        Returns a pandas DataFrame with columns: time, open, high, low, close, volume.
        Falls back to generated mock data if MT5 is unavailable.
        """
        if timeframe is None and MT5_AVAILABLE:
            timeframe = mt5.TIMEFRAME_M4
        elif timeframe is None:
            timeframe = "M4"

        if not MT5_AVAILABLE or not self._connected:
            return self._generate_mock_ohlcv(count)

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                # Symbol may not be subscribed — select and retry once
                mt5.symbol_select(symbol, True)
                import time; time.sleep(0.3)
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"No OHLCV data returned for {symbol}. Using mock data.")
                return self._generate_mock_ohlcv(count)

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={"tick_volume": "volume"})
            df = df[["time", "open", "high", "low", "close", "volume"]]
            df = df.reset_index(drop=True)
            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df

        except Exception as exc:
            logger.error(f"get_ohlcv error: {exc}")
            return self._generate_mock_ohlcv(count)

    def _generate_mock_ohlcv(self, count: int = 200) -> pd.DataFrame:
        """Generate realistic mock OHLCV data for NAS100."""
        now = datetime.utcnow()
        # Round down to nearest 4-minute bar
        minutes_offset = now.minute % 4
        bar_time = now - timedelta(minutes=minutes_offset, seconds=now.second)

        times = [bar_time - timedelta(minutes=4 * i) for i in range(count)]
        times.reverse()

        base = self._mock_base_price
        opens, highs, lows, closes, volumes = [], [], [], [], []

        price = base
        for i in range(count):
            open_p = price
            change = random.gauss(0, 8)
            close_p = open_p + change
            high_p = max(open_p, close_p) + abs(random.gauss(0, 5))
            low_p = min(open_p, close_p) - abs(random.gauss(0, 5))
            vol = int(random.gauss(5000, 1000))

            opens.append(round(open_p, 2))
            highs.append(round(high_p, 2))
            lows.append(round(low_p, 2))
            closes.append(round(close_p, 2))
            volumes.append(max(100, vol))

            price = close_p

        self._mock_base_price = price

        return pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

    def get_account_info(self) -> Dict[str, Any]:
        """
        Returns account info: balance, equity, profit, margin, margin_level, currency.
        """
        if not MT5_AVAILABLE or not self._connected:
            pnl = sum(p.get("profit", 0) for p in self._mock_positions)
            self._mock_equity = self._mock_balance + pnl
            return {
                "balance": self._mock_balance,
                "equity": self._mock_equity,
                "profit": pnl,
                "margin": 0.0,
                "margin_level": 0.0,
                "currency": "USD",
                "login": 0,
                "server": "Mock",
                "leverage": 100,
            }

        try:
            info = mt5.account_info()
            if info is None:
                raise ValueError("account_info() returned None")

            return {
                "balance": float(info.balance),
                "equity": float(info.equity),
                "profit": float(info.profit),
                "margin": float(info.margin),
                "margin_level": float(info.margin_level),
                "currency": info.currency,
                "login": int(info.login),
                "server": str(info.server),
                "leverage": int(info.leverage),
            }
        except Exception as exc:
            logger.error(f"get_account_info error: {exc}")
            return {
                "balance": 0.0,
                "equity": 0.0,
                "profit": 0.0,
                "margin": 0.0,
                "margin_level": 0.0,
                "currency": "USD",
                "login": 0,
                "server": "Error",
                "leverage": 0,
            }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Returns list of open positions."""
        if not MT5_AVAILABLE or not self._connected:
            # Update mock position P&L
            current_price = self._mock_base_price
            for pos in self._mock_positions:
                if pos["type"] == "BUY":
                    pos["profit"] = round((current_price - pos["open_price"]) * pos["lot"] * 10, 2)
                else:
                    pos["profit"] = round((pos["open_price"] - current_price) * pos["lot"] * 10, 2)
                pos["current_price"] = round(current_price, 2)
            return list(self._mock_positions)

        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            result = []
            for pos in positions:
                result.append({
                    "ticket": int(pos.ticket),
                    "symbol": str(pos.symbol),
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "lot": float(pos.volume),
                    "open_price": float(pos.price_open),
                    "current_price": float(pos.price_current),
                    "sl": float(pos.sl),
                    "tp": float(pos.tp),
                    "profit": float(pos.profit),
                    "swap": float(pos.swap),
                    "open_time": datetime.utcfromtimestamp(pos.time).isoformat(),
                    "comment": str(pos.comment),
                })
            return result

        except Exception as exc:
            logger.error(f"get_open_positions error: {exc}")
            return []

    def place_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        sl: float,
        tp: float,
        comment: str = "AI Scalper",
    ) -> Dict[str, Any]:
        """
        Place a market order.
        order_type: 'BUY' or 'SELL'
        Returns dict with success status and ticket/error.
        """
        if not MT5_AVAILABLE or not self._connected:
            return self._mock_place_order(symbol, order_type, lot, sl, tp, comment)

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Symbol {symbol} not found"}

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": "Failed to get current tick"}

            if order_type.upper() == "BUY":
                price = tick.ask
                mt5_type = mt5.ORDER_TYPE_BUY
            elif order_type.upper() == "SELL":
                price = tick.bid
                mt5_type = mt5.ORDER_TYPE_SELL
            else:
                return {"success": False, "error": f"Invalid order type: {order_type}"}

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 20241201,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"success": False, "error": "order_send returned None"}

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False,
                    "error": f"Order failed: {result.retcode} - {result.comment}",
                    "retcode": result.retcode,
                }

            return {
                "success": True,
                "ticket": int(result.order),
                "price": float(result.price),
                "volume": float(result.volume),
            }

        except Exception as exc:
            logger.error(f"place_order error: {exc}")
            return {"success": False, "error": str(exc)}

    def _mock_place_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        sl: float,
        tp: float,
        comment: str,
    ) -> Dict[str, Any]:
        """Mock order placement for testing."""
        price = self._mock_base_price + (2.0 if order_type.upper() == "BUY" else -2.0)
        self._mock_ticket_counter += 1
        ticket = self._mock_ticket_counter

        position = {
            "ticket": ticket,
            "symbol": symbol,
            "type": order_type.upper(),
            "lot": lot,
            "open_price": round(price, 2),
            "current_price": round(price, 2),
            "sl": sl,
            "tp": tp,
            "profit": 0.0,
            "swap": 0.0,
            "open_time": datetime.utcnow().isoformat(),
            "comment": comment,
        }
        self._mock_positions.append(position)

        logger.info(f"Mock order placed: {order_type} {lot} lots at {price} (ticket={ticket})")
        return {
            "success": True,
            "ticket": ticket,
            "price": round(price, 2),
            "volume": lot,
        }

    def close_position(self, ticket: int) -> Dict[str, Any]:
        """
        Close an open position by ticket number.
        Returns dict with success status.
        """
        if not MT5_AVAILABLE or not self._connected:
            return self._mock_close_position(ticket)

        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {"success": False, "error": f"Position {ticket} not found"}

            pos = positions[0]
            symbol = pos.symbol
            tick = mt5.symbol_info_tick(symbol)

            if pos.type == 0:  # BUY
                close_price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:  # SELL
                close_price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 20241201,
                "comment": "AI Scalper Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"success": False, "error": "order_send returned None"}

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False,
                    "error": f"Close failed: {result.retcode} - {result.comment}",
                }

            return {"success": True, "ticket": ticket, "price": float(result.price)}

        except Exception as exc:
            logger.error(f"close_position error: {exc}")
            return {"success": False, "error": str(exc)}

    def _mock_close_position(self, ticket: int) -> Dict[str, Any]:
        """Mock position closing."""
        for i, pos in enumerate(self._mock_positions):
            if pos["ticket"] == ticket:
                closed_pos = self._mock_positions.pop(i)
                logger.info(f"Mock position closed: ticket={ticket}, profit={closed_pos.get('profit', 0)}")
                return {
                    "success": True,
                    "ticket": ticket,
                    "price": closed_pos["current_price"],
                }
        return {"success": False, "error": f"Position {ticket} not found"}

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get current bid/ask price for a symbol.
        Returns dict with bid, ask, spread.
        """
        if not MT5_AVAILABLE or not self._connected:
            # Simulate price movement
            self._mock_tick += 1
            drift = random.gauss(0, 3)
            self._mock_base_price = max(18000, self._mock_base_price + drift)
            bid = round(self._mock_base_price, 2)
            ask = round(bid + random.uniform(0.5, 2.0), 2)
            return {
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 2),
                "symbol": symbol,
            }

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                # Symbol might not be subscribed yet — select it and retry
                mt5.symbol_select(symbol, True)
                import time; time.sleep(0.3)
                tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise ValueError(f"No tick data for {symbol}")

            return {
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "spread": round(float(tick.ask) - float(tick.bid), 2),
                "symbol": symbol,
            }

        except Exception as exc:
            logger.error(f"get_current_price error: {exc}")
            return {"bid": 0.0, "ask": 0.0, "spread": 0.0, "symbol": symbol}


# Global singleton instance
mt5_connector = MT5Connector()
