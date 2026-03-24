"""
Technical analysis module for NAS100 scalping.
Calculates indicators and generates trading signals.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float safely, returning None if not possible."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _last(series: pd.Series) -> Optional[float]:
    """Get the last non-NaN value of a pandas Series."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    return _safe_float(val)


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands: upper, middle (SMA), lower."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    range_ = highest_high - lowest_low
    k = 100 * (close - lowest_low) / range_.replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive technical analysis on OHLCV data.

    Args:
        df: DataFrame with columns [time, open, high, low, close, volume]

    Returns:
        Dictionary with all indicators, trend direction, and signal strength.
    """
    if df is None or df.empty or len(df) < 50:
        return {"error": "Insufficient data for analysis", "signal": "NEUTRAL", "strength": 0}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Moving Averages
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)

    # RSI
    rsi = _rsi(close, 14)

    # MACD
    macd_line, macd_signal, macd_hist = _macd(close, 12, 26, 9)

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, 20, 2.0)

    # ATR
    atr = _atr(high, low, close, 14)

    # Stochastic
    stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)

    # Volume SMA
    vol_sma = volume.rolling(window=20).mean()

    # Current values
    current_price = _last(close)
    current_ema9 = _last(ema9)
    current_ema21 = _last(ema21)
    current_ema50 = _last(ema50)
    current_rsi = _last(rsi)
    current_macd = _last(macd_line)
    current_macd_signal = _last(macd_signal)
    current_macd_hist = _last(macd_hist)
    current_bb_upper = _last(bb_upper)
    current_bb_mid = _last(bb_mid)
    current_bb_lower = _last(bb_lower)
    current_atr = _last(atr)
    current_stoch_k = _last(stoch_k)
    current_stoch_d = _last(stoch_d)
    current_vol = _last(volume)
    current_vol_sma = _last(vol_sma)

    # Previous values for crossover detection
    prev_macd_hist = _safe_float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else None
    prev_ema9 = _safe_float(ema9.iloc[-2]) if len(ema9) > 1 else None
    prev_ema21 = _safe_float(ema21.iloc[-2]) if len(ema21) > 1 else None

    # --- Trend Analysis ---
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0

    # EMA alignment
    if all(v is not None for v in [current_price, current_ema9, current_ema21, current_ema50]):
        total_signals += 1
        if current_price > current_ema9 > current_ema21 > current_ema50:
            bullish_signals += 1
        elif current_price < current_ema9 < current_ema21 < current_ema50:
            bearish_signals += 1

    # EMA9/EMA21 crossover
    if all(v is not None for v in [current_ema9, current_ema21, prev_ema9, prev_ema21]):
        total_signals += 1
        if current_ema9 > current_ema21:
            bullish_signals += 1
        else:
            bearish_signals += 1

    # RSI
    if current_rsi is not None:
        total_signals += 1
        if current_rsi > 55:
            bullish_signals += 1
        elif current_rsi < 45:
            bearish_signals += 1

    # MACD histogram direction
    if current_macd_hist is not None and prev_macd_hist is not None:
        total_signals += 1
        if current_macd_hist > 0 and current_macd_hist > prev_macd_hist:
            bullish_signals += 1
        elif current_macd_hist < 0 and current_macd_hist < prev_macd_hist:
            bearish_signals += 1

    # Stochastic
    if current_stoch_k is not None and current_stoch_d is not None:
        total_signals += 1
        if current_stoch_k > current_stoch_d and current_stoch_k < 80:
            bullish_signals += 1
        elif current_stoch_k < current_stoch_d and current_stoch_k > 20:
            bearish_signals += 1

    # Bollinger Band position
    if all(v is not None for v in [current_price, current_bb_upper, current_bb_lower, current_bb_mid]):
        bb_range = current_bb_upper - current_bb_lower
        if bb_range > 0:
            bb_position = (current_price - current_bb_lower) / bb_range
            total_signals += 1
            if bb_position > 0.6:
                bullish_signals += 1
            elif bb_position < 0.4:
                bearish_signals += 1

    # Volume confirmation
    volume_surge = False
    if current_vol is not None and current_vol_sma is not None and current_vol_sma > 0:
        volume_surge = current_vol > current_vol_sma * 1.3

    # Determine trend direction and signal strength
    if total_signals > 0:
        bull_pct = bullish_signals / total_signals
        bear_pct = bearish_signals / total_signals

        if bull_pct >= 0.6:
            trend = "BULLISH"
            signal = "BUY"
            strength = min(100, int(bull_pct * 100))
        elif bear_pct >= 0.6:
            trend = "BEARISH"
            signal = "SELL"
            strength = min(100, int(bear_pct * 100))
        else:
            trend = "SIDEWAYS"
            signal = "NEUTRAL"
            strength = 50
    else:
        trend = "UNKNOWN"
        signal = "NEUTRAL"
        strength = 0

    # Boost strength with volume
    if volume_surge and signal != "NEUTRAL":
        strength = min(100, strength + 10)

    # Overbought/Oversold conditions
    is_overbought = (current_rsi is not None and current_rsi > 70) or \
                    (current_stoch_k is not None and current_stoch_k > 80)
    is_oversold = (current_rsi is not None and current_rsi < 30) or \
                  (current_stoch_k is not None and current_stoch_k < 20)

    # MACD crossover detection
    macd_bullish_cross = (
        prev_macd_hist is not None and current_macd_hist is not None and
        prev_macd_hist <= 0 < current_macd_hist
    )
    macd_bearish_cross = (
        prev_macd_hist is not None and current_macd_hist is not None and
        prev_macd_hist >= 0 > current_macd_hist
    )

    # Bollinger Band squeeze (low volatility, potential breakout)
    bb_squeeze = False
    if current_bb_upper is not None and current_bb_lower is not None and current_bb_mid is not None and current_bb_mid != 0:
        bb_width = (current_bb_upper - current_bb_lower) / current_bb_mid
        if len(bb_upper) > 20:
            historical_width = ((bb_upper - bb_lower) / bb_mid).rolling(20).mean().iloc[-1]
            bb_squeeze = bb_width < historical_width * 0.7 if not np.isnan(historical_width) else False

    return {
        "signal": signal,
        "trend": trend,
        "strength": strength,
        "indicators": {
            "ema9": current_ema9,
            "ema21": current_ema21,
            "ema50": current_ema50,
            "rsi": round(current_rsi, 2) if current_rsi is not None else None,
            "macd": round(current_macd, 4) if current_macd is not None else None,
            "macd_signal": round(current_macd_signal, 4) if current_macd_signal is not None else None,
            "macd_histogram": round(current_macd_hist, 4) if current_macd_hist is not None else None,
            "bb_upper": round(current_bb_upper, 2) if current_bb_upper is not None else None,
            "bb_mid": round(current_bb_mid, 2) if current_bb_mid is not None else None,
            "bb_lower": round(current_bb_lower, 2) if current_bb_lower is not None else None,
            "atr": round(current_atr, 2) if current_atr is not None else None,
            "stoch_k": round(current_stoch_k, 2) if current_stoch_k is not None else None,
            "stoch_d": round(current_stoch_d, 2) if current_stoch_d is not None else None,
            "volume": int(current_vol) if current_vol is not None else None,
            "volume_sma": round(current_vol_sma, 1) if current_vol_sma is not None else None,
        },
        "conditions": {
            "is_overbought": is_overbought,
            "is_oversold": is_oversold,
            "volume_surge": volume_surge,
            "macd_bullish_cross": macd_bullish_cross,
            "macd_bearish_cross": macd_bearish_cross,
            "bb_squeeze": bb_squeeze,
        },
        "signal_counts": {
            "bullish": bullish_signals,
            "bearish": bearish_signals,
            "total": total_signals,
        },
        "current_price": current_price,
    }


def get_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[float]]:
    """
    Identify key support and resistance levels using pivot points.

    Args:
        df: OHLCV DataFrame
        lookback: Number of candles to analyze

    Returns:
        Dict with 'support' and 'resistance' lists.
    """
    if df is None or len(df) < lookback:
        lookback = len(df) if df is not None else 0
    if lookback < 5:
        return {"support": [], "resistance": []}

    recent = df.tail(lookback).copy()
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values

    support_levels: List[float] = []
    resistance_levels: List[float] = []
    window = 3

    for i in range(window, len(recent) - window):
        # Local low (support)
        if lows[i] == min(lows[i - window:i + window + 1]):
            support_levels.append(round(float(lows[i]), 2))
        # Local high (resistance)
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistance_levels.append(round(float(highs[i]), 2))

    current_price = float(closes[-1])

    # Filter: keep only levels near current price (within 2%)
    price_range = current_price * 0.02
    support_levels = sorted(
        [s for s in support_levels if s < current_price and current_price - s <= price_range],
        reverse=True
    )[:3]
    resistance_levels = sorted(
        [r for r in resistance_levels if r > current_price and r - current_price <= price_range]
    )[:3]

    # Add pivot points as additional S/R
    if len(recent) >= 1:
        last_high = float(recent["high"].max())
        last_low = float(recent["low"].min())
        pivot = round((last_high + last_low + current_price) / 3, 2)
        r1 = round(2 * pivot - last_low, 2)
        s1 = round(2 * pivot - last_high, 2)

        if s1 < current_price and s1 not in support_levels:
            support_levels.append(s1)
        if r1 > current_price and r1 not in resistance_levels:
            resistance_levels.append(r1)

    return {
        "support": sorted(support_levels, reverse=True)[:3],
        "resistance": sorted(resistance_levels)[:3],
    }


def get_technical_summary(df: pd.DataFrame) -> str:
    """
    Generate a human-readable technical analysis summary.

    Args:
        df: OHLCV DataFrame

    Returns:
        Multi-line string with analysis summary.
    """
    analysis = analyze(df)
    sr = get_support_resistance(df)

    if "error" in analysis:
        return f"Analysis error: {analysis['error']}"

    ind = analysis.get("indicators", {})
    cond = analysis.get("conditions", {})
    signal_counts = analysis.get("signal_counts", {})

    lines = [
        f"=== NAS100 Technical Analysis ===",
        f"Signal: {analysis['signal']} | Trend: {analysis['trend']} | Strength: {analysis['strength']}%",
        f"",
        f"Moving Averages:",
        f"  EMA9:  {ind.get('ema9', 'N/A'):.2f}" if ind.get('ema9') else "  EMA9:  N/A",
        f"  EMA21: {ind.get('ema21', 'N/A'):.2f}" if ind.get('ema21') else "  EMA21: N/A",
        f"  EMA50: {ind.get('ema50', 'N/A'):.2f}" if ind.get('ema50') else "  EMA50: N/A",
        f"",
        f"Oscillators:",
        f"  RSI(14): {ind.get('rsi', 'N/A')}",
        f"  Stoch %K: {ind.get('stoch_k', 'N/A')} | %D: {ind.get('stoch_d', 'N/A')}",
        f"",
        f"MACD (12,26,9):",
        f"  MACD: {ind.get('macd', 'N/A')} | Signal: {ind.get('macd_signal', 'N/A')}",
        f"  Histogram: {ind.get('macd_histogram', 'N/A')}",
        f"",
        f"Bollinger Bands (20,2):",
        f"  Upper: {ind.get('bb_upper', 'N/A')} | Mid: {ind.get('bb_mid', 'N/A')} | Lower: {ind.get('bb_lower', 'N/A')}",
        f"",
        f"Volatility:",
        f"  ATR(14): {ind.get('atr', 'N/A')}",
        f"",
        f"Conditions:",
        f"  Overbought: {cond.get('is_overbought', False)} | Oversold: {cond.get('is_oversold', False)}",
        f"  Volume Surge: {cond.get('volume_surge', False)} | BB Squeeze: {cond.get('bb_squeeze', False)}",
        f"  MACD Bull Cross: {cond.get('macd_bullish_cross', False)} | Bear Cross: {cond.get('macd_bearish_cross', False)}",
        f"",
        f"Signal Count: {signal_counts.get('bullish', 0)} bullish / {signal_counts.get('bearish', 0)} bearish out of {signal_counts.get('total', 0)}",
    ]

    if sr["support"] or sr["resistance"]:
        lines.append("")
        lines.append(f"Key Levels:")
        if sr["resistance"]:
            lines.append(f"  Resistance: {', '.join(str(r) for r in sr['resistance'])}")
        if sr["support"]:
            lines.append(f"  Support: {', '.join(str(s) for s in sr['support'])}")

    return "\n".join(lines)
