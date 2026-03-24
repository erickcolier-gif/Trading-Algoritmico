"""
Historical pattern matching for NASDAQ 100.
Finds similar past price patterns and analyzes what happened next.
Provides statistical edge based on historical behavior.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_window(prices: np.ndarray) -> np.ndarray:
    """Normalize a price window to percentage returns from first bar."""
    if len(prices) == 0 or prices[0] == 0:
        return np.zeros(len(prices))
    return (prices - prices[0]) / prices[0] * 100


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class PatternMatcher:
    """
    Finds historical price patterns similar to current market conditions.
    Uses normalized price windows and cosine similarity for matching.
    """

    def __init__(self, window: int = 20, top_k: int = 10, future_bars: int = 20):
        self.window = window
        self.top_k = top_k
        self.future_bars = future_bars
        self._historical_df: Optional[pd.DataFrame] = None

    def load_data(self, df: pd.DataFrame) -> None:
        """Load historical OHLCV data for pattern matching."""
        self._historical_df = df.copy()
        logger.info(f"Pattern matcher loaded {len(df)} bars of historical data")

    def find_similar_patterns(self, current_window: pd.Series) -> dict:
        """
        Find historical windows most similar to the current price action.
        Returns statistics on what happened next.
        """
        if self._historical_df is None or len(self._historical_df) < self.window + self.future_bars + 10:
            return self._empty_result()

        # Normalize current window
        current_prices = current_window.values[-self.window:]
        if len(current_prices) < self.window:
            return self._empty_result()

        current_norm = _normalize_window(current_prices)

        # Extract features: include RSI-like momentum and trend direction
        hist_close = self._historical_df['close'].values
        matches = []

        max_start = len(hist_close) - self.window - self.future_bars - 1
        if max_start < self.window:
            return self._empty_result()

        for i in range(self.window, max_start, 2):  # Step 2 for speed
            hist_window = hist_close[i - self.window: i]
            hist_norm = _normalize_window(hist_window)
            similarity = _cosine_similarity(current_norm, hist_norm)

            if similarity > 0.85:  # Only high-similarity matches
                # What happened in the next N bars?
                future_prices = hist_close[i: i + self.future_bars]
                if len(future_prices) < self.future_bars:
                    continue

                entry_price = hist_close[i]
                future_high = float(np.max(future_prices))
                future_low = float(np.min(future_prices))
                future_close = float(future_prices[-1])

                move_pct = (future_close - entry_price) / entry_price * 100
                max_up_pct = (future_high - entry_price) / entry_price * 100
                max_down_pct = (future_low - entry_price) / entry_price * 100

                t = self._historical_df.index[i]
                t_str = t.isoformat() if hasattr(t, 'isoformat') else str(t)

                matches.append({
                    'date': t_str,
                    'similarity': round(float(similarity), 4),
                    'entry_price': round(float(entry_price), 2),
                    'move_pct': round(move_pct, 2),
                    'max_up_pct': round(max_up_pct, 2),
                    'max_down_pct': round(max_down_pct, 2),
                    'direction': 'UP' if move_pct > 0.1 else ('DOWN' if move_pct < -0.1 else 'FLAT'),
                })

        if not matches:
            return self._empty_result()

        # Sort by similarity, take top K
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:self.top_k]

        # Statistics
        directions = [m['direction'] for m in matches]
        moves = [m['move_pct'] for m in matches]
        max_ups = [m['max_up_pct'] for m in matches]
        max_downs = [m['max_down_pct'] for m in matches]

        bullish_count = directions.count('UP')
        bearish_count = directions.count('DOWN')
        total = len(matches)

        win_rate_long = round(bullish_count / total * 100, 1) if total else 50
        avg_move = round(float(np.mean(moves)), 2) if moves else 0
        avg_max_up = round(float(np.mean(max_ups)), 2) if max_ups else 0
        avg_max_down = round(float(np.mean(max_downs)), 2) if max_downs else 0
        avg_volatility = round(float(np.mean([m['max_up_pct'] - m['max_down_pct'] for m in matches])), 2) if matches else 0
        bias = 'BULLISH' if bullish_count > bearish_count else ('BEARISH' if bearish_count > bullish_count else 'NEUTRAL')

        return {
            'matches': matches,
            'statistics': {
                'total_matches': total,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'win_rate_long': win_rate_long,
                'win_rate_short': round(100 - win_rate_long, 1),
                'avg_move_pct': avg_move,
                'avg_max_up_pct': avg_max_up,
                'avg_max_down_pct': avg_max_down,
                'avg_volatility_pct': avg_volatility,
                'bias': bias,
                'confidence': round(abs(win_rate_long - 50) * 2, 1),
            },
            'summary': (
                f"Found {total} similar patterns. "
                f"{win_rate_long}% were bullish, {100 - win_rate_long:.0f}% bearish. "
                f"Avg move: {avg_move:+.2f}%, max range: {avg_max_up:.2f}% up / {avg_max_down:.2f}% down. "
                f"Historical bias: {bias}."
            ),
            'analysis_window_bars': self.window,
            'forecast_bars': self.future_bars,
        }

    def _empty_result(self) -> dict:
        return {
            'matches': [],
            'statistics': {
                'total_matches': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'win_rate_long': 50,
                'win_rate_short': 50,
                'avg_move_pct': 0,
                'avg_max_up_pct': 0,
                'avg_max_down_pct': 0,
                'avg_volatility_pct': 0,
                'bias': 'NEUTRAL',
                'confidence': 0,
            },
            'summary': 'Not enough historical data for pattern matching.',
            'analysis_window_bars': self.window,
            'forecast_bars': self.future_bars,
        }


# Global instance
pattern_matcher = PatternMatcher(window=20, top_k=10, future_bars=20)
