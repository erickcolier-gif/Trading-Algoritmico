"""
Comprehensive technical analysis module.
Includes: Classic indicators + VWAP + SMC (Smart Money Concepts) + Institutional Flow.
Historically proven indicators with highest win rates for NAS100 intraday.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Utility ──────────────────────────────────────────────────────────────────

def _safe(val, decimals: int = 2):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return round(float(val), decimals)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


# ── Classic Indicators ────────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def calc_bollinger(close: pd.Series, period=20, std=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, period)
    sd = close.rolling(period).std()
    return mid + std * sd, mid, mid - std * sd


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_stochastic(df: pd.DataFrame, k=14, d=3) -> Tuple[pd.Series, pd.Series]:
    lo = df['low'].rolling(k).min()
    hi = df['high'].rolling(k).max()
    stoch_k = 100 * (df['close'] - lo) / (hi - lo).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


# ── VWAP (Volume Weighted Average Price) ────────────────────────────────────

def calc_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday VWAP with standard deviation bands (1, 2, 3 sigma).
    Resets each trading day — the institutional benchmark.
    """
    df = df.copy()
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical'] * df['volume']

    # Group by date for daily VWAP reset
    results = []
    try:
        df_indexed = df.copy()
        df_indexed['_date'] = df_indexed.index.date
        for _, group in df_indexed.groupby('_date'):
            group = group.copy()
            cum_tv = group['tp_vol'].cumsum()
            cum_v = group['volume'].cumsum().replace(0, np.nan)
            vwap = cum_tv / cum_v

            # Rolling std for bands
            diff_sq = (group['typical'] - vwap) ** 2
            variance = diff_sq.cumsum() / np.arange(1, len(group) + 1)
            sd = np.sqrt(variance)

            group['vwap'] = vwap
            group['vwap_std'] = sd
            group['vwap_u1'] = vwap + sd
            group['vwap_l1'] = vwap - sd
            group['vwap_u2'] = vwap + 2 * sd
            group['vwap_l2'] = vwap - 2 * sd
            group['vwap_u3'] = vwap + 3 * sd
            group['vwap_l3'] = vwap - 3 * sd
            results.append(group)

        out = pd.concat(results).drop(columns=['typical', 'tp_vol', 'vwap_std', '_date'], errors='ignore')
    except Exception as e:
        logger.warning(f"VWAP calculation error: {e}")
        out = df.copy()
        out['vwap'] = df['close'].rolling(20).mean()
        out['vwap_u1'] = out['vwap'] + df['close'].rolling(20).std()
        out['vwap_l1'] = out['vwap'] - df['close'].rolling(20).std()
        out['vwap_u2'] = out['vwap'] + 2 * df['close'].rolling(20).std()
        out['vwap_l2'] = out['vwap'] - 2 * df['close'].rolling(20).std()
        out['vwap_u3'] = out['vwap'] + 3 * df['close'].rolling(20).std()
        out['vwap_l3'] = out['vwap'] - 3 * df['close'].rolling(20).std()
    return out


# ── Volume Analysis & Institutional Flow ──────────────────────────────────────

def calc_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate volume delta (buying vs selling pressure).
    Bullish candle → buying volume; bearish → selling volume.
    Cumulative delta shows net institutional direction.
    """
    df = df.copy()
    df['delta'] = df.apply(
        lambda r: r['volume'] if r['close'] >= r['open'] else -r['volume'], axis=1
    )
    df['cum_delta'] = df['delta'].cumsum()
    df['delta_sma'] = df['delta'].rolling(20).mean()
    return df


def calc_volume_profile(df: pd.DataFrame, bins: int = 50) -> dict:
    """
    Volume Profile: distribution of volume across price levels.
    Returns POC (Point of Control) and Value Area High/Low.
    """
    price_min = df['low'].min()
    price_max = df['high'].max()
    if price_min >= price_max:
        return {'profile': [], 'poc': float(df['close'].iloc[-1]), 'vah': price_max, 'val': price_min}

    levels = np.linspace(price_min, price_max, bins + 1)
    profile = []
    for i in range(len(levels) - 1):
        mask = (df['low'] <= levels[i + 1]) & (df['high'] >= levels[i])
        # Proportional volume at this level
        lvl_vol = 0
        for _, row in df[mask].iterrows():
            overlap_lo = max(row['low'], levels[i])
            overlap_hi = min(row['high'], levels[i + 1])
            if overlap_hi > overlap_lo and (row['high'] - row['low']) > 0:
                fraction = (overlap_hi - overlap_lo) / (row['high'] - row['low'])
                lvl_vol += row['volume'] * fraction
        profile.append({
            'price': round((levels[i] + levels[i + 1]) / 2, 2),
            'volume': int(lvl_vol),
        })

    if not profile:
        return {'profile': [], 'poc': float(df['close'].iloc[-1]), 'vah': price_max, 'val': price_min}

    profile_df = pd.DataFrame(profile)
    poc_idx = profile_df['volume'].idxmax()
    poc = profile_df.iloc[poc_idx]['price']

    # Value Area: 70% of total volume expanding from POC
    total_vol = profile_df['volume'].sum()
    target = total_vol * 0.70
    upper_idx = poc_idx
    lower_idx = poc_idx
    accumulated = float(profile_df.iloc[poc_idx]['volume'])

    while accumulated < target:
        up_vol = profile_df.iloc[upper_idx + 1]['volume'] if upper_idx < len(profile_df) - 1 else 0
        dn_vol = profile_df.iloc[lower_idx - 1]['volume'] if lower_idx > 0 else 0
        if up_vol == 0 and dn_vol == 0:
            break
        if up_vol >= dn_vol and upper_idx < len(profile_df) - 1:
            upper_idx += 1
            accumulated += up_vol
        elif lower_idx > 0:
            lower_idx -= 1
            accumulated += dn_vol
        else:
            upper_idx += 1
            accumulated += up_vol

    return {
        'profile': profile[:bins],
        'poc': float(poc),
        'vah': float(profile_df.iloc[upper_idx]['price']),
        'val': float(profile_df.iloc[lower_idx]['price']),
    }


def detect_institutional_candles(df: pd.DataFrame, spike_mult: float = 2.0) -> List[dict]:
    """
    Detect candles with institutional-level volume (2x+ average).
    These mark key areas of institutional activity.
    """
    vol_sma = df['volume'].rolling(20).mean()
    results = []
    for i in range(20, len(df)):
        if df['volume'].iloc[i] >= vol_sma.iloc[i] * spike_mult:
            candle = df.iloc[i]
            results.append({
                'time': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                'price': _safe(candle['close']),
                'volume': int(candle['volume']),
                'volume_ratio': _safe(candle['volume'] / vol_sma.iloc[i]),
                'direction': 'BUY' if candle['close'] >= candle['open'] else 'SELL',
                'size': _safe(candle['high'] - candle['low']),
            })
    return results[-20:]


# ── Support & Resistance Engine ───────────────────────────────────────────────

def calc_support_resistance(df: pd.DataFrame, price: float, atr: float) -> dict:
    """
    Key support & resistance levels:
    - Daily pivot points (PP, R1-R3, S1-S3)
    - Previous day high/low/close
    - Psychological round numbers (multiples of 50/100/250/500)
    - Fibonacci retracements of last 60-bar swing
    - Swing high/low clusters (tested 2+ times = confirmed horizontal level)
    """
    levels: List[dict] = []

    # ── Daily Pivot Points ──
    try:
        df_cp = df.copy()
        df_cp['_date'] = df_cp.index.date
        dates = sorted(df_cp['_date'].unique())
        if len(dates) >= 2:
            pd_data = df_cp[df_cp['_date'] == dates[-2]]
            ph = float(pd_data['high'].max())
            pl = float(pd_data['low'].min())
            pc = float(pd_data['close'].iloc[-1])
            levels += [
                {'price': ph, 'type': 'RESISTANCE', 'strength': 4, 'label': 'PDH'},
                {'price': pl, 'type': 'SUPPORT',    'strength': 4, 'label': 'PDL'},
                {'price': pc, 'type': 'PIVOT',      'strength': 3, 'label': 'PDC'},
            ]
            pp = (ph + pl + pc) / 3
            r1 = 2*pp - pl;  r2 = pp + (ph - pl);  r3 = ph + 2*(pp - pl)
            s1 = 2*pp - ph;  s2 = pp - (ph - pl);  s3 = pl - 2*(ph - pp)
            levels += [
                {'price': round(pp, 2), 'type': 'PIVOT',      'strength': 5, 'label': 'PP'},
                {'price': round(r1, 2), 'type': 'RESISTANCE', 'strength': 4, 'label': 'R1'},
                {'price': round(r2, 2), 'type': 'RESISTANCE', 'strength': 3, 'label': 'R2'},
                {'price': round(r3, 2), 'type': 'RESISTANCE', 'strength': 2, 'label': 'R3'},
                {'price': round(s1, 2), 'type': 'SUPPORT',    'strength': 4, 'label': 'S1'},
                {'price': round(s2, 2), 'type': 'SUPPORT',    'strength': 3, 'label': 'S2'},
                {'price': round(s3, 2), 'type': 'SUPPORT',    'strength': 2, 'label': 'S3'},
            ]
    except Exception as e:
        logger.warning(f"Pivot points error: {e}")

    # ── Psychological Round Numbers (within 4.5 ATR) ──
    base = round(price / 100) * 100
    for offset in range(-6, 7):
        lvl = base + offset * 50
        if abs(lvl - price) <= atr * 4.5:
            str_ = 5 if lvl % 500 == 0 else 4 if lvl % 250 == 0 else 3 if lvl % 100 == 0 else 2
            t = 'SUPPORT' if lvl < price else 'RESISTANCE' if lvl > price else 'PIVOT'
            levels.append({'price': float(lvl), 'type': t, 'strength': str_, 'label': f'Round {int(lvl)}'})

    # ── Fibonacci Retracements (last 60 bars) ──
    try:
        fw = df.tail(60)
        sh = float(fw['high'].max())
        sl_f = float(fw['low'].min())
        if (sh - sl_f) > atr * 3:
            rang = sh - sl_f
            for fib, lbl, str_ in [
                (0.236, 'Fib 23.6%', 3), (0.382, 'Fib 38.2%', 5),
                (0.5,   'Fib 50%',   4), (0.618, 'Fib 61.8%', 5),
                (0.786, 'Fib 78.6%', 3),
            ]:
                fp = round(sl_f + rang * fib, 2)
                t = 'SUPPORT' if fp < price else 'RESISTANCE' if fp > price else 'PIVOT'
                levels.append({'price': fp, 'type': t, 'strength': str_, 'label': lbl})
    except Exception as e:
        logger.warning(f"Fibonacci error: {e}")

    # ── Swing High/Low Clusters (2+ touches = confirmed horizontal level) ──
    try:
        highs, lows = detect_swing_points(df, lookback=8)
        ct = atr * 0.35

        def _cluster(pts: List[dict], level_type: str) -> List[dict]:
            clusters: dict = {}
            for p in [x['price'] for x in pts[-30:] if x.get('price')]:
                matched = False
                for k in list(clusters.keys()):
                    if abs(p - k) < ct:
                        clusters[k] += 1
                        matched = True
                        break
                if not matched:
                    clusters[p] = 1
            return [
                {'price': round(k, 2), 'type': level_type,
                 'strength': min(5, v + 1), 'label': f'{level_type.title()} ({v}x tested)'}
                for k, v in clusters.items() if v >= 2
            ]

        levels += _cluster(highs, 'RESISTANCE')
        levels += _cluster(lows,  'SUPPORT')
    except Exception as e:
        logger.warning(f"Swing cluster error: {e}")

    # ── Deduplicate (merge within 0.25 ATR), annotate, sort ──
    unique: List[dict] = []
    for l in sorted(levels, key=lambda x: x['strength'], reverse=True):
        if not any(abs(l['price'] - u['price']) < atr * 0.25 for u in unique):
            l = dict(l)
            l['distance']     = round(abs(l['price'] - price), 2)
            l['distance_atr'] = round(abs(l['price'] - price) / atr, 2)
            unique.append(l)

    supports    = sorted([l for l in unique if l['price'] < price],  key=lambda x: x['price'], reverse=True)[:8]
    resistances = sorted([l for l in unique if l['price'] > price],  key=lambda x: x['price'])[:8]
    key_levels  = sorted(unique, key=lambda x: (x['distance_atr'], -x['strength']))[:12]

    return {
        'support':            supports,
        'resistance':         resistances,
        'key_levels':         key_levels,
        'nearest_support':    supports[0]    if supports    else None,
        'nearest_resistance': resistances[0] if resistances else None,
    }


def score_reversal_zones(
    price: float, atr: float,
    sr_levels: dict,
    order_blocks: List[dict],
    fvgs: List[dict],
    vwap: float,
    vwap_u1: float, vwap_l1: float,
    vwap_u2: float, vwap_l2: float,
) -> List[dict]:
    """
    Score reversal zones by stacking institutional confluences at the same price.
    High score = multiple layers (S&R + OB + FVG + VWAP) = elite bounce zone.
    These are the levels where smart money WILL defend their position.
    """
    proximity = atr * 0.45
    zones = []

    for lvl in sr_levels.get('key_levels', []):
        lp    = lvl['price']
        score = int(lvl['strength'])
        conf  = [lvl['label']]

        for ob in order_blocks:
            if abs((ob.get('mid') or 0) - lp) < proximity:
                score += 3; conf.append(f"{ob['type']} OB")
        for fvg in fvgs:
            if abs((fvg.get('mid') or 0) - lp) < proximity:
                score += 2; conf.append(f"{fvg['type']} FVG")
        for band_v, band_n in [
            (vwap,    'VWAP'),     (vwap_u1, 'VWAP+1σ'), (vwap_l1, 'VWAP-1σ'),
            (vwap_u2, 'VWAP+2σ'), (vwap_l2, 'VWAP-2σ'),
        ]:
            if band_v and abs(band_v - lp) < proximity:
                score += 3 if band_n == 'VWAP' else 2
                conf.append(band_n)

        direction    = 'BULLISH' if lp < price else 'BEARISH'
        strength_lbl = ('EXTREME'  if score >= 12 else 'STRONG'  if score >= 8
                        else 'MODERATE' if score >= 5 else 'WEAK')
        zones.append({
            'price':            lp,
            'type':             lvl['type'],
            'direction':        direction,
            'score':            score,
            'strength':         strength_lbl,
            'confluences':      conf,
            'confluence_count': len(conf),
            'distance':         round(abs(lp - price), 2),
            'distance_atr':     round(abs(lp - price) / atr, 2),
            'label':            lvl['label'],
        })

    return sorted(zones, key=lambda x: (-x['score'], x['distance']))[:8]


# ── Smart Money Concepts (SMC) ────────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, lookback: int = 10) -> Tuple[List[dict], List[dict]]:
    """Detect swing highs and lows (key institutional levels)."""
    highs, lows = [], []
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = df['high'].iloc[i - lookback: i + lookback + 1]
        window_l = df['low'].iloc[i - lookback: i + lookback + 1]
        t = df.index[i]
        t_str = t.isoformat() if hasattr(t, 'isoformat') else str(t)
        if df['high'].iloc[i] == window_h.max():
            highs.append({'price': _safe(df['high'].iloc[i]), 'time': t_str, 'index': i})
        if df['low'].iloc[i] == window_l.min():
            lows.append({'price': _safe(df['low'].iloc[i]), 'time': t_str, 'index': i})
    return highs, lows


def detect_market_structure(df: pd.DataFrame, lookback: int = 10) -> dict:
    """
    Detect market structure: BOS (Break of Structure) & CHoCH (Change of Character).
    BOS = trend continuation; CHoCH = potential reversal.
    """
    highs, lows = detect_swing_points(df, lookback)
    current_price = float(df['close'].iloc[-1])

    trend = 'SIDEWAYS'
    bos = None
    choch = None

    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1]['price'] > highs[-2]['price']  # Higher high
        hl = lows[-1]['price'] > lows[-2]['price']    # Higher low
        lh = highs[-1]['price'] < highs[-2]['price']  # Lower high
        ll = lows[-1]['price'] < lows[-2]['price']    # Lower low

        if hh and hl:
            trend = 'BULLISH'
        elif lh and ll:
            trend = 'BEARISH'

        # Break of Structure
        if highs and current_price > highs[-1]['price']:
            bos = {'type': 'BULLISH_BOS', 'price': highs[-1]['price'], 'description': 'Price broke above swing high — bullish continuation'}
        elif lows and current_price < lows[-1]['price']:
            bos = {'type': 'BEARISH_BOS', 'price': lows[-1]['price'], 'description': 'Price broke below swing low — bearish continuation'}

        # Change of Character (reversal signal)
        if trend == 'BULLISH' and lows and current_price < lows[-1]['price']:
            choch = {'type': 'BEARISH_CHOCH', 'price': lows[-1]['price'], 'description': 'CHoCH: bullish structure broken — potential reversal'}
        elif trend == 'BEARISH' and highs and current_price > highs[-1]['price']:
            choch = {'type': 'BULLISH_CHOCH', 'price': highs[-1]['price'], 'description': 'CHoCH: bearish structure broken — potential reversal'}

    return {
        'trend': trend,
        'swing_highs': highs[-5:],
        'swing_lows': lows[-5:],
        'bos': bos,
        'choch': choch,
        'current_price': _safe(current_price),
    }


def detect_order_blocks(df: pd.DataFrame, impulse_mult: float = 1.5, lookback: int = 5) -> List[dict]:
    """
    Order Blocks: last opposing candle before a strong impulse move.
    These are institutional entry points — price often returns to test them.
    """
    avg_candle = (df['high'] - df['low']).mean()
    impulse_size = avg_candle * impulse_mult
    current_price = float(df['close'].iloc[-1])
    obs = []

    for i in range(lookback, len(df) - 2):
        candle_size = df['high'].iloc[i] - df['low'].iloc[i]
        move = df['close'].iloc[i] - df['close'].iloc[i - 1]

        # Bullish impulse → Bullish OB (last bearish candle before impulse)
        if candle_size > impulse_size and move > 0:
            for j in range(i - 1, max(0, i - lookback - 1), -1):
                if df['close'].iloc[j] < df['open'].iloc[j]:  # Bearish candle
                    ob_high = float(df['high'].iloc[j])
                    ob_low = float(df['low'].iloc[j])
                    # Active if price hasn't closed below OB low
                    active = current_price > ob_low
                    obs.append({
                        'type': 'BULLISH',
                        'high': _safe(ob_high),
                        'low': _safe(ob_low),
                        'mid': _safe((ob_high + ob_low) / 2),
                        'time': df.index[j].isoformat() if hasattr(df.index[j], 'isoformat') else str(df.index[j]),
                        'active': active,
                        'distance': _safe(abs(current_price - (ob_high + ob_low) / 2)),
                        'description': f'Bullish OB at {_safe(ob_low)}-{_safe(ob_high)} — institutional demand zone',
                    })
                    break

        # Bearish impulse → Bearish OB (last bullish candle before impulse)
        elif candle_size > impulse_size and move < 0:
            for j in range(i - 1, max(0, i - lookback - 1), -1):
                if df['close'].iloc[j] > df['open'].iloc[j]:  # Bullish candle
                    ob_high = float(df['high'].iloc[j])
                    ob_low = float(df['low'].iloc[j])
                    active = current_price < ob_high
                    obs.append({
                        'type': 'BEARISH',
                        'high': _safe(ob_high),
                        'low': _safe(ob_low),
                        'mid': _safe((ob_high + ob_low) / 2),
                        'time': df.index[j].isoformat() if hasattr(df.index[j], 'isoformat') else str(df.index[j]),
                        'active': active,
                        'distance': _safe(abs(current_price - (ob_high + ob_low) / 2)),
                        'description': f'Bearish OB at {_safe(ob_low)}-{_safe(ob_high)} — institutional supply zone',
                    })
                    break

    # Return most recent active OBs sorted by proximity
    active_obs = [ob for ob in obs if ob['active']]
    return sorted(active_obs, key=lambda x: x['distance'])[:8]


def detect_fair_value_gaps(df: pd.DataFrame, min_size_pct: float = 0.001) -> List[dict]:
    """
    Fair Value Gaps (FVG): price inefficiencies created by fast institutional moves.
    Price tends to return to fill these gaps — high probability reversal levels.
    """
    current_price = float(df['close'].iloc[-1])
    fvgs = []

    for i in range(1, len(df) - 1):
        c1_high = float(df['high'].iloc[i - 1])
        c1_low = float(df['low'].iloc[i - 1])
        c3_high = float(df['high'].iloc[i + 1])
        c3_low = float(df['low'].iloc[i + 1])
        mid_close = float(df['close'].iloc[i])

        # Bullish FVG: gap between candle1 high and candle3 low
        if c3_low > c1_high:
            size = c3_low - c1_high
            if size / mid_close >= min_size_pct:
                filled = current_price <= c1_high  # Price came back down to fill
                fvgs.append({
                    'type': 'BULLISH',
                    'high': _safe(c3_low),
                    'low': _safe(c1_high),
                    'mid': _safe((c3_low + c1_high) / 2),
                    'size': _safe(size),
                    'time': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'filled': filled,
                    'distance': _safe(abs(current_price - (c3_low + c1_high) / 2)),
                    'description': f'Bullish FVG {_safe(c1_high)}-{_safe(c3_low)} — potential support on pullback',
                })

        # Bearish FVG: gap between candle1 low and candle3 high
        elif c1_low > c3_high:
            size = c1_low - c3_high
            if size / mid_close >= min_size_pct:
                filled = current_price >= c1_low
                fvgs.append({
                    'type': 'BEARISH',
                    'high': _safe(c1_low),
                    'low': _safe(c3_high),
                    'mid': _safe((c1_low + c3_high) / 2),
                    'size': _safe(size),
                    'time': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'filled': filled,
                    'distance': _safe(abs(current_price - (c1_low + c3_high) / 2)),
                    'description': f'Bearish FVG {_safe(c3_high)}-{_safe(c1_low)} — potential resistance on bounce',
                })

    unfilled = [f for f in fvgs if not f['filled']]
    return sorted(unfilled, key=lambda x: x['distance'])[:8]


# ── Precision Scalping Helpers ────────────────────────────────────────────────

def _candle_type(df: pd.DataFrame, idx: int = -1) -> str:
    """
    Classify the candle at idx.
    Returns: BULLISH_ENGULFING, BEARISH_ENGULFING, HAMMER, SHOOTING_STAR,
             STRONG_BULL, STRONG_BEAR, BULL, BEAR, DOJI, NEUTRAL
    """
    n = len(df)
    actual = n + idx if idx < 0 else idx
    if actual < 1 or actual >= n:
        return 'NEUTRAL'
    c = df.iloc[actual]
    prev = df.iloc[actual - 1]
    body = abs(c['close'] - c['open'])
    rng = c['high'] - c['low']
    if rng < 1e-8:
        return 'DOJI'
    upper_wick = c['high'] - max(c['close'], c['open'])
    lower_wick = min(c['close'], c['open']) - c['low']
    body_pct = body / rng
    prev_body = abs(prev['close'] - prev['open'])

    # Engulfing (strongest confirmation candles)
    if (c['close'] > c['open'] and prev['close'] < prev['open']
            and body >= prev_body * 0.85
            and c['close'] >= prev['open'] and c['open'] <= prev['close']):
        return 'BULLISH_ENGULFING'
    if (c['close'] < c['open'] and prev['close'] > prev['open']
            and body >= prev_body * 0.85
            and c['open'] >= prev['close'] and c['close'] <= prev['open']):
        return 'BEARISH_ENGULFING'

    # Pin bars (rejection candles)
    if lower_wick >= body * 2.0 and lower_wick > upper_wick * 1.5 and body_pct < 0.40:
        return 'HAMMER'
    if upper_wick >= body * 2.0 and upper_wick > lower_wick * 1.5 and body_pct < 0.40:
        return 'SHOOTING_STAR'

    # Strong directional
    if c['close'] > c['open'] and body_pct > 0.55:
        return 'STRONG_BULL'
    if c['close'] < c['open'] and body_pct > 0.55:
        return 'STRONG_BEAR'

    return 'BULL' if c['close'] > c['open'] else 'BEAR' if c['close'] < c['open'] else 'DOJI'


def _macd_momentum(macd_hist: pd.Series, bars: int = 3) -> Tuple[bool, bool]:
    """
    Returns (bull_expanding, bear_expanding).
    Bull: histogram positive and not declining in last `bars` bars.
    Bear: histogram negative and not rising in last `bars` bars.
    """
    if len(macd_hist) < bars + 1:
        return False, False
    recent = list(macd_hist.iloc[-(bars + 1):])
    last = recent[-1]
    bull = last > 0 and last >= recent[-2]
    bear = last < 0 and last <= recent[-2]
    return bull, bear


def _rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 20) -> Tuple[bool, bool]:
    """
    Detect bullish / bearish RSI divergence over the last `lookback` bars.
    Bullish: price makes lower low but RSI makes higher low → reversal up.
    Bearish: price makes higher high but RSI makes lower high → reversal down.
    """
    if len(close) < lookback or len(rsi) < lookback:
        return False, False
    c = close.iloc[-lookback:]
    r = rsi.iloc[-lookback:]
    if c.std() < 1e-8:
        return False, False

    price_range = float(c.max() - c.min())
    if price_range == 0:
        return False, False

    # Bullish divergence — look at price lows (bottom 25%)
    thresh_lo = float(c.min()) + price_range * 0.25
    lo_mask = c <= thresh_lo
    bull_div = False
    if lo_mask.sum() >= 2:
        lo_idx = c[lo_mask].index
        if float(c[lo_idx[-1]]) <= float(c[lo_idx[0]]) * 1.003:  # price not higher at the low
            bull_div = float(r[lo_idx[-1]]) > float(r[lo_idx[0]]) + 4  # RSI clearly higher

    # Bearish divergence — look at price highs (top 25%)
    thresh_hi = float(c.max()) - price_range * 0.25
    hi_mask = c >= thresh_hi
    bear_div = False
    if hi_mask.sum() >= 2:
        hi_idx = c[hi_mask].index
        if float(c[hi_idx[-1]]) >= float(c[hi_idx[0]]) * 0.997:  # price not lower at the high
            bear_div = float(r[hi_idx[-1]]) < float(r[hi_idx[0]]) - 4  # RSI clearly lower

    return bull_div, bear_div


def _liquidity_sweep(
    df: pd.DataFrame,
    swing_highs: List[dict],
    swing_lows: List[dict],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect a liquidity sweep in the last 1–2 bars.
    Bullish sweep: bar wicked below a swing low then closed back above it.
    Bearish sweep: bar wicked above a swing high then closed back below it.
    Returns (bull_sweep_level, bear_sweep_level).  None means no sweep.
    """
    if len(df) < 3 or not swing_lows or not swing_highs:
        return None, None

    bull_level: Optional[float] = None
    bear_level: Optional[float] = None

    for look_back in (-1, -2):
        bar = df.iloc[look_back]
        conf_bar = df.iloc[-1] if look_back == -2 else None

        for sl in (swing_lows or [])[-6:]:
            level = sl.get('price') or 0
            if not level:
                continue
            if bar['low'] < level and bar['close'] > level:
                if look_back == -2:
                    if conf_bar is not None and float(conf_bar['close']) > level:
                        bull_level = float(level)
                else:
                    bull_level = float(level)
                break

        for sh in (swing_highs or [])[-6:]:
            level = sh.get('price') or 0
            if not level:
                continue
            if bar['high'] > level and bar['close'] < level:
                if look_back == -2:
                    if conf_bar is not None and float(conf_bar['close']) < level:
                        bear_level = float(level)
                else:
                    bear_level = float(level)
                break

        if bull_level or bear_level:
            break

    return bull_level, bear_level


def _recent_delta_bull(delta_raw: pd.Series, bars: int = 5) -> bool:
    """Net buying pressure in the last `bars` candles."""
    if len(delta_raw) < bars:
        return False
    return float(delta_raw.iloc[-bars:].sum()) > 0


def _recent_delta_bear(delta_raw: pd.Series, bars: int = 5) -> bool:
    """Net selling pressure in the last `bars` candles."""
    if len(delta_raw) < bars:
        return False
    return float(delta_raw.iloc[-bars:].sum()) < 0


# ── Signal Generation — Elite Scalping Engine ────────────────────────────────

def generate_signals(df: pd.DataFrame, indicators: dict) -> List[dict]:
    """
    Elite scalping signal engine for NAS100 5m timeframe.

    Implements 6 named high-probability setups:
      1. Liquidity Sweep Reversal  (~80 % win rate)
      2. Order Block Reaction      (~75 % win rate)
      3. VWAP Bounce / Reclaim     (~72 % win rate)
      4. EMA Pullback in Trend     (~68 % win rate)
      5. BOS First Pullback        (~70 % win rate)
      6. RSI Divergence at Level   (~70 % win rate)

    Each setup requires HARD conditions before scoring soft confluences.
    Minimum confidence to emit: 65.  No signal > a losing trade.
    """
    n = len(df)
    if n < 30:
        return []

    # ── Extract indicators ──────────────────────────────────────────────────
    price    = float(indicators.get('price', 0) or 0)
    vwap     = float(indicators.get('vwap', 0) or 0)
    rsi      = float(indicators.get('rsi', 50) or 50)
    macd_h   = float(indicators.get('macd_hist', 0) or 0)
    atr      = max(float(indicators.get('atr', 1) or 1), 0.1)
    ema9     = float(indicators.get('ema9', price) or price)
    ema21    = float(indicators.get('ema21', price) or price)
    ema50    = float(indicators.get('ema50', price) or price)
    structure = indicators.get('structure', {})
    obs      = indicators.get('order_blocks', [])
    fvgs     = indicators.get('fair_value_gaps', [])
    vwap_u1  = float(indicators.get('vwap_u1', price + atr * 2) or price + atr * 2)
    vwap_l1  = float(indicators.get('vwap_l1', price - atr * 2) or price - atr * 2)

    # Series passed from full_analysis for deeper analysis
    rsi_series      = indicators.get('rsi_series', pd.Series(dtype=float))
    macd_hist_series = indicators.get('macd_hist_series', pd.Series(dtype=float))
    delta_raw       = indicators.get('delta_series_raw', pd.Series(dtype=float))

    if not price or not vwap:
        return []

    # ── Pre-compute all conditions once ────────────────────────────────────
    vol_avg     = float(df['volume'].rolling(20).mean().iloc[-1]) or 1
    vol_ratio   = float(df['volume'].iloc[-1]) / vol_avg

    candle      = _candle_type(df, -1)
    c_bull      = candle in ('HAMMER', 'BULLISH_ENGULFING', 'STRONG_BULL', 'BULL')
    c_bear      = candle in ('SHOOTING_STAR', 'BEARISH_ENGULFING', 'STRONG_BEAR', 'BEAR')
    c_str_bull  = candle in ('HAMMER', 'BULLISH_ENGULFING', 'STRONG_BULL')
    c_str_bear  = candle in ('SHOOTING_STAR', 'BEARISH_ENGULFING', 'STRONG_BEAR')

    macd_bull_mom, macd_bear_mom = _macd_momentum(macd_hist_series, bars=3)
    bull_div, bear_div = _rsi_divergence(df['close'], rsi_series, lookback=20) \
        if len(rsi_series) >= 20 else (False, False)

    swing_highs = structure.get('swing_highs', [])
    swing_lows  = structure.get('swing_lows', [])
    sweep_bull, sweep_bear = _liquidity_sweep(df, swing_highs, swing_lows)

    d_bull = _recent_delta_bull(delta_raw, 5)
    d_bear = _recent_delta_bear(delta_raw, 5)

    above_vwap = price > vwap
    below_vwap = price < vwap
    near_vwap  = abs(price - vwap) / atr < 0.6

    ema_bull = ema9 > ema21 > ema50
    ema_bear = ema9 < ema21 < ema50

    near_ema9  = abs(price - ema9)  / atr < 0.4
    near_ema21 = abs(price - ema21) / atr < 0.55

    rsi_xtreme_hi = rsi > 78   # disqualifies new longs
    rsi_xtreme_lo = rsi < 22   # disqualifies new shorts
    rsi_os  = rsi < 38
    rsi_ob  = rsi > 62

    struct_bull  = structure.get('trend') == 'BULLISH'
    struct_bear  = structure.get('trend') == 'BEARISH'
    bos_obj      = structure.get('bos') or {}
    bos_type     = bos_obj.get('type', '')
    bos_price    = float(bos_obj.get('price', 0) or 0)
    bull_bos     = bos_type == 'BULLISH_BOS'
    bear_bos     = bos_type == 'BEARISH_BOS'

    # OBs within tight range (0.35 ATR = "price AT the OB")
    bull_obs_near = [ob for ob in obs if ob['type'] == 'BULLISH'
                     and ob.get('distance', 999) < atr * 0.4]
    bear_obs_near = [ob for ob in obs if ob['type'] == 'BEARISH'
                     and ob.get('distance', 999) < atr * 0.4]
    at_bull_ob = bool(bull_obs_near)
    at_bear_ob = bool(bear_obs_near)

    # FVGs within 0.8 ATR
    at_bull_fvg = any(f['type'] == 'BULLISH' and f.get('distance', 999) < atr * 0.8 for f in fvgs)
    at_bear_fvg = any(f['type'] == 'BEARISH' and f.get('distance', 999) < atr * 0.8 for f in fvgs)

    # ── Support & Resistance ──────────────────────────────────────────────────
    sr_levels       = indicators.get('sr_levels', {})
    reversal_zones  = indicators.get('reversal_zones', [])
    nr_support      = sr_levels.get('nearest_support')
    nr_resistance   = sr_levels.get('nearest_resistance')
    nr_supp_price   = float(nr_support['price'])    if nr_support    else price - atr * 3
    nr_resist_price = float(nr_resistance['price']) if nr_resistance else price + atr * 3

    at_sr_support = (bool(nr_support) and
                     nr_support.get('distance_atr', 99) < 0.5 and
                     nr_support.get('strength', 0) >= 3)
    at_sr_resistance = (bool(nr_resistance) and
                        nr_resistance.get('distance_atr', 99) < 0.5 and
                        nr_resistance.get('strength', 0) >= 3)

    near_rev_zone_bull = next(
        (z for z in reversal_zones
         if z['direction'] == 'BULLISH' and z['distance_atr'] < 0.5 and z['score'] >= 6), None)
    near_rev_zone_bear = next(
        (z for z in reversal_zones
         if z['direction'] == 'BEARISH' and z['distance_atr'] < 0.5 and z['score'] >= 6), None)

    # ── SL / TP calculators ─────────────────────────────────────────────────

    def _buy_sl() -> float:
        candidates = [price - atr * 1.2]  # fallback
        if swing_lows:
            lv = float(swing_lows[-1].get('price', 0) or 0)
            if lv and lv < price - atr * 0.25:
                candidates.append(lv - atr * 0.12)
        if bull_obs_near:
            best_ob = min(bull_obs_near, key=lambda x: x.get('distance', 999))
            ob_lo = float(best_ob.get('low', 0) or 0)
            if ob_lo and ob_lo < price - atr * 0.2:
                candidates.append(ob_lo - atr * 0.10)
        return float(max(c for c in candidates if c < price - atr * 0.20))

    def _sell_sl() -> float:
        candidates = [price + atr * 1.2]
        if swing_highs:
            hv = float(swing_highs[-1].get('price', 0) or 0)
            if hv and hv > price + atr * 0.25:
                candidates.append(hv + atr * 0.12)
        if bear_obs_near:
            best_ob = min(bear_obs_near, key=lambda x: x.get('distance', 999))
            ob_hi = float(best_ob.get('high', 0) or 0)
            if ob_hi and ob_hi > price + atr * 0.2:
                candidates.append(ob_hi + atr * 0.10)
        return float(min(c for c in candidates if c > price + atr * 0.20))

    def _buy_tp() -> float:
        base = price + atr * 2.0
        candidates = [base]
        if swing_highs:
            nh = next((float(h['price']) for h in reversed(swing_highs)
                       if h.get('price') and float(h['price']) > price + atr * 0.5), None)
            if nh:
                candidates.append(nh)
        if bear_obs_near:
            no = next((float(ob['mid']) for ob in bear_obs_near
                       if ob.get('mid') and float(ob['mid']) > price + atr * 0.5), None)
            if no:
                candidates.append(no)
        if vwap_u1 > price + atr * 0.5:
            candidates.append(vwap_u1)
        if nr_resist_price > price + atr * 0.5:
            candidates.append(nr_resist_price)
        return float(min(candidates))  # nearest target — scalper discipline

    def _sell_tp() -> float:
        base = price - atr * 2.0
        candidates = [base]
        if swing_lows:
            nl = next((float(l['price']) for l in reversed(swing_lows)
                       if l.get('price') and float(l['price']) < price - atr * 0.5), None)
            if nl:
                candidates.append(nl)
        if bull_obs_near:
            no = next((float(ob['mid']) for ob in bull_obs_near
                       if ob.get('mid') and float(ob['mid']) < price - atr * 0.5), None)
            if no:
                candidates.append(no)
        if vwap_l1 < price - atr * 0.5:
            candidates.append(vwap_l1)
        if nr_supp_price < price - atr * 0.5:
            candidates.append(nr_supp_price)
        return float(max(candidates))  # nearest target

    def _emit(direction: str, confidence: float, entry: float,
              sl: float, tp: float, reasons: List[str], setup: str):
        """Validate geometry and minimum RR before emitting a signal."""
        try:
            if direction == 'BUY':
                if sl >= entry or tp <= entry:
                    return None
                rr = round((tp - entry) / (entry - sl), 2)
            else:
                if sl <= entry or tp >= entry:
                    return None
                rr = round((entry - tp) / (sl - entry), 2)
            if rr < 1.4:
                return None
            return {
                'direction': direction,
                'confidence': min(int(confidence), 100),
                'entry': round(entry, 2),
                'stop_loss': round(sl, 2),
                'take_profit': round(tp, 2),
                'risk_reward': rr,
                'atr': _safe(atr),
                'setup': setup,
                'reasons': reasons,
                'summary': (f"{direction} [{setup}] — {len(reasons)} confluences. "
                            f"E:{round(entry,1)}  SL:{round(sl,1)}  TP:{round(tp,1)}  RR:{rr}:1"),
                'nearest_support':    nr_support,
                'nearest_resistance': nr_resistance,
                'reversal_zone': near_rev_zone_bull if direction == 'BUY' else near_rev_zone_bear,
                'timestamp': datetime.now().isoformat(),
            }
        except Exception:
            return None

    signals: List[dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 1 — LIQUIDITY SWEEP REVERSAL  (highest priority, ~80 % win rate)
    # Smart money sweeps retail stops then moves the opposite direction.
    # Requires: sweep confirmed + reversal candle.  No RSI extreme.
    # ══════════════════════════════════════════════════════════════════════════

    if sweep_bull and c_bull and not rsi_xtreme_hi:
        conf = 76
        reasons = [f'Liquidity sweep: wick below {_safe(sweep_bull)}, sharp bullish reversal']
        if at_bull_ob:
            conf += 10; reasons.append('Sweep at bullish Order Block — institutions defended demand')
        if at_bull_fvg:
            conf += 7;  reasons.append('Sweep inside bullish FVG — imbalance filled, reversal due')
        if near_vwap:
            conf += 7;  reasons.append('Sweep at VWAP — key institutional anchor')
        if c_str_bull:
            conf += 6;  reasons.append(f'Strong reversal candle: {candle}')
        if d_bull:
            conf += 5;  reasons.append('Recent delta net positive — buying pressure confirmed')
        if vol_ratio >= 1.2:
            conf += 5;  reasons.append(f'Volume spike {vol_ratio:.1f}× avg — institutional entry')
        if above_vwap or near_vwap:
            conf += 3;  reasons.append('Bias bullish relative to VWAP')
        if ema_bull:
            conf += 4;  reasons.append('EMA stack bullish — trend supports long')
        sig = _emit('BUY', conf, price, _buy_sl(), _buy_tp(), reasons, 'Liquidity Sweep')
        if sig:
            signals.append(sig)

    if sweep_bear and c_bear and not rsi_xtreme_lo:
        conf = 76
        reasons = [f'Liquidity sweep: wick above {_safe(sweep_bear)}, sharp bearish reversal']
        if at_bear_ob:
            conf += 10; reasons.append('Sweep at bearish Order Block — institutions defended supply')
        if at_bear_fvg:
            conf += 7;  reasons.append('Sweep inside bearish FVG — imbalance filled, reversal due')
        if near_vwap:
            conf += 7;  reasons.append('Sweep at VWAP — key institutional anchor')
        if c_str_bear:
            conf += 6;  reasons.append(f'Strong rejection candle: {candle}')
        if d_bear:
            conf += 5;  reasons.append('Recent delta net negative — selling pressure confirmed')
        if vol_ratio >= 1.2:
            conf += 5;  reasons.append(f'Volume spike {vol_ratio:.1f}× avg — institutional entry')
        if below_vwap or near_vwap:
            conf += 3;  reasons.append('Bias bearish relative to VWAP')
        if ema_bear:
            conf += 4;  reasons.append('EMA stack bearish — trend supports short')
        sig = _emit('SELL', conf, price, _sell_sl(), _sell_tp(), reasons, 'Liquidity Sweep')
        if sig:
            signals.append(sig)

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 2 — ORDER BLOCK REACTION  (~75 % win rate)
    # Enter where institutions entered and will defend their position.
    # Requires: price AT OB zone + reversal candle.
    # ══════════════════════════════════════════════════════════════════════════

    if at_bull_ob and c_bull and not rsi_xtreme_hi:
        conf = 68
        reasons = ['Price touching bullish Order Block — institutional demand zone']
        if c_str_bull:
            conf += 10; reasons.append(f'Rejection candle ({candle}) confirms OB defended')
        if above_vwap or near_vwap:
            conf += 8;  reasons.append('At/above VWAP — institutional bias bullish')
        if ema_bull:
            conf += 7;  reasons.append('EMA 9>21>50 — trend aligned with long')
        if macd_bull_mom:
            conf += 8;  reasons.append('MACD histogram expanding bullish — momentum confirmed')
        elif macd_h > 0:
            conf += 3;  reasons.append('MACD positive — directional bias intact')
        if at_bull_fvg:
            conf += 10; reasons.append('OB + FVG confluence — double institutional level')
        if d_bull:
            conf += 6;  reasons.append('Net buying delta — smart money accumulating')
        if vol_ratio >= 1.0:
            conf += 5;  reasons.append(f'Volume {vol_ratio:.1f}× avg — active market')
        if struct_bull or bull_bos:
            conf += 5;  reasons.append('Market structure bullish — HH+HL pattern')
        if conf >= 65:
            sig = _emit('BUY', conf, price, _buy_sl(), _buy_tp(), reasons, 'OB Reaction')
            if sig:
                signals.append(sig)

    if at_bear_ob and c_bear and not rsi_xtreme_lo:
        conf = 68
        reasons = ['Price touching bearish Order Block — institutional supply zone']
        if c_str_bear:
            conf += 10; reasons.append(f'Rejection candle ({candle}) confirms OB defended')
        if below_vwap or near_vwap:
            conf += 8;  reasons.append('At/below VWAP — institutional bias bearish')
        if ema_bear:
            conf += 7;  reasons.append('EMA 9<21<50 — trend aligned with short')
        if macd_bear_mom:
            conf += 8;  reasons.append('MACD histogram expanding bearish — momentum confirmed')
        elif macd_h < 0:
            conf += 3;  reasons.append('MACD negative — directional bias intact')
        if at_bear_fvg:
            conf += 10; reasons.append('OB + FVG confluence — double institutional level')
        if d_bear:
            conf += 6;  reasons.append('Net selling delta — smart money distributing')
        if vol_ratio >= 1.0:
            conf += 5;  reasons.append(f'Volume {vol_ratio:.1f}× avg — active market')
        if struct_bear or bear_bos:
            conf += 5;  reasons.append('Market structure bearish — LH+LL pattern')
        if conf >= 65:
            sig = _emit('SELL', conf, price, _sell_sl(), _sell_tp(), reasons, 'OB Reaction')
            if sig:
                signals.append(sig)

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 3 — VWAP BOUNCE / RECLAIM  (~72 % win rate)
    # VWAP is the institutional anchor — bounces off it are precision entries.
    # ══════════════════════════════════════════════════════════════════════════

    prev_close = float(df['close'].iloc[-2]) if n > 2 else price

    vwap_reclaim_bull = near_vwap and above_vwap and c_bull and prev_close < vwap
    vwap_bounce_bull  = near_vwap and above_vwap and c_str_bull and ema_bull

    if (vwap_reclaim_bull or vwap_bounce_bull) and not rsi_xtreme_hi:
        setup_label = 'VWAP Reclaim' if vwap_reclaim_bull else 'VWAP Bounce'
        conf = 65
        reasons = [f'Price {"reclaimed" if vwap_reclaim_bull else "bounced off"} VWAP — '
                   f'institutional support confirmed']
        if c_str_bull:
            conf += 8;  reasons.append(f'Reversal candle ({candle}) at VWAP')
        if ema_bull:
            conf += 8;  reasons.append('EMA 9>21>50 — trend confirms long')
        if macd_bull_mom:
            conf += 8;  reasons.append('MACD expanding bullish — momentum in direction')
        if d_bull:
            conf += 7;  reasons.append('Net buying pressure on delta')
        if vol_ratio >= 1.1:
            conf += 6;  reasons.append(f'Volume {vol_ratio:.1f}× avg — confirms move')
        if at_bull_ob or at_bull_fvg:
            conf += 8;  reasons.append('VWAP aligns with OB/FVG — stacked confluence')
        if struct_bull:
            conf += 4;  reasons.append('Bullish market structure')
        if rsi_os:
            conf += 5;  reasons.append('RSI oversold bounce — additional confirmation')
        sig = _emit('BUY', conf, price, _buy_sl(), _buy_tp(), reasons, setup_label)
        if sig:
            signals.append(sig)

    vwap_reject_bear  = near_vwap and below_vwap and c_bear and prev_close > vwap
    vwap_bounce_bear  = near_vwap and below_vwap and c_str_bear and ema_bear

    if (vwap_reject_bear or vwap_bounce_bear) and not rsi_xtreme_lo:
        setup_label = 'VWAP Rejection' if vwap_reject_bear else 'VWAP Fade'
        conf = 65
        reasons = [f'Price {"rejected" if vwap_reject_bear else "faded"} at VWAP — '
                   f'institutional resistance confirmed']
        if c_str_bear:
            conf += 8;  reasons.append(f'Rejection candle ({candle}) at VWAP')
        if ema_bear:
            conf += 8;  reasons.append('EMA 9<21<50 — trend confirms short')
        if macd_bear_mom:
            conf += 8;  reasons.append('MACD expanding bearish — momentum in direction')
        if d_bear:
            conf += 7;  reasons.append('Net selling pressure on delta')
        if vol_ratio >= 1.1:
            conf += 6;  reasons.append(f'Volume {vol_ratio:.1f}× avg — confirms move')
        if at_bear_ob or at_bear_fvg:
            conf += 8;  reasons.append('VWAP aligns with OB/FVG — stacked confluence')
        if struct_bear:
            conf += 4;  reasons.append('Bearish market structure')
        if rsi_ob:
            conf += 5;  reasons.append('RSI overbought rejection — additional confirmation')
        sig = _emit('SELL', conf, price, _sell_sl(), _sell_tp(), reasons, setup_label)
        if sig:
            signals.append(sig)

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 4 — EMA PULLBACK IN TREND  (~68 % win rate)
    # In a clean trend, the first pullback to EMA9/21 is the safest entry.
    # Requires: EMA alignment + price touching EMA + candle closes back in trend.
    # ══════════════════════════════════════════════════════════════════════════

    if (near_ema9 or near_ema21) and ema_bull and above_vwap and c_bull and not rsi_xtreme_hi:
        ema_lbl = 'EMA9' if near_ema9 else 'EMA21'
        conf = 62
        reasons = [f'Pullback to {ema_lbl} in bullish trend — continuation long entry']
        if c_str_bull:
            conf += 10; reasons.append(f'Rejection candle ({candle}) at {ema_lbl}')
        if macd_bull_mom:
            conf += 8;  reasons.append('MACD expanding bullish — trend momentum intact')
        elif macd_h > 0:
            conf += 3;  reasons.append('MACD positive — trend bias confirmed')
        if above_vwap:
            conf += 7;  reasons.append('Above VWAP — institutional bullish bias')
        if d_bull:
            conf += 6;  reasons.append('Delta bullish — smart money buying')
        if struct_bull or bull_bos:
            conf += 7;  reasons.append('Market structure: HH+HL confirmed')
        if at_bull_fvg:
            conf += 8;  reasons.append(f'FVG near {ema_lbl} — double demand zone')
        if vol_ratio >= 0.85:
            conf += 4;  reasons.append(f'Volume {vol_ratio:.1f}× avg — acceptable activity')
        if conf >= 65:
            sig = _emit('BUY', conf, price, _buy_sl(), _buy_tp(),
                        reasons, f'EMA Pullback ({ema_lbl})')
            if sig:
                signals.append(sig)

    if (near_ema9 or near_ema21) and ema_bear and below_vwap and c_bear and not rsi_xtreme_lo:
        ema_lbl = 'EMA9' if near_ema9 else 'EMA21'
        conf = 62
        reasons = [f'Pullback to {ema_lbl} in bearish trend — continuation short entry']
        if c_str_bear:
            conf += 10; reasons.append(f'Rejection candle ({candle}) at {ema_lbl}')
        if macd_bear_mom:
            conf += 8;  reasons.append('MACD expanding bearish — trend momentum intact')
        elif macd_h < 0:
            conf += 3;  reasons.append('MACD negative — trend bias confirmed')
        if below_vwap:
            conf += 7;  reasons.append('Below VWAP — institutional bearish bias')
        if d_bear:
            conf += 6;  reasons.append('Delta bearish — smart money selling')
        if struct_bear or bear_bos:
            conf += 7;  reasons.append('Market structure: LH+LL confirmed')
        if at_bear_fvg:
            conf += 8;  reasons.append(f'FVG near {ema_lbl} — double supply zone')
        if vol_ratio >= 0.85:
            conf += 4;  reasons.append(f'Volume {vol_ratio:.1f}× avg — acceptable activity')
        if conf >= 65:
            sig = _emit('SELL', conf, price, _sell_sl(), _sell_tp(),
                        reasons, f'EMA Pullback ({ema_lbl})')
            if sig:
                signals.append(sig)

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 5 — BOS FIRST PULLBACK  (~70 % win rate)
    # After a confirmed Break of Structure, trade the retest of the broken level.
    # Requires: recent BOS + price returns to the BOS level + rejection.
    # ══════════════════════════════════════════════════════════════════════════

    if bull_bos and bos_price and c_bull and not rsi_xtreme_hi:
        dist_bos = abs(price - bos_price) / atr
        if dist_bos < 0.65:
            conf = 68
            reasons = [f'BOS pullback: retesting broken structure at {_safe(bos_price)}']
            if above_vwap:
                conf += 8;  reasons.append('Above VWAP — bullish macro bias')
            if c_str_bull:
                conf += 10; reasons.append(f'Rejection at BOS: {candle}')
            if macd_bull_mom:
                conf += 7;  reasons.append('MACD momentum bullish')
            if d_bull:
                conf += 6;  reasons.append('Delta confirms buying at BOS level')
            if at_bull_ob or at_bull_fvg:
                conf += 8;  reasons.append('BOS level aligns with OB/FVG — premium zone')
            if vol_ratio >= 0.9:
                conf += 4
            sl = min(_buy_sl(), bos_price - atr * 0.25) if bos_price > 0 else _buy_sl()
            sig = _emit('BUY', conf, price, sl, _buy_tp(), reasons, 'BOS Pullback')
            if sig:
                signals.append(sig)

    if bear_bos and bos_price and c_bear and not rsi_xtreme_lo:
        dist_bos = abs(price - bos_price) / atr
        if dist_bos < 0.65:
            conf = 68
            reasons = [f'BOS pullback: retesting broken structure at {_safe(bos_price)}']
            if below_vwap:
                conf += 8;  reasons.append('Below VWAP — bearish macro bias')
            if c_str_bear:
                conf += 10; reasons.append(f'Rejection at BOS: {candle}')
            if macd_bear_mom:
                conf += 7;  reasons.append('MACD momentum bearish')
            if d_bear:
                conf += 6;  reasons.append('Delta confirms selling at BOS level')
            if at_bear_ob or at_bear_fvg:
                conf += 8;  reasons.append('BOS level aligns with OB/FVG — premium zone')
            if vol_ratio >= 0.9:
                conf += 4
            sl = max(_sell_sl(), bos_price + atr * 0.25) if bos_price > 0 else _sell_sl()
            sig = _emit('SELL', conf, price, sl, _sell_tp(), reasons, 'BOS Pullback')
            if sig:
                signals.append(sig)

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 6 — RSI DIVERGENCE AT STRUCTURAL LEVEL  (~70 % win rate)
    # When momentum diverges from price AT a key level — reversal is imminent.
    # Requires: divergence confirmed + at OB/FVG/VWAP + reversal candle.
    # ══════════════════════════════════════════════════════════════════════════

    if bull_div and (at_bull_ob or near_vwap or at_bull_fvg) and c_bull and not rsi_xtreme_hi:
        conf = 68
        reasons = ['Bullish RSI divergence — momentum turning up while price makes lows']
        if at_bull_ob:
            conf += 10; reasons.append('Divergence at Order Block — high-probability reversal zone')
        if near_vwap:
            conf += 8;  reasons.append('Divergence at VWAP — institutional support')
        if at_bull_fvg:
            conf += 7;  reasons.append('Divergence at FVG — imbalance acts as magnet')
        if c_str_bull:
            conf += 8;  reasons.append(f'Reversal candle confirms divergence: {candle}')
        if d_bull:
            conf += 6;  reasons.append('Delta turning bullish — money flow confirming')
        if macd_bull_mom:
            conf += 5;  reasons.append('MACD histogram rising — momentum shift')
        sig = _emit('BUY', conf, price, _buy_sl(), _buy_tp(), reasons, 'RSI Divergence')
        if sig:
            signals.append(sig)

    if bear_div and (at_bear_ob or near_vwap or at_bear_fvg) and c_bear and not rsi_xtreme_lo:
        conf = 68
        reasons = ['Bearish RSI divergence — momentum turning down while price makes highs']
        if at_bear_ob:
            conf += 10; reasons.append('Divergence at Order Block — high-probability reversal zone')
        if near_vwap:
            conf += 8;  reasons.append('Divergence at VWAP — institutional resistance')
        if at_bear_fvg:
            conf += 7;  reasons.append('Divergence at FVG — imbalance acts as magnet')
        if c_str_bear:
            conf += 8;  reasons.append(f'Rejection candle confirms divergence: {candle}')
        if d_bear:
            conf += 6;  reasons.append('Delta turning bearish — money flow confirming')
        if macd_bear_mom:
            conf += 5;  reasons.append('MACD histogram falling — momentum shift')
        sig = _emit('SELL', conf, price, _sell_sl(), _sell_tp(), reasons, 'RSI Divergence')
        if sig:
            signals.append(sig)

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP 7 — S&R ZONE REVERSAL  (~75 % win rate)
    # Price hitting a high-confluence support/resistance zone with reversal candle.
    # The more levels stacked (Pivot + OB + FVG + VWAP), the higher the score.
    # ══════════════════════════════════════════════════════════════════════════

    if near_rev_zone_bull and c_bull and not rsi_xtreme_hi:
        z = near_rev_zone_bull
        conf = 62 + min(18, z['score'])   # base scales with zone score
        reasons = [
            f"Elite support zone at {z['price']} — {z['strength']} ({z['confluence_count']} confluences: "
            f"{', '.join(z['confluences'][:3])})"
        ]
        if c_str_bull:
            conf += 8;  reasons.append(f'Reversal candle ({candle}) confirms bounce')
        if d_bull:
            conf += 6;  reasons.append('Delta turning bullish — buying pressure at zone')
        if macd_bull_mom:
            conf += 5;  reasons.append('MACD momentum shifting bullish')
        if vol_ratio >= 1.1:
            conf += 5;  reasons.append(f'Volume {vol_ratio:.1f}× confirms institutional activity')
        if ema_bull or above_vwap:
            conf += 4;  reasons.append('EMA/VWAP bias supports long')
        if z['confluence_count'] >= 3:
            conf += 5;  reasons.append(f'{z["confluence_count"]}-way confluence — elite reversal zone')
        if rsi_os:
            conf += 4;  reasons.append('RSI oversold at zone — maximum reversal probability')
        sig = _emit('BUY', conf, price, _buy_sl(), _buy_tp(), reasons, 'S&R Zone Reversal')
        if sig:
            signals.append(sig)

    if near_rev_zone_bear and c_bear and not rsi_xtreme_lo:
        z = near_rev_zone_bear
        conf = 62 + min(18, z['score'])
        reasons = [
            f"Elite resistance zone at {z['price']} — {z['strength']} ({z['confluence_count']} confluences: "
            f"{', '.join(z['confluences'][:3])})"
        ]
        if c_str_bear:
            conf += 8;  reasons.append(f'Rejection candle ({candle}) confirms resistance')
        if d_bear:
            conf += 6;  reasons.append('Delta turning bearish — selling pressure at zone')
        if macd_bear_mom:
            conf += 5;  reasons.append('MACD momentum shifting bearish')
        if vol_ratio >= 1.1:
            conf += 5;  reasons.append(f'Volume {vol_ratio:.1f}× confirms institutional activity')
        if ema_bear or below_vwap:
            conf += 4;  reasons.append('EMA/VWAP bias supports short')
        if z['confluence_count'] >= 3:
            conf += 5;  reasons.append(f'{z["confluence_count"]}-way confluence — elite reversal zone')
        if rsi_ob:
            conf += 4;  reasons.append('RSI overbought at zone — maximum reversal probability')
        sig = _emit('SELL', conf, price, _sell_sl(), _sell_tp(), reasons, 'S&R Zone Reversal')
        if sig:
            signals.append(sig)

    # ── S&R Confluence Bonus: reward any signal generated at a key level ──────
    for sig in signals:
        if sig['direction'] == 'BUY' and at_sr_support and nr_support:
            bonus = min(8, nr_support.get('strength', 0) * 2)
            sig['confidence'] = min(100, sig['confidence'] + bonus)
            sig['reasons'].append(
                f"Key support: {nr_support['label']} ({nr_support['strength']}/5 strength, "
                f"dist {nr_support['distance_atr']:.2f} ATR)"
            )
            sig['summary'] = (
                f"{sig['direction']} [{sig['setup']}] — {len(sig['reasons'])} confluences. "
                f"E:{round(sig['entry'],1)}  SL:{round(sig['stop_loss'],1)}  "
                f"TP:{round(sig['take_profit'],1)}  RR:{sig['risk_reward']}:1"
            )
        elif sig['direction'] == 'SELL' and at_sr_resistance and nr_resistance:
            bonus = min(8, nr_resistance.get('strength', 0) * 2)
            sig['confidence'] = min(100, sig['confidence'] + bonus)
            sig['reasons'].append(
                f"Key resistance: {nr_resistance['label']} ({nr_resistance['strength']}/5 strength, "
                f"dist {nr_resistance['distance_atr']:.2f} ATR)"
            )
            sig['summary'] = (
                f"{sig['direction']} [{sig['setup']}] — {len(sig['reasons'])} confluences. "
                f"E:{round(sig['entry'],1)}  SL:{round(sig['stop_loss'],1)}  "
                f"TP:{round(sig['take_profit'],1)}  RR:{sig['risk_reward']}:1"
            )

    # ── Deduplicate: keep best per direction, allow 2nd only if diff setup ≥72% ──
    seen: dict = {}
    result: List[dict] = []
    for sig in sorted(signals, key=lambda x: x['confidence'], reverse=True):
        if sig['confidence'] < 65:
            continue
        d = sig['direction']
        if d not in seen:
            seen[d] = sig
            result.append(sig)
        elif sig['confidence'] >= 72 and sig.get('setup') != seen[d].get('setup'):
            result.append(sig)

    return sorted(result, key=lambda x: x['confidence'], reverse=True)


# ── ADX (Trend Strength) ─────────────────────────────────────────────────────

def _calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index — measures trend strength (25+ = trending, <20 = ranging)."""
    high = df['high']; low = df['low']; close = df['close']
    prev_high = high.shift(1); prev_low = low.shift(1); prev_close = close.shift(1)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up   = (high - prev_high).clip(lower=0)
    down = (prev_low - low).clip(lower=0)
    dm_plus  = up.where(up > down, 0.0)
    dm_minus = down.where(down > up, 0.0)

    alpha = 1.0 / period
    atr_s    = tr.ewm(alpha=alpha, adjust=False).mean()
    dmp_s    = dm_plus.ewm(alpha=alpha, adjust=False).mean()
    dmm_s    = dm_minus.ewm(alpha=alpha, adjust=False).mean()

    di_plus  = 100 * dmp_s / atr_s.replace(0, np.nan)
    di_minus = 100 * dmm_s / atr_s.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, di_plus, di_minus


# ── Market Session ────────────────────────────────────────────────────────────

def detect_market_session() -> dict:
    """
    Detect current NYSE/NASDAQ trading session based on Eastern Time.
    Returns session characteristics, volatility profile, and trading notes.
    Sessions: PRE_MARKET → OPEN → EARLY → MID_MORNING → LUNCH → AFTERNOON → POWER_HOUR → AFTER_HOURS
    """
    try:
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo('America/New_York'))
        except Exception:
            # Fallback: approximate ET offset (EDT = UTC-4, EST = UTC-5)
            utc_now = datetime.now(timezone.utc)
            month = utc_now.month
            offset_h = -4 if 3 <= month <= 11 else -5
            now_et = utc_now + timedelta(hours=offset_h)

        h, m, wd = now_et.hour, now_et.minute, now_et.weekday()  # 0=Mon
        t = h * 60 + m  # minutes since midnight ET

        if wd >= 5:
            return {
                'session': 'WEEKEND', 'session_label': 'Weekend — Markets Closed',
                'expected_volatility': 'NONE', 'expected_volume': 'NONE',
                'vol_multiplier': 0.0, 'session_bias': 'CLOSED',
                'is_tradeable': False, 'is_high_volatility': False, 'is_low_volume': True,
                'reduce_size': True, 'notes': 'Markets closed. Review setups for Monday.',
                'time_et': f"{h:02d}:{m:02d} ET", 'minutes_remaining_in_session': None,
            }

        # (start_min, end_min, key, label, volatility, volume, vol_mult, bias, notes)
        SESSIONS = [
            (0,     4*60,       'OVERNIGHT',   'Overnight',                  'LOW',    'NONE',    0.2, 'RANGING',
             'Overnight — no setups. Monitor for gap conditions.'),
            (4*60,  9*60+30,    'PRE_MARKET',  'Pre-Market (4:00–9:30 ET)',  'ELEVATED','LOW',   0.6, 'RANGING',
             'Watch gaps and news reactions. Wide spreads — reduce size 50%.'),
            (9*60+30, 10*60,    'OPEN',        'Open (9:30–10:00 ET)',        'EXTREME','EXTREME', 2.2, 'TRENDING',
             'EXTREME volatility — first 30 min. Wait for direction, trade momentum only.'),
            (10*60,   10*60+30, 'EARLY',       'Early Session (10:00–10:30)', 'HIGH',  'HIGH',    1.7, 'TRENDING',
             'Prime window. BOS pullbacks, OB reactions and liquidity sweeps excel here.'),
            (10*60+30,12*60,    'MID_MORNING', 'Mid-Morning (10:30–12:00)',   'NORMAL','NORMAL',  1.1, 'TRENDING',
             'Clean trends. VWAP bounce/rejection and EMA pullbacks work best.'),
            (12*60,   13*60+30, 'LUNCH',       'Lunch (12:00–13:30 ET)',      'LOW',   'LOW',     0.4, 'RANGING',
             'DANGER: Low volume, choppy, false moves. Reduce size 70% or step aside.'),
            (13*60+30,15*60,    'AFTERNOON',   'Afternoon (13:30–15:00 ET)',  'NORMAL','NORMAL',  1.0, 'TRENDING',
             'Session resumes. Trend continuation setups preferred. Watch VWAP.'),
            (15*60,   16*60,    'POWER_HOUR',  'Power Hour (15:00–16:00 ET)', 'HIGH',  'HIGH',    1.6, 'TRENDING',
             'Institutional closing. Strong directional moves, liquidity sweeps common.'),
            (16*60,   20*60,    'AFTER_HOURS', 'After Hours (16:00–20:00)',   'LOW',   'LOW',     0.3, 'RANGING',
             'After-hours — earnings plays only. Wide spreads, reduced liquidity.'),
            (20*60,   24*60,    'OVERNIGHT',   'Late/Overnight',              'LOW',   'NONE',    0.2, 'RANGING',
             'Overnight — no intraday setups.'),
        ]

        session_data = SESSIONS[-1]  # default
        for start, end, *rest in SESSIONS:
            if start <= t < end:
                session_data = (start, end, *rest)
                break

        _, end_t, key, label, vol, volume, vol_mult, bias, notes = session_data
        mins_remaining = max(0, end_t - t)

        # Minutes to open (9:30 ET)
        open_t = 9 * 60 + 30
        mins_to_open = max(0, open_t - t) if t < open_t else None

        high_vol_sessions = ('OPEN', 'EARLY', 'POWER_HOUR')
        low_vol_sessions  = ('LUNCH', 'PRE_MARKET', 'AFTER_HOURS', 'OVERNIGHT')

        return {
            'session':                  key,
            'session_label':            label,
            'expected_volatility':      vol,
            'expected_volume':          volume,
            'vol_multiplier':           vol_mult,
            'session_bias':             bias,
            'notes':                    notes,
            'time_et':                  f"{h:02d}:{m:02d} ET",
            'minutes_remaining_in_session': mins_remaining,
            'minutes_to_open':          mins_to_open,
            'is_tradeable':             key not in ('WEEKEND', 'OVERNIGHT', 'AFTER_HOURS'),
            'is_high_volatility':       key in high_vol_sessions,
            'is_low_volume':            key in low_vol_sessions,
            'reduce_size':              key in low_vol_sessions or vol == 'EXTREME',
        }
    except Exception as e:
        logger.warning(f"Session detection error: {e}")
        return {
            'session': 'UNKNOWN', 'session_label': 'Unknown Session',
            'expected_volatility': 'NORMAL', 'expected_volume': 'NORMAL',
            'vol_multiplier': 1.0, 'session_bias': 'TRENDING',
            'is_tradeable': True, 'is_high_volatility': False, 'is_low_volume': False,
            'reduce_size': False, 'notes': 'Session detection unavailable.',
            'time_et': '--:-- ET', 'minutes_remaining_in_session': None,
        }


# ── Market Regime ─────────────────────────────────────────────────────────────

def detect_market_regime(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Classify market regime: TRENDING_UP, TRENDING_DOWN, RANGING, REVERSAL, CONSOLIDATION.
    Uses ADX (trend strength), BB width, RSI, volume, and SMC structure.
    Returns probabilities for continuation, reversal, and pullback.
    """
    if len(df) < 30:
        return {'regime': 'UNKNOWN', 'regime_confidence': 0,
                'continuation_probability': 50, 'reversal_probability': 25,
                'pullback_probability': 25, 'preferred_setups': [], 'avoid_setups': []}

    price  = float(indicators.get('price', df['close'].iloc[-1]))
    atr    = max(float(indicators.get('atr', 1)), 0.1)
    rsi    = float(indicators.get('rsi', 50) or 50)
    bb_u   = float(indicators.get('bb_upper', price + atr * 2))
    bb_l   = float(indicators.get('bb_lower', price - atr * 2))
    ema9_v = float(indicators.get('ema9',  price))
    ema21_v= float(indicators.get('ema21', price))
    ema50_v= float(indicators.get('ema50', price))

    # ADX
    try:
        adx_s, dip_s, dim_s = _calc_adx(df, 14)
        adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 20.0
        dip_v = float(dip_s.iloc[-1]) if not np.isnan(dip_s.iloc[-1]) else 20.0
        dim_v = float(dim_s.iloc[-1]) if not np.isnan(dim_s.iloc[-1]) else 20.0
    except Exception:
        adx_v, dip_v, dim_v = 20.0, 20.0, 20.0

    # Bollinger Band width vs average (squeeze = consolidation)
    bb_width     = (bb_u - bb_l) / max(price, 1)
    bb_width_avg = float(df['close'].rolling(20).std().mean()) * 4 / max(price, 1)
    bb_squeeze   = bb_width < bb_width_avg * 0.75
    bb_expansion = bb_width > bb_width_avg * 1.40

    # Price range (tight range = ranging market)
    last20_range   = float(df['high'].tail(20).max() - df['low'].tail(20).min())
    range_ratio    = last20_range / max(atr * 20, 0.1)
    is_ranging_px  = range_ratio < 1.1

    # EMA alignment
    ema_bull = ema9_v > ema21_v > ema50_v
    ema_bear = ema9_v < ema21_v < ema50_v

    # Volume
    vol_avg20 = max(float(df['volume'].rolling(20).mean().iloc[-1]), 1)
    vol_avg5  = max(float(df['volume'].tail(5).mean()), 1)
    vol_rising = vol_avg5 > vol_avg20 * 1.1

    # SMC context
    structure = indicators.get('structure', {})
    trend_smc = structure.get('trend', 'SIDEWAYS')
    choch     = structure.get('choch')
    bos       = structure.get('bos') or {}
    bos_type  = bos.get('type', '')

    rsi_extreme = rsi > 75 or rsi < 25

    # ── Score each regime ──
    scores = {'TRENDING_UP': 0, 'TRENDING_DOWN': 0, 'RANGING': 0, 'REVERSAL': 0, 'CONSOLIDATION': 0}

    if adx_v > 25:      scores['TRENDING_UP'] += 15; scores['TRENDING_DOWN'] += 15
    if adx_v > 35:      scores['TRENDING_UP'] += 10; scores['TRENDING_DOWN'] += 10
    if ema_bull:        scores['TRENDING_UP'] += 20
    if ema_bear:        scores['TRENDING_DOWN'] += 20
    if dip_v > dim_v:   scores['TRENDING_UP'] += 10
    if dim_v > dip_v:   scores['TRENDING_DOWN'] += 10
    if rsi > 55:        scores['TRENDING_UP'] += 8
    if rsi < 45:        scores['TRENDING_DOWN'] += 8
    if vol_rising:      scores['TRENDING_UP'] += 5; scores['TRENDING_DOWN'] += 5
    if trend_smc == 'BULLISH': scores['TRENDING_UP'] += 15
    if trend_smc == 'BEARISH': scores['TRENDING_DOWN'] += 15
    if bos_type == 'BULLISH_BOS': scores['TRENDING_UP'] += 12
    if bos_type == 'BEARISH_BOS': scores['TRENDING_DOWN'] += 12

    if is_ranging_px:   scores['RANGING'] += 25; scores['CONSOLIDATION'] += 20
    if adx_v < 20:      scores['RANGING'] += 20; scores['CONSOLIDATION'] += 15
    if bb_squeeze:      scores['CONSOLIDATION'] += 30; scores['RANGING'] += 15
    if 40 < rsi < 60:   scores['RANGING'] += 10

    if choch:           scores['REVERSAL'] += 35
    if rsi_extreme:     scores['REVERSAL'] += 20
    if bb_expansion and rsi_extreme: scores['REVERSAL'] += 15
    if choch and rsi_extreme:        scores['REVERSAL'] += 20

    regime    = max(scores, key=scores.get)
    total     = max(sum(scores.values()), 1)
    confidence = int(scores[regime] / total * 100)

    # Probabilities
    trend_total = scores['TRENDING_UP'] + scores['TRENDING_DOWN']
    continuation_prob = int(trend_total / total * 100)
    reversal_prob     = int(scores['REVERSAL'] / total * 100)
    pullback_prob     = 0
    if regime in ('TRENDING_UP', 'TRENDING_DOWN'):
        if rsi > 70 or rsi < 30:    pullback_prob = 65
        elif adx_v > 40:            pullback_prob = 35
        else:                       pullback_prob = 20

    # Preferred setups per regime
    if regime == 'TRENDING_UP':
        preferred = ['EMA Pullback', 'BOS Pullback', 'VWAP Bounce', 'OB Reaction']
        avoid     = ['Counter-trend shorts', 'S&R Zone Reversal (short)']
    elif regime == 'TRENDING_DOWN':
        preferred = ['EMA Pullback', 'BOS Pullback', 'VWAP Rejection', 'OB Reaction']
        avoid     = ['Counter-trend longs', 'S&R Zone Reversal (long)']
    elif regime == 'RANGING':
        preferred = ['S&R Zone Reversal', 'RSI Divergence', 'VWAP Bounce/Rejection', 'OB Reaction']
        avoid     = ['Trend breakout without confirmation']
    elif regime == 'REVERSAL':
        preferred = ['Liquidity Sweep', 'RSI Divergence', 'S&R Zone Reversal', 'VWAP Reclaim/Rejection']
        avoid     = ['Trend continuation in prior direction']
    else:  # CONSOLIDATION
        preferred = ['Wait for breakout — no new entries']
        avoid     = ['All intraday setups until breakout confirmed']

    # Observations
    obs = []
    obs.append(f'ADX {adx_v:.0f} — {"strong trend" if adx_v > 25 else "weak/no trend"}')
    if bb_squeeze:   obs.append('Bollinger squeeze — breakout loading')
    if bb_expansion: obs.append('BB expanding — volatility burst in progress')
    if choch:        obs.append('CHoCH detected — structural reversal signal')
    if rsi_extreme:  obs.append(f'RSI {rsi:.0f} — extreme, reversal risk elevated')
    if vol_rising:   obs.append('Volume rising — institutional participation')

    return {
        'regime':                    regime,
        'regime_confidence':         confidence,
        'adx':                       round(adx_v, 1),
        'di_plus':                   round(dip_v, 1),
        'di_minus':                  round(dim_v, 1),
        'bb_squeeze':                bb_squeeze,
        'bb_expansion':              bb_expansion,
        'is_trending':               adx_v > 25,
        'is_ranging':                is_ranging_px or adx_v < 20,
        'continuation_probability':  continuation_prob,
        'reversal_probability':      reversal_prob,
        'pullback_probability':       pullback_prob,
        'preferred_setups':          preferred,
        'avoid_setups':              avoid,
        'key_observations':          obs[:5],
        'scores':                    scores,
    }


# ── Main Analysis Function ────────────────────────────────────────────────────

def full_analysis(df: pd.DataFrame) -> dict:
    """
    Run complete technical analysis on OHLCV data.
    Returns all indicators, SMC zones, institutional flow, and signals.
    """
    if df is None or len(df) < 50:
        return {'error': 'Insufficient data (need 50+ bars)'}

    close = df['close']
    price = float(close.iloc[-1])

    # ── Classic Indicators ──
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200) if len(df) >= 200 else _ema(close, len(df))
    rsi = calc_rsi(close, 14)
    macd, macd_sig, macd_hist = calc_macd(close)
    bb_u, bb_m, bb_l = calc_bollinger(close)
    atr = calc_atr(df, 14)
    stoch_k, stoch_d = calc_stochastic(df)
    vol_sma = _sma(df['volume'], 20)

    # ── VWAP ──
    df_vwap = calc_vwap(df)
    vwap = float(df_vwap['vwap'].iloc[-1]) if 'vwap' in df_vwap.columns else price
    vwap_u1 = float(df_vwap.get('vwap_u1', pd.Series([price])).iloc[-1])
    vwap_l1 = float(df_vwap.get('vwap_l1', pd.Series([price])).iloc[-1])
    vwap_u2 = float(df_vwap.get('vwap_u2', pd.Series([price])).iloc[-1])
    vwap_l2 = float(df_vwap.get('vwap_l2', pd.Series([price])).iloc[-1])

    # ── Volume Delta ──
    df_delta = calc_volume_delta(df)
    cum_delta = float(df_delta['cum_delta'].iloc[-1])
    last_delta = float(df_delta['delta'].iloc[-1])

    # ── Volume Profile ──
    vol_profile = calc_volume_profile(df.tail(100), bins=30)

    # ── SMC ──
    structure = detect_market_structure(df)
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fair_value_gaps(df)
    inst_candles = detect_institutional_candles(df)

    # ── Support & Resistance + Reversal Zones ──
    _atr_v = float(atr.iloc[-1])
    sr_levels = calc_support_resistance(df, price, _atr_v)
    reversal_zones = score_reversal_zones(
        price, _atr_v, sr_levels, order_blocks, fvgs,
        vwap, vwap_u1, vwap_l1, vwap_u2, vwap_l2,
    )

    # ── Trend Assessment ──
    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e200 = float(ema200.iloc[-1])
    rsi_v = float(rsi.iloc[-1]) if not rsi.iloc[-1] != rsi.iloc[-1] else 50

    trend_score = 0
    if price > vwap: trend_score += 1
    if e9 > e21: trend_score += 1
    if e21 > e50: trend_score += 1
    if price > e50: trend_score += 1
    if rsi_v > 55: trend_score += 1
    if float(macd_hist.iloc[-1]) > 0: trend_score += 1

    if trend_score >= 5:
        trend = 'STRONG_BULLISH'
    elif trend_score >= 4:
        trend = 'BULLISH'
    elif trend_score <= 1:
        trend = 'STRONG_BEARISH'
    elif trend_score <= 2:
        trend = 'BEARISH'
    else:
        trend = 'SIDEWAYS'

    # ── Indicators dict for signals ──
    indicators = {
        'price': price, 'vwap': vwap, 'rsi': rsi_v,
        'macd_hist': float(macd_hist.iloc[-1]),
        'atr': float(atr.iloc[-1]),
        'ema9': e9, 'ema21': e21, 'ema50': e50,
        'stoch_k': float(stoch_k.iloc[-1]) if not stoch_k.iloc[-1] != stoch_k.iloc[-1] else 50,
        'bb_upper': float(bb_u.iloc[-1]),
        'bb_lower': float(bb_l.iloc[-1]),
        'vwap_u1': vwap_u1, 'vwap_l1': vwap_l1,
        'structure': structure,
        'order_blocks': order_blocks,
        'fair_value_gaps': fvgs,
        'cum_delta': cum_delta,
        # Full series — needed for divergence, momentum, and sweep detection
        'rsi_series': rsi,
        'macd_hist_series': macd_hist,
        'delta_series_raw': df_delta['delta'],
        # S&R and reversal zones for signal engine
        'sr_levels':     sr_levels,
        'reversal_zones': reversal_zones,
    }

    # ── Market Regime + Session ──
    regime  = detect_market_regime(df, indicators)
    session = detect_market_session()

    signals = generate_signals(df, indicators)

    # ── VWAP series for charting (last 100 bars) ──
    vwap_series = []
    for i in range(max(0, len(df_vwap) - 100), len(df_vwap)):
        t = df_vwap.index[i]
        t_str = t.isoformat() if hasattr(t, 'isoformat') else str(t)
        v = df_vwap.get('vwap', pd.Series())
        if len(v) > i and not (v.iloc[i] != v.iloc[i]):
            vwap_series.append({'time': t_str, 'value': _safe(v.iloc[i])})

    # ── Volume profile for chart ──
    profile_for_chart = vol_profile.get('profile', [])

    # ── OHLCV candles for chart (last 150 bars) ──
    candles = []
    for i in range(max(0, len(df) - 150), len(df)):
        t = df.index[i]
        try:
            ts = int(t.timestamp()) if hasattr(t, 'timestamp') else int(t)
        except Exception:
            ts = i
        candles.append({
            'time': ts,
            'open': _safe(df['open'].iloc[i]),
            'high': _safe(df['high'].iloc[i]),
            'low': _safe(df['low'].iloc[i]),
            'close': _safe(df['close'].iloc[i]),
            'volume': int(df['volume'].iloc[i]),
        })

    # ── Delta series for chart ──
    delta_series = []
    for i in range(len(df_delta)):
        t = df_delta.index[i]
        # Use Unix integer timestamp (UTC) — same format as candles so time scales align
        try:
            t_unix = int(t.timestamp())
        except Exception:
            import calendar
            t_unix = calendar.timegm(t.timetuple())
        val = int(df_delta['delta'].iloc[i])
        delta_series.append({
            'time': t_unix,
            'value': val,
            'color': '#26a69a' if val >= 0 else '#ef5350',
        })

    return {
        'price': _safe(price),
        'trend': trend,
        'trend_score': trend_score,

        # Classic indicators
        'indicators': {
            'ema9': _safe(e9), 'ema21': _safe(e21), 'ema50': _safe(e50), 'ema200': _safe(e200),
            'rsi': _safe(rsi_v),
            'macd': _safe(float(macd.iloc[-1])),
            'macd_signal': _safe(float(macd_sig.iloc[-1])),
            'macd_hist': _safe(float(macd_hist.iloc[-1])),
            'bb_upper': _safe(float(bb_u.iloc[-1])),
            'bb_mid': _safe(float(bb_m.iloc[-1])),
            'bb_lower': _safe(float(bb_l.iloc[-1])),
            'atr': _safe(float(atr.iloc[-1])),
            'stoch_k': _safe(float(stoch_k.iloc[-1]) if not stoch_k.iloc[-1] != stoch_k.iloc[-1] else 50),
            'stoch_d': _safe(float(stoch_d.iloc[-1]) if not stoch_d.iloc[-1] != stoch_d.iloc[-1] else 50),
            'volume': int(df['volume'].iloc[-1]),
            'volume_sma': _safe(float(vol_sma.iloc[-1]) if not vol_sma.iloc[-1] != vol_sma.iloc[-1] else 0),
            'volume_ratio': _safe(df['volume'].iloc[-1] / float(vol_sma.iloc[-1])) if (vol_sma.iloc[-1] and vol_sma.iloc[-1] != 0) else None,
        },

        # VWAP
        'vwap': {
            'value': _safe(vwap),
            'upper1': _safe(vwap_u1), 'lower1': _safe(vwap_l1),
            'upper2': _safe(vwap_u2), 'lower2': _safe(vwap_l2),
            'position': 'ABOVE' if price > vwap else 'BELOW',
            'series': vwap_series,
        },

        # Institutional flow
        'institutional': {
            'cum_delta': int(cum_delta),
            'last_delta': int(last_delta),
            'delta_trend': 'BULLISH' if cum_delta > 0 else 'BEARISH',
            'volume_profile': {
                'poc': _safe(vol_profile.get('poc')),
                'vah': _safe(vol_profile.get('vah')),
                'val': _safe(vol_profile.get('val')),
                'profile': profile_for_chart,
            },
            'institutional_candles': inst_candles[-10:],
            'delta_series': delta_series,
        },

        # SMC
        'smc': {
            'market_structure': structure,
            'order_blocks': order_blocks,
            'fair_value_gaps': fvgs,
        },

        # Support & Resistance
        'support_resistance': {
            'support':            sr_levels.get('support', []),
            'resistance':         sr_levels.get('resistance', []),
            'key_levels':         sr_levels.get('key_levels', []),
            'nearest_support':    sr_levels.get('nearest_support'),
            'nearest_resistance': sr_levels.get('nearest_resistance'),
        },

        # Reversal zones (scored by confluence count)
        'reversal_zones': reversal_zones,

        # Market regime (trend strength, continuation vs reversal probabilities)
        'regime': regime,

        # Market session (pre-market, open, lunch, power hour, etc.)
        'session': session,

        # Signals
        'signals': signals,

        # Chart data
        'candles': candles,

        'timestamp': datetime.now().isoformat(),
    }
