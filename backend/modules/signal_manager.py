"""
Signal Window Manager
─────────────────────
• 30-minute rolling window — max 3 active signals at a time
• Real-time confidence updates every ~10 s based on:
    - Price movement relative to entry (in ATR units)
    - Market regime alignment
    - Market session (avoid LUNCH / low-volume sessions)
    - Time remaining in the 30-min window (gentle decay in last 20 %)
• Signal lifecycle:  ACTIVE → TP_HIT | SL_HIT | EXPIRED | INVALIDATED
• Smart deduplication: same-direction signal replaces existing only if
  confidence gain ≥ 12 pts AND different setup.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 1800          # 30-minute window
MAX_SIGNALS    = 3             # maximum concurrent active signals


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ActiveSignal:
    id:                  str
    direction:           str        # BUY | SELL
    setup:               str
    entry:               float
    stop_loss:           float
    take_profit:         float
    atr:                 float
    risk_reward:         float
    initial_confidence:  int
    current_confidence:  int
    status:              str        # ACTIVE | TP_HIT | SL_HIT | EXPIRED | INVALIDATED
    created_at:          datetime
    window_start:        datetime
    last_updated:        datetime
    last_price:          float
    initial_price:       float
    pnl_pts:             float = 0.0
    max_favorable_pts:   float = 0.0
    max_adverse_pts:     float = 0.0
    invalidation_reason: str = ""
    reasons:             list = field(default_factory=list)
    confidence_history:  list = field(default_factory=list)
    # Context at signal creation
    regime_at_creation:  str = ""
    session_at_creation: str = ""


# ── Manager ───────────────────────────────────────────────────────────────────

class SignalWindowManager:
    """30-minute rolling signal window with live probability tracking."""

    def __init__(self):
        self._window_start: datetime = datetime.now()
        self._active:  List[ActiveSignal] = []
        self._history: List[ActiveSignal] = []
        self._last_regime:  str = ""
        self._last_session: str = ""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _roll_window_if_needed(self):
        """Move to a new 30-min window when the current one expires."""
        now = datetime.now()
        if (now - self._window_start).total_seconds() >= WINDOW_SECONDS:
            for s in self._active:
                if s.status == 'ACTIVE':
                    s.status = 'EXPIRED'
                    s.last_updated = now
            self._history.extend(self._active)
            self._active = []
            self._window_start = now
            logger.info("New 30-min signal window started")

    def _window_elapsed_s(self) -> float:
        return (datetime.now() - self._window_start).total_seconds()

    def _window_remaining_s(self) -> int:
        return max(0, int(WINDOW_SECONDS - self._window_elapsed_s()))

    def _window_elapsed_pct(self) -> float:
        return min(1.0, self._window_elapsed_s() / WINDOW_SECONDS)

    def _active_count(self) -> int:
        return sum(1 for s in self._active if s.status == 'ACTIVE')

    # ── Confidence update formula ─────────────────────────────────────────────

    def _recalc_confidence(self, sig: ActiveSignal, price: float,
                           regime: str, session: str) -> int:
        """
        Update signal confidence based on:
         1. Price movement (±ATR scaled, cap ±20 pts)
         2. Regime alignment with signal direction (+5 / −10)
         3. Regime reversal penalty (−15 if regime is REVERSAL against signal)
         4. Session quality: LUNCH / low-vol → −8; high-vol → +3
         5. Time decay in last 30 % of window (−0 to −12)
        """
        if sig.direction == 'BUY':
            move = price - sig.entry
        else:
            move = sig.entry - price

        price_score = move / max(sig.atr, 0.1)
        price_bonus = int(min(20, max(-20, price_score * 10)))

        regime_bonus = 0
        if regime:
            if sig.direction == 'BUY'  and 'UP'   in regime: regime_bonus =  5
            if sig.direction == 'SELL' and 'DOWN'  in regime: regime_bonus =  5
            if 'REVERSAL' in regime:
                # Reversal regime favours counter-trend setups
                if sig.setup in ('Liquidity Sweep', 'RSI Divergence', 'S&R Zone Reversal'):
                    regime_bonus += 5
                else:
                    regime_bonus -= 10
            if regime == 'CONSOLIDATION':
                regime_bonus -= 8          # low edge in consolidation

        session_bonus = 0
        if session in ('LUNCH',):          session_bonus = -8
        if session in ('OPEN', 'EARLY', 'POWER_HOUR'): session_bonus = 3

        elapsed_pct = self._window_elapsed_pct()
        # Decay only in last 30 % of window
        time_decay  = int(max(0, (elapsed_pct - 0.70) / 0.30) ** 2 * 12)

        new_conf = sig.initial_confidence + price_bonus + regime_bonus + session_bonus - time_decay
        return max(10, min(100, new_conf))

    # ── Public API ────────────────────────────────────────────────────────────

    def update_price(self, price: float, regime: str = "", session: str = "") -> List[dict]:
        """
        Called every ~10 s with current market price.
        Checks SL/TP hits, updates confidence, returns current window signals.
        """
        self._roll_window_if_needed()
        self._last_regime  = regime  or self._last_regime
        self._last_session = session or self._last_session
        now = datetime.now()

        for sig in self._active:
            if sig.status != 'ACTIVE':
                continue

            # ── SL / TP check ──────────────────────────────────────────────
            if sig.direction == 'BUY':
                hit_sl = price <= sig.stop_loss
                hit_tp = price >= sig.take_profit
                move   = price - sig.entry
            else:
                hit_sl = price >= sig.stop_loss
                hit_tp = price <= sig.take_profit
                move   = sig.entry - price

            if hit_sl:
                sig.status = 'SL_HIT'
                sig.current_confidence = 0
                logger.info(f"Signal {sig.id} SL_HIT at {price:.2f}")
            elif hit_tp:
                sig.status = 'TP_HIT'
                sig.current_confidence = 100
                logger.info(f"Signal {sig.id} TP_HIT at {price:.2f}")

            # ── PnL tracking ───────────────────────────────────────────────
            sig.pnl_pts          = round(move, 2)
            sig.max_favorable_pts = round(max(sig.max_favorable_pts, move), 2)
            sig.max_adverse_pts   = round(min(sig.max_adverse_pts, move), 2)

            # ── Confidence update if still active ──────────────────────────
            if sig.status == 'ACTIVE':
                new_c = self._recalc_confidence(sig, price, regime, session)
                sig.current_confidence = new_c
                sig.confidence_history.append({
                    'ts': now.isoformat()[:19], 'conf': new_c, 'price': round(price, 2)
                })

            sig.last_price   = price
            sig.last_updated = now

        return self._window_snapshot()

    def add_signals(self, new_signals: List[dict],
                    regime: str = "", session: str = "") -> List[dict]:
        """
        Ingest signals from a fresh analysis cycle.
        Enforces MAX_SIGNALS per window and smart deduplication.
        Returns current window snapshot.
        """
        self._roll_window_if_needed()
        self._last_regime  = regime  or self._last_regime
        self._last_session = session or self._last_session

        # Sort by confidence descending so best signals enter first
        for raw in sorted(new_signals, key=lambda x: x.get('confidence', 0), reverse=True):
            if self._active_count() >= MAX_SIGNALS:
                break

            direction = raw.get('direction', '')
            setup     = raw.get('setup', '')
            conf      = int(raw.get('confidence', 50))

            # Check existing same-direction signal
            existing = next(
                (s for s in self._active if s.status == 'ACTIVE' and s.direction == direction), None
            )
            if existing:
                gain        = conf - existing.current_confidence
                diff_setup  = setup != existing.setup
                # Replace only if clearly better (12+ pts higher AND different setup)
                if gain >= 12 and diff_setup:
                    existing.status             = 'INVALIDATED'
                    existing.invalidation_reason = (
                        f'Superseded by {setup} (conf {conf} vs {existing.current_confidence})'
                    )
                    logger.info(f"Signal {existing.id} invalidated — replaced by {setup}")
                else:
                    continue   # keep existing, skip this duplicate

            sig = ActiveSignal(
                id                 = f"sig_{uuid.uuid4().hex[:8]}",
                direction          = direction,
                setup              = setup,
                entry              = float(raw.get('entry', 0)),
                stop_loss          = float(raw.get('stop_loss', 0)),
                take_profit        = float(raw.get('take_profit', 0)),
                atr                = float(raw.get('atr', 1)),
                risk_reward        = float(raw.get('risk_reward', 1.5)),
                initial_confidence = conf,
                current_confidence = conf,
                status             = 'ACTIVE',
                created_at         = datetime.now(),
                window_start       = self._window_start,
                last_updated       = datetime.now(),
                last_price         = float(raw.get('entry', 0)),
                initial_price      = float(raw.get('entry', 0)),
                reasons            = list(raw.get('reasons', [])),
                regime_at_creation = regime,
                session_at_creation= session,
            )
            self._active.append(sig)
            logger.info(
                f"Signal added [{sig.id}]: {direction} [{setup}] conf={conf} "
                f"E={sig.entry} SL={sig.stop_loss} TP={sig.take_profit}"
            )

        return self._window_snapshot()

    def get_active(self) -> List[dict]:
        """Return full current-window snapshot (active + completed)."""
        self._roll_window_if_needed()
        return self._window_snapshot()

    # ── Serialization ─────────────────────────────────────────────────────────

    def _sig_to_dict(self, sig: ActiveSignal) -> dict:
        rem   = self._window_remaining_s()
        elp   = self._window_elapsed_pct()
        pnl_r = round(sig.pnl_pts / max(sig.atr, 0.1), 2)   # PnL in ATR units

        # Dynamic recommendation based on current signal health
        if sig.status != 'ACTIVE':
            recommendation = sig.status
        elif sig.current_confidence >= 80:
            recommendation = 'STRONG — hold position'
        elif sig.current_confidence >= 65:
            recommendation = 'VALID — maintain'
        elif sig.current_confidence >= 50:
            recommendation = 'WEAKENING — tighten SL or reduce size'
        else:
            recommendation = 'FADING — consider exit'

        return {
            'id':                   sig.id,
            'direction':            sig.direction,
            'setup':                sig.setup,
            'entry':                round(sig.entry, 2),
            'stop_loss':            round(sig.stop_loss, 2),
            'take_profit':          round(sig.take_profit, 2),
            'risk_reward':          round(sig.risk_reward, 2),
            'atr':                  round(sig.atr, 2),
            'initial_confidence':   sig.initial_confidence,
            'current_confidence':   sig.current_confidence,
            'confidence_change':    sig.current_confidence - sig.initial_confidence,
            'confidence_trend':     ('UP' if sig.current_confidence > sig.initial_confidence
                                     else 'DOWN' if sig.current_confidence < sig.initial_confidence
                                     else 'FLAT'),
            'status':               sig.status,
            'recommendation':       recommendation,
            'created_at':           sig.created_at.isoformat(),
            'last_updated':         sig.last_updated.isoformat(),
            'last_price':           round(sig.last_price, 2),
            'pnl_pts':              sig.pnl_pts,
            'pnl_atr_units':        pnl_r,
            'max_favorable_pts':    sig.max_favorable_pts,
            'max_adverse_pts':      sig.max_adverse_pts,
            'reasons':              sig.reasons,
            'invalidation_reason':  sig.invalidation_reason,
            'regime_at_creation':   sig.regime_at_creation,
            'session_at_creation':  sig.session_at_creation,
            # Window context
            'window_start':         sig.window_start.isoformat(),
            'window_remaining_s':   rem,
            'window_remaining_pct': round(1 - elp, 2),
            'window_elapsed_pct':   round(elp, 2),
            # Recent confidence history (last 10 ticks)
            'confidence_history':   sig.confidence_history[-10:],
            # One-line summary
            'summary': (
                f"{sig.direction} [{sig.setup}] conf={sig.current_confidence}% "
                f"E:{sig.entry:.1f} SL:{sig.stop_loss:.1f} TP:{sig.take_profit:.1f} "
                f"RR:{sig.risk_reward} | {sig.status}"
            ),
        }

    def _window_snapshot(self) -> List[dict]:
        return [self._sig_to_dict(s) for s in self._active]

    def get_stats(self) -> dict:
        recent = self._history[-100:]
        tp  = sum(1 for s in recent if s.status == 'TP_HIT')
        sl  = sum(1 for s in recent if s.status == 'SL_HIT')
        exp = sum(1 for s in recent if s.status == 'EXPIRED')
        inv = sum(1 for s in recent if s.status == 'INVALIDATED')
        return {
            'window_start':        self._window_start.isoformat(),
            'window_remaining_s':  self._window_remaining_s(),
            'window_elapsed_pct':  round(self._window_elapsed_pct(), 2),
            'active_count':        self._active_count(),
            'total_in_window':     len(self._active),
            'history_count':       len(self._history),
            'tp_hits':             tp,
            'sl_hits':             sl,
            'expired':             exp,
            'invalidated':         inv,
            'win_rate_pct':        round(tp / max(1, tp + sl) * 100, 1),
            'last_regime':         self._last_regime,
            'last_session':        self._last_session,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
signal_manager = SignalWindowManager()
