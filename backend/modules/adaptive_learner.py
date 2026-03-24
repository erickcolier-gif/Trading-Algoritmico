"""
Adaptive Learner — self-improving signal engine for NAS100 scalping.

How it works:
  1. Every time a signal is emitted, it's stored with entry price, SL, TP, and setup name.
  2. Every price tick is checked against open "pending" signals to see if SL or TP was hit.
  3. Results are written to a local JSON database (adaptive_memory.json).
  4. The learner computes per-setup win rates and per-level reaction scores.
  5. signal_weight(setup_name) returns a multiplier [0.6, 1.4] used in generate_signals
     to boost or penalize setups based on their recent historical performance.

This means the system gets SMARTER over time — setups that consistently work in the
current market regime get boosted; setups that fail get demoted automatically.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'adaptive_memory.json')

# ── Default structure ─────────────────────────────────────────────────────────
_DEFAULT_DB: dict = {
    "signals": [],          # list of all tracked signals (capped at 500)
    "setup_stats": {},      # {setup_name: {wins, losses, total, win_rate}}
    "level_reactions": [],  # [{price_level, type, direction, outcome, timestamp}]
    "regime": "UNKNOWN",    # current detected market regime
    "last_updated": None,
}

# How many recent signals to use for weight calculation
_LOOKBACK = 30
# Max signals stored in DB
_MAX_SIGNALS = 500


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_db() -> dict:
    try:
        if os.path.exists(_DB_PATH):
            with open(_DB_PATH, 'r') as f:
                db = json.load(f)
                # Ensure all keys exist (forward compatibility)
                for k, v in _DEFAULT_DB.items():
                    if k not in db:
                        db[k] = v
                return db
    except Exception as e:
        logger.warning(f"[Learner] Could not load adaptive_memory.json: {e}")
    return dict(_DEFAULT_DB)


def _save_db(db: dict) -> None:
    try:
        db['last_updated'] = datetime.now(timezone.utc).isoformat()
        with open(_DB_PATH, 'w') as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        logger.warning(f"[Learner] Could not save adaptive_memory.json: {e}")


# ── Core class ────────────────────────────────────────────────────────────────

class AdaptiveLearner:
    """
    Learns from past signal outcomes to improve future signal quality.
    Thread-safe for single-process async usage (no multiprocessing).
    """

    def __init__(self):
        self._db = _load_db()
        logger.info(f"[Learner] Loaded {len(self._db['signals'])} historical signals.")

    # ── Signal tracking ───────────────────────────────────────────────────────

    def register_signal(self, signal: dict) -> str:
        """
        Store a new signal for outcome tracking.
        Returns the signal's tracking ID.
        """
        sid = f"{signal.get('direction','?')}-{signal.get('setup','?')}-{datetime.now().strftime('%H%M%S%f')}"
        record = {
            "id": sid,
            "direction": signal.get('direction'),
            "setup": signal.get('setup', 'Unknown'),
            "confidence": signal.get('confidence', 0),
            "entry": signal.get('entry', 0),
            "stop_loss": signal.get('stop_loss', 0),
            "take_profit": signal.get('take_profit', 0),
            "atr": signal.get('atr', 0),
            "status": "PENDING",   # PENDING | WIN | LOSS | EXPIRED
            "outcome": None,
            "pnl_pts": None,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "resolved_at": None,
        }
        self._db['signals'].append(record)
        # Cap memory
        if len(self._db['signals']) > _MAX_SIGNALS:
            self._db['signals'] = self._db['signals'][-_MAX_SIGNALS:]
        _save_db(self._db)
        logger.info(f"[Learner] Registered signal {sid}")
        return sid

    def update_price(self, current_price: float) -> List[dict]:
        """
        Check all PENDING signals against current_price.
        Resolves signals that hit TP or SL.
        Returns list of newly resolved signals.
        """
        resolved = []
        changed = False

        for sig in self._db['signals']:
            if sig['status'] != 'PENDING':
                continue
            entry = sig.get('entry', 0)
            sl = sig.get('stop_loss', 0)
            tp = sig.get('take_profit', 0)
            direction = sig.get('direction')
            if not entry:
                continue

            # Check expiry (signal older than 4 hours = expired)
            try:
                reg = datetime.fromisoformat(sig['registered_at'].replace('Z', '+00:00'))
                age_h = (datetime.now(timezone.utc) - reg).total_seconds() / 3600
                if age_h > 4:
                    sig['status'] = 'EXPIRED'
                    sig['resolved_at'] = datetime.now(timezone.utc).isoformat()
                    changed = True
                    continue
            except Exception:
                pass

            hit_tp = hit_sl = False
            if direction == 'BUY':
                hit_tp = current_price >= tp > 0
                hit_sl = current_price <= sl > 0
            elif direction == 'SELL':
                hit_tp = current_price <= tp > 0
                hit_sl = current_price >= sl > 0

            if hit_tp:
                sig['status'] = 'WIN'
                sig['outcome'] = 'TP_HIT'
                sig['pnl_pts'] = round(abs(tp - entry), 2)
            elif hit_sl:
                sig['status'] = 'LOSS'
                sig['outcome'] = 'SL_HIT'
                sig['pnl_pts'] = -round(abs(sl - entry), 2)

            if hit_tp or hit_sl:
                sig['resolved_at'] = datetime.now(timezone.utc).isoformat()
                changed = True
                resolved.append(sig)
                self._update_setup_stats(sig)
                logger.info(f"[Learner] {sig['status']} — {sig['setup']} "
                            f"{sig['direction']} @ {entry}  P&L: {sig['pnl_pts']} pts")

        if changed:
            _save_db(self._db)

        return resolved

    def _update_setup_stats(self, sig: dict) -> None:
        """Recalculate win rate for the setup that just resolved."""
        setup = sig.get('setup', 'Unknown')
        stats = self._db['setup_stats'].get(setup, {'wins': 0, 'losses': 0, 'total': 0})

        if sig['status'] == 'WIN':
            stats['wins'] += 1
        elif sig['status'] == 'LOSS':
            stats['losses'] += 1
        stats['total'] = stats['wins'] + stats['losses']
        stats['win_rate'] = round(stats['wins'] / stats['total'] * 100, 1) if stats['total'] > 0 else 50.0

        # Recent-only win rate (last _LOOKBACK resolved for this setup)
        recent = [s for s in self._db['signals']
                  if s.get('setup') == setup and s['status'] in ('WIN', 'LOSS')][-_LOOKBACK:]
        if recent:
            recent_wins = sum(1 for s in recent if s['status'] == 'WIN')
            stats['recent_win_rate'] = round(recent_wins / len(recent) * 100, 1)
            stats['recent_sample'] = len(recent)
        self._db['setup_stats'][setup] = stats

    # ── Adaptive weight ───────────────────────────────────────────────────────

    def signal_weight(self, setup_name: str) -> float:
        """
        Returns a confidence multiplier [0.65, 1.35] for a given setup.
        Based on recent win rate vs expected 65% baseline:
          - 80%+ win rate  → 1.35× boost
          - 70–80%         → 1.15×
          - 55–70%         → 1.00× (neutral)
          - 45–55%         → 0.85×
          - <45%           → 0.65× (penalize)
        Requires at least 5 resolved signals to apply adjustment.
        """
        stats = self._db['setup_stats'].get(setup_name, {})
        sample = stats.get('recent_sample', 0)
        if sample < 5:
            return 1.0  # not enough data — don't adjust

        wr = stats.get('recent_win_rate', 65.0)
        if wr >= 80:
            return 1.35
        elif wr >= 70:
            return 1.15
        elif wr >= 55:
            return 1.00
        elif wr >= 45:
            return 0.85
        else:
            return 0.65

    # ── Level reaction memory ─────────────────────────────────────────────────

    def record_level_reaction(self, level: float, level_type: str,
                               direction: str, outcome: str) -> None:
        """
        Record how price reacted at a key level.
        level_type: 'OB', 'FVG', 'VWAP', 'EMA9', 'EMA21', 'SwingHigh', 'SwingLow', 'BOS'
        direction: 'BUY' or 'SELL'
        outcome: 'BOUNCE', 'BREAK'
        """
        self._db['level_reactions'].append({
            "level": round(level, 2),
            "type": level_type,
            "direction": direction,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 200 level reactions
        self._db['level_reactions'] = self._db['level_reactions'][-200:]
        _save_db(self._db)

    def level_bounce_rate(self, level_type: str) -> float:
        """
        Returns the historical bounce rate (0–1) for a given level type.
        Used to upweight setups at levels with high bounce history.
        """
        reactions = [r for r in self._db['level_reactions'] if r['type'] == level_type]
        if len(reactions) < 3:
            return 0.65  # assume 65% baseline
        bounces = sum(1 for r in reactions if r['outcome'] == 'BOUNCE')
        return round(bounces / len(reactions), 3)

    # ── Stats export ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return learning stats for the API / dashboard."""
        total_resolved = sum(
            1 for s in self._db['signals']
            if s['status'] in ('WIN', 'LOSS')
        )
        total_wins = sum(
            1 for s in self._db['signals']
            if s['status'] == 'WIN'
        )
        total_pending = sum(
            1 for s in self._db['signals']
            if s['status'] == 'PENDING'
        )
        overall_wr = round(total_wins / total_resolved * 100, 1) if total_resolved > 0 else None

        # Recent pnl (last 20 resolved)
        recent = [s for s in self._db['signals']
                  if s['status'] in ('WIN', 'LOSS') and s.get('pnl_pts') is not None][-20:]
        recent_pnl = round(sum(s['pnl_pts'] for s in recent), 1) if recent else 0

        return {
            "total_signals": len(self._db['signals']),
            "total_resolved": total_resolved,
            "total_pending": total_pending,
            "overall_win_rate": overall_wr,
            "recent_pnl_pts": recent_pnl,
            "setup_stats": self._db['setup_stats'],
            "last_updated": self._db.get('last_updated'),
        }

    def get_recent_signals(self, limit: int = 20) -> List[dict]:
        """Return most recent tracked signals (newest first)."""
        return list(reversed(self._db['signals'][-limit:]))


# ── Singleton ─────────────────────────────────────────────────────────────────
learner = AdaptiveLearner()
