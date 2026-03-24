"""
Risk management module for PropXP prop firm compliance.
Validates trades against daily loss limits, drawdown limits, and position sizing rules.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# PropXP point value for NAS100 (approximate: $1 per point per 0.01 lot)
NAS100_POINT_VALUE = 1.0  # USD per point per micro lot (0.01)
NAS100_LOT_POINT_VALUE = 100.0  # USD per point per standard lot (1.0)


def check_trade_allowed(
    account_info: Dict[str, Any],
    config_account: Any,  # AccountConfig
    open_positions_count: int = 0,
) -> Dict[str, Any]:
    """
    Check if a new trade is allowed under PropXP rules.

    Args:
        account_info: From mt5_connector.get_account_info()
        config_account: AccountConfig instance
        open_positions_count: Number of currently open positions

    Returns:
        Dict with 'allowed' (bool) and 'reasons' (list of strings).
    """
    reasons = []
    allowed = True

    balance = account_info.get("balance", 0.0)
    equity = account_info.get("equity", 0.0)
    current_profit = account_info.get("profit", 0.0)

    # Use config balance as reference if account balance is 0 (mock)
    ref_balance = balance if balance > 0 else config_account.balance

    # Calculate daily P&L (approximation using open profit)
    # In production, this should track realized P&L from the day's history
    daily_loss_used = min(0.0, current_profit)  # Only losses count
    daily_loss_abs = abs(daily_loss_used)

    # Check daily loss limit
    if daily_loss_abs >= config_account.max_daily_loss:
        reasons.append(
            f"Daily loss limit reached: ${daily_loss_abs:.2f} >= ${config_account.max_daily_loss:.2f}"
        )
        allowed = False
    elif daily_loss_abs >= config_account.max_daily_loss * 0.8:
        reasons.append(
            f"WARNING: Approaching daily loss limit (${daily_loss_abs:.2f} / ${config_account.max_daily_loss:.2f})"
        )

    # Check max drawdown
    drawdown = ref_balance - equity
    if drawdown >= config_account.max_drawdown:
        reasons.append(
            f"Max drawdown limit reached: ${drawdown:.2f} >= ${config_account.max_drawdown:.2f}"
        )
        allowed = False
    elif drawdown >= config_account.max_drawdown * 0.8:
        reasons.append(
            f"WARNING: Approaching max drawdown (${drawdown:.2f} / ${config_account.max_drawdown:.2f})"
        )

    # Check open positions limit (from PropXP config)
    max_positions = getattr(config_account, "max_positions", 5)
    if not hasattr(config_account, "max_positions"):
        # Fallback: read from PropXP global config if available
        from config import config as app_config
        max_positions = app_config.propxp.max_open_positions
    if open_positions_count >= max_positions:
        reasons.append(f"Maximum open positions reached: {open_positions_count}/{max_positions}")
        allowed = False

    # For challenge accounts: block trading when profit target is reached
    if getattr(config_account, "profit_target", None) is not None:
        balance = account_info.get("balance", config_account.balance)
        equity = account_info.get("equity", balance)
        ref_balance = balance if balance > 0 else config_account.balance
        total_profit = equity - ref_balance
        if total_profit >= config_account.profit_target:
            reasons.append(
                f"Profit target reached: ${total_profit:.2f} >= ${config_account.profit_target:.2f}. "
                f"Stop trading and request payout."
            )
            allowed = False

    return {
        "allowed": allowed,
        "reasons": reasons,
        "daily_loss_used": daily_loss_abs,
        "daily_loss_limit": config_account.max_daily_loss,
        "drawdown": drawdown,
        "drawdown_limit": config_account.max_drawdown,
    }


def calculate_lot_size(
    account_balance: float,
    risk_percent: float,
    sl_pips: float,
    symbol_point_value: float = NAS100_LOT_POINT_VALUE,
    min_lot: float = 0.01,
    max_lot: float = 1.0,
    lot_step: float = 0.01,
) -> float:
    """
    Calculate appropriate lot size based on risk percentage.

    Risk formula: lot_size = (account_balance * risk_percent / 100) / (sl_pips * point_value)

    Args:
        account_balance: Account balance in USD
        risk_percent: Risk as percentage of balance (e.g., 1.0 for 1%)
        sl_pips: Stop loss distance in pips/points
        symbol_point_value: Dollar value per point per standard lot
        min_lot: Minimum lot size
        max_lot: Maximum lot size
        lot_step: Lot size step increment

    Returns:
        Calculated lot size rounded to lot_step.
    """
    if account_balance <= 0 or risk_percent <= 0 or sl_pips <= 0:
        return min_lot

    risk_amount = account_balance * (risk_percent / 100.0)
    raw_lot = risk_amount / (sl_pips * symbol_point_value)

    # Round to nearest lot_step
    lot = round(raw_lot / lot_step) * lot_step

    # Clamp to min/max
    lot = max(min_lot, min(max_lot, lot))

    return round(lot, 2)


def get_account_risk_status(
    account_info: Dict[str, Any],
    config_account: Any,  # AccountConfig
    open_positions_count: int = 0,
) -> Dict[str, Any]:
    """
    Compute a comprehensive risk status for the account.

    Returns dict with:
    - daily_pnl: estimated daily P&L
    - daily_loss_remaining: how much more can be lost today
    - drawdown_percent: current drawdown as percentage
    - drawdown_remaining: remaining drawdown buffer
    - can_trade: bool
    - warnings: list of warning strings
    - account_type, balance, equity
    - profit_target_remaining (for challenge accounts)
    """
    balance = account_info.get("balance", config_account.balance)
    equity = account_info.get("equity", balance)
    profit = account_info.get("profit", 0.0)

    # Reference balance (use config if actual is 0)
    ref_balance = balance if balance > 0 else config_account.balance

    # Daily P&L (approximate — in production track realized + unrealized from session start)
    daily_pnl = profit  # Unrealized only in this simplified version

    # Daily loss calculation
    daily_loss = min(0.0, daily_pnl)
    daily_loss_abs = abs(daily_loss)
    daily_loss_remaining = max(0.0, config_account.max_daily_loss - daily_loss_abs)
    daily_loss_pct = (daily_loss_abs / config_account.max_daily_loss) * 100 if config_account.max_daily_loss > 0 else 0

    # Drawdown calculation
    drawdown = max(0.0, ref_balance - equity)
    drawdown_pct = (drawdown / config_account.max_drawdown) * 100 if config_account.max_drawdown > 0 else 0
    drawdown_remaining = max(0.0, config_account.max_drawdown - drawdown)
    drawdown_balance_pct = (drawdown / ref_balance * 100) if ref_balance > 0 else 0

    # Check if trading is allowed
    trade_check = check_trade_allowed(account_info, config_account, open_positions_count)
    can_trade = trade_check["allowed"]

    # Build warnings list
    warnings = [r for r in trade_check["reasons"] if "WARNING" in r]
    errors = [r for r in trade_check["reasons"] if "WARNING" not in r]

    # Profit target (challenge accounts)
    profit_target_remaining = None
    profit_target_pct = None
    if config_account.profit_target:
        total_profit = max(0.0, equity - ref_balance)
        profit_target_remaining = max(0.0, config_account.profit_target - total_profit)
        profit_target_pct = min(100.0, (total_profit / config_account.profit_target) * 100)

    return {
        "account_type": config_account.account_type,
        "account_name": config_account.name,
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_loss": round(daily_loss_abs, 2),
        "daily_loss_remaining": round(daily_loss_remaining, 2),
        "daily_loss_pct": round(daily_loss_pct, 1),
        "drawdown": round(drawdown, 2),
        "drawdown_pct": round(drawdown_pct, 1),
        "drawdown_balance_pct": round(drawdown_balance_pct, 2),
        "drawdown_remaining": round(drawdown_remaining, 2),
        "can_trade": can_trade,
        "warnings": warnings,
        "errors": errors,
        "open_positions_count": open_positions_count,
        "profit_target": config_account.profit_target,
        "profit_target_remaining": profit_target_remaining,
        "profit_target_pct": profit_target_pct,
        "max_daily_loss": config_account.max_daily_loss,
        "max_drawdown": config_account.max_drawdown,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def validate_trade(
    order_params: Dict[str, Any],
    account_info: Dict[str, Any],
    config_account: Any,  # AccountConfig
    open_positions_count: int = 0,
) -> Dict[str, Any]:
    """
    Perform full trade validation before placement.

    Args:
        order_params: {direction, lot, sl, tp, entry_price}
        account_info: From mt5_connector.get_account_info()
        config_account: AccountConfig
        open_positions_count: Current open positions

    Returns:
        Dict with 'valid' (bool), 'errors' (list), 'warnings' (list),
        'recommended_lot' (float), 'risk_amount' (float)
    """
    errors = []
    warnings = []

    direction = order_params.get("direction", "").upper()
    lot = order_params.get("lot", 0.0)
    sl = order_params.get("sl", 0.0)
    tp = order_params.get("tp", 0.0)
    entry_price = order_params.get("entry_price", account_info.get("current_price", 0.0))

    balance = account_info.get("balance", config_account.balance)
    ref_balance = balance if balance > 0 else config_account.balance

    # Validate direction
    if direction not in ("BUY", "SELL"):
        errors.append(f"Invalid direction: '{direction}'. Must be BUY or SELL.")

    # Validate lot size
    if lot <= 0:
        errors.append("Lot size must be greater than 0.")
    elif lot > 1.0:
        warnings.append(f"Large lot size: {lot}. Consider reducing position size.")

    # Validate stop loss and take profit
    if direction == "BUY":
        if sl > 0 and entry_price > 0 and sl >= entry_price:
            errors.append(f"BUY stop loss ({sl}) must be below entry price ({entry_price}).")
        if tp > 0 and entry_price > 0 and tp <= entry_price:
            errors.append(f"BUY take profit ({tp}) must be above entry price ({entry_price}).")
    elif direction == "SELL":
        if sl > 0 and entry_price > 0 and sl <= entry_price:
            errors.append(f"SELL stop loss ({sl}) must be above entry price ({entry_price}).")
        if tp > 0 and entry_price > 0 and tp >= entry_price:
            errors.append(f"SELL take profit ({tp}) must be below entry price ({entry_price}).")

    # Check risk/reward ratio
    if entry_price > 0 and sl > 0 and tp > 0:
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        rr = reward / risk if risk > 0 else 0
        if rr < 1.5:
            warnings.append(f"Risk/reward ratio {rr:.2f} is below minimum 1.5.")

    # Calculate risk amount
    sl_pips = abs(entry_price - sl) if entry_price > 0 and sl > 0 else 50.0
    risk_amount = lot * sl_pips * NAS100_LOT_POINT_VALUE if sl_pips > 0 else 0.0
    risk_pct = (risk_amount / ref_balance * 100) if ref_balance > 0 else 0.0

    # Check risk percentage
    if risk_pct > 2.0:
        warnings.append(
            f"Risk per trade ({risk_pct:.1f}%) exceeds recommended 1-2% of balance."
        )
    if risk_pct > 5.0:
        errors.append(
            f"Risk per trade ({risk_pct:.1f}%) is dangerously high. Maximum 5%."
        )

    # Check PropXP trade allowance
    trade_check = check_trade_allowed(account_info, config_account, open_positions_count)
    if not trade_check["allowed"]:
        errors.extend([r for r in trade_check["reasons"] if "WARNING" not in r])
    warnings.extend([r for r in trade_check["reasons"] if "WARNING" in r])

    # Calculate recommended lot size (1% risk)
    recommended_lot = calculate_lot_size(
        account_balance=ref_balance,
        risk_percent=config_account.risk_per_trade_pct,
        sl_pips=sl_pips,
        symbol_point_value=NAS100_LOT_POINT_VALUE,
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "risk_amount": round(risk_amount, 2),
        "risk_pct": round(risk_pct, 2),
        "recommended_lot": recommended_lot,
        "sl_pips": round(sl_pips, 1),
        "risk_reward": round(abs(tp - entry_price) / abs(entry_price - sl), 2) if sl > 0 and tp > 0 and entry_price > 0 else None,
    }
