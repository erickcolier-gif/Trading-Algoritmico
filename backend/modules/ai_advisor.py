"""
Claude AI advisor module for NAS100 scalping trade analysis.
Uses claude-opus-4-6 with adaptive thinking for deep market analysis.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic library not available. AI advisor will use mock responses.")

SYSTEM_PROMPT = """You are an expert NASDAQ (NAS100/US100) scalping trader working for a PropXP prop firm account.

Your primary role:
- Analyze 4-minute NAS100 chart data and provide precise scalping trade recommendations
- All trades MUST be held for a MINIMUM of 4 minutes (PropXP rule)
- Maximum hold time: 30 minutes (scalping discipline)
- Focus on high-probability setups with clear risk/reward ratios

PropXP Compliance Rules you MUST follow:
- Funded $10k account: max daily loss $500, max drawdown $1000
- Challenge $3k accounts: max daily loss $150, max drawdown $300, profit target $300
- NEVER recommend a trade if daily loss limit is near (within 20%)
- NEVER recommend trading during high-impact news events
- Maximum risk per trade: 1% of account balance
- Preferred risk/reward ratio: minimum 1:1.5 (better 1:2 or higher)
- Maximum open positions: 5

Your analysis framework:
1. Trend identification (EMA alignment, price structure)
2. Momentum confirmation (RSI, MACD, Stochastic)
3. Entry timing (pullbacks, breakouts, reversals)
4. Risk management (ATR-based stops, key S/R levels)
5. News sentiment impact assessment

Response format: You MUST respond with a valid JSON object only (no markdown, no extra text).
"""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this NAS100 scalping opportunity and respond with JSON only.

CURRENT MARKET DATA:
- Symbol: NAS100 (4-minute chart)
- Current Price: {current_price}
- Bid: {bid}
- Ask: {ask}
- Spread: {spread}

TECHNICAL INDICATORS:
{technical_summary}

NEWS SENTIMENT:
- Overall: {news_sentiment} (score: {news_score})
- High-impact events: {high_impact_count}
- Top headline: {top_headline}
- Analysis: {news_analysis}

ACCOUNT STATUS:
- Account Type: {account_type}
- Balance: ${account_balance}
- Current Equity: ${equity}
- Daily P&L: ${daily_pnl}
- Daily Loss Remaining: ${daily_loss_remaining}
- Max Drawdown Remaining: ${drawdown_remaining}
- Can Trade: {can_trade}
- Open Positions: {open_positions}
- Warnings: {warnings}

SIGNAL CONTEXT:
- Primary Signal: {signal}
- Trend: {trend}
- Signal Strength: {strength}%
- ATR: {atr}
- Support Levels: {support}
- Resistance Levels: {resistance}

TASK:
Based on all the above data, provide a precise scalping recommendation.

RESPOND WITH THIS EXACT JSON STRUCTURE (no markdown, no code blocks, just raw JSON):
{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "entry_price": <float or null>,
  "stop_loss": <float or null>,
  "take_profit": <float or null>,
  "risk_reward": <float or null>,
  "confidence": <integer 0-100>,
  "hold_minutes": <integer minimum 4, maximum 30>,
  "reasoning": "<2-3 sentence explanation>",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "propxp_compliant": true | false,
  "analysis_timestamp": "{timestamp}"
}}

CRITICAL RULES:
- hold_minutes must be >= 4 (PropXP minimum)
- If can_trade is false, set recommendation to "HOLD"
- If news has high-impact events, increase risk_level to HIGH and be more conservative
- stop_loss must be at least 1 ATR away from entry
- take_profit must achieve minimum 1.5x risk/reward
"""


def _build_mock_response(signal: str = "HOLD", price: float = 21000.0) -> Dict[str, Any]:
    """Build a mock AI response for testing without API key."""
    import random

    if signal == "BUY":
        entry = round(price + 2, 2)
        sl = round(entry - 50, 2)
        tp = round(entry + 100, 2)
        recommendation = "BUY"
        confidence = random.randint(55, 75)
        reasoning = (
            f"Technical analysis shows bullish EMA alignment with RSI in the 55-65 range. "
            f"MACD histogram is positive and expanding. Entering long at {entry} with "
            f"tight stop at {sl} for a 1:2 risk/reward scalp."
        )
        key_factors = [
            "EMA9 crossing above EMA21 — bullish momentum",
            "RSI at 58 — room to run before overbought",
            "MACD histogram positive and increasing",
        ]
    elif signal == "SELL":
        entry = round(price - 2, 2)
        sl = round(entry + 50, 2)
        tp = round(entry - 100, 2)
        recommendation = "SELL"
        confidence = random.randint(55, 75)
        reasoning = (
            f"Bearish pressure evident with price below all EMAs. RSI declining from overbought levels. "
            f"Entering short at {entry} targeting {tp} with stop at {sl}."
        )
        key_factors = [
            "Price below EMA9, EMA21, and EMA50 — confirmed downtrend",
            "RSI declining from 68 — momentum fading",
            "MACD bearish crossover confirmed",
        ]
    else:
        entry = None
        sl = None
        tp = None
        recommendation = "HOLD"
        confidence = 30
        reasoning = (
            "Market conditions are unclear with mixed signals across indicators. "
            "Waiting for a higher-probability setup before entering. "
            "Preserving capital is the priority in uncertain conditions."
        )
        key_factors = [
            "Mixed EMA signals — no clear trend",
            "RSI in neutral zone (45-55)",
            "Low volume — insufficient conviction",
        ]

    return {
        "recommendation": recommendation,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "risk_reward": 2.0 if recommendation != "HOLD" else None,
        "confidence": confidence,
        "hold_minutes": 8 if recommendation != "HOLD" else 0,
        "reasoning": reasoning,
        "key_factors": key_factors,
        "risk_level": "MEDIUM",
        "propxp_compliant": True,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "mock": True,
    }


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text that may contain other content."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    matches = json_pattern.findall(text)
    for match in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


async def analyze_trade_opportunity(
    market_data: Dict[str, Any],
    technical_analysis: Dict[str, Any],
    news_summary: Dict[str, Any],
    account_status: Dict[str, Any],
    current_price: Dict[str, float],
) -> Dict[str, Any]:
    """
    Use Claude AI to analyze a NAS100 scalping opportunity.

    Args:
        market_data: OHLCV and market context
        technical_analysis: Results from market_analysis.analyze()
        news_summary: Results from news_analyzer.get_news_summary()
        account_status: Results from risk_manager.get_account_risk_status()
        current_price: {bid, ask, spread} from mt5_connector

    Returns:
        Structured dict with trading recommendation.
    """
    from config import config

    api_key = config.anthropic_api_key
    if not api_key or not ANTHROPIC_AVAILABLE:
        logger.info("No API key or anthropic unavailable — returning mock analysis.")
        signal = technical_analysis.get("signal", "HOLD")
        price = current_price.get("bid", 21000.0)
        return _build_mock_response(signal, price)

    # Build the prompt
    ind = technical_analysis.get("indicators", {})
    conditions = technical_analysis.get("conditions", {})
    sr_levels = market_data.get("support_resistance", {"support": [], "resistance": []})

    top_headline = ""
    top_headlines = news_summary.get("top_headlines", [])
    if top_headlines:
        top_headline = top_headlines[0].get("title", "No headline available")

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        current_price=current_price.get("bid", "N/A"),
        bid=current_price.get("bid", "N/A"),
        ask=current_price.get("ask", "N/A"),
        spread=current_price.get("spread", "N/A"),
        technical_summary=f"""
  EMA9={ind.get('ema9', 'N/A')} | EMA21={ind.get('ema21', 'N/A')} | EMA50={ind.get('ema50', 'N/A')}
  RSI={ind.get('rsi', 'N/A')} | Stoch K={ind.get('stoch_k', 'N/A')} D={ind.get('stoch_d', 'N/A')}
  MACD={ind.get('macd', 'N/A')} | Signal={ind.get('macd_signal', 'N/A')} | Hist={ind.get('macd_histogram', 'N/A')}
  BB Upper={ind.get('bb_upper', 'N/A')} | Mid={ind.get('bb_mid', 'N/A')} | Lower={ind.get('bb_lower', 'N/A')}
  ATR={ind.get('atr', 'N/A')} | Volume={ind.get('volume', 'N/A')} | Vol SMA={ind.get('volume_sma', 'N/A')}
  Overbought={conditions.get('is_overbought', False)} | Oversold={conditions.get('is_oversold', False)}
  MACD Bull Cross={conditions.get('macd_bullish_cross', False)} | Bear Cross={conditions.get('macd_bearish_cross', False)}
  Volume Surge={conditions.get('volume_surge', False)} | BB Squeeze={conditions.get('bb_squeeze', False)}""",
        news_sentiment=news_summary.get("overall_sentiment", "NEUTRAL"),
        news_score=news_summary.get("sentiment_score", 0.0),
        high_impact_count=news_summary.get("high_impact_count", 0),
        top_headline=top_headline,
        news_analysis=news_summary.get("analysis_text", "No analysis available"),
        account_type=account_status.get("account_type", "unknown"),
        account_balance=account_status.get("balance", 0),
        equity=account_status.get("equity", 0),
        daily_pnl=account_status.get("daily_pnl", 0),
        daily_loss_remaining=account_status.get("daily_loss_remaining", 0),
        drawdown_remaining=account_status.get("drawdown_remaining", 0),
        can_trade=account_status.get("can_trade", False),
        open_positions=account_status.get("open_positions_count", 0),
        warnings=", ".join(account_status.get("warnings", [])) or "None",
        signal=technical_analysis.get("signal", "NEUTRAL"),
        trend=technical_analysis.get("trend", "UNKNOWN"),
        strength=technical_analysis.get("strength", 0),
        atr=ind.get("atr", "N/A"),
        support=", ".join(str(s) for s in sr_levels.get("support", [])) or "None identified",
        resistance=", ".join(str(r) for r in sr_levels.get("resistance", [])) or "None identified",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)

        full_response = ""
        thinking_content = ""

        # Use adaptive thinking for deeper analysis (Opus 4.6 — budget_tokens deprecated)
        with client.messages.stream(
            model=config.ai_model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                pass
            message = stream.get_final_message()

        # Extract text and thinking content
        for block in message.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    full_response += block.text
                elif block.type == "thinking":
                    thinking_content = getattr(block, "thinking", "")

        logger.debug(f"Claude response length: {len(full_response)}")

        # Parse JSON from response
        parsed = _extract_json_from_text(full_response)
        if parsed is None:
            logger.error(f"Failed to parse JSON from Claude response: {full_response[:500]}")
            return _build_mock_response(
                technical_analysis.get("signal", "HOLD"),
                current_price.get("bid", 21000.0)
            )

        # Validate and enforce rules
        recommendation = parsed.get("recommendation", "HOLD").upper()
        if recommendation not in ("BUY", "SELL", "HOLD"):
            recommendation = "HOLD"
        parsed["recommendation"] = recommendation

        # Enforce minimum hold time
        hold_minutes = parsed.get("hold_minutes", 4)
        if hold_minutes < 4:
            parsed["hold_minutes"] = 4

        # Enforce PropXP compliance
        if not account_status.get("can_trade", True):
            parsed["recommendation"] = "HOLD"
            parsed["propxp_compliant"] = True
            parsed["reasoning"] = "Trade blocked: account risk limits reached. " + parsed.get("reasoning", "")

        # Add timestamp if missing
        if "analysis_timestamp" not in parsed:
            parsed["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add thinking summary if available
        if thinking_content:
            parsed["thinking_summary"] = thinking_content[:200] + "..." if len(thinking_content) > 200 else thinking_content

        parsed["mock"] = False
        return parsed

    except anthropic.APIConnectionError as exc:
        logger.error(f"Claude API connection error: {exc}")
        return _build_mock_response(
            technical_analysis.get("signal", "HOLD"),
            current_price.get("bid", 21000.0)
        )
    except anthropic.APIStatusError as exc:
        logger.error(f"Claude API status error {exc.status_code}: {exc.message}")
        return _build_mock_response(
            technical_analysis.get("signal", "HOLD"),
            current_price.get("bid", 21000.0)
        )
    except anthropic.BadRequestError as exc:
        logger.error(f"Claude bad request (possibly model doesn't support thinking): {exc}")
        # Retry without thinking
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=config.ai_model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            full_response = ""
            for block in message.content:
                if hasattr(block, "text"):
                    full_response += block.text

            parsed = _extract_json_from_text(full_response)
            if parsed:
                parsed["mock"] = False
                if parsed.get("hold_minutes", 0) < 4:
                    parsed["hold_minutes"] = 4
                return parsed
        except Exception as retry_exc:
            logger.error(f"Retry without thinking also failed: {retry_exc}")

        return _build_mock_response(
            technical_analysis.get("signal", "HOLD"),
            current_price.get("bid", 21000.0)
        )
    except Exception as exc:
        logger.error(f"Unexpected Claude error: {exc}")
        return _build_mock_response(
            technical_analysis.get("signal", "HOLD"),
            current_price.get("bid", 21000.0)
        )
