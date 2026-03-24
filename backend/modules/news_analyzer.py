"""
AI news analysis module.
Priority: Groq (free, Llama 3.3 70B) → OpenAI → keyword fallback.
Groq uses the same OpenAI SDK — just a different base_url. No extra package needed.
"""
import json
import logging
import re
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    HAS_AI_SDK = True
except ImportError:
    HAS_AI_SDK = False
    logger.warning("openai SDK not installed — using keyword fallback only")


# ── Keyword fallback ─────────────────────────────────────────────────────────

BULLISH_WORDS = {
    'surge', 'rally', 'soar', 'gain', 'rise', 'jump', 'climb', 'beat', 'exceed',
    'record', 'high', 'strong', 'bull', 'optimism', 'growth', 'profit', 'upgrade',
    'recovery', 'rebound', 'boost', 'positive', 'outperform', 'momentum',
}
BEARISH_WORDS = {
    'fall', 'drop', 'decline', 'plunge', 'crash', 'tumble', 'sink', 'miss',
    'weak', 'bear', 'recession', 'fear', 'sell', 'loss', 'downgrade', 'inflation',
    'rate hike', 'concern', 'risk', 'warning', 'slowdown', 'layoffs', 'default',
}
HIGH_IMPACT_WORDS = {
    'fed', 'federal reserve', 'fomc', 'interest rate', 'inflation', 'cpi', 'gdp',
    'jobs report', 'nonfarm', 'earnings', 'guidance', 'recession', 'tariff',
}


def _keyword_sentiment(text: str) -> float:
    tl = text.lower()
    score = sum(0.1 for w in BULLISH_WORDS if w in tl) - sum(0.1 for w in BEARISH_WORDS if w in tl)
    return max(-1.0, min(1.0, score))


def _is_high_impact(text: str) -> bool:
    return any(w in text.lower() for w in HIGH_IMPACT_WORDS)


# ── Shared prompt builder ─────────────────────────────────────────────────────

def _build_prompt(news_items, technical_context, pattern_context) -> str:
    news_text = "\n".join(
        f"[{i+1}] {item.get('title','')} — {item.get('source','')} ({item.get('timestamp','')[:10]})"
        for i, item in enumerate(news_items[:15])
    )

    tech_text = ""
    if technical_context:
        ind = technical_context.get('indicators', {})
        tech_text = f"""
Current Technical Context:
- Price: {technical_context.get('price')} | Trend: {technical_context.get('trend')}
- RSI: {ind.get('rsi')} | MACD Hist: {ind.get('macd_hist')} | ATR: {ind.get('atr')}
- VWAP: {technical_context.get('vwap', {}).get('position')} | Structure: {technical_context.get('smc', {}).get('market_structure', {}).get('trend')}
"""

    pattern_text = ""
    if pattern_context:
        stats = pattern_context.get('statistics', {})
        if stats.get('total_matches', 0) > 0:
            pattern_text = f"""
Historical Pattern Context:
- {stats.get('total_matches')} similar patterns found
- {stats.get('win_rate_long')}% bullish, {stats.get('win_rate_short')}% bearish
- Avg move: {stats.get('avg_move_pct',0):+.2f}% | Range: +{stats.get('avg_max_up_pct',0):.2f}% / {stats.get('avg_max_down_pct',0):.2f}%
- Bias: {stats.get('bias')}
"""

    sr_text = ""
    if technical_context:
        sr = technical_context.get('support_resistance', {})
        nr_s = sr.get('nearest_support')
        nr_r = sr.get('nearest_resistance')
        rev_zones = technical_context.get('reversal_zones', [])
        if nr_s or nr_r or rev_zones:
            sr_text = "\nKey Support & Resistance Levels:\n"
            if nr_r:
                sr_text += (f"- Nearest Resistance: {nr_r.get('price')} "
                            f"({nr_r.get('label')}, strength {nr_r.get('strength')}/5, "
                            f"dist {nr_r.get('distance_atr', '?')} ATR)\n")
            if nr_s:
                sr_text += (f"- Nearest Support: {nr_s.get('price')} "
                            f"({nr_s.get('label')}, strength {nr_s.get('strength')}/5, "
                            f"dist {nr_s.get('distance_atr', '?')} ATR)\n")
            for z in rev_zones[:3]:
                sr_text += (f"- Reversal Zone: {z.get('price')} ({z.get('direction')}, "
                            f"{z.get('strength')} score={z.get('score')}, "
                            f"confluences: {', '.join(z.get('confluences', [])[:3])})\n")

    return f"""You are an expert NASDAQ 100 (QQQ ETF) intraday trader and quant analyst specializing in reversal detection.

LATEST NEWS:
{news_text}
{tech_text}{pattern_text}{sr_text}
Analyze whether the news supports or opposes the technical picture. Pay special attention to:
1. Whether news catalysts align with the key S&R levels shown above
2. Reversal probability at those levels given the news context
3. Which direction is most likely to BOUNCE or REJECT at the nearest levels

Provide a JSON response with EXACTLY these fields (no extra text, pure JSON):
{{
  "overall_sentiment": <-1.0 to 1.0>,
  "sentiment_label": "<STRONGLY_BEARISH|BEARISH|NEUTRAL|BULLISH|STRONGLY_BULLISH>",
  "direction_bias": "<UP|DOWN|SIDEWAYS>",
  "direction_confidence": <0-100>,
  "volatility_expected": "<LOW|NORMAL|ELEVATED|HIGH|EXTREME>",
  "volatility_increase_pct": <number>,
  "price_range_estimate": {{"low_pct": <number>, "high_pct": <number>}},
  "time_horizon": "<string>",
  "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "high_impact_events": ["<event1>"],
  "historical_comparison": "<string>",
  "reasoning": "<3-4 sentence explanation>",
  "risk_warnings": ["<warning1>", "<warning2>"],
  "trading_notes": "<actionable note>",
  "reversal_probability": <0-100>,
  "reversal_direction": "<UP|DOWN|NONE>",
  "reversal_reasoning": "<1-2 sentence explanation of most likely reversal scenario>",
  "key_price_levels": [
    {{"level": <price>, "type": "<support|resistance>", "significance": "<why this level matters given the news>"}}
  ]
}}"""


def _extract_json(text: str) -> dict:
    """Extract JSON from response even if wrapped in markdown or extra text."""
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract from code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Find first JSON object in text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError("No valid JSON found in response")


# ── AI call (works for Groq AND OpenAI — same SDK) ───────────────────────────

async def _call_ai(
    api_key: str,
    model: str,
    prompt: str,
    base_url: Optional[str] = None,
    provider: str = "openai",
) -> dict:
    """Make a chat completion call. base_url=None → OpenAI. Set for Groq."""
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = AsyncOpenAI(**kwargs)

    # Some models/providers support json_object mode
    supports_json_mode = provider in ("openai",) or "llama" in model or "mixtral" in model

    create_kwargs = dict(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert quantitative analyst specializing in NASDAQ 100 intraday trading. "
                    "Always respond with valid JSON only, no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    if supports_json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**create_kwargs)
    content = response.choices[0].message.content
    return _extract_json(content)


# ── Main public function ──────────────────────────────────────────────────────

async def analyze_news_with_ai(
    news_items: List[dict],
    technical_context: Optional[dict] = None,
    pattern_context: Optional[dict] = None,
    # Groq (free — primary)
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    # OpenAI (paid — secondary)
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
) -> dict:
    """
    Analyze news with AI. Priority: Groq → OpenAI → keyword fallback.
    """
    if not news_items:
        return _fallback_analysis([])

    if not HAS_AI_SDK:
        return _fallback_analysis(news_items)

    prompt = _build_prompt(news_items, technical_context, pattern_context)

    # 1️⃣ Try Groq (free)
    if groq_api_key:
        try:
            result = await _call_ai(
                api_key=groq_api_key,
                model=groq_model,
                prompt=prompt,
                base_url="https://api.groq.com/openai/v1",
                provider="groq",
            )
            result.update({
                'source': 'groq',
                'model': groq_model,
                'analyzed_at': datetime.now().isoformat(),
                'news_count': len(news_items),
            })
            logger.info(f"Groq analysis OK — {result.get('sentiment_label')} / {result.get('direction_bias')}")
            return result
        except Exception as e:
            logger.warning(f"Groq analysis failed: {e} — trying OpenAI")

    # 2️⃣ Try OpenAI (paid fallback)
    if openai_api_key:
        try:
            result = await _call_ai(
                api_key=openai_api_key,
                model=openai_model,
                prompt=prompt,
                base_url=None,
                provider="openai",
            )
            result.update({
                'source': 'openai',
                'model': openai_model,
                'analyzed_at': datetime.now().isoformat(),
                'news_count': len(news_items),
            })
            logger.info(f"OpenAI analysis OK — {result.get('sentiment_label')}")
            return result
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e} — using keyword fallback")

    # 3️⃣ Keyword fallback
    return _fallback_analysis(news_items)


def _fallback_analysis(news_items: List[dict]) -> dict:
    """Pure keyword-based sentiment when no AI is available."""
    if not news_items:
        return {
            'overall_sentiment': 0, 'sentiment_label': 'NEUTRAL',
            'direction_bias': 'SIDEWAYS', 'direction_confidence': 0,
            'volatility_expected': 'NORMAL', 'volatility_increase_pct': 0,
            'price_range_estimate': {'low_pct': -0.5, 'high_pct': 0.5},
            'time_horizon': '1-4 hours', 'key_themes': ['No news'],
            'high_impact_events': [], 'historical_comparison': 'N/A',
            'reasoning': 'No news available for analysis.',
            'risk_warnings': ['No AI configured — add GROQ_API_KEY for free analysis'],
            'trading_notes': 'Use technical analysis only.',
            'source': 'keyword_fallback', 'analyzed_at': datetime.now().isoformat(), 'news_count': 0,
        }

    scores, high_impact_count, themes = [], 0, set()
    for item in news_items:
        text = item.get('title', '') + ' ' + item.get('summary', '')
        score = _keyword_sentiment(text)
        weight = 1.5 if _is_high_impact(text) else 1.0
        if _is_high_impact(text):
            high_impact_count += 1
        scores.append(score * weight)
        for word in HIGH_IMPACT_WORDS:
            if word in text.lower():
                themes.add(word.title())

    overall = max(-1.0, min(1.0, sum(scores) / len(scores) if scores else 0))

    if overall > 0.3:
        label, direction, conf = ('STRONGLY_BULLISH' if overall > 0.6 else 'BULLISH'), 'UP', int(min(90, abs(overall) * 100))
    elif overall < -0.3:
        label, direction, conf = ('STRONGLY_BEARISH' if overall < -0.6 else 'BEARISH'), 'DOWN', int(min(90, abs(overall) * 100))
    else:
        label, direction, conf = 'NEUTRAL', 'SIDEWAYS', 20

    return {
        'overall_sentiment': round(overall, 2),
        'sentiment_label': label,
        'direction_bias': direction,
        'direction_confidence': conf,
        'volatility_expected': 'ELEVATED' if high_impact_count >= 2 else 'NORMAL',
        'volatility_increase_pct': 30 if high_impact_count >= 2 else 10,
        'price_range_estimate': {
            'low_pct': round(-abs(overall) * 0.8 - 0.3, 2),
            'high_pct': round(abs(overall) * 0.8 + 0.3, 2),
        },
        'time_horizon': '1-4 hours',
        'key_themes': list(themes)[:5] or ['General Market'],
        'high_impact_events': [
            item['title'] for item in news_items[:3] if _is_high_impact(item.get('title', ''))
        ],
        'historical_comparison': 'N/A — add GROQ_API_KEY for free AI analysis',
        'reasoning': (
            f"Keyword analysis of {len(news_items)} headlines. Sentiment: {label}. "
            f"{high_impact_count} high-impact events detected. Score: {overall:+.2f}."
        ),
        'risk_warnings': [
            'Using keyword fallback — add GROQ_API_KEY (free at groq.com) for deep analysis',
        ],
        'trading_notes': f"Bias: {label}. Consider {'long' if direction == 'UP' else 'short' if direction == 'DOWN' else 'neutral'} positions.",
        'source': 'keyword_fallback',
        'analyzed_at': datetime.now().isoformat(),
        'news_count': len(news_items),
    }
