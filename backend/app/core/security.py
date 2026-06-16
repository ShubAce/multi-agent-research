"""
app/core/security.py
API key authentication and basic input sanitisation.
"""

import re
from fastapi import Header, HTTPException, status
from app.core.config import get_settings

settings = get_settings()

# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"you\s+are\s+now",
    r"forget\s+everything",
    r"system\s+prompt",
    r"<\|endoftext\|>",
    r"\[INST\]",
    r"###\s*instruction",
    r"act\s+as\s+if",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> bool:
    """Return True if the text looks like a prompt injection attempt."""
    return any(pattern.search(text) for pattern in _COMPILED)


def sanitise_query(query: str) -> str:
    """Strip leading/trailing whitespace and basic HTML entities."""
    q = query.strip()
    q = q.replace("<", "&lt;").replace(">", "&gt;")
    return q


# ── FastAPI dependency ────────────────────────────────────────────────────────
async def verify_api_key(x_api_key: str = Header(..., alias="X-Api-Key")) -> str:
    """
    Simple API-key gate.
    Pass `X-Api-Key: <your key>` in every request.
    In development, it is bypassed for local ease-of-use.
    In production, swap this for JWT or OAuth2.
    """
    if settings.environment == "development":
        return x_api_key
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return x_api_key
