import hmac

from fastapi import Header, HTTPException

from app.core.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail='Invalid or missing API key.',
            headers={'WWW-Authenticate': 'ApiKey'},
        )
