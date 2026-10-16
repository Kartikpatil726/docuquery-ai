import logging
import random
import time
from typing import Callable, TypeVar

from google.genai import errors

log = logging.getLogger(__name__)

T = TypeVar('T')
RETRYABLE_CODES = {429, 500, 503}


def call_with_retry(fn: Callable[[], T], max_attempts: int = 5, base_delay: float = 1.5) -> T:
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except errors.APIError as exc:
            code = getattr(exc, 'code', None)
            if code not in RETRYABLE_CODES or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning('Gemini error %s (attempt %d/%d), retrying in %.1fs', code, attempt, max_attempts, delay)
            time.sleep(delay)
    raise RuntimeError('unreachable')
