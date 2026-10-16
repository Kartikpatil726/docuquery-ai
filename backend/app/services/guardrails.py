import logging
import re

from app.services.vector_store import RetrievedChunk

log = logging.getLogger(__name__)

FALLBACK_MESSAGE = 'I cannot find this in the uploaded documents.'

_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r'ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)',
        r'disregard (all |any |the )?(previous|prior|above|system)',
        r'you are now\b',
        r'(reveal|print|show|repeat) (your |the )?(system )?(prompt|instructions)',
        r'</?\s*(source|sources|system|assistant)\b',
        r'new instructions?\s*:',
    )
]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def drop_suspicious(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    safe: list[RetrievedChunk] = []
    for hit in hits:
        if looks_like_injection(hit.text):
            log.warning('Dropped suspicious chunk %s (possible prompt injection)', hit.chunk_id)
            continue
        safe.append(hit)
    return safe


def filter_by_score(hits: list[RetrievedChunk], min_similarity: float) -> list[RetrievedChunk]:
    return [h for h in hits if h.score >= min_similarity]


def _safe_attr(value: str) -> str:
    return re.sub(r'[^\w.\- ]', '_', value)[:120]


def _escape(text: str) -> str:
    return text.replace('<', '&lt;').replace('>', '&gt;')


def build_context(
    hits: list[RetrievedChunk], max_chars: int
) -> tuple[str, dict[str, RetrievedChunk]]:
    labeled: dict[str, RetrievedChunk] = {}
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        source_id = f'S{index}'
        block = (
            f'<source id="{source_id}" file="{_safe_attr(hit.filename)}" page="{hit.page}">\n'
            f'{_escape(hit.text)}\n'
            f'</source>'
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        labeled[source_id] = hit
        used += len(block)
    return '\n\n'.join(blocks), labeled


def validate_citations(cited_ids: list[str], labeled: dict[str, RetrievedChunk]) -> list[str]:
    valid: list[str] = []
    for raw in cited_ids:
        source_id = str(raw).strip().upper()
        if source_id in labeled and source_id not in valid:
            valid.append(source_id)
    return valid


def compute_confidence(scores: list[float], floor: float, ceiling: float = 0.75) -> float:
    if not scores:
        return 0.0
    blended = 0.6 * max(scores) + 0.4 * (sum(scores) / len(scores))
    scaled = (blended - floor) / (ceiling - floor)
    return round(min(max(scaled, 0.0), 1.0), 2)
