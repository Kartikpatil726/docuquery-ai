import logging
import re
import time
from app.core.config import get_settings
from app.models.schemas import Citation, QueryResponse
from app.services import guardrails
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.vector_store import get_vector_store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = '''You are DocuQuery, an intelligent and helpful assistant.

GUIDELINES:
1. Base your core answer on the uploaded documents provided in the <source> blocks.
2. If the user asks a question related to the document, provide the direct answer from the document first and cite it with [S1], [S2], etc.
3. You MAY also provide helpful, relevant, and related context or insights based on standard professional knowledge to give a complete answer.
4. If sources are present, always include the relevant source IDs in the "sources" list.
5. If the question is completely irrelevant and unrelated to the document topic, set answerable to false and answer: I cannot find this in the uploaded documents.'''

def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

def _fallback(started: float, reason: str) -> QueryResponse:
    log.info('Fallback answer used: %s', reason)
    return QueryResponse(
        answer=guardrails.FALLBACK_MESSAGE,
        answerable=False,
        confidence=0.0,
        citations=[],
        latency_ms=_elapsed_ms(started),
    )

def _excerpt(text: str, limit: int = 300) -> str:
    flat = re.sub(r'\s+', ' ', text).strip()
    return flat if len(flat) <= limit else flat[:limit].rstrip() + '...'

def answer_question(question: str, doc_id: str | None = None) -> QueryResponse:
    started = time.perf_counter()
    s = get_settings()
    query_vector = get_embedder().embed_query(question)
    hits = get_vector_store().search(query_vector, k=s.top_k, doc_id=doc_id)
    hits = guardrails.filter_by_score(hits, s.min_similarity)
    hits = guardrails.drop_suspicious(hits)
    
    if not hits:
        return _fallback(started, 'no chunk above the similarity threshold')

    context, labeled = guardrails.build_context(hits, s.max_context_chars)
    user_prompt = f'SOURCES:\n{context}\n\nQUESTION: {question}\n\nReturn the JSON object now.'
    llm_answer = get_llm().generate_answer(SYSTEM_PROMPT, user_prompt)
    
    if not llm_answer.answerable:
        return _fallback(started, 'model reported it was not answerable')

    cited_ids = guardrails.validate_citations(llm_answer.sources, labeled)
    
    # Agar model ne related generic answer diya par exact citation skip ki, toh top hit fallback rakhein
    if not cited_ids and labeled:
        cited_ids = [list(labeled.keys())[0]]

    citations = [
        Citation(
            source_id=source_id,
            filename=labeled[source_id].filename,
            page=labeled[source_id].page,
            quote=_excerpt(labeled[source_id].text),
            score=labeled[source_id].score,
        )
        for source_id in cited_ids
    ]
    confidence = guardrails.compute_confidence(
        [labeled[source_id].score for source_id in cited_ids], s.min_similarity
    )
    return QueryResponse(
        answer=llm_answer.answer.strip(),
        answerable=True,
        confidence=confidence,
        citations=citations,
        latency_ms=_elapsed_ms(started),
    )
