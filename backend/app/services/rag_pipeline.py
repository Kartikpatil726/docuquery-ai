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

SYSTEM_PROMPT = '''You are DocuQuery, a smart, friendly, and helpful AI assistant.

BEHAVIOR RULES:
1. Treat the <source> blocks as primary reference data from uploaded documents.
2. If the user asks a question that can be answered from the sources, give the answer clearly and cite the sources using tags like [S1], [S2] in your text, and list them in the "sources" field.
3. If the user's question is NOT found in the documents (or is general advice like "how to increase cgpa", greeting, casual conversation, coding tips, etc.):
   - Set "answerable": true (DO NOT refuse).
   - Set "sources": [] (empty list).
   - Give a warm, natural, and genuinely helpful AI answer using your general knowledge.
   - Mention briefly and casually that this specific advice is from general knowledge since it was not directly mentioned in the uploaded file.
4. Always maintain a friendly, conversational, and direct tone. Never sound like a robotic refusal.'''

def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

def _excerpt(text: str, limit: int = 300) -> str:
    flat = re.sub(r'\s+', ' ', text).strip()
    return flat if len(flat) <= limit else flat[:limit].rstrip() + '...'

def answer_question(question: str, doc_id: str | None = None) -> QueryResponse:
    started = time.perf_counter()
    s = get_settings()
    
    # 1. Search vector database
    query_vector = get_embedder().embed_query(question)
    hits = get_vector_store().search(query_vector, k=s.top_k, doc_id=doc_id)
    hits = guardrails.filter_by_score(hits, s.min_similarity)
    hits = guardrails.drop_suspicious(hits)

    # Context build karo agar matches milein
    context, labeled = guardrails.build_context(hits, s.max_context_chars) if hits else ("", {})
    
    user_prompt = f"SOURCES:\n{context}\n\nQUESTION: {question}\n\nReturn the JSON object now."
    llm_answer = get_llm().generate_answer(SYSTEM_PROMPT, user_prompt)

    # Citations map karo agar model ne cite kiya ho
    cited_ids = guardrails.validate_citations(llm_answer.sources, labeled) if labeled else []
    
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
    
    confidence = (
        guardrails.compute_confidence([labeled[sid].score for sid in cited_ids], s.min_similarity)
        if cited_ids else 0.0
    )

    return QueryResponse(
        answer=llm_answer.answer.strip(),
        answerable=True,
        confidence=confidence,
        citations=citations,
        latency_ms=_elapsed_ms(started),
    )
