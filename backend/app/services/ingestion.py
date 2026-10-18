import logging
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import UploadResponse
from app.services.chunking import chunk_document
from app.services.document_loader import (
    DocumentLoadError,
    compute_doc_id,
    load_document,
)
from app.services.embeddings import get_embedder
from app.services.vector_store import get_vector_store

log = logging.getLogger(__name__)


def ingest_document(path: Path, original_filename: str) -> UploadResponse:
    s = get_settings()
    store = get_vector_store()

    # Cheap duplicate check first, so a re-upload never repeats OCR or embedding calls.
    doc_id = compute_doc_id(path)
    if store.doc_exists(doc_id):
        chunks = next((d['chunks'] for d in store.list_documents() if d['doc_id'] == doc_id), 0)
        log.info('Skipped %s: already indexed', original_filename)
        return UploadResponse(
            doc_id=doc_id, filename=original_filename, pages=0,
            skipped_empty_pages=0, chunks=chunks, already_indexed=True,
        )

    document = load_document(path, original_filename, s.max_pdf_pages)
    chunks = chunk_document(document, s.chunk_size, s.chunk_overlap)
    if not chunks:
        raise DocumentLoadError('The document contained no usable text after cleaning.')

    embeddings = get_embedder().embed_documents([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    log.info('Indexed %s: %d pages, %d chunks', original_filename, document.total_pages, len(chunks))

    return UploadResponse(
        doc_id=document.doc_id,
        filename=document.filename,
        pages=document.total_pages,
        skipped_empty_pages=document.empty_pages,
        chunks=len(chunks),
        already_indexed=False,
    )


# Backward compatibility: routes.py and older code import ingest_pdf.
ingest_pdf = ingest_document
