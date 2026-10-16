import logging
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import UploadResponse
from app.services.chunking import chunk_document
from app.services.document_loader import DocumentLoadError, load_pdf
from app.services.embeddings import get_embedder
from app.services.vector_store import get_vector_store

log = logging.getLogger(__name__)


def ingest_pdf(path: Path, original_filename: str) -> UploadResponse:
    s = get_settings()
    document = load_pdf(path, original_filename, s.max_pdf_pages)
    chunks = chunk_document(document, s.chunk_size, s.chunk_overlap)
    if not chunks:
        raise DocumentLoadError('The PDF contained no usable text after cleaning.')

    store = get_vector_store()
    already_indexed = store.doc_exists(document.doc_id)
    if not already_indexed:
        embeddings = get_embedder().embed_documents([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)
        log.info('Indexed %s: %d pages, %d chunks', original_filename, document.total_pages, len(chunks))
    else:
        log.info('Skipped %s: already indexed', original_filename)

    return UploadResponse(
        doc_id=document.doc_id,
        filename=document.filename,
        pages=document.total_pages,
        skipped_empty_pages=document.empty_pages,
        chunks=len(chunks),
        already_indexed=already_indexed,
    )
