import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from google.genai import errors as genai_errors

from app.core.config import get_settings
from app.core.security import require_api_key
from app.models.schemas import (
    DocumentInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.document_loader import DocumentLoadError
from app.services.ingestion import ingest_pdf
from app.services.llm import LLMError
from app.services.rag_pipeline import answer_question
from app.services.vector_store import get_vector_store

log = logging.getLogger(__name__)
router = APIRouter()


@router.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status='ok',
        indexed_chunks=get_vector_store().count(),
        embedding_provider=s.embedding_provider,
        llm_configured=bool(s.gemini_api_key),
    )


@router.post('/upload', response_model=UploadResponse, dependencies=[Depends(require_api_key)])
def upload(file: UploadFile = File(...)) -> UploadResponse:
    s = get_settings()
    filename = Path(file.filename or 'upload.pdf').name
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=415, detail='Only .pdf files are accepted.')

    s.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    destination = s.raw_docs_dir / f'{uuid.uuid4().hex}.pdf'
    max_bytes = s.max_upload_mb * 1024 * 1024
    keep_file = False
    try:
        written = 0
        with destination.open('wb') as out:
            while block := file.file.read(1024 * 1024):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413, detail=f'File is larger than {s.max_upload_mb} MB.'
                    )
                out.write(block)

        with destination.open('rb') as check:
            if check.read(5) != b'%PDF-':
                raise HTTPException(status_code=400, detail='This file is not a valid PDF.')

        result = ingest_pdf(destination, filename)
        keep_file = not result.already_indexed
        return result
    except DocumentLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except genai_errors.APIError as exc:
        log.error('Embedding provider failed: %s', exc)
        raise HTTPException(status_code=502, detail='The embedding service failed. Try again.') from exc
    except RuntimeError as exc:
        log.exception('Server configuration problem during upload')
        raise HTTPException(status_code=503, detail='The service is not configured correctly.') from exc
    finally:
        if not keep_file:
            destination.unlink(missing_ok=True)


@router.post('/query', response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    try:
        return answer_question(request.question, request.doc_id)
    except LLMError as exc:
        log.error('LLM failure: %s', exc)
        raise HTTPException(
            status_code=502, detail='The language model could not produce an answer. Try again.'
        ) from exc
    except genai_errors.APIError as exc:
        log.error('Embedding provider failed: %s', exc)
        raise HTTPException(status_code=502, detail='The embedding service failed. Try again.') from exc
    except RuntimeError as exc:
        log.exception('Server configuration problem during query')
        raise HTTPException(status_code=503, detail='The service is not configured correctly.') from exc


@router.get(
    '/documents', response_model=list[DocumentInfo], dependencies=[Depends(require_api_key)]
)
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**doc) for doc in get_vector_store().list_documents()]


@router.delete('/documents/{doc_id}', dependencies=[Depends(require_api_key)])
def delete_document(doc_id: str) -> dict:
    deleted = get_vector_store().delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail='Document not found.')
    return {'deleted_chunks': deleted}
