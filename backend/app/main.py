import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.services.embeddings import get_embedder
from app.services.vector_store import get_vector_store

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.environment == 'production' and not settings.api_key:
        raise RuntimeError('API_KEY must be set when ENVIRONMENT=production')
    settings.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    get_embedder()
    get_vector_store()
    log.info('%s is ready (embeddings: %s)', settings.app_name, settings.embedding_provider)
    yield


app = FastAPI(title=settings.app_name, version='1.0.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['X-API-Key', 'Content-Type'],
)

app.include_router(router)
