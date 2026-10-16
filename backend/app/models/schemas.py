from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    filename: str
    page: int
    quote: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    answerable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]
    latency_ms: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    skipped_empty_pages: int
    chunks: int
    already_indexed: bool


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    embedding_provider: str
    llm_configured: bool
