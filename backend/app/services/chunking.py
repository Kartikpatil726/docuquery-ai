from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.document_loader import LoadedDocument

MIN_CHUNK_CHARS = 20


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', ''],
        length_function=len,
    )


def chunk_document(doc: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be smaller than chunk_size')

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    index = 0
    for page in doc.pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f'{doc.doc_id}-p{page.page}-c{index}',
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page=page.page,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks
