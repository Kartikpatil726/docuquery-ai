import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class DocumentLoadError(Exception):
    '''Raised when a PDF cannot be read or has no usable text.'''


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    doc_id: str
    filename: str
    pages: list[PageText]
    total_pages: int
    empty_pages: int


def compute_doc_id(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()[:16]


def clean_text(text: str) -> str:
    text = text.replace('\x00', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' ?\n ?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_pdf(path: Path, original_filename: str, max_pages: int) -> LoadedDocument:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise DocumentLoadError('This PDF is password-protected. Remove the password and upload it again.')
        total = len(reader.pages)
        if total == 0:
            raise DocumentLoadError('This PDF has no pages.')
        if total > max_pages:
            raise DocumentLoadError(f'This PDF has {total} pages; the limit is {max_pages}.')

        pages: list[PageText] = []
        empty = 0
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or '')
            if text:
                pages.append(PageText(page=number, text=text))
            else:
                empty += 1
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f'Could not read this PDF: {exc}') from exc

    if not pages:
        raise DocumentLoadError(
            'No text could be extracted. The PDF is probably a scanned image; run OCR on it first.'
        )

    return LoadedDocument(
        doc_id=compute_doc_id(path),
        filename=original_filename,
        pages=pages,
        total_pages=total,
        empty_pages=empty,
    )
