import pytest
from app.services.chunking import chunk_document
from app.services.document_loader import LoadedDocument, PageText

def make_doc(pages):
    return LoadedDocument(
        doc_id='abc123',
        filename='sample.pdf',
        pages=[PageText(page=n, text=t) for n, t in pages],
        total_pages=len(pages),
        empty_pages=0,
    )

def test_chunks_keep_their_page_number():
    doc = make_doc([(1, 'Alpha clause. ' * 100), (2, 'Beta clause. ' * 100)])
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        expected = 'Alpha' if c.page == 1 else 'Beta'
        assert expected in c.text

def test_chunks_never_exceed_chunk_size():
    doc = make_doc([(1, 'word ' * 1000)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert chunks
    assert all(len(c.text) <= 200 for c in chunks)

def test_chunk_ids_are_unique():
    doc = make_doc([(1, 'sentence number one. ' * 80), (2, 'sentence number two. ' * 80)])
    chunks = chunk_document(doc, 300, 50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

def test_tiny_fragments_are_dropped():
    doc = make_doc([(1, '3')])
    assert chunk_document(doc, 300, 50) == []

def test_overlap_must_be_smaller_than_size():
    doc = make_doc([(1, 'hello world ' * 10)])
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)
