import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

# Secrets -> environment, BEFORE the backend reads its settings.
try:
    for _k in st.secrets:
        if isinstance(st.secrets[_k], (str, int, float)):
            os.environ.setdefault(str(_k), str(st.secrets[_k]))
except Exception:
    pass
os.environ.setdefault('EMBEDDING_PROVIDER', 'gemini')
os.environ.setdefault('DATA_DIR', str(Path(tempfile.gettempdir()) / 'docuquery'))

from google.genai import errors as genai_errors  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.document_loader import DocumentLoadError  # noqa: E402
from app.services.ingestion import ingest_pdf  # noqa: E402
from app.services.llm import LLMError  # noqa: E402
from app.services.rag_pipeline import answer_question  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402

st.set_page_config(page_title='DocuQuery AI', page_icon='📄', layout='wide')
settings = get_settings()
store = get_vector_store()


def render_assistant(message: dict) -> None:
    st.markdown(message['content'])
    meta = message.get('meta')
    if not meta:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric('Confidence', f"{meta['confidence']:.0%}")
    c2.metric('Sources', len(meta['citations']))
    c3.metric('Latency', f"{meta['latency_ms']} ms")
    st.progress(min(max(meta['confidence'], 0.0), 1.0))
    if not meta['answerable']:
        st.caption('No supporting passage was found, so no sources are shown.')
    for cite in meta['citations']:
        label = f"[{cite['source_id']}] {cite['filename']} · page {cite['page']} · match {cite['score']:.2f}"
        with st.expander(label):
            st.markdown('> ' + cite['quote'])


if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header('📄 DocuQuery AI')
    if not settings.gemini_api_key:
        st.error('GEMINI_API_KEY is missing in Secrets.')
        st.stop()
    st.success(f'Ready · {store.count()} chunks indexed')

    st.subheader('Upload')
    uploaded = st.file_uploader('Choose a PDF', type=['pdf'])
    if uploaded and st.button('Index document', type='primary'):
        data = uploaded.getvalue()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            st.error(f'File is larger than {settings.max_upload_mb} MB.')
        elif data[:5] != b'%PDF-':
            st.error('This file is not a valid PDF.')
        else:
            tmp = Path(tempfile.gettempdir()) / f'{os.urandom(8).hex()}.pdf'
            tmp.write_bytes(data)
            try:
                with st.spinner('Reading, chunking and embedding...'):
                    result = ingest_pdf(tmp, uploaded.name)
                if result.already_indexed:
                    st.info('This document was already indexed.')
                else:
                    st.success(f'Indexed {result.pages} pages into {result.chunks} chunks.')
            except DocumentLoadError as exc:
                st.error(str(exc))
            except genai_errors.APIError as exc:
                st.error(f'Embedding service failed: {exc}')
            finally:
                tmp.unlink(missing_ok=True)

    st.subheader('Documents')
    docs = store.list_documents()
    options = {'All documents': None}
    for d in docs:
        options[f"{d['filename']} ({d['doc_id'][:6]})"] = d['doc_id']
    scope_doc_id = options[st.selectbox('Ask about', list(options))]
    for d in docs:
        a, b = st.columns([4, 1])
        a.caption(f"{d['filename']} · {d['chunks']} chunks")
        if b.button('🗑', key='del-' + d['doc_id']):
            store.delete_document(d['doc_id'])
            st.rerun()
    if st.button('Clear chat'):
        st.session_state.messages = []
        st.rerun()

st.title('Chat with your documents')
st.caption('Answers come only from your uploaded PDFs, with page citations.')

for m in st.session_state.messages:
    with st.chat_message(m['role']):
        render_assistant(m) if m['role'] == 'assistant' else st.markdown(m['content'])

if not docs:
    st.info('Upload a PDF in the sidebar to begin.')

question = st.chat_input('Ask a question about your documents', disabled=not docs)
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Searching your documents...'):
            try:
                result = answer_question(question, scope_doc_id).model_dump()
                reply = {'role': 'assistant', 'content': result['answer'], 'meta': result}
            except (LLMError, genai_errors.APIError) as exc:
                reply = {'role': 'assistant', 'content': f'⚠️ Model error: {exc}', 'meta': None}
        render_assistant(reply)
    st.session_state.messages.append(reply)
