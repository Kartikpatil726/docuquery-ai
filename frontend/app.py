import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title='DocuQuery AI', page_icon='📄', layout='wide')

def get_config(name: str, default: str = '') -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return os.getenv(name, default)

BACKEND_URL = get_config('BACKEND_URL', 'http://localhost:8000').rstrip('/')
API_KEY = get_config('API_KEY', '')
HEADERS = {'X-API-Key': API_KEY} if API_KEY else {}

def call_api(method: str, path: str, timeout: int = 300, **kwargs):
    try:
        response = requests.request(
            method, f'{BACKEND_URL}{path}', headers=HEADERS, timeout=timeout, **kwargs
        )
    except requests.RequestException as exc:
        return None, f'Cannot reach the backend at {BACKEND_URL} ({exc.__class__.__name__}).'
    if response.ok:
        return response.json(), None
    try:
        detail = response.json().get('detail', response.text)
    except ValueError:
        detail = response.text
    return None, f'{response.status_code}: {detail}'

def render_assistant(message: dict) -> None:
    st.markdown(message['content'])
    meta = message.get('meta')
    if not meta:
        return
    confidence = meta['confidence']
    sources = len(meta['citations'])
    latency = meta['latency_ms']
    col1, col2, col3 = st.columns(3)
    col1.metric('Confidence', f'{confidence:.0%}')
    col2.metric('Sources', sources)
    col3.metric('Latency', f'{latency} ms')
    st.progress(min(max(confidence, 0.0), 1.0))
    if not meta['answerable']:
        st.caption('No supporting passage was found, so no sources are shown.')
    for cite in meta['citations']:
        sid = cite['source_id']
        name = cite['filename']
        page = cite['page']
        score = cite['score']
        with st.expander(f'[{sid}] {name} · page {page} · match {score:.2f}'):
            st.markdown('> ' + cite['quote'])

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header('📄 DocuQuery AI')
    health, error = call_api('GET', '/health', timeout=90)
    if error:
        st.error(error)
        st.stop()
    indexed = health['indexed_chunks']
    st.success(f'Backend online · {indexed} chunks indexed')
    st.subheader('Upload')
    uploaded = st.file_uploader('Choose a PDF', type=['pdf'])
    if uploaded and st.button('Index document', type='primary'):
        with st.spinner('Reading, chunking and embedding...'):
            result, error = call_api(
                'POST',
                '/upload',
                timeout=600,
                files={'file': (uploaded.name, uploaded.getvalue(), 'application/pdf')},
            )
        if error:
            st.error(error)
        elif result['already_indexed']:
            st.info('This document was already indexed.')
        else:
            pages = result['pages']
            chunks = result['chunks']
            st.success(f'Indexed {pages} pages into {chunks} chunks.')
    st.subheader('Documents')
    docs, error = call_api('GET', '/documents')
    if error:
        st.error(error)
    docs = docs or []
    scope_options = {'All documents': None}
    for doc in docs:
        label = doc['filename'] + ' (' + doc['doc_id'][:6] + ')'
        scope_options[label] = doc['doc_id']
    scope_label = st.selectbox('Ask about', list(scope_options.keys()))
    scope_doc_id = scope_options[scope_label]
    for doc in docs:
        name = doc['filename']
        chunk_count = doc['chunks']
        col_name, col_btn = st.columns([4, 1])
        col_name.caption(f'{name} · {chunk_count} chunks')
        if col_btn.button('🗑', key='del-' + doc['doc_id']):
            _, delete_error = call_api('DELETE', '/documents/' + doc['doc_id'])
            if delete_error:
                st.error(delete_error)
            else:
                st.rerun()
    if st.button('Clear chat'):
        st.session_state.messages = []
        st.rerun()

st.title('Chat with your documents')
st.caption('Answers come only from your uploaded PDFs, with page citations.')
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_assistant(message)
        else:
            st.markdown(message['content'])

if not docs:
    st.info('Upload a PDF in the sidebar to begin.')

question = st.chat_input('Ask a question about your documents', disabled=not docs)
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Searching your documents...'):
            result, error = call_api(
                'POST',
                '/query',
                timeout=300,
                json={'question': question, 'doc_id': scope_doc_id},
            )
        if error:
            reply = {'role': 'assistant', 'content': '⚠️ ' + error, 'meta': None}
        else:
            reply = {'role': 'assistant', 'content': result['answer'], 'meta': result}
        render_assistant(reply)
    st.session_state.messages.append(reply)
