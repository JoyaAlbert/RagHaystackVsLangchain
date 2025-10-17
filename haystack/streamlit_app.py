import os
import logging
from dotenv import load_dotenv
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils import chunk_text
import google.generativeai as genai


load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# Force Gemini model to 2.5 flash unconditionally (ignore .env MODEL_NAME)
MODEL_NAME = 'gemini-2.5-flash'
CHROMA_DIR = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
TOP_K = int(os.getenv('TOP_K', '5'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

genai.configure(api_key=GOOGLE_API_KEY)

client = chromadb.Client(settings=Settings(is_persistent=True, persist_directory=CHROMA_DIR))
try:
    collection = client.get_collection('documents')
except Exception:
    collection = None


def retrieve(query: str, top_k: int = TOP_K):
    if collection is None:
        return []
    # simple approach: embed query and use collection.query
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas'])
    docs = []
    for docs_list in res.get('documents', []):
        docs.extend(docs_list)
    return docs


def list_available_models() -> list:
    try:
        models = genai.list_models()
        names = []
        # support SDK variations: list of objects or dict
        if isinstance(models, dict) and 'models' in models:
            for m in models['models']:
                if isinstance(m, dict) and 'name' in m:
                    names.append(m['name'])
                elif hasattr(m, 'name'):
                    names.append(m.name)
        elif isinstance(models, (list, tuple)):
            for m in models:
                if isinstance(m, dict) and 'name' in m:
                    names.append(m['name'])
                elif hasattr(m, 'name'):
                    names.append(m.name)
        return names
    except Exception:
        return []


def call_gemini(prompt: str, model_name: str = None) -> str:
    # Use the GenerativeModel API from google.generativeai and handle model-not-found
    use_model = model_name if model_name else MODEL_NAME
    model_ref = f'models/{use_model}' if not use_model.startswith('models/') else use_model
    try:
        model = genai.GenerativeModel(model_ref)
        resp = model.generate_content(prompt)
        # SDKs differ; try common response shapes
        if hasattr(resp, 'text') and resp.text:
            return resp.text
        if hasattr(resp, 'candidates'):
            return "\n".join([getattr(c, 'text', str(c)) for c in resp.candidates])
        if isinstance(resp, dict) and 'candidates' in resp:
            return "\n".join([c.get('text', '') for c in resp.get('candidates', [])])
        return str(resp)
    except Exception as e:
        msg = str(e)
        if 'NotFound' in msg or 'not found' in msg.lower():
            avail = list_available_models()
            if avail:
                sample = "\n".join(avail[:20])
                return (f"Model '{use_model}' not found or not supported for this API.\n"
                        f"Available models (sample):\n{sample}\n"
                        "Please set MODEL_NAME in .env to one of the above or use a supported model.")
            else:
                return (f"Model '{use_model}' not found or not supported for this API."
                        " Also failed to list models (check GOOGLE_API_KEY and network).\n"
                        f"Original error: {e}")
        return f'Error calling Gemini: {e}'



def build_prompt(contexts, question):
    header = "Eres un asistente que responde preguntas basándose en la siguiente información extraída de documentos:\n\n"
    ctx_text = "\n\n---\n\n".join(contexts)
    prompt = f"{header}{ctx_text}\n\nPregunta: {question}\n\nPor favor responde de forma clara y cita la fuente (ruta de archivo) cuando sea posible."
    return prompt


def main():
    st.title('ENETECH - RAG')
    # Force the MODEL_NAME (no override)
    available_model = MODEL_NAME

    question = st.text_input('Escribe tu pregunta:')
    if st.button('Preguntar') and question:
        with st.spinner('Recuperando documentos...'):
            docs = retrieve(question)
        if not docs:
            st.warning('No hay documentos indexados. Ejecuta el indexador primero.')
            return
        # take texts and limit size
        contexts = docs
        prompt = build_prompt(contexts, question)
        with st.spinner('Consultando Gemini...'):
            try:
                answer = call_gemini(prompt, available_model)
            except Exception as e:
                st.error(f'Error al llamar a Gemini: {e}')
                # try to list models for the user
                try:
                    models = genai.list_models()
                    st.info('Modelos disponibles:')
                    for m in models:
                        st.write(f'- {m.name} (supports: {getattr(m, "capabilities", None)})')
                except Exception:
                    st.warning('No se pudieron listar los modelos (problema de autenticación o red).')
                return
        st.subheader('Respuesta')
        st.write(answer)
        st.subheader('Contexto usado (fragmentos)')
        for i, c in enumerate(contexts):
            st.write(f'Fragmento {i+1}:')
            st.write(c)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
