import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import chainlit as cl
from utils import chunk_text


# Load .env located next to this script (langchain/.env)
here = Path(__file__).parent
load_dotenv(dotenv_path=here / '.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# Force the model to gemini-2.5-flash unconditionally
MODEL_NAME = 'gemini-2.5-flash'
import logging
logging.getLogger(__name__).info(f"Forzando modelo Gemini a: {MODEL_NAME}")
CHROMA_DIR = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
TOP_K = int(os.getenv('TOP_K', '5'))

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # don't call configure with None; log a warning and we'll return friendly errors when calling Gemini
    import logging
    logging.warning('GOOGLE_API_KEY not set in environment. Gemini calls will fail until it is configured.')

client = chromadb.Client(settings=Settings(is_persistent=True, persist_directory=CHROMA_DIR))
try:
    collection = client.get_collection('documents')
except Exception:
    collection = None

# Load embedding model once
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Prefer CPU unless CUDA is available; SentenceTransformer will pick device
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def retrieve(query: str, top_k: int = TOP_K):
    if collection is None:
        return []
    # ensure query is a raw string
    if not isinstance(query, str):
        query = str(query)
    model = get_embedding_model()
    q_emb = model.encode([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas'])
    docs = []
    for docs_list in res.get('documents', []):
        docs.extend(docs_list)
    return docs


def build_prompt(contexts, question):
    header = "Eres un asistente que responde preguntas bas\u00e1ndose en la siguiente informaci\u00f3n extra\u00edda de documentos:\n\n"
    ctx_text = "\n\n---\n\n".join(contexts)
    prompt = f"{header}{ctx_text}\n\nPregunta: {question}\n\nPor favor responde de forma clara y cita la fuente (ruta de archivo) cuando sea posible."
    return prompt


def call_gemini(prompt: str, model_name: str = None) -> str:
    use_model = model_name if model_name else MODEL_NAME
    model_ref = f'models/{use_model}' if not use_model.startswith('models/') else use_model
    try:
        model = genai.GenerativeModel(model_ref)
        resp = model.generate_content(prompt)
        if hasattr(resp, 'text') and resp.text:
            return resp.text
        if hasattr(resp, 'candidates'):
            return "\n".join([getattr(c, 'text', str(c)) for c in resp.candidates])
        if isinstance(resp, dict) and 'candidates' in resp:
            return "\n".join([c.get('text', '') for c in resp.get('candidates', [])])
        return str(resp)
    except Exception as e:
        return f'Error calling Gemini: {e}'


@cl.on_message
async def main(message):
    # Chainlit may pass a Message object; extract text safely
    user_text = None
    try:
        # message can be chainlit Message or raw str
        if hasattr(message, 'content'):
            user_text = message.content
        else:
            user_text = str(message)
    except Exception:
        user_text = str(message)

    # This handler is invoked for each user message in Chainlit
    await cl.Message(content="Recuperando documentos...").send()
    docs = retrieve(user_text)
    if not docs:
        await cl.Message(content="No hay documentos indexados. Ejecuta el indexador primero.").send()
        return

    prompt = build_prompt(docs, user_text)
    await cl.Message(content="Consultando Gemini...").send()
    answer = call_gemini(prompt)

    # Prepare an Action button that will send the fragments when clicked
    from chainlit import Action
    # Build a compact payload with fragments (truncate each to a reasonable length)
    safe_fragments = [ (c[:2000] + '...') if len(c) > 2000 else c for c in docs ]
    action = Action(name="show_fragments", label="Mostrar fragmentos", payload={"fragments": safe_fragments})

    # Send the assistant message with the action button attached. Fragments will be sent only after user clicks the button.
    sent = await cl.Message(content=answer, actions=[action]).send()

    # Note: we do NOT send the fragments here to avoid auto-scrolling. They will be sent in the action callback below.


@cl.action_callback('show_fragments')
async def handle_action(action: "Action"):
    """Callback invoked when an action button is clicked in the UI."""
    try:
        if action.name == 'show_fragments':
            fragments = action.payload.get('fragments', [])
            await cl.Message(content="Contexto usado (fragmentos):").send()
            for i, f in enumerate(fragments):
                # send each fragment as a separate message; UI will keep viewport at the response (user can scroll)
                await cl.Message(content=f'Fragmento {i+1}:\n{f}').send()
    except Exception as e:
        await cl.Message(content=f'Error al enviar fragmentos: {e}').send()
