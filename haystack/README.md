# RAG con Haystack-style + Gemini

Este directorio contiene un ejemplo mínimo para crear un sistema RAG usando:
- ChromaDB para persistencia de embeddings
- SentenceTransformers para embeddings
- Google Gemini (via google-generativeai) como LLM
- Streamlit como UI

Archivos importantes:
- `.env` - configuración y claves API
- `index_docs.py` - script para indexar los documentos en `doc/`
- `streamlit_app.py` - aplicación Streamlit para preguntar al sistema
- `utils.py` - utilidades para leer PDFs/txt y chunkear texto
- `requirements.txt` - dependencias Python

Instrucciones rápidas:

1) Crea e activa un virtualenv (opcional pero recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instala dependencias:

```bash
pip install -r haystack/requirements.txt
```

3) Revisa y actualiza `haystack/.env` con tu API key de Google Gemini.

4) Indexa los documentos (usa la ruta por defecto `../doc` que apunta al directorio `doc/` del repo):

```bash
python haystack/index_docs.py
```

5) Ejecuta la app Streamlit:

```bash
streamlit run haystack/streamlit_app.py --server.port 8501
```

Notas:
- Este es un ejemplo mínimo pensado para ser ampliado. Manejo de errores, control de costes y paginación de embeddings deben mejorarse para producción.
- Si hay muchos documentos, la generación de embeddings puede tardar y consumir memoria; se recomienda enviarlos en batches y/o usar una solución de embeddings remota.
