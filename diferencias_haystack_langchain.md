# Diferencias entre Haystack y LangChain

Este documento compara Haystack y LangChain en términos de arquitectura, funcionamiento, experiencia de desarrollo y rendimiento, con ejemplos de uso, ventajas, limitaciones y recomendaciones.

## Resumen ejecutivo

- Haystack: framework centrado en construcción de sistemas de recuperación-augmented generation (RAG) y búsqueda de documentos con integración para indexadores (FAISS, Milvus, Chroma...), encoders, pipelines y orquestación de documentos. Muy sólido para soluciones de producción enfocadas en ingestion, preprocesamiento y búsqueda semántica.
- LangChain: librería centrada en orquestación de LLMs y cadenas (chains), con utilidades para prompts, memoria, agentes, y conectores a vectores. Se orienta a construir flujos aplicacionales con LLMs (chatbots, agentes, herramientas), menos prescriptiva sobre ingestion masiva de docs.

## Contrato (inputs/outputs, criterios de éxito)

- Inputs: colección de documentos (PDF, HTML, texto), embeddings, modelo LLM, consultas de usuario.
- Outputs: respuestas enriquecidas por contexto (RAG), metadatos de búsqueda, latencias por consulta.
- Criterios de éxito: precisión de respuestas, latencia aceptable para usuario, coste computacional.

## Arquitectura y componentes principales

### Haystack

- Pipeline de ingestión: extractores de texto, limpiadores, segmentadores (splitting), metadata tagging.
- Indexadores: soporta FAISS, Milvus, Elasticsearch, Chroma, etc.; maneja actualizaciones y búsqueda por similitud.
- Modelos de embeddings y motores de recuperación (retrievers): retrievers como DensePassageRetriever, BM25 (sparse), DenseRetriever.
- Generación: integración con LLMs para respuesta (reader/generator) y pipelines de RAG.
- Orquestación: Pipelines configurables (ExtractiveQAPipeline, GenerativeQAPipeline).
- Ventaja: muy orientado a workflows de ingestión y búsqueda, buen manejo de documentos en producción.

### LangChain

- Primitivas: Chains (encadenamiento de pasos), Agents (agentes que llaman herramientas), Tools (conectores), Memory (estado conversacional).
- Integraciones: wrappers para LLMs, prompts, embeddings, vectores y también retrievers.
- Flexibilidad: alto nivel de abstracción para construir cualquier flujo que combine prompts, LLMs y vectores.
- Ventaja: excelente para prototipar agentes conversacionales y pipelines LLM-centric.

## Flujo de datos y funcionamiento

- Haystack: ingestion → splitting → embeddings → indexación → retrieval → reranking → LLM generation. Pipeline optimizada para documentos.
- LangChain: ingestion (opcional) → embeddings → vectorstore ↔ retriever → chain/agent → LLM. Es más flexible en el paso de orquestación (multiple chains, agents).

## Desarrollo y experiencia del desarrollador

- Haystack:
  - Orientado a equipos que implementan RAG en producción.
  - Proporciona utilidades listas para usar (splitters, converters, pipelines), menos necesidad de ensamblarlo todo manualmente.
  - Documentación y ejemplos centrados en ingestion y QA.
  - Requiere entender indexadores y pipeline tuning para escalar.

- LangChain:
  - API modular y minimalista, muy popular en prototipado rápido.
  - Gran ecosistema de integrations (OpenAI, HuggingFace, Pinecone, Chroma, etc.) y ejemplos de agentes.
  - Ideal para lógica de aplicación (memory, agents) y manipulación de prompts.
  - Puede requerir trabajo extra para ingestion masiva y control fino del pipeline de búsqueda.

## Rendimiento y escalabilidad

- Latencia:
  - Haystack: optimizado para retrieval y re-ranking; con un indexador eficiente (FAISS, Milvus) puede ofrecer latencias bajas para retrieval. Sin embargo, ingestión y reindexado pueden ser costosos.
  - LangChain: la latencia depende del vectorstore elegido y de cuánta lógica de chain se ejecute; no impone optimizaciones de retrieval por defecto.

- Throughput y escalado:
  - Haystack: diseñado pensando en producción — manejo de índices, shard/repl, y carga de documentos a gran escala; mejor para ingestion de grandes corpora.
  - LangChain: más orientado a orquestación de llamadas a LLMs; la escalabilidad de la búsqueda dependerá del vector database.

- Coste computacional:
  - Ambos dependen de los modelos usados; Haystack añade coste de mantenimiento de indexadores y procesos de ingestión.

## Casos de uso típicos

- Haystack:
  - Sistemas de QA sobre documentación extensa.
  - Búsqueda semántica empresarial y buscadores internos.
  - Pipelines de extracción y extracción+generación en producción.

- LangChain:
  - Agentes que orquestan herramientas (ej.: llamar APIs, buscar en web, ejecutar código).
  - Chatbots con memoria y flujos conversacionales complejos.
  - Prototipos que mezclan prompt engineering y manipulación de contexto.

## Ventajas y limitaciones (resumen)

- Haystack:
  - + Fuerte en ingestion y pipelines de búsqueda.
  - + Buenas integraciones para indexadores y re-ranker.
  - - Menos centrado en agentes y memoria conversacional.
  - - Mayor curva si quieres flexibilidad fuera de RAG.

- LangChain:
  - + Gran flexibilidad para construir chains y agentes.
  - + Amplio ecosistema y documentación comunitaria.
  - - Menos batería de herramientas out-of-the-box para ingestion masiva.

## Recomendaciones

- Si necesitas construir un sistema RAG/QA robusto sobre grandes colecciones: Haystack.
- Si tu foco es orquestación con LLMs, agentes, o prototipado rápido: LangChain.
- Con combinación: usar LangChain para la orquestación de aplicación y Haystack (o un vector DB gestionado) para la indexación y retrieval.
## Explicación paso a paso de cada etapa

A continuación se explica en detalle qué significa cada paso que se menciona en los pipelines (ingestión → splitting → embeddings → indexación → retrieval → re-ranking → generación) y se ofrecen snippets de ejemplo para Haystack y LangChain.

1) Ingestión
- Qué es: cargar y convertir los documentos fuente (PDF, DOCX, HTML, TXT) a un formato texto estructurado (documentos con metadatos).
- Objetivo: extraer texto y metadatos (títulos, fecha, autor) listos para procesar.

Snippet (Python, lectura básica de archivos):

```python
# Ingesta básica: leer archivos y crear una lista de documentos
from pathlib import Path

def ingest_folder(folder_path):
  docs = []
  for p in Path(folder_path).rglob("*.txt"):
    text = p.read_text(encoding='utf-8')
    docs.append({"content": text, "meta": {"source": str(p)}})
  return docs

docs = ingest_folder("./doc/ELECTRIC MARKET - LEARNING STUFF")
print(f"Documentos cargados: {len(docs)}")
```

2) Splitting (segmentación)
- Qué es: dividir documentos largos en fragmentos (chunks) manejables para embeddings y retrieval.
- Objetivo: controlar el tamaño del contexto y mejorar la granularidad de la búsqueda.

Snippet (simple splitter):

```python
def chunk_text(text, chunk_size=500, overlap=50):
  tokens = text.split()
  chunks = []
  i = 0
  while i < len(tokens):
    chunk = " ".join(tokens[i:i+chunk_size])
    chunks.append(chunk)
    i += chunk_size - overlap
  return chunks

chunks = chunk_text(docs[0]['content'])
print(len(chunks), "chunks created")
```

3) Embeddings
- Qué es: convertir cada chunk a un vector numérico (embedding) que capture semántica.
- Objetivo: poder comparar similitud entre query y fragmentos mediante distancia vectorial.

Nota: existen varios modelos de embeddings (sentence-transformers, OpenAI, Cohere, etc.).

4) Indexación (vector store)
- Qué es: almacenar los embedding vectors en una base/vector store (FAISS, Milvus, Chroma, Pinecone, Elastic Vector Search).
- Objetivo: consultas de similitud rápidas y escalables.

5) Retrieval
- Qué es: por cada consulta (query) se transforma la consulta a embedding y se busca los vectores más cercanos en el index.
- Objetivo: recuperar los mejores candidatos (contexto) para pasarlos al LLM.

6) Re-ranking
- Qué es: opcionalmente aplicar un re-ranker (modelo de cross-encoder o heurística) para ordenar mejor los candidatos.
- Objetivo: aumentar la precisión antes de la generación.

7) Generación (LLM)
- Qué es: usar un modelo de lenguaje para responder la consulta usando el contexto recuperado (RAG) o para extraer directamente la respuesta (extractive QA).
- Objetivo: producir la respuesta final para el usuario.

## Snippets de ejemplo

Las muestras de abajo muestran un flujo mínimo para Haystack y para LangChain. Son ejemplos educativos: adáptalos según la versión de la librería y tus credenciales (API keys).

### Ejemplo Haystack (flujo mínimo)

Notas previas:
- Instalar: `pip install farm-haystack[all]` o la versión moderna `pip install "haystack==<compatible>"` según tu entorno.
- Ajusta los nombres de clases si usas una versión diferente de Haystack.

```python
# Ejemplo simplificado con InMemoryDocumentStore + EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline

# 1) Document store
document_store = InMemoryDocumentStore()

# 2) Ingesta + split
raw_docs = [{"content": "Texto muy largo ...", "meta": {"source": "mi_doc"}}]
preprocessor = PreProcessor(split_by="word", split_length=300, split_overlap=50)
docs = preprocessor.process(raw_docs)

# 3) Escribir en store
document_store.write_documents(docs)

# 4) Embeddings (usa un modelo de sentence-transformers)
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
document_store.update_embeddings(retriever)

# 5) Pipeline de búsqueda
pipeline = DocumentSearchPipeline(retriever)
res = pipeline.run(query="¿Qué es RAG?", params={"Retriever": {"top_k": 5}})
print(res["documents"][0].content)
```

### Ejemplo LangChain (flujo mínimo)

Notas previas:
- Instalar: `pip install langchain chromadb openai` (si usas Chroma + OpenAI). Setea `OPENAI_API_KEY`.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1) Crear embeddings y vectorstore
emb = OpenAIEmbeddings()
texts = ["Documento split 1...", "Documento split 2..."]
chromadb = Chroma.from_texts(texts, embedding=emb)

# 2) Crear retriever
retriever = chromadb.as_retriever(search_kwargs={"k": 4})

# 3) LLM + chain de retrieval
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
resp = qa.run("Explica brevemente qué es RAG")
print(resp)
```

## Consejos prácticos y diferencias en el código

- Haystack suele dar primitives listas para ingestion y pipelines (PreProcessor, DocumentStore, Nodes). Es conveniente cuando tienes un corpus grande y necesitas control del indexado.
- LangChain te ofrece bloques de construcción para orquestar LLMs y retrieval, ideal para agentes y lógica de aplicación.
- En producción, mezcla ambos: usa Haystack o un vector DB robusto para el indexado, y LangChain para orquestar la lógica del LLM/agent.

