#!/usr/bin/env python3
import os
import argparse
import logging
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from utils import list_files, read_file, chunk_text


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '2')


def create_client(persist_dir: str):
    return chromadb.Client(settings=Settings(is_persistent=True, persist_directory=persist_dir))


def index_files(
    docs_path: str,
    persist_dir: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    limit: int | None,
):
    client = create_client(persist_dir)
    try:
        collection = client.get_collection('documents')
    except Exception:
        collection = client.create_collection('documents')

    logging.info(f'Usando modelo de embeddings: {embedding_model_name} (CPU)')
    model = SentenceTransformer(embedding_model_name, device='cpu')

    files = list_files(docs_path)
    logging.info(f'{len(files)} archivos encontrados en {docs_path} (limit={limit})')
    if limit:
        files = files[:limit]

    # Obtener rutas ya indexadas y sus hashes (si existen) para evitar re-indexar
    indexed_sources_hash: dict[str, str] = {}
    try:
        existing = collection.get(include=['metadatas'])
        for mlist in existing.get('metadatas', []):
            for m in mlist:
                if isinstance(m, dict) and 'source' in m:
                    src = os.path.abspath(m['source'])
                    src_hash = None
                    if 'source_hash' in m and m['source_hash']:
                        src_hash = m['source_hash']
                    if src and src_hash and src not in indexed_sources_hash:
                        indexed_sources_hash[src] = src_hash
                    elif src and src not in indexed_sources_hash:
                        indexed_sources_hash[src] = None
    except Exception:
        indexed_sources_hash = {}

    total_indexed = 0
    for path in files:
        abs_path = os.path.abspath(path)

        logging.info(f'Procesando: {abs_path}')
        try:
            text = read_file(abs_path)
            if not text or not text.strip():
                logging.warning(f'Archivo vacio o ilegible: {abs_path}, se omite')
                continue

            # calcular hash del contenido para detectar cambios y evitar re-indexar
            path_hash_bytes = hashlib.sha1(text.encode('utf-8')).hexdigest()
            existing_hash = indexed_sources_hash.get(abs_path)
            if existing_hash and existing_hash == path_hash_bytes:
                logging.info(f'Se omite (ya indexado y sin cambios): {abs_path}')
                continue

            chunks = chunk_text(text, chunk_size, chunk_overlap)
            if not chunks:
                logging.warning(f'No se generaron chunks para {abs_path}, se omite')
                continue

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                short_path_hash = hashlib.sha1((abs_path + path_hash_bytes).encode('utf-8')).hexdigest()[:12]
                ids = [f"{short_path_hash}_{i + j}" for j in range(len(batch_chunks))]
                metadatas = [{"source": abs_path, "chunk_index": i + j, "source_hash": path_hash_bytes} for j in range(len(batch_chunks))]
                embeddings = model.encode(batch_chunks, show_progress_bar=False)
                collection.add(ids=ids, documents=batch_chunks, metadatas=metadatas, embeddings=embeddings)

            try:
                client.persist()
            except Exception:
                pass

            total_indexed += len(chunks)
            logging.info(f'Indexados {len(chunks)} chunks desde {abs_path}')

        except Exception as e:
            logging.exception(f'Error procesando {abs_path}: {e}')

    logging.info(f'\u2705 Indexado completado. Total de nuevos chunks: {total_indexed}')


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Indexador de documentos en ChromaDB (LangChain)')
    parser.add_argument('--limit', type=int, default=None, help='Procesar solo N archivos (útil para pruebas)')
    parser.add_argument('--batch-size', type=int, default=64, help='Tamaño de batch para embeddings')
    args = parser.parse_args()

    docs_path = os.getenv('DOCUMENTS_PATH', '../doc')
    persist_dir = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
    embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    index_files(docs_path, persist_dir, embedding_model_name, chunk_size, chunk_overlap, args.batch_size, args.limit)


if __name__ == '__main__':
    main()
