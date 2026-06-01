"""
Se leen todos los archivos de /docs (txt, md, pdf, json),
limpia el texto, lo divide en chunks y los indexa en ChromaDB.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

# Configuración

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Rutas
DOCS_DIR = Path(os.getenv("DOCS_DIR", "./docs"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "./vectors"))

# Chunks
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))  # caracteres por chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # solapamiento

# OpenAI para embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME = "docs"


# Funciones de lectura por tipo de archivo

def read_txt(path: Path) -> str:
    """Lee archivos .txt y .md con fallback de encoding."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    logger.warning(f"No se pudo decodificar {path}, se omite.")
    return ""


def read_json(path: Path) -> str:
    """
    Lee archivos .json y convierte su contenido a texto plano.
    Soporta: string, lista de strings/dicts, dict con campos de texto.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON inválido en {path}: {e}")
        return ""

    if isinstance(data, str):
        return data

    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Concatena todos los valores de texto del dict
                parts.append(" ".join(str(v) for v in item.values()))
        return "\n\n".join(parts)

    if isinstance(data, dict):
        # Concatena clave: valor para cada campo
        parts = [f"{k}: {v}" for k, v in data.items() if isinstance(v, (str, int, float))]
        return "\n".join(parts)

    return str(data)


def read_pdf(path: Path) -> str:
    """Lee archivos .pdf extrayendo el texto de cada página."""
    if not PDF_SUPPORT:
        logger.warning(f"pypdf no está instalado, se omite {path}. Instalá con: pip install pypdf")
        return ""
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"Error leyendo PDF {path}: {e}")
        return ""


def read_file(path: Path) -> Optional[str]:
    """Dispatcher: elige la función de lectura según la extensión."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return read_txt(path)
    elif ext == ".json":
        return read_json(path)
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        logger.info(f"Extensión no soportada, se omite: {path.name}")
        return None


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normaliza el texto eliminando ruido:
    - líneas vacías múltiples
    - espacios extra
    - caracteres de control (excepto saltos de línea y tabulaciones)
    """
    if not text:
        return ""

    # Eliminar caracteres de control raros (mantiene \n y \t)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", text)

    # Colapsar espacios en blanco dentro de una línea
    text = re.sub(r"[ \t]+", " ", text)

    # Colapsar más de 2 saltos de línea consecutivos
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Eliminar líneas que solo tengan espacios
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide el texto en fragmentos de tamaño aproximado `chunk_size` caracteres,
    con solapamiento `overlap` para no perder contexto entre chunks.

    Intenta cortar en saltos de párrafo primero, luego en oraciones, y
    como último recurso en el límite de caracteres exacto.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            # Último fragmento — tomar lo que queda
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Intentar cortar en el último \n\n antes de `end`
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start:
            # Fallback: último \n
            cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            # Fallback: último espacio
            cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            # Sin opción — corte duro
            cut = end

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        # Avanzar con solapamiento para no perder contexto
        start = max(cut - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# Indexación en ChromaDB
# ---------------------------------------------------------------------------

def get_collection(vectorstore_dir: Path, collection_name: str):
    """
    Crea (o abre) la colección ChromaDB con embeddings de OpenAI.
    Si no hay API key, usa el modelo local por defecto de ChromaDB.
    """
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(vectorstore_dir))

    if OPENAI_API_KEY:
        logger.info("Usando embeddings de OpenAI (text-embedding-3-small)")
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )
    else:
        logger.warning(
            "OPENAI_API_KEY no encontrada. "
            "Usando embedding local (all-MiniLM-L6-v2). "
            "Asegurate de tener sentence-transformers instalado."
        )
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_documents(docs_dir: Path, vectorstore_dir: Path) -> int:
    """
    Pipeline completo: lee → limpia → chunkea → indexa.
    Devuelve la cantidad total de chunks indexados.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"La carpeta de documentos no existe: {docs_dir}")

    collection = get_collection(vectorstore_dir, COLLECTION_NAME)

    # IDs ya indexados (para no duplicar en re-ejecuciones)
    existing = set(collection.get(include=[])["ids"])
    logger.info(f"Chunks ya existentes en la colección: {len(existing)}")

    total_chunks = 0
    files = [f for f in docs_dir.iterdir() if f.is_file()]

    if not files:
        logger.warning(f"No se encontraron archivos en {docs_dir}")
        return 0

    for file_path in files:
        logger.info(f"Procesando: {file_path.name}")

        raw_text = read_file(file_path)
        if not raw_text:
            logger.info(f"  → Sin contenido extraíble, se omite.")
            continue

        clean = clean_text(raw_text)
        if not clean:
            logger.info(f"  → Sin texto después de limpiar, se omite.")
            continue

        chunks = split_into_chunks(clean)
        logger.info(f"  → {len(chunks)} chunks generados")

        # Filtrar chunks ya indexados
        new_documents = []
        new_ids = []
        new_metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path.stem}_chunk_{i}"
            if chunk_id in existing:
                continue
            new_documents.append(chunk)
            new_ids.append(chunk_id)
            new_metadatas.append({
                "source": file_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        if not new_documents:
            logger.info(f"  → Todos los chunks ya estaban indexados.")
            continue

        # Indexar en lotes de 100 (límite recomendado de ChromaDB)
        batch_size = 100
        for batch_start in range(0, len(new_documents), batch_size):
            collection.add(
                documents=new_documents[batch_start:batch_start + batch_size],
                ids=new_ids[batch_start:batch_start + batch_size],
                metadatas=new_metadatas[batch_start:batch_start + batch_size],
            )

        total_chunks += len(new_documents)
        logger.info(f"  → {len(new_documents)} chunks nuevos indexados")

    logger.info(f"Indexación completa. Total de chunks nuevos: {total_chunks}")
    return total_chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Carpeta de documentos: {DOCS_DIR.resolve()}")
    logger.info(f"Vectorstore: {VECTORSTORE_DIR.resolve()}")
    logger.info(f"Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    try:
        total = index_documents(DOCS_DIR, VECTORSTORE_DIR)
        print(f"\n✅ Listo. {total} chunks nuevos indexados en '{VECTORSTORE_DIR}'.")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        logger.exception("Error inesperado durante la indexación")
        print(f"\n❌ Error inesperado: {e}")
