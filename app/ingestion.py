"""
Se leen todos los archivos de /docs (txt, md, pdf, json),
limpia el texto, lo divide en chunks y los indexa en ChromaDB.
"""

# Librerias necesarias
import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PdfReader = None
    PDF_SUPPORT = False


# Configuración
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Resolución dinámica de rutas relativas basadas en la ubicación del script
BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "docs")))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectors")))


# Chunks
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))  # caracteres por chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # solapamiento

# OpenAI para los embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME = "docs"



# Funciones de lectura por tipo de archivo

def read_txt(path: Path) -> str:
    """Lee archivos .txt y .md."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    logger.warning(f"No se pudo decodificar {path}, se omite.")
    return ""


def read_json(path: Path) -> str:
    """Lee archivos .json y convierte su contenido a texto plano."""
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
                parts.append(" ".join(str(v) for v in item.values()))
        return "\n\n".join(parts)

    if isinstance(data, dict):
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
    """Elige la función de lectura según la extensión."""
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


# Limpieza de texto

def clean_text(text: str) -> str:
    """Normalización del texto eliminando ruido y caracteres de control."""
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


# Implementación de Chunking

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide el texto en fragmentos de tamaño aproximado con solapamiento semanticamente controlado.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Intentar buscar cortes lógicos descendentes
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start:
            cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            cut = end

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        # Controlar el avance evitando bucles infinitos o sobre-solapamientos en fragmentos cortos
        if cut - overlap > start:
            start = cut - overlap
        else:
            start = cut

    return chunks


# Indexación en ChromaDB

def get_collection(vectorstore_dir: Path, collection_name: str):
    """Crea o abre la colección ChromaDB configurando la función de embeddings."""
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
            "OPENAI_API_KEY no encontrada. Usando embedding local (all-MiniLM-L6-v2)."
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
    """Pipeline completo: lee → limpia → chunkea → indexa sin duplicados."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"La carpeta de documentos no existe: {docs_dir}")

    collection = get_collection(vectorstore_dir, COLLECTION_NAME)

    # EVITAR LA TRAMPA DEL LÍMITE DE 100: Paginamos para obtener TODOS los IDs existentes
    existing = set()
    offset = 0
    batch_limit = 100
    while True:
        existing_batch = collection.get(include=[], limit=batch_limit, offset=offset)
        if not existing_batch["ids"]:
            break
        existing.update(existing_batch["ids"])
        offset += batch_limit

    logger.info(f"Chunks ya existentes en la base de datos vectorial: {len(existing)}")

    total_chunks = 0
    files = [f for f in docs_dir.iterdir() if f.is_file()]

    if not files:
        logger.warning(f"No se encontraron archivos en {docs_dir}")
        return 0

    for file_path in files:
        logger.info(f"Procesando: {file_path.name}")

        raw_text = read_file(file_path)
        if not raw_text:
            logger.info("  → Sin contenido extraíble, se omite.")
            continue

        clean = clean_text(raw_text)
        if not clean:
            logger.info("  → Sin texto después de limpiar, se omite.")
            continue

        chunks = split_into_chunks(clean)
        logger.info(f"  → {len(chunks)} chunks generados")

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
            logger.info("  → Todos los chunks ya estaban indexados.")
            continue

        # Indexar en lotes controlados
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


if __name__ == "__main__":
    logger.info(f"Carpeta de documentos detectada: {DOCS_DIR.resolve()}")
    logger.info(f"Vectorstore detectado: {VECTORSTORE_DIR.resolve()}")
    logger.info(f"Configuración - Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    try:
        total = index_documents(DOCS_DIR, VECTORSTORE_DIR)
        print(f"\n✅ {total} chunks nuevos procesados en '{VECTORSTORE_DIR.name}'.")
    except FileNotFoundError as e:
        print(f"\n❌ Error de rutas: {e}")
    except Exception as e:
        logger.exception("Error inesperado durante la indexación")
        print(f"\n❌ Error inesperado: {e}")