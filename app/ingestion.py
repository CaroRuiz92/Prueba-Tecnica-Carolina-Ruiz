"""
Se leen todos los archivos de /docs (txt, md, pdf, json),
limpia el texto, lo divide en chunks y los indexa en ChromaDB.
"""

# Librerias necesarias
import os
import re
import json
import hashlib
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

# Backend de embeddings: "local" (gratis) u "openai".
# Para la Opción B usamos "local": no consume cuota de OpenAI.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME = "docs"


def make_embedding_function():
    """
    Devuelve la función de embeddings según EMBEDDING_BACKEND
    """
    if EMBEDDING_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("EMBEDDING_BACKEND=openai pero no hay OPENAI_API_KEY.")
        logger.info("Embeddings: OpenAI (text-embedding-3-small)")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )
    # Local (gratis): all-MiniLM-L6-v2 vía ONNX, no requiere torch ni API key.
    logger.info("Embeddings: locales y gratuitos (all-MiniLM-L6-v2, ONNX)")
    return embedding_functions.DefaultEmbeddingFunction()



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


def _flatten_json(obj, lines: list) -> None:
    """
    Aplana recursivamente cualquier estructura JSON a líneas de texto.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}:")
                _flatten_json(v, lines)
            else:
                lines.append(f"{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            _flatten_json(item, lines)
    else:
        lines.append(str(obj))


def read_json(path: Path) -> str:
    """Lee archivos .json y convierte TODO su contenido (incl. anidado) a texto plano."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON inválido en {path}: {e}")
        return ""

    lines: list[str] = []
    _flatten_json(data, lines)
    return "\n".join(lines)


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

    # Se eliminan SOLO caracteres de control (mantiene \t \n \r y TODO el
    # Unicode imprimible: viñetas, comillas tipográficas, rayas, etc.).
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)

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
    ef = make_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def chunk_id_for(file_path: Path, index: int, chunk: str) -> str:
    """
    ID estable por contenido: extensión + índice + hash del contenido.
    Evita colisiones entre archivos con el mismo nombre y permite reindexar
    al editar un documento.
    """
    h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:12]
    ext = file_path.suffix.lower().lstrip(".") or "noext"
    return f"{file_path.stem}__{ext}__{index:04d}__{h}"


def index_documents(docs_dir: Path, vectorstore_dir: Path) -> int:
    """Pipeline completo: lee -> limpia -> chunkea -> indexa de forma idempotente."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"La carpeta de documentos no existe: {docs_dir}")

    collection = get_collection(vectorstore_dir, COLLECTION_NAME)

    total_new = 0
    files = [f for f in docs_dir.iterdir() if f.is_file()]

    if not files:
        logger.warning(f"No se encontraron archivos en {docs_dir}")
        return 0

    for file_path in files:
        logger.info(f"Procesando: {file_path.name}")

        raw_text = read_file(file_path)
        if not raw_text:
            logger.info("  -> Sin contenido extraíble, se omite.")
            continue

        clean = clean_text(raw_text)
        if not clean:
            logger.info("  -> Sin texto después de limpiar, se omite.")
            continue

        chunks = split_into_chunks(clean)
        logger.info(f"  -> {len(chunks)} chunks generados")

        # Construir el conjunto de chunks "deseados" para este archivo
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        seen: set[str] = set()

        for i, chunk in enumerate(chunks):
            cid = chunk_id_for(file_path, i, chunk)
            if cid in seen:  # chunk idéntico repetido en el mismo archivo
                continue
            seen.add(cid)
            ids.append(cid)
            documents.append(chunk)
            metadatas.append({
                "source": file_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        # Qué hay ya indexado para ESTE archivo
        try:
            existing_for_source = collection.get(where={"source": file_path.name})
            existing_ids = set(existing_for_source.get("ids", []))
        except Exception as e:
            logger.warning(f"  -> No se pudieron leer chunks existentes de {file_path.name}: {e}")
            existing_ids = set()

        # Limpieza de huérfanos: chunks viejos del archivo que ya no existen
        orphans = existing_ids - set(ids)
        if orphans:
            collection.delete(ids=list(orphans))
            logger.info(f"  -> {len(orphans)} chunks obsoletos eliminados")

        # Agregar solo los chunks nuevos
        new_documents, new_ids, new_metadatas = [], [], []
        for cid, doc, meta in zip(ids, documents, metadatas):
            if cid in existing_ids:
                continue
            new_ids.append(cid)
            new_documents.append(doc)
            new_metadatas.append(meta)

        if not new_documents:
            logger.info("  -> Sin cambios, todos los chunks ya estaban indexados.")
            continue

        # Indexar en lotes controlados
        batch_size = 100
        for batch_start in range(0, len(new_documents), batch_size):
            collection.add(
                documents=new_documents[batch_start:batch_start + batch_size],
                ids=new_ids[batch_start:batch_start + batch_size],
                metadatas=new_metadatas[batch_start:batch_start + batch_size],
            )

        total_new += len(new_documents)
        logger.info(f"  -> {len(new_documents)} chunks nuevos indexados")

    logger.info(f"Indexación completa. Total de chunks nuevos: {total_new}")
    return total_new


if __name__ == "__main__":
    logger.info(f"Carpeta de documentos detectada: {DOCS_DIR.resolve()}")
    logger.info(f"Vectorstore detectado: {VECTORSTORE_DIR.resolve()}")
    logger.info(f"Backend de embeddings: {EMBEDDING_BACKEND}")
    logger.info(f"Configuración - Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    try:
        total = index_documents(DOCS_DIR, VECTORSTORE_DIR)
        print(f"\n✅ {total} chunks nuevos procesados en '{VECTORSTORE_DIR.name}'.")
    except FileNotFoundError as e:
        print(f"\n❌ Error de rutas: {e}")
    except Exception as e:
        logger.exception("Error inesperado durante la indexación")
        print(f"\n❌ Error inesperado: {e}")
