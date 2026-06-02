"""
API REST para el Asistente de Soporte Técnico.
Recibe consultas, busca en ChromaDB el contexto relevante y
usa un LLM para generar una respuesta.

OPCIÓN B (gratis):
- Embeddings locales (EMBEDDING_BACKEND=local), sin costo de API.
- Generación de respuestas vía un endpoint compatible con OpenAI (por defecto Groq,
  que tiene capa gratuita). Se configura con LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, APITimeoutError, APIError, AuthenticationError
from dotenv import load_dotenv

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Rutas y configuración
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectors")))
COLLECTION_NAME = "docs"

# Backend de embeddings (DEBE ser el mismo que en ingestion.py)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM compatible con la API de OpenAI.
# Por defecto: Groq (gratis, sin tarjeta). Para fully-local con Ollama, usar:
#   LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama  LLM_MODEL=llama3.2
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Umbral de relevancia (distancia coseno, rango 0-2; menor = más parecido).
# OJO: el valor "bueno" depende del modelo de embeddings; recalibrar si cambiás de backend.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.5"))

NO_INFO_MSG = "Lo siento, no tengo información en mi documentación para responder esa pregunta."

if not LLM_API_KEY:
    logger.warning("LLM_API_KEY no encontrada. La generación de respuestas fallará.")


def make_embedding_function():
    """Misma lógica que en ingestion.py: el backend tiene que coincidir."""
    if EMBEDDING_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("EMBEDDING_BACKEND=openai pero no hay OPENAI_API_KEY.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )
    return embedding_functions.DefaultEmbeddingFunction()


# Inicializar clientes
app = FastAPI(title="API de Soporte - Prueba Técnica Unilink")
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=30.0)
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
embedding_fn = make_embedding_function()

# Obtener la colección. Capturamos Exception (no solo ValueError) porque las
# versiones recientes de Chroma lanzan NotFoundError/InvalidCollectionException.
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
except Exception as e:
    logger.error(
        f"No se pudo abrir la colección '{COLLECTION_NAME}': {e}. "
        "Ejecutá ingestion.py primero."
    )
    collection = None


# Modelos de datos para la API
class QuestionRequest(BaseModel):
    pregunta: str = Field(..., description="La pregunta del usuario")

class AnswerResponse(BaseModel):
    respuesta: str
    contexto_usado: bool
    fuentes: list[str] = []


def generar_prompt(pregunta: str, contexto: str) -> list[dict]:
    """Genera los mensajes con las instrucciones estrictas (System Prompt)."""
    return [
        {
            "role": "system",
            "content": (
                "Eres un asistente de soporte técnico automatizado. "
                "Tu tarea es responder a las preguntas de los usuarios basándote ÚNICAMENTE en "
                "el contexto proporcionado a continuación.\n\n"
                "REGLAS ESTRICTAS:\n"
                "1. Si la respuesta no se encuentra en el contexto, DEBES responder exactamente "
                f"esto: '{NO_INFO_MSG}'\n"
                "2. NO inventes información (no alucines).\n"
                "3. Mantén un tono profesional y claro.\n\n"
                f"CONTEXTO RECUPERADO:\n{contexto}"
            )
        },
        {"role": "user", "content": pregunta}
    ]


# Endpoint SÍNCRONO (def, no async def): las llamadas a ChromaDB y al LLM son
# bloqueantes; con 'def' FastAPI las corre en un threadpool y no traba el event loop.
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    # 1. Manejo de inputs vacíos
    query_text = request.pregunta.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía."
        )

    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La base de datos vectorial no está inicializada."
        )

    # 2. Búsqueda semántica
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Error consultando ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Error en la búsqueda de información.")

    documentos = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distancias = results["distances"][0] if results["distances"] else []

    # Filtrar por distancia (umbral de relevancia)
    context_chunks = []
    fuentes_utilizadas = set()

    for doc, meta, dist in zip(documentos, metadatas, distancias):
        if dist < RELEVANCE_THRESHOLD:
            context_chunks.append(doc)
            fuentes_utilizadas.add(meta.get("source", "Desconocida"))

    # 3. Preguntas sin respuesta (si no hay contexto relevante)
    if not context_chunks:
        return AnswerResponse(
            respuesta=NO_INFO_MSG,
            contexto_usado=False,
            fuentes=[]
        )

    contexto_unido = "\n\n---\n\n".join(context_chunks)
    mensajes = generar_prompt(query_text, contexto_unido)

    # 4. Generación con el LLM (Groq/OpenAI-compatible) y manejo de errores
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=mensajes,
            temperature=0.0,  # determinista, para evitar alucinaciones
            max_tokens=300
        )
        respuesta_ia = response.choices[0].message.content or NO_INFO_MSG

        return AnswerResponse(
            respuesta=respuesta_ia,
            contexto_usado=True,
            fuentes=sorted(fuentes_utilizadas)
        )

    except APITimeoutError:
        logger.error("Timeout del LLM")
        raise HTTPException(status_code=504, detail="El servicio de IA tardó demasiado en responder (Timeout).")
    except AuthenticationError:
        logger.error("Error de autenticación con el LLM (revisá LLM_API_KEY)")
        raise HTTPException(status_code=401, detail="Error de credenciales en el servicio de IA.")
    except APIError as e:
        logger.error(f"Error en la API del LLM: {e}")
        raise HTTPException(status_code=502, detail="Error temporal en el servicio de generación de respuestas.")
    except Exception as e:
        logger.exception("Error inesperado generando la respuesta")
        raise HTTPException(status_code=500, detail="Error interno procesando la solicitud.")


# Endpoint de salud para verificar que la API está viva
@app.get("/health")
def health_check():
    return {"status": "ok", "vectorstore_ready": collection is not None}
