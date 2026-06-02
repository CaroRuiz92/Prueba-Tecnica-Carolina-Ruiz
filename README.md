# Asistente de Soporte Técnico (RAG)

**Prueba Técnica — Unilink**

Asistente automatizado que responde preguntas de soporte basándose **únicamente** en
la documentación técnica proporcionada. Implementa un pipeline RAG: recupera los fragmentos relevantes de la documentación mediante búsqueda semántica y genera una respuesta con un LLM, orquestado mediante un workflow de n8n.

---

## Arquitectura

```
Usuario
  │  POST { "pregunta": "..." }
  ▼
┌─────────────────────── n8n ───────────────────────┐
│  Webhook  ──►  HTTP Request  ──►  Respond to Webhook│
└──────────────────────│─────────────────────────────┘
                        ▼
              API REST (FastAPI, local)
                        │
                        ├─ 1. Embedding de la pregunta
                        ├─ 2. Búsqueda semántica en ChromaDB (top 3)
                        ├─ 3. Filtro por umbral de relevancia
                        ├─ 4. Armado del prompt con el contexto recuperado
                        └─ 5. Generación de la respuesta con el LLM
                              (solo a partir del contexto)
```

La respuesta incluye el texto generado, si se usó contexto y las **fuentes**
(documentos concretos que consultó).

---

## Stack

- **Python** — procesamiento de documentos y API.
- **FastAPI** — API REST (`/ask`, `/health`, documentación automática en `/docs`).
- **ChromaDB** — base de datos vectorial (similitud coseno).
- **Embeddings locales** — `all-MiniLM-L6-v2` (vía ONNX), sin costo ni API key.
- **LLM** — vía Groq (capa gratuita), usando su **API compatible con OpenAI**.
- **n8n** — orquestación del flujo mediante Webhook HTTP.

---

## Cobertura de la entrega

| Parte | Requisito | Dónde se resuelve |
|-------|-----------|-------------------|
| 1 | Ingesta: lectura, limpieza, normalización, chunking | `app/ingestion.py` |
| 2 | Workflow en n8n con Webhook HTTP | `n8n_workflows/` |
| 3 | Recuperación + no inventar + indicar si no existe | `app/main.py` (búsqueda + guardrail) |
| 4 | Integración con IA (prompts, contexto, flujo) | `app/main.py` (`generar_prompt`, `/ask`) |
| 5 | Procesamiento con Python (embeddings, indexación, búsqueda) | `app/ingestion.py`, `app/main.py` |
| 6 | Manejo de errores (sin respuesta, API, timeouts, inputs vacíos) | `app/main.py` |
| 7 | Deployment local, README, `.env.example` | este archivo, `.env.example` |

---

## Requisitos

- Python 3.10 – 3.12 (recomendado).
- Cuenta gratuita en [Groq](https://console.groq.com) para obtener una API key.
- [n8n](https://n8n.io) para el workflow.
- [ngrok](https://ngrok.com) si n8n corre en la nube y la API en local.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd <carpeta-del-repo>

# 2. (Recomendado) entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración

Copiá `.env.example` a `.env` y completá los valores:

```bash
cp .env.example .env
```

Configuración por defecto (Groq + embeddings locales):

```env
EMBEDDING_BACKEND=local

LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_tu_clave_de_groq
LLM_MODEL=llama-3.3-70b-versatile

DOCS_DIR=./docs
VECTORSTORE_DIR=./vectorstore
CHUNK_SIZE=300
CHUNK_OVERLAP=60
```

---

## Levantamiento

### 1. Indexar la documentación

Desde la raíz del proyecto:

```bash
python app/ingestion.py
```

Lee los archivos de `docs/`, limpia y normaliza el texto, lo divide en chunks e indexa
todo en `./vectorstore`. La primera ejecución descarga el modelo de embeddings (~80 MB)
una sola vez.

### 2. Levantar la API

```bash
python -m uvicorn app.main:app --port 8000
```

Verificación: `http://127.0.0.1:8000/health` debe responder
`{"status": "ok", "vectorstore_ready": true}`.

La documentación interactiva (Swagger) queda en `http://127.0.0.1:8000/docs`, donde se
puede probar `/ask` directamente desde el navegador.

### 3. Consultar

Hay tres formas:

**a) Swagger** — `http://127.0.0.1:8000/docs` → `POST /ask` → "Try it out".

**b) Cliente de consola incluido:**

```bash
# Interactivo
python app/preguntar.py

# Consulta única
python app/preguntar.py "Me aparece Usuario o contraseña incorrectos"
```

**c) Cualquier cliente HTTP** (POST a `http://127.0.0.1:8000/ask`):

```json
{ "pregunta": "Me aparece Usuario o contraseña incorrectos" }
```

Respuesta:

```json
{
  "respuesta": "...",
  "contexto_usado": true,
  "fuentes": ["Documentación 3.md"]
}
```

---

## Conexión con n8n

El workflow exportado está en `n8n_workflows/`. Se importa desde n8n
(**⋯ → Import from File**). Estructura: **Webhook → HTTP Request → Respond to Webhook**.

Como n8n Cloud no puede acceder a `localhost`, se expone la API local con un túnel:

```bash
ngrok http 8000
```

ngrok devuelve una URL pública (p. ej. `https://xxxx.ngrok-free.app`). En el nodo
**HTTP Request** del workflow, el campo **URL** debe apuntar a esa dirección seguida de
`/ask`:

```
https://xxxx.ngrok-free.app/ask
```

> **Importante:** la URL gratuita de ngrok cambia en cada reinicio. El valor que viene
> en el `.json` exportado corresponde a una sesión anterior y debe reemplazarse por la
> URL del entorno donde se levante la API.

El nodo envía la pregunta en el cuerpo como JSON
(`{ "pregunta": "{{ $json.body.pregunta }}" }`).

---

## Estructura del proyecto

```
.
├── app/
│   ├── ingestion.py        # Ingesta: lectura, limpieza, chunking, indexación
│   ├── main.py             # API REST (FastAPI): /ask y /health
│   └── preguntar.py        # Cliente de consola para consultar la API
├── docs/                   # Documentación técnica fuente (txt, md, pdf, json)
├── n8n_workflows/          # Workflow de n8n exportado (.json)
├── tests/
│   └── test_webhook.py     # Prueba de envío al Webhook de n8n
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

> `vectorstore/` (base vectorial) y `ngrok.exe` no se versionan: se generan/descargan
> localmente.
