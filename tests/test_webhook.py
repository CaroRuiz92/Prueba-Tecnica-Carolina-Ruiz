import os
import json
import urllib.request

# FIX: la URL /webhook-test/ de n8n solo responde mientras el workflow está
# "escuchando" (botón "Test workflow" / "Listen for test event"). Para algo
# estable usá la URL de producción /webhook/preguntar. La dejamos configurable.
URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "https://carolinaruiz92.app.n8n.cloud/webhook-test/preguntar",
)

payload = json.dumps({
    "pregunta": "¿Cómo reinicio el servicio de autenticación?"
}).encode("utf-8")

req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})

try:
    # FIX: timeout para evitar que el script se cuelgue indefinidamente
    with urllib.request.urlopen(req, timeout=10) as response:
        print("¡Éxito! La pregunta entró a n8n. Código de estado:", response.getcode())
        body = response.read().decode("utf-8", errors="replace")
        if body:
            print("Respuesta:", body)
except Exception as e:
    print("Hubo un error al enviar:", e)
