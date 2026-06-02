"""
Cliente de consola para consultar el asistente de soporte.
Uso:
    python preguntar.py
        -> modo interactivo: se escribe la pregunta o consulta y se ve la respuesta.
    python preguntar.py "¿...?"
        -> consulta única y termina.
"""

import sys
import json
import urllib.request

API_URL = "http://127.0.0.1:8000/ask"


def preguntar(texto: str) -> None:
    payload = json.dumps({"pregunta": texto}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"\n[Error al consultar la API: {e}]")
        print("¿Está corriendo uvicorn en http://127.0.0.1:8000 ?\n")
        return

    print("\n" + "=" * 60)
    print("RESPUESTA:")
    print(data.get("respuesta", "(sin respuesta)"))
    fuentes = data.get("fuentes", [])
    if fuentes:
        print("\nFUENTES: " + ", ".join(fuentes))
    print(f"Contexto usado: {data.get('contexto_usado')}")
    print("=" * 60 + "\n")


def main() -> None:
    # Modo consulta única: python preguntar.py "mi pregunta"
    if len(sys.argv) > 1:
        preguntar(" ".join(sys.argv[1:]))
        return

    # Modo interactivo
    print("Asistente de soporte. Escribí tu pregunta (o 'salir' para terminar).")
    while True:
        try:
            texto = input("\nPregunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break
        if texto.lower() in ("salir", "exit", "quit", ""):
            print("Hasta luego.")
            break
        preguntar(texto)


if __name__ == "__main__":
    main()
