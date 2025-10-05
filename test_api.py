
# ======================================================================
# test_api.py — Pruebas automáticas para API Iris (Flask)
# Autor: John Gómez
# Fecha: 2025-09-23
# Uso:
#   python test_api.py                 # usa http://127.0.0.1:5000
#   python test_api.py --base-url ...  # para otro puerto/host
# Requisitos: requests
# ======================================================================

from __future__ import annotations

import argparse
import json
import sys
from typing import Tuple

import requests

# -- Función helper para GET
def get(base: str, path: str) -> Tuple[int, dict]:
    r = requests.get(base + path, timeout=10)
    try:
        return r.status_code, r.json()  # -- Devuelve código y JSON
    except Exception:
        return r.status_code, {"raw": r.text}  # -- Fallback si no es JSON

# -- Función helper para POST
def post(base: str, path: str, payload: dict) -> Tuple[int, dict]:
    r = requests.post(base + path, json=payload, timeout=10)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

# -- Función principal de pruebas
def main() -> int:
    # -- Argumentos de línea de comando
    parser = argparse.ArgumentParser(description="Pruebas para API Iris")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="URL base de la API")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    failures = []  # -- Lista para registrar errores

    # 1) /health
    sc, js = get(base, "/health")
    print("[/health]", sc, json.dumps(js, ensure_ascii=False))
    if sc != 200 or js.get("status") != "ok" or not js.get("model_loaded", False):
        failures.append("Health check falló")

    # 2) / (raíz)
    sc, js = get(base, "/")
    print("[/]", sc, json.dumps(js, ensure_ascii=False))
    if sc != 200 or js.get("status") != "success":
        failures.append("Root no devolvió success")

    # 3) /predict válido
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    sc, js = post(base, "/predict", payload)
    print("[/predict válido]", sc, json.dumps(js, ensure_ascii=False))
    if sc != 200 or js.get("status") != "success" or "prediction" not in js:
        failures.append("Predict válido no devolvió success")

    # 4) /predict inválido (menos features)
    payload_bad = {"features": [5.1, 3.5, 1.4]}
    sc, js = post(base, "/predict", payload_bad)
    print("[/predict inválido (3 features)]", sc, json.dumps(js, ensure_ascii=False))
    if sc != 400:
        failures.append("Predict inválido debería devolver 400")

    # 5) /predict inválido (tipo incorrecto)
    payload_type = {"features": [5.1, "tres", 1.4, 0.2]}
    sc, js = post(base, "/predict", payload_type)
    print("[/predict inválido (tipo)]", sc, json.dumps(js, ensure_ascii=False))
    if sc != 400:
        failures.append("Predict inválido (tipo) debería devolver 400")

    # -- Reporte final de errores o éxito
    if failures:
        print("\nFALLAS:")
        for f in failures:
            print(" -", f)
        return 1

    print("\nOK: todas las pruebas pasaron exitosamente.")
    return 0

# -- Ejecutar
if __name__ == "__main__":
    sys.exit(main())
