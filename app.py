# ============================================================================
# app.py — API Flask para clasificación Iris (Actividad 2 - Docker)
# Autor: John Gómez
# Fecha: 2025-09-23
# Descripción:
#   API REST para clasificar flores Iris usando un modelo entrenado
#   (Random Forest). Expone los siguientes endpoints:
#     - GET  /        → documentación breve de la API
#     - GET  /health  → estado del servicio y metadatos del modelo
#     - POST /predict → predicción a partir de 4 características numéricas
# Uso:
#   python app.py  → http://127.0.0.1:5002
# ============================================================================

from __future__ import annotations  # -- Anotaciones de tipo modernas

from flask import Flask, request, jsonify  # -- Framework web y utilidades para JSON
import numpy as np  # -- Operaciones numéricas y arrays
import joblib  # -- Para cargar el modelo entrenado
from sklearn.datasets import load_iris  # -- Dataset de referencia (por compatibilidad)
from typing import Tuple, Union  # -- Tipado estático
import os  # -- Operaciones del sistema, paths y variables de entorno

# -- Versión de la API
API_VERSION = "1.0.0"

# -- Ruta del artefacto (modelo entrenado + metadatos), configurable por env
ARTIFACT_PATH = os.getenv("MODEL_PATH", "modelo.pkl")

# -- Crear instancia Flask
app = Flask(__name__)  # -- Aplicación principal de Flask


# --------------------------- Carga del modelo -------------------------------
try:
    # -- Intentar cargar el artefacto desde disco
    artifact = joblib.load(ARTIFACT_PATH)

    # -- Si el artefacto es un diccionario con metadatos
    if isinstance(artifact, dict):
        model = artifact.get("model")  # -- Modelo entrenado
        class_names = artifact.get("class_names", load_iris().target_names.tolist())  # -- Clases
        feature_names = artifact.get("feature_names", ["f1", "f2", "f3", "f4"])  # -- Nombres de features
        n_features = int(artifact.get("n_features", 4))  # -- Número de features esperadas
        test_accuracy = float(artifact.get("test_accuracy", 0.0))  # -- Accuracy de test guardado

    else:
        # -- Compatibilidad si solo se guardó el modelo sin diccionario
        model = artifact
        class_names = load_iris().target_names.tolist()
        feature_names = ["f1", "f2", "f3", "f4"]
        n_features = 4
        test_accuracy = 0.0

except FileNotFoundError:
    # -- Si el archivo no existe, inicializamos variables vacías
    model, class_names, feature_names, n_features, test_accuracy = None, [], [], 0, 0.0


# ----------------------------- Endpoints -----------------------------------

@app.get("/")
def root():
    """Endpoint raíz — devuelve documentación breve de la API."""
    return jsonify(
        {
            "message": "API de Clasificación Iris",  # -- Mensaje principal
            "status": "success",                     # -- Estado de respuesta
            "version": API_VERSION,                  # -- Versión de la API
            "endpoints": {                           # -- Lista de endpoints disponibles
                "GET /": "Documentación de la API",
                "GET /health": "Estado del servicio y metadatos del modelo",
                "POST /predict": "Clasificación de flores Iris",
            },
            "usage": {"predict_example": {"features": [5.1, 3.5, 1.4, 0.2]}},  # -- Ejemplo de uso de /predict
            "schema": {"POST /predict": {"features": ["float", "float", "float", "float"]}},  # -- Formato esperado
            "model_info": {                          # -- Información del modelo cargado
                "loaded": model is not None,
                "n_features": n_features,
                "feature_names": feature_names,
                "class_names": class_names,
                "test_accuracy": test_accuracy,
            },
        }
    )


@app.get("/health")
def health():
    """Endpoint de salud del servicio — devuelve estado y metadatos."""
    ok = model is not None  # -- Determinar si el modelo está cargado
    return (
        jsonify(
            {
                "status": "ok" if ok else "error",  # -- Estado general
                "message": "Servicio saludable" if ok else "Modelo no cargado",  # -- Mensaje descriptivo
                "model_loaded": ok,                 # -- Booleano de carga de modelo
                "version": API_VERSION,             # -- Versión de la API
                "n_features": n_features,           # -- Número de features esperadas
                "class_names": class_names,         # -- Nombres de clases
                "test_accuracy": test_accuracy,     # -- Accuracy de test del modelo
            }
        ),
        200 if ok else 500,  # -- Código HTTP (200 si todo bien, 500 si hay error)
    )


# ----------------------- Validación de entrada ------------------------------
def _validate(payload: dict) -> Tuple[bool, Union[str, np.ndarray]]:
    """Valida el JSON de entrada para /predict."""
    if not isinstance(payload, dict):
        return False, "El cuerpo debe ser un objeto JSON con la clave 'features'."
    if "features" not in payload:
        return False, "Falta la clave 'features' en el JSON."
    feats = payload["features"]
    if not isinstance(feats, list):
        return False, "La clave 'features' debe ser una lista."
    if len(feats) != n_features:
        return False, f"Se esperaban {n_features} valores numéricos, se recibieron {len(feats)}."
    try:
        x = np.asarray(feats, dtype=float).reshape(1, -1)  # -- Convertir a array 2D para sklearn
    except Exception:
        return False, "Todos los elementos de 'features' deben ser numéricos (int/float)."
    return True, x


# ----------------------- Endpoint /predict ----------------------------------
@app.post("/predict")
def predict():
    """Clasificación de flores Iris a partir de 4 características."""
    # -- Validar que el modelo esté cargado
    if model is None:
        return jsonify({"status": "error", "error": "Modelo no disponible."}), 500

    # -- Validar que el Content-Type sea JSON
    if not request.is_json:
        return jsonify({"status": "error", "error": "Content-Type debe ser application/json"}), 400

    # -- Obtener JSON del cuerpo de la solicitud
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"status": "error", "error": "JSON inválido o vacío."}), 400

    # -- Validar la estructura y contenido
    ok, x = _validate(payload)
    if not ok:
        return jsonify({"status": "error", "error": x}), 400

    # -- Predicción de la clase
    idx = int(model.predict(x)[0])

    # -- Obtener probabilidades si el modelo soporta predict_proba
    if hasattr(model, "predict_proba"):
        proba_list = model.predict_proba(x)[0].tolist()
        probs_dict = {class_names[i]: f"{proba_list[i]:.3f}" for i in range(len(class_names))}
    else:
        proba_list = []
        probs_dict = {class_names[idx]: "1.000"}

    # -- Respuesta JSON final
    return (
        jsonify(
            {
                "status": "success",
                "prediction": class_names[idx],
                "prediction_index": idx,
                "probabilities": probs_dict,
                "proba": proba_list,
                "target_names": class_names,
            }
        ),
        200,
    )


# --------------------------------- Main -------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("INICIANDO API FLASK")
    print("=" * 50)

    # -- Puerto por variable de entorno (útil para despliegue) o 5002 por defecto
    port = int(os.getenv("PORT", "5002"))

    # -- Ejecutar servidor Flask (producción: debug=False)
    app.run(host="0.0.0.0", port=port, debug=False)


