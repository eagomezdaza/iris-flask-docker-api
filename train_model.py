# ============================================================================
# train_model.py — Entrenamiento y guardado del modelo Iris (Random Forest)
# Autor: John Gómez
# Fecha: 2025-09-23
# Descripción:
#   - Carga el dataset Iris de sklearn
#   - Entrena un clasificador Random Forest
#   - Evalúa la precisión (accuracy) en el conjunto de prueba
#   - Guarda el modelo entrenado y metadatos en 'modelo.pkl'
# Uso:
#   python train_model.py
#   # opcionales:
#   python train_model.py --output modelo.pkl --test-size 0.2 --n-estimators 200 --seed 42
# ============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -- Función para parsear argumentos de línea de comando
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena RandomForest sobre Iris y guarda modelo.pkl")
    p.add_argument("--output", type=Path, default=Path("modelo.pkl"), help="Ruta del artefacto a guardar")
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción para el conjunto de prueba")
    p.add_argument("--n-estimators", type=int, default=200, help="Número de árboles en el RandomForest")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    return p.parse_args()

# -- Función principal de entrenamiento
def main() -> None:
    args = parse_args()

    # -- Semilla para reproducibilidad
    np.random.seed(args.seed)

    # -- Cargar dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    # -- División en entrenamiento y prueba (estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # -- Entrenar RandomForest
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
    model.fit(X_train, y_train)

    # -- Evaluar en test
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # -- Preparar artefacto que incluye modelo + metadatos
    artifact = {
        "model": model,
        "class_names": class_names,
        "feature_names": feature_names,
        "n_features": int(X.shape[1]),
        "test_accuracy": acc,
        "training_meta": {
            "seed": args.seed,
            "test_size": args.test_size,
            "n_estimators": args.n_estimators,
            "sklearn_version": getattr(model, "__module__", "sklearn"),
        },
    }

    # -- Guardar artefacto comprimido con joblib (xz, nivel 3)
    joblib.dump(artifact, args.output, compress=("xz", 3))

    # -- Resumen legible por consola
    summary = {
        "output": str(args.output),
        "accuracy_test": round(acc, 3),
        "classes": class_names,
        "features": feature_names,
    }
    print(json.dumps(summary, ensure_ascii=False))

# -- Ejecutar solo si se llama directamente
if __name__ == "__main__":
    main()
