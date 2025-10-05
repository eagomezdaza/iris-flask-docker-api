# ============================================================================
# frontend.py — Interfaz Streamlit para consumir la API Flask
# Autor: John Gómez
# Fecha: 2025-09-23
# Descripción:
#   Interfaz web con Streamlit para probar la API de clasificación Iris.
#   Incluye botones de prueba rápida y maneja dos formatos de respuesta:
#   - {"probabilities": {"setosa":"1.000", ...}}
#   - {"proba": [0.9, 0.1, 0.0], "target_names": [...]}
# Uso:
#   streamlit run frontend.py   → http://localhost:8501  (o el que definas)
# ============================================================================

import os
import json
import requests
import streamlit as st

# -- Configuración de la página Streamlit
st.set_page_config(page_title="Iris Predictor", layout="centered")
st.title("Predicción Iris — Frontend (Streamlit)")

st.markdown("""
Esta interfaz envía peticiones a la API Flask:
- Salud: `GET /health`
- Predicción: `POST /predict` con `{"features":[...]}`
""")

# -------- Config de API (lee de variable de entorno y permite editar) --------
default_api = os.getenv("API_URL", "http://127.0.0.1:5002")  # -- URL por defecto
api_base = st.text_input("URL base de la API", default_api)   # -- Permite cambiar URL
st.caption(f"API_URL (env): {default_api}")

# -------- Botón /health --------
col = st.columns(2)
with col[0]:
    if st.button("Probar /health"):
        try:
            r = requests.get(f"{api_base}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()

                # -- Mensaje de estado
                st.success("API saludable")

                # -- Métricas rápidas en 3 columnas
                m1, m2, m3 = st.columns(3)
                m1.metric("Versión", data.get("version", "-"))
                m2.metric("Features", data.get("n_features", "-"))

                acc = data.get("test_accuracy")
                if isinstance(acc, (int, float)):
                    m3.metric("Accuracy (test)", f"{acc*100:.1f}%")
                else:
                    m3.metric("Accuracy (test)", "-")

                # -- Mostrar clases
                clases = data.get("class_names") or []
                if clases:
                    st.write("**Clases:** " + ", ".join(map(str, clases)))

                # -- JSON completo en expander
                with st.expander("Ver JSON completo de /health"):
                    st.json(data)
            else:
                st.error(f"Error en /health (código {r.status_code})")
                try:
                    st.json(r.json())
                except Exception:
                    st.text(r.text)
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")

st.divider()  # -- Separador visual

# -------- Estado inicial para inputs --------
defaults = {"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2}  # -- Valores iniciales de ejemplo
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.subheader("Predicción")

# -------- Botones de prueba rápida --------
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Cargar Setosa"):
        st.session_state.update(sl=5.1, sw=3.5, pl=1.4, pw=0.2)  # -- Carga ejemplo Setosa
        st.rerun()
with c2:
    if st.button("Cargar Versicolor"):
        st.session_state.update(sl=5.9, sw=3.0, pl=4.2, pw=1.5)  # -- Carga ejemplo Versicolor
        st.rerun()
with c3:
    if st.button("Cargar Virginica"):
        st.session_state.update(sl=6.7, sw=3.0, pl=5.2, pw=2.3)  # -- Carga ejemplo Virginica
        st.rerun()

# -------- Inputs numéricos --------
sl = st.number_input("Sepal length", key="sl", step=0.1)
sw = st.number_input("Sepal width",  key="sw", step=0.1)
pl = st.number_input("Petal length", key="pl", step=0.1)
pw = st.number_input("Petal width",  key="pw", step=0.1)

# -------- Enviar predicción --------
if st.button("Enviar a /predict"):
    payload = {"features": [sl, sw, pl, pw]}  # -- Construir JSON de entrada
    try:
        r = requests.post(f"{api_base}/predict", json=payload, timeout=5)
        if r.status_code == 200:
            result = r.json()

            # -- Mostrar clase predicha
            pred_name = result.get("prediction", "N/A")
            st.success(f"Clase predicha: {pred_name}")

            # -- Extraer probabilidades (soporta dict o lista)
            probs = {}
            if isinstance(result.get("probabilities"), dict):
                probs = {k: float(v) for k, v in result["probabilities"].items() if v is not None}
            elif result.get("proba") is not None:
                names = result.get("target_names") or result.get("class_names") or []
                for name, p in zip(names, result["proba"]):
                    probs[name] = float(p)

            # -- Mostrar probabilidades por clase en columnas
            if probs:
                st.write("**Probabilidades por clase:**")
                ordenadas = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                cols = st.columns(len(ordenadas))
                for (name, p), c in zip(ordenadas, cols):
                    c.metric(name, f"{p*100:.1f}%")

                # -- JSON completo en expander
                with st.expander("Ver detalle y JSON de respuesta"):
                    for name, p in ordenadas:
                        st.write(f"- {name}: {p:.3f}")
                    st.json(result)
            else:
                # -- Si no hay probabilidades, al menos mostrar JSON
                with st.expander("Ver respuesta JSON completa"):
                    st.json(result)
        else:
            st.error(f"Error en la predicción (Código: {r.status_code})")
            try:
                st.json(r.json())
            except Exception:
                st.text(r.text)
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")

# -- Nota de ejecución
st.caption("Ejecuta:  `streamlit run frontend.py`  →  Local: http://localhost:8501 (o el puerto que fijes)")
