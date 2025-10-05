# ============================================================================
# Dockerfile — Contenedor de la API Iris (Actividad 2)
# Autor: John Gómez
# Descripción:
#   Imagen ligera con Python 3.12 que sirve la API Flask en puerto 5002.
# Uso:
#   docker build -t iris-api .
#   docker run -d --name irisapi -p 5002:5002 iris-api
# ============================================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PORT=5002

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Código fuente (sin modelo)
COPY app.py .
COPY train_model.py .

# ⬇️ Entrena el modelo en build -> genera modelo.pkl dentro de la imagen
RUN python -u train_model.py

# Exponer puerto
EXPOSE 5002

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys;r=urllib.request.urlopen('http://127.0.0.1:5002/health',timeout=3);sys.exit(0 if r.status==200 else 1)" || exit 1

# Arranque
CMD ["python", "app.py"]






