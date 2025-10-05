# ============================================================================
# Makefile — Atajos para construir, correr y testear la API Iris
# Autor: John Gómez
# Fecha: 2025-09-23
# Uso:
#   make build   → construir imagen
#   make run     → ejecutar contenedor en 5000
#   make stop    → detener y eliminar contenedor
#   make logs    → ver logs en tiempo real
#   make test    → correr pruebas automáticas
# ============================================================================

# -- Variables de entorno / nombres
IMAGE=iris-api           # -- Nombre de la imagen Docker
CONTAINER=irisapi        # -- Nombre del contenedor
PORT=5000                # -- Puerto expuesto en host y contenedor

# -- Construir imagen Docker
build:
	docker build -t $(IMAGE) .  # -- Construye la imagen con tag $(IMAGE) desde el Dockerfile actual

# -- Ejecutar contenedor en background
run:
	docker run -d --name $(CONTAINER) -p $(PORT):5000 $(IMAGE)  # -- Ejecuta el contenedor, mapea puerto 5000

# -- Detener y eliminar contenedor (seguro)
stop:
	-@docker stop $(CONTAINER) || true  # -- Detiene si existe
	-@docker rm $(CONTAINER) || true    # -- Elimina si existe

# -- Ver logs en tiempo real
logs:
	docker logs -f $(CONTAINER)         # -- Fila continua de logs

# -- Probar API usando script de tests
test:
	python3 test_api.py --base-url http://127.0.0.1:$(PORT)  # -- Llama al test client con la URL correcta

# -- Reconstruir todo: stop, build y run
rebuild: stop build run
	@echo "Rebuild completo."  # -- Mensaje final

# -- Ejecutar frontend Streamlit local
frontend:
	streamlit run frontend.py  # -- Levanta la interfaz web de predicción
