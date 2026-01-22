# api_app/Dockerfile
# ============================================================
# Image de base : Python 3.11 (slim = légère)
# ============================================================
FROM python:3.11-slim

# ============================================================
# Settings Python : logs immédiats + pas de fichiers .pyc
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# Dossier de travail dans le container
# ============================================================
WORKDIR /app

# ============================================================
# Dépendances système minimales
# (utile pour certaines libs scientifiques)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Installer les dépendances Python (cache Docker optimisé)
# ============================================================
COPY api_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ============================================================
# Copier le code API
# ============================================================
COPY api_app /app/api_app

# ============================================================
# Exposer le port FastAPI
# ============================================================
EXPOSE 8000

# ============================================================
# Lancer l'API
# --host 0.0.0.0 : obligatoire en Docker pour exposer le port
# ============================================================
CMD ["uvicorn", "api_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
