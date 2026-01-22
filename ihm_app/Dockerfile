# ihm_app/Dockerfile
# ============================================================
# Image de base : Python 3.11 (slim)
# ============================================================
FROM python:3.11-slim

# Logs immédiats + pas de .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ============================================================
# Installer les dépendances Streamlit
# ============================================================
COPY ihm_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ============================================================
# Copier le code Streamlit
# ============================================================
COPY ihm_app /app/ihm_app

# Exposer le port Streamlit
EXPOSE 8501

# ============================================================
# Lancer Streamlit
# --server.address=0.0.0.0 : obligatoire en Docker
# ============================================================
CMD ["streamlit", "run", "ihm_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
