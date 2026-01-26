"""
Configuration globale pytest.

Objectif : ajouter la racine du projet au PYTHONPATH pour permettre
l'import de `api_app.main` dans les tests.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# IMPORTANT : on ajoute d'abord la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensuite seulement, on peut importer l'app
from api_app.main import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
