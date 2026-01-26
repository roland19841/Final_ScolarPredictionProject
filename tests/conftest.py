"""
Configuration globale pytest.

Objectif : ajouter la racine du projet au PYTHONPATH pour permettre
l'import de `api_app.main` dans les tests.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api_app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
