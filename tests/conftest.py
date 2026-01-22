# pour permettre de lancer les tests "pytest -q" en local
# Les tests utilisent un conftest.py pour injecter la racine du projet dans le PYTHONPATH
# ce qui permet d’importer l’API sans packager le code.
# tests/conftest.py
"""
Configuration globale pytest.

Objectif : ajouter la racine du projet au PYTHONPATH pour permettre
l'import de `api_app.main` dans les tests.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
