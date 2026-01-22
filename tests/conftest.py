# pour permettre de lancer les tests "pytest -q" en local
# Les tests utilisent un conftest.py pour injecter la racine du projet dans le PYTHONPATH, ce qui permet d’importer l’API sans packager le code.
# tests/conftest.py
"""
Configuration pytest globale.
On ajoute la racine du projet au PYTHONPATH pour permettre :
    import api_app.main
"""

import sys
from pathlib import Path

# Racine du projet (parent de tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ajout au PYTHONPATH si absent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
