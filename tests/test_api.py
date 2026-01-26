# tests/test_api.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api_app.main as main


def load_s3_features() -> List[str]:
    """Charge le contrat d'entrée (liste des features) depuis les artifacts."""
    features_path = Path("artifacts/scenario3_features.json")
    assert features_path.exists(), "artifacts/scenario3_features.json doit exister dans le repo"
    return json.loads(features_path.read_text(encoding="utf-8"))


class DummyModel:
    """
    Faux modèle minimal pour tester /predict sans dépendre d'un joblib réel.
    On simule predict + predict_proba (format sklearn).
    """

    def predict(self, X):  # noqa: N802 (signature sklearn-like)
        # Retourne toujours "1" (succès) pour rendre le test stable
        return np.array([1])

    def predict_proba(self, X):  # noqa: N802
        # proba classe0=0.2, classe1=0.8 (succès)
        return np.array([[0.2, 0.8]])

@pytest.mark.skipif(
    os.getenv("CI", "").lower() == "true",
    reason=(
        "Le endpoint /train dépend de PostgreSQL. "
        "En CI GitHub Actions, la DB n'est pas disponible, on skip ce smoke test."
    ),
)

@pytest.fixture()
def client_with_dummy_model() -> TestClient:
    """
    Fixture : injecte EXPECTED_FEATURES + un modèle dummy
    afin de tester /predict (validation + réponse).
    """
    main.EXPECTED_FEATURES = load_s3_features()
    main.MODEL = DummyModel()
    return TestClient(main.app)


def build_valid_payload(features: List[str]) -> Dict[str, Any]:
    """
    Construit un payload valide conforme au scenario 3.
    On respecte les types attendus (int + yes/no).
    """
    base = {
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "yes",
        "famsup": "no",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 2,
        "health": 3,
        "absences": 0,
        "G1": 12,
    }
    # On renvoie exactement les features demandées par le contrat
    return {k: base[k] for k in features}


def test_health_returns_ok() -> None:
    """
    Test simple : /health répond 200 et contient des champs clés.
    (Ne dépend pas du modèle)
    """
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "expected_features_count" in data


def test_predict_valid_payload(client_with_dummy_model: TestClient) -> None:
    """Payload valide => 200 + prediction + proba_success."""
    features = load_s3_features()
    payload = {"user_id": "ci-test", "data": build_valid_payload(features)}

    r = client_with_dummy_model.post("/predict", json=payload)
    assert r.status_code == 200

    out = r.json()
    assert "prediction" in out
    assert "proba_success" in out
    assert out["prediction"] in [0, 1]
    assert 0.0 <= out["proba_success"] <= 1.0


def test_predict_missing_feature_returns_422(client_with_dummy_model: TestClient) -> None:
    """Feature manquante => 422 missing_features."""
    features = load_s3_features()
    data = build_valid_payload(features)

    # Retire une feature obligatoire
    data.pop("G1")

    payload = {"user_id": "ci-test", "data": data}
    r = client_with_dummy_model.post("/predict", json=payload)

    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "missing_features"
    assert "G1" in detail["missing"]


def test_predict_out_of_range_returns_422(client_with_dummy_model: TestClient) -> None:
    """Valeur hors bornes => 422 invalid_feature_values."""
    features = load_s3_features()
    data = build_valid_payload(features)

    # Hors bornes (G1 doit être dans [0, 20])
    data["G1"] = 999

    payload = {"user_id": "ci-test", "data": data}
    r = client_with_dummy_model.post("/predict", json=payload)

    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "invalid_feature_values"
    fields = {e["field"] for e in detail["details"]}
    assert "G1" in fields


def test_train_smoke(tmp_path: Path) -> None:
    """
    - Vérifier que l'endpoint répond correctement en environnement complet (local/Docker).
    - En CI, on SKIP car PostgreSQL n'est pas démarré (test d'infra, pas un test unitaire).
    """
    r = client.post("/train", json={"force": True})
    assert r.status_code == 200
    
    """
    Smoke test /train :
    - crée un petit dataset synthétique conforme scenario 3 + G3
    - lance /train avec dataset_path temporaire
    - vérifie réponse 200

    """
    features = load_s3_features()

    # Dataset synthétique : il faut assez de lignes pour CV 5-fold + 2 classes
    n = 60
    X = pd.DataFrame([build_valid_payload(features) for _ in range(n)])

    # Alternance pour obtenir 2 classes (target via G3 >= 10)
    G3 = np.array([15 if i % 2 == 0 else 5 for i in range(n)])
    df = X.copy()
    df["G3"] = G3

    csv_path = tmp_path / "student_synth.csv"
    df.to_csv(csv_path, index=False)

    # Important : s'assurer que l'API connaît les features attendues
    main.EXPECTED_FEATURES = features

    client = TestClient(main.app)
    r = client.post("/train", json={"dataset_path": str(csv_path), "force": True})

    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
