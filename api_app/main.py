# api_app/main.py
from __future__ import annotations

import time
import os  # ✅ pour lire CORS_ORIGINS depuis l'environnement (utile en Docker)
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# ✅ Middleware CORS : indispensable si un navigateur (front web) appelle l'API
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

import sys
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sklearn  # pour logger sklearn.__version__

import mlflow  # ✅ MLflow tracking (runs, metrics, artifacts)
import mlflow.sklearn  # ✅ logging modèle sklearn (optionnel)

# --- Monitoring Prometheus : instrumentation FastAPI ---
from prometheus_fastapi_instrumentator import Instrumentator

# ============================================================
# Configuration fichiers (artefacts & logs)
# ============================================================

FEATURES_PATH = Path("artifacts/scenario3_features.json")
MODEL_PATH = Path("artifacts/model_s3_logreg.joblib")

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "inference_log.jsonl"

DATASET_DEFAULT_PATH = Path("data/student-final.csv")

# Dossier où on versionne les modèles entraînés
MODELS_DIR = Path("artifacts/models")

# Rapport de training (dernière exécution)
TRAIN_REPORT_PATH = Path("artifacts/train_report.json")


# ============================================================
# Initialisation FastAPI
# ============================================================

app = FastAPI(
    title="School Success Predictor API",
    version="1.0.0",
    description="API de prédiction : Scenario 3 + Logistic Regression (sklearn pipeline).",
)


# ============================================================
# Monitoring Prometheus : exposition /metrics
# ============================================================

# On instrumente l'API pour exporter des métriques Prometheus :
# - nombre de requêtes
# - latences
# - codes HTTP (2xx/4xx/5xx)
# Les métriques seront disponibles sur GET /metrics.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ============================================================
# MLflow (tracking des entraînements)
# ============================================================
# Objectif : chaque /train = 1 run traçable (params, metrics, artifacts, modèle).
# Par défaut, MLflow écrit dans ./mlruns (local).
# Tu peux changer via MLFLOW_TRACKING_URI si besoin (plus tard en Docker).
# ============================================================

# Dossier racine du projet (on remonte depuis ce fichier : api_app/main.py -> projet/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossier mlruns unique, stable, au niveau racine du projet
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Tracking URI absolu (format MLflow file store)
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")

# Nom d'expérience (configurable via variable d'env)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "school-success-s3")

# Création / sélection de l'expérience
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ============================================================
# CORS (Cross-Origin Resource Sharing)
# ============================================================
# CORS est nécessaire si l'appel à l'API est effectué depuis un navigateur
# (ex: front React, page web, etc.) et que l'API tourne sur un autre domaine/port.
#
# - Streamlit "pur" peut parfois appeler en serveur->serveur (requests Python) :
#   dans ce cas le CORS ne s'applique pas.
# - MAIS : le livrable "industrialisation" est plus propre si l'API est CORS-ready.
#
# Stratégie :
# - En DEV, autoriser au minimum Streamlit : http://localhost:8501
# - En option : pilotable via variable d'env CORS_ORIGINS
#
# Format CORS_ORIGINS :
#   "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000"
#
# ⚠️ En prod, éviter allow_origins=["*"] si allow_credentials=True (interdit).
# ============================================================

# Origines par défaut (dev) : Streamlit + un port classique de front (3000) au besoin
DEFAULT_CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Si la variable d'environnement existe, elle remplace la liste par défaut
origins_env = os.getenv("CORS_ORIGINS")
if origins_env:
    # On découpe la chaîne CSV en liste propre ["http://...", "http://..."]
    ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = DEFAULT_CORS_ORIGINS

# Ajout du middleware CORS à l'application FastAPI
app.add_middleware(
    CORSMiddleware,
    # Liste des origines autorisées (doit matcher exactement le Origin du navigateur)
    allow_origins=ALLOWED_ORIGINS,
    # Credentials = cookies / auth navigateur.
    # Ici on n'utilise pas d'auth cookie, donc False (plus sûr).
    allow_credentials=False,
    # Méthodes autorisées.
    # On inclut OPTIONS car les navigateurs font un "preflight" avant certains POST.
    allow_methods=["GET", "POST", "OPTIONS"],
    # Headers autorisés. "*" convient en dev. En prod, tu peux restreindre.
    allow_headers=["*"],
)


# Timestamp de démarrage (pour calculer l'uptime)
START_TIME = time.time()

# Cache du dernier report (optionnel, mais pratique)
LAST_TRAIN_REPORT: Optional[Dict[str, Any]] = None

# Variables globales (chargées au démarrage)
EXPECTED_FEATURES: list[str] = []
MODEL = None  # on stocke ici le pipeline sklearn chargé


# ============================================================
# Schémas Pydantic (validation des inputs)
# ============================================================

class PredictRequest(BaseModel):
    # user_id : utile pour tracer les appels côté log (optionnel)
    user_id: Optional[str] = Field(None, description="Identifiant user/session (optionnel)")
    # data : dict de features (scenario 3)
    data: Dict[str, Any] = Field(..., description="Features de l'élève (scenario 3)")


class PredictResponse(BaseModel):
    prediction: int
    proba_success: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str

    # Monitoring / infos utiles
    uptime_seconds: float
    expected_features_count: int
    model_path: str

    # Dernier entraînement (si train_report.json existe)
    last_train_timestamp_utc: Optional[str] = None
    dataset_used: Optional[str] = None
    model_versioned: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    env: Optional[Dict[str, str]] = None

    # Dernière inférence (si logs existent)
    last_event_timestamp_utc: Optional[str] = None
    last_event_endpoint: Optional[str] = None


class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    force: bool = False


class TrainResponse(BaseModel):
    status: str
    detail: str


# ============================================================
# Utilitaires : charger features + logger + validation
# ============================================================

def log_event(*, endpoint: str, user_id: Optional[str], inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Log JSONL : 1 ligne = 1 événement (audit + debug)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "user_id": user_id,
        "inputs": inputs,
        "outputs": outputs,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def validate_features(payload_data: Dict[str, Any], expected: list[str]) -> None:
    """
    Vérifie missing/extra features -> HTTP 422
    (validation structurelle : on ne vérifie pas encore les types/bornes ici).
    """
    missing = [f for f in expected if f not in payload_data]
    extra = [k for k in payload_data.keys() if k not in expected]

    if missing:
        raise HTTPException(status_code=422, detail={"error": "missing_features", "missing": missing})
    if extra:
        raise HTTPException(status_code=422, detail={"error": "unexpected_features", "extra": extra})

# ============================================================
# Validation renforcée (types + bornes + normalisation)
# Scenario 3 : basé sur artifacts/scenario3_features.json
# ============================================================


def _to_int(value: Any) -> int:
    """
    Convertit une valeur en int de façon robuste.
    - Accepte int
    - Accepte float entier (ex: 2.0)
    - Accepte str numérique (ex: "12")
    - Refuse bool et le reste
    """
    if isinstance(value, bool):
        # bool est un sous-type de int en Python -> on le refuse explicitement
        raise ValueError("boolean is not a valid int")

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)

    raise ValueError("cannot convert to int")


def _to_yes_no(value: Any) -> str:
    """
    Normalise une valeur yes/no.
    - Accepte : Yes/YES/y/true/1 -> "yes"
    - Accepte : No/NO/n/false/0 -> "no"
    - Accepte bool
    """
    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, (int, float)):
        if value == 1:
            return "yes"
        if value == 0:
            return "no"

    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return "yes"
        if s in {"no", "n", "false", "0"}:
            return "no"

    raise ValueError("expected yes/no")


def validate_and_clean_data_s3(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validation + normalisation adaptée EXACTEMENT au Scenario 3.

    Features attendues (19) :
      traveltime, studytime, failures,
      schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
      famrel, freetime, goout, Dalc, Walc, health,
      absences, G1

    Renvoie un dict "clean" (types cohérents) prêt pour sklearn.
    En cas d'erreurs : HTTP 422 avec détails par champ.
    """
    errors: list[Dict[str, Any]] = []
    clean = dict(data)  # copie défensive

    # ----------------------------
    # 1) Champs numériques (int) + bornes
    # ----------------------------
    # Bornes standard du dataset Student Performance.
    int_ranges: Dict[str, tuple[int, int]] = {
        "traveltime": (1, 4),
        "studytime": (1, 4),
        "failures": (0, 4),
        "famrel": (1, 5),
        "freetime": (1, 5),
        "goout": (1, 5),
        "Dalc": (1, 5),
        "Walc": (1, 5),
        "health": (1, 5),
        "absences": (0, 93),
        "G1": (0, 20),
    }

    # ----------------------------
    # 2) Champs yes/no à normaliser
    # ----------------------------
    yes_no_fields = {
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
    }

    # ----------------------------
    # 3) Conversion + contrôle des int bornés
    # ----------------------------
    for field, (mn, mx) in int_ranges.items():
        if field not in clean:
            # Le missing est déjà géré par validate_features, mais on reste safe
            continue
        try:
            v = _to_int(clean[field])
            if v < mn or v > mx:
                raise ValueError(f"out of range [{mn}, {mx}]")
            clean[field] = v
        except Exception as e:
            errors.append(
                {
                    "field": field,
                    "value": clean[field],
                    "error": str(e),
                    "expected": f"int in [{mn}, {mx}]",
                }
            )

    # ----------------------------
    # 4) Normalisation yes/no
    # ----------------------------
    for field in yes_no_fields:
        if field not in clean:
            continue
        try:
            clean[field] = _to_yes_no(clean[field])
        except Exception as e:
            errors.append(
                {
                    "field": field,
                    "value": clean[field],
                    "error": str(e),
                    "expected": "yes/no",
                }
            )

    # ----------------------------
    # 5) Si erreurs -> 422 explicite
    # ----------------------------
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_feature_values", "details": errors},
        )

    return clean

# ============================================================
# Chargement au démarrage (startup)
# ============================================================


@app.on_event("startup")
def startup_load_artifacts() -> None:
    """
    Chargement des artefacts au démarrage :
    - liste des features attendues
    - pipeline sklearn (joblib)
    - report d'entraînement (pour /health)
    """
    global EXPECTED_FEATURES, MODEL, LAST_TRAIN_REPORT

    # 1) Charger features attendues
    if not FEATURES_PATH.exists():
        print(f"⚠️ FEATURES file missing: {FEATURES_PATH}")
        EXPECTED_FEATURES = []
    else:
        EXPECTED_FEATURES = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))

    # 2) Charger modèle
    if not MODEL_PATH.exists():
        print(f"⚠️ MODEL file missing: {MODEL_PATH}")
        MODEL = None
    else:
        try:
            MODEL = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            MODEL = None

    # 3) Charger le dernier train report en cache (optionnel)
    LAST_TRAIN_REPORT = load_train_report()


def build_s3_logreg_pipeline(X_example: pd.DataFrame) -> Pipeline:
    """
    Construit le pipeline Scenario 3 + Logistic Regression, identique à celui du notebook :
    - Numérique : imputer médiane + scaler
    - Catégoriel : imputer mode + OneHot (handle_unknown='ignore')
    - Modèle : LogisticRegression
    """
    # Colonnes numériques/catégorielles détectées à partir du DataFrame
    num_cols = X_example.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_example.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Pipeline pour les colonnes numériques
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Pipeline pour les colonnes catégorielles
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Prétraitement global : appliquer chaque pipeline à ses colonnes
    preprocess = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # Modèle : régression logistique (classif binaire)
    model = LogisticRegression(max_iter=2000)

    # Pipeline complet
    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])


def save_train_report(report: Dict[str, Any]) -> None:
    """Sauvegarde un rapport JSON d'entraînement (audit + /health)."""
    TRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def load_train_report() -> Optional[Dict[str, Any]]:
    """Charge le dernier rapport d'entraînement si disponible, sinon None."""
    try:
        if not TRAIN_REPORT_PATH.exists():
            return None
        return json.loads(TRAIN_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_last_inference_event() -> Optional[Dict[str, Any]]:
    """
    Lit le dernier événement dans logs/inference_log.jsonl.
    Renvoie None si le fichier n'existe pas ou est vide.
    """
    try:
        if not LOG_FILE.exists():
            return None

        # Lecture simple : OK si log de taille raisonnable.
        # Si gros volume : lire la fin du fichier (optimisation possible).
        lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def versioned_model_path(prefix: str = "model_s3_logreg") -> Path:
    """Construit un nom de fichier versionné artifacts/models/<prefix>_YYYYMMDD_HHMMSS.joblib"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / f"{prefix}_{ts}.joblib"


# ============================================================
# ROUTES
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check enrichi :
    - API up
    - modèle chargé ou non
    - uptime
    - nb features attendues
    - infos dernier entraînement (train_report)
    - dernier event de log
    """
    report = LAST_TRAIN_REPORT or load_train_report()
    last_event = get_last_inference_event()

    # Uptime
    uptime = time.time() - START_TIME

    # Statut modèle
    model_loaded = (MODEL is not None)
    mode = "real_model" if model_loaded else "no_model"

    # Champs issus du report (si présent)
    last_ts = None
    dataset_used = None
    model_versioned = None
    metrics = None
    env = None

    if report:
        last_ts = report.get("timestamp_utc")
        dataset_used = report.get("dataset_used")
        metrics = report.get("metrics")
        env = report.get("env")

        # Chemin modèle versionné dans report["artifacts"]["model_versioned"]
        artifacts = report.get("artifacts", {})
        model_versioned = artifacts.get("model_versioned")

    # Champs issus du dernier log (si présent)
    last_event_ts = None
    last_event_endpoint = None
    if last_event:
        last_event_ts = last_event.get("timestamp_utc")
        last_event_endpoint = last_event.get("endpoint")

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        mode=mode,
        uptime_seconds=float(uptime),
        expected_features_count=len(EXPECTED_FEATURES),
        model_path=str(MODEL_PATH),
        last_train_timestamp_utc=last_ts,
        dataset_used=dataset_used,
        model_versioned=model_versioned,
        metrics=metrics,
        env=env,
        last_event_timestamp_utc=last_event_ts,
        last_event_endpoint=last_event_endpoint,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Prédiction :
    - vérifie modèle chargé
    - validate_features (missing/extra)
    - DataFrame 1 ligne dans l'ordre attendu
    - predict + predict_proba
    - log JSONL
    """
    # Vérifier que le modèle est chargé
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded (joblib missing or failed to load).")

    # Vérifier qu'on a la liste de features
    if not EXPECTED_FEATURES:
        raise HTTPException(status_code=500, detail="Expected features list is empty or missing.")

    # Validation structurelle stricte des features
    validate_features(req.data, EXPECTED_FEATURES)

    # 2bis) Validation renforcée + normalisation (Scenario 3)
    clean_data = validate_and_clean_data_s3(req.data)

    # 3) Construire X sous forme DataFrame (1 ligne) dans l'ordre EXACT attendu
    X = pd.DataFrame([clean_data], columns=EXPECTED_FEATURES)

    # Construire X = DataFrame 1 ligne dans l'ordre EXACT des features attendues
    X = pd.DataFrame([req.data], columns=EXPECTED_FEATURES)

    # Prédiction
    try:
        pred = int(MODEL.predict(X)[0])

        # LogisticRegression => predict_proba disponible (mais on garde un fallback)
        if hasattr(MODEL, "predict_proba"):
            proba_success = float(MODEL.predict_proba(X)[0][1])
        else:
            proba_success = 1.0 if pred == 1 else 0.0

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Log d'audit
    log_event(
        endpoint="/predict",
        user_id=req.user_id,
        inputs=clean_data,  # on loggue les valeurs normalisées
        outputs={"prediction": pred, "proba_success": proba_success},
    )

    return PredictResponse(prediction=pred, proba_success=proba_success)


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    """
    Entraînement monitoré :
    - charge CSV
    - crée target si besoin (G3 >= 10)
    - split train/test + CV
    - réentraîne final sur tout le dataset
    - sauvegarde versionnée + modèle courant
    - écrit report JSON
    - remplace MODEL en mémoire
    - log JSONL
    """
    global MODEL, LAST_TRAIN_REPORT

    # 1) Choix du dataset
    dataset_path = Path(req.dataset_path) if req.dataset_path else DATASET_DEFAULT_PATH
    if not dataset_path.exists():
        raise HTTPException(status_code=422, detail=f"Dataset not found: {dataset_path}")

    # 2) Vérifier la liste de features attendues
    if not EXPECTED_FEATURES:
        raise HTTPException(status_code=500, detail="Expected features list is empty or missing.")

    # 3) Charger les données
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}")

    # 4) Créer la cible binaire si nécessaire
    if "target" not in df.columns:
        if "G3" not in df.columns:
            raise HTTPException(status_code=422, detail="Dataset must contain 'G3' (or already have 'target').")
        df["target"] = (df["G3"] >= 10).astype(int)

    y = df["target"].copy()

    # 5) Construire X (Scenario 3)
    try:
        X = df[EXPECTED_FEATURES].copy()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Dataset missing expected features: {e}")

    # 6) Évaluation robustesse (split train/test + CV)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe = build_s3_logreg_pipeline(X_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        # NB : on appelle cross_val_score deux fois (accuracy + f1) — OK pour un examen,
        # mais en prod on optimiserait (cross_validate) pour éviter de refitter deux fois.
        cv_acc = float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")))
        cv_f1 = float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {e}")

    # Fit train + score test
    try:
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        test_acc = float(accuracy_score(y_te, preds))
        test_f1 = float(f1_score(y_te, preds))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Train/test evaluation failed: {e}")

    # 7) Entraînement final sur tout le dataset
    final_pipe = build_s3_logreg_pipeline(X)
    try:
        final_pipe.fit(X, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Final training failed: {e}")

    # 8) Sauvegarde modèle (versioning + modèle courant)
    if MODEL_PATH.exists() and not req.force:
        raise HTTPException(status_code=409, detail="Model already exists. Use force=true to overwrite.")

    ver_path = versioned_model_path(prefix="model_s3_logreg")

    try:
        joblib.dump(final_pipe, ver_path)    # sauvegarde versionnée (historique)
        joblib.dump(final_pipe, MODEL_PATH)    # modèle "courant" (chemin stable)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {e}")

    # 9) Rapport d'entraînement (pour /health + audit)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_used": str(dataset_path),
        "scenario": "Scenario 3",
        "model": "LogisticRegression",
        "artifacts": {
            "model_current": str(MODEL_PATH),
            "model_versioned": str(ver_path),
            "features_path": str(FEATURES_PATH),
        },
        "metrics": {
            "cv_accuracy": cv_acc,
            "cv_f1": cv_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        },
        "data_info": {
            "n_rows": int(df.shape[0]),
            "n_features": int(X.shape[1]),
            "class_ratio_success": float(y.mean()),
        },
        "env": {
            "python": sys.version,
            "sklearn": sklearn.__version__,
        }
    }

    save_train_report(report)

    # ----------------------------
    # 9bis) MLflow tracking (1 run par appel /train)
    # ----------------------------
    # MLflow ne remplace PAS les logs JSONL :
    # - JSONL = audit des appels (predict/train)
    # - MLflow = suivi des entraînements (params/metrics/artifacts/modèles)
    try:
        with mlflow.start_run(run_name=f"s3_logreg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
            # --- Params : configuration d'entraînement (reproductibilité)
            mlflow.log_param("scenario", "Scenario 3")
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("dataset_path", str(dataset_path))
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_n_splits", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("force_overwrite", req.force)

            # --- Info utile : dimensions
            mlflow.log_param("n_rows", int(df.shape[0]))
            mlflow.log_param("n_features", int(X.shape[1]))

            # --- Versions (utile pour debug compatibilité)
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("sklearn_version", sklearn.__version__)

            # --- Metrics : résultats de ton évaluation
            mlflow.log_metric("cv_accuracy", cv_acc)
            mlflow.log_metric("cv_f1", cv_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_f1", test_f1)

            # --- Artifacts : fichiers utiles pour audit / rendu
            # On s'assure que le train_report est bien écrit avant de l'uploader
            # (si tu appelles ce bloc après save_train_report, c'est parfait)
            if TRAIN_REPORT_PATH.exists():
                mlflow.log_artifact(str(TRAIN_REPORT_PATH), artifact_path="reports")

            if FEATURES_PATH.exists():
                mlflow.log_artifact(str(FEATURES_PATH), artifact_path="contracts")

            # --- Modèle : 2 options
            # Option A (simple) : loguer le joblib versionné
            if ver_path.exists():
                mlflow.log_artifact(str(ver_path), artifact_path="models_joblib")

            # Option B (bonus) : loguer le modèle en format MLflow (plus “standard”)
            # Cela permet plus tard de le recharger via mlflow.sklearn.load_model(...)
            mlflow.sklearn.log_model(
                sk_model=final_pipe,
                artifact_path="model",
            )

            # Tag : permet de filtrer facilement dans l'UI
            mlflow.set_tag("endpoint", "/train")
            mlflow.set_tag("project", "school-success-predictor")
    except Exception as e:
        # En prod, on ne veut pas qu'un souci MLflow casse /train
        # => on loggue (print) et on continue.
        print(f"⚠️ MLflow logging failed: {e}")

    save_train_report(report)
    LAST_TRAIN_REPORT = report

    # 10) Reload en mémoire (sans redémarrer l'API)
    MODEL = final_pipe

    # 11) Log d'audit
    log_event(
        endpoint="/train",
        user_id=None,
        inputs={"dataset_path": str(dataset_path), "force": req.force},
        outputs={"status": "trained", "metrics": report["metrics"], "model_versioned": str(ver_path)},
    )

    return TrainResponse(
        status="ok",
        detail=(
            f"Training complete. "
            f"CV acc={cv_acc:.3f}, CV f1={cv_f1:.3f} | "
            f"Test acc={test_acc:.3f}, Test f1={test_f1:.3f} | "
            f"Saved: {ver_path.name}"
        ),
    )
