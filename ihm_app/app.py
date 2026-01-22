# ihm_app/app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

import os # ✅ pour lire API_URL depuis l'environnement (docker-compose)

# ============================================================
# Descriptions lisibles des variables (issues du dataset)
# ============================================================

FEATURE_DESCRIPTIONS = {
    "traveltime": "home to school travel time",
    "studytime": "weekly study time",
    "failures": "number of past class failures",
    "school": "student's school",
    "sex": "student's sex",
    "age": "student's age",
    "address": "student's home address type",
    "famsize": "family size",
    "Pstatus": "parent's cohabitation status",
    "Medu": "mother's education level",
    "Fedu": "father's education level",
    "Mjob": "mother's job",
    "Fjob": "father's job",
    "reason": "reason to choose this school",
    "guardian": "student's guardian",
    "schoolsup": "extra educational support",
    "famsup": "family educational support",
    "paid": "extra paid classes",
    "activities": "extra-curricular activities",
    "nursery": "attended nursery school",
    "higher": "wants to take higher education",
    "internet": "internet access at home",
    "romantic": "romantic relationship",
    "famrel": "quality of family relationships",
    "freetime": "free time after school",
    "goout": "going out with friends",
    "Dalc": "workday alcohol consumption",
    "Walc": "weekend alcohol consumption",
    "health": "current health status",
    "absences": "number of school absences",
    "G1": "first period grade",
}


# ============================================================
# Config générale
# ============================================================

st.set_page_config(page_title="School Success Predictor", layout="wide")
st.title("🎓 Prédiction de réussite scolaire (Scénario 3)")

# URL de l'API :
# - en local : http://localhost:8000
# - en docker-compose : http://api:8000 (nom du service docker)
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

# On garde l'input Streamlit pour debug, mais on pré-remplit avec la bonne valeur
API_BASE_URL = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)

FEATURES_PATH = Path("artifacts/scenario3_features.json")


# ============================================================
# Chargement features (artefact)
# ============================================================

@st.cache_data
def load_features(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return json.loads(path.read_text(encoding="utf-8"))

try:
    feature_cols = load_features(FEATURES_PATH)
except Exception as e:
    st.error(f"Impossible de charger {FEATURES_PATH} : {e}")
    st.stop()

st.sidebar.success(f"Features chargées : {len(feature_cols)}")
st.sidebar.caption(f"Source : {FEATURES_PATH}")


# ============================================================
# Schéma des variables (d’après ta description)
# ============================================================

# Catégorielles nominales
NOMINAL = {
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
}

# Binaires avec valeurs explicites
BINARY = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    # binaires yes/no
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
}

# Ordinales (valeurs entières avec labels)
ORDINAL_1_4 = {
    "traveltime": {
        1: "<15 min",
        2: "15–30 min",
        3: "30 min – 1 hour",
        4: ">1 hour",
    },
    "studytime": {
        1: "<2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: ">10 hours",
    },
    # failures : n si 1<=n<3, else 4
    # dans les données originales, failures est souvent 0..3 (ou 4 comme “>=3”)
    # on propose 0..4 pour rester compatible.
    "failures": {
        0: "0",
        1: "1",
        2: "2",
        4: "≥3",
    }
}

ORDINAL_1_5 = {
    "famrel": {
        1: "Très mauvais",
        2: "Mauvais",
        3: "Moyen",
        4: "Bon",
        5: "Excellent",
    },
    "freetime": {
        1: "Très faible",
        2: "Faible",
        3: "Moyen",
        4: "Élevé",
        5: "Très élevé",
    },
    "goout": {
        1: "Très faible",
        2: "Faible",
        3: "Moyen",
        4: "Élevé",
        5: "Très élevé",
    },
    "Dalc": {
        1: "Très faible",
        2: "Faible",
        3: "Moyen",
        4: "Élevé",
        5: "Très élevé",
    },
    "Walc": {
        1: "Très faible",
        2: "Faible",
        3: "Moyen",
        4: "Élevé",
        5: "Très élevé",
    },
    "health": {
        1: "Très mauvais",
        2: "Mauvais",
        3: "Moyen",
        4: "Bon",
        5: "Très bon",
    },
}

# Numériques bornées (slider / int)
NUMERIC_BOUNDED = {
    "age": (15, 22, 16),
    "absences": (0, 93, 0),
    "G1": (0, 20, 10),
    "G2": (0, 20, 10),  # normalement absent du scénario 3, mais au cas où
    "G3": (0, 20, 10),  # cible, normalement absente des features
    "Medu": (0, 4, 2),
    "Fedu": (0, 4, 2),
}

# ============================================================
# Helpers widgets
# ============================================================

def render_feature_input(feat: str) -> Any:
    """
    Affiche un widget Streamlit adapté à la feature
    et renvoie la valeur correcte à envoyer à l'API.
    """

    # Label utilisateur : nom + description
    description = FEATURE_DESCRIPTIONS.get(feat, "")
    label = f"{feat} ({description})" if description else feat

    # 1) Binaires
    if feat in BINARY:
        return st.selectbox(label, BINARY[feat], index=0)

    # 2) Nominales
    if feat in NOMINAL:
        return st.selectbox(label, NOMINAL[feat], index=0)

    # 3) Ordinales 1..4 (traveltime, studytime, failures)
    if feat in ORDINAL_1_4:
        mapping = ORDINAL_1_4[feat]
        labels = list(mapping.values())          # ce que voit l'utilisateur
        reverse = {v: k for k, v in mapping.items()}  # label → entier

        selected_label = st.selectbox(label, labels, index=0)
        return reverse[selected_label]

    # 4) Ordinales 1..5
    if feat in ORDINAL_1_5:
        mapping = ORDINAL_1_5[feat]
        labels = list(mapping.values())
        reverse = {v: k for k, v in mapping.items()}

        selected_label = st.selectbox(label, labels, index=2)
        return reverse[selected_label]

    # 5) Numériques bornées
    if feat in NUMERIC_BOUNDED:
        mn, mx, default = NUMERIC_BOUNDED[feat]
        return st.slider(label, min_value=mn, max_value=mx, value=default, step=1)

    # 6) Fallback
    return st.text_input(label, value="")


# ============================================================
# Health check
# ============================================================

st.subheader("🩺 Santé de l’API")
col_h1, col_h2 = st.columns([1, 2])

with col_h1:
    if st.button("GET /health"):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=5)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Erreur /health : {e}")

with col_h2:
    st.caption("Cette route sert à vérifier que l’API tourne et que le modèle est chargé (plus tard).")


# ============================================================
# Formulaire prédiction
# ============================================================

st.subheader("🧾 Formulaire de saisie (features Scénario 3)")

with st.form("predict_form"):
    user_id = st.text_input("user_id (optionnel)", value="demo-session-001")

    # Formulaire en 3 colonnes pour réduire le scroll
    c1, c2, c3 = st.columns(3)

    data: Dict[str, Any] = {}

    for i, feat in enumerate(feature_cols):
        container = [c1, c2, c3][i % 3]
        with container:
            data[feat] = render_feature_input(feat)

    submitted = st.form_submit_button("🔮 POST /predict")

if submitted:
    payload = {"user_id": user_id, "data": data}

    st.markdown("### 📤 Payload envoyé")
    st.json(payload)

    try:
        r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        st.write("Status:", r.status_code)

        if r.status_code != 200:
            st.error("Erreur API")
            try:
                st.json(r.json())
            except Exception:
                st.write(r.text)
        else:
            res = r.json()
            pred = res.get("prediction")
            proba = res.get("proba_success")

            st.markdown("### ✅ Résultat")
            if pred == 1:
                st.success(f"✅ Réussite probable — proba={proba:.3f}")
            else:
                st.warning(f"⚠️ Échec probable — proba={proba:.3f}")

            st.markdown("### 📥 Réponse complète")
            st.json(res)

    except Exception as e:
        st.error(f"Erreur /predict : {e}")
