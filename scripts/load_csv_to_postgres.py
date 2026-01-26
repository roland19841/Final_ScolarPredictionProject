"""
Import 1 fois le dataset CSV dans PostgreSQL pour l'entraînement.

Pourquoi ?
- En prod, on évite de lire un fichier CSV "local" dans /train.
- La DB devient la source de vérité du dataset d'entraînement.
- Le contrat des features (scenario3_features.json) ne change pas.

Usage :
1) Démarrer Postgres (docker compose up -d db)
2) Exécuter ce script en local
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# Chemin de ton CSV (dataset d'entraînement)
CSV_PATH = Path("data/student-final.csv")

# URL DB (mode local : host=localhost)
# Si tu exécutes ce script depuis ta machine, garde localhost.
DATABASE_URL = os.getenv(
    "DATABASE_URL_LOCAL",
    "postgresql+psycopg2://school_user:school_pwd@localhost:5432/school",
)

TABLE_NAME = "student_data"


def main() -> None:
    # 1) Charger le dataset CSV
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # 2) Connexion SQLAlchemy
    engine = create_engine(DATABASE_URL)

    # 3) Import dans la table (replace = reconstruit la table à partir du CSV)
    # index=False : évite une colonne d'index inutile
    df.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)

    print(f"✅ Import terminé : {len(df)} lignes dans la table '{TABLE_NAME}'")


if __name__ == "__main__":
    main()
