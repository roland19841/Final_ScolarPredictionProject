# 🎓 Prédiction de la Réussite Scolaire — Phase d’industrialisation

## 🚀 Cas d'usage standard en local

Cette section décrit **le scénario de démonstration standard** permettant de présenter l’application de bout en bout.

### 1️⃣ Lancement de l’application (Docker recommandé)

Prérequis :
- Docker
- Docker Compose

Depuis la racine du projet :

```bash
docker compose up --build
```

Tous les services sont alors démarrés automatiquement.

### 2️⃣ Accès aux interfaces

- **Interface utilisateur (IHM Streamlit)**  
  👉 http://localhost:8501  

- **API FastAPI (Swagger)**  
  👉 http://localhost:8000/docs  

- **MLflow (suivi des entraînements)**  
  👉 http://localhost:5000  

- **Prometheus (collecte des métriques)**  
  👉 http://localhost:9090  

- **Grafana (dashboards & visualisation)**  
  👉 http://localhost:3000  
  *Identifiants par défaut (si non modifiés) :* `admin` / `admin`

- **Uptime Kuma (supervision disponibilité)**  
  👉 http://localhost:3001  
---

### 3️⃣ Vérification de la santé de l’API

Dans Swagger :
- Appeler `GET /health`
- Vérifier :
  - API active
  - modèle chargé
  - uptime
  - métriques du dernier entraînement

➡️ Objectif : montrer que l’API est **monitorée et opérationnelle**.

---

### 4️⃣ Entraînement du modèle (endpoint /train)

Dans Swagger :
- Appeler `POST /train`
- (optionnel) spécifier un chemin de dataset
- Observer :
  - calcul des métriques
  - sauvegarde du modèle
  - création d’un run MLflow

Dans MLflow :
- ouvrir le run
- montrer :
  - paramètres
  - métriques
  - artefacts

➡️ Objectif : démontrer le **réentraînement monitoré et traçable**.

---

### 5️⃣ Prédiction via l’IHM

Dans l’IHM Streamlit :
- renseigner les caractéristiques d’un élève
- cliquer sur *Prédire*
- observer :
  - prédiction
  - probabilité associée

➡️ Objectif : montrer l’usage **non technique** du modèle.

---

### 6️⃣ Traçabilité des prédictions

Dans le dossier :
```
logs/inference_log.jsonl
```

Montrer qu’une ligne est ajoutée à chaque prédiction :
- inputs
- outputs
- timestamp
- user_id

➡️ Objectif : démontrer l’**auditabilité**.

---

## 🧠 Présentation générale du projet

Ce projet propose une **application de machine learning industrialisée** permettant de prédire la réussite scolaire d’un élève à partir de caractéristiques socio-éducatives (scénario 3 du dataset *Student Performance*).

L’objectif n’est pas uniquement de produire un modèle performant, mais de démontrer la capacité à :
- déployer un modèle sous forme de service
- assurer sa traçabilité
- garantir sa robustesse
- automatiser son cycle de vie

---

## 🧩 Architecture globale

La solution repose sur plusieurs composants indépendants :
- **API FastAPI** : exposition du modèle, entraînement, prédiction
- **IHM Streamlit** : interface utilisateur non technique
- **MLflow** : suivi des entraînements
- **Prometheus / Grafana / Uptime Kuma** : monitoring
- **Docker** : déploiement reproductible
- **GitHub Actions** : CI/CD

---

## 🔧 Stack technique

| Couche | Technologie |
|------|------------|
| API | FastAPI |
| IHM | Streamlit |
| ML | scikit-learn |
| Tracking | MLflow |
| Monitoring | Prometheus, Grafana, Uptime Kuma |
| CI/CD | GitHub Actions |
| Conteneurisation | Docker / Docker Compose |
| Langage | Python 3.11 |

---

## 📂 Structure du projet

```
SCOLAR_PREDICTION_PROJECT/
├── api_app/                # API FastAPI
├── ihm_app/                # Interface Streamlit
├── artifacts/              # Modèles et features
├── logs/                   # Logs d'inférence
├── mlruns/                 # MLflow
├── data/                   # Données CSV
├── docker-compose.yml
├── README.md
```

---

## 🔁 Cycle de vie Machine Learning

### Entraînement
- déclenché via `/train`
- validation croisée
- métriques sauvegardées
- modèle versionné

### Prédiction
- validation des entrées
- inférence
- journalisation automatique

---

## 📌 Bonnes pratiques mises en œuvre

- séparation claire API / IHM
- validation des données en plusieurs couches
- versioning (Semantic Versioning)
- CI/CD automatisée
- monitoring applicatif
- traçabilité des prédictions

---

## 👤 Auteur

Roland RENIER - Projet réalisé dans le cadre d’un **livrable de certification Expert IT / IA**.


