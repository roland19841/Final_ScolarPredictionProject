# 🎓 School Success Prediction – Industrialisation IA

## 🧱 Architecture locale (Docker)
Tous les services sont orchestrés via **Docker Compose**.

Ouvrir **Docker Desktop** et lancer la commande depuis un terminal :
```
docker compose up -d
```

---

## 🖥️ Services et accès

| Composant | Rôle | URL |
|---------|-----|-----|
| IHM Streamlit | Interface utilisateur | http://localhost:8501 |
| Swagger UI | Documentation API | http://localhost:8000/docs |
| MLflow UI | Suivi des entraînements | http://localhost:5000 |
| Adminer | Interface BDD | http://localhost:8080 |
| Prometheus | Metrics | http://localhost:9090 |
| Grafana | Dashboards | http://localhost:3000 |
| Uptime Kuma | Disponibilité API | http://localhost:3001 |

---

## 🗄️ Base de données PostgreSQL

### 🎯 Objectif
Remplacer la lecture directe d’un fichier CSV par une **base persistante**, plus proche d’un environnement de production.

- La route `/train` lit désormais les données depuis PostgreSQL
- Le CSV `student-final.csv` sert uniquement de **seed initial**
- La base est inspectable via **Adminer**

### 🔐 Connexion Adminer

| Champ | Valeur |
|-----|-------|
| Système | PostgreSQL |
| Serveur | db |
| Utilisateur | school_user |
| Mot de passe | school_pwd |
| Base de données | school |

---

## 🤖 API FastAPI

### Routes principales
- `POST /predict` : prédiction de réussite scolaire
- `POST /train` : entraînement monitoré
- `GET /health` : état de santé de l’API
- `GET /metrics` : métriques Prometheus

### Fonctionnement
- Validation des entrées avec **Pydantic**
- Modèle chargé en mémoire au démarrage
- Logs d’inférence en JSONL
- Rechargement du modèle après `/train`

---

## 📊 MLflow (MLOps)

Chaque appel à `/train` :
- crée un **run MLflow**
- enregistre paramètres, métriques, artefacts
- versionne le modèle

MLflow permet :
- comparaison des modèles
- audit des entraînements
- reproductibilité

---

## 📈 Monitoring

- **Prometheus** scrappe `/metrics`
- **Grafana** affiche latence, erreurs, trafic
- **Uptime Kuma** surveille `/health`

Objectif : observabilité sans outils cloud externes.

---

## 🔄 CI/CD (GitHub Actions)

### Workflows
- **CI** : tests (`pytest`) + lint (`flake8`)
- **Docker** : build & push image API vers Docker Hub

### Versioning
- Tags Git `vX.Y.Z` (Semantic Versioning)
- Le tag déclenche une image Docker du même nom

---

## 🧪 Démo orale type

1. `docker compose up --build`
2. Ouvrir Swagger → `/health`
3. Lancer `/train`
4. Montrer MLflow (nouveau run)
5. Tester `/predict`
6. Montrer Grafana / Kuma
7. Accéder à Adminer

---
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

Roland RENIER - Projet réalisé dans le cadre d’un **livrable de certification Expert IT / IA de SIMPLON**.


