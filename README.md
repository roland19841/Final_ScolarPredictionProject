# 🎓 School Success Prediction — Industrialized ML Project

## 📌 Overview

This project delivers an **end-to-end industrialized Machine Learning application** for predicting student success.
It demonstrates the full lifecycle of a data product:

- Model training and evaluation
- Model versioning and tracking
- REST API for inference
- Web-based user interface
- Observability and monitoring
- Reproducible deployment with Docker

The project is aligned with **MLOps best practices** and suitable for an academic or professional evaluation.

---

## 🧠 Use Case

Predict whether a student is likely to **succeed or fail** based on socio‑educational indicators  
(using **Scenario 3** from the *Student Performance* dataset).

- Target: binary success indicator
- Prediction returned with probability
- Scenario 3 excludes final grade (G3) from inputs

---

## 🧩 Technical Stack

| Layer | Technology |
|-----|-----------|
| API | FastAPI |
| IHM | Streamlit |
| ML | scikit-learn |
| Tracking | MLflow |
| Serialization | joblib |
| Containerization | Docker / Docker Compose |
| Language | Python 3.11 |

---

## 📂 Project Structure

```
SCOLAR_PREDICTION_PROJECT/
├── api_app/                # FastAPI application
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── ihm_app/                # Streamlit interface
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── artifacts/              # ML artifacts
│   ├── scenario3_features.json
│   └── models/
│
├── logs/                   # Inference logs (JSONL)
│
├── mlruns/                 # MLflow runs (file store)
│
├── data/                   # Datasets (CSV)
│
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Execution

### Option 1 — Run locally (without Docker)

#### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate    # Windows
```

#### 2. Install dependencies
```bash
pip install -r api_app/requirements.txt
pip install -r ihm_app/requirements.txt
```

#### 3. Start the API
```bash
uvicorn api_app.main:app --reload --port 8000
```

#### 4. Start the IHM
```bash
streamlit run ihm_app/app.py
```

#### 5. (Optional) Start MLflow UI
```bash
mlflow ui --port 5000
```

Access:
- API docs: http://localhost:8000/docs
- IHM: http://localhost:8501
- MLflow UI: http://localhost:5000

---

### Option 2 — Run with Docker (recommended)

This is the **preferred method** for evaluation.

#### Prerequisites
- Docker
- Docker Compose

#### 1. Build & start all services
```bash
docker compose up --build
```

#### 2. Access services
- IHM (Streamlit): http://localhost:8501
- API (FastAPI Swagger): http://localhost:8000/docs
- MLflow UI: http://localhost:5000

Everything runs with **one command**, no Python installation required.

---

## 🔁 Machine Learning Workflow

### Training (`POST /train`)
- Loads dataset from `data/`
- Performs train/test split + cross-validation
- Computes accuracy and F1-score
- Trains final model on full dataset
- Saves:
  - Versioned model
  - Training report (`train_report.json`)
- Logs run in MLflow (params, metrics, artifacts)

### Prediction (`POST /predict`)
- Validates input features
- Applies trained pipeline
- Returns:
  - Prediction (0 / 1)
  - Probability of success
- Logs inference in `logs/inference_log.jsonl`

---

## 📊 MLflow Usage

MLflow is used **only for training runs**, not for predictions.

Each training:
- Creates one MLflow run
- Logs parameters, metrics, artifacts
- Stores runs in `mlruns/` (file-based store)

The MLflow UI allows:
- Comparing experiments
- Inspecting metrics
- Downloading models and reports

---

## 🧪 Monitoring & Observability

### `/health` endpoint
Provides:
- API status
- Model loaded or not
- Uptime
- Last training metrics
- Last inference event

### Inference logs
All predictions are logged in:
```
logs/inference_log.jsonl
```

Each line contains:
- Timestamp
- Endpoint
- User ID
- Input payload
- Output prediction

---

## 📄 Input Contract

The expected input features are defined in:
```
artifacts/scenario3_features.json
```

This file is shared by:
- API (validation)
- IHM (form generation)

It guarantees **API–UI consistency**.

---

## ✅ Key Deliverables

- ✔ Industrialized ML pipeline
- ✔ REST API with validation & monitoring
- ✔ Web interface
- ✔ MLflow experiment tracking
- ✔ Dockerized deployment
- ✔ Clean repository structure

---

## 👤 Author

Project developed as part of an **AI / IT Expert certification deliverable**.

---

## 📎 Notes for Evaluation

- Use Docker for fastest evaluation
- Train the model via Swagger or IHM
- Inspect runs in MLflow UI
- Check logs and artifacts for traceability
