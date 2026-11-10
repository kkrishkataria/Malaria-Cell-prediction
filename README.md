# CellScope AI - Malaria Cell Classification

CellScope AI is an end-to-end deep-learning project that classifies microscopic blood-cell images as **Parasitized** or **Uninfected**. It combines a custom TensorFlow/Keras convolutional neural network (CNN), a FastAPI inference service, and a polished React dashboard.

> **Educational use only.** This project is a machine-learning demonstration and must not be used as a clinical diagnostic tool or as a substitute for qualified medical advice.

## Live service

- API: <https://malaria-cell-prediction.onrender.com>
- Health check: <https://malaria-cell-prediction.onrender.com/api/health>
- Interactive API documentation: <https://malaria-cell-prediction.onrender.com/docs>

## Features

- Binary image classification: Parasitized vs Uninfected cells
- Custom CNN built with TensorFlow and Keras
- Image preprocessing with NumPy and Pillow
- FastAPI REST API with file validation and JSON responses
- React/Vite interface with image preview, confidence score, and risk level
- Training metrics exported as CSV and a Matplotlib accuracy chart
- Scikit-learn classification report and confusion matrix after training

## Technology stack

| Area | Tools |
| --- | --- |
| ML and image processing | Python, TensorFlow, Keras, NumPy, Pillow |
| Data and evaluation | Pandas, Matplotlib, Scikit-learn |
| API | FastAPI, Uvicorn |
| Frontend | React, JavaScript, CSS, Vite |
| Deployment | Render (API), Vercel (frontend) |

## Project structure

```text
app/
  main.py                 FastAPI application and prediction endpoint
  train.py                CNN training and evaluation pipeline
data/
  raw/cell_images/        Dataset folders (not tracked in Git)
    Parasitized/
    Uninfected/
  processed/              Generated metrics and training chart
models/                   Trained .keras model and metadata
frontend/
  src/main.jsx            React interface
  src/styles.css          Dashboard styling
  .env.example            Deployment environment-variable template
notebooks/                Space for EDA and experiments
requirements.txt          Python dependencies
```

## Dataset

Place the image folders in this exact layout:

```text
data/raw/cell_images/
├── Parasitized/
└── Uninfected/
```

The local project dataset contains 13,780 images in each class. Dataset files are excluded from Git to keep the repository lightweight.

## Local setup

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the CNN

```bash
python app/train.py
```

Training saves these generated files:

```text
models/malaria_cnn.keras
models/model_metadata.json
data/processed/training_metrics.csv
data/processed/training_history.png
```

Training is required only once. Retrain when you change the dataset or the CNN architecture.

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at <http://localhost:8000>, and Swagger documentation is available at <http://localhost:8000/docs>.

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | API landing page with useful links |
| `GET` | `/api/health` | Service and model availability |
| `GET` | `/api/model-info` | Saved model metadata and metrics |
| `POST` | `/api/predict` | Predict a class from a PNG or JPEG cell image |

Example prediction request:

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@data/raw/cell_images/Parasitized/example.png"
```

Example response:

```json
{
  "prediction": "Parasitized",
  "confidence": 98.42,
  "infection_probability": 98.42,
  "risk_level": "High"
}
```

## Deployment

### Backend - Render

Create a Render Python Web Service connected to this repository.

```text
Build command: pip install -r requirements.txt
Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Commit the trained `models/malaria_cnn.keras` and `models/model_metadata.json` files before deploying. The dataset itself is not needed by the live API.

### Frontend - Vercel

Import the `frontend` folder as a Vite project. In Vercel's environment variables, add:

```text
VITE_API_URL=https://malaria-cell-prediction.onrender.com
```

Use only the base URL: do **not** add `/api`, `/api/health`, or `/api/predict`. Redeploy the frontend after saving the variable.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `model_trained: false` | Run `python app/train.py`, then deploy the generated model files. |
| `The prediction service returned an unexpected response` | Confirm `VITE_API_URL` is the Render base URL only, then redeploy Vercel. |
| Render logs mention CUDA drivers | Safe to ignore. Render runs this project on CPU. |
| API returns `503 Model not trained yet` | The deployment has no `models/malaria_cnn.keras` file. Add the trained model and redeploy. |

## Verification

The deployed API has been checked successfully:

```text
GET /api/health   -> 200, model_trained: true
POST /api/predict -> 200, JSON prediction response
```
