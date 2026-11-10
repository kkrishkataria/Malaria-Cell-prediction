# CellScope AI - Malaria Cell Classification

An end-to-end malaria cell image classifier with a TensorFlow/Keras CNN, a FastAPI inference API, and a React dashboard.

## Project structure

```text
app/                  FastAPI API and CNN training script
data/raw/cell_images/ Dataset: Parasitized/ and Uninfected/
data/processed/       Generated metrics and training plots
models/               Saved Keras model and evaluation metadata
notebooks/            EDA or experimentation notebooks
frontend/             React + Vite dashboard
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/train.py
uvicorn app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The model API runs at `http://localhost:8000`.

## Deployment

Deploy `frontend/` to Vercel and `app/` to a Python web service such as Render.
In Vercel, add an environment variable named `VITE_API_URL` with the public
Render service URL, for example `https://cellscope-api.onrender.com`. Redeploy
the frontend after saving the variable.

> This project is for learning and screening support only. It is not a clinical diagnostic device.
