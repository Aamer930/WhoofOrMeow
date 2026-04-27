# 🐾 WhoofOrMeow

> A deep learning web app that classifies images as **cat** or **dog** — with Grad-CAM visual explanations, a React frontend, and a FastAPI backend.

**Arab Academy for Science, Technology and Maritime Transport**
Faculty of Computer Science — Introduction To Artificial Intelligence (CAI3101) — Cyber Security

> Supervised by **Dr. Mohamed Hamhme** | Author: **Ahmed Aamer**

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Training](#training)
  - [Quick Mode](#quick-mode)
  - [Full Training](#full-training)
  - [Transfer Learning](#transfer-learning)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Demo

| Predict Tab | Train Tab |
|-------------|-----------|
| Upload any photo → instant cat/dog prediction with confidence score and Grad-CAM heatmap showing where the model looked | Epoch slider, quick-mode toggle, and a live terminal log streaming training progress in real time |

---

## Features

- **Binary image classifier** — cat vs dog with confidence score
- **Grad-CAM heatmap** — visual explanation of model attention per prediction
- **Live training UI** — start training from the browser, watch logs stream in real time
- **Quick mode** — 2-minute smoke test on 2k images to verify the pipeline
- **Transfer learning** — MobileNetV2-based model for ~95%+ accuracy
- **FastAPI backend** — REST API with Server-Sent Events for training stream
- **React + Vite frontend** — hot-reloading dev server, no build step needed
- **Docker Compose** — single command to run everything

---

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   React Frontend    │        │     FastAPI Backend       │
│   localhost:5173    │◄──────►│     localhost:8000        │
│                     │  HTTP  │                           │
│  • Predict tab      │        │  POST /predict            │
│  • Train tab        │        │  POST /train  (SSE)       │
│  • Grad-CAM viewer  │        │  GET  /health             │
└─────────────────────┘        └──────────┬───────────────┘
                                          │
                               ┌──────────▼───────────────┐
                               │    TensorFlow Model       │
                               │    models/best_model.keras│
                               │                           │
                               │  • CNN inference          │
                               │  • Grad-CAM generation    │
                               │  • Training subprocess    │
                               └──────────────────────────┘
```

---

## Project Structure

```
WhoofOrMeow/
│
├── data/
│   └── dog-vs-cat/               # Dataset (not included — see Getting Started)
│       ├── cat/                  # 12,500 cat images
│       └── dog/                  # 12,500 dog images
│
├── frontend/                     # React + Vite web app
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js            # Dev server + API proxy
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── components/
│           ├── PredictTab.jsx    # Image upload + prediction display
│           └── TrainTab.jsx      # Training controls + live log
│
├── models/                       # Saved model files (generated after training)
│   ├── best_model.keras
│   ├── dog_cat_classifier.keras
│   └── training_curves.png
│
├── notebooks/
│   └── final.ipynb               # Jupyter notebook version
│
├── samples/                      # Test images
│
├── src/                          # Python backend
│   ├── api.py                    # FastAPI app
│   ├── train.py                  # CNN training script
│   ├── train_transfer.py         # MobileNetV2 transfer learning
│   ├── predict.py                # CLI prediction script
│   ├── evaluate.py               # Confusion matrix + metrics
│   └── gradcam.py                # Grad-CAM heatmap generation
│
├── Dockerfile                    # Backend container (FastAPI)
├── docker-compose.yml
├── requirements.txt
└── .vscode/settings.json         # Points VS Code to .venv interpreter
```

---

## Getting Started

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 | TensorFlow requires ≤ 3.11 |
| Node.js | 18+ | For the React frontend |
| Docker Desktop | Any | For Docker setup |

### Dataset

Download the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle and extract it so the structure matches:

```
data/
└── dog-vs-cat/
    ├── cat/   ← cat.0.jpg … cat.12499.jpg
    └── dog/   ← dog.0.jpg … dog.12499.jpg
```

---

### Local Setup

**1. Create virtual environment**

```bash
python3.11 -m venv .venv
```

**2. Activate it**

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Install frontend dependencies**

```bash
cd frontend
npm install
```

**5. Train a model** *(skip if you already have `models/best_model.keras`)*

```bash
# Quick smoke test (~2 min, CPU)
python src/train.py --quick

# Full training (~30-45 min on CPU, much faster with GPU)
python src/train.py --epochs 30
```

**6. Start the backend**

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

**7. Start the frontend** *(new terminal)*

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**

---

### Docker Setup

> Requires Docker Desktop to be running.

**Build and start both services:**

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

**Stop:**

```bash
docker compose down
```

> **Note:** The `data/` and `models/` folders are mounted as volumes. Train a model first before running predictions.

---

## Training

### Quick Mode

Runs 3 epochs on ~2,000 images. Takes ~2 minutes on CPU. Use this to verify the pipeline is working before a full run.

**CLI:**
```bash
python src/train.py --quick
```

**UI:** Open the Train tab → tick **⚡ Quick mode** → Start Training.

---

### Full Training

Trains on all 25,000 images with:
- Data augmentation (rotation, flip, zoom, shift)
- BatchNormalization + Dropout for regularization
- EarlyStopping (patience=5) — stops automatically when validation loss plateaus
- ModelCheckpoint — saves only the best weights

```bash
python src/train.py --epochs 30
```

Expected results after full training:

| Metric | Value |
|--------|-------|
| Validation accuracy | ~85–88% |
| Training time (CPU) | ~30–45 min |
| Training time (GPU) | ~3–5 min |

> **GPU note:** TensorFlow dropped native Windows GPU support after v2.10. For GPU training on Windows, use **WSL2** with `pip install tensorflow[and-cuda]`.

---

### Transfer Learning

Uses MobileNetV2 pretrained on ImageNet. Two-phase training:
1. Train only the custom head (base frozen)
2. Fine-tune the top 30 layers of the base

```bash
python src/train_transfer.py
```

Expected validation accuracy: **~95%+**

---

### Evaluate a Trained Model

Generates a confusion matrix and per-class accuracy chart, saved to `models/evaluation.png`.

```bash
python src/evaluate.py
```

---

### CLI Prediction

```bash
python src/predict.py samples/cat.116.jpg
# Prediction : Cat
# Confidence : 91.4%

# Use a different model
python src/predict.py samples/cat.116.jpg models/transfer_model.keras
```

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

### `GET /health`

Returns backend status and whether a trained model exists.

**Response**
```json
{
  "status": "ok",
  "model_ready": true
}
```

---

### `POST /predict`

Classifies an uploaded image.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | image file | JPG, PNG, or WEBP |

**Response**
```json
{
  "label": "Dog",
  "is_dog": true,
  "confidence": 87.3,
  "gradcam": "<base64-encoded PNG>"
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/cat.116.jpg"
```

---

### `POST /train`

Starts model training. Returns a **Server-Sent Events** stream of log lines.

**Request body**
```json
{
  "epochs": 30,
  "quick": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | int | 30 | Max training epochs (early stopping may end sooner) |
| `quick` | bool | false | Quick mode — 3 epochs on ~2k images |

**Response** — `text/event-stream`

Each event is a line of training output:
```
data: Epoch 1/30
data: 313/313 [===] - loss: 0.64 - accuracy: 0.61
data: ✅ Training complete! Model saved to models/
data: [DONE]
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "quick": false}'
```

---

## Model Details

### CNN Architecture

```
Input (100×100×3)
  └── Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
  └── Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
  └── Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
  └── Flatten
  └── Dense(256) → BatchNorm → Dropout(0.5)
  └── Dense(1, sigmoid)
```

| Parameter | Value |
|-----------|-------|
| Input size | 100 × 100 × 3 |
| Loss | Binary crossentropy |
| Optimizer | Adam |
| Regularization | BatchNorm + Dropout |
| Augmentation | Rotation, flip, zoom, shift, shear |

### Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image regions that most influenced the prediction. The heatmap is computed from the last convolutional layer (`conv2d_2`) and overlaid on the original image using a plasma colormap.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML framework | TensorFlow 2.21 / Keras |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + Vite 5 |
| Containerisation | Docker + Docker Compose |
| Data augmentation | Keras ImageDataGenerator |
| Evaluation | scikit-learn |
| Transfer learning | MobileNetV2 (ImageNet weights) |

---

## Documentation

In-depth technical docs live in [`docs/`](docs/):

| Doc | Content |
|-----|---------|
| [docs/backend.md](docs/backend.md) | All Python files, API endpoints, data flow, Docker internals |
| [docs/model.md](docs/model.md) | CNN architecture, MobileNetV2 transfer learning, Grad-CAM, design decisions |

---

## License

MIT — see [LICENSE](LICENSE) for details.
