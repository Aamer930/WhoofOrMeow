import os
import sys
import io
import base64
import subprocess

import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradcam import gradcam_for_image

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_base, 'models', 'best_model.keras')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 100

app = FastAPI(title="WhoofOrMeow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Use /train to train first.",
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": os.path.exists(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()

    data = await file.read()
    original_pil = Image.open(io.BytesIO(data)).convert("RGB")

    img = original_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img).astype(np.float32) / 255.0, axis=0)

    score = float(model.predict(img_array, verbose=0)[0][0])
    is_dog = score >= 0.5
    label = "Dog" if is_dog else "Cat"
    confidence = round((score if is_dog else 1 - score) * 100, 1)

    gradcam_pil = gradcam_for_image(img_array, original_pil, model)
    buf = io.BytesIO()
    gradcam_pil.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "label": label,
        "is_dog": is_dog,
        "confidence": confidence,
        "gradcam": gradcam_b64,
    }


class TrainRequest(BaseModel):
    epochs: int = 30
    quick: bool = False


@app.post("/train")
def train(req: TrainRequest):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    def stream():
        cmd = [sys.executable, "train.py", "--epochs", str(req.epochs)]
        if req.quick:
            cmd.append("--quick")
        proc = subprocess.Popen(
            cmd,
            cwd=SRC_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                yield f"data: {line}\n\n"
        proc.wait()
        if proc.returncode == 0:
            global _model
            _model = None  # force reload on next predict
            yield "data: ✅ Training complete! Model saved to models/\n\n"
        else:
            yield f"data: ❌ Training failed (exit code {proc.returncode})\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
