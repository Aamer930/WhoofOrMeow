# Backend Structure

The backend is a Python application built with **FastAPI**. It handles model inference, Grad-CAM generation, and training orchestration. All backend code lives in `src/`.

---

## File Map

```
src/
├── api.py              ← HTTP server (entry point)
├── gradcam.py          ← Grad-CAM heatmap generation
├── train.py            ← CNN training script
├── train_transfer.py   ← MobileNetV2 transfer learning
├── predict.py          ← CLI prediction utility
└── evaluate.py         ← Confusion matrix + metrics report
```

---

## `api.py` — FastAPI Server

The HTTP entry point. Exposes three endpoints consumed by the React frontend.

### How it starts

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` makes Uvicorn watch `src/` for file changes and restart automatically — useful in development and in Docker with the `src/` volume mount.

### Model loading strategy

The model is loaded **lazily** — only on the first request, not at startup. This means:
- The server starts instantly even with no trained model
- Startup does not fail if `models/best_model.keras` is missing
- After training completes, `_model` is reset to `None` so the next prediction reloads the fresh weights

```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model
```

### CORS

CORS is set to `allow_origins=["*"]` so the Vite dev server (port 5173) can call the API (port 8000) without browser blocking. In production this should be restricted to the frontend's domain.

---

### `GET /health`

Lightweight liveness check. Returns whether a trained model file exists on disk.

```json
{ "status": "ok", "model_ready": true }
```

Used by the frontend to show a warning if no model has been trained yet.

---

### `POST /predict`

Accepts a multipart image upload, runs inference, and returns the result with a Grad-CAM heatmap encoded as base64.

**Flow:**

```
Upload (multipart) → PIL decode → resize to 100×100
  → normalize [0,1] → model.predict → score
  → Grad-CAM overlay → PNG → base64
  → JSON response
```

The image is read into memory with `PIL.Image.open(io.BytesIO(data))` — no disk writes, no temp files.

The Grad-CAM PNG is base64-encoded so it can be embedded directly in the JSON response and rendered by the frontend as a `data:image/png;base64,...` `src` attribute — no second HTTP request needed.

**Response shape:**
```json
{
  "label":      "Dog",
  "is_dog":     true,
  "confidence": 87.3,
  "gradcam":    "<base64 PNG string>"
}
```

---

### `POST /train`

Starts a training subprocess and streams its stdout back to the client as **Server-Sent Events (SSE)**.

**Why a subprocess?**

TensorFlow training blocks the Python thread. Running it inside the FastAPI process would freeze the entire API server for the duration of training — no health checks, no predictions. A subprocess keeps the API responsive.

**Why SSE instead of WebSocket?**

SSE is unidirectional (server → client), which is exactly what training logs need. It works over plain HTTP, requires no handshake, and is natively supported by the browser's `EventSource` API. WebSockets add complexity with no benefit here.

**Flow:**

```
POST /train → spawn subprocess (train.py)
  → read stdout line by line
  → yield "data: <line>\n\n"   ← SSE format
  → on exit code 0: reset _model = None
  → yield "data: [DONE]\n\n"
```

**Quick mode:**

Passing `"quick": true` appends `--quick` to the subprocess command, which limits training to ~2,000 images and 3 epochs. This completes in ~2 minutes on CPU and is useful for verifying the pipeline without committing to a full training run.

---

## `gradcam.py` — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap that shows which spatial regions of the input image were most influential in the model's prediction.

### How it works

1. Build a sub-model that outputs both the **last convolutional layer's feature maps** and the **final prediction**
2. Run a forward pass with `tf.GradientTape` to record gradients
3. Compute gradients of the predicted class score with respect to the feature maps
4. Pool the gradients spatially (global average) to get per-channel importance weights
5. Weight each feature map channel by its importance and sum → heatmap
6. Apply ReLU (keep only positive activations) and normalize to `[0, 1]`
7. Resize heatmap to original image size and overlay using a plasma colormap

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d_2'):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    ...
```

**Why `conv2d_2`?**

This is the third (final) convolutional layer in the CNN. It has the highest semantic content — earlier layers detect edges and textures, the final conv layer detects dog/cat-specific features. Grad-CAM on earlier layers produces noisier, less interpretable heatmaps.

**Why plasma colormap?**

Plasma goes from dark purple (low attention) → orange/yellow (high attention). It is perceptually uniform and colorblind-friendly, unlike the more common `jet` colormap which has misleading brightness jumps.

---

## `train.py` — CNN Training

Trains the custom CNN model from scratch.

### Data pipeline

```python
ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
)
```

Two separate generators are created from the same `ImageDataGenerator`:
- `subset='training'` — gets 80% of images, with augmentation applied
- `subset='validation'` — gets 20% of images

**Why two generators instead of one?**

Augmentation must only be applied to training data. The validation set must always see the original, unmodified images so the accuracy metric is not affected by random transforms. Using the same `ImageDataGenerator` instance with two subsets ensures the 80/20 split is consistent and that augmentation only fires on the training subset.

### Callbacks

**`EarlyStopping`** — monitors `val_loss`. If it does not improve for 5 consecutive epochs, training stops and the best weights are restored. This prevents wasted compute and overfitting.

**`ModelCheckpoint`** — saves `models/best_model.keras` only when `val_accuracy` improves. The final model file always contains the best weights seen across all epochs, not just the last epoch's weights.

### Arguments

| Flag | Default | Effect |
|------|---------|--------|
| `--epochs` | 30 | Maximum epochs before stopping |
| `--quick` | off | 3 epochs, ~2k images, ~2 min on CPU |

---

## `train_transfer.py` — Transfer Learning

Uses **MobileNetV2** pretrained on ImageNet. Training happens in two phases:

**Phase 1 — Head only (10 epochs)**
The MobileNetV2 base is fully frozen (`base_model.trainable = False`). Only the custom classification head is trained. Learning rate: `1e-4`.

**Phase 2 — Fine-tuning (10 more epochs)**
The top 30 layers of the base are unfrozen and trained at a much lower learning rate (`1e-5`). Lower LR is critical here — large updates would destroy the pretrained ImageNet features.

**Why freeze first, then fine-tune?**

If you unfreeze the base immediately with a randomly initialised head, the large gradients from the untrained head flow backwards and corrupt the pretrained weights. Freezing first lets the head reach a reasonable state before the base is touched.

---

## `evaluate.py` — Evaluation

Generates a two-panel chart saved to `models/evaluation.png`:

- **Left panel:** Confusion matrix — shows true positives, false positives, true negatives, false negatives
- **Right panel:** Per-class accuracy bar chart — shows whether the model performs equally well on cats and dogs, or has a class bias

Metrics also printed to stdout:
- Precision, Recall, F1-score per class
- Overall accuracy

---

## `predict.py` — CLI Prediction

Standalone script for quick one-off predictions without starting the API server.

```bash
python src/predict.py samples/cat.116.jpg
python src/predict.py samples/dog.168.jpg models/transfer_model.keras
```

Defaults to `models/best_model.keras`. Accepts an optional second argument to target a different model file (e.g., the transfer learning model).

---

## Path Resolution

All scripts resolve paths relative to their own file location using `__file__`, not the current working directory. This means they work correctly regardless of where they are invoked from — from the project root, from `src/`, or from inside a Docker container.

```python
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_base, 'data', 'dog-vs-cat')
MODELS_DIR  = os.path.join(_base, 'models')
```

---

## Docker

Two services defined in `docker-compose.yml`:

| Service | Image | Port | Hot-reload |
|---------|-------|------|-----------|
| `backend` | `python:3.11-slim` | 8000 | `Dockerfile` + `src/` volume + uvicorn `--reload` |
| `frontend` | `node:20-slim` | 5173 | `src/` volume + Vite HMR + `usePolling: true` |

`usePolling: true` in `vite.config.js` is required on Windows because Docker volumes on Windows do not propagate inotify filesystem events — polling is the only reliable way for Vite to detect file changes inside a volume.

The frontend proxy in `vite.config.js` rewrites `/api/*` → `http://backend:8000/*`. This means the React app always calls `/api/predict` and Vite transparently forwards it to the FastAPI container — no CORS issues, no hardcoded backend URLs in frontend code.
