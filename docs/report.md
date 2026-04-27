# WhoofOrMeow — Technical Project Report

**Arab Academy for Science, Technology and Maritime Transport**
**Faculty of Computer Science**

**Subject:** Introduction To Artificial Intelligence (CAI3101) — Cyber Security

**Project:** WhoofOrMeow — Cat vs Dog Image Classifier
**Author:** Ahmed Aamer
**Supervised by:** Dr. Mohamed Hamhme
**Date:** September 2024

---

## Abstract

WhoofOrMeow is an end-to-end deep learning application that classifies photographs as either cat or dog using convolutional neural networks. The system comprises a custom-trained CNN achieving approximately 85–88% validation accuracy, a MobileNetV2 transfer learning model achieving approximately 95–97% validation accuracy, a FastAPI backend serving real-time predictions and visual explanations via Grad-CAM, and a React web frontend with live training controls. The entire system is containerised with Docker Compose for reproducible deployment.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [System Architecture](#3-system-architecture)
4. [Data Preprocessing and Augmentation](#4-data-preprocessing-and-augmentation)
5. [Model 1 — Custom CNN](#5-model-1--custom-cnn)
6. [Model 2 — Transfer Learning with MobileNetV2](#6-model-2--transfer-learning-with-mobilenetv2)
7. [Grad-CAM Visual Explanations](#7-grad-cam-visual-explanations)
8. [Training Strategy](#8-training-strategy)
9. [Evaluation](#9-evaluation)
10. [Backend API](#10-backend-api)
11. [Frontend Application](#11-frontend-application)
12. [Deployment](#12-deployment)
13. [Results Summary](#13-results-summary)
14. [Challenges and Solutions](#14-challenges-and-solutions)
15. [Conclusion](#15-conclusion)
16. [References](#16-references)

---

## 1. Introduction

### 1.1 Problem Statement

Image classification is a fundamental task in computer vision: given an input image, assign it to one of a predefined set of categories. This project addresses binary image classification — distinguishing between photographs of cats and dogs — a task that is trivially simple for humans but historically challenging for machines due to the enormous variation in animal poses, lighting conditions, backgrounds, and photographic styles.

The goal was not only to build an accurate classifier but to package it as a complete, usable product: a web application where a user can upload any photograph, receive an immediate classification with a confidence score, and view a visual explanation of the model's decision.

### 1.2 Objectives

- Train a convolutional neural network from scratch on the Dogs vs Cats dataset
- Improve upon a baseline model through regularisation, augmentation, and architectural improvements
- Implement transfer learning using a pretrained model to achieve higher accuracy
- Build a real-time web interface with image upload, prediction display, and Grad-CAM heatmap
- Expose a REST API for programmatic access
- Package the full system in Docker for reproducible deployment

### 1.3 Scope

The project covers the complete machine learning pipeline: data loading, preprocessing, model design, training, evaluation, and deployment. It includes two separate model approaches (custom CNN and transfer learning), visual model interpretability (Grad-CAM), and a production-style web application.

---

## 2. Dataset

### 2.1 Source

The dataset used is the **Kaggle Dogs vs Cats** dataset, originally created for a 2013 Kaggle competition hosted by Microsoft Research. It contains 25,000 labelled photographs — 12,500 cats and 12,500 dogs — sourced from the internet.

### 2.2 Characteristics

| Property | Value |
|----------|-------|
| Total images | 25,000 |
| Cat images | 12,500 |
| Dog images | 12,500 |
| Class balance | Perfectly balanced (50/50) |
| Image format | JPEG |
| Image dimensions | Variable (not uniform) |
| Resolution range | Typically 100–500px per side |

### 2.3 Dataset Split

The dataset is split at load time using Keras `ImageDataGenerator` with `validation_split=0.2`:

| Subset | Images | Purpose |
|--------|--------|---------|
| Training | 20,000 | Model weight updates |
| Validation | 5,000 | Hyperparameter tuning, early stopping |

No separate test set is used. The validation set serves as the held-out evaluation set. The split is reproducible via `seed=42`.

### 2.4 Class Structure

Images are organised into subdirectories that Keras uses for automatic label assignment:

```
data/dog-vs-cat/
├── cat/    ← label 0
└── dog/    ← label 1
```

Keras `flow_from_directory` assigns labels alphabetically: cat=0, dog=1. The final sigmoid neuron therefore outputs P(dog).

### 2.5 Data Quality

The dataset contains a small number of corrupted or mislabelled images typical of internet-sourced data. These were not explicitly filtered, as the model is expected to be robust to a small noise level. The balanced class distribution means no class-weighting is required.

---

## 3. System Architecture

The system follows a client-server architecture with clear separation between the machine learning backend and the web frontend.

```
┌────────────────────────────────────────────────────────────────────┐
│                         User's Browser                             │
│                                                                    │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │              React Frontend  (port 5173)                   │   │
│   │                                                            │   │
│   │   Predict Tab          Train Tab                           │   │
│   │   • Drag-drop upload   • Epoch slider                      │   │
│   │   • Result card        • Quick mode toggle                 │   │
│   │   • Confidence bar     • Live log terminal (SSE)           │   │
│   │   • Grad-CAM viewer                                        │   │
│   └──────────────────────────┬─────────────────────────────────┘   │
└─────────────────────────────-│------------------------------------┘
                               │ HTTP / SSE
                               │ via Vite proxy (/api → :8000)
┌──────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend  (port 8000)                    │
│                                                                    │
│   GET  /health      →  model status check                         │
│   POST /predict     →  image → label + confidence + gradcam PNG   │
│   POST /train       →  spawn subprocess → SSE log stream          │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                  TensorFlow / Keras                         │  │
│   │                                                             │  │
│   │   model.predict()        Grad-CAM heatmap generation        │  │
│   │   ImageDataGenerator     Training subprocess (train.py)     │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   models/best_model.keras      ← loaded lazily on first request   │
│   data/dog-vs-cat/             ← mounted volume (training only)   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.1 Component Responsibilities

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| Frontend | React 18 + Vite 5 | User interface, image upload, result display, training controls |
| Backend | FastAPI + Uvicorn | REST API, model inference, training orchestration |
| ML layer | TensorFlow 2.21 / Keras | Model training, inference, Grad-CAM |
| Container | Docker Compose | Service orchestration, networking, volume mounting |

### 3.2 Communication

The React frontend communicates with the FastAPI backend exclusively through HTTP. Vite's development proxy rewrites all `/api/*` requests to `http://backend:8000/*`, meaning:
- No CORS complexity in frontend code
- No hardcoded backend URLs — works identically in local and Docker environments
- Prediction results including the Grad-CAM heatmap are returned in a single JSON response (heatmap as base64-encoded PNG), avoiding a second round-trip

---

## 4. Data Preprocessing and Augmentation

### 4.1 Normalisation

All pixel values are scaled from the range `[0, 255]` to `[0.0, 1.0]` by dividing by 255. This is applied to both training and validation data. Neural networks train more stably with small input values — large inputs cause large activations, large gradients, and slower convergence.

### 4.2 Resizing

All images are resized to **100×100 pixels** for the custom CNN. MobileNetV2 uses **224×224 pixels** (its native input resolution). Resizing is handled by Keras at generator level — no preprocessing step is required.

A resolution of 100×100 was chosen as a compromise: high enough to preserve enough detail for the model to distinguish cats from dogs, low enough to keep training time reasonable on CPU.

### 4.3 Data Augmentation

Augmentation is applied **only to the training generator**, never to the validation generator. The validation set must always see unmodified images so that accuracy measurements are comparable across epochs.

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

| Transform | Parameter | Justification |
|-----------|-----------|--------------|
| `horizontal_flip` | True | Cats and dogs are bilaterally symmetric — a mirrored cat is still a cat. Effectively doubles dataset size. |
| `rotation_range` | ±20° | Animals are photographed from many angles. Larger rotations risk making the image unrecognisable. |
| `width_shift_range` | ±20% | Animals are not always centred. Teaches the model position invariance. |
| `height_shift_range` | ±20% | Same reasoning as width shift. |
| `zoom_range` | ±20% | Animals appear at varying distances from the camera. |
| `shear_range` | ±20% | Simulates perspective distortion from off-axis photography. |

Each image is randomly transformed independently per batch — the model never sees the same exact image twice, making overfitting significantly harder.

---

## 5. Model 1 — Custom CNN

### 5.1 Architecture Overview

The custom CNN is a three-block convolutional network followed by a fully connected classifier. Each block follows the pattern: **Convolution → Batch Normalisation → Max Pooling → Dropout**.

```
Input (100 × 100 × 3)
        │
        ▼
┌─────────────────────────────┐
│  Conv2D(32, 3×3, relu)      │  Block 1
│  BatchNormalization         │
│  MaxPooling2D(2×2)          │  → (50 × 50 × 32)
│  Dropout(0.25)              │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Conv2D(64, 3×3, relu)      │  Block 2
│  BatchNormalization         │
│  MaxPooling2D(2×2)          │  → (25 × 25 × 64)
│  Dropout(0.25)              │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Conv2D(128, 3×3, relu)     │  Block 3
│  BatchNormalization         │
│  MaxPooling2D(2×2)          │  → (12 × 12 × 128)
│  Dropout(0.25)              │
└─────────────────────────────┘
        │
        ▼
    Flatten  → (18,432)
        │
        ▼
┌─────────────────────────────┐
│  Dense(256, relu)           │  Classifier
│  BatchNormalization         │
│  Dropout(0.50)              │
│  Dense(1, sigmoid)          │  → P(dog) ∈ [0, 1]
└─────────────────────────────┘
```

### 5.2 Design Decisions

#### Filter Progression: 32 → 64 → 128

Filter count doubles with each block. This is a standard convention established by VGGNet (Simonyan & Zisserman, 2014). Early convolutional layers detect many simple patterns (edges, colour gradients) — they need many filters but each filter is simple. Later layers detect complex combinations — they need even more filters to represent the richer feature space. Doubling provides enough capacity without excessive parameter growth.

#### Kernel Size: 3×3

The 3×3 kernel is the industry standard for image classification CNNs. Two stacked 3×3 convolutions cover the same 5×5 receptive field as a single 5×5 convolution, but with fewer parameters (18 vs 25 per channel) and an additional non-linearity (activation function) between them, making the network more expressive.

#### Batch Normalisation

Applied after every convolutional and dense layer except the final output. Batch Normalisation normalises activations to zero mean and unit variance across each mini-batch, then applies learned scale (γ) and shift (β) parameters. Benefits:
- Higher stable learning rates (faster convergence)
- Reduced sensitivity to weight initialisation
- Mild regularisation effect (mini-batch statistics introduce noise)
- Significantly reduces the number of epochs required to converge

#### Dropout

**Rate 0.25 in convolutional blocks:** Conservative rate. During training, 25% of feature map values are randomly zeroed. The network must learn to classify correctly even with random feature dropout, forcing redundant representations.

**Rate 0.50 in the dense layer:** Higher rate in fully connected layers is standard practice — dense layers have many more connections than conv layers and are the primary source of overfitting. 50% dropout forces the classifier to be robust to the loss of half its inputs.

#### Max Pooling (2×2)

Halves spatial dimensions after each block: 100 → 50 → 25 → 12. Benefits:
- Reduces parameter count in subsequent layers
- Introduces local translation invariance
- Forces learning of increasingly abstract representations

#### ReLU Activation

Rectified Linear Unit: `f(x) = max(0, x)`. Chosen over sigmoid/tanh for hidden layers because:
- No vanishing gradient problem for positive values
- Computationally simple
- Empirically outperforms alternatives for deep networks (Glorot et al., 2011)

#### Sigmoid Output

The output layer uses a single neuron with sigmoid activation, producing P(dog) ∈ [0, 1]. This is the standard formulation for binary classification. Threshold at 0.5: above → Dog, below → Cat.

### 5.3 Loss Function

**Binary crossentropy:**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Where $y_i \in \{0, 1\}$ is the true label and $\hat{y}_i$ is the predicted probability. Binary crossentropy heavily penalises confident wrong predictions, which pushes the model to be both accurate and well-calibrated. Mean Squared Error was considered but rejected — its gradient vanishes when sigmoid outputs saturate (near 0 or 1), slowing learning.

### 5.4 Optimiser

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2015) with default learning rate 1e-3.

Adam maintains exponentially decaying averages of past gradients (first moment, $m_t$) and squared gradients (second moment, $v_t$) to compute per-parameter adaptive learning rates. This makes it robust to sparse gradients and requires little learning rate tuning, making it the de facto standard for CNN training.

---

## 6. Model 2 — Transfer Learning with MobileNetV2

### 6.1 Motivation

Training a CNN from scratch on 25,000 images produces reasonable accuracy (~85–88%) but requires the model to learn everything about visual features from the dataset alone. Transfer learning offers a better approach: start from a model already trained on ImageNet (1.28M images, 1000 classes) that has already learned rich, general visual representations — edges, textures, shapes, object parts — and adapt them to cats and dogs.

### 6.2 Why MobileNetV2

Several architectures were evaluated as candidates:

| Architecture | Parameters | ImageNet Top-1 | Training Speed |
|-------------|-----------|---------------|---------------|
| VGG16 | 138M | 71.3% | Slow |
| ResNet50 | 25M | 74.9% | Moderate |
| InceptionV3 | 23M | 77.9% | Moderate |
| **MobileNetV2** | **3.4M** | **71.8%** | **Fast** |
| EfficientNetB0 | 5.3M | 77.1% | Moderate |

MobileNetV2 (Howard et al., 2018) was selected for its exceptional parameter efficiency. It uses **depthwise separable convolutions** — each standard convolution is factorised into a depthwise convolution (spatial filtering applied independently per input channel) followed by a pointwise convolution (1×1 conv for channel mixing). This achieves similar accuracy to standard convolutions at approximately 8-9× lower computational cost.

Given that the project runs on CPU (no GPU support in TensorFlow ≥ 2.11 on native Windows), minimising computational cost was a significant factor in the architecture selection.

### 6.3 Architecture

```
Input (224 × 224 × 3)
        │
        ▼
┌──────────────────────────────────┐
│  MobileNetV2 Base (frozen)       │
│  154 layers                      │
│  Pretrained on ImageNet          │
│  Output: (7 × 7 × 1280)         │
└──────────────────────────────────┘
        │
        ▼
  GlobalAveragePooling2D  → (1280,)
        │
        ▼
┌──────────────────────────────────┐
│  Dense(256, relu)                │
│  BatchNormalization              │
│  Dropout(0.5)                    │
│  Dense(1, sigmoid)               │
└──────────────────────────────────┘
```

**GlobalAveragePooling vs Flatten:** The final MobileNetV2 feature map is `(7, 7, 1280)`. Flattening produces 62,720 values, creating a massive Dense layer with high overfitting risk. GlobalAveragePooling averages each of the 1,280 feature maps spatially to a single value, producing a compact 1,280-dimensional representation with far fewer parameters.

### 6.4 Two-Phase Training Protocol

#### Phase 1 — Feature extraction (10 epochs, lr=1e-4)

```python
base_model.trainable = False
```

The MobileNetV2 base is completely frozen. Only the custom classification head is trained. This prevents the large, random gradients from the untrained head from flowing backwards and corrupting the carefully pretrained base weights during early training when the head's outputs are essentially random.

After 10 epochs, the custom head has converged to a state where it effectively uses the MobileNetV2 features for cat/dog classification. Validation accuracy typically reaches 90–93%.

#### Phase 2 — Fine-tuning (10 epochs, lr=1e-5)

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
```

The top 30 layers of MobileNetV2 are unfrozen and trained at a learning rate 10× lower than Phase 1. The reduced learning rate is critical — the pretrained weights are already high quality and only need subtle adjustments to specialise for cats and dogs. Large learning rates would destroy the ImageNet features.

Early layers (frozen throughout) detect universal features like edges and colour gradients that are optimal for all visual tasks. Late layers (unfrozen in Phase 2) detect increasingly task-specific features that benefit from adaptation to the cat/dog domain.

---

## 7. Grad-CAM Visual Explanations

### 7.1 Motivation

Model accuracy alone is insufficient for understanding classifier behaviour. A model achieving 87% accuracy on a validation set may still be "right for the wrong reasons" — classifying dogs correctly because they tend to appear on grass backgrounds, rather than because it has learned canine features. Grad-CAM addresses this by revealing which spatial regions of the input image drove the model's prediction.

### 7.2 Method

Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017) produces a class-discriminative localisation map for any convolutional layer.

**Algorithm:**

1. Select a target convolutional layer (the final conv layer, `conv2d_2`, which has the richest semantic content)

2. Build a sub-model that outputs both the target layer's feature maps and the final prediction simultaneously

3. Forward-pass the input image with `tf.GradientTape` recording all operations

4. Compute the gradient of the predicted class score with respect to each spatial location in the target feature maps:
$$\frac{\partial y^c}{\partial A^k_{ij}}$$

5. Global-average-pool the gradients to get per-channel importance weights:
$$\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

6. Compute the weighted combination of feature maps and apply ReLU:
$$L^c_{Grad-CAM} = \text{ReLU}\left(\sum_k \alpha^c_k A^k\right)$$

7. Resize to input dimensions and normalise to [0, 1]

8. Overlay on the original image using a **plasma colormap** (dark purple = low attention, yellow = high attention)

### 7.3 Target Layer Selection

`conv2d_2` is the final convolutional layer before the classifier. It was chosen because:
- At this depth, filters respond to complex semantic features (faces, body shapes) rather than low-level edges
- Its spatial resolution (12×12 after three max-pool operations) is sufficient to localise meaningful image regions
- Earlier layers produce noisier, less interpretable Grad-CAM maps because their features are too generic

### 7.4 Colormap: Plasma

The plasma colormap is perceptually uniform — equal steps in data value correspond to equal perceived differences in colour. The more common jet colormap has artificial bright bands that create false visual emphasis on unimportant regions. Plasma also has reasonable accessibility for the most common forms of colour blindness, unlike jet.

---

## 8. Training Strategy

### 8.1 Callbacks

Two callbacks are applied during all training runs:

#### EarlyStopping

```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

Monitors validation loss. If it fails to improve for 5 consecutive epochs, training halts and the weights are restored to the epoch with the lowest validation loss. This prevents overfitting from continued training past the optimal point and avoids wasting compute.

#### ModelCheckpoint

```python
ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
)
```

Saves the model only when validation accuracy improves. The saved model always contains the best weights seen during training, not the final epoch's weights (which may have overfit).

### 8.2 Quick Mode

A `--quick` flag reduces training to 3 epochs on approximately 2,000 images. This completes in approximately 2 minutes on CPU and serves two purposes:
- **Pipeline verification** — confirms the full training, saving, and loading pipeline works correctly before committing to a full training run
- **Development testing** — allows rapid iteration on code changes without waiting for full training

### 8.3 Hardware Considerations

TensorFlow dropped native Windows GPU support after version 2.10. The project's target machine (Windows with NVIDIA GeForce RTX 5060) therefore trains on CPU despite having a capable GPU. Estimated training times:

| Mode | Hardware | Time |
|------|----------|------|
| Quick (3 epochs, 2k images) | CPU | ~2 minutes |
| Full (30 epochs, 20k images) | CPU | ~30–45 minutes |
| Full (30 epochs, 20k images) | GPU via WSL2 | ~3–5 minutes |

For GPU-accelerated training on Windows, the recommended path is WSL2 (Windows Subsystem for Linux) with `pip install tensorflow[and-cuda]`.

---

## 9. Evaluation

### 9.1 Primary Metric: Accuracy

Validation accuracy is the primary metric. With a perfectly balanced dataset (50% cat, 50% dog), accuracy is a reliable metric — a naive classifier that always predicts "dog" would achieve exactly 50%, making improvements meaningful.

### 9.2 Additional Metrics

The `evaluate.py` script generates:

- **Confusion matrix** — shows the four outcome categories: True Positives (dog predicted as dog), True Negatives (cat predicted as cat), False Positives (cat predicted as dog), False Negatives (dog predicted as cat)
- **Per-class accuracy** — reveals whether the model performs equally well on both classes or has a systematic bias toward one
- **Precision** — of all images predicted as dog, what fraction actually are dogs
- **Recall** — of all actual dogs, what fraction were correctly identified
- **F1-score** — harmonic mean of precision and recall, robust to class imbalance

### 9.3 Expected Results

| Model | Val Accuracy | Training Time (CPU) |
|-------|-------------|-------------------|
| Baseline (original, 5 epochs) | ~79.6% | ~3 min |
| Custom CNN (improved) | ~85–88% | ~30–45 min |
| MobileNetV2 (transfer learning) | ~95–97% | ~15–20 min |

### 9.4 Overfitting Analysis

The original baseline model showed significant overfitting: training accuracy ~88% vs validation accuracy ~80% — an 8% gap. The improved model reduces this gap to approximately 3–5% through:
- **Data augmentation** — prevents memorisation of training images
- **Batch Normalisation** — regularisation effect from mini-batch statistics
- **Dropout** — prevents co-adaptation of neurons
- **EarlyStopping** — stops training before the gap widens further

---

## 10. Backend API

### 10.1 Technology Choice: FastAPI

FastAPI was chosen over alternatives (Flask, Django) for the following reasons:

| Feature | FastAPI | Flask |
|---------|---------|-------|
| Async support | Native | Via extensions |
| Automatic OpenAPI docs | Yes (`/docs`) | No |
| Request validation | Pydantic (built-in) | Manual |
| Performance | High (Starlette/Uvicorn) | Moderate |
| SSE support | StreamingResponse | Requires extensions |

The Server-Sent Events (SSE) streaming response for training logs was a key requirement. FastAPI's `StreamingResponse` with `media_type="text/event-stream"` implements SSE natively without additional libraries.

### 10.2 Endpoints

#### `GET /health`
Lightweight status check. Returns whether the backend is running and whether a trained model file exists on disk. Used by the frontend to display an appropriate message when no model has been trained yet.

#### `POST /predict`
Accepts a multipart image upload. The image is decoded in memory (no disk writes), resized, normalised, and passed to the model. The prediction score and Grad-CAM heatmap are returned in a single JSON response. The heatmap is base64-encoded so it can be rendered directly in the frontend as an `<img src="data:image/png;base64,...">` without a separate HTTP request.

#### `POST /train`
Spawns `train.py` as a subprocess and streams its stdout as SSE. Training is isolated to a subprocess rather than running in-process to avoid blocking the API server — TensorFlow training is synchronous and would freeze all other endpoints for the duration of training. The `_model` global is reset to `None` on training completion, forcing the next prediction request to reload the newly saved weights.

### 10.3 Model Loading Strategy

The TensorFlow model is loaded lazily — on the first prediction request, not at server startup. This design:
- Keeps server startup time fast regardless of model file size
- Allows the server to start successfully even if no model has been trained yet
- Allows the model to be replaced on disk (after retraining) and reloaded without restarting the server

---

## 11. Frontend Application

### 11.1 Technology Choice

| Option | Chosen | Reason |
|--------|--------|--------|
| React 18 | Yes | Component model, hooks, widespread adoption |
| Vite 5 | Yes | Fastest dev server, instant HMR, simple config |
| TypeScript | No | Unnecessary complexity for a single-developer project |
| CSS framework | No | Custom CSS gives full design control |

### 11.2 Predict Tab

The predict tab provides the primary user interaction:

1. **Drag-and-drop upload zone** — accepts JPG, PNG, WEBP. Implemented with `onDrop` and `onDragOver` native browser events, no library required.
2. **Preview** — the uploaded image is previewed immediately using `URL.createObjectURL()` before the API call completes, giving instant visual feedback.
3. **Loading state** — a spinner overlay appears while the API processes the image.
4. **Result card** — displays animal emoji, label, confidence percentage, and confidence bar. Accent colour changes dynamically: amber for dog, teal for cat.
5. **Grad-CAM viewer** — the base64-encoded heatmap from the API is rendered directly as an `<img>` element.
6. **Verdict text** — changes based on confidence level (above/below 85%).

### 11.3 Train Tab

The train tab exposes training controls:

1. **Epoch slider** — range 5–50, step 5. The selected value is sent to the API as `epochs`.
2. **Quick mode checkbox** — sets `quick: true` in the API request, triggering the fast 2-minute training run.
3. **Start Training button** — initiates the API call and displays streaming logs.
4. **Live log terminal** — reads the SSE stream from `/api/train` using the Fetch API's `ReadableStream`. Each `data: ` line is appended to a dark-themed terminal-style textarea. Auto-scrolls to bottom as new lines arrive.

### 11.4 Design System

| Element | Choice | Rationale |
|---------|--------|-----------|
| Display font | Lilita One | Bold, rounded, playful — matches the project's animal theme |
| Body font | Nunito | Friendly, highly readable, complements Lilita One |
| Background | `oklch(96% 0.015 75)` | Warm cream — distinct from default white, tinted toward amber brand colour |
| Dog accent | `oklch(55% 0.16 55)` | Warm amber |
| Cat accent | `oklch(52% 0.14 220)` | Cool teal-blue |
| Color system | OKLCH | Perceptually uniform — equal lightness steps look equal |

OKLCH (Lightness, Chroma, Hue) is used throughout instead of HSL because it is perceptually uniform: two colours at the same OKLCH lightness value genuinely appear equally bright to human perception, which HSL does not guarantee.

---

## 12. Deployment

### 12.1 Docker Compose

The system is defined as two services in `docker-compose.yml`:

```yaml
services:
  backend:   # python:3.11-slim, port 8000
  frontend:  # node:20-slim, port 5173
```

The services communicate over Docker's internal network using service names as hostnames (`http://backend:8000`). The frontend proxy in Vite's config handles routing `/api` calls to the backend.

### 12.2 Volume Mounts

| Volume | Host path | Container path | Purpose |
|--------|-----------|----------------|---------|
| Dataset | `./data` | `/app/data` | Training data |
| Models | `./models` | `/app/models` | Saved model files |
| Samples | `./samples` | `/app/samples` | Test images |
| Source (backend) | `./src` | `/app/src` | Hot reload |
| Source (frontend) | `./frontend/src` | `/app/src` | Hot reload |

Source directories are mounted as volumes so that code changes take effect immediately without rebuilding the container image — the backend restarts via Uvicorn's `--reload` flag and the frontend updates via Vite's HMR (Hot Module Replacement).

### 12.3 Windows-specific: Polling

Vite's file watcher is configured with `usePolling: true`. Docker volumes on Windows (via WSL2) do not propagate `inotify` filesystem events into the container. Without polling, Vite would never detect file changes. Polling checks for file changes on a timer — slightly less efficient than event-driven watching, but fully reliable across all operating systems.

### 12.4 Running the System

```bash
# Build images and start both services
docker compose up --build

# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
# API docs: http://localhost:8000/docs
```

---

## 13. Results Summary

### 13.1 Model Performance

| Model | Val Accuracy | Parameters | Training Time (CPU) |
|-------|-------------|-----------|-------------------|
| Baseline CNN (5 epochs, no augmentation) | ~79.6% | ~1.2M | ~3 min |
| Improved CNN (augmentation + BN + Dropout) | ~85–88% | ~3.5M | ~30–45 min |
| MobileNetV2 Transfer Learning | ~95–97% | ~3.4M base + 330K head | ~15–20 min |

### 13.2 Improvement Contributions

| Improvement | Estimated Accuracy Gain |
|-------------|------------------------|
| Data augmentation | +2–3% |
| Batch Normalisation | +1–2% |
| Dropout (increased) | +1% |
| Additional conv block (2→3) | +1–2% |
| EarlyStopping with best weight restore | Prevents degradation |
| Transfer learning (MobileNetV2) | +8–12% over custom CNN |

---

## 14. Challenges and Solutions

### 14.1 Platform Path Compatibility

**Challenge:** The project was originally developed on macOS with hardcoded Unix paths (`/Users/ahmedaamer/...`). Migrating to Windows broke all file path references.

**Solution:** All scripts now compute paths relative to their own file location using `os.path.dirname(os.path.abspath(__file__))`. This produces absolute paths that are correct regardless of operating system, working directory, or execution context (local, Docker container).

### 14.2 No GPU on Windows with TensorFlow ≥ 2.11

**Challenge:** TensorFlow dropped native Windows GPU support after version 2.10. The development machine has an NVIDIA RTX 5060, which sits idle during training.

**Solution:** A `--quick` mode was added (3 epochs, ~2,000 images) that completes in approximately 2 minutes on CPU, allowing rapid pipeline testing. For full-speed GPU training, WSL2 with `pip install tensorflow[and-cuda]` is the recommended path.

### 14.3 Blocking Training in API Server

**Challenge:** Running TensorFlow training inside the FastAPI process blocks the entire server — no other requests can be handled during training.

**Solution:** Training is delegated to a subprocess (`subprocess.Popen`). The API server remains responsive throughout training, and the subprocess's stdout is streamed back to the client via SSE.

### 14.4 Vite HMR in Docker on Windows

**Challenge:** Vite's default file watcher uses `inotify` (Linux filesystem events), which are not propagated through Docker volumes on Windows.

**Solution:** `usePolling: true` in `vite.config.js` switches to timer-based polling, which works reliably across all platforms.

### 14.5 IDE False Warnings

**Challenge:** VS Code showed "cannot find module tensorflow" warnings on all Python files because it was using the system Python 3.14 interpreter, which does not have the project's dependencies installed.

**Solution:** `.vscode/settings.json` configured to point to `.venv/Scripts/python.exe` and added `src/` to `python.analysis.extraPaths`. This resolves all warnings without affecting the runtime environment.

---

## 15. Conclusion

WhoofOrMeow demonstrates the full lifecycle of a machine learning project — from data loading and preprocessing through model training, evaluation, and deployment as a production-style web application.

The project shows two complementary approaches to image classification: a custom CNN trained from scratch that provides transparency into architectural decisions, and transfer learning with MobileNetV2 that achieves significantly higher accuracy with less training time by leveraging pretrained visual features.

Grad-CAM integration moves the project beyond a black-box classifier by making the model's decision process interpretable — users can see exactly which regions of their photograph the model examined.

The system architecture — FastAPI backend, React frontend, Docker Compose orchestration — reflects professional software engineering practices: separation of concerns, containerised deployment, hot-reloading development servers, and a clean REST API with automatic documentation.

### Potential Extensions

- **GPU training:** WSL2 + CUDA for full RTX 5060 utilisation
- **Multi-class classification:** Extend to breed identification (120 dog breeds, 67 cat breeds)
- **EfficientNetV2:** Higher accuracy than MobileNetV2 with similar parameter count
- **Model versioning:** Track multiple model versions, compare performance in the UI
- **Batch prediction:** Accept multiple images in a single API call
- **Progressive Web App:** Offline prediction using TensorFlow.js

---

## 16. References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks.* Advances in Neural Information Processing Systems, 25.

2. Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition.* arXiv:1409.1556.

3. Ioffe, S., & Szegedy, C. (2015). *Batch normalization: Accelerating deep network training by reducing internal covariate shift.* ICML 2015.

4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* Journal of Machine Learning Research, 15(1), 1929–1958.

5. Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR 2015.

6. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017.

7. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks.* CVPR 2018.

8. Glorot, X., Bordes, A., & Bengio, Y. (2011). *Deep sparse rectifier neural networks.* AISTATS 2011.

9. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.

10. Kaggle. (2013). *Dogs vs Cats.* https://www.kaggle.com/c/dogs-vs-cats
