# Model Documentation

This document explains the models used in WhoofOrMeow, the design decisions behind each one, and how they relate to the problem.

---

## The Problem

Binary image classification — given a photo, output one of two classes: **Cat** or **Dog**.

This is one of the classic benchmark problems in computer vision. The dataset (Kaggle Dogs vs Cats) contains 25,000 labelled photographs sourced from the internet, making it large enough to train a small CNN from scratch but also a good candidate for transfer learning.

---

## Model 1 — Custom CNN

**File:** `src/train.py`
**Saved to:** `models/best_model.keras` / `models/best_model.keras`

### Architecture

```
Input: (100, 100, 3)
│
├── Conv2D(32, 3×3, relu)
├── BatchNormalization
├── MaxPooling2D(2×2)
├── Dropout(0.25)
│
├── Conv2D(64, 3×3, relu)
├── BatchNormalization
├── MaxPooling2D(2×2)
├── Dropout(0.25)
│
├── Conv2D(128, 3×3, relu)
├── BatchNormalization
├── MaxPooling2D(2×2)
├── Dropout(0.25)
│
├── Flatten
├── Dense(256, relu)
├── BatchNormalization
├── Dropout(0.5)
│
└── Dense(1, sigmoid)        ← Output: probability of "Dog"
```

**Total parameters:** ~3.5 million

---

### Why a CNN?

Convolutional Neural Networks are the standard architecture for image classification. Unlike fully connected networks, CNNs exploit the **spatial structure** of images:

- **Weight sharing** — the same filter is applied across the entire image. A filter that detects ears works wherever in the image the ear appears.
- **Translation invariance** — a cat in the top-left corner is recognised the same as a cat in the bottom-right.
- **Hierarchical feature learning** — early layers detect low-level features (edges, textures), middle layers detect shapes (eyes, snouts), and deep layers detect high-level features (faces, body shapes).

A fully connected network on 100×100×3 images would have 30,000 input neurons, leading to enormous parameter counts and poor generalisation. CNNs solve this through local connectivity and pooling.

---

### Layer-by-layer reasoning

#### Convolutional blocks (×3)

Each block follows the pattern: `Conv2D → BatchNorm → MaxPool → Dropout`

**Filter progression: 32 → 64 → 128**

Filter count doubles with each block. Early layers detect many simple features (edges in many directions); later layers need more filters to represent the increasingly complex combinations of those features. Doubling is a standard heuristic — it balances expressiveness against parameter count.

**Kernel size: 3×3**

3×3 is the industry standard since VGGNet (2014). Two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution but with fewer parameters and an extra non-linearity, making the network more expressive at lower cost.

**MaxPooling(2×2)**

Halves the spatial dimensions after each conv block (100→50→25→12). This progressively reduces the number of parameters, introduces a small degree of translation invariance, and forces the network to learn increasingly abstract representations.

#### BatchNormalization

Normalises the output of each layer to have zero mean and unit variance during training. This:
- Allows higher learning rates without divergence
- Reduces sensitivity to weight initialisation
- Acts as mild regularisation (adds slight noise during training via mini-batch statistics)
- Significantly speeds up convergence — the model reaches good accuracy in fewer epochs

Without BatchNorm, training a 3-block CNN on this dataset would typically require more careful learning rate tuning and take longer to converge.

#### Dropout

Applied after each pooling layer (rate=0.25) and after the dense layer (rate=0.5).

During training, Dropout randomly zeroes out a fraction of neuron outputs. This forces the network to learn **redundant representations** — no single neuron can be relied upon, so the network builds multiple independent pathways to the same conclusion. The result is better generalisation to unseen data.

Rate=0.25 in conv layers is conservative — too much dropout early on prevents the filters from learning useful features. Rate=0.5 in the dense layer is standard — fully connected layers are the primary source of overfitting in CNNs.

**Why Dropout matters here:**

Without augmentation and dropout, the original baseline model trained to ~88% training accuracy but only ~80% validation accuracy — a clear sign of overfitting. The combination of augmentation + BatchNorm + Dropout closes this gap significantly.

#### Dense(256) head

After flattening the spatial feature maps, a 256-neuron dense layer combines them into a global prediction. 256 was chosen as a balance:
- Too small (e.g. 64): the model lacks capacity to combine complex spatial features
- Too large (e.g. 1024): increases overfitting risk and parameter count with diminishing returns for a binary task

#### Dense(1, sigmoid) output

A single neuron with sigmoid activation outputs a probability in [0, 1].
- Output ≥ 0.5 → **Dog**
- Output < 0.5 → **Cat**

This is the standard output for binary classification. The alternative — two softmax neurons — would work identically but adds unnecessary parameters.

---

### Loss function: Binary Crossentropy

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Binary crossentropy penalises confident wrong predictions heavily. If the model outputs 0.95 (very confident Dog) but the label is Cat, the loss is large. If it outputs 0.6 (somewhat confident Dog) for the same label, the loss is smaller. This pushes the model to be both accurate and calibrated.

Mean Squared Error (MSE) is sometimes used for binary classification but is a poor choice — its gradient vanishes when the sigmoid output saturates, slowing learning significantly.

---

### Optimizer: Adam

Adam (Adaptive Moment Estimation) maintains per-parameter learning rates that adapt based on gradient history. It combines:
- **Momentum** — accumulates gradients to smooth out oscillations
- **RMSProp** — scales learning rates by recent gradient magnitude

Default learning rate: `1e-3`. Adam typically requires no tuning for standard CNN training and converges faster than plain SGD on this type of problem.

---

### Data Augmentation

```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
)
```

Augmentation artificially expands the dataset by applying random transforms at training time. The model never sees the exact same image twice.

| Transform | Why it helps |
|-----------|-------------|
| `horizontal_flip` | Cats and dogs look the same mirrored. Doubles effective dataset size for free. |
| `rotation_range=20` | Animals are photographed at angles. ±20° covers most real-world variation without distorting the image too much. |
| `width/height_shift` | Animals are not always centred. Teaches position invariance. |
| `zoom_range=0.2` | Animals are photographed at varying distances. |
| `shear_range=0.2` | Handles perspective distortion from camera angle. |

**Augmentation is only applied to training data.** The validation generator uses `rescale=1./255` only. Augmenting validation data would make the accuracy metric unreliable.

---

### Expected Performance

| Metric | Baseline (original) | With improvements |
|--------|--------------------|--------------------|
| Training accuracy | ~88% | ~90%+ |
| Validation accuracy | ~80% | ~85–88% |
| Overfitting gap | ~8% | ~3–5% |

---

## Model 2 — Transfer Learning (MobileNetV2)

**File:** `src/train_transfer.py`
**Saved to:** `models/transfer_model.keras`

### What is Transfer Learning?

Transfer learning reuses a model trained on a large dataset (ImageNet — 1.2M images, 1000 classes) for a different but related task. The pretrained model has already learned powerful, general visual features — edges, textures, shapes, object parts — that are directly useful for classifying cats and dogs.

Instead of learning from scratch with 25,000 images, we start from weights that already encode rich visual knowledge and adapt them to our specific problem.

---

### Why MobileNetV2?

Several pretrained architectures were candidates:

| Model | Params | ImageNet Top-1 | Notes |
|-------|--------|---------------|-------|
| VGG16 | 138M | 71.3% | Very large, slow inference |
| ResNet50 | 25M | 74.9% | Good accuracy, moderate size |
| InceptionV3 | 23M | 77.9% | Complex, requires 299×299 input |
| **MobileNetV2** | **3.4M** | **71.8%** | **Lightweight, fast, designed for efficiency** |
| EfficientNetB0 | 5.3M | 77.1% | Strong accuracy/size ratio |

MobileNetV2 was chosen because:
1. **Small size** — 3.4M parameters fits comfortably in CPU memory and trains quickly even without a GPU
2. **Fast inference** — designed for mobile/edge deployment; predictions are near-instant
3. **Good enough accuracy** — 71.8% on 1000-class ImageNet translates to excellent performance on 2-class cat/dog
4. **Depthwise separable convolutions** — MobileNetV2's core building block is computationally cheap: it factorises a standard convolution into a depthwise convolution (spatial filtering per channel) followed by a pointwise convolution (channel mixing). This is ~8-9× fewer operations than a standard convolution at similar accuracy.

---

### Architecture

```
MobileNetV2 base (frozen)
  └── 154 layers, pretrained on ImageNet
  └── Input: (224, 224, 3)     ← MobileNetV2's native resolution
  └── Output: (7, 7, 1280)     ← feature map

Custom head (trained)
  └── GlobalAveragePooling2D   ← (7,7,1280) → (1280,)
  └── Dense(256, relu)
  └── BatchNormalization
  └── Dropout(0.5)
  └── Dense(1, sigmoid)        ← cat/dog probability
```

**Why GlobalAveragePooling instead of Flatten?**

Flattening `(7, 7, 1280)` produces a vector of 62,720 values, leading to a large Dense layer and high overfitting risk. GlobalAveragePooling averages each of the 1280 feature maps to a single value, giving a 1280-dimensional vector. This is a much more compact representation and reduces overfitting significantly.

---

### Two-phase training

#### Phase 1 — Train head only (learning rate: 1e-4)

```python
base_model.trainable = False
```

The entire MobileNetV2 base is frozen. Only the custom head's weights are updated. This prevents the randomly initialised head's large gradients from immediately corrupting the pretrained base weights.

After 10 epochs (with early stopping), the head has converged to a reasonable state. Validation accuracy typically reaches ~90-93% at this point.

#### Phase 2 — Fine-tune top layers (learning rate: 1e-5)

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
```

The top 30 layers of MobileNetV2 are unfrozen and trained at a 10× lower learning rate. The lower LR is critical — the pretrained weights are already good and only need subtle adjustments to specialise for cats and dogs. Large updates would destroy the learned features.

Only the top 30 layers are unfrozen because:
- **Early layers** (frozen) learn universal features like edges and textures that apply to all image tasks. There is no benefit in retraining them on 25k images.
- **Late layers** (unfrozen) learn task-specific features. These are the ones that need to shift from "ImageNet objects in general" to "cats and dogs specifically".

---

### Expected Performance

| Phase | Validation Accuracy |
|-------|-------------------|
| After Phase 1 (head only) | ~90–93% |
| After Phase 2 (fine-tuning) | ~95–97% |

This is a significant improvement over the custom CNN (~85-88%) using the same dataset, because MobileNetV2 arrives with 1.2M training images worth of visual knowledge.

---

## Grad-CAM: Understanding What the Model Sees

Grad-CAM is applied to the custom CNN to produce the heatmap overlay shown in the UI.

### Why interpretability matters

A model that achieves 87% accuracy is useful. A model that shows *why* it made a decision is more trustworthy, easier to debug, and more convincing in a presentation.

Common failure modes Grad-CAM can expose:
- **Background bias** — the model classifies based on grass (dogs are often outdoors) rather than the animal
- **Texture shortcut** — the model detects fur texture rather than animal shape
- **Spurious correlations** — the model latches onto collars, humans in the background, or photo framing

If Grad-CAM shows attention concentrated on the animal's face/body, the model is learning the right features. If it lights up the background or irrelevant objects, the model is using shortcuts that will fail on out-of-distribution images.

### Why the last conv layer specifically?

The final convolutional layer before the classifier (`conv2d_2`) has the best combination of:
- **Spatial resolution** — 12×12 feature maps (after 3 max-pooling steps from 100×100), enough to localise attention to meaningful regions
- **Semantic content** — at this depth, filters respond to complex features like snouts, ears, and fur patterns rather than raw edges

Earlier layers have higher resolution but lower semantic content. The gradient signal at shallow layers is noisy and produces less interpretable heatmaps.

---

## Summary: Why These Choices?

| Decision | Choice | Why |
|----------|--------|-----|
| Architecture | CNN | Spatial invariance, weight sharing, proven for images |
| Filter progression | 32→64→128 | Balance expressiveness vs parameters |
| Kernel size | 3×3 | Efficient, deep receptive field when stacked |
| Regularisation | BatchNorm + Dropout | Reduce overfitting gap (train-val ~8% → ~3%) |
| Augmentation | 6 transforms | Artificial dataset expansion, forces position invariance |
| Loss | Binary crossentropy | Correct for probabilistic binary output |
| Optimizer | Adam | Adaptive LR, fast convergence, no tuning needed |
| Transfer model | MobileNetV2 | Best accuracy/size ratio for CPU deployment |
| Grad-CAM layer | `conv2d_2` | Highest semantic content, sufficient spatial resolution |
| Colormap | Plasma | Perceptually uniform, colorblind-safe |
