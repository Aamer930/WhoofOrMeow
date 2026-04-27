import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_base, 'data', 'dog-vs-cat')
MODELS_DIR = os.path.join(_base, 'models')
IMG_SIZE = 100
BATCH_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--quick', action='store_true', help='Train on 2k images for 3 epochs (smoke test)')
args = parser.parse_args()
EPOCHS = 3 if args.quick else args.epochs
BATCH_SIZE = 32 if args.quick else 64

os.makedirs(MODELS_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=42,
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=42,
)

# Quick mode: limit steps to ~2k train / 500 val images
train_steps = (2000 // BATCH_SIZE) if args.quick else None
val_steps   = (500  // BATCH_SIZE) if args.quick else None

if args.quick:
    print(f"\n⚡ Quick mode: {EPOCHS} epochs, ~2k images — smoke test only\n")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
]

print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience=5)...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
)

loss, accuracy = model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

model.save(os.path.join(MODELS_DIR, 'dog_cat_classifier.keras'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.tight_layout()
curves_path = os.path.join(MODELS_DIR, 'training_curves.png')
plt.savefig(curves_path, dpi=150)
print(f"Training curves saved to {curves_path}")
