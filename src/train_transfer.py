import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_base, 'data', 'dog-vs-cat')
MODELS_DIR = os.path.join(_base, 'models')
IMG_SIZE = 224  # MobileNetV2 native size
BATCH_SIZE = 32

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

# Frozen MobileNetV2 base
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy'],
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_transfer_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
]

# Phase 1: train head only
print("\n--- Phase 1: Training head ---")
history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks,
)

# Phase 2: fine-tune top 30 layers of base
print("\n--- Phase 2: Fine-tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['accuracy'],
)

history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks,
)

loss, accuracy = model.evaluate(validation_generator)
print(f"\nTransfer Learning Validation Accuracy: {accuracy * 100:.2f}%")

model.save(os.path.join(MODELS_DIR, 'transfer_model.keras'))

# Combine histories for plot
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_vals = history1.history['loss'] + history2.history['loss']
val_loss_vals = history1.history['val_loss'] + history2.history['val_loss']
phase2_start = len(history1.history['accuracy'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for ax, train, val, title in [
    (ax1, acc, val_acc, 'Accuracy'),
    (ax2, loss_vals, val_loss_vals, 'Loss'),
]:
    ax.plot(train, label='Train')
    ax.plot(val, label='Validation')
    ax.axvline(phase2_start, color='gray', linestyle='--', label='Fine-tune start')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()

plt.tight_layout()
curves_path = os.path.join(MODELS_DIR, 'transfer_training_curves.png')
plt.savefig(curves_path, dpi=150)
plt.show()
print(f"Training curves saved to {curves_path}")
