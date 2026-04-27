import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

DEFAULT_MODEL = '../models/best_model.keras'
IMG_SIZE = 100  # matches train.py; use 224 for transfer model


def predict(img_path, model_path=DEFAULT_MODEL, img_size=IMG_SIZE):
    if not os.path.exists(img_path):
        print(f"Error: image not found: {img_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: model not found: {model_path}")
        print("Run train.py first to generate a model.")
        sys.exit(1)

    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    score = model.predict(img_array, verbose=0)[0][0]
    label = 'Dog' if score >= 0.5 else 'Cat'
    confidence = score if score >= 0.5 else 1 - score

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.1f}%")
    return label, float(confidence)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        print("  image_path   path to .jpg/.png image")
        print("  model_path   (optional) path to .keras model file")
        sys.exit(1)

    img_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    predict(img_path, model_path)
