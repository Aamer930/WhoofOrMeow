import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_base, 'data', 'dog-vs-cat')
MODELS_DIR = os.path.join(_base, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.keras')
IMG_SIZE = 100
BATCH_SIZE = 64


def run_evaluation(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}\nRun train.py first.")
        return

    print("Loading model...")
    model = load_model(model_path)

    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42,
    )

    print("Running predictions on validation set...")
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    y_true = val_gen.classes

    class_names = ['Cat', 'Dog']

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#FAF7F2')

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], colorbar=False, cmap='YlOrBr')
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_facecolor('#FAF7F2')

    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bars = axes[1].bar(
        class_names, per_class_acc * 100,
        color=['#0891B2', '#D97706'], width=0.5, edgecolor='none'
    )
    axes[1].set_ylim(0, 110)
    axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_facecolor('#FAF7F2')
    axes[1].spines[['top', 'right']].set_visible(False)

    for bar, acc in zip(bars, per_class_acc):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{acc * 100:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12
        )

    plt.suptitle('WhoofOrMeow — Model Evaluation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(MODELS_DIR, 'evaluation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAF7F2')
    plt.show()
    print(f"\nEvaluation chart saved to {out_path}")


if __name__ == '__main__':
    run_evaluation()
