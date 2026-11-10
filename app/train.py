"""Train and evaluate a compact TensorFlow/Keras CNN on malaria-cell images."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "cell_images"
MODEL_PATH = ROOT / "models" / "malaria_cnn.keras"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
CHART_PATH = ROOT / "data" / "processed" / "training_history.png"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1 / 255.0),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET}")
    train = tf.keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="training", seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="validation", seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    )
    autotune = tf.data.AUTOTUNE
    train, validation = train.prefetch(autotune), validation.prefetch(autotune)
    model = build_model()
    history = model.fit(train, validation_data=validation, epochs=EPOCHS)
    model.save(MODEL_PATH)

    labels = np.concatenate([y.numpy().ravel() for _, y in validation])
    scores = model.predict(validation, verbose=0).ravel()
    predictions = (scores >= 0.5).astype(int)
    report = classification_report(labels, predictions, target_names=["Parasitized", "Uninfected"], output_dict=True)
    metadata = {
        "model_trained": True, "architecture": "Custom CNN", "image_size": "128 x 128",
        "dataset": "NIH Malaria Cell Images", "classes": ["Parasitized", "Uninfected"],
        "validation_accuracy": round(float(report["accuracy"]) * 100, 2),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(history.history).to_csv(ROOT / "data" / "processed" / "training_metrics.csv", index=False)
    plt.figure(figsize=(7, 4)); plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation"); plt.title("CNN accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.savefig(CHART_PATH, dpi=160)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
