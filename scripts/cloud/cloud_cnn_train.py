"""
Cloud CNN Training (CPU-only)

Behavior:
- Ask: Load prepared data (NPZ)?
  - Yes: load prepared train+test NPZ and train CNN directly
  - No : call run_cloud_filter() which internally calls data_prep_interactive.py
        and returns X_train_sel, y_train, X_test_sel, y_test (selected features only)

Fix:
- Adds PROJECT ROOT to sys.path so imports work from any run location.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------
# Ensure project root is on sys.path (so `scripts.*` imports work)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now this import will work
from scripts.cloud.cloud_bbfs_filter import run_cloud_filter

# ---------------------------------------------------------
# TensorFlow CPU (silence logs)
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    BatchNormalization, Activation, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


# ---------------- Prompts ----------------
def prompt_str(msg: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(msg).strip()
    x = input(f"{msg} [default: {default}] ").strip()
    return x if x else default


def prompt_yes_no(msg: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    x = input(f"{msg} ({default}): ").strip().lower()
    if not x:
        return default_yes
    return x in ("y", "yes", "1", "true")


def prompt_int(msg: str, default: int) -> int:
    x = input(f"{msg} [default: {default}] ").strip()
    return int(x) if x else default


def load_prepared_npz(train_npz: str, test_npz: str):
    with np.load(train_npz, allow_pickle=True) as d:
        X_train = d["X_train"].astype(np.float32)
        y_train = d["y_train"].astype(np.int32)
    with np.load(test_npz, allow_pickle=True) as d:
        X_test = d["X_test"].astype(np.float32)
        y_test = d["y_test"].astype(np.int32)
    return X_train, y_train, X_test, y_test


# ---------------- CNN model ----------------
def build_model(input_features: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_features, 1)),

        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(256, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(128, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------- Train/Eval ----------------
def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int, batch_size: int,
    plot_path: str
) -> tf.keras.Model:

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = build_model(input_features=X_train.shape[1])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8, min_lr=1e-6)
    ]

    history = model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='right')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

    st = time.time()
    y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    et = time.time()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    print("\n=== CLOUD CNN RESULTS ===")
    print(f"Prediction Time (s): {et - st:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return model


# ---------------- Main ----------------
def main():
    print("\n=== CLOUD CNN TRAINING ===")

    load_data = prompt_yes_no("Load prepared data (NPZ)?", default_yes=True)

    if load_data:
        train_npz = prompt_str("Train NPZ path:", default=os.path.expanduser("~/Jupyter/PhD/PreparedData/train_prepared.npz"))
        test_npz = prompt_str("Test NPZ path:", default=os.path.expanduser("~/Jupyter/PhD/PreparedData/test_prepared.npz"))
        X_train, y_train, X_test, y_test = load_prepared_npz(os.path.expanduser(train_npz), os.path.expanduser(test_npz))
        print("\nPrepared data loaded. (No filter applied here)")
    else:
        print("\nLoad:NO → Running BBFS filter (which calls data_prep_interactive internally) ...")
        X_train, y_train, X_test, y_test, selected, selected_names = run_cloud_filter(return_selected_data=True)
        print(f"\nUsing selected-feature dataset for CNN. Selected #features: {len(selected)}")

    print(f"\nFinal training shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

    epochs = prompt_int("Epochs?", default=100)
    batch_size = prompt_int("Batch size?", default=256)
    plot_path = prompt_str("Plot file name?", default="CNN_accuracy_loss.png")

    model = train_and_evaluate(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        plot_path=plot_path
    )

    save_model = prompt_yes_no("Save trained model?", default_yes=False)
    if save_model:
        out_dir = prompt_str("Output directory:", default=os.path.expanduser("~/Jupyter/PhD/CloudModels"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cloud_cnn_model.keras")
        model.save(out_path)
        print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()
