"""
Compute and report per-class precision/recall/F1 and confusion matrices for
both MediScan AI classifiers (issue #2).

This script does NOT contain any results itself — it's a runnable scaffold.
Point it at a held-out test directory (ImageFolder-style: one subfolder per
class) and it will:
  1. Load the model from Hugging Face (same models used in app.py)
  2. Run inference over the test set
  3. Print a full classification report (per-class precision/recall/F1/support)
  4. Save a confusion matrix image to outputs/

Usage:
    python eval_per_class_metrics.py --task brain --test_dir /path/to/brain_test
    python eval_per_class_metrics.py --task chest --test_dir /path/to/chest_test

Once run, paste the printed classification_report table (and the confusion
matrix image) into the README's "Model Evaluation" section, replacing the
"not yet reported" placeholder — see issue #2.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

HF_REPO = "neural-shubh/mediscan-models"

TASK_CONFIG = {
    "brain": {
        "hf_filename": "brain_model_final.keras",
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "img_size": (224, 224),
        "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
    },
    "chest": {
        "hf_filename": "chest_model_final.keras",
        "class_names": ["NORMAL", "PNEUMONIA"],
        "img_size": (224, 224),
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
}


def load_model(task):
    cfg = TASK_CONFIG[task]
    weights_path = hf_hub_download(repo_id=HF_REPO, filename=cfg["hf_filename"])
    return tf.keras.models.load_model(weights_path)


def load_test_set(test_dir, cfg):
    """ImageFolder-style loader: test_dir/<class_name>/*.jpg"""
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=cfg["class_names"],
        image_size=cfg["img_size"],
        shuffle=False,
        batch_size=32,
    )
    return ds


def evaluate(task, test_dir, output_dir="outputs"):
    cfg = TASK_CONFIG[task]
    model = load_model(task)
    ds = load_test_set(test_dir, cfg)

    y_true, y_pred = [], []
    for batch_x, batch_y in ds:
        batch_x = cfg["preprocess"](tf.cast(batch_x, tf.float32))
        preds = model.predict(batch_x, verbose=0)

        if preds.shape[-1] == 1:
            batch_pred = (preds.ravel() > 0.5).astype(int)
        else:
            batch_pred = np.argmax(preds, axis=1)

        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(batch_pred.tolist())

    print(f"\n=== {task.upper()} classifier — per-class report ===\n")
    print(
        classification_report(
            y_true, y_pred, target_names=cfg["class_names"], digits=3
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cfg["class_names"])))
    ax.set_yticks(range(len(cfg["class_names"])))
    ax.set_xticklabels(cfg["class_names"], rotation=45, ha="right")
    ax.set_yticklabels(cfg["class_names"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{task} classifier — confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{task}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["brain", "chest"], required=True)
    parser.add_argument(
        "--test_dir",
        required=True,
        help="ImageFolder-style directory: test_dir/<class_name>/*.jpg",
    )
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    evaluate(args.task, args.test_dir, args.output_dir)
