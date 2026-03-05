import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from data_preprocessing import RAW_DATA_PATH, prepare_training_data
from dataset import SequenceDataset
from model_utils import load_trained_model


def evaluate_model():
    model, artifact = load_trained_model()
    features = artifact["features"]
    scaler = artifact["scaler"]
    seq_len = artifact.get("seq_len", 30)
    batch_size = artifact.get("batch_size", 64)

    # Reload data and apply the same preprocessing
    X, y, _, _ = prepare_training_data(RAW_DATA_PATH)

    dataset = SequenceDataset(X, y, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            probs = model(xb).squeeze()
            all_probs.extend(probs.numpy())
            all_labels.extend(yb.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, preds)

    print("Evaluation on full dataset")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(all_labels, preds, zero_division=0))


if __name__ == "__main__":
    evaluate_model()

