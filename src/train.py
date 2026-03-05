import os

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc

from data_preprocessing import RAW_DATA_PATH, prepare_training_data
from dataset import SequenceDataset
from lstm_model import LSTMModel
from transformer_model import TransformerModel


def train_and_evaluate(model, loader, epochs: int = 5, lr: float = 1e-3):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(loader):.4f}")

    # -------- Evaluation --------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            probs = model(xb).squeeze()
            all_preds.extend(probs.numpy())
            all_labels.extend(yb.numpy())

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)

    print(f"PR-AUC: {pr_auc:.4f}")


def save_model_artifact(
    model,
    features,
    scaler,
    model_type: str = "transformer",
    seq_len: int = 30,
    batch_size: int = 64,
    path: str = os.path.join("models", "model.joblib"),
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    artifact = {
        "model_state": model.state_dict(),
        "model_type": model_type,
        "input_dim": len(features),
        "features": features,
        "scaler": scaler,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }

    joblib.dump(artifact, path)
    print(f"Saved model artifact to {path}")


if __name__ == "__main__":
    # Load and preprocess data using the new pipeline
    X, y, features, scaler = prepare_training_data(RAW_DATA_PATH)

    seq_len = 30
    batch_size = 64

    dataset = SequenceDataset(X, y, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Choose model (default: Transformer to match original behavior)
    model_type = "transformer"
    model = TransformerModel(input_dim=len(features))
    # To switch to LSTM, change the following two lines:
    # model_type = "lstm"
    # model = LSTMModel(input_dim=len(features))

    train_and_evaluate(model, loader)
    save_model_artifact(
        model,
        features=features,
        scaler=scaler,
        model_type=model_type,
        seq_len=seq_len,
        batch_size=batch_size,
    )

