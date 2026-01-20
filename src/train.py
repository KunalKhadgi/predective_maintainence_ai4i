import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc

from prepare_data import load_and_prepare_data
from dataset import SequenceDataset
from lstm_model import LSTMModel
from transformer_model import TransformerModel


def train_and_evaluate(model, loader, epochs=5, lr=1e-3):
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


if __name__ == "__main__":
    X, y, features = load_and_prepare_data("data/ai4i2020.csv")

    dataset = SequenceDataset(X, y, seq_len=30)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Choose model
    model = TransformerModel(input_dim=len(features))
    # model = LSTMModel(input_dim=len(features))

    train_and_evaluate(model, loader)
