import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_label = self.y[idx + self.seq_len]

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_label, dtype=torch.float32),
        )
