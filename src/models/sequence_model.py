import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS
from src.generator.config import ALL_LABELS

WINDOW = 10


def build_sequences(features_df, window=WINDOW):
    df = features_df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}
    n_features = len(FEATURE_COLUMNS)

    sequences = []
    targets = []
    session_ids = []

    # labels are training-only; at inference they are hidden, so the returned
    # targets are placeholders in that case.
    has_labels = "label" in df.columns

    for entity_id, group in df.groupby("entity_id", sort=False):
        feats = group[FEATURE_COLUMNS].values.astype(np.float32)
        if has_labels:
            labels = group["label"].map(label_to_idx).fillna(0).values
        else:
            labels = np.zeros(len(group), dtype=np.int64)
        sids = group["session_id"].values
        n = len(group)
        for idx in range(n):
            start = max(0, idx - window + 1)
            chunk = feats[start:idx + 1]
            if len(chunk) < window:
                pad = np.zeros((window - len(chunk), n_features), dtype=np.float32)
                chunk = np.vstack([pad, chunk])
            sequences.append(chunk)
            targets.append(labels[idx])
            session_ids.append(sids[idx])

    x = np.stack(sequences)
    y = np.array(targets, dtype=np.int64)
    return x, y, np.array(session_ids)


class SessionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=48, n_classes=len(ALL_LABELS)):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class SequenceAnomalyModel:
    def __init__(self, window=WINDOW, hidden_size=48, epochs=10, lr=1e-3, device="cpu"):
        self.window = window
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.scaler = None
        self.model = None

    def _scale(self, x):
        n, w, f = x.shape
        flat = x.reshape(-1, f)
        flat = self.scaler.transform(flat)
        return flat.reshape(n, w, f).astype(np.float32)

    def fit(self, x_train, y_train):
        flat = x_train.reshape(-1, x_train.shape[-1])
        self.scaler = StandardScaler().fit(flat)
        x_scaled = self._scale(x_train)

        class_counts = np.bincount(y_train, minlength=len(ALL_LABELS)).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        weights = (1.0 / class_counts)
        weights = weights / weights.sum() * len(ALL_LABELS)

        self.model = SessionGRU(input_size=x_train.shape[-1], hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))

        x_t = torch.tensor(x_scaled)
        y_t = torch.tensor(y_train)

        n = len(x_t)
        batch_size = 256
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                out = self.model(x_t[idx])
                loss = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
        return self

    def predict_proba(self, x):
        self.model.eval()
        x_scaled = self._scale(x)
        with torch.no_grad():
            logits = self.model(torch.tensor(x_scaled))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs
