"""Reproducible tabular baselines for the Table 7 comparison."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 42
    hidden_dims: Tuple[int, ...] = (128, 64)
    latent_dim: int = 128
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 200
    patience: int = 10
    scheduler_patience: int = 5
    gamma: float = 2.0
    ldam_max_margin: float = 0.5
    ldam_scale: float = 30.0
    deferred_reweight_epoch: int = 160
    smote_penalty: float = 0.1
    smote_k_neighbors: int = 5
    validation_fraction: float = 0.2

    def to_dict(self) -> Dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(256, 128, 64)):
        super().__init__()
        encoder = []
        previous = input_dim
        for width in hidden_dims:
            encoder.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        encoder.extend((nn.Linear(previous, latent_dim), nn.ReLU()))
        decoder = [nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU(), nn.Linear(hidden_dims[-1], input_dim)]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)


def _class_counts(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float32)
    return torch.tensor(counts, dtype=torch.float32)


class LDAMLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, max_margin=0.5, scale=30.0, reweight=None):
        super().__init__()
        margins = 1.0 / torch.sqrt(torch.sqrt(counts.clamp_min(1.0)))
        margins = margins * (max_margin / margins.max())
        self.register_buffer("margins", margins)
        self.scale = scale
        self.reweight = reweight

    def forward(self, logits, targets):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        adjusted = logits - index.float() * self.margins[targets].unsqueeze(1)
        loss = F.cross_entropy(self.scale * adjusted, targets, weight=self.reweight)
        return loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, temperature=1.0):
        super().__init__()
        self.register_buffer("log_counts", torch.log(counts.clamp_min(1.0)))
        self.temperature = temperature

    def forward(self, logits, targets):
        return F.cross_entropy(logits / self.temperature + self.log_counts, targets)


def _smote_latent(z: np.ndarray, y: np.ndarray, seed: int, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    synthetic = [z]
    labels = [y]
    for cls, count in zip(classes, counts):
        if count >= target or count < 2:
            continue
        class_z = z[y == cls]
        neighbors = NearestNeighbors(n_neighbors=min(k_neighbors + 1, count)).fit(class_z)
        indices = neighbors.kneighbors(class_z, return_distance=False)[:, 1:]
        generated = []
        for _ in range(target - count):
            source = rng.integers(len(class_z))
            partner = indices[source, rng.integers(indices.shape[1])]
            generated.append(class_z[source] + rng.random() * (class_z[partner] - class_z[source]))
        synthetic.append(np.asarray(generated, dtype=np.float32))
        labels.append(np.full(target - count, cls, dtype=y.dtype))
    return np.concatenate(synthetic), np.concatenate(labels)


def _batch_metrics(y_true, logits):
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_score": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "y_true": y_true.tolist(),
        "y_pred": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }
    try:
        result["auc_roc"] = float(roc_auc_score(y_true, probabilities[:, 1])) if probabilities.shape[1] == 2 else float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
    except ValueError:
        result["auc_roc"] = float("nan")
    return result


def train_classifier(X_train, y_train, X_valid, y_valid, method: str, config: BaselineConfig, device="cpu",
                       validation_fraction: float = 0.2, validation_seed: int = 0):
    seed_everything(config.seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.int64)
    num_classes = int(max(y_train.max(), y_valid.max()) + 1)

    # Split a held-out selection fold from the training fold. Early stopping,
    # the learning-rate schedule, and checkpoint choice use this selection fold
    # only; the reporting fold (X_valid / y_valid) is never consulted until the
    # final evaluation, so the reported Accuracy / AUC / F1 are unbiased test
    # estimates under the same stratified 5-fold protocol.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=validation_seed)
    fit_idx, select_idx = next(splitter.split(X_train, y_train))
    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    X_select, y_select = X_train[select_idx], y_train[select_idx]

    if method == "deepsmote":
        pretraining = EncoderDecoder(X_fit.shape[1], config.latent_dim).to(device)
        optimizer = torch.optim.Adam(pretraining.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        x_tensor = torch.as_tensor(X_fit, device=device)
        for _ in range(min(config.epochs, 100)):
            optimizer.zero_grad()
            latent, reconstruction = pretraining(x_tensor)
            loss = F.mse_loss(reconstruction, x_tensor) + config.smote_penalty * latent.pow(2).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            latent = pretraining.encoder(x_tensor).cpu().numpy()
        latent, y_fit = _smote_latent(latent, y_fit, config.seed, config.smote_k_neighbors)
        with torch.no_grad():
            X_fit = pretraining.decoder(torch.as_tensor(latent, dtype=torch.float32, device=device)).cpu().numpy()
        method = "mlp"

    model = TabularMLP(X_fit.shape[1], num_classes, config.hidden_dims, config.dropout).to(device)
    counts = _class_counts(y_fit, num_classes).to(device)
    if method == "ldam":
        criterion = LDAMLoss(counts, config.ldam_max_margin, config.ldam_scale).to(device)
    elif method == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(counts).to(device)
    elif method == "mlp_focal":
        criterion = FocalLoss(counts, config.gamma).to(device)
    elif method == "mlp":
        # DeepSMOTE trains a standard classifier on the balanced augmented set,
        # so its downstream loss is plain cross-entropy (not focal loss).
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError(f"Unknown baseline: {method}")

    train_x = torch.as_tensor(X_fit, device=device)
    train_y = torch.as_tensor(y_fit, device=device)
    select_x = torch.as_tensor(X_select, device=device)
    select_y = torch.as_tensor(y_select, device=device)
    report_x = torch.as_tensor(X_valid, device=device)
    report_y = torch.as_tensor(y_valid, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=config.scheduler_patience)
    best_state, best_loss, stale = None, float("inf"), 0
    generator = torch.Generator().manual_seed(config.seed)

    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator, device=device)
        batches = list(permutation.split(config.batch_size))
        # BatchNorm requires at least two observations. Merge a singleton tail
        # into the preceding batch so that no training sample is discarded.
        if len(batches) > 1 and batches[-1].numel() == 1:
            batches[-2] = torch.cat((batches[-2], batches[-1]))
            batches.pop()
        losses = []
        if method == "ldam" and epoch == config.deferred_reweight_epoch:
            criterion.reweight = (counts.sum() / (num_classes * counts)).to(device)
        for batch in batches:
            optimizer.zero_grad()
            loss = criterion(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        select_loss = criterion(model(select_x), select_y).item()
        scheduler.step(select_loss)
        if select_loss < best_loss - 1e-6:
            best_loss, stale, best_state = select_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(report_x)
    metrics = _batch_metrics(y_valid, logits)
    return metrics, {"epochs_trained": epoch + 1, "selection_split": float(len(select_idx)) / float(max(len(y_train), 1)), "config": config.to_dict()}


class FocalLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, gamma=2.0):
        super().__init__()
        weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        return ((1 - torch.exp(-ce)).pow(self.gamma) * ce).mean()
