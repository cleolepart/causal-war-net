import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.wwi import WWIDataset, collate_wwi
from src.models.risk_model import WWIRiskModel


def train_one_epoch(
    model: WWIRiskModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(batch)
        loss = loss_fn(logits, batch["y"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: WWIRiskModel,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    model.eval()

    correct = 0
    total = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(batch)
        preds = logits.argmax(dim=1)

        correct += (preds == batch["y"]).sum().item()
        total += batch["y"].numel()

    return correct / max(1, total)
