from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.abba import ABBA


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    return (output.argmax(1) == target).sum().item()


def mse(output: torch.Tensor, target: torch.Tensor) -> float:
    return (output - target).square().sum().item()


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metric: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device,
) -> tuple[float, float]:
    loss = 0
    metric_value = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss += loss.item()
        metric_value += metric(output, target)

    return loss / len(train_loader.dataset), metric_value / len(train_loader.dataset)  # type: ignore


def test_loop(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    metric: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device,
) -> tuple[float, float]:
    loss = 0
    metric_value = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            metric_value += metric(output, target)

    return loss / len(test_loader.dataset), metric_value / len(test_loader.dataset)  # type: ignore


def forecast(
    model: torch.nn.Module,
    test_dataset: Dataset,
    auto_regressive: bool = False,
    abba: ABBA | None = None,
) -> np.ndarray:
    model.eval()

    predictions = []
    if auto_regressive:
        x = test_dataset[0][0]

        with torch.no_grad():
            for _ in range(len(test_dataset)):  # type: ignore
                y_hat = model(x.unsqueeze(0))
                if abba is not None:
                    y_hat = torch.argmax(y_hat, dim=1)
                predictions.append(y_hat.item())
                x = torch.cat((x[1:], y_hat))
    else:
        with torch.no_grad():
            for x, _ in test_dataset:
                y_hat = model(x.unsqueeze(0))
                if abba is not None:
                    y_hat = torch.argmax(y_hat, dim=1)
                predictions.append(y_hat.item())

    predictions = np.array(predictions)

    if abba is not None:
        return abba.apply_inverse_transform(predictions)

    return predictions
