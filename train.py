import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Tuple
import numpy as np


def train(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
        clip_value: float = 0.25
) -> float:
    """
    Train the model for one epoch
    :param model: The model to be trained.
    :param dataloader: The DataLoader providing the training data.
    :param loss_fn: The loss function to use.
    :param optimizer: The optimizer to use.
    :param epoch: The current epoch number for logging.
    :param device: The device to train on (GPU or CPU).
    :param clip_value: Gradient clipping value (default: 0.0).
    :return:The average loss for the epoch.
    """

    current_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler()

    torch.backends.cudnn.benchmark = True

    for idx, (inputs, targets) in enumerate(dataloader):
        if device.type == 'cuda':
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        outputs = model(inputs.float())
        loss = loss_fn(outputs, targets.float().view(-1, 1))

        scaler.scale(loss).backward()

        if clip_value > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        if (idx + 1) % 2 == 0 or (idx + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        current_loss += loss.item() * inputs.size(0)

        if idx % 100 == 0:
            correlation = np.corrcoef(outputs.cpu().detach().numpy().flat, targets.cpu())
            print(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss.item():.4f}, Correlation: {correlation.min().item():.4f}")

    avg_loss = current_loss / len(dataloader.dataset)
    return avg_loss


