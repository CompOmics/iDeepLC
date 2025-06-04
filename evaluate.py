from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device
) -> Tuple[float, float, list, list]:
    """
    Validate the model on a given dataset.
    :param model: The trained model.
    :param dataloader: The DataLoader providing the validation/test data.
    :param loss_fn: The loss function to use.
    :param device: The device to train on (GPU or CPU).
    :return: Average loss, correlation coefficient, predictions, and ground truth values.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    outputs, targets = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs_batch = model(inputs.float())
            loss = loss_fn(outputs_batch, labels.float().view(-1, 1))

            total_loss += loss.item() * inputs.size(0)
            outputs.extend(outputs_batch.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    correlation = np.corrcoef(outputs, targets)[0, 1]

    return avg_loss, correlation, outputs, targets


def evaluate_model(
        model: nn.Module,
        dataloader_test: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        model_path: str,
        save_results: bool = True
):
    """
    Load a trained model and evaluate it on test datasets.

    :param model: The trained model.
    :param dataloader_test: Test dataset loader.
    :param loss_fn: Loss function.
    :param device: Computation device.
    :param model_path: Path to the trained model.
    :param save_results: If True, saves the evaluation results.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Ensure model is on correct device

    # Validate on the primary test set
    loss_test, corr_test, output_test, y_test = validate(model, dataloader_test, loss_fn, device)
    print(f'Test Loss: {loss_test:.4f}, Correlation: {corr_test:.4f}')

    # Save results
    if save_results:
        filename = model_path.replace('.pth', '_output_results.csv')

        data_to_save = np.column_stack((y_test, output_test))
        header = "y_test,output_test"

        np.savetxt(filename, data_to_save, delimiter=',', header=header, fmt='%.6f', comments='')
        print(f"Results saved to {filename}")

    return loss_test, corr_test, output_test, y_test
