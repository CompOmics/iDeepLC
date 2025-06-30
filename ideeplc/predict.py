import os.path
from pathlib import Path
from typing import Tuple
import datetime
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
    predictions, ground_truth = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs_batch = model(inputs.float())
            loss = loss_fn(outputs_batch, labels.float().view(-1, 1))

            total_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs_batch.cpu().numpy().flatten())
            ground_truth.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]

    return avg_loss, correlation, predictions, ground_truth


def predict(
        model: nn.Module,
        dataloader_test: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        input_file: str,
        save_results: bool = True
):
    """
    Load a trained model and evaluate it on test datasets.

    :param model: The trained model.
    :param dataloader_test: Test dataset loader.
    :param loss_fn: Loss function.
    :param device: Computation device.
    :param input_file: Path to the input file containing peptide sequences.
    :param save_results: If True, saves the evaluation results.
    """
    
    # Validate on the primary test set
    loss, correlation, predictions, ground_truth = validate(model, dataloader_test, loss_fn, device)
    print(f'Test Loss: {loss:.4f}, Correlation: {correlation:.4f}')

    # Save results
    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = Path("data/output") / f"{input_file_name}_predictions_{timestamp}.csv"
        data_to_save = np.column_stack((ground_truth, predictions))
        header = "ground_truth,predictions"
        np.savetxt(output_path, data_to_save, delimiter=',', header=header, fmt='%.6f', comments='')
        print(f"Results saved to {output_path}")

    return loss, correlation, predictions, ground_truth
