import os
from pathlib import Path
from typing import Tuple
import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ideeplc.calibrate import SplineTransformerCalibration
import logging

LOGGER = logging.getLogger(__name__)
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
    LOGGER.info(f"Validation complete. Loss: {avg_loss:.4f}, Correlation: {correlation:.4f}")

    return avg_loss, correlation, predictions, ground_truth


def predict(
        model: nn.Module,
        dataloader_input: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        input_file: str,
        calibrate:  bool,
        save_results: bool
):
    """
    Load a trained model and evaluate it on test datasets.

    :param model: The trained model.
    :param dataloader_input: Test dataset loader.
    :param loss_fn: Loss function.
    :param device: Computation device.
    :param input_file: Path to the input file containing peptide sequences.
    :param calibrate: If True, calibrates the results.
    :param save_results: If True, saves the evaluation results.
    :return: Loss, correlation, predictions, and ground truth values.
    """
    LOGGER.info("Starting prediction process.")

    try:
        # Validate on the primary test set
        loss, correlation, predictions, ground_truth = validate(model, dataloader_input, loss_fn, device)

        if calibrate:
            LOGGER.info("Fitting calibration model.")
            calibration_model = SplineTransformerCalibration()
            calibration_model.fit(ground_truth, predictions)
            calibrated_preds = calibration_model.transform(predictions)
            correlation_preds = np.corrcoef(calibrated_preds, ground_truth)[0, 1]

            loss_calibrated = loss_fn(torch.tensor(calibrated_preds).float().view(-1, 1), torch.tensor(ground_truth).float().view(-1, 1))
            LOGGER.info(f"Calibration Loss: {loss_calibrated.item():.4f}")
            predictions = calibrated_preds  # Use calibrated predictions for further analysis
            loss = loss_calibrated.item()
            correlation = correlation_preds
        # Save results
        if save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            input_file_name = os.path.splitext(os.path.basename(input_file))[0]
            output_path = Path("ideeplc/output") / f"{input_file_name}_predictions_{timestamp}.csv"
            if calibrate:
                data_to_save = np.column_stack((ground_truth, predictions, calibrated_preds))
                header = "ground_truth,predictions,calibrated_predictions"

            else:
                data_to_save = np.column_stack((ground_truth, predictions))
                header = "ground_truth,predictions"
            np.savetxt(output_path, data_to_save, delimiter=',', header=header, fmt='%.6f', comments='')
            LOGGER.info(f"Results saved to {output_path}")

        return loss, correlation, predictions, ground_truth

    except Exception as e:
        LOGGER.error(f"An error occurred during prediction: {e}")
        raise e

