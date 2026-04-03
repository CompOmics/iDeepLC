import os
from pathlib import Path
from typing import Tuple
import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from ideeplc.calibrate import SplineTransformerCalibration
import logging

LOGGER = logging.getLogger(__name__)


def validate(
    model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device
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
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs_batch = model(inputs.float())
            loss = loss_fn(outputs_batch, labels.float().view(-1, 1))

            total_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs_batch.cpu().numpy().flatten())
            ground_truth.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]
    LOGGER.info(
        f"Validation complete. Loss: {avg_loss:.4f}, Correlation: {correlation:.4f}"
    )

    return avg_loss, correlation, predictions, ground_truth


def predict(
    model: nn.Module,
    dataloader_input: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    input_file: str,
    calibrate: bool,
    save_results: bool,
    batch_size: int = None,
    chunk_size: int = 10000,
    dataloader_input: DataLoader = None,
    mod_features_csv: str = None,
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
    LOGGER.info(
        f"Starting prediction process with batch size {batch_size} and chunk size {chunk_size}."
    )

    all_predictions = []
    all_ground_truth = []
    total_loss = 0.0
    total_samples = 0

    calibrated_preds = None

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = (
        Path("ideeplc_output") / f"{input_file_name}_predictions_{timestamp}.csv"
    )

    try:
        if dataloader_input is not None:
            LOGGER.info("Using provided dataloader_input for prediction.")
            loss, correlation, all_predictions, all_ground_truth = validate(
                model=model,
                dataloader=dataloader_input,
                loss_fn=loss_fn,
                device=device,
            )

            if calibrate:
                LOGGER.info("Fitting calibration model.")
                calibration_model = SplineTransformerCalibration()
                calibration_model.fit(all_ground_truth, all_predictions)
                calibrated_preds = calibration_model.transform(all_predictions)

                if len(calibrated_preds) > 1 and len(all_ground_truth) > 1:
                    correlation = np.corrcoef(calibrated_preds, all_ground_truth)[0, 1]
                else:
                    correlation = np.nan

                loss_calibrated = loss_fn(
                    torch.tensor(calibrated_preds).float().view(-1, 1),
                    torch.tensor(all_ground_truth).float().view(-1, 1),
                )
                loss = loss_calibrated.item()
                return loss, correlation, calibrated_preds, all_ground_truth

            return loss, correlation, all_predictions, all_ground_truth

        if batch_size is None:
            raise ValueError("batch_size must be provided when dataloader_input is not used.")

        if save_results:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                output_path.unlink()

        for chunk_idx, (df_chunk, dataset_chunk, x_shape) in enumerate(
            data_initialize_chunked(
                csv_path=input_file,
                chunk_size=chunk_size,
                mod_features_csv=mod_features_csv,
            ),
            start=1,
        ):
            LOGGER.info(
                f"Processing chunk {chunk_idx} with {len(dataset_chunk)} entries and shape {x_shape}."
            )

            dataloader_input = DataLoader(
                dataset_chunk,
                batch_size=batch_size,
                shuffle=False,
            )

            chunk_loss, _, chunk_predictions, chunk_ground_truth = validate(
                model=model,
                dataloader=dataloader_input,
                loss_fn=loss_fn,
                device=device,
            )

            n_chunk = len(dataset_chunk)
            total_loss += chunk_loss * n_chunk
            total_samples += n_chunk

            all_predictions.extend(chunk_predictions)
            all_ground_truth.extend(chunk_ground_truth)

            if save_results:
                result_data = {
                    "sequences": df_chunk.get("seq", None),
                    "modifications": df_chunk.get("modifications", None),
                    "ground_truth": chunk_ground_truth,
                    "predictions": chunk_predictions,
                }

                result_df = pd.DataFrame(result_data)
                result_df.to_csv(
                    output_path,
                    mode="a",
                    index=False,
                    header=not output_path.exists(),
                )
                LOGGER.info(f"Chunk {chunk_idx} results appended to {output_path}")

        if total_samples == 0:
            LOGGER.error("No valid samples were processed during prediction.")
            raise ValueError("No valid samples were processed during prediction.")

        loss = total_loss / total_samples

        if len(all_predictions) > 1 and len(all_ground_truth) > 1:
            correlation = np.corrcoef(all_predictions, all_ground_truth)[0, 1]
        else:
            correlation = np.nan

        if calibrate:
            LOGGER.info("Fitting calibration model.")
            calibration_model = SplineTransformerCalibration()
            calibration_model.fit(ground_truth, predictions)
            calibrated_preds = calibration_model.transform(predictions)
            correlation_preds = np.corrcoef(calibrated_preds, ground_truth)[0, 1]

            loss_calibrated = loss_fn(
                torch.tensor(calibrated_preds).float().view(-1, 1),
                torch.tensor(ground_truth).float().view(-1, 1),
            )
            LOGGER.info(f"Calibration Loss: {loss_calibrated.item():.4f}")
            loss = loss_calibrated.item()
            correlation = correlation_preds
        # Save results
        if save_results:
            input_df = pd.read_csv(input_file)

            # Extract sequences and modifications from the input file
            sequences = input_df.get("seq", None)
            modifications = input_df.get("modifications", None)

            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            input_file_name = os.path.splitext(os.path.basename(input_file))[0]
            output_path = (
                Path("ideeplc_output")
                / f"{input_file_name}_predictions_{timestamp}.csv"
            )
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists

            result_data = {
                "sequences": sequences,
                "modifications": modifications,
                "ground_truth": ground_truth,
                "predictions": predictions,
            }

            if calibrate:
                result_data["calibrated_predictions"] = calibrated_preds

            result_df = pd.DataFrame(result_data)
            result_df.to_csv(output_path, index=False)
            LOGGER.info(f"Results saved to {output_path}")

        if calibrate:
            return loss, correlation, calibrated_preds, ground_truth
        else:
            return loss, correlation, predictions, ground_truth

    except Exception as e:
        LOGGER.error(f"An error occurred during prediction: {e}")
        raise e
