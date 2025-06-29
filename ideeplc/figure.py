from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from predict import evaluate_model


def make_figures(
        model: torch.nn.Module,
        dataloader_test: DataLoader,
        loss_fn: torch.nn.Module,
        model_path: str,
        save_results: bool = False,
        eval_results: tuple = None
):
    """
    Generate figures based on evaluation type.

    :param model: Trained PyTorch model.
    :param dataloader_test: DataLoader for the test set.
    :param loss_fn: Loss function for evaluation.
    :param model_path: Path to the trained model.
    :param save_results: Whether to save results as a CSV file.
    :param eval_results: Evaluation results from `evaluate_model`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set correct device

    # If eval_results is provided, use it instead of re-running evaluation
    if eval_results:
        loss_test, corr_test, output_test, y_test = eval_results
    else:
        # If eval_results is not provided, fall back to running evaluate_model (not recommended)
        loss_test, corr_test, output_test, y_test = evaluate_model(model, dataloader_test,
                                                                   loss_fn, torch.device("cpu"), model_path,
                                                                   save_results)

    # generate figures
    plot_20datasets(y_test, output_test, model_path)


def plot_20datasets(y_test, output_test, model_path):
    """Generate scatter plot for 20datasets."""
    mae_test = mean_absolute_error(y_test, output_test)
    max_value = max(output_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, output_test, c="b",
               label=f"MAE: {mae_test:.3f}, R: {np.corrcoef(y_test, output_test)[0, 1]:.3f}", s=3)
    plt.legend(loc="upper left")
    plt.xlabel("Observed Retention Time")
    plt.ylabel("Predicted Retention Time")
    dataset_name = Path(model_path).parent.name
    plt.title(f"Dataset: {dataset_name}")  # Add dataset name as title
    plt.axis("scaled")
    ax.plot([0, max_value], [0, max_value], ls="--", c=".5")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    save_path = model_path.replace("best.pth", "scatter_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
