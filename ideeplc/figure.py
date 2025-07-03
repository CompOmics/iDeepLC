from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import datetime
import logging

LOGGER = logging.getLogger(__name__)

def make_figures(predictions: list, ground_truth: list, input_file: str, calibrated: bool = False ,save_results: bool = True,

                 ):
    """
    Create and save scatter plot of predicted vs observed retention times.

    Args:
        predictions (list): List of predicted retention times.
        ground_truth (list): List of observed retention times.
        input_file (str): Path to the input file used for predictions.
        calibrated (bool): Whether the predictions are calibrated. Default is False.
        save_results (bool): Whether to save the figure. Default is True.

    """
    try:
        mae_predictions = mean_absolute_error(ground_truth, predictions)
        max_value = max(max(ground_truth), max(predictions)) * 1.05 # Extend the max value by 5% for better visualization

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(ground_truth, predictions, c="b",
                   label=f"MAE: {mae_predictions:.3f}, R: {np.corrcoef(ground_truth, predictions)[0, 1]:.3f}", s=3)
        plt.legend(loc="upper left")
        plt.xlabel("Observed Retention Time")
        plt.ylabel("Predicted Retention Time")

        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        suffix = "_calibrated.png" if calibrated else ".png"
        output_path = Path("data/output") / f"{input_file_name}_predictions_{timestamp}{suffix}"
        plt.title(f"scatterplot({'calibrated' if calibrated else 'not calibrated'})\n")
        plt.axis("scaled")
        ax.plot([0, max_value], [0, max_value], ls="--", c=".5")
        plt.xlim(0, max_value)
        plt.ylim(0, max_value)

        if save_results:
            plt.savefig(output_path, dpi=300)

        plt.show()
        plt.close(fig)
    except Exception as e:
        LOGGER.error(f"Error in generating scatter plot: {e}")
        raise
