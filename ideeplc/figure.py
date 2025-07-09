from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import datetime
import logging

LOGGER = logging.getLogger(__name__)

def make_figures(predictions: list, ground_truth: list, input_file: str, calibrated: bool = False, finetuned: bool = False, save_results: bool = True,

                 ):
    """
    Create and save scatter plot of predicted vs observed retention times.

    :param predictions: List of predicted retention times.
    :param ground_truth: List of observed retention times.
    :param input_file: Path to the input file used for predictions.
    :param calibrated: Boolean indicating if the predictions are calibrated.
    :param finetuned: Boolean indicating if the model was fine-tuned.
    :param save_results: Boolean indicating if the results should be saved to disk.

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
        status = "finetuned" if finetuned else ("calibrated" if calibrated else "not_calibrated")
        output_path = Path("ideeplc/output") / f"{input_file_name}_predictions_{timestamp}{status}.png"
        plt.title(f"scatterplot({status})\n")
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
