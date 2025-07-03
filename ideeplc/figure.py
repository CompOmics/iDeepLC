from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime

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

    mae_test = mean_absolute_error(ground_truth, predictions)
    max_value = max(predictions)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ground_truth, predictions, c="b",
               label=f"MAE: {mae_test:.3f}, R: {np.corrcoef(ground_truth, predictions)[0, 1]:.3f}", s=3)
    plt.legend(loc="upper left")
    plt.xlabel("Observed Retention Time")
    plt.ylabel("Predicted Retention Time")
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = Path("data/output") / f"{input_file_name}_predictions_{timestamp}"
    if calibrated:
        plt.title(f"scatterplot (calibrated)")
        output_path = output_path + "_calibrated.png"
    else:
        plt.title(f"scatterplot (not calibrated)")
        output_path = output_path + ".png"
    plt.axis("scaled")
    ax.plot([0, max_value], [0, max_value], ls="--", c=".5")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.savefig(output_path, dpi=300)
    plt.show()
