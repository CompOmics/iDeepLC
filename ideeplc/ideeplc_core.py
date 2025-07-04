import argparse
import datetime
import logging
import torch
from pathlib import Path
from torch import nn
from ideeplc.model import MyNet
from ideeplc.config import get_config
from ideeplc.data_initialize import data_initialize
from ideeplc.predict import predict
from ideeplc.figure import make_figures
from ideeplc.fine_tuning import iDeepLCFineTuner
# Logging configuration
LOGGER = logging.getLogger(__name__)


def get_model_save_path():
    """
    Determines the correct directory and filename for saving the model.
    Appends a timestamp to the filename to prevent overwriting.


    Returns:
        tuple: (model_save_path, model_dir)
    """
    timestamp = datetime.datetime.now().strftime("%m%d")
    dataset_name = 'proteometools'
    model_dir = Path(f"data/saved_models/{dataset_name}_{timestamp}")
    pretrained_path = f"data/saved_models/{dataset_name}/best.pth"
    model_name = "best.pth"
    return model_dir / model_name, model_dir, pretrained_path


def main(args):
    """
    Main function that executes training/evaluation for the iDeepLC package based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments from the CLI.
    """

    LOGGER.info("Starting iDeepLC prediction...")
    try:
        # Load configuration
        config = get_config(epoch=10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize data
        LOGGER.info(f"Loading data from {args.input}")
        dataloader_pred, x_shape = data_initialize(csv_path=args.input, batch_size=config["batch_size"])

        # Initialize model
        LOGGER.info("Initializing model")
        model = MyNet(x_shape=x_shape, config=config).to(device)

        # Load pre-trained model
        LOGGER.info("Loading pre-trained model")
        best_model_path, model_dir, pretrained_model = get_model_save_path()
        model.load_state_dict(torch.load(pretrained_model, map_location=device), strict=False)

        if args.finetune:
            LOGGER.info("Fine-tuning the model")
            fine_tuner = iDeepLCFineTuner(
                model=model,
                train_data=dataloader_pred,
                device=device,
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                validation_data=None,  # No validation data provided for prediction
                validation_split=0.1,
                patience=5
            )
            model = fine_tuner.fine_tune(layers_to_freeze=config["layers_to_freeze"])


        loss_function = nn.L1Loss()

        # Prediction on provided data
        LOGGER.info("Starting prediction")
        pred_loss, pred_cor, pred_results, ground_truth = predict(model=model, dataloader_test=dataloader_pred,
                                                                  loss_fn=loss_function,
                                                                  device=device,
                                                                  calibrate=args.calibrate,
                                                                  input_file=args.input, save_results=args.save_results)
        LOGGER.info(f"Prediction completed.")
        # Generate Figures
        make_figures(predictions=pred_results, ground_truth=ground_truth,
                     input_file=args.input, calibrated=args.calibrate, save_results=args.save_results)

    except Exception as e:
        LOGGER.error(f"An error occurred during execution: {e}")
        raise e
