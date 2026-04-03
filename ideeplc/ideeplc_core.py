import datetime
import logging
import torch
from pathlib import Path
import sys
from torch import nn
from ideeplc.model import MyNet
from ideeplc.config import get_config
from ideeplc.data_initialize import data_initialize, get_input_shape_from_first_chunk
from ideeplc.predict import predict
from ideeplc.figure import make_figures
from ideeplc.fine_tuning import iDeepLCFineTuner
from importlib.resources import files

# Logging configuration
LOGGER = logging.getLogger(__name__)


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "ideeplc.log")
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler],
    )


setup_logging()


def get_model_save_path():
    """
    Determines the correct directory and filename for saving the model.
    Appends a timestamp to the filename to prevent overwriting.

    :return: Tuple containing model_save_path and model_dir.
    """
    timestamp = datetime.datetime.now().strftime("%m%d")
    model_dir = Path("ideeplc/models") / timestamp
    model_name = "pretrained_model.pth"

    if getattr(sys, "frozen", False):
        # If frozen (PyInstaller)
        base_path = Path(sys._MEIPASS)
        model_path = base_path / "ideeplc" / "models" / model_name
    else:
        # If normal Python environment
        model_path = files("ideeplc.models").joinpath(model_name)

    return model_path, model_dir


def main(args):
    """
    Main function that executes training/evaluation for the iDeepLC package based on the provided arguments.

    :param args: Parsed arguments from the CLI.
    """
    LOGGER.info("Starting iDeepLC prediction...")

    try:
        # Load configuration
        config = get_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chunk_size = config.get("chunk_size", 10000)
        batch_size = config["batch_size"]

        LOGGER.info(f"Using device: {device}")
        LOGGER.info(f"Loading data from {args.input}")

        # For model initialization, only inspect the first valid chunk
        x_shape = get_input_shape_from_first_chunk(
            csv_path=args.input,
            chunk_size=chunk_size,
            mod_features_csv=getattr(args, "mod_features", None),
        )

        # Initialize model
        LOGGER.info("Initializing model")
        model = MyNet(x_shape=x_shape, config=config).to(device)

        # Load pre-trained model
        LOGGER.info("Loading pre-trained model")
        pretrained_model, model_dir = get_model_save_path()
        if args.model:
            try:
                LOGGER.info(f"Using user-specified model path: {args.model}")
                pretrained_model = Path(args.model)
            except Exception as e:
                LOGGER.error(f"Invalid model path provided: {e}")
                raise e

        model.load_state_dict(
            torch.load(pretrained_model, map_location=device), strict=False
        )
        loss_function = nn.L1Loss()

        if args.finetune:
            LOGGER.info("Fine-tuning the model")

            matrix_input, _ = data_initialize(
                csv_path=args.input,
                mod_features_csv=getattr(args, "mod_features", None),
            )

            fine_tuner = iDeepLCFineTuner(
                model=model,
                train_data=matrix_input,
                loss_function=loss_function,
                device=device,
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                batch_size=batch_size,
                validation_data=None,
                validation_split=0.1,
                patience=20,
            )
            model = fine_tuner.fine_tune(layers_to_freeze=config["layers_to_freeze"])
            torch.save(model.state_dict(), "finetuned_model_.pth")

        # Prediction on provided data
        LOGGER.info("Starting prediction")
        pred_loss, pred_cor, pred_results, ground_truth = predict(
            model=model,
            loss_fn=loss_function,
            device=device,
            calibrate=args.calibrate,
            input_file=args.input,
            save_results=args.save,
            batch_size=batch_size,
            chunk_size=chunk_size,
            mod_features_csv=getattr(args, "mod_features", None),
        )
        LOGGER.info("Prediction completed.")

        # Generate Figures
        make_figures(
            predictions=pred_results,
            ground_truth=ground_truth,
            input_file=args.input,
            calibrated=args.calibrate,
            finetuned=args.finetune,
            save_results=args.save,
        )

    except Exception as e:
        LOGGER.error(f"An error occurred during execution: {e}")
        raise e
