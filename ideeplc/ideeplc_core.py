import argparse
import datetime
import pandas as pd
import torch
from pathlib import Path
from torch import nn, optim
from ideeplc.model import MyNet
from ideeplc.config import get_config
from ideeplc.data_initialize import data_initialize
from ideeplc.evaluate import predict


# from ideeplc.figure import make_figures

def get_model_save_path():
    """
    Determines the correct directory and filename for saving the model.
    Appends a timestamp to the filename to prevent overwriting.


    Returns:
        tuple: (model_save_path, model_dir)
    """
    timestamp = datetime.datetime.now().strftime("%m%d")
    dataset_name = 'heladeeprt'
    model_dir = Path(f"data/saved_models/{dataset_name}_{timestamp}")
    pretrained_path = f"data/saved_models/{dataset_name}/best.pth"
    model_name = f"best.pth"
    return model_dir / model_name, model_dir, pretrained_path


def main(args):
    """
    Main function that executes training/evaluation for the iDeepLC package based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments from the CLI.
    """
    # Load configuration
    config = get_config(epoch=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data
    dataloader_pred, x_shape = data_initialize(csv_path=args.csv_file, batch_size=config["batch_size"])

    # Initialize model
    model = MyNet(x_shape=x_shape, config=config).to(device)

    # Load pre-trained model
    best_model_path, model_dir, pretrained_model = get_model_save_path()
    model.load_state_dict(torch.load(pretrained_model, map_location=device), strict=False)
    model_to_use = pretrained_model
    loss_function = nn.L1Loss()

    # Prediction on provided data
    eval_results = predict(model=model, dataloader_test=dataloader_pred, loss_fn=loss_function, device=device,
                           save_results=args.save_results)

    # Generate Figures
    # make_figures(model=model, dataloader_test=dataloader_pred, loss_fn=loss_function, model_path=model_to_use,
    #              save_results=args.save_results, eval_results=eval_results)
