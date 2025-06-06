import argparse
import datetime

import torch
import wandb
from pathlib import Path
from torch import nn, optim
from iDeepLC.ideeplc.model import MyNet
from iDeepLC.ideeplc.config import get_config
from iDeepLC.ideeplc.data_initialize import data_initialize
from iDeepLC.ideeplc.train import train
from iDeepLC.ideeplc.evaluate import evaluate_model
from iDeepLC.ideeplc.figure import make_figures

def get_model_save_path(dataset_name):
    """
    Determines the correct directory and filename for saving the model.
    Appends a timestamp to the filename to prevent overwriting.

    Args:
        dataset_name (str): The dataset name.

    Returns:
        tuple: (model_save_path, model_dir)
    """
    timestamp = datetime.datetime.now().strftime("%m%d")

    model_dir = Path(f"../saved_model/20_datasets/{dataset_name}_{timestamp}")
    pretrained_path = Path(f"../saved_model/20_datasets/{dataset_name}/best.pth")
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
    data_loaders = data_initialize(dataset_name=args.dataset_name, batch_size=config["batch_size"])

    dataloader_train, dataloader_val, dataloader_test, x_shape = data_loaders

    # Initialize model
    model = MyNet(x_shape=x_shape, config=config).to(device)

    # Get model save path
    best_model_path, model_dir, pretrained_model_path = get_model_save_path(args.dataset_name)

    # Define loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if args.train:
        # Weights & Biases setup
        wandb.init(config=config, project=args.eval_type, entity="alirezak2", name=args.dataset_name, mode="online")

        model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Training loop
        best_loss = float("inf")
        best_model_state = None

        for epoch in range(config["epochs"]):
            loss_train = train(model, dataloader_train, loss_function, optimizer, epoch, device,
                               config["clipping_size"])
            loss_valid, corr_valid, _, _ = validate(model, dataloader_val, loss_function, device)

            print(
                f"Epoch {epoch}: Train Loss {loss_train:.4f}, Val Loss {loss_valid:.4f}, Correlation {corr_valid:.4f}")

            if loss_valid < best_loss:
                best_loss = loss_valid
                best_model_state = model.state_dict()

            wandb.log({"Train Loss": loss_train, "Validation Loss": loss_valid, "Validation Correlation": corr_valid})

        # Save best model
        torch.save(best_model_state, best_model_path)
        model_to_use = str(best_model_path)
        wandb.finish()
    else:
        # Load pre-trained model if training is skipped
        pretrained_model = str(pretrained_model_path)
        if not pretrained_model:
            raise FileNotFoundError(f"No pre-trained model found in {model_dir}. Please enable training first.")

        model.load_state_dict(torch.load(pretrained_model, map_location=device), strict=False)
        print(f"Loaded pre-trained model: {pretrained_model}")
        model_to_use = pretrained_model

    # Evaluate on the test set
    eval_results = evaluate_model(model=model, dataloader_test=dataloader_test, loss_fn=loss_function, device=device,
                                  model_path=model_to_use, save_results=args.save_results)

    # Generate Figures
    make_figures(model=model, dataloader_test=dataloader_test, loss_fn=loss_function, model_path=model_to_use,
                 save_results=args.save_results, eval_results=eval_results)

