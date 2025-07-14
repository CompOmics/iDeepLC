# Python
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ideeplc.predict import predict
from ideeplc.model import MyNet
from ideeplc.data_initialize import data_initialize
from ideeplc.config import get_config
from ideeplc.ideeplc_core import get_model_save_path

def test_predict():
    """Test the predict function."""
    # Mock data and model
    config = get_config()
    best_model_path, model_dir, pretrained_model = get_model_save_path()

    test_csv_path = "ideeplc/example_input/Hela_deeprt.csv"  # Path to a sample test CSV file
    matrix_input, x_shape = data_initialize(csv_path=test_csv_path)
    dataloader = DataLoader(matrix_input, batch_size=config["batch_size"], shuffle=False)
    model = MyNet(x_shape=x_shape, config=config)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(pretrained_model, map_location=device), strict=False)
    loss_fn = torch.nn.L1Loss()


    pred_loss, pred_cor, pred_results, ground_truth = predict(
        model=model,
        dataloader_input=dataloader,
        loss_fn=loss_fn,
        device=device,
        calibrate=False,
        input_file=test_csv_path,
        save_results=False
    )
    assert pred_loss is not None, "Prediction loss should not be None"
    assert pred_results is not None, "Prediction results should not be None"
    assert pred_loss < 5000, "Prediction loss should be less than 5000"
    assert pred_cor > 0.9, "Prediction correlation should be greater than 0.9"

    pred_loss, pred_cor, pred_results, ground_truth = predict(
        model=model,
        dataloader_input=dataloader,
        loss_fn=loss_fn,
        device=device,
        calibrate=True,
        input_file=test_csv_path,
        save_results=False
    )

    assert pred_loss < 200, "Calibrated prediction loss should be less than 200"
    assert pred_cor > 0.95, "Calibrated prediction correlation should be greater than 0.95"

