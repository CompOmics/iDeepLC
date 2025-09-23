# Python
from pathlib import Path
from ideeplc.ideeplc_core import get_model_save_path


def test_get_model_save_path():
    """Test the get_model_save_path function."""
    pretrained_model, model_dir = get_model_save_path()
    assert isinstance(pretrained_model, Path), "Model path should be a Path object"
    assert isinstance(model_dir, Path), "Model directory should be a Path object"
    assert (
        pretrained_model.name == "pretrained_model.pth"
    ), "Model name should be 'pretrained_model.pth'"
