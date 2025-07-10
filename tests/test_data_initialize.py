# Python
from pathlib import Path
from ideeplc.ideeplc_core import get_model_save_path


def test_get_model_save_path():
    """Test the get_model_save_path function."""
    model_path, model_dir, pretrained_path = get_model_save_path()
    assert isinstance(model_path, Path), "Model path should be a Path object"
    assert isinstance(model_dir, Path), "Model directory should be a Path object"
    assert isinstance(pretrained_path, str), "Pretrained path should be a string"
    assert model_path.name == "pretrained_model.pth", "Model name should be 'pretrained_model.pth'"


