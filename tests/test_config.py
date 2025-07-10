from ideeplc.config import get_config

def test_get_config():
    """Test the get_config function to ensure it returns a valid configuration."""
    config = get_config(epoch=5)  # Adjust epoch for testing purposes
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "epochs" in config, "Config should contain 'epoch' key"
    assert config["epochs"] == 5, "Epoch in config should match the input value"
    assert "learning_rate" in config, "'learning_rate' should be in config"

test_get_config()