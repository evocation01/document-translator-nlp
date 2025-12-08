import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent

def load_config():
    """Loads the configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration once when the module is imported
config = load_config()
