"""
Configuration loader and validator for the Diabetic Retinopathy pipeline.
"""

import os
import yaml
from typing import Any, Dict


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
