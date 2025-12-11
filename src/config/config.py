"""Configuration management for the forestfire prediction project"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration class for managing project settings"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._interpolate_paths()

    def _interpolate_paths(self):
        """Interpolate path variables in configuration"""
        # Load environment variables
        load_dotenv()

        # Override root path from environment if available
        if "filepath" in os.environ:
            self._config["paths"]["root"] = os.environ["filepath"]

        # Recursively interpolate string values
        self._config = self._interpolate_dict(self._config)

    def _interpolate_dict(self, d: Dict) -> Dict:
        """Recursively interpolate string values in dictionary"""
        result = {}
        for key, value in d.items():
            if isinstance(value, str):
                result[key] = self._interpolate_string(value)
            elif isinstance(value, dict):
                result[key] = self._interpolate_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._interpolate_string(v) if isinstance(v, str) else v for v in value
                ]
            else:
                result[key] = value
        return result

    def _interpolate_string(self, s: str) -> str:
        """Interpolate variables in string (e.g., ${paths.root})"""
        if "${" not in s:
            return s

        # Simple variable interpolation
        import re

        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, s)

        for match in matches:
            keys = match.split(".")
            value = self._config
            for key in keys:
                value = value.get(key, "")
                if not isinstance(value, (str, int, float)):
                    break

            if isinstance(value, (str, int, float)):
                s = s.replace(f"${{{match}}}", str(value))

        return s

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation)

        Args:
            key: Configuration key (e.g., 'model.cnn.backbone')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value by key (supports dot notation)

        Args:
            key: Configuration key (e.g., 'model.cnn.backbone')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set configuration value by key"""
        self.set(key, value)

    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"Config({self._config})"


def load_config(
    config_path: Optional[str] = None, override_config: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file (default: src/config/default.yaml)
        override_config: Dictionary to override configuration values

    Returns:
        Config object
    """
    # Default config path
    if config_path is None:
        current_dir = Path(__file__).parent
        config_path = current_dir / "default.yaml"

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Override with provided config
    if override_config:
        config_dict = _deep_update(config_dict, override_config)

    return Config(config_dict)


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep update dictionary"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict
