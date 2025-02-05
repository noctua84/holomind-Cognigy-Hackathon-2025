from typing import Dict, Any
import os
import yaml
from pathlib import Path

class ConfigLoader:
    """Configuration loader for HoloMind"""
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = {}
        
    def load_all(self) -> Dict[str, Any]:
        """Load all configuration files"""
        config_files = {
            'model': 'model_config.yaml',
            'training': 'training_config.yaml',
            'data': 'data_config.yaml',
            'logging': 'logging_config.yaml',
            'system': 'system_config.yaml'
        }
        
        for key, filename in config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.config[key] = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
                
        return self.config
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get specific configuration section"""
        if not self.config:
            self.load_all()
        return self.config.get(name, {})
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """Update configuration values"""
        if not self.config:
            self.load_all()
            
        def deep_update(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        if name in self.config:
            self.config[name] = deep_update(self.config[name], updates)
        else:
            self.config[name] = updates
            
    def save_config(self, name: str):
        """Save configuration to file"""
        if name in self.config:
            file_path = self.config_dir / f"{name}_config.yaml"
            with open(file_path, 'w') as f:
                yaml.safe_dump(self.config[name], f, default_flow_style=False) 