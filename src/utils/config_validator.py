from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging
from .config_schema import ConfigSchema

@dataclass
class NetworkConfig:
    input_dim: int
    feature_dim: int
    output_dim: int
    memory_size: int
    task_hidden_dims: list[int]
    dropout: float

@dataclass
class DataConfig:
    root_dir: Path
    train_split: float
    val_split: float
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    
    def __post_init__(self):
        if self.train_split + self.val_split >= 1.0:
            raise ValueError("Train and validation splits must sum to less than 1.0")

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    ewc_lambda: float
    optimizer: Dict[str, Any]
    
    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.ewc_lambda < 0:
            raise ValueError("EWC lambda must be non-negative")

@dataclass
class MonitoringConfig:
    tensorboard: Dict[str, Any]
    mlflow: Dict[str, Any]
    visualization: Dict[str, Any]

@dataclass
class CheckpointConfig:
    base_dir: Path
    save_frequency: int = 1
    keep_last: int = 3
    save_optimizer: bool = True
    save_metrics: bool = True

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates and processes configuration"""
    
    CURRENT_VERSION = "1.0.0"
    REQUIRED_SECTIONS = ['model', 'data', 'training', 'monitoring']
    
    def __init__(self, config: Dict[str, Any]):
        self.raw_config = config
        self.validated = None
        
    def validate(self) -> Dict[str, Any]:
        """Validate configuration using schema"""
        if self.validated is not None:
            return self.validated
            
        # Add version if missing
        config = self.raw_config.copy()
        config['version'] = config.get('version', self.CURRENT_VERSION)
        
        try:
            # Validate against schema
            validated_config = ConfigSchema(**config)
            
            # Additional validation for memory size
            memory = validated_config.model['network']['memory']
            if memory.get('size', 0) <= 0:
                raise ValueError("Memory size must be positive")
            
            self.validated = validated_config.model_dump()
            return self.validated
        except Exception as e:
            logging.error(f"Configuration validation failed: {str(e)}")
            raise ValueError(str(e))
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ConfigValidator':
        """Load and validate configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with path.open() as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse config file: {e}")
                raise
                
        return cls(config)
    
    def save_yaml(self, path: Union[str, Path]):
        """Save validated configuration to YAML file"""
        if not self.validated:
            self.validate()
            
        path = Path(path)
        with path.open('w') as f:
            yaml.safe_dump(self.validated, f, default_flow_style=False)
    
    def get_network_config(self) -> NetworkConfig:
        """Get validated network configuration"""
        if 'network' not in self.validated:
            self.validate()
        return NetworkConfig(
            input_dim=self.validated['model']['network']['input_dim'],
            feature_dim=self.validated['model']['network']['feature_dim'],
            output_dim=self.validated['model']['network']['output_dim'],
            memory_size=self.validated['model']['network']['memory']['size'],
            task_hidden_dims=self.validated['model']['network']['task_columns']['hidden_dims'],
            dropout=self.validated['model']['network']['task_columns'].get('dropout', 0.0)
        )
    
    def get_data_config(self) -> DataConfig:
        """Get validated data configuration"""
        if 'data' not in self.validated:
            self.validate()
        return DataConfig(
            root_dir=Path(self.validated['data']['datasets']['root_dir']),
            train_split=self.validated['data'].get('train_split', 0.7),
            val_split=self.validated['data'].get('val_split', 0.15),
            batch_size=self.validated['data']['dataloader']['batch_size'],
            num_workers=self.validated['data']['dataloader'].get('num_workers', 2),
            pin_memory=self.validated['data']['dataloader'].get('pin_memory', True),
            shuffle=self.validated['data']['dataloader'].get('shuffle', True)
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Get validated training configuration"""
        if self.validated is None:
            self.validate()
        return TrainingConfig(
            epochs=self.validated['training']['epochs'],
            batch_size=self.validated['training']['batch_size'],
            learning_rate=self.validated['training']['learning_rate'],
            ewc_lambda=self.validated['training'].get('ewc_lambda', 0.4),
            optimizer=self.validated['training'].get('optimizer', {'type': 'adam'})
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get and validate monitoring configuration"""
        monitoring = self.validated.get('monitoring', {})
        
        # Validate MLflow config
        mlflow_config = monitoring.get('mlflow', {})
        if not isinstance(mlflow_config, dict):
            raise ValueError("MLflow configuration must be a dictionary")
        
        # Validate database config
        db_config = monitoring.get('database', {
            'url': 'sqlite:///metrics.db',
            'echo': False
        })
        if not isinstance(db_config, dict):
            raise ValueError("Database configuration must be a dictionary")
        if 'url' not in db_config:
            raise ValueError("Database URL must be specified")
        
        return MonitoringConfig(
            tensorboard=monitoring.get('tensorboard', {'enabled': False}),
            mlflow=mlflow_config,
            visualization=monitoring.get('visualization', {})
        )
    
    def get_checkpoint_config(self) -> CheckpointConfig:
        """Get validated checkpoint configuration"""
        if self.validated is None:
            self.validate()
        
        checkpoint_config = self.validated.get('checkpoints', {
            'base_dir': 'checkpoints/',
            'save_frequency': 1,
            'keep_last': 3,
            'save_optimizer': True,
            'save_metrics': True
        })
        
        return CheckpointConfig(
            base_dir=Path(checkpoint_config['base_dir']),
            save_frequency=checkpoint_config.get('save_frequency', 1),
            keep_last=checkpoint_config.get('keep_last', 3),
            save_optimizer=checkpoint_config.get('save_optimizer', True),
            save_metrics=checkpoint_config.get('save_metrics', True)
        ) 