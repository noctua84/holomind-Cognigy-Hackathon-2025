import pytest
from pathlib import Path
from src.utils.config_validator import ConfigValidator, NetworkConfig, DataConfig, TrainingConfig

def test_config_validation(test_config):
    """Test configuration validation"""
    validator = ConfigValidator(test_config)
    validated = validator.validate()
    
    network_config = validator.get_network_config()
    assert network_config.input_dim == 784
    assert network_config.feature_dim == 256
    
    data_config = validator.get_data_config()
    assert isinstance(data_config.root_dir, Path)
    assert 0 < data_config.train_split < 1

def test_invalid_config():
    """Test invalid configuration handling"""
    invalid_config = {
        'model': {
            'network': {
                'input_dim': -1  # Invalid input dimension
            }
        }
    }
    
    with pytest.raises(ValueError):
        ConfigValidator(invalid_config).validate()

def test_missing_required_sections():
    """Test handling of missing required sections"""
    incomplete_config = {
        'model': {}  # Missing other required sections
    }
    
    with pytest.raises(ValueError) as exc:
        ConfigValidator(incomplete_config).validate()
    assert "Field required" in str(exc.value)

def test_data_split_validation():
    """Test data split validation"""
    invalid_config = {
        'version': '1.0.0',
        'model': {
            'network': {
                'input_dim': 784,
                'feature_dim': 256,
                'output_dim': 10,
                'memory': {'size': 1000, 'feature_dim': 256},
                'task_columns': {'hidden_dims': [256, 128]}
            }
        },
        'data': {
            'datasets': {'root_dir': 'data/'},
            'preprocessing': {'normalization': {'type': 'standard'}},
            'dataloader': {
                'batch_size': 32,
                'num_workers': 2,
                'pin_memory': True,
                'shuffle': True
            },
            'train_split': 0.8,
            'val_split': 0.3,  # Sum > 1.0
            'test_split': 0.1  # Changed from 0.0 to 0.1
        },
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': {'type': 'adam'}
        },
        'monitoring': {
            'tensorboard': {'enabled': False},
            'mlflow': {'tracking_uri': 'mlruns'},
            'visualization': {'output_dir': 'viz/'}
        }
    }
    
    with pytest.raises(ValueError) as exc:
        ConfigValidator(invalid_config).validate()
    assert "Data splits must sum to less than or equal to 1.0" in str(exc.value)

def test_schema_validation():
    """Test configuration schema validation"""
    invalid_config = {
        'version': '1.0.0',
        'model': {
            'network': {
                'input_dim': 784,
                'feature_dim': 256,
                'output_dim': 10,
                'memory': {'size': 0},  # Invalid size
                'task_columns': {'hidden_dims': [256, 128]}
            }
        },
        'data': {
            'datasets': {'root_dir': 'data/'},
            'preprocessing': {'normalization': {'type': 'standard'}},
            'dataloader': {
                'batch_size': 32,
                'num_workers': 2,
                'pin_memory': True,
                'shuffle': True
            }
        },
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': {'type': 'adam'}
        },
        'monitoring': {
            'tensorboard': {'enabled': False},
            'mlflow': {'tracking_uri': 'mlruns'}
        }
    }
    
    with pytest.raises(ValueError) as exc:
        ConfigValidator(invalid_config).validate()
    assert "Memory size must be positive" in str(exc.value)

def test_yaml_roundtrip(tmp_path, test_config):
    """Test YAML loading and saving"""
    config_path = tmp_path / "config.yaml"
    
    # Save config
    validator = ConfigValidator(test_config)
    validator.save_yaml(config_path)
    
    # Load and validate
    loaded = ConfigValidator.from_yaml(config_path)
    validated = loaded.validate()
    
    assert validated['model']['network']['input_dim'] == test_config['model']['network']['input_dim'] 