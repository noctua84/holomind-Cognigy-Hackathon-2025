import pytest
import torch
import os
from pathlib import Path
import tempfile
import shutil

from src.utils.config import ConfigLoader
from src.core.network import ContinualLearningNetwork
from src.core.trainer import ContinualTrainer
from src.data.dataloader import DataManager
from src.monitoring.metrics import MetricsTracker
from src.database.manager import DatabaseManager

@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    config = {
        'model': {
            'network': {
                'input_dim': 784,
                'feature_dim': 256,
                'output_dim': 10,
                'feature_extractor': {
                    'hidden_layers': [512, 256],
                    'activation': 'relu',
                    'dropout': 0.2
                }
            }
        },
        'training': {
            'training': {
                'epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'ewc_lambda': 0.4
            }
        },
        'monitoring': {
            'tensorboard': {
                'enabled': True,
                'log_dir': 'test_runs/'
            },
            'visualization': {
                'output_dir': 'test_visualizations/',
                'plots': {
                    'task_performance': {'update_frequency': 1}
                }
            }
        }
    }
    return config

@pytest.fixture
def temp_dir():
    """Fixture providing temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_database():
    """Fixture providing mock database connections"""
    class MockDB:
        def __init__(self):
            self.stored_data = {}
        
        def save(self, key, value):
            self.stored_data[key] = value
            
        def get(self, key):
            return self.stored_data.get(key)
    
    return MockDB()

@pytest.fixture
def sample_batch():
    """Fixture providing a sample batch of data"""
    batch_size = 32
    input_dim = 784
    num_classes = 10
    
    inputs = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    return inputs, targets

@pytest.fixture
def model(test_config):
    """Fixture providing initialized model"""
    return ContinualLearningNetwork(test_config['model']['network']) 