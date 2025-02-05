import pytest
import torch
import os
from pathlib import Path
import tempfile
import shutil
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

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
                },
                'memory': {
                    'size': 1000,
                    'feature_dim': 256
                },
                'task_columns': {
                    'hidden_dims': [256, 128, 64],
                    'activation': 'relu',
                    'dropout': 0.2
                }
            }
        },
        'data': {
            'preprocessing': {
                'normalize': True,
                'augment': False,
                'input_shape': [784],
                'normalization': {
                    'type': 'standard',
                    'mean': 0.0,
                    'std': 1.0,
                    'per_feature': False
                },
                'augmentation': {
                    'enabled': False,
                    'types': []
                }
            }
        },
        'training': {
            'epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'training': {
                'ewc_lambda': 0.4,
                'batch_size': 32
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

@pytest.fixture(scope="session")
def postgres_db():
    """Create test database and clean it up after tests"""
    # Connection parameters for creating database
    admin_params = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }
    
    test_db_name = 'test_holomind'
    
    try:
        # Connect to default postgres database to create test database
        conn = psycopg2.connect(**admin_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Drop test database if it exists
        cur.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
        # Create fresh test database
        cur.execute(f"CREATE DATABASE {test_db_name}")
        
        cur.close()
        conn.close()
        
        # Now connect to the test database and create schema
        test_conn = psycopg2.connect(dbname=test_db_name, **{k:v for k,v in admin_params.items() if k != 'dbname'})
        test_cur = test_conn.cursor()
        
        # Create tables
        test_cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                hyperparameters JSONB,
                status TEXT
            )
        """)
        
        test_cur.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                experiment_id INTEGER REFERENCES experiments(experiment_id),
                task_id TEXT,
                epoch INTEGER,
                metrics JSONB,
                PRIMARY KEY (experiment_id, task_id, epoch)
            )
        """)
        
        test_conn.commit()
        test_cur.close()
        test_conn.close()
        
        # Yield database parameters
        yield {
            'dbname': test_db_name,
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432'
        }
        
    finally:
        # Cleanup: drop test database
        conn = psycopg2.connect(**admin_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
        cur.close()
        conn.close() 