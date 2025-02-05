import pytest
import torch
from pathlib import Path

from src.data.preprocessing import PreprocessingPipeline
from src.data.dataset import ContinualDataset
from src.data.dataloader import DataManager

@pytest.fixture
def sample_data(temp_dir):
    """Fixture providing sample dataset"""
    task_id = "task1"
    task_dir = temp_dir / task_id
    task_dir.mkdir(parents=True)
    
    # Create sample data
    features = torch.randn(100, 784)
    targets = torch.randint(0, 10, (100,))
    
    # Save data
    torch.save(features, task_dir / 'features.pt')
    torch.save(targets, task_dir / 'targets.pt')
    
    return task_dir, features, targets

def test_preprocessing_pipeline(test_config):
    """Test preprocessing pipeline"""
    pipeline = PreprocessingPipeline(test_config['data']['preprocessing'])
    
    # Test transform application
    x = torch.randn(1, 784)
    transformed = pipeline(x)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == x.shape

def test_dataset_loading(sample_data):
    """Test dataset loading and access"""
    task_dir, features, targets = sample_data
    
    dataset = ContinualDataset(
        data_dir=task_dir.parent,
        task_id="task1"
    )
    
    assert len(dataset) == len(features)
    
    # Test item access
    x, y, task_id = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert task_id == "task1"

def test_augmentation_pipeline(test_config):
    """Test data augmentation pipeline"""
    test_config['data']['preprocessing']['augmentation']['enabled'] = True
    pipeline = PreprocessingPipeline(test_config['data']['preprocessing'])
    
    x = torch.randn(1, 784)
    transformed = pipeline(x)
    assert transformed.shape == x.shape

def test_batch_preprocessing(test_config):
    """Test batch preprocessing"""
    pipeline = PreprocessingPipeline(test_config['data']['preprocessing'])
    
    batch = torch.randn(32, 784)
    transformed = pipeline(batch)
    assert transformed.shape == batch.shape
    
    # Test normalization
    assert torch.abs(transformed.mean()) < 1.0
    assert torch.abs(transformed.std() - 1.0) < 0.1

def test_dataloader_configuration(test_config):
    """Test dataloader configuration"""
    config = {
        'datasets': {
            'root_dir': 'data/'
        },
        'preprocessing': test_config['data']['preprocessing'],  # Use preprocessing from test_config
        'batch_size': 32,
        'num_workers': 2,
        'shuffle': True
    }
    
    manager = DataManager(config)
    
    assert manager.batch_size == 32
    assert manager.num_workers == 2
    assert manager.shuffle == True 