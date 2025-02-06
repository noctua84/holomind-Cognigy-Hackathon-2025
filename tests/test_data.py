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

@pytest.fixture
def data_manager(test_config, temp_dir):
    test_config['data'] = {
        'datasets': {'root_dir': str(temp_dir)},
        'preprocessing': {
            'normalization': {'type': 'standard', 'mean': 0, 'std': 1},
            'augmentation': {'enabled': False}
        },
        'dataloader': {
            'batch_size': 32,
            'num_workers': 0,
            'pin_memory': False
        }
    }
    return DataManager(test_config['data'])

def test_data_preparation(data_manager, temp_dir):
    """Test data preparation and versioning"""
    data = torch.randn(100, 784)
    targets = torch.randint(0, 10, (100,))
    task_id = "test_task"
    
    data_manager.prepare_task_data(task_id, data, targets)
    
    # Check files exist
    task_dir = temp_dir / task_id
    assert (task_dir / 'features.pt').exists()
    assert (task_dir / 'targets.pt').exists()
    assert (task_dir / 'metadata.yaml').exists()

def test_data_loading(data_manager, temp_dir):
    """Test data loading functionality"""
    # First prepare some data
    data = torch.randn(100, 784)
    targets = torch.randint(0, 10, (100,))
    task_id = "test_task"
    
    data_manager.prepare_task_data(task_id, data, targets)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_manager.get_task_loaders(task_id)
    
    # Check loaders
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    # Check batch
    batch = next(iter(train_loader))
    assert len(batch) == 3  # (data, target, task_id)
    assert batch[0].shape[1] == 784

def test_preprocessing(test_config):
    """Test preprocessing pipeline"""
    config = {
        'normalization': {'type': 'standard', 'mean': 0, 'std': 1},
        'augmentation': {'enabled': False}
    }
    pipeline = PreprocessingPipeline(config)
    
    # Test normalization
    x = torch.randn(32, 784)
    transformed = pipeline(x)
    
    assert transformed.shape == x.shape
    assert abs(transformed.mean()) < 0.1
    assert abs(transformed.std() - 1.0) < 0.1 