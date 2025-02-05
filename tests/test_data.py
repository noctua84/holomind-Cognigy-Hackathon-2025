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