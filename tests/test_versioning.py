import pytest
from pathlib import Path
import torch
import yaml
from datetime import datetime

from src.data.dataloader import DataManager

@pytest.fixture
def data_manager(test_config, temp_dir):
    if 'data' not in test_config:
        test_config['data'] = {
            'datasets': {'root_dir': str(temp_dir)},
            'preprocessing': {
                'normalization': {'type': 'standard'},
                'augmentation': {'enabled': False}
            }
        }
    else:
        test_config['data']['datasets']['root_dir'] = str(temp_dir)
    return DataManager(test_config['data'])

def test_data_versioning(data_manager, temp_dir):
    """Test data versioning with DVC"""
    # Create sample data
    data = torch.randn(100, 784)
    targets = torch.randint(0, 10, (100,))
    task_id = "test_task"
    
    # Save data
    data_manager.prepare_task_data(task_id, data, targets)
    
    # Check files exist
    task_dir = temp_dir / task_id
    assert (task_dir / 'features.pt').exists()
    assert (task_dir / 'targets.pt').exists()
    assert (task_dir / 'metadata.yaml').exists()
    
    # Check metadata
    with open(task_dir / 'metadata.yaml') as f:
        metadata = yaml.safe_load(f)
        assert metadata['task_id'] == task_id
        assert metadata['samples'] == 100
        assert metadata['feature_dim'] == 784
        assert metadata['classes'] == 10
        assert datetime.fromisoformat(metadata['created'])

def test_data_loading_after_versioning(data_manager, temp_dir):
    """Test loading versioned data"""
    # First save some data
    data = torch.randn(100, 784)
    targets = torch.randint(0, 10, (100,))
    task_id = "test_task"
    
    data_manager.prepare_task_data(task_id, data, targets)
    
    # Now try to load it
    train_loader, val_loader, test_loader = data_manager.get_task_loaders(task_id)
    
    # Check data integrity
    batch = next(iter(train_loader))
    assert len(batch) == 3  # (data, target, task_id)
    assert batch[0].shape[1] == 784
    # Task ID is now a tuple for each sample in the batch
    assert all(tid == task_id for tid in batch[2]) 