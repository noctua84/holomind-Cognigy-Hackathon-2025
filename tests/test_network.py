import pytest
import torch
import torch.nn as nn

from src.core.network import ContinualLearningNetwork, TaskColumn, ExternalMemory

@pytest.fixture
def network(test_config):
    return ContinualLearningNetwork(test_config['model']['network'])

def test_network_initialization(network):
    """Test proper network initialization"""
    assert isinstance(network.feature_extractor, nn.Module)
    assert isinstance(network.memory, ExternalMemory)
    assert len(network.task_columns) == 0
    assert network.ewc_loss is None

def test_feature_extractor(network, sample_batch):
    """Test feature extraction"""
    inputs, _ = sample_batch
    features = network.feature_extractor(inputs)
    assert features.shape == (inputs.shape[0], network.config['feature_dim'])

def test_task_column_addition(network):
    """Test adding new task columns"""
    task_id = "task1"
    network.add_task(task_id)
    
    assert task_id in network.task_columns
    assert isinstance(network.task_columns[task_id], TaskColumn)
    
    # Adding same task twice should not create duplicate
    network.add_task(task_id)
    assert len(network.task_columns) == 1

def test_forward_pass(network, sample_batch):
    """Test complete forward pass"""
    inputs, _ = sample_batch
    task_id = "task1"
    network.add_task(task_id)
    
    outputs = network(inputs, task_id)
    assert outputs.shape == (inputs.shape[0], network.config['output_dim'])
    
    # Test with unknown task
    with pytest.raises(ValueError):
        network(inputs, "unknown_task")

def test_ewc_importance(network, sample_batch):
    """Test EWC importance update"""
    inputs, targets = sample_batch
    task_id = "task1"
    network.add_task(task_id)
    
    # Forward pass and loss computation
    outputs = network(inputs, task_id)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    # Update importance
    network.update_importance(loss)
    assert len(network.fisher_tracker.fisher_diagonal) > 0

def test_multiple_tasks(network, sample_batch):
    """Test handling multiple tasks"""
    inputs, _ = sample_batch
    
    # Add multiple tasks
    tasks = ["task1", "task2", "task3"]
    for task_id in tasks:
        network.add_task(task_id)
        outputs = network(inputs, task_id)
        assert outputs.shape == (inputs.shape[0], network.config['output_dim'])
    
    assert len(network.task_columns) == len(tasks)

def test_ewc_loss_computation(model, sample_batch):
    """Test EWC loss computation"""
    inputs, targets = sample_batch
    task_id = "task1"
    model.add_task(task_id)
    
    # Initial forward and backward pass
    outputs = model(inputs, task_id)
    loss = nn.functional.cross_entropy(outputs, targets)
    
    # Update importance weights with retain_graph=True
    model.update_importance(loss, retain_graph=True)
    
    # Check Fisher diagonal values
    assert len(model.fisher_tracker.fisher_diagonal) > 0

def test_memory_operations(model):
    """Test external memory operations"""
    batch_size = 4
    features = torch.randn(batch_size, model.config['feature_dim'])
    importance = torch.ones(batch_size)
    
    # Test memory query
    memory_output = model.memory.query(features)
    assert memory_output.shape == features.shape
    
    # Test memory update
    model.memory.update(features, importance)
    assert not torch.all(model.memory.memory == 0)

def test_fisher_tracking(model, sample_batch):
    """Test Fisher information tracking"""
    inputs, targets = sample_batch
    task_id = "task1"
    model.add_task(task_id)
    model.current_task = task_id  # Set current task
    
    criterion = nn.CrossEntropyLoss()
    outputs = model(inputs, task_id)
    loss = criterion(outputs, targets)
    
    # Test importance update
    model.update_importance(loss)
    assert len(model.fisher_tracker.fisher_diagonal) > 0

def test_lateral_connections(model):
    """Test lateral connections between task columns"""
    task1_id = "task1"
    task2_id = "task2"
    
    # Add two tasks
    model.add_task(task1_id)
    model.add_task(task2_id)
    
    # Verify task columns
    assert task1_id in model.task_columns
    assert task2_id in model.task_columns 