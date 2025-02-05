import pytest
import torch
import torch.nn as nn

from src.core.network import ContinualLearningNetwork, TaskColumn

def test_network_initialization(model):
    """Test proper network initialization"""
    assert isinstance(model, ContinualLearningNetwork)
    assert isinstance(model.feature_extractor, nn.Module)
    assert len(model.task_columns) == 0

def test_add_task(model):
    """Test adding new task columns"""
    task_id = "task1"
    model.add_task(task_id)
    
    assert task_id in model.task_columns
    assert isinstance(model.task_columns[task_id], TaskColumn)

def test_forward_pass(model, sample_batch):
    """Test forward pass through network"""
    inputs, _ = sample_batch
    task_id = "task1"
    model.add_task(task_id)
    
    outputs = model(inputs, task_id)
    
    assert outputs.shape[0] == inputs.shape[0]
    assert outputs.shape[1] == model.config['output_dim']

def test_ewc_loss_computation(model, sample_batch):
    """Test EWC loss computation"""
    inputs, targets = sample_batch
    task_id = "task1"
    model.add_task(task_id)
    
    # Initial forward and backward pass
    outputs = model(inputs, task_id)
    loss = nn.functional.cross_entropy(outputs, targets)
    loss.backward()
    
    # Update importance weights
    model.update_importance(loss)
    
    # Check Fisher diagonal values
    assert len(model.fisher_tracker.fisher_diagonal) > 0
    for name, values in model.fisher_tracker.fisher_diagonal.items():
        assert torch.all(values >= 0)  # Fisher values should be non-negative 