import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.core.trainer import ContinualTrainer

@pytest.fixture
def trainer(test_config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=test_config['training']['learning_rate'])
    return ContinualTrainer(model, optimizer, test_config)

def test_trainer_initialization(trainer):
    """Test trainer initialization"""
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert trainer.current_task is None
    assert trainer.ewc_lambda == 0.4

def test_training_step(trainer, sample_batch):
    """Test single training step"""
    inputs, targets = sample_batch
    task_id = "task1"
    trainer.current_task = task_id
    trainer.model.add_task(task_id)
    
    loss = trainer._training_step((inputs, targets))
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

def test_validation(trainer, sample_batch):
    """Test validation process"""
    inputs, targets = sample_batch
    task_id = "task1"
    trainer.current_task = task_id
    trainer.model.add_task(task_id)
    
    # Create validation loader
    val_dataset = TensorDataset(inputs, targets)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    val_loss = trainer._validate(val_loader)
    assert isinstance(val_loss, float)
    assert val_loss > 0

def test_full_training(trainer, sample_batch):
    """Test complete training cycle"""
    inputs, targets = sample_batch
    task_id = "task1"
    
    # Create data loaders
    train_dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(train_dataset, batch_size=16)
    
    # Train for one epoch
    trainer.train_task(
        task_id=task_id,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1
    )
    
    assert task_id in trainer.model.task_columns

def test_mlflow_logging(trainer, sample_batch, temp_dir):
    """Test MLflow metric logging"""
    inputs, targets = sample_batch
    task_id = "test_task"
    
    # Create data loader
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=16)
    
    # Train with MLflow logging
    trainer.train_task(task_id, loader, epochs=1)
    
    # Check metrics were logged
    assert len(trainer.metrics_tracker.metrics_history[task_id]) > 0 