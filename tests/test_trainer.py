import pytest
import torch
import torch.optim as optim

from src.core.trainer import ContinualTrainer

@pytest.fixture
def trainer(model, test_config):
    """Fixture providing initialized trainer"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return ContinualTrainer(model, optimizer, test_config['training'])

def test_trainer_initialization(trainer):
    """Test proper trainer initialization"""
    assert isinstance(trainer, ContinualTrainer)
    assert trainer.current_task is None
    assert trainer.ewc_lambda > 0

def test_training_step(trainer, sample_batch):
    """Test single training step"""
    trainer.current_task = "task1"
    trainer.model.add_task("task1")
    
    loss = trainer._training_step(sample_batch)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

def test_validation(trainer, sample_batch):
    """Test validation process"""
    trainer.current_task = "task1"
    trainer.model.add_task("task1")
    
    # Create small validation loader
    val_loader = [(sample_batch[0], sample_batch[1]) for _ in range(2)]
    
    val_loss = trainer._validate(val_loader)
    
    assert isinstance(val_loss, float)
    assert val_loss > 0 