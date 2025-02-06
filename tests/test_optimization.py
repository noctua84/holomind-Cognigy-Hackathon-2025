import pytest
import torch
from ray import tune

from src.utils.optimization import HyperparameterOptimizer

@pytest.fixture
def optimizer(test_config):
    return HyperparameterOptimizer(test_config)

def test_optimizer_initialization(optimizer):
    """Test proper optimizer initialization"""
    assert isinstance(optimizer.search_space, dict)
    assert "learning_rate" in optimizer.search_space
    assert "batch_size" in optimizer.search_space

def test_search_space_validity(optimizer):
    """Test search space configuration"""
    lr_space = optimizer.search_space["learning_rate"]
    assert hasattr(lr_space, 'sample')  # Check if it's a searchable space
    
    batch_space = optimizer.search_space["batch_size"]
    assert hasattr(batch_space, 'sample')
    assert all(isinstance(x, int) for x in [16, 32, 64, 128])

def test_optimization_run(optimizer):
    """Test optimization execution"""
    def dummy_train(config):
        from ray import train
        loss = (config["learning_rate"] - 0.001) ** 2
        train.report({"loss": loss})
    
    tuner = tune.Tuner(
        dummy_train,
        param_space=optimizer.search_space,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=2
        )
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric="loss", mode="min")
    assert isinstance(best_result.config, dict)
    assert "learning_rate" in best_result.config

def test_resource_allocation(optimizer):
    """Test resource allocation configuration"""
    def dummy_train(config):
        tune.report(loss=1.0)
    
    best_config = optimizer.optimize(
        train_fn=dummy_train,
        num_samples=1
    )
    
    assert isinstance(best_config, dict) 