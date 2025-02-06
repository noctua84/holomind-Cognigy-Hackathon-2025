import pytest
import torch
import torch.nn as nn
import numpy as np

from src.core.network import ExternalMemory

@pytest.fixture
def memory():
    return ExternalMemory(memory_size=100, feature_dim=32)

def test_memory_initialization(memory):
    """Test proper memory initialization"""
    assert isinstance(memory.memory, nn.Parameter)
    assert memory.memory.shape == (100, 32)
    assert torch.all(memory.usage_counts == 0)

def test_memory_update(memory):
    """Test memory update mechanism"""
    features = torch.randn(10, 32)
    importance = torch.ones(10)
    
    memory.update(features, importance)
    
    # Check that memory was updated
    assert not torch.all(memory.memory.data == 0)
    assert torch.sum(memory.usage_counts > 0) == 10

def test_memory_query(memory):
    """Test memory query operation"""
    # First add some known patterns
    features = torch.randn(10, 32)
    importance = torch.ones(10)
    memory.update(features, importance)
    
    # Query with one of the stored patterns
    query = features[0].unsqueeze(0)
    result = memory.query(query)
    
    assert result.shape == query.shape
    # The closest match should be the query itself
    assert torch.allclose(result, query, atol=1e-5)

def test_importance_scoring(memory):
    """Test importance score computation"""
    features = np.random.randn(1, 32)
    memory_data = np.random.randn(100, 32)
    
    scores = memory._compute_importance_scores(features, memory_data)
    
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))  # No NaN or inf values 