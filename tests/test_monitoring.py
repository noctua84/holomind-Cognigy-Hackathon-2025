import pytest
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any

from src.monitoring.metrics import MetricsTracker
from src.monitoring.visualization import PerformanceVisualizer

def test_metrics_tracking(test_config):
    """Test metrics tracking functionality"""
    config = {
        'enabled': True,
        'log_dir': 'test_runs/'
    }
    tracker = MetricsTracker(config)
    
    # Track some metrics
    metrics = {'loss': 0.5, 'accuracy': 0.95}
    tracker.log_training_metrics(metrics, task_id='task1', step=0)
    
    assert 'task1' in tracker.metrics_history
    assert len(tracker.metrics_history['task1']) == 1
    assert tracker.metrics_history['task1'][0]['metrics']['loss'] == 0.5

def test_visualization(test_config, temp_dir):
    """Test visualization generation"""
    config = {
        'output_dir': str(temp_dir),
        'plots': {'task_performance': {'update_frequency': 1}}
    }
    visualizer = PerformanceVisualizer(config)
    
    # Create sample metrics history
    metrics_history = {
        "task1": [
            {
                'step': 0,
                'metrics': {'loss': 1.0, 'accuracy': 0.5},
                'memory': {'rss': 1000, 'vms': 2000, 'gpu_used': 500}
            },
            {
                'step': 1,
                'metrics': {'loss': 0.5, 'accuracy': 0.8},
                'memory': {'rss': 1100, 'vms': 2100, 'gpu_used': 600}
            }
        ]
    }
    
    # Test plot generation
    save_path = "test_performance.png"
    visualizer.plot_task_performance(metrics_history, save_path)
    assert (temp_dir / save_path).exists()

def test_metrics_logging(test_config):
    """Test metrics logging to tensorboard"""
    config = {
        'enabled': True,
        'log_dir': 'test_runs/'
    }
    tracker = MetricsTracker(config)
    
    # Log metrics
    metrics = {'loss': 0.5, 'accuracy': 0.95}
    tracker.log_training_metrics(metrics, task_id='task1', step=0)
    
    # Verify writer was created
    assert tracker.writer is not None
    
    # Clean up
    tracker.close()

def test_gradient_tracking(model, test_config):
    """Test gradient tracking"""
    tracker = MetricsTracker(test_config['monitoring'])
    
    # Generate some gradients
    x = torch.randn(1, 784)
    y = torch.tensor([0])
    task_id = 'task1'
    model.add_task(task_id)
    
    output = model(x, task_id)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    # Track gradients
    tracker.log_model_gradients(model, step=0)
    assert len(tracker.gradient_history) > 0 