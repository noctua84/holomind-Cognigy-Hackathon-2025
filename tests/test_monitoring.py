import pytest
from pathlib import Path

from src.monitoring.metrics import MetricsTracker
from src.monitoring.visualization import PerformanceVisualizer

def test_metrics_tracking(test_config, temp_dir):
    """Test metrics tracking functionality"""
    test_config['monitoring']['tensorboard']['log_dir'] = str(temp_dir)
    tracker = MetricsTracker(test_config['monitoring'])
    
    # Test logging metrics
    metrics = {
        'loss': 0.5,
        'accuracy': 0.85
    }
    tracker.log_training_metrics(metrics, "task1", step=0)
    
    assert "task1" in tracker.metrics_history
    assert len(tracker.metrics_history["task1"]) == 1
    
    tracker.close()

def test_visualization(test_config, temp_dir):
    """Test visualization generation"""
    test_config['monitoring']['visualization']['output_dir'] = str(temp_dir)
    visualizer = PerformanceVisualizer(test_config['monitoring'])
    
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
    visualizer.plot_task_performance(metrics_history, "test_performance.png")
    assert (temp_dir / "test_performance.png").exists() 