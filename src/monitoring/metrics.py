from typing import Dict, Any, Optional, List
import torch
from torch.utils.tensorboard import SummaryWriter
import psutil
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict

class MetricsTracker:
    """Tracks and logs training metrics and system performance"""
    def __init__(self, config: Dict[str, Any]):
        """Initialize metrics tracker with configuration"""
        self.config = config
        self.metrics_history = {}
        
        # Setup tensorboard if enabled
        self.writer = None
        if config.get('enabled', False):
            run_dir = Path(config.get('log_dir', 'runs')) / datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(run_dir))
        self.process = psutil.Process()
        self.gradient_history = []
        
    def log_training_metrics(self, metrics: Dict[str, float], task_id: str, step: int):
        """Log training metrics to history and tensorboard"""
        if task_id not in self.metrics_history:
            self.metrics_history[task_id] = []
        
        metrics_entry = {
            'step': step,
            'metrics': metrics,
            'memory': self._get_memory_metrics()
        }
        
        self.metrics_history[task_id].append(metrics_entry)
        
        # Log to tensorboard if enabled
        if self.writer:
            for metric_name, value in metrics.items():
                if value is not None:
                    self.writer.add_scalar(f"{task_id}/{metric_name}", value, step)
            
            # Log memory metrics
            for name, value in metrics_entry['memory'].items():
                if self.writer:
                    self.writer.add_scalar(f'system/memory/{name}', value, step)
        
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """Log model gradient statistics"""
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_stats[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'norm': param.grad.norm().item()
                }
                
        self.gradient_history.append({
            'step': step,
            'stats': gradient_stats
        })
    
    def log_forgetting_metrics(self, task_performances: Dict[str, float], step: int):
        """Log metrics related to catastrophic forgetting"""
        for task_id, performance in task_performances.items():
            self.writer.add_scalar(f'forgetting/{task_id}', performance, step)
            
        # Calculate average forgetting
        if len(task_performances) > 1:
            avg_forgetting = sum(task_performances.values()) / len(task_performances)
            self.writer.add_scalar('forgetting/average', avg_forgetting, step)
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get current memory usage metrics"""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            'gpu_used': self._get_gpu_memory() if torch.cuda.is_available() else 0
        }
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB"""
        return torch.cuda.memory_allocated() / (1024 * 1024)
    
    def close(self):
        """Clean up resources"""
        self.writer.close() 