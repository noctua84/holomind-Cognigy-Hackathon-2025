from typing import Dict, Any, Optional, List
import torch
from torch.utils.tensorboard import SummaryWriter
import psutil
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

class MetricsTracker:
    """Tracks and logs training metrics and system performance"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_dir = Path(config['tensorboard']['log_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(self.run_dir)
        self.process = psutil.Process()
        self.metrics_history = {}
        
    def log_training_metrics(self, metrics: Dict[str, float], 
                           task_id: str, step: int):
        """Log training metrics to tensorboard"""
        # Basic training metrics
        for name, value in metrics.items():
            if value is not None:  # Skip None values
                self.writer.add_scalar(f'training/{task_id}/{name}', value, step)
        
        # Track memory usage
        memory_metrics = self._get_memory_metrics()
        for name, value in memory_metrics.items():
            self.writer.add_scalar(f'system/memory/{name}', value, step)
            
        # Store in history
        if task_id not in self.metrics_history:
            self.metrics_history[task_id] = []
        self.metrics_history[task_id].append({
            'step': step,
            'metrics': metrics,
            'memory': memory_metrics
        })
        
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """Log model gradient statistics"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', 
                                        param.grad.data.cpu().numpy(), step)
                
                grad_norm = torch.norm(param.grad.data)
                self.writer.add_scalar(f'gradient_norms/{name}', grad_norm.item(), step)
    
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