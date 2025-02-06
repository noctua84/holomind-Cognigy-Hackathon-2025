from typing import Dict, Any, Optional, List
import torch
from torch.utils.tensorboard import SummaryWriter
import psutil
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict
import mlflow

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and logs training metrics using MLflow"""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = {}  # History for metrics
        self.gradient_history = []  # History for gradients
        self.forgetting_history = {}  # Add history for forgetting metrics
        
        # Set up MLflow
        try:
            # Create mlruns directory if it doesn't exist
            Path("mlruns").mkdir(exist_ok=True)
            
            # Set tracking URI
            mlflow.set_tracking_uri(config.get('tracking_uri', 'mlruns'))
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=str(Path("mlruns").absolute() / experiment_name)
                )
            else:
                self.experiment_id = experiment.experiment_id
                
            # Start new run
            self.run = mlflow.start_run(experiment_id=self.experiment_id)
            
            # Log config parameters
            mlflow.log_params(self._flatten_dict(config))
            
            logger.info(f"MLflow tracking initialized: {experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking: {e}")
            raise
            
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary for MLflow params"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics for current step"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            
    def log_model(self, model, name: str):
        """Log model with input example and signature"""
        try:
            # Create example input
            input_example = torch.randn(1, self.config['model']['network']['input_dim'])
            
            # Log model with signature
            mlflow.pytorch.log_model(
                model,
                name,
                input_example=input_example,
                registered_model_name=f"{name}_registered"
            )
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            
    def end_run(self):
        """End current MLflow run"""
        try:
            mlflow.end_run()
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_training_metrics(self, metrics: Dict[str, float], task_id: str, step: int):
        """Log training metrics to history and MLflow"""
        if task_id not in self.metrics_history:
            self.metrics_history[task_id] = []
            
        metrics_entry = {
            'step': step,
            'metrics': metrics
        }
        
        self.metrics_history[task_id].append(metrics_entry)
        
        # Log to MLflow
        try:
            mlflow.log_metrics(
                {f"{task_id}/{k}": v for k, v in metrics.items() if v is not None},
                step=step
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """Log model gradient statistics to MLflow and history"""
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_stats[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'norm': param.grad.norm().item()
                }
                
                # Log to MLflow directly
                try:
                    mlflow.log_metrics({
                        f"gradients/{name}/mean": gradient_stats[name]['mean'],
                        f"gradients/{name}/std": gradient_stats[name]['std'],
                        f"gradients/{name}/norm": gradient_stats[name]['norm']
                    }, step=step)
                except Exception as e:
                    logger.warning(f"Failed to log gradient metrics to MLflow: {e}")
        
        # Store in history
        self.gradient_history.append({
            'step': step,
            'stats': gradient_stats
        })
    
    def log_forgetting_metrics(self, task_performances: Dict[str, float], step: int):
        """Log metrics related to catastrophic forgetting"""
        # Store in history
        self.forgetting_history[step] = task_performances
        
        # Log to MLflow
        try:
            metrics = {
                f"forgetting/{task_id}": performance 
                for task_id, performance in task_performances.items()
            }
            if len(task_performances) > 1:
                metrics['forgetting/average'] = sum(task_performances.values()) / len(task_performances)
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log forgetting metrics to MLflow: {e}")
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get current memory usage metrics"""
        try:
            memory_info = psutil.Process().memory_info()
            metrics = {
                'memory/rss': memory_info.rss / (1024 * 1024),  # RSS in MB
                'memory/vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            }
            if torch.cuda.is_available():
                metrics['memory/gpu'] = torch.cuda.memory_allocated() / (1024 * 1024)
            mlflow.log_metrics(metrics)
            return metrics
        except Exception as e:
            logger.warning(f"Failed to log memory metrics: {e}")
            return {}
    
    def close(self):
        """Clean up resources and end MLflow run"""
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}") 