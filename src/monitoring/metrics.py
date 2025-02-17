from typing import Dict, Any, Optional, List
import torch
import psutil
import logging
from pathlib import Path
from datetime import datetime
from src.database import MetricsDB, ModelArtifact

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and logs training metrics using database storage"""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.metrics_history: Dict[str, List[Dict]] = {}
        
        # Initialize database connection
        db_url = config.get('database', {}).get('url', 'sqlite:///metrics.db')
        self.metrics_db = MetricsDB(db_url=db_url)
        
        # Create experiment
        self.experiment_id = self.metrics_db.create_experiment(
            name=experiment_name,
            config=config
        )
        
        # Create run
        self.run_id = self.metrics_db.create_run(
            experiment_id=self.experiment_id,
            config=config
        )
        
        logger.info(f"Database tracking initialized: {experiment_name}")
    
    def log_training_metrics(self, metrics: Dict[str, float], task_id: str, step: int):
        """Log training metrics to database"""
        # Add memory metrics
        metrics.update(self._get_memory_metrics())
        
        # Store in history
        if task_id not in self.metrics_history:
            self.metrics_history[task_id] = []
        
        self.metrics_history[task_id].append({
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log to database
        self.metrics_db.log_metrics(
            task_id=task_id,
            run_id=self.run_id,
            epoch=step,
            step=step,
            metrics={
                'training_metrics': metrics,
                'memory_metrics': self._get_memory_metrics(),
                'gradient_stats': {}
            }
        )
    
    def log_model(self, model: torch.nn.Module, name: str):
        """Log model state to database"""
        try:
            # Save model state
            state_dict = model.state_dict()
            self.metrics_db.save_model(
                run_id=self.run_id,
                name=name,
                state_dict=state_dict,
                metadata={
                    'architecture': str(model.__class__.__name__),
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """Log gradient statistics to database"""
        grad_metrics = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_metrics[f'gradients/{name}/mean'] = param.grad.mean().item()
                grad_metrics[f'gradients/{name}/std'] = param.grad.std().item()
                grad_metrics[f'gradients/{name}/norm'] = param.grad.norm().item()
        
        self.metrics_db.log_gradients(
            run_id=self.run_id,
            step=step,
            gradients=grad_metrics
        )
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get current memory usage metrics"""
        memory_metrics = {}
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_metrics['memory/rss'] = memory_info.rss / (1024 * 1024)
        memory_metrics['memory/vms'] = memory_info.vms / (1024 * 1024)
        if torch.cuda.is_available():
            memory_metrics['memory/gpu'] = torch.cuda.memory_allocated() / (1024 * 1024)
        return memory_metrics
    
    def close(self):
        """Clean up resources"""
        try:
            self.metrics_db.complete_run(self.run_id)
        except Exception as e:
            logger.error(f"Failed to close metrics tracker: {e}") 