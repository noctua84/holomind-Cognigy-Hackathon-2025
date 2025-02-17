from pathlib import Path
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints and training history"""
    
    def __init__(self, 
                 base_dir: str = 'checkpoints',
                 save_frequency: int = 1,
                 keep_last: int = 3,
                 save_optimizer: bool = True,
                 save_metrics: bool = True):
        self.base_dir = Path(base_dir)
        self.checkpoints_dir = self.base_dir / 'model_states'
        self.history_dir = self.base_dir / 'history'
        self.metrics_dir = self.base_dir / 'metrics'
        
        self.save_frequency = save_frequency
        self.keep_last = keep_last
        self.save_optimizer = save_optimizer
        self.save_metrics = save_metrics
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any], task_id: str):
        """Save model checkpoint with task-specific information"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            **state
        }
        path = self.checkpoints_dir / f'checkpoint_task_{task_id}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint for task {task_id} at {path}")
    
    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a specific task"""
        path = self.checkpoints_dir / f'checkpoint_task_{task_id}.pt'
        if path.exists():
            checkpoint = torch.load(path)
            logger.info(f"Loaded checkpoint for task {task_id}")
            return checkpoint
        return None
    
    def save_history(self, task_id: str, history: Dict[str, Any]):
        """Save training history for a task"""
        path = self.history_dir / f'history_task_{task_id}.json'
        with open(path, 'w') as f:
            json.dump(history, f)
        logger.info(f"Saved training history for task {task_id}")
    
    def load_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load training history for a task"""
        path = self.history_dir / f'history_task_{task_id}.json'
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def get_latest_task_id(self) -> Optional[str]:
        """Get the most recent task ID from checkpoints"""
        checkpoints = list(self.checkpoints_dir.glob('checkpoint_task_*.pt'))
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest.stem.split('_')[-1] 