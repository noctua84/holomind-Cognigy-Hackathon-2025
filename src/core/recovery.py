from pathlib import Path
import torch
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StateRecovery:
    """Handles state recovery and error recovery during training"""
    
    def __init__(self, checkpoint_manager, metrics_db):
        self.checkpoint_manager = checkpoint_manager
        self.metrics_db = metrics_db
        
    def save_recovery_point(self, task_id: str, state: Dict[str, Any]):
        """Save a recovery point with full state"""
        recovery_state = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'state': state,
            'metrics': self.metrics_db.get_task_metrics(task_id)
        }
        path = self.checkpoint_manager.base_dir / 'recovery' / f'recovery_{task_id}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(recovery_state, path)
        
    def recover_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Recover from the latest valid state"""
        # Try recovery point first
        recovery_path = self.checkpoint_manager.base_dir / 'recovery' / f'recovery_{task_id}.pt'
        if recovery_path.exists():
            try:
                state = torch.load(recovery_path)
                logger.info(f"Recovered from recovery point for task {task_id}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load recovery point: {e}")
        
        # Fall back to latest checkpoint
        return self.checkpoint_manager.load_checkpoint(task_id) 