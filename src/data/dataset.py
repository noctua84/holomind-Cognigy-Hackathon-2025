from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging

class ContinualDataset(Dataset):
    """Dataset class for continual learning tasks"""
    def __init__(self, 
                 data_dir: str,
                 task_id: str,
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None):
        self.data_dir = Path(data_dir)
        self.task_id = task_id
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = []
        self.targets = []
        self._load_data()
        
    def _load_data(self):
        """Load data for the specific task"""
        try:
            # Load data from task-specific directory
            data_path = self.data_dir / self.task_id
            if not data_path.exists():
                raise FileNotFoundError(f"Data not found for task {self.task_id}")
                
            # Load features and targets
            self.data = torch.load(data_path / 'features.pt')
            self.targets = torch.load(data_path / 'targets.pt')
            
            logging.info(f"Loaded {len(self.data)} samples for task {self.task_id}")
            
        except Exception as e:
            logging.error(f"Error loading data for task {self.task_id}: {str(e)}")
            raise
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        x, y = self.data[idx], self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y, self.task_id 