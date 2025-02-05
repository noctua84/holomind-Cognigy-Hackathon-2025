from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from .dataset import ContinualDataset
from .preprocessing import PreprocessingPipeline

class DataManager:
    """Manages data loading and preprocessing for continual learning"""
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['datasets']['root_dir'])
        self.preprocessor = PreprocessingPipeline(config['preprocessing'])
        
    def get_task_loaders(self, 
                        task_id: str) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Get data loaders for a specific task"""
        # Create datasets
        train_dataset = ContinualDataset(
            data_dir=self.data_dir,
            task_id=task_id,
            transform=self.preprocessor,
            target_transform=None
        )
        
        # Split into train/val/test
        train_size = int(len(train_dataset) * self.config['datasets']['train_split'])
        val_size = int(len(train_dataset) * self.config['datasets']['val_split'])
        test_size = len(train_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=self.config['dataloader']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=self.config['dataloader']['pin_memory']
        ) if val_size > 0 else None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=self.config['dataloader']['pin_memory']
        )
        
        return train_loader, val_loader, test_loader
    
    def prepare_task_data(self, task_id: str, data: torch.Tensor, 
                         targets: torch.Tensor):
        """Prepare and save data for a new task"""
        task_dir = self.data_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features and targets
        torch.save(data, task_dir / 'features.pt')
        torch.save(targets, task_dir / 'targets.pt')
        
        logging.info(f"Prepared data for task {task_id}: {len(data)} samples") 