from typing import Dict, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
import dvc.api
import yaml
from datetime import datetime, UTC

from .dataset import ContinualDataset
from .preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data loading and preprocessing"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config['datasets']['root_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get splits from config
        self.train_split = config.get('train_split', 0.7)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        
        # Initialize preprocessing
        self.preprocessor = PreprocessingPipeline(config['preprocessing'])
        
        # Initialize DVC repo if needed
        try:
            import dvc.repo
            if not (self.data_dir / '.dvc').exists():
                self.dvc = dvc.repo.Repo.init(str(self.data_dir), no_scm=True)
            else:
                self.dvc = dvc.repo.Repo(str(self.data_dir))
            self.use_dvc = True
        except (ImportError, Exception) as e:
            logging.warning(f"DVC initialization failed - {e}")
            self.use_dvc = False
        
        # Store dataloader configuration
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 2)
        self.shuffle = config.get('shuffle', True)
        
    def get_task_loaders(self, task_id: str) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Get data loaders for a specific task"""
        # Create datasets
        full_dataset = ContinualDataset(
            data_dir=self.data_dir,
            task_id=task_id,
            transform=self.preprocessor,
            target_transform=None
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size, test_size]
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
        """Prepare and save data for a new task with versioning"""
        task_dir = self.data_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = task_dir / 'features.pt'
        target_path = task_dir / 'targets.pt'
        
        torch.save(data, data_path)
        torch.save(targets, target_path)
        
        # Add metadata
        metadata_path = task_dir / 'metadata.yaml'
        with metadata_path.open('w') as f:
            yaml.safe_dump({
                'task_id': task_id,
                'samples': len(data),
                'feature_dim': data.shape[1],
                'classes': len(torch.unique(targets)),
                'created': datetime.now(UTC).isoformat()
            }, f)
        
        # Track with DVC if available
        if self.use_dvc:
            try:
                self.dvc.add([str(data_path), str(target_path), str(metadata_path)])
                self.dvc.scm.add([
                    str(data_path) + '.dvc',
                    str(target_path) + '.dvc',
                    str(metadata_path) + '.dvc'
                ])
                self.dvc.scm.commit(f"Add data for task {task_id}")
            except Exception as e:
                logging.warning(f"DVC tracking failed - {e}")
        
        logging.info(f"Prepared data for task {task_id}: {len(data)} samples") 