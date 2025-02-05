from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

class PreprocessingPipeline:
    """Configurable preprocessing pipeline for input data"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transforms = self._build_transforms()
        
    def _build_transforms(self) -> nn.Module:
        """Build transformation pipeline from config"""
        transform_list = []
        
        # Normalization
        if self.config['normalization']['type'] == 'standard':
            transform_list.append(
                T.Normalize(mean=[0.5], std=[0.5])
                if self.config['normalization']['per_feature']
                else T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            )
            
        # Augmentations
        if self.config['augmentation']['enabled']:
            for aug in self.config['augmentation']['methods']:
                transform = self._get_augmentation(aug)
                if transform:
                    transform_list.append(transform)
        
        return T.Compose(transform_list)
    
    def _get_augmentation(self, aug_config: Dict) -> Optional[nn.Module]:
        """Get augmentation transform from config"""
        name = aug_config['name']
        params = aug_config['params']
        
        if name == 'random_crop':
            return T.RandomCrop(**params)
        elif name == 'random_rotation':
            return T.RandomRotation(**params)
        elif name == 'random_horizontal_flip':
            return T.RandomHorizontalFlip(**params)
        else:
            return None
            
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing pipeline to input"""
        return self.transforms(x) 