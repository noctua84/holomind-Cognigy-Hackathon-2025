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
            # For flat vectors, use simple normalization
            mean = self.config['normalization']['mean']
            std = self.config['normalization']['std']
            transform_list.append(
                lambda x: (x - mean) / std
            )
        
        # Augmentations
        if self.config['augmentation']['enabled']:
            # Add augmentations for flat vectors if needed
            pass
        
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
        """Apply transformations to input"""
        if len(x.shape) == 2:  # Batch of flat vectors
            return self.transforms(x)
        else:  # Single flat vector
            return self.transforms(x.unsqueeze(0)).squeeze(0) 