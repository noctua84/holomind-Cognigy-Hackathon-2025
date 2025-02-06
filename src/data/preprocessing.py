from typing import List, Dict, Any, Optional, Union, Callable
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from functools import partial

class PreprocessingPipeline:
    """Configurable preprocessing pipeline for input data"""
    def __init__(self, config: dict):
        self.config = config
        self.transforms = self._build_transforms()
        
    def _build_transforms(self) -> List[Callable]:
        """Build preprocessing transforms"""
        transforms = []
        
        if self.config.get('normalization'):
            norm_config = self.config['normalization']
            transforms.append(
                partial(
                    self._normalize,
                    mean=norm_config.get('mean', 0.0),
                    std=norm_config.get('std', 1.0)
                )
            )
        
        return transforms
    
    @staticmethod
    def _normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Normalize input tensor"""
        return (x - mean) / std
    
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
        """Apply preprocessing transforms"""
        for transform in self.transforms:
            x = transform(x)
        return x 