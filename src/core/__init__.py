"""Core components for continual learning"""

from .network import ContinualLearningNetwork, TaskColumn
from .trainer import ContinualTrainer

__all__ = [
    'ContinualLearningNetwork',
    'TaskColumn',
    'ContinualTrainer',
] 