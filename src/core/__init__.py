"""Core components for model training and inference"""

from .network import ContinualLearningNetwork, AdaptiveTaskColumn
from .trainer import ContinualTrainer
from .ewc import EWC
from .memory import ExperienceReplay

__all__ = [
    'ContinualLearningNetwork',
    'AdaptiveTaskColumn',
    'ContinualTrainer',
    'EWC',
    'ExperienceReplay'
] 