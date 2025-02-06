import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class MemorySystem:
    """Memory system with different backend options"""
    
    class FaissMixin:
        """Use FAISS for efficient similarity search"""
        def __init__(self):
            import faiss
            self.index = faiss.IndexFlatL2(self.feature_dim)
            
        def query(self, features):
            return self.index.search(features, k=self.top_k)
    
    class QuantizedMemory:
        """8-bit quantization for memory storage"""
        def __init__(self):
            self.scale = nn.Parameter(torch.ones(1))
            self.zero_point = nn.Parameter(torch.zeros(1))
            self.memory_int8 = torch.zeros(size, dim, dtype=torch.int8)
            
        def dequantize(self, x_int8):
            return self.scale * (x_int8 - self.zero_point)
    
    class LSHMemory:
        """Locality-Sensitive Hashing for approximate search"""
        def __init__(self):
            self.hash_tables = []
            self.projection_matrices = nn.ParameterList([
                nn.Parameter(torch.randn(dim, hash_dim))
                for _ in range(num_tables)
            ])

class ExperienceReplay:
    """Memory buffer for experience replay"""
    def __init__(self, capacity: int, feature_dim: int):
        self.capacity = capacity
        self.memory = {}
        self.feature_dim = feature_dim
        
    def add_task(self, task_id: str, examples: torch.Tensor, labels: torch.Tensor):
        """Store important examples for a task"""
        if len(examples) > self.capacity:
            # Select most representative examples using herding
            selected_indices = self._herding_selection(examples, self.capacity)
            examples = examples[selected_indices]
            labels = labels[selected_indices]
            
        self.memory[task_id] = {
            'examples': examples,
            'labels': labels
        }
        
    def _herding_selection(self, examples: torch.Tensor, k: int) -> torch.Tensor:
        """Select k most representative examples using herding"""
        # Compute mean of all examples
        mu = examples.mean(0)
        
        selected = []
        for _ in range(k):
            if not selected:
                # Select example closest to mean
                distances = ((examples - mu) ** 2).sum(1)
                selected.append(distances.argmin().item())
            else:
                # Select example that makes selected set closest to mean
                current_mean = examples[selected].mean(0)
                distances = ((current_mean - mu) ** 2).sum()
                selected.append(distances.argmin().item())
                
        return torch.tensor(selected) 