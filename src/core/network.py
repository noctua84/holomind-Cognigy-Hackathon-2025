from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from numba import jit

class ContinualLearningNetwork(nn.Module):
    """
    Core neural network implementation for HoloMind v3.
    Combines progressive networks, external memory, and weight consolidation.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        # Task-specific columns (Progressive Networks component)
        self.task_columns = nn.ModuleDict()
        
        # External memory system
        self.memory = ExternalMemory(
            memory_size=config['memory']['size'],
            feature_dim=config['memory']['feature_dim']
        )
        
        # Replace the fisher tracking with FisherDiagonal
        self.fisher_tracker = FisherDiagonal(self)
        self.ewc_loss = None

    def _build_feature_extractor(self) -> nn.Module:
        """Constructs the shared feature extraction network"""
        return nn.Sequential(
            nn.Linear(self.config['input_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.config['feature_dim'])
        )
    
    def add_task(self, task_id: str):
        """Adds a new task column with lateral connections to previous columns"""
        if task_id in self.task_columns:
            return
            
        new_column = TaskColumn(
            input_dim=self.config['feature_dim'],  # This will be doubled internally
            hidden_dims=self.config['task_columns']['hidden_dims'],
            output_dim=self.config['output_dim'],
            prev_columns=list(self.task_columns.values())
        )
        
        self.task_columns[task_id] = new_column
        
    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        # Extract base features
        features = self.feature_extractor(x)
        
        # Query external memory
        memory_output = self.memory.query(features)
        
        # Combine features with memory context
        enhanced_features = torch.cat([features, memory_output], dim=-1)
        
        # Process through task-specific column
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not initialized")
            
        output = self.task_columns[task_id](enhanced_features)
        
        return output
    
    def update_importance(self, loss: torch.Tensor, retain_graph: bool = False):
        """Updates weight importance metrics for EWC"""
        # Make sure to compute gradients first
        loss.backward(retain_graph=retain_graph)
        
        # Update Fisher information
        for name, param in self.named_parameters():
            if param.grad is not None:
                fisher = param.grad.data.pow(2)
                if name in self.fisher_tracker.fisher_diagonal:
                    self.fisher_tracker.fisher_diagonal[name] += fisher
                else:
                    self.fisher_tracker.fisher_diagonal[name] = fisher
                    
        # Zero gradients after computing Fisher
        self.zero_grad()

    def prepare_ewc_loss(self, data_loader, criterion):
        """Prepare EWC loss for the next task by computing Fisher diagonal"""
        # Compute Fisher information
        self.fisher_tracker.compute_fisher(data_loader, criterion)
        
        # Initialize EWC loss with current Fisher values
        self.ewc_loss = EWCLoss(self, self.fisher_tracker.fisher_diagonal)
    
    def get_ewc_loss(self) -> Optional[torch.Tensor]:
        """Get the EWC loss if available"""
        if self.ewc_loss is not None:
            return self.ewc_loss()
        return None

class TaskColumn(nn.Module):
    """Implementation of a single task column with lateral connections"""
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 prev_columns: Optional[list] = None):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim * 2  # Double because of concatenated memory features
        
        # Build main layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim
            
        self.output_layer = nn.Linear(current_dim, output_dim)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through task column"""
        current = x
        
        for layer in self.layers:
            current = layer(current)
            
        return self.output_layer(current)

class ExternalMemory(nn.Module):
    """Memory system using efficient tensor operations"""
    def __init__(self, memory_size: int, feature_dim: int):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(memory_size, feature_dim))
        self.usage_counts = torch.zeros(memory_size)
        
    def update(self, features: torch.Tensor, importance: torch.Tensor):
        """Update memory using dense operations"""
        # Find least used locations
        _, indices = torch.topk(self.usage_counts, 
                              k=features.size(0), 
                              largest=False)
        
        # Update memory directly
        self.memory.data[indices] = features
        
        # Update usage counts
        self.usage_counts[indices] = importance
        
    def query(self, features: torch.Tensor) -> torch.Tensor:
        """Query memory using importance scoring"""
        # Convert to numpy for numba acceleration
        features_np = features.detach().cpu().numpy()
        memory_np = self.memory.detach().cpu().numpy()
        
        # Compute importance scores
        scores = self._compute_importance_scores(features_np, memory_np)
        scores = torch.from_numpy(scores).to(features.device)
        
        # Get top k matches
        _, indices = torch.topk(scores, k=min(self.memory.size(0), features.size(0)))
        return self.memory[indices]

    @staticmethod
    @jit(nopython=True)
    def _compute_importance_scores(features: np.ndarray, memory: np.ndarray) -> np.ndarray:
        """Compute importance scores using numba-accelerated function"""
        scores = np.zeros(len(memory))
        for i in range(len(memory)):
            diff = features - memory[i]
            scores[i] = -np.sum(diff * diff)  # Negative L2 distance
        return scores

class LateralAdapter(nn.Module):
    """Adapter for lateral connections between task columns"""
    def __init__(self, source_column: TaskColumn):
        super().__init__()
        self.source = source_column
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.source(x) * 0.1  # Scale lateral connections 

class FisherDiagonal:
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_diagonal = {}
        self.parameter_importance = {}
        
    def compute_fisher(self, data_loader, criterion):
        # Initialize Fisher diagonal for each parameter
        for name, param in self.model.named_parameters():
            self.fisher_diagonal[name] = torch.zeros_like(param.data)
            
        self.model.eval()  # Set to evaluation mode
        for batch in data_loader:
            inputs, targets = batch
            
            # Forward pass with task_id if it's a batch tuple of 3 elements
            if len(batch) == 3:
                inputs, targets, task_id = batch
                outputs = self.model(inputs, task_id)
            else:
                outputs = self.model(inputs, self.model.current_task)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_diagonal[name] += param.grad.data.pow(2)
                    
        # Average the Fisher values
        n_samples = len(data_loader.dataset)
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] /= n_samples

class EWCLoss:
    def __init__(self, model: nn.Module, fisher_diagonal: Dict[str, torch.Tensor]):
        self.model = model
        self.fisher_diagonal = fisher_diagonal
        self.old_parameters = {}
        
        # Store current parameter values
        for name, param in model.named_parameters():
            self.old_parameters[name] = param.data.clone()
            
    def __call__(self) -> torch.Tensor:
        loss = 0
        for name, param in self.model.named_parameters():
            # Skip if parameter wasn't in previous task
            if name not in self.fisher_diagonal:
                continue
                
            # Compute EWC loss: Fisher * (θ - θ_old)²
            loss += (self.fisher_diagonal[name] * 
                    (param - self.old_parameters[name]).pow(2)).sum()
                    
        return loss 