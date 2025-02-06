# Core API Reference

## ContinualLearningNetwork

```python
class ContinualLearningNetwork(nn.Module):
    """Neural network supporting continual learning with progressive architecture"""
    
    def __init__(self, config: Dict):
        """
        Initialize network with configuration.
        
        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - feature_dim: Feature dimension for feature extractor
                - output_dim: Task output dimension
                - memory: Memory configuration (size, feature_dim)
                - task_columns: Task-specific column configuration
        """
    
    def add_task(self, task_id: str):
        """
        Add new task column to network.
        
        Args:
            task_id: Unique identifier for the task
        """
    
    def update_importance(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Updates weight importance metrics for EWC.
        
        Args:
            loss: Loss tensor to compute gradients from
            retain_graph: Whether to retain computation graph
        """

class ExternalMemory(nn.Module):
    """Memory system using sparse tensors for efficient storage"""
    
    def __init__(self, memory_size: int, feature_dim: int):
        """
        Initialize sparse memory system.
        
        Args:
            memory_size: Number of memory slots
            feature_dim: Dimension of feature vectors
        """
    
    def query(self, features: torch.Tensor) -> torch.Tensor:
        """
        Query memory using importance scoring.
        
        Args:
            features: Input features to query with
            
        Returns:
            Retrieved memory vectors
        """
```

## ContinualTrainer

```python
class ContinualTrainer:
    """Trainer class for continual learning"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            optimizer: PyTorch optimizer
            config: Training configuration
        """
        
    def train_task(self,
                   task_id: str,
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None) -> None:
        """
        Train model on specific task.
        
        Args:
            task_id: Task identifier
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
``` 