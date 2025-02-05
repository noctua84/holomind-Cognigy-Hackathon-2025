# Core API Reference

## ContinualLearningNetwork

```python
class ContinualLearningNetwork(nn.Module):
    """Neural network supporting continual learning with progressive architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize network with configuration.
        
        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - feature_dim: Feature extractor output dimension
                - output_dim: Task output dimension
                - feature_extractor: Feature extractor configuration
        """
        
    def add_task(self, task_id: str) -> None:
        """
        Add new task column to network.
        
        Args:
            task_id: Unique identifier for the task
        """
        
    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """
        Forward pass for specific task.
        
        Args:
            x: Input tensor
            task_id: Task identifier
            
        Returns:
            Task-specific output
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