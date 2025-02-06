# Data Management API

## DataManager

```python
class DataManager:
    """Manages data loading and versioning"""
    
    def __init__(self, config: Dict):
        """
        Initialize data manager.
        
        Args:
            config: Configuration containing:
                - datasets: Dataset configuration
                - preprocessing: Preprocessing pipeline config
                - dataloader: DataLoader configuration
        """
    
    def prepare_task_data(self, task_id: str, data: torch.Tensor, 
                         targets: torch.Tensor):
        """
        Prepare and version task data using DVC.
        
        Args:
            task_id: Task identifier
            data: Feature tensors
            targets: Target tensors
        """
    
    def get_task_loaders(self, task_id: str) -> Tuple[DataLoader, 
                                                     Optional[DataLoader], 
                                                     DataLoader]:
        """
        Get data loaders for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
``` 