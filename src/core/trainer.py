from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class ContinualTrainer:
    """Trainer class for continuous learning with HoloMind"""
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: Dict):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        self.current_task = None
        self.ewc_lambda = config['training']['ewc_lambda']
        self.criterion = nn.CrossEntropyLoss()  # Add default criterion
        
    def train_task(self, 
                   task_id: str,
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   epochs: int = None):
        """Train on a specific task while preserving knowledge"""
        epochs = epochs or self.config['training']['epochs']
        self.current_task = task_id
        self.model.current_task = task_id  # Set current task in model
        
        # Add new task column if needed
        self.model.add_task(task_id)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                loss = self._training_step(batch)
                train_loss += loss.item()
                
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
                      f"Val Loss = {val_loss:.4f}")
            
            # Prepare EWC loss for next task at the end of training
            if epoch == epochs - 1:
                self.model.prepare_ewc_loss(train_loader, self.criterion)
    
    def _training_step(self, batch) -> torch.Tensor:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        inputs, targets = batch
        outputs = self.model(inputs, self.current_task)
        
        # Calculate task loss
        task_loss = self.criterion(outputs, targets)
        
        # Add EWC loss if available
        ewc_loss = self.model.get_ewc_loss()
        total_loss = task_loss
        if ewc_loss is not None:
            total_loss += self.ewc_lambda * ewc_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate current model state"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = self.model(inputs, self.current_task)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader) 