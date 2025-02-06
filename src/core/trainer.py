from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.monitoring.metrics import MetricsTracker
from src.monitoring.visualization import PerformanceVisualizer
import mlflow
from mlflow.tracking import MlflowClient
import ray
from src.utils.config_validator import ConfigValidator

class ContinualTrainer:
    """Trainer class for continuous learning with HoloMind"""
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: Dict):
        """Initialize trainer with validated configuration"""
        self.config_validator = ConfigValidator(config)
        self.validated_config = self.config_validator.validate()
        
        self.model = model
        self.optimizer = optimizer
        
        # Get validated training config
        training_config = self.config_validator.get_training_config()
        self.ewc_lambda = training_config.ewc_lambda
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize monitoring with validated config
        monitoring_config = self.config_validator.get_monitoring_config()
        self.metrics_tracker = MetricsTracker(monitoring_config.tensorboard)
        self.visualizer = PerformanceVisualizer(monitoring_config.visualization)
        
        # Setup MLflow with validated config
        if monitoring_config.mlflow:
            mlflow.set_tracking_uri(monitoring_config.mlflow.get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(monitoring_config.mlflow.get('experiment_name', 'default'))
            self.mlflow_client = MlflowClient()
        
        self.current_task = None
    
    def _setup_distributed(self):
        """Setup distributed training if enabled"""
        if self.validated_config['system']['parallel']['distributed_training']:
            ray.init()
    
    def _distributed_training_step(self, batch):
        """Distributed training step using Ray"""
        @ray.remote(num_gpus=1)
        def train_shard(model_shard, batch_shard):
            outputs = model_shard(batch_shard[0])
            loss = self.criterion(outputs, batch_shard[1])
            return loss
            
        # Split batch across workers
        batch_shards = self._split_batch(batch)
        futures = [train_shard.remote(self.model, shard) for shard in batch_shards]
        losses = ray.get(futures)
        return sum(losses) / len(losses)
    
    def train_task(self, 
                   task_id: str,
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   epochs: int = None):
        """Train on a specific task while preserving knowledge"""
        training_config = self.config_validator.get_training_config()
        
        with mlflow.start_run(run_name=f"task_{task_id}"):
            # Log parameters
            mlflow.log_params({
                "task_id": task_id,
                "epochs": epochs or training_config.epochs,
                "batch_size": training_config.batch_size,
                "learning_rate": training_config.learning_rate,
                "ewc_lambda": training_config.ewc_lambda
            })
            
            epochs = epochs or training_config.epochs
            self.current_task = task_id
            self.model.current_task = task_id
            
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
                
                # Log metrics
                step = epoch + len(self.metrics_tracker.metrics_history.get(task_id, []))
                metrics = {
                    'train_loss': train_loss / len(train_loader),
                    'train_accuracy': None,  # Assuming no accuracy metric in the training loop
                    'val_loss': val_loss if val_loader else None,
                    'val_accuracy': None  # Assuming no accuracy metric in the validation loop
                }
                self._log_metrics(metrics, step)
                
                # Log gradients
                self.metrics_tracker.log_model_gradients(self.model, step)
                
                # Generate visualizations periodically
                if epoch % self.validated_config['monitoring']['visualization']['plots']['task_performance']['update_frequency'] == 0:
                    self.visualizer.plot_task_performance(
                        self.metrics_tracker.metrics_history,
                        save_path=f'task_performance_epoch_{epoch}.png'
                    )
                    
                    self.visualizer.plot_memory_usage(
                        self.metrics_tracker.metrics_history,
                        save_path=f'memory_usage_epoch_{epoch}.png'
                    )
            
            # Log final metrics
            final_metrics = {
                "final_train_loss": train_loss / len(train_loader)
            }
            if val_loader:
                final_metrics["final_val_loss"] = val_loss
            
            # Only log metrics with non-None values
            mlflow.log_metrics({k: v for k, v in final_metrics.items() if v is not None})
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
    
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
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow and tensorboard"""
        # Filter out None values for MLflow
        valid_metrics = {k: v for k, v in metrics.items() if v is not None}
        metrics_with_step = {
            **valid_metrics,
            "step": step
        }
        
        # Log to MLflow
        if valid_metrics:  # Only log if we have valid metrics
            mlflow.log_metrics(metrics_with_step)
        
        # Log to metrics tracker (can handle None values)
        self.metrics_tracker.log_training_metrics(
            metrics=metrics,
            task_id=self.current_task,
            step=step
        ) 