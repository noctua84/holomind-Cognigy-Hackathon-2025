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
from src.core.ewc import EWC
import torch.nn.functional as F
import logging
from pathlib import Path
from datetime import datetime
from src.core.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

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
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Get validated training config
        training_config = self.config_validator.get_training_config()
        self.ewc_lambda = training_config.ewc_lambda
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize monitoring with validated config
        monitoring_config = self.config_validator.get_monitoring_config()
        self.metrics_tracker = MetricsTracker(
            experiment_name=monitoring_config.mlflow.get('experiment_name', 'default'),
            config=monitoring_config.mlflow
        )
        self.visualizer = PerformanceVisualizer(monitoring_config.visualization)
        
        # Setup MLflow with validated config
        if monitoring_config.mlflow:
            mlflow.set_tracking_uri(monitoring_config.mlflow.get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(monitoring_config.mlflow.get('experiment_name', 'default'))
            self.mlflow_client = MlflowClient()
        
        self.current_task = None
        
        # Learning rate scheduling
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        # Gradient clipping
        self.grad_clip_value = config.get('training', {}).get('grad_clip', 1.0)
        
        # Knowledge distillation
        self.temperature = config.get('training', {}).get('distill_temp', 2.0)
        self.distill_lambda = config.get('training', {}).get('distill_lambda', 0.5)
        self.old_model = None
        
        # Initialize checkpoint manager with validated config
        checkpoint_config = self.config_validator.get_checkpoint_config()
        self.checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_config.base_dir,
            save_frequency=checkpoint_config.save_frequency,
            keep_last=checkpoint_config.keep_last,
            save_optimizer=checkpoint_config.save_optimizer,
            save_metrics=checkpoint_config.save_metrics
        )
        
        # Load latest state if available
        self._restore_latest_state()
    
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
        # Create task-specific visualization directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_base_dir = Path(self.validated_config['monitoring']['visualization']['output_dir'].replace('visualizations/', ''))
        task_viz_dir = viz_base_dir / f"task_{task_id}_{timestamp}"
        
        # Create directory if it doesn't exist
        (self.visualizer.output_dir / task_viz_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize task-specific variables
        epochs = epochs or self.config_validator.get_training_config().epochs
        self.current_task = task_id
        self.model.current_task = task_id
        
        # Add new task column if needed
        self.model.add_task(task_id)
        
        # Initialize EWC with validation
        if not hasattr(self, 'ewc'):
            try:
                ewc_lambda = self.config_validator.get_training_config().ewc_lambda
                if not isinstance(ewc_lambda, (int, float)) or ewc_lambda < 0:
                    raise ValueError(f"Invalid EWC lambda value: {ewc_lambda}")
                
                self.ewc = EWC(
                    model=self.model,
                    initial_lambda=ewc_lambda,
                    adaptive_lambda=True,
                    lambda_decay=0.95,
                    importance_scaling=2.0
                )
                logger.info(f"Initialized EWC with lambda={ewc_lambda}")
            except Exception as e:
                logger.error(f"Failed to initialize EWC: {e}")
                raise
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc=f"Task {task_id}", position=0, leave=True)
        
        for epoch in epoch_pbar:
            # Compute Fisher information at the start of training
            if epoch == 0:
                try:
                    self.ewc.compute_fisher(task_id, train_loader)
                    logger.info(f"Computed Fisher information for task {task_id}")
                except Exception as e:
                    logger.error(f"Failed to compute Fisher information: {e}")
                    raise
            
            self.model.train()
            total_loss = 0.0
            
            # Create batch progress bar
            batch_pbar = tqdm(enumerate(train_loader), 
                             desc=f"Epoch {epoch}", 
                             total=len(train_loader),
                             position=1, 
                             leave=False)
            
            for batch_idx, (data, target) in batch_pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data, task_id)
                
                # Combine task loss with EWC loss
                task_loss = self.criterion(output, target)
                ewc_loss = self.ewc.ewc_loss() if epoch > 0 else 0
                loss = task_loss + ewc_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ewc_loss': f'{ewc_loss if epoch > 0 else 0:.4f}'
                })
            
            # Compute average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'val_loss': f'{self._validate(val_loader):.4f}' if val_loader else 'N/A'
            })
            
            # Validate and log metrics
            val_loss = self._validate(val_loader)
            metrics = {
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'ewc_loss': ewc_loss.item() if epoch > 0 else 0,
                **self.metrics_tracker._get_memory_metrics()
            }
            
            # Log metrics using the correct method
            self.metrics_tracker.log_training_metrics(
                metrics=metrics,
                task_id=task_id,
                step=epoch
            )
            
            # Log gradients
            self.metrics_tracker.log_model_gradients(self.model, epoch)
            
            # Generate visualizations periodically
            if epoch % self.validated_config['monitoring']['visualization']['plots']['task_performance']['update_frequency'] == 0:
                # Only keep the latest epoch visualization
                self._cleanup_epoch_visualizations(task_viz_dir)
                
                # Save new visualizations with timestamp
                self.visualizer.plot_task_performance(
                    self.metrics_tracker.metrics_history,
                    save_path=task_viz_dir / f'performance_epoch_{epoch}_{timestamp}.png'
                )
                
                self.visualizer.plot_memory_usage(
                    self.metrics_tracker.metrics_history,
                    save_path=task_viz_dir / f'memory_epoch_{epoch}_{timestamp}.png'
                )
            
            # Log final metrics
            final_metrics = {
                "final_train_loss": avg_loss
            }
            if val_loader:
                final_metrics["final_val_loss"] = val_loss
            
            # Only log metrics with non-None values
            mlflow.log_metrics({k: v for k, v in final_metrics.items() if v is not None})
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # After training, visualize EWC importance in task directory
            self.ewc.visualize_importance(save_dir=self.visualizer.output_dir / task_viz_dir / 'ewc')
            
            # Log additional metrics
            if hasattr(self, 'ewc'):
                mlflow.log_metric(f"ewc_lambda_{task_id}", self.ewc.task_lambdas[task_id])
            
            # Save checkpoint after each epoch
            state = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }
            if hasattr(self, 'ewc'):
                state['ewc_state'] = {
                    'fisher_dict': self.ewc.fisher_dict,
                    'optpar_dict': self.ewc.optpar_dict,
                    'task_lambdas': self.ewc.task_lambdas
                }
            self.checkpoint_manager.save_checkpoint(state, task_id)
            
            # Save training history
            history = {
                'metrics': metrics,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat()
            }
            self.checkpoint_manager.save_history(task_id, history)
        
        # Log final metrics
        final_metrics = {
            "final_train_loss": avg_loss
        }
        if val_loader:
            final_metrics["final_val_loss"] = val_loss
        
        # Only log metrics with non-None values
        mlflow.log_metrics({k: v for k, v in final_metrics.items() if v is not None})
        
        # Log model
        mlflow.pytorch.log_model(self.model, "model")
    
    def _training_step(self, batch):
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        inputs, targets = batch
        outputs = self.model(inputs, self.current_task)
        
        # Task loss
        task_loss = self.criterion(outputs, targets)
        
        # Knowledge distillation loss
        distill_loss = 0
        if self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(inputs)
            distill_loss = self._distillation_loss(outputs, old_outputs)
        
        # EWC loss
        ewc_loss = self.model.get_ewc_loss() if hasattr(self.model, 'get_ewc_loss') else 0
        
        # Total loss
        total_loss = task_loss + self.ewc_lambda * ewc_loss + self.distill_lambda * distill_loss
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
        
        # Optimizer step and LR scheduling
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return total_loss
    
    def _distillation_loss(self, student_outputs, teacher_outputs):
        """Knowledge distillation loss"""
        return nn.KLDivLoss()(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)
    
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
    
    def _cleanup_old_visualizations(self, viz_dir: Path, task_prefix: str):
        """Archive old visualization directories for a task by overriding"""
        if not viz_dir.exists():
            return
        
        # Instead of removing, we'll just override the files in the latest directory
        task_dirs = [d for d in viz_dir.iterdir() 
                    if d.is_dir() and d.name.startswith(task_prefix)]
        
        # No need to do anything if this is the first run
        if not task_dirs:
            return
        
        # Sort by timestamp to get the latest directory
        task_dirs.sort(key=lambda x: x.name.split('_')[-1])
        logger.info(f"Will override visualizations in {task_dirs[-1]}")

    def _cleanup_epoch_visualizations(self, task_dir: Path):
        """Override old epoch visualizations in a task directory"""
        if not task_dir.exists():
            task_dir.mkdir(parents=True, exist_ok=True)
        
        # No need to remove old files, they'll be overridden when saving new ones
        pass 

    def _restore_latest_state(self):
        """Restore the latest training state"""
        latest_task = self.checkpoint_manager.get_latest_task_id()
        if latest_task:
            checkpoint = self.checkpoint_manager.load_checkpoint(latest_task)
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                if 'ewc_state' in checkpoint:
                    self.ewc = EWC(self.model)
                    self.ewc.fisher_dict = checkpoint['ewc_state']['fisher_dict']
                    self.ewc.optpar_dict = checkpoint['ewc_state']['optpar_dict']
                    self.ewc.task_lambdas = checkpoint['ewc_state']['task_lambdas']
                logger.info(f"Restored state from task {latest_task}") 