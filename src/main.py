import torch
import torch.optim as optim
from pathlib import Path
import logging
from datetime import datetime

# Project imports
from src.utils.config import ConfigLoader
from src.core.network import ContinualLearningNetwork
from src.core.trainer import ContinualTrainer
from src.data.dataloader import DataManager
from src.database.manager import DatabaseManager

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config['logging']['checkpoints']['dir']) / 'training.log'),
            logging.StreamHandler()
        ]
    )

def setup_device(config):
    """Setup compute device"""
    device = config['system']['compute']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        logging.warning("MPS not available, falling back to CPU")
        device = 'cpu'
    
    return torch.device(device)

def main():
    # Load configurations
    config_loader = ConfigLoader()
    config = config_loader.load_all()
    
    # Setup logging and device
    setup_logging(config)
    device = setup_device(config)
    
    logging.info(f"Using device: {device}")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(config['database'])
        
        # Start experiment
        experiment_name = f"holomind_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db_manager.start_experiment(
            name=experiment_name,
            model_config=config['model']
        )
        
        # Initialize model with configurations
        model = ContinualLearningNetwork(config['model']['network'])
        model = model.to(device)
        
        # Setup optimizer
        optimizer_config = config['training']['training']['optimizer']
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['training']['learning_rate'],
            betas=optimizer_config['betas'],
            weight_decay=optimizer_config['weight_decay']
        )
        
        # Initialize trainer
        trainer = ContinualTrainer(
            model=model,
            optimizer=optimizer,
            config=config['training']['training']
        )
        
        logging.info("Model and trainer initialized successfully")
        
        # Initialize data manager
        data_manager = DataManager(config['data']['data'])
        
        # Example training loop (assuming sequential tasks)
        tasks = ["task1", "task2", "task3"]  # Define your tasks
        
        for task_id in tasks:
            logging.info(f"Starting training for task: {task_id}")
            
            # Get data loaders for current task
            train_loader, val_loader, test_loader = data_manager.get_task_loaders(task_id)
            
            for epoch in range(config['training']['training']['epochs']):
                # Train on current task
                train_loss, val_loss, accuracy = trainer.train_task(
                    task_id=task_id,
                    train_loader=train_loader,
                    val_loader=val_loader
                )
                
                # Log metrics
                metrics = {
                    'loss': train_loss,
                    'val_loss': val_loss if val_loader else None,
                    'accuracy': accuracy
                }
                db_manager.log_training_metrics(task_id, epoch, metrics)
                
                # Save checkpoint periodically
                if (epoch + 1) % config['training']['training']['checkpointing']['save_frequency'] == 0:
                    db_manager.save_model_checkpoint(
                        task_id=task_id,
                        epoch=epoch,
                        state_dict=model.state_dict(),
                        metrics=metrics
                    )
            
            # Evaluate on test set
            test_loss = trainer._validate(test_loader)
            logging.info(f"Test loss for task {task_id}: {test_loss:.4f}")
            
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 