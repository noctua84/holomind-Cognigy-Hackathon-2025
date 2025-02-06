import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
import logging
import mlflow
from src.utils.config_validator import ConfigValidator
from src.core.network import ContinualLearningNetwork
from src.core.trainer import ContinualTrainer
from src.data.synthetic import create_task_datasets
from src.database.manager import DatabaseManager
from src.monitoring.metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_mlflow():
    """Cleanup any active MLflow runs"""
    try:
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"Ending active run: {active_run.info.run_id}")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error during MLflow cleanup: {e}")

def main():
    # Cleanup any existing MLflow runs
    cleanup_mlflow()
    
    try:
        # Load and validate configuration
        config = ConfigValidator.from_yaml('config.yaml').validate()
        
        # Initialize databases
        db_manager = DatabaseManager(config.get('database', {}))
        if not db_manager.initialize_databases():
            logger.error("Database initialization failed")
            return
        
        # Create synthetic datasets
        data_config = {
            'num_tasks': 5,
            'samples_per_task': 1000,
            'input_dim': config['model']['network']['input_dim'],
            'num_classes': config['model']['network']['output_dim'],
            'batch_size': config['training']['batch_size'],
            'train_split': config['data'].get('train_split', 0.7),
            'val_split': config['data'].get('val_split', 0.15)
        }
        
        task_loaders = create_task_datasets(data_config)
        
        # Initialize model and trainer
        model = ContinualLearningNetwork(config['model']['network'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        trainer = ContinualTrainer(model, optimizer, config)
        
        # Train on each task sequentially
        for task_id, loaders in task_loaders.items():
            logger.info(f"Training on {task_id}")
            trainer.train_task(
                task_id=task_id,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                epochs=config['training']['epochs']
            )
            
            # Evaluate on all previous tasks
            all_task_metrics = {}
            for prev_task_id, prev_loaders in task_loaders.items():
                if prev_task_id <= task_id:
                    val_loss = trainer._validate(prev_loaders['val'])
                    logger.info(f"Validation loss on {prev_task_id}: {val_loss:.4f}")
                    all_task_metrics[f"val_loss_{prev_task_id}"] = val_loss
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Always cleanup MLflow runs
        cleanup_mlflow()

if __name__ == "__main__":
    main() 