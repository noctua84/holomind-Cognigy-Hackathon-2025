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
from src.utils.config_validator import ConfigValidator

logger = logging.getLogger(__name__)

def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_dir = Path(log_config.get('directory', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'holomind.log'),
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
    """Main entry point"""
    # Load and validate configuration
    try:
        config = ConfigValidator.from_yaml('config.yaml').validate()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
        
    # Setup logging
    setup_logging(config)
    logger.info("Starting HoloMind...")
    
    # Initialize databases
    db_manager = DatabaseManager(config.get('database', {}))
    if not db_manager.initialize_databases():
        logger.error("Database initialization failed")
        return False
    
    logger.info("System initialization complete")
    return True

if __name__ == "__main__":
    main() 