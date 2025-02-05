import torch
import torch.optim as optim
from pathlib import Path
import logging

# Project imports
from src.utils.config import ConfigLoader
from src.core.network import ContinualLearningNetwork
from src.core.trainer import ContinualTrainer

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
        
        # TODO: Add data loading and training loop here
        # This will be implemented when we create the data loading modules
        
    except Exception as e:
        logging.error(f"Error during initialization: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 