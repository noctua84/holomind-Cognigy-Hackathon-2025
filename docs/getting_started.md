# Getting Started with HoloMind

## Installation

1. Prerequisites:
   - Python 3.8 or higher
   - pip/pipenv
   - PostgreSQL
   - MongoDB

2. Install HoloMind:
```bash
# Clone the repository
git clone https://github.com/yourusername/holomind-v3.git
cd holomind-v3

# Install dependencies using pipenv
pipenv install
pipenv install --dev  # for development dependencies
```

3. Set up databases:
```bash
# PostgreSQL setup
createdb holomind

# MongoDB setup is automatic when first connecting
```

4. Configure environment:
```bash
# Copy example config files
cp config/example.yaml config/config.yaml
# Edit configuration as needed
```

## Quick Start

1. Activate the virtual environment:
```bash
pipenv shell
```

2. Run a basic training example:
```python
from src.core.network import ContinualLearningNetwork
from src.core.trainer import ContinualTrainer
from src.utils.config import ConfigLoader

# Load configuration
config = ConfigLoader().load_all()

# Initialize model and trainer
model = ContinualLearningNetwork(config['model'])
trainer = ContinualTrainer(model, optimizer, config['training'])

# Train on tasks
trainer.train_task(task_id="task1", train_loader=train_loader)
```

## Basic Configuration

Key configuration files are located in the `config/` directory:
- `model_config.yaml`: Network architecture settings
- `training_config.yaml`: Training parameters
- `database_config.yaml`: Database connections
- `monitoring_config.yaml`: Monitoring and visualization settings 