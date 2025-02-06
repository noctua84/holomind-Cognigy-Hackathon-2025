import pytest
import torch
import tempfile
import mlflow
from mlflow.tracking import MlflowClient
from torch.utils.data import TensorDataset, DataLoader

from src.core.trainer import ContinualTrainer

@pytest.fixture
def mlflow_trainer(model, test_config):
    # Setup temporary MLflow tracking
    tracking_uri = f"sqlite:///{tempfile.mktemp()}"
    test_config['monitoring']['mlflow']['tracking_uri'] = tracking_uri
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return ContinualTrainer(model, optimizer, test_config)

def test_mlflow_experiment_tracking(mlflow_trainer, sample_batch):
    """Test MLflow experiment tracking"""
    task_id = "test_task"
    mlflow_trainer.model.add_task(task_id)
    
    # Create proper data loader
    inputs, targets = sample_batch
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=16)
    
    # Train for one epoch
    mlflow_trainer.train_task(task_id, train_loader, epochs=1)
    
    # Verify metrics were logged
    client = MlflowClient()
    experiment = client.get_experiment_by_name("test")  # From test_config fixture
    assert experiment is not None
    
    # Get the latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    assert len(runs) > 0
    assert "train_loss" in runs[0].data.metrics 