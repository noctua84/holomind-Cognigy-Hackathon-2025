from typing import Dict, Any, Optional
import logging
from pathlib import Path
import torch
import platform
from psycopg2.extras import Json
import h5py
from datetime import datetime, UTC

from .postgres import TrainingDatabase, PostgresConnector
from .mongo import ModelArchiveDB, MongoConnector
from .migrations import MigrationManager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages interactions with both PostgreSQL and MongoDB databases"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize connections only if enabled
        self.training_db = None
        self.model_db = None
        
        if config.get('postgres', {}).get('enabled', False):
            self.training_db = TrainingDatabase(config['postgres'])
        
        if config.get('mongo', {}).get('enabled', False):
            mongo_config = config['mongo']
            # Construct MongoDB URI from config
            uri = f"mongodb://{mongo_config.get('host', 'localhost')}:{mongo_config.get('port', 27017)}"
            self.model_db = ModelArchiveDB(
                uri=uri,
                database=mongo_config.get('database', 'holomind')
            )
        
        self.current_experiment_id = None
        
    def _ensure_connections(self) -> bool:
        """Ensure all enabled database connections are working"""
        success = True
        
        if self.training_db:
            try:
                # Test PostgreSQL connection
                with self.training_db.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                logger.info("PostgreSQL connection verified")
            except Exception as e:
                logger.error(f"PostgreSQL connection failed: {e}")
                success = False
        
        if self.model_db:
            try:
                # Test MongoDB connection
                self.model_db.client.server_info()
                logger.info("MongoDB connection verified")
            except Exception as e:
                logger.error(f"MongoDB connection failed: {e}")
                success = False
        
        return success

    def initialize_databases(self) -> bool:
        """Initialize and migrate databases"""
        if not self._ensure_connections():
            return False
        
        # Run migrations
        migration_manager = MigrationManager(
            postgres_connector=self.postgres if hasattr(self, 'postgres') else None,
            mongo_connector=self.mongo if hasattr(self, 'mongo') else None
        )
        
        if not migration_manager.migrate():
            logger.error("Database migrations failed")
            return False
        
        return True
        
    def start_experiment(self, name: str, model_config: Dict[str, Any]):
        """Start new training experiment"""
        try:
            # Create experiment in PostgreSQL
            self.current_experiment_id = self.training_db.create_experiment(
                name=name,
                hyperparameters=model_config
            )
            
            # Save initial architecture in MongoDB
            self.model_db.save_model_architecture(
                architecture_id=f"exp_{self.current_experiment_id}",
                architecture=model_config
            )
            
            logging.info(f"Started experiment {name} with ID {self.current_experiment_id}")
            
        except Exception as e:
            logging.error(f"Failed to start experiment: {str(e)}")
            raise
            
    def log_training_metrics(self, task_id: str, epoch: int, 
                           metrics: Dict[str, float]):
        """Log training metrics to PostgreSQL"""
        if not self.current_experiment_id:
            raise ValueError("No active experiment")
            
        self.training_db.log_metrics(
            experiment_id=self.current_experiment_id,
            task_id=task_id,
            epoch=epoch,
            metrics=metrics
        )
        
    def save_model_checkpoint(self, task_id: str, epoch: int,
                            state_dict: Dict[str, Any], metrics: Dict[str, float],
                            checkpoint_dir: Optional[Path] = None):
        """Save model checkpoint and record in both databases"""
        if not self.current_experiment_id:
            raise ValueError("No active experiment")
            
        # Create checkpoint path
        if checkpoint_dir is None:
            checkpoint_dir = Path(self.config['checkpoints']['dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save large tensors efficiently with h5py
        h5_path = checkpoint_dir / f"task_{task_id}_epoch_{epoch}.h5"
        with h5py.File(h5_path, 'w') as f:
            # Create groups for organization
            state_group = f.create_group('state_dict')
            for key, tensor in state_dict.items():
                state_group.create_dataset(key, data=tensor.cpu().numpy())
            
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['task_id'] = task_id
            meta_group.attrs['epoch'] = epoch
            meta_group.attrs['timestamp'] = str(datetime.now(UTC))
            
            # Save metrics as attributes
            metrics_group = f.create_group('metrics')
            for key, value in metrics.items():
                metrics_group.attrs[key] = value
        
        # Record in PostgreSQL
        self.training_db.save_model_state(
            experiment_id=self.current_experiment_id,
            task_id=task_id,
            epoch=epoch,
            state_path=str(h5_path),
            metrics=metrics
        )
        
        # Record in MongoDB
        self.model_db.save_training_state(
            architecture_id=f"exp_{self.current_experiment_id}",
            state={
                'task_id': task_id,
                'epoch': epoch,
                'checkpoint_path': str(h5_path)
            },
            metadata={
                'metrics': metrics,
                'environment': self._get_environment_info()
            }
        )
        
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu'
        }

    def save_metrics(self, task_id: str, metrics: Dict):
        """Save training metrics"""
        # Create experiment if it doesn't exist
        experiment_id = self._ensure_experiment_exists()
        
        # Save metrics
        self.training_db.log_metrics(
            experiment_id=experiment_id,
            task_id=task_id,
            epoch=0,
            metrics=metrics
        )

    def _ensure_experiment_exists(self) -> int:
        """Ensure test experiment exists and return its ID"""
        with self.training_db.conn.cursor() as cur:
            # Check if test experiment exists
            cur.execute("SELECT experiment_id FROM experiments WHERE name = 'test'")
            result = cur.fetchone()
            
            if result:
                return result[0]
            
            # Create new test experiment
            cur.execute(
                """
                INSERT INTO experiments (name, hyperparameters, status)
                VALUES (%s, %s, %s)
                RETURNING experiment_id
                """,
                ('test', Json({}), 'running')
            )
            self.training_db.conn.commit()
            return cur.fetchone()[0]

    def get_metrics(self, task_id: str) -> Dict:
        """Retrieve training metrics"""
        # For testing, return the last metrics for the task
        with self.training_db.conn.cursor() as cur:
            cur.execute(
                """
                SELECT metrics FROM training_metrics 
                WHERE task_id = %s 
                ORDER BY epoch DESC LIMIT 1
                """,
                (task_id,)
            )
            result = cur.fetchone()
            return result[0] if result else {}
            
    def save_checkpoint(self, model_id: str, state_dict: Dict):
        """Save model checkpoint"""
        self.model_db.save_model_architecture(
            architecture_id=model_id,
            architecture=state_dict
        )
        
    def load_checkpoint(self, model_id: str) -> Optional[Dict]:
        """Load model checkpoint"""
        results = self.model_db.query_architecture_evolution(model_id)
        if results:
            return results[0]['structure']  # Get latest version
        return None 