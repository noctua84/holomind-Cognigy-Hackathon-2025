from typing import Dict, List, Any
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import logging

class TrainingDatabase:
    """PostgreSQL database for tracking training metrics and experiments"""
    def __init__(self, connection_params: Dict[str, str]):
        try:
            self.conn = psycopg2.connect(**connection_params)
            self._init_tables()
            logging.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
        
    def _init_tables(self):
        """Initialize database tables"""
        with self.conn.cursor() as cur:
            # Experiments table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status VARCHAR(50),
                    hyperparameters JSONB,
                    metrics JSONB
                )
            ''')
            
            # Training metrics table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    experiment_id INTEGER REFERENCES experiments(experiment_id),
                    task_id VARCHAR(50),
                    epoch INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metrics JSONB,
                    UNIQUE(experiment_id, task_id, epoch)
                )
            ''')
            
            # Model states table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS model_states (
                    state_id SERIAL PRIMARY KEY,
                    experiment_id INTEGER REFERENCES experiments(experiment_id),
                    task_id VARCHAR(50),
                    epoch INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    state_path VARCHAR(255),
                    metrics JSONB,
                    UNIQUE(experiment_id, task_id, epoch)
                )
            ''')
            
        self.conn.commit()
        
    def create_experiment(self, name: str, hyperparameters: Dict[str, Any]) -> int:
        """Create new experiment entry"""
        with self.conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO experiments (name, hyperparameters, status)
                VALUES (%s, %s, %s)
                RETURNING experiment_id
                ''',
                (name, Json(hyperparameters), 'running')
            )
            experiment_id = cur.fetchone()[0]
        self.conn.commit()
        return experiment_id
    
    def log_metrics(self, experiment_id: int, task_id: str, 
                   epoch: int, metrics: Dict[str, float]):
        """Log training metrics"""
        with self.conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO training_metrics 
                (experiment_id, task_id, epoch, metrics)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (experiment_id, task_id, epoch) 
                DO UPDATE SET metrics = EXCLUDED.metrics
                ''',
                (experiment_id, task_id, epoch, Json(metrics))
            )
        self.conn.commit()
    
    def save_model_state(self, experiment_id: int, task_id: str,
                        epoch: int, state_path: str, metrics: Dict[str, float]):
        """Save model state information"""
        with self.conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO model_states 
                (experiment_id, task_id, epoch, state_path, metrics)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id, task_id, epoch) 
                DO UPDATE SET state_path = EXCLUDED.state_path,
                             metrics = EXCLUDED.metrics
                ''',
                (experiment_id, task_id, epoch, state_path, Json(metrics))
            )
        self.conn.commit() 