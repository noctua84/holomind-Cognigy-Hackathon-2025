from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class TrainingDatabase:
    """PostgreSQL database for tracking training metrics and experiments"""
    def __init__(self, config: Dict[str, Any]):
        # Filter out non-connection parameters
        connection_params = {
            k: v for k, v in config.items() 
            if k in ['host', 'port', 'database', 'user', 'password']
        }
        
        try:
            self.conn = psycopg2.connect(**connection_params)
            self.cur = self.conn.cursor()
            self._create_tables()
            logging.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
        
    def _create_tables(self):
        """Initialize database tables"""
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                config JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id SERIAL PRIMARY KEY,
                experiment_id INTEGER REFERENCES experiments(id),
                task_id VARCHAR(255),
                epoch INTEGER,
                metrics JSONB,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
    def create_experiment(self, name: str, config: Dict) -> int:
        """Create a new experiment and return its ID"""
        self.cur.execute(
            "INSERT INTO experiments (name, config) VALUES (%s, %s) RETURNING id",
            (name, Json(config))
        )
        experiment_id = self.cur.fetchone()[0]
        self.conn.commit()
        return experiment_id
    
    def log_metrics(self, experiment_id: int, task_id: str, 
                   epoch: int, metrics: Dict[str, float]):
        """Log metrics for an experiment"""
        self.cur.execute(
            """INSERT INTO metrics 
               (experiment_id, task_id, epoch, metrics) 
               VALUES (%s, %s, %s, %s)""",
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

    def close(self):
        """Close database connection"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

class PostgresConnector:
    """PostgreSQL connector for storing metrics and model checkpoints"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conn = None
        self.cur = None
        
        # Filter connection parameters
        self.connection_params = {
            'host': config.get('host', 'localhost'),
            'port': config.get('port', 5432),
            'database': config.get('database', 'holomind'),
            'user': config.get('user', 'postgres'),
            'password': config.get('password', '')
        }
        
    def get_connection(self):
        """Get a database connection"""
        try:
            # Try to connect to specified database
            return psycopg2.connect(**self.connection_params)
        except psycopg2.OperationalError as e:
            if "database" in str(e):
                # If database doesn't exist, connect to default postgres db
                temp_params = self.connection_params.copy()
                temp_params['database'] = 'postgres'
                try:
                    conn = psycopg2.connect(**temp_params)
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute(f"CREATE DATABASE {self.connection_params['database']}")
                    conn.close()
                    # Now try connecting to the new database
                    return psycopg2.connect(**self.connection_params)
                except Exception as e2:
                    logger.error(f"Failed to create database: {e2}")
                    raise
            else:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.conn is not None and self.cur is not None
        
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                key TEXT PRIMARY KEY,
                data JSONB
            )
        """)
        self.conn.commit()
        
    def save(self, key: str, data: Dict):
        """Save data to PostgreSQL"""
        if not self.is_connected():
            raise Exception("Not connected to PostgreSQL")
            
        self.cur.execute(
            "INSERT INTO metrics (key, data) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET data = %s",
            (key, json.dumps(data), json.dumps(data))
        )
        self.conn.commit()
        
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve data from PostgreSQL"""
        if not self.is_connected():
            raise Exception("Not connected to PostgreSQL")
            
        self.cur.execute("SELECT data FROM metrics WHERE key = %s", (key,))
        result = self.cur.fetchone()
        return json.loads(result[0]) if result else None
        
    def test_connection(self) -> bool:
        """Test if database exists and is accessible"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception:
            return False
        
    def setup_database(self):
        """Create database and required tables"""
        try:
            # Connect to default database to create new one
            with psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database="postgres"
            ) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Create database if it doesn't exist
                    cur.execute(f"CREATE DATABASE {self.config['database']}")
                    
            # Now connect to new database and create tables
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS tasks (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS models (
                            id SERIAL PRIMARY KEY,
                            task_id INTEGER REFERENCES tasks(id),
                            path VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}")
            return False 