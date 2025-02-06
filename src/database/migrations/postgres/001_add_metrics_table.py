from ..base import Migration

class AddMetricsTable(Migration):
    def up(self) -> bool:
        """Add metrics table with JSON support"""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS metrics (
                            id SERIAL PRIMARY KEY,
                            experiment_id INTEGER REFERENCES experiments(id),
                            task_id VARCHAR(255),
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metrics JSONB
                        )
                    """)
                    cur.execute("""
                        CREATE INDEX idx_metrics_experiment 
                        ON metrics(experiment_id)
                    """)
            return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def down(self) -> bool:
        """Remove metrics table"""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DROP TABLE IF EXISTS metrics")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False 