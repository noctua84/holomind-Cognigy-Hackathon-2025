from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
from pathlib import Path
import os
from alembic import command
from alembic.config import Config
from .models import Base, Experiment, Run, MetricsRecord, ModelArtifact

logger = logging.getLogger(__name__)

class MetricsDB:
    # ... rest of the MetricsDB class ... 

    def _run_migrations(self):
        """Run database migrations"""
        try:
            # Get the migrations directory relative to this file
            migrations_dir = os.path.join(
                os.path.dirname(__file__), 
                'migrations'
            )
            
            # Create Alembic config
            alembic_cfg = Config()
            alembic_cfg.set_main_option('script_location', migrations_dir)
            alembic_cfg.set_main_option('sqlalchemy.url', str(self.engine.url))
            
            # Run migrations
            command.upgrade(alembic_cfg, 'head')
            
        except Exception as e:
            logger.error(f"Failed to run migrations: {e}")
            raise 