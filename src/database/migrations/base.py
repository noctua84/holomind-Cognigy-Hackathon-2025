from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import importlib
import pkgutil
from pathlib import Path
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class Migration(ABC):
    """Base class for database migrations"""
    
    @abstractmethod
    def up(self) -> bool:
        """Perform the migration"""
        pass
        
    @abstractmethod
    def down(self) -> bool:
        """Rollback the migration"""
        pass

class MigrationManager:
    """Manages database migrations"""
    def __init__(self, postgres_connector=None, mongo_connector=None):
        self.postgres = postgres_connector
        self.mongo = mongo_connector
        
    def _get_current_version(self, db_type: str) -> int:
        """Get current migration version from version table"""
        if db_type == 'postgres' and self.postgres:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS schema_version (
                            version INTEGER PRIMARY KEY,
                            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    cur.execute("SELECT MAX(version) FROM schema_version")
                    version = cur.fetchone()[0]
                    return version if version is not None else 0
        elif db_type == 'mongo' and self.mongo:
            db = self.mongo.client[self.mongo.db_name]
            version_doc = db.schema_version.find_one({}, sort=[('version', -1)])
            return version_doc['version'] if version_doc else 0
        return 0
    
    def _load_migrations(self, db_type: str) -> Dict[int, Migration]:
        """Load all migration classes for the given database type"""
        migrations = {}
        migrations_path = Path(__file__).parent / db_type
        
        if not migrations_path.exists():
            return migrations
            
        # Import all migration modules
        for _, name, _ in pkgutil.iter_modules([str(migrations_path)]):
            if name.startswith('_'):
                continue
                
            try:
                module = importlib.import_module(f".{db_type}.{name}", package="src.database.migrations")
                
                # Find migration class in module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Migration) and attr != Migration:
                        # Extract version from filename (e.g., 001_add_metrics.py -> 1)
                        version = int(name.split('_')[0])
                        migrations[version] = attr()
                        break
                        
            except Exception as e:
                logger.error(f"Failed to load migration {name}: {e}")
                
        return migrations
    
    def _record_migration(self, db_type: str, version: int):
        """Record that a migration has been applied"""
        if db_type == 'postgres':
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO schema_version (version) VALUES (%s)",
                        (version,)
                    )
        elif db_type == 'mongo':
            db = self.mongo.client[self.mongo.db_name]
            db.schema_version.insert_one({
                'version': version,
                'applied_at': datetime.now(UTC)
            })
    
    def migrate(self, target_version: int = None) -> bool:
        """Run all pending migrations or up to target_version"""
        success = True
        
        for db_type in ['postgres', 'mongo']:
            if getattr(self, db_type):
                current = self._get_current_version(db_type)
                migrations = self._load_migrations(db_type)
                
                if target_version is None:
                    target_version = max(migrations.keys()) if migrations else 0
                
                logger.info(f"Migrating {db_type} from version {current} to {target_version}")
                
                try:
                    if current < target_version:
                        # Apply forward migrations
                        for version in range(current + 1, target_version + 1):
                            if version in migrations:
                                migration = migrations[version]
                                if migration.up():
                                    self._record_migration(db_type, version)
                                else:
                                    success = False
                                    break
                    elif current > target_version:
                        # Apply rollback migrations
                        for version in range(current, target_version, -1):
                            if version in migrations:
                                migration = migrations[version]
                                if not migration.down():
                                    success = False
                                    break
                except Exception as e:
                    logger.error(f"Migration failed: {str(e)}")
                    success = False
                    
        return success 