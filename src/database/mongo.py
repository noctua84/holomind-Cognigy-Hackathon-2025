from typing import Dict, List, Any, Optional
from pymongo import MongoClient
from datetime import datetime, UTC
import logging
import pymongo

class MongoConnector:
    """MongoDB connector for storing metrics and model checkpoints"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.db = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = pymongo.MongoClient(self.config['uri'])
            self.db = self.client.holomind
            return True
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            return False
            
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.client is not None and self.db is not None
        
    def save(self, key: str, data: Dict):
        """Save data to MongoDB"""
        if not self.is_connected():
            raise Exception("Not connected to MongoDB")
            
        collection = self.db.metrics
        collection.update_one(
            {'_id': key},
            {'$set': data},
            upsert=True
        )
        
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve data from MongoDB"""
        if not self.is_connected():
            raise Exception("Not connected to MongoDB")
            
        collection = self.db.metrics
        doc = collection.find_one({'_id': key})
        return doc if doc else None
        
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    def test_connection(self) -> bool:
        """Test if database exists and is accessible"""
        try:
            self.client.server_info()
            return self.db_name in self.client.list_database_names()
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
        
    def setup_database(self):
        """Create database and required collections"""
        try:
            db = self.client[self.db_name]
            # Create collections with validation
            db.create_collection("tasks")
            db.create_collection("models")
            db.create_collection("metrics")
            return True
        except Exception as e:
            logger.error(f"MongoDB setup failed: {e}")
            return False

class ModelArchiveDB:
    """MongoDB database for storing model architectures and states"""
    def __init__(self, uri: str, database: str = 'holomind'):
        try:
            self.client = MongoClient(uri)
            self.db = self.client[database]
            # Test connection
            self.client.server_info()
            logging.info("Successfully connected to MongoDB")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def save_model_architecture(self, 
                              architecture_id: str, 
                              architecture: Dict[str, Any]):
        """Store model architecture with evolution history"""
        collection = self.db.model_architectures
        
        document = {
            'architecture_id': architecture_id,
            'timestamp': datetime.now(UTC),
            'structure': architecture,
            'version': 1
        }
        
        # Check for existing architecture
        existing = collection.find_one({'architecture_id': architecture_id})
        if existing:
            document['version'] = existing['version'] + 1
            document['previous_version'] = existing['_id']
        
        collection.insert_one(document)
        
    def save_training_state(self, 
                          architecture_id: str, 
                          state: Dict[str, Any],
                          metadata: Dict[str, Any]):
        """Store training state with metadata"""
        collection = self.db.training_states
        
        document = {
            'architecture_id': architecture_id,
            'timestamp': datetime.now(UTC),
            'state': state,
            'metadata': metadata,
            'training_context': {
                'environment': metadata.get('environment'),
                'dependencies': metadata.get('dependencies'),
                'hardware': metadata.get('hardware_specs')
            }
        }
        
        collection.insert_one(document)
        
    def query_architecture_evolution(self, architecture_id: str) -> List[Dict]:
        """Retrieve evolution history of a model architecture"""
        collection = self.db.model_architectures
        
        pipeline = [
            {'$match': {'architecture_id': architecture_id}},
            {'$sort': {'version': -1}},
            {'$project': {
                'version': 1,
                'timestamp': 1,
                'structure': 1,
                'changes_from_previous': 1
            }}
        ]
        
        return list(collection.aggregate(pipeline)) 