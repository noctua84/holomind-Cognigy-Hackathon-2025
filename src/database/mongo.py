from typing import Dict, List, Any
from pymongo import MongoClient
from datetime import datetime
import logging

class ModelArchiveDB:
    """MongoDB database for storing model architectures and states"""
    def __init__(self, connection_uri: str):
        try:
            self.client = MongoClient(connection_uri)
            self.db = self.client.holomind
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
            'timestamp': datetime.utcnow(),
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
            'timestamp': datetime.utcnow(),
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