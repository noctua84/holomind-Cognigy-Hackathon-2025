import pytest
from src.database.manager import DatabaseManager
from src.database.mongo import MongoConnector
from src.database.postgres import PostgresConnector

def test_database_manager(mock_database, postgres_db):
    """Test database manager functionality"""
    manager = DatabaseManager({
        'mongo': {'uri': 'mongodb://localhost:27017'},
        'mongodb': {'uri': 'mongodb://localhost:27017'},
        'postgres': postgres_db
    })
    
    # Test data storage and retrieval
    data = {'test_key': 'test_value'}
    manager.save_metrics('task1', data)
    retrieved = manager.get_metrics('task1')
    assert retrieved == data
    
    # Test model checkpoint
    state_dict = {'layer1.weight': [1.0, 2.0]}
    manager.save_checkpoint('model_v1', state_dict)
    loaded = manager.load_checkpoint('model_v1')
    assert loaded == state_dict

def test_mongo_connector():
    """Test MongoDB connector"""
    connector = MongoConnector({'uri': 'mongodb://localhost:27017'})
    
    # Test connection handling
    assert not connector.is_connected()
    
    # Test error handling
    with pytest.raises(Exception):
        connector.save('test', {})

def test_postgres_connector():
    """Test PostgreSQL connector"""
    connector = PostgresConnector({'uri': 'postgresql://localhost:5432/test'})
    
    # Test connection handling
    assert not connector.is_connected()
    
    # Test error handling
    with pytest.raises(Exception):
        connector.save('test', {}) 