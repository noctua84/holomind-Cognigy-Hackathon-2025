from ..base import Migration

class AddModelValidation(Migration):
    def up(self) -> bool:
        """Add validation rules to models collection"""
        try:
            db = self.mongo.client[self.mongo.db_name]
            db.command({
                'collMod': 'models',
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['architecture_id', 'version', 'structure'],
                        'properties': {
                            'architecture_id': {'bsonType': 'string'},
                            'version': {'bsonType': 'int'},
                            'structure': {'bsonType': 'object'}
                        }
                    }
                }
            })
            return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def down(self) -> bool:
        """Remove validation rules"""
        try:
            db = self.mongo.client[self.mongo.db_name]
            db.command({'collMod': 'models', 'validator': {}})
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False 