from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey, Table, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# Move all model classes here from metrics_db.py
class Experiment(Base):
    __tablename__ = 'experiments'
    # ... rest of the Experiment class ...

class Run(Base):
    __tablename__ = 'runs'
    # ... rest of the Run class ...

# ... rest of the model classes ... 