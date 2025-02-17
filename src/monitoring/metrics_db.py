from sqlalchemy import (
    create_engine, Column, Integer, Float, String, 
    DateTime, JSON, ForeignKey, Table, Text, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Dict, Any, List
import json
import io
import torch
from alembic import command
from alembic.config import Config
import os

Base = declarative_base()

# Association table for task dependencies
task_dependencies = Table(
    'task_dependencies', Base.metadata,
    Column('dependent_task_id', String, ForeignKey('tasks.task_id')),
    Column('prerequisite_task_id', String, ForeignKey('tasks.task_id'))
)

class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    creation_time = Column(DateTime, default=datetime.utcnow)
    config = Column(JSON)
    
    runs = relationship("Run", back_populates="experiment")

class Run(Base):
    __tablename__ = 'runs'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    status = Column(String)  # 'running', 'completed', 'failed'
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    config = Column(JSON)
    
    experiment = relationship("Experiment", back_populates="runs")
    metrics = relationship("MetricsRecord", back_populates="run")
    models = relationship("ModelArtifact", back_populates="run")

class ModelArtifact(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    name = Column(String)
    state_dict = Column(LargeBinary)  # Serialized model state
    model_metadata = Column(JSON)  # Renamed from metadata to model_metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    run = relationship("Run", back_populates="models")

class Task(Base):
    """Store task-specific information"""
    __tablename__ = 'tasks'
    
    task_id = Column(String, primary_key=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String)  # 'running', 'completed', 'failed'
    configuration = Column(JSON)  # Task-specific config
    
    # Relationships
    metrics = relationship("MetricsRecord", back_populates="task")
    performance = relationship("TaskPerformance", back_populates="task")
    model_states = relationship("ModelState", back_populates="task")

class MetricsRecord(Base):
    """Store training metrics"""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String, ForeignKey('tasks.task_id'))
    run_id = Column(Integer, ForeignKey('runs.id'))
    epoch = Column(Integer)
    step = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)  # Training metrics
    memory_usage = Column(JSON)  # Memory metrics
    gradient_stats = Column(JSON)  # Gradient statistics
    
    task = relationship("Task", back_populates="metrics")
    run = relationship("Run", back_populates="metrics")

class TaskPerformance(Base):
    """Store task performance metrics"""
    __tablename__ = 'task_performance'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String, ForeignKey('tasks.task_id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    loss = Column(Float)
    forgetting_score = Column(Float)  # Measure of catastrophic forgetting
    cross_task_performance = Column(JSON)  # Performance on other tasks
    
    task = relationship("Task", back_populates="performance")

class ModelState(Base):
    """Store model checkpoints and states"""
    __tablename__ = 'model_states'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String, ForeignKey('tasks.task_id'))
    epoch = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    checkpoint_path = Column(String)  # Path to saved checkpoint
    fisher_information = Column(JSON)  # EWC Fisher information
    parameters_importance = Column(JSON)  # Parameter importance scores
    
    task = relationship("Task", back_populates="model_states")

class MetricsDB:
    def __init__(self, db_url: str = 'sqlite:///metrics.db'):
        self.engine = create_engine(db_url)
        
        # Run migrations before creating tables
        self._run_migrations()
        
        # Create any remaining tables
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
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
    
    def create_experiment(self, name: str, config: Dict[str, Any]) -> int:
        """Create new experiment"""
        session = self.Session()
        try:
            experiment = Experiment(name=name, config=config)
            session.add(experiment)
            session.commit()
            return experiment.id
        finally:
            session.close()
    
    def create_run(self, experiment_id: int, config: Dict[str, Any]) -> int:
        """Create new run"""
        session = self.Session()
        try:
            run = Run(
                experiment_id=experiment_id,
                status='running',
                config=config
            )
            session.add(run)
            session.commit()
            return run.id
        finally:
            session.close()
    
    def save_model(self, run_id: int, name: str, state_dict: Dict, metadata: Dict):
        """Save model state to database"""
        session = self.Session()
        try:
            # Serialize model state
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            
            model = ModelArtifact(
                run_id=run_id,
                name=name,
                state_dict=buffer.getvalue(),
                model_metadata=metadata
            )
            session.add(model)
            session.commit()
        finally:
            session.close()
            
    def complete_run(self, run_id: int):
        """Mark run as completed"""
        session = self.Session()
        try:
            run = session.query(Run).filter_by(id=run_id).first()
            if run:
                run.status = 'completed'
                run.end_time = datetime.utcnow()
                session.commit()
        finally:
            session.close()
    
    def create_task(self, task_id: str, config: Dict[str, Any]):
        """Create new task record"""
        session = self.Session()
        try:
            task = Task(
                task_id=task_id,
                configuration=config,
                status='running'
            )
            session.add(task)
            session.commit()
        finally:
            session.close()
    
    def update_task_status(self, task_id: str, status: str):
        """Update task status"""
        session = self.Session()
        try:
            task = session.query(Task).filter_by(task_id=task_id).first()
            if task:
                task.status = status
                if status in ['completed', 'failed']:
                    task.end_time = datetime.utcnow()
                session.commit()
        finally:
            session.close()
    
    def log_metrics(self, task_id: str, run_id: int, epoch: int, step: int, metrics: Dict[str, Any]):
        """Log training metrics"""
        session = self.Session()
        try:
            record = MetricsRecord(
                task_id=task_id,
                run_id=run_id,
                epoch=epoch,
                step=step,
                metrics=metrics.get('training_metrics', {}),
                memory_usage=metrics.get('memory_metrics', {}),
                gradient_stats=metrics.get('gradient_stats', {})
            )
            session.add(record)
            session.commit()
        finally:
            session.close()
    
    def log_performance(self, task_id: str, performance_metrics: Dict[str, Any]):
        """Log task performance"""
        session = self.Session()
        try:
            performance = TaskPerformance(
                task_id=task_id,
                accuracy=performance_metrics.get('accuracy'),
                loss=performance_metrics.get('loss'),
                forgetting_score=performance_metrics.get('forgetting_score'),
                cross_task_performance=performance_metrics.get('cross_task_performance', {})
            )
            session.add(performance)
            session.commit()
        finally:
            session.close()
    
    def log_model_state(self, task_id: str, epoch: int, state_info: Dict[str, Any]):
        """Log model state information"""
        session = self.Session()
        try:
            model_state = ModelState(
                task_id=task_id,
                epoch=epoch,
                checkpoint_path=state_info.get('checkpoint_path'),
                fisher_information=state_info.get('fisher_information'),
                parameters_importance=state_info.get('parameters_importance')
            )
            session.add(model_state)
            session.commit()
        finally:
            session.close()
    
    def get_task_metrics(self, task_id: str) -> List[Dict]:
        """Get all metrics for a task"""
        session = self.Session()
        try:
            records = session.query(MetricsRecord)\
                           .filter(MetricsRecord.task_id == task_id)\
                           .order_by(MetricsRecord.epoch, MetricsRecord.step).all()
            return [
                {
                    'epoch': r.epoch,
                    'step': r.step,
                    'metrics': r.metrics,
                    'memory_usage': r.memory_usage,
                    'gradient_stats': r.gradient_stats,
                    'timestamp': r.timestamp.isoformat()
                } for r in records
            ]
        finally:
            session.close() 