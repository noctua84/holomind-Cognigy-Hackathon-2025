from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Dict, Any, Optional
from pathlib import Path

class NetworkSchema(BaseModel):
    input_dim: int = Field(gt=0, description="Input dimension size")
    feature_dim: int = Field(gt=0, description="Feature dimension size")
    output_dim: int = Field(gt=0, description="Output dimension size")
    memory: Dict[str, int] = Field(
        default_factory=lambda: {"size": 1000, "feature_dim": 256},
        description="Memory configuration"
    )
    task_columns: Dict[str, Any] = Field(
        default_factory=lambda: {
            "hidden_dims": [256, 128, 64],
            "activation": "relu",
            "dropout": 0.2
        },
        description="Task column configuration"
    )

    @field_validator('memory')
    @classmethod
    def validate_memory(cls, v: Dict[str, int]) -> Dict[str, int]:
        if not v.get('size', 0) > 0:
            raise ValueError("Memory size must be positive")
        return v

    @field_validator('task_columns')
    @classmethod
    def validate_task_columns(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if 'hidden_dims' not in v or not v['hidden_dims']:
            raise ValueError("Must specify hidden dimensions")
        return v

class DataloaderSchema(BaseModel):
    batch_size: int = Field(gt=0, description="Batch size")
    num_workers: int = Field(ge=0, description="Number of worker processes")
    pin_memory: bool = Field(default=True, description="Pin memory for GPU transfer")
    shuffle: bool = Field(default=True, description="Shuffle training data")

class DataSchema(BaseModel):
    datasets: Dict[str, Any] = Field(..., description="Dataset configuration")
    preprocessing: Dict[str, Any] = Field(..., description="Preprocessing configuration")
    dataloader: DataloaderSchema = Field(..., description="Dataloader configuration")
    train_split: float = Field(gt=0, lt=1, default=0.7)
    val_split: float = Field(gt=0, lt=1, default=0.15)
    test_split: float = Field(gt=0, lt=1, default=0.15)

    @model_validator(mode='after')
    def validate_splits(self) -> 'DataSchema':
        total = self.train_split + self.val_split + self.test_split
        if total > 1.0:
            raise ValueError("Data splits must sum to less than or equal to 1.0")
        return self

class OptimizerSchema(BaseModel):
    type: str = Field(..., description="Optimizer type")
    momentum: Optional[float] = Field(ge=0, lt=1, default=None)
    weight_decay: Optional[float] = Field(ge=0, default=None)

class TrainingSchema(BaseModel):
    epochs: int = Field(gt=0, description="Number of training epochs")
    batch_size: int = Field(gt=0, description="Training batch size")
    learning_rate: float = Field(gt=0, description="Learning rate")
    ewc_lambda: float = Field(ge=0, default=0.4, description="EWC regularization strength")
    optimizer: OptimizerSchema

class MonitoringSchema(BaseModel):
    tensorboard: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "log_dir": "runs/"}
    )
    mlflow: Dict[str, Any] = Field(
        default_factory=lambda: {"tracking_uri": "mlruns", "experiment_name": "default"}
    )
    visualization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "output_dir": "visualizations/",
            "plots": {"task_performance": {"update_frequency": 1}}
        }
    )

class SystemSchema(BaseModel):
    parallel: Dict[str, Any] = Field(default_factory=dict)

class ConfigSchema(BaseModel):
    version: str = Field(default="1.0.0")
    model: Dict[str, Any] = Field(..., description="Model configuration")
    data: DataSchema
    training: TrainingSchema
    monitoring: MonitoringSchema
    system: SystemSchema = Field(default_factory=SystemSchema)
    
    model_config = ConfigDict(
        extra='allow',  # Allow extra fields for backward compatibility
        validate_assignment=True  # Validate on attribute assignment
    ) 