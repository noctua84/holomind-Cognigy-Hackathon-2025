from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from typing import Dict, Any
import torch
from pathlib import Path

class HyperparameterOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "optimizer": tune.choice(["adam", "sgd"]),
            "momentum": tune.uniform(0.1, 0.9)
        }
        
    def optimize(self, train_fn, num_samples: int = 10, **kwargs):
        """Run hyperparameter optimization"""
        metric = kwargs.get('metric', 'loss')
        mode = kwargs.get('mode', 'min')
        
        scheduler = tune.schedulers.ASHAScheduler(
            metric=metric,
            mode=mode
        )
        
        # Create absolute path for Ray results
        storage_path = Path.cwd() / "ray_results"
        storage_path.mkdir(exist_ok=True)
        
        tuner = tune.Tuner(
            train_fn,
            param_space=self.search_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=num_samples
            ),
            run_config=tune.RunConfig(
                name="holomind_tune",
                storage_path=str(storage_path.absolute())
            )
        )
        
        results = tuner.fit()
        return results.get_best_result().config 