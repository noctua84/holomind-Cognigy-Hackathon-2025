import torch
import torch.nn as nn
from typing import Dict, List
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn not installed. Some visualizations will use matplotlib instead.")

class EWC:
    """Elastic Weight Consolidation with adaptive lambda and importance visualization"""
    
    def __init__(self, model: nn.Module, initial_lambda: float = 0.4, 
                 adaptive_lambda: bool = True, lambda_decay: float = 0.8):
        """
        Args:
            model: The neural network model
            initial_lambda: Initial importance of old tasks
            adaptive_lambda: Whether to adapt lambda based on task difficulty
            lambda_decay: Decay factor for lambda between tasks
        """
        self.model = model
        self.initial_lambda = initial_lambda
        self.adaptive_lambda = adaptive_lambda
        self.lambda_decay = lambda_decay
        
        # Store Fisher information and optimal parameters for each task
        self.fisher_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        self.optpar_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_lambdas: Dict[str, float] = {}
        
        # Track task difficulties
        self.task_losses: Dict[str, List[float]] = {}
        
    def compute_fisher(self, task_id: str, data_loader: torch.utils.data.DataLoader):
        """Compute Fisher Information Matrix for a task"""
        if not task_id:
            raise ValueError("task_id cannot be empty")
        if not data_loader:
            raise ValueError("data_loader cannot be empty")
        if len(data_loader.dataset) == 0:
            raise ValueError("data_loader contains no samples")
        
        # Check if model has parameters
        if not list(self.model.parameters()):
            raise ValueError("Model has no parameters")
        
        fisher_dict = {}
        optpar_dict = {}
        initial_losses = []
        
        try:
            # Initialize Fisher matrices for current parameters
            for n, p in self.model.named_parameters():
                if p.requires_grad:  # Only track parameters that require gradients
                    fisher_dict[n] = torch.zeros_like(p, device=p.device)
                    optpar_dict[n] = p.data.clone()
            
            if not fisher_dict:
                raise ValueError("No trainable parameters found in model")
            
            # Set model to training mode
            self.model.train()
            
            # Compute Fisher Information Matrix
            for input, target in data_loader:
                input = input.to(next(self.model.parameters()).device)
                target = target.to(next(self.model.parameters()).device)
                
                self.model.zero_grad()
                output = self.model(input, task_id)
                loss = nn.functional.cross_entropy(output, target)
                initial_losses.append(loss.item())
                loss.backward()
                
                # Accumulate squared gradients
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        fisher_dict[n] += p.grad.data ** 2
            
            # Normalize by number of samples
            for n in fisher_dict.keys():
                fisher_dict[n] /= len(data_loader)
            
            self.fisher_dict[task_id] = fisher_dict
            self.optpar_dict[task_id] = optpar_dict
            
            # Compute adaptive lambda based on task difficulty
            if self.adaptive_lambda:
                task_difficulty = np.mean(initial_losses)
                self.task_losses[task_id] = initial_losses
                
                # Adjust lambda based on task difficulty and number of previous tasks
                num_prev_tasks = len(self.fisher_dict) - 1
                task_lambda = self.initial_lambda * (self.lambda_decay ** num_prev_tasks)
                task_lambda *= (1 + task_difficulty)
                
                self.task_lambdas[task_id] = task_lambda
                logger.info(f"Task {task_id} lambda set to {task_lambda:.4f} "
                           f"(difficulty: {task_difficulty:.4f})")
            else:
                self.task_lambdas[task_id] = self.initial_lambda
        
        except Exception as e:
            logger.error(f"Error during Fisher Information Matrix computation: {e}")
            raise
        
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC loss for all previous tasks with adaptive lambda"""
        if not self.fisher_dict:
            logger.warning("No Fisher information available. EWC loss will be zero.")
            return torch.tensor(0., device=next(self.model.parameters()).device)
        
        if not self.task_lambdas:
            logger.warning("No task lambdas available. EWC loss will be zero.")
            return torch.tensor(0., device=next(self.model.parameters()).device)
        
        loss = torch.tensor(0., device=next(self.model.parameters()).device)
        
        # Get current model parameters
        current_params = dict(self.model.named_parameters())
        
        for task_id in self.fisher_dict.keys():
            task_lambda = self.task_lambdas[task_id]
            
            # Only compute EWC loss for parameters that existed during this task
            for n, p in current_params.items():
                # Skip if parameter didn't exist for this task
                if n not in self.fisher_dict[task_id] or n not in self.optpar_dict[task_id]:
                    continue
                
                # Get Fisher information and optimal parameters for this task
                fisher = self.fisher_dict[task_id][n]
                optpar = self.optpar_dict[task_id][n]
                
                # Add EWC loss term with task-specific lambda
                loss += (fisher * (p - optpar) ** 2).sum() * task_lambda
        
        return loss
        
    def visualize_importance(self, save_dir: str = 'visualizations/ewc'):
        """Visualize parameter importance across tasks"""
        # Basic validation
        if not self.fisher_dict:
            logger.warning("No Fisher information available for visualization. "
                          "Make sure compute_fisher() has been called first.")
            return
        
        if not self.task_lambdas:
            logger.warning("No task lambdas available for visualization.")
            return
        
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate we have data to visualize
            param_names = list(next(iter(self.fisher_dict.values())).keys())
            task_ids = list(self.fisher_dict.keys())
            
            if not param_names:
                logger.error("No parameters found in Fisher information")
                return
            
            if not task_ids:
                logger.error("No tasks found in Fisher information")
                return
            
            # Create importance heatmap
            plt.figure(figsize=(12, 8))
            importance_matrix = np.zeros((len(task_ids), len(param_names)))
            
            for i, task_id in enumerate(task_ids):
                if task_id not in self.fisher_dict:
                    logger.warning(f"Missing Fisher information for task {task_id}")
                    continue
                
                for j, param_name in enumerate(param_names):
                    if param_name not in self.fisher_dict[task_id]:
                        logger.warning(f"Missing parameter {param_name} for task {task_id}")
                        continue
                    importance_matrix[i, j] = self.fisher_dict[task_id][param_name].mean().item()
            
            # Check if we have valid data
            if np.all(importance_matrix == 0):
                logger.error("All importance values are zero")
                plt.close()
                return
            
            # Normalize importance values
            importance_matrix = (importance_matrix - importance_matrix.min()) / \
                              (importance_matrix.max() - importance_matrix.min() + 1e-8)
            
            # Plot heatmap using seaborn if available, otherwise use matplotlib
            if HAS_SEABORN:
                sns.heatmap(importance_matrix, xticklabels=param_names, yticklabels=task_ids,
                           cmap='viridis', annot=True, fmt='.2f')
            else:
                plt.imshow(importance_matrix, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
                plt.yticks(range(len(task_ids)), task_ids)
            
            plt.title('Parameter Importance Across Tasks')
            plt.xlabel('Model Parameters')
            plt.ylabel('Tasks')
            plt.tight_layout()
            plt.savefig(save_dir / 'parameter_importance.png')
            plt.close()
            
            # Plot task difficulties if using adaptive lambda
            if self.adaptive_lambda:
                plt.figure(figsize=(10, 6))
                for task_id, losses in self.task_losses.items():
                    plt.plot(losses, label=f'Task {task_id}')
                plt.title('Task Learning Difficulties')
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_dir / 'task_difficulties.png')
                plt.close()
                
                # Plot lambda evolution
                plt.figure(figsize=(8, 6))
                plt.bar(task_ids, [self.task_lambdas[tid] for tid in task_ids])
                plt.title('EWC Lambda Values per Task')
                plt.xlabel('Task')
                plt.ylabel('Lambda Value')
                plt.tight_layout()
                plt.savefig(save_dir / 'lambda_values.png')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error during importance visualization: {e}")
            plt.close()  # Cleanup any open figures
            raise
        
    def analyze_parameter_importance(self, save_dir: str = 'visualizations/ewc'):
        """Detailed analysis of parameter importance across tasks"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        param_names = list(next(iter(self.fisher_dict.values())).keys())
        task_ids = list(self.fisher_dict.keys())
        
        # 1. Layer-wise Importance Analysis
        layer_importance = {}
        for task_id in task_ids:
            layer_importance[task_id] = {}
            for name in param_names:
                layer = name.split('.')[0]  # Get layer name
                if layer not in layer_importance[task_id]:
                    layer_importance[task_id][layer] = 0
                layer_importance[task_id][layer] += self.fisher_dict[task_id][name].mean().item()
        
        # Plot layer importance
        plt.figure(figsize=(10, 6))
        layers = list(layer_importance[task_ids[0]].keys())
        x = np.arange(len(layers))
        width = 0.8 / len(task_ids)
        
        for i, task_id in enumerate(task_ids):
            values = [layer_importance[task_id][layer] for layer in layers]
            plt.bar(x + i * width, values, width, label=f'Task {task_id}')
        
        plt.xlabel('Layers')
        plt.ylabel('Importance')
        plt.title('Layer-wise Parameter Importance')
        plt.xticks(x + width * len(task_ids) / 2, layers, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / 'layer_importance.png')
        plt.close()
        
        # 2. Parameter Stability Analysis
        param_stability = {}
        for name in param_names:
            changes = []
            for i in range(len(task_ids) - 1):
                curr_params = self.optpar_dict[task_ids[i]][name]
                next_params = self.optpar_dict[task_ids[i + 1]][name]
                change = torch.norm(next_params - curr_params) / torch.norm(curr_params)
                changes.append(change.item())
            param_stability[name] = np.mean(changes)
        
        # Plot parameter stability
        plt.figure(figsize=(12, 6))
        stability_values = list(param_stability.values())
        plt.bar(range(len(param_stability)), stability_values)
        plt.xlabel('Parameters')
        plt.ylabel('Average Parameter Change')
        plt.title('Parameter Stability Across Tasks')
        plt.xticks(range(len(param_stability)), list(param_stability.keys()), rotation=90)
        plt.tight_layout()
        plt.savefig(save_dir / 'parameter_stability.png')
        plt.close()
        
        # 3. Task Interference Analysis
        interference_matrix = np.zeros((len(task_ids), len(task_ids)))
        for i, task1 in enumerate(task_ids):
            for j, task2 in enumerate(task_ids):
                if i != j:
                    interference = 0
                    for name in param_names:
                        f1 = self.fisher_dict[task1][name]
                        f2 = self.fisher_dict[task2][name]
                        # Compute normalized dot product of Fisher matrices
                        interference += torch.sum(f1 * f2) / (torch.norm(f1) * torch.norm(f2))
                    interference_matrix[i, j] = interference.item()
        
        # Plot task interference
        plt.figure(figsize=(8, 6))
        sns.heatmap(interference_matrix, xticklabels=task_ids, yticklabels=task_ids,
                    cmap='coolwarm', center=0, annot=True)
        plt.title('Task Interference Matrix')
        plt.xlabel('Task')
        plt.ylabel('Task')
        plt.tight_layout()
        plt.savefig(save_dir / 'task_interference.png')
        plt.close()
        
        # 4. Generate Analysis Report
        with open(save_dir / 'importance_analysis.txt', 'w') as f:
            f.write("Parameter Importance Analysis Report\n")
            f.write("===================================\n\n")
            
            # Most important parameters per task
            f.write("Most Important Parameters per Task:\n")
            for task_id in task_ids:
                f.write(f"\nTask {task_id}:\n")
                importances = {name: self.fisher_dict[task_id][name].mean().item() 
                             for name in param_names}
                top_5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                for name, imp in top_5:
                    f.write(f"  {name}: {imp:.4f}\n")
            
            # Most stable parameters
            f.write("\nMost Stable Parameters:\n")
            stable_params = sorted(param_stability.items(), key=lambda x: x[1])[:5]
            for name, stability in stable_params:
                f.write(f"  {name}: {stability:.4f}\n")
            
            # Task interference summary
            f.write("\nTask Interference Summary:\n")
            for i, task1 in enumerate(task_ids):
                for j, task2 in enumerate(task_ids):
                    if i != j:
                        f.write(f"  Task {task1} → Task {task2}: {interference_matrix[i,j]:.4f}\n")
        
        return {
            'layer_importance': layer_importance,
            'param_stability': param_stability,
            'task_interference': interference_matrix
        } 