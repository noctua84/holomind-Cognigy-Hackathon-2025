from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import logging

class PerformanceVisualizer:
    """Generates visualizations for model performance and system metrics"""
    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer with configuration"""
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'visualizations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plot settings
        self.plot_config = config.get('plots', {
            'task_performance': {'update_frequency': 1}
        })
        
        # Setup basic matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2
        })
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def plot_task_performance(self, metrics_history: Dict[str, List[Dict]],
                            save_path: Optional[str] = None):
        """Plot performance across all tasks"""
        plt.figure(figsize=(12, 6))
        
        for task_id, history in metrics_history.items():
            steps = [h['step'] for h in history]
            accuracy = [h['metrics'].get('accuracy', 0) for h in history]
            loss = [h['metrics'].get('loss', 0) for h in history]
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(steps, accuracy, label=f'Task {task_id}')
            plt.title('Accuracy Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(steps, loss, label=f'Task {task_id}')
            plt.title('Loss Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.legend()
            
        plt.tight_layout()
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close()
        
    def plot_memory_usage(self, metrics_history: Dict[str, List[Dict]],
                         save_path: Optional[str] = None):
        """Plot memory usage over time"""
        plt.figure(figsize=(10, 6))
        
        # Combine memory metrics from all tasks
        steps = []
        rss = []
        vms = []
        gpu = []
        
        for task_history in metrics_history.values():
            for entry in task_history:
                steps.append(entry['step'])
                rss.append(entry['memory']['rss'])
                vms.append(entry['memory']['vms'])
                gpu.append(entry['memory']['gpu_used'])
        
        plt.plot(steps, rss, label='RSS Memory')
        plt.plot(steps, vms, label='Virtual Memory')
        if any(gpu):  # Only plot GPU if used
            plt.plot(steps, gpu, label='GPU Memory')
            
        plt.title('Memory Usage Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close()
        
    def plot_forgetting_analysis(self, task_performances: Dict[str, List[float]],
                               save_path: Optional[str] = None):
        """Plot analysis of catastrophic forgetting"""
        plt.figure(figsize=(12, 6))
        
        # Plot performance retention
        tasks = list(task_performances.keys())
        final_performances = [perf[-1] for perf in task_performances.values()]
        
        plt.subplot(1, 2, 1)
        plt.bar(tasks, final_performances)
        plt.title('Final Performance by Task')
        plt.xlabel('Task')
        plt.ylabel('Performance')
        
        # Plot performance evolution
        plt.subplot(1, 2, 2)
        for task_id, performances in task_performances.items():
            plt.plot(range(len(performances)), performances, 
                    label=f'Task {task_id}')
            
        plt.title('Performance Evolution')
        plt.xlabel('Evaluation Point')
        plt.ylabel('Performance')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close() 