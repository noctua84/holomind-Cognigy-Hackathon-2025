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
        
        if not metrics_history:
            logger.warning("No metrics history available for plotting")
            plt.close()
            return
        
        for task_id, history in metrics_history.items():
            if not history:  # Skip empty histories
                continue
            
            steps = []
            accuracy = []
            loss = []
            
            for h in history:
                if 'step' not in h or 'metrics' not in h:
                    continue
                
                steps.append(h['step'])
                metrics = h['metrics']
                # Use get() with default 0 and handle None values
                acc = metrics.get('accuracy', 0)
                los = metrics.get('loss', 0)
                if acc is not None and los is not None:
                    accuracy.append(acc)
                    loss.append(los)
            
            if not steps:  # Skip if no valid data points
                logger.warning(f"No valid metrics found for task {task_id}")
                continue
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(steps, accuracy, label=f'Task {task_id}', marker='.')
            plt.title('Accuracy Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(steps, loss, label=f'Task {task_id}', marker='.')
            plt.title('Loss Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            # Ensure the directory exists
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.output_dir / save_path)
        plt.close()
        
    def plot_memory_usage(self, metrics_history: Dict[str, List[Dict]], save_path: str = None):
        """Plot memory usage over time"""
        plt.figure(figsize=(10, 6))
        
        steps = []
        rss = []
        vms = []
        gpu = []
        
        # Collect memory metrics from all tasks
        for task_metrics in metrics_history.values():
            for entry in task_metrics:
                if 'metrics' in entry and any(k.startswith('memory/') for k in entry['metrics']):
                    steps.append(entry['step'])
                    metrics = entry['metrics']
                    rss.append(metrics.get('memory/rss', 0))
                    vms.append(metrics.get('memory/vms', 0))
                    if 'memory/gpu' in metrics:
                        gpu.append(metrics['memory/gpu'])
        
        if steps:  # Only plot if we have data
            plt.plot(steps, rss, label='RSS Memory (MB)', marker='.')
            plt.plot(steps, vms, label='Virtual Memory (MB)', marker='.')
            if gpu:  # Plot GPU memory if available
                plt.plot(steps, gpu, label='GPU Memory (MB)', marker='.')
            
            plt.title('Memory Usage Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Memory (MB)')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                # Ensure the directory exists
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(self.output_dir / save_path)
            plt.close()
        else:
            logger.warning("No memory metrics found in history")
        
    def plot_forgetting_analysis(self, task_performances: Dict[str, List[float]],
                               save_path: Optional[str] = None):
        """Plot analysis of catastrophic forgetting"""
        plt.figure(figsize=(12, 6))
        
        if not task_performances:
            logger.warning("No task performance data available for plotting")
            plt.close()
            return
        
        # Filter out empty or invalid performances
        valid_performances = {
            task_id: perfs for task_id, perfs in task_performances.items()
            if perfs and all(p is not None for p in perfs)
        }
        
        if not valid_performances:
            logger.warning("No valid performance data found")
            plt.close()
            return
        
        tasks = list(valid_performances.keys())
        final_performances = [perfs[-1] for perfs in valid_performances.values()]
        
        # Plot performance retention
        plt.subplot(1, 2, 1)
        plt.bar(tasks, final_performances)
        plt.title('Final Performance by Task')
        plt.xlabel('Task')
        plt.ylabel('Performance')
        plt.grid(True)
        
        # Plot performance evolution
        plt.subplot(1, 2, 2)
        for task_id, performances in valid_performances.items():
            plt.plot(range(len(performances)), performances, 
                    label=f'Task {task_id}', marker='.')
        
        plt.title('Performance Evolution')
        plt.xlabel('Evaluation Point')
        plt.ylabel('Performance')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close() 