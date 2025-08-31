from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


@dataclass
class PerformanceMetrics:

    batch_processing_time: float
    generation_time: float
    reward_computation_time: float
    gpu_memory_usage: float
    rollout_time: float
    update_policy_time: float
    cpu_usage: float
    timestamp: float


class TrainingPhase(Enum):
    WARMUP = "warmup"
    TRAINING = "training"
    EVALUATION = "evaluation"
    CHECKPOINTING = "checkpointing"


class PerformanceMonitor:

    def __init__(self, window_size: int = 100):
        """
        Initializes the training monitoring object.

        Args:
            window_size (int, optional): The size of the window for tracking metrics history. Defaults to 100.

        Attributes:
            window_size (int): The size of the window for metrics history.
            metrics_history (List[PerformanceMetrics]): List to store historical performance metrics.
            phase_metrics (Dict[TrainingPhase, List[float]]): Dictionary mapping each training phase to its list of metric values.
        """
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.phase_metrics: Dict[TrainingPhase, List[float]] = {
            phase: [] for phase in TrainingPhase
        }

    def record_metric(self, metric: PerformanceMetrics):
        """
        Records a new performance metric and maintains a fixed-size history window.

        Appends the provided metric to the metrics history. If the history exceeds
        the specified window size, removes the oldest metric to keep the history
        within the window.

        Args:
            metric (PerformanceMetrics): The performance metric to record.
        """
        self.metrics_history.append(metric)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Computes and returns a summary of performance metrics collected during training.

        Returns:
            Dict[str, float]: A dictionary containing the following keys:
                - "avg_batch_time": Average batch processing time.
                - "p95_batch_time": 95th percentile of batch processing times.
                - "avg_memory_usage": Average GPU memory usage.
                - "max_memory_usage": Maximum GPU memory usage observed.
                - "throughput_batches_per_sec": Estimated throughput in batches per second.

        If no metrics are available, returns an empty dictionary.
        """
        if not self.metrics_history:
            return {}

        processing_times = [m.batch_processing_time for m in self.metrics_history]
        memory_usage = [m.gpu_memory_usage for m in self.metrics_history]

        summary = {
            "avg_batch_time": np.mean(processing_times),
            "p95_batch_time": np.percentile(processing_times, 95),
            "avg_memory_usage": np.mean(memory_usage),
            "max_memory_usage": np.max(memory_usage),
            "throughput_batches_per_sec": (
                1.0 / np.mean(processing_times) if processing_times else 0
            ),
        }

        return summary
