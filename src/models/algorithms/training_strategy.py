import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import evaluate
import mlflow
import numpy as np
import torch
from torch.amp import GradScaler
from tqdm import tqdm

from models.training_monitoring import PerformanceMetrics, PerformanceMonitor
from models.utils.distributed_manager import DistributedManager
from models.utils.metrics import mlflow_log_metrics
from models.utils.vizualisation import (
    plot_layer_wise_expert_heatmap,
    plot_overall_expert_utilization,
)
from utils.batch_data import BatchData
from utils.configurations import MoERLConfig
from utils.exceptions import TrainingError
from utils.ressource_manager import TrainingResourceManager

if TYPE_CHECKING:
    from models.trainer import MixtureOfExpertsTrainer


class TrainingStrategy:

    def __init__(
        self,
        resource_manager: TrainingResourceManager,
        performance_monitor: PerformanceMonitor,
        logger: logging.Logger,
        config: MoERLConfig,
        mlflow_client: mlflow.MlflowClient,
        distributed_manager: DistributedManager,
    ):
        self.resource_manager = resource_manager
        self.performance_monitor = performance_monitor
        self.logger = logger
        self.config = config
        self.mlflow_client = mlflow_client
        self.distributed_manager = distributed_manager
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.expert_usage_stats: List[Dict[str, Any]] = []
        self.float16_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.scaler = GradScaler(
            init_scale=self.config.training_params.amp_init_scale,
            growth_factor=self.config.training_params.amp_growth_factor,
            backoff_factor=self.config.training_params.amp_backoff_factor,
            growth_interval=self.config.training_params.amp_growth_interval,
            enabled=self.config.training_params.use_amp,
        )

    def run_training_step(
        self, trainer: "MixtureOfExpertsTrainer", dataloader, epoch: int, run_id: str
    ) -> Dict[str, float]:

        epoch_metrics = []

        if hasattr(trainer.policy_model, "expert_tracker"):
            trainer.policy_model.expert_tracker.reset_stats()

        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}"
        )

        self.logger.info(f"Starting epoch {epoch} with {len(dataloader)} batches.")

        for batch_idx, batch_data in progress_bar:
            with self.resource_manager.managed_computation("batch_processing"):
                try:
                    if batch_data is None:
                        continue

                    batch_start_time = time.perf_counter()

                    rollout_start_time = time.perf_counter()
                    local_rollout_data, local_metrics = self._collect_rollout(
                        trainer, batch_data
                    )
                    rollout_time = time.perf_counter() - rollout_start_time

                    update_policy_start_time = time.perf_counter()
                    update_metrics = self._update_policy(trainer, local_rollout_data)
                    update_policy_time = time.perf_counter() - update_policy_start_time

                    if self.distributed_manager.is_main_process:
                        local_metrics = self.distributed_manager.reduce_metrics(
                            local_metrics
                        )
                        update_metrics = self.distributed_manager.reduce_metrics(
                            update_metrics
                        )

                    local_metrics.update(update_metrics)
                    epoch_metrics.append(local_metrics)

                    mlflow_log_metrics(
                        self.mlflow_client,
                        self.config,
                        local_metrics,
                        step=epoch * len(dataloader) + batch_idx,
                        run_id=run_id,
                    )

                    processing_time = time.perf_counter() - batch_start_time
                    memory_stats = self.resource_manager.check_gpu_memory()

                    perf_metric = PerformanceMetrics(
                        batch_processing_time=processing_time,
                        generation_time=local_metrics.get("generation_time", 0),
                        reward_computation_time=local_metrics.get("reward_time", 0),
                        gpu_memory_usage=memory_stats["used"],
                        rollout_time=rollout_time,
                        update_policy_time=update_policy_time,
                        cpu_usage=0.0,
                        timestamp=time.time(),
                    )
                    self.performance_monitor.record_metric(perf_metric)

                    display_metrics = {
                        "reward": f"{local_metrics.get('reward', 0):.4f}",
                        "policy_loss": f"{local_metrics.get('policy_loss', 0):.4f}",
                        "entropy_loss": f"{local_metrics.get('entropy_loss', 0):.4f}",
                        "kl_div": f"{local_metrics.get('kl_div', 0):.4f}",
                    }
                    progress_bar.set_postfix(**display_metrics)

                except Exception as e:
                    self.logger.error(
                        f"Batch processing failed: epoch={epoch}, batch_idx={batch_idx}, error={e}",
                        exc_info=True,
                    )
                    raise

            if not epoch_metrics:
                raise TrainingError(f"No valid batches processed in epoch {epoch}")

            expert_usage = trainer.policy_model.module.expert_tracker.get_usage_stats()
            self.expert_usage_stats.append(expert_usage)
            mlflow_log_metrics(
                self.mlflow_client, self.config, expert_usage, step=epoch, run_id=run_id
            )
        vizualization_path = Path(self.config.checkpoint_path / "visualizations")
        vizualization_path.mkdir(parents=True, exist_ok=True)

        plot_layer_wise_expert_heatmap(
            self.expert_usage_stats, vizualization_path, epoch
        )
        plot_overall_expert_utilization(
            self.expert_usage_stats, vizualization_path, epoch
        )

        return self._aggregate_metrics(epoch_metrics)

    def _collect_rollout(
        self, trainer: "MixtureOfExpertsTrainer", batch_data: BatchData
    ) -> Tuple[Dict, Dict]:
        raise NotImplementedError("Subclasses must implement collect rollout")

    def _update_policy(
        self, trainer: "MixtureOfExpertsTrainer", rollout_data: Dict
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement policy update")

    def _aggregate_metrics(
        self, batch_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregates a list of metric dictionaries by computing the mean and standard deviation for each metric key.

        Args:
            batch_metrics (List[Dict[str, float]]): A list of dictionaries containing metric names and their float values.

        Returns:
            Dict[str, float]: A dictionary where each key is prefixed with 'avg_' or 'std_' followed by the metric name,
                              representing the average and standard deviation of each metric across the batch.
                              Returns an empty dictionary if batch_metrics is empty.
        """
        if not batch_metrics:
            return {}
        aggregated = {}
        all_keys = set().union(*[m.keys() for m in batch_metrics])
        for k in all_keys:
            vals = [m[k] for m in batch_metrics if k in m and m[k] is not None]
            if vals:
                aggregated[f"avg_{k}"] = float(np.mean(vals))
                aggregated[f"std_{k}"] = float(np.std(vals))
        return aggregated

    def load_checkpoint(self, trainer: "MixtureOfExpertsTrainer", checkpoint_path: str):
        """
        Load model and optimizer state from checkpoint.

        Args:
            trainer: The MixtureOfExpertsTrainer instance
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.policy_model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
