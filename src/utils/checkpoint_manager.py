import json
import logging
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from models.trainer import MixtureOfExpertsTrainer

from utils.exceptions import TrainingError

logger = logging.getLogger(__name__)


class CheckpointManager:

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        metric_name: str = "avg_reward",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.001,
    ):
        """
        Initializes the CheckpointManager.

        Args:
            checkpoint_dir (str): Directory where checkpoints will be saved.
            keep_last_n (int, optional): Number of most recent checkpoints to keep. Defaults to 3.
            metric_name (str, optional): Name of the metric to monitor for improvement. Defaults to "avg_reward".
            mode (str, optional): Mode for monitoring the metric, either "max" or "min". Defaults to "max".
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.001.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_metrics = {}

    def is_better(self, current_value: float) -> bool:
        """
        Determines whether the current value is better than the best recorded value based on the specified mode.

        In "max" mode, a value is considered better if it exceeds the best value by at least `min_delta`.
        In other modes (typically "min"), a value is considered better if it is less than the best value minus `min_delta`.

        Args:
            current_value (float): The value to compare against the best recorded value.

        Returns:
            bool: True if the current value is better according to the mode and `min_delta`, False otherwise.
        """
        if self.mode == "max":
            return current_value > (self.best_value + self.min_delta)
        else:
            return current_value < (self.best_value - self.min_delta)

    def update(
        self,
        trainer: "MixtureOfExpertsTrainer",
        current_metrics: Dict[str, float],
        epoch: int,
    ) -> bool:
        """
        Updates the checkpoint based on the provided metrics and epoch.

        If the specified metric has improved according to `is_better`, saves a new checkpoint,
        resets the epochs without improvement counter, and updates the best value and epoch.
        Otherwise, increments the epochs without improvement counter.

        Args:
            trainer (PPOMoETrainer): The trainer instance used for saving the checkpoint.
            current_metrics (Dict[str, float]): Dictionary of current metric values.
            epoch (int): The current epoch number.

        Returns:
            bool: True if a new checkpoint was saved (i.e., the metric improved), False otherwise.
        """
        if self.metric_name not in current_metrics:
            return False

        current_value = current_metrics[self.metric_name]

        if self.is_better(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self.save_checkpoint(trainer, epoch, self.best_metrics)

            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_early_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def get_best_info(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "metric_name": self.metric_name,
            "epochs_without_improvement": self.epochs_without_improvement,
            "best_metrics": self.best_metrics.copy(),
        }

    def save_checkpoint(
        self, trainer: "MixtureOfExpertsTrainer", epoch: int, metrics: Dict[str, float]
    ) -> str:
        """
        Saves the current training checkpoint, including the policy model, tokenizer, and metadata.

        Args:
            trainer (PPOMoETrainer): The trainer instance containing the policy model and tokenizer to be saved.
            epoch (int): The current training epoch.
            metrics (Dict[str, float]): A dictionary of training metrics to be saved as metadata.

        Returns:
            str: The path to the saved checkpoint directory.

        Raises:
            TrainingError: If saving the checkpoint fails for any reason.
        """
        checkpoint_dir = self.checkpoint_dir
        checkpoint_dir.mkdir(exist_ok=True)

        try:

            trainer.policy_model.save_pretrained(checkpoint_dir)
            trainer.tokenizer.save_pretrained(checkpoint_dir)

            self._save_metadata(checkpoint_dir, trainer, epoch, metrics)

            logger.info(
                f"checkpoint_saved: epoch = {epoch}, path = {str(checkpoint_dir)}",
            )

            return str(checkpoint_dir)
        except Exception as e:
            logger.error(f"checkpoint_save_failed: epoch = {epoch}, error = {str(e)}")
            logger.debug(traceback.format_exc())
            raise TrainingError(f"Failed to save checkpoint: {e}") from e

    def _save_metadata(
        self,
        checkpoint_dir: Path,
        trainer: "MixtureOfExpertsTrainer",
        epoch: int,
        metrics: Dict[str, float],
    ):
        metadata = {
            "epoch": epoch,
            "config": trainer.config.model_dump(),
            "metrics": metrics,
            "timestamp": time.time(),
            "device": str(trainer.device),
            "torch_version": torch.__version__,
        }

        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4, default=str)
