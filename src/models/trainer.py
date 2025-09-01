import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import bitsandbytes as bnb
import mlflow
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from models.algorithms.a2c_training_strategy import A2CTrainingStrategy
from models.algorithms.ppo_training_strategy import PPOTrainingStrategy
from models.environnement import GSM8KEnvironment
from models.expert_usage_tracker import (
    MoEModelWithTracking,
    PatchedAutoModelForCausalLMWithValueHead,
)
from models.training_monitoring import PerformanceMonitor
from models.utils.distributed_manager import DistributedManager
from utils.checkpoint_manager import CheckpointManager
from utils.configurations import MoERLConfig
from utils.exceptions import ResourceError, TrainingError
from utils.ressource_manager import TrainingResourceManager


class MixtureOfExpertsTrainer:

    def __init__(
        self,
        config: MoERLConfig,
        tokenizer: AutoTokenizer,
        policy_model: MoEModelWithTracking,
        reference_model: PatchedAutoModelForCausalLMWithValueHead,
        environment: GSM8KEnvironment,
        logger: logging.Logger,
        mlflow_client: mlflow.MlflowClient,
        distributed_manager: DistributedManager,
        strategy: Optional[PPOTrainingStrategy | A2CTrainingStrategy] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
    ):
        self.config = config
        self.training_config = self.config.training_params

        self.logger = logger
        self.mlflow_client = mlflow_client
        self.distributed_manager = distributed_manager
        self.device = self.distributed_manager.device

        self.policy_model = self.distributed_manager.wrap_model(policy_model)
        self.reference_model = reference_model.to(self.device)
        self.tokenizer: AutoTokenizer = tokenizer
        self.environment = environment

        if self.config.training_params.use_8bit_optimizer:
            self.optimizer = bnb.optim.AdamW8bit(
                self.policy_model.parameters(), lr=self.training_config.learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.policy_model.parameters(), lr=self.training_config.learning_rate
            )

        self.resource_manager = TrainingResourceManager(self.logger)
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            config.checkpoint_path
        )
        if strategy is not None:
            self.strategy = strategy
        elif self.config.algorithm == "ppo":
            self.strategy = PPOTrainingStrategy(
                self.resource_manager,
                self.performance_monitor,
                self.logger,
                self.config,
                self.mlflow_client,
                distributed_manager=distributed_manager,
            )
        elif self.config.algorithm == "a2c":
            self.strategy = A2CTrainingStrategy(
                self.resource_manager,
                self.performance_monitor,
                self.logger,
                self.config,
                self.mlflow_client,
                distributed_manager=distributed_manager,
            )

        self.generation_kwargs = self._get_generation_kwargs()

        self.logger.info(
            f"trainer_initialized: device {str(self.device)}, config={self.training_config.model_dump()}",
        )

    def _get_generation_kwargs(self) -> Dict[str, Any]:
        """
        Constructs a dictionary of generation parameters for text generation.

        Returns:
            Dict[str, Any]: A dictionary containing generation keyword arguments:
                - min_length (int): Minimum length of generated sequences (default: 10).
                - top_k (int): Number of highest probability vocabulary tokens to keep for top-k-filtering (default: 50).
                - top_p (float): Cumulative probability for nucleus sampling (default: 0.9).
                - temperature (float): Sampling temperature (default: 0.7).
                - do_sample (bool): Whether to use sampling; if False, greedy decoding is used (default: True).
                - pad_token_id (int): Token ID used for padding.
                - max_new_tokens (int): Maximum number of new tokens to generate (default: 256).
        """
        return {
            "min_length": getattr(self.config.training_params, "min_length", 10),
            "top_k": getattr(self.config.training_params, "top_k", 50),
            "top_p": getattr(self.config.training_params, "top_p", 0.9),
            "temperature": getattr(self.config.training_params, "temperature", 0.7),
            "do_sample": getattr(self.config.training_params, "do_sample", True),
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": getattr(
                self.config.training_params, "max_new_tokens", 256
            ),
        }

    def _generate_responses(
        self, query_tensors: torch.Tensor, max_retries: int = 3
    ) -> torch.Tensor:
        """
        Generates response tensors from the given query tensors using the policy model,
        with retry logic for handling GPU out-of-memory and other exceptions.

        Args:
            query_tensors (torch.Tensor): Input tensor(s) representing queries for response generation.
            max_retries (int, optional): Maximum number of retry attempts in case of failure. Defaults to 3.

        Returns:
            torch.Tensor: Generated response tensors.

        Raises:
            ResourceError: If GPU out-of-memory error persists after all retry attempts.
            TrainingError: If any other exception persists after all retry attempts.
        """
        unwrapped_model = (
            self.policy_model.module
            if self.distributed_manager.is_distributed
            else self.policy_model
        )
        for attempt in range(max_retries):
            try:
                with self.resource_manager.managed_computation("response_generation"):
                    with torch.inference_mode():  # Use inference_mode for better performance
                        return unwrapped_model.generate(
                            query_tensors,
                            attention_mask=torch.ones_like(query_tensors),
                            **self.generation_kwargs,
                        )

            except torch.cuda.OutOfMemoryError as e:
                self.logger.warning(
                    f"oom_during_generation: attempt {attempt}, max_retries {max_retries}"
                )
                if attempt == max_retries - 1:
                    raise ResourceError(f"OOM after {max_retries} attempts: {e}") from e

                self.resource_manager.cleanup_gpu_memory(force=True)
                time.sleep(2**attempt)

            except Exception as e:
                self.logger.error(
                    f"generation_failed: attempt={attempt}, error={str(e)}"
                )
                if attempt == max_retries - 1:
                    raise TrainingError(
                        f"Generation failed after {max_retries} attempts: {e}"
                    ) from e
                time.sleep(1)

    def _compute_rewards_batch(
        self,
        queries: List[str],
        response_tensors: torch.Tensor,
        ground_truths: List[str],
    ) -> List[torch.Tensor]:
        """
        Computes rewards for a batch of queries, responses, and ground truths.

        Args:
            queries (List[str]): List of input queries.
            response_tensors (torch.Tensor): Tensor containing model-generated responses.
            ground_truths (List[str]): List of ground truth answers.

        Returns:
            torch.Tensor: Tensor of computed rewards for each query-response-ground truth triplet.

        Notes:
            - Decodes response tensors to text using the tokenizer.
            - Computes reward for each triplet using the environment's `compute_reward` method.
            - If reward computation fails for a triplet, logs a warning and assigns a reward of 0.0.
            - Returns rewards as a float32 tensor on the appropriate device.
        """
        responses_text = self.tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        rewards = []
        for query, response, ground_truth in zip(
            queries, responses_text, ground_truths
        ):
            try:
                reward = self.environment.compute_reward(query, response, ground_truth)
                rewards.append(reward)
            except Exception as e:
                self.logger.warning(
                    f"Reward computation failed for query '{query[:50]}...': {e}"
                )
                rewards.append(0.0)
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def train(self, run_id: str) -> Dict[str, Any]:
        """
        Trains the PPO-MoE model for a specified number of epochs, tracking performance and managing checkpoints.

        Args:
            run_id (str): Unique identifier for the training run.

        Returns:
            Dict[str, Any]: A summary dictionary containing training statistics, including:
                - epochs_completed (int): Number of epochs completed.
                - total_batches (int): Total number of batches processed.
                - best_reward (float): Highest average reward achieved during training.
                - training_time (float): Total training duration in seconds.
                - early_stopped (bool): Whether early stopping was triggered.
                - best_model_info (dict): Information about the best model checkpoint.
                - performance (dict): Performance metrics summary.

        Raises:
            Exception: Propagates any exception encountered during training after logging the error.
        """

        self.logger.info(
            f"training_started: num_epochs={self.training_config.num_epochs}, "
            f"tracking_metric={self.checkpoint_manager.metric_name}"
        )

        training_summary = {
            "epochs_completed": 0,
            "total_batches": 0,
            "best_reward": float("-inf"),
            "training_time": 0,
            "early_stopped": False,
            "best_model_info": {},
        }

        train_dataset = self.environment.train_dataset

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.distributed_manager.world_size,
            rank=self.distributed_manager.rank,
            shuffle=True,
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=self.environment.data_collator,
            pin_memory=True,
        )
        training_summary["total_batches"] += len(data_loader)

        training_start = time.perf_counter()
        try:
            for epoch in tqdm(range(self.training_config.num_epochs)):
                epoch_start = time.perf_counter()
                sampler.set_epoch(epoch)

                if self.distributed_manager.is_main_process:
                    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
                else:
                    progress_bar = data_loader

                epoch_metrics = self.strategy.run_training_step(
                    self, progress_bar, epoch, run_id
                )

                if self.distributed_manager.is_distributed:
                    torch.distributed.barrier()

                if self.distributed_manager.is_main_process:

                    self.on_epoch_end(
                        epoch,
                        epoch_duration=time.perf_counter() - epoch_start,
                        epoch_metrics=epoch_metrics,
                        training_summary=training_summary,
                        epoch_start=epoch_start,
                    )

        except Exception as e:
            self.logger.error(
                f"training_failed: error={str(e)}, error_type={type(e).__name__}"
            )
            self.logger.debug(traceback.format_exc())
            raise
        finally:
            self.on_train_end(training_summary, training_start)

        return training_summary

    def on_epoch_end(
        self, epoch, epoch_duration, epoch_metrics, training_summary, epoch_start
    ):
        model_to_save = (
            self.policy_model.module
            if self.distributed_manager.is_distributed
            else self.policy_model
        )

        tokenizer_to_save = self.tokenizer

        self.logger.debug("Epoch ended")

        training_summary["epochs_completed"] = epoch + 1

        current_reward = epoch_metrics.get("avg_reward", 0)
        if current_reward > training_summary["best_reward"]:
            training_summary["best_reward"] = current_reward

        self.logger.debug("Epoch ended on main process.")

        model_updated = self.checkpoint_manager.update(
            model_to_save,
            tokenizer_to_save,
            self.config,
            self.device,
            current_metrics=epoch_metrics,
            epoch=epoch,
        )

        if model_updated:
            self.logger.info(
                f"New best model found at epoch {epoch}! "
                f"{self.checkpoint_manager.metric_name}={self.checkpoint_manager.best_value:.4f}"
            )

        epoch_duration = time.perf_counter() - epoch_start
        self.logger.info(f"epoch_completed: epoch={epoch}, duration={epoch_duration}")
        self._display_epoch_end(epoch, epoch_duration, epoch_metrics)

        if self.checkpoint_manager.should_early_stop():
            self.logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"No improvement for {self.checkpoint_manager.patience} epochs."
            )
            training_summary["early_stopped"] = True
            training_summary["early_stopping_epoch"] = epoch

        return training_summary

    def on_train_end(self, training_summary, training_start):

        training_summary["best_model_info"] = self.checkpoint_manager.get_best_info()
        training_summary["training_time"] = time.perf_counter() - training_start

        perf_summary = self.performance_monitor.get_performance_summary()
        training_summary["performance"] = perf_summary

        self.logger.info(f"training_completed: summary={training_summary}")

        best_info = training_summary["best_model_info"]
        self.logger.info(
            f"Best model: Epoch {best_info['best_epoch']} with "
            f"{best_info['metric_name']}={best_info['best_value']:.4f}"
        )

        self.distributed_manager.cleanup()
        training_duration = time.perf_counter() - training_start
        self.logger.info(f"Training finished in {training_duration:.2f}s.")
        perf_summary = self.performance_monitor.get_performance_summary()
        self.logger.info(f"Performance Summary: {perf_summary}")

    def _display_epoch_end(self, epoch, epoch_duration, epoch_metrics):
        epoch_metrics_str = ""
        for key, value in epoch_metrics.items():
            epoch_metrics_str += f"  {key}: {float(value)}\n"
        self.logger.info(
            f"Epoch {epoch} ended with duration: {epoch_duration}, \n metrics:\n {epoch_metrics_str}"
        )

    def load_checkpoint(self, checkpoint_path: str):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.strategy.load_checkpoint(self, checkpoint_path)

    def evaluate(self):
        self.logger.info("Starting evaluation")
        eval_results = self.strategy.evaluate(self, self.environment.test_dataloader)
        self.logger.info(f"Evaluation results: {eval_results}")
        return eval_results

    def evaluate_sample(self, question: str) -> str:
        try:
            prompt = self.environment.format_prompt(question)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                with self.resource_manager.managed_computation("inference"):
                    outputs = self.policy_model.generate(
                        inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt) :].strip()

        except Exception as e:
            self.logger.error(
                f"evaluation_failed: question={question[:50]}, error={str(e)}"
            )
            return f"Error during evaluation: {str(e)}"
