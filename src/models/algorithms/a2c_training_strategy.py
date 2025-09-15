import logging
import time
from typing import TYPE_CHECKING, Dict, Tuple

import mlflow
import torch
from torch.amp import autocast
from tqdm import tqdm

from models.algorithms.training_strategy import TrainingStrategy
from models.utils.distributed_manager import DistributedManager

if TYPE_CHECKING:
    from models.trainer import MixtureOfExpertsTrainer

from models.training_monitoring import PerformanceMonitor
from models.utils.loss import (
    compute_advantages,
    compute_entropy_loss,
    compute_value_loss,
    get_log_probs,
)
from models.utils.metrics import compute_nlp_metrics, mlflow_log_metrics
from utils.batch_data import BatchData
from utils.configurations import MoERLConfig
from utils.exceptions import TrainingError
from utils.ressource_manager import TrainingResourceManager


class A2CTrainingStrategy(TrainingStrategy):

    def __init__(
        self,
        resource_manager: TrainingResourceManager,
        performance_monitor: PerformanceMonitor,
        logger: logging.Logger,
        config: MoERLConfig,
        mlflow_client: mlflow.MlflowClient,
        distributed_manager: DistributedManager,
    ):
        super().__init__(
            resource_manager,
            performance_monitor,
            logger,
            config,
            mlflow_client,
            distributed_manager=distributed_manager,
        )

    def _collect_rollout(
        self, trainer: "MixtureOfExpertsTrainer", batch_data: BatchData
    ) -> Tuple[Dict, Dict]:
        if batch_data is None:
            raise TrainingError("Failed to prepare batch.")

        device = trainer.device
        query_tensors = batch_data.input_ids.to(device)

        generation_start = time.perf_counter()
        with torch.inference_mode():
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=trainer.config.training_params.use_mixed_precision,
                dtype=self.float16_type,
            ):
                response_tensors = trainer._generate_responses(query_tensors)
        generation_time = time.perf_counter() - generation_start

        reward_start = time.perf_counter()
        rewards = trainer._compute_rewards_batch(
            batch_data.queries, response_tensors, batch_data.ground_truths
        )
        reward_time = time.perf_counter() - reward_start

        full_tensors = torch.cat([query_tensors, response_tensors], dim=1)
        attention_mask = (full_tensors != trainer.tokenizer.pad_token_id).long()

        with torch.inference_mode():
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=trainer.config.training_params.use_mixed_precision,
                dtype=self.float16_type,
            ):
                logits, _, values = trainer.policy_model(
                    full_tensors, attention_mask=attention_mask, return_dict=True
                )
                log_probs = get_log_probs(logits, full_tensors)

        B, T = full_tensors.shape
        qlen = query_tensors.shape[1]
        resp_mask = torch.zeros_like(full_tensors, dtype=torch.bool)
        resp_mask[:, qlen:] = True
        resp_mask &= attention_mask.bool()

        eos_id = trainer.tokenizer.eos_token_id
        if eos_id is not None:
            resp_part = full_tensors[:, qlen:]
            eos_pos = (resp_part == eos_id).float()
            first_eos = eos_pos.argmax(dim=1)
            has_eos = eos_pos.max(dim=1).values.bool()
            idx = (
                torch.arange(resp_part.size(1), device=device)
                .unsqueeze(0)
                .expand(B, -1)
            )
            before_eos = idx < first_eos.unsqueeze(1)
            before_eos = torch.where(
                has_eos.unsqueeze(1),
                before_eos,
                torch.ones_like(before_eos, dtype=torch.bool),
            )
            trimmed_mask = torch.zeros_like(resp_mask)
            trimmed_mask[:, qlen:] = before_eos
            resp_mask &= trimmed_mask

        metrics = {
            "reward": rewards.mean().item(),
            "generation_time": generation_time,
            "reward_time": reward_time,
        }

        metrics.update(
            compute_nlp_metrics(
                self.bleu_metric,
                self.rouge_metric,
                query_tensors=query_tensors,
                logits=logits,
                response_tensors=response_tensors,
                batch_data=batch_data,
                tokenizer=trainer.tokenizer,
                prefix="train_rollout_",
            )
        )

        rollout_data = {
            "queries": query_tensors,
            "responses": response_tensors,
            "full_tensors": full_tensors,
            "log_probs": log_probs.detach(),
            "values": values.detach(),
            "rewards": rewards.detach(),
            "resp_mask": resp_mask,
            "attn_mask": attention_mask,
        }

        return rollout_data, metrics

    def _update_policy(
        self, trainer: "MixtureOfExpertsTrainer", rollout_data: Dict
    ) -> Dict[str, float]:
        trainer.optimizer.zero_grad()

        advantages, returns = compute_advantages(
            rollout_data["rewards"],
            rollout_data["values"],
            torch.zeros_like(rollout_data["rewards"], dtype=torch.bool),
            trainer.config.training_params,
        )

        resp_mask_f = rollout_data["resp_mask"].float()
        count = resp_mask_f.sum().clamp_min(1.0)
        adv_mean = (advantages * resp_mask_f).sum() / count
        adv_var = ((advantages - adv_mean) * resp_mask_f).pow(2).sum() / count
        adv_std = adv_var.sqrt().clamp_min(1e-6)
        normalized_advantages = (advantages - adv_mean) / adv_std
        normalized_advantages = normalized_advantages * resp_mask_f

        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=trainer.config.training_params.use_mixed_precision,
            dtype=self.float16_type,
        ):
            logits, _, new_values = trainer.policy_model(
                rollout_data["full_tensors"], attention_mask=rollout_data["attn_mask"]
            )
            new_log_probs = get_log_probs(logits, rollout_data["full_tensors"])

            actor_loss = (
                -(normalized_advantages * new_log_probs * resp_mask_f).sum() / count
            )

            critic_loss = compute_value_loss(
                new_values=new_values,
                old_values=rollout_data["values"],
                returns=returns,
                loss_mask=rollout_data["resp_mask"],
                training_params=trainer.config.training_params,
            )

            entropy_loss = compute_entropy_loss(logits, rollout_data["resp_mask"])

            total_loss = (
                actor_loss
                + trainer.config.training_params.value_coeff * critic_loss
                - trainer.config.training_params.entropy_coeff * entropy_loss
            )

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(trainer.optimizer)

        torch.nn.utils.clip_grad_norm_(
            trainer.policy_model.parameters(),
            trainer.config.training_params.max_grad_norm,
        )

        scale_before = self.scaler.get_scale()
        self.scaler.step(trainer.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        overflow_occurred = scale_after < scale_before

        metrics = {}
        if not overflow_occurred:
            metrics = {
                "actor_loss": float(actor_loss.detach().item()),
                "critic_loss": float(critic_loss.detach().item()),
                "entropy_loss": float(-entropy_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "grad_scale": float(scale_after),
                "advantage_mean": float(normalized_advantages.mean().item()),
                "advantage_std": float(adv_std.item()),
                "value_mean": float(new_values.mean().item()),
                "return_mean": float(returns.mean().item()),
            }

        return metrics

    def evaluate(
        self, trainer: "MixtureOfExpertsTrainer", test_dataloader, run_id: str = None
    ) -> Dict[str, float]:
        self.logger.info("Starting A2C model evaluation...")

        trainer.policy_model.eval()

        if hasattr(trainer.policy_model, "expert_tracker"):
            trainer.policy_model.expert_tracker.reset_stats()

        eval_metrics = []
        total_samples = 0

        progress_bar = tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            desc="A2C Evaluating",
        )

        eval_start_time = time.perf_counter()

        with torch.inference_mode():
            for batch_idx, batch_data in progress_bar:
                try:
                    if batch_data is None:
                        continue

                    batch_start_time = time.perf_counter()

                    device = trainer.device
                    query_tensors = batch_data.input_ids.to(device)
                    batch_size = query_tensors.size(0)
                    total_samples += batch_size

                    with autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                        enabled=trainer.config.training_params.use_mixed_precision,
                        dtype=self.float16_type,
                    ):
                        generation_start_time = time.perf_counter()
                        response_tensors = trainer._generate_responses(query_tensors)
                        generation_time = time.perf_counter() - generation_start_time

                    reward_start_time = time.perf_counter()
                    rewards = trainer._compute_rewards_batch(
                        batch_data.queries, response_tensors, batch_data.ground_truths
                    )
                    reward_time = time.perf_counter() - reward_start_time

                    full_tensors = torch.cat([query_tensors, response_tensors], dim=1)
                    attention_mask = (
                        full_tensors != trainer.tokenizer.pad_token_id
                    ).long()

                    with autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                        enabled=trainer.config.training_params.use_mixed_precision,
                        dtype=self.float16_type,
                    ):
                        logits, _, values = trainer.policy_model(
                            full_tensors,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )

                    B, T = full_tensors.shape
                    qlen = query_tensors.shape[1]
                    resp_mask = torch.zeros_like(full_tensors, dtype=torch.bool)
                    resp_mask[:, qlen:] = True
                    resp_mask &= attention_mask.bool()

                    eos_id = trainer.tokenizer.eos_token_id
                    if eos_id is not None:
                        resp_part = full_tensors[:, qlen:]
                        eos_pos = (resp_part == eos_id).float()
                        first_eos = eos_pos.argmax(dim=1)
                        has_eos = eos_pos.max(dim=1).values.bool()
                        idx = (
                            torch.arange(resp_part.size(1), device=device)
                            .unsqueeze(0)
                            .expand(B, -1)
                        )
                        before_eos = idx < first_eos.unsqueeze(1)
                        before_eos = torch.where(
                            has_eos.unsqueeze(1),
                            before_eos,
                            torch.ones_like(before_eos, dtype=torch.bool),
                        )
                        trimmed_mask = torch.zeros_like(resp_mask)
                        trimmed_mask[:, qlen:] = before_eos
                        resp_mask &= trimmed_mask

                    entropy = compute_entropy_loss(logits, resp_mask)

                    nlp_metrics = compute_nlp_metrics(
                        self.bleu_metric,
                        self.rouge_metric,
                        query_tensors=query_tensors,
                        logits=logits,
                        response_tensors=response_tensors,
                        batch_data=batch_data,
                        tokenizer=trainer.tokenizer,
                        prefix="eval_",
                    )

                    batch_metrics = {
                        "reward": rewards.mean().item(),
                        "entropy": entropy.item(),
                        "value_estimate": values.mean().item(),
                        "response_length": resp_mask.sum(dim=1).float().mean().item(),
                        "generation_time": generation_time,
                        "reward_computation_time": reward_time,
                        "batch_processing_time": time.perf_counter() - batch_start_time,
                        "batch_size": batch_size,
                    }

                    batch_metrics.update(nlp_metrics)
                    eval_metrics.append(batch_metrics)

                    display_metrics = {
                        "reward": f"{batch_metrics['reward']:.4f}",
                        "entropy": f"{batch_metrics['entropy']:.4f}",
                    }
                    if "eval_bleu" in batch_metrics:
                        display_metrics["bleu"] = f"{batch_metrics['eval_bleu']:.4f}"

                    progress_bar.set_postfix(display_metrics)

                except Exception as e:
                    self.logger.error(
                        f"Evaluation batch failed: batch_idx={batch_idx}, error={e}",
                        exc_info=True,
                    )
                    continue

        total_eval_time = time.perf_counter() - eval_start_time

        if not eval_metrics:
            self.logger.warning("No valid evaluation batches processed")
            return {}

        aggregated_metrics = self._aggregate_metrics(eval_metrics)

        aggregated_metrics.update(
            {
                "eval_total_time": total_eval_time,
                "eval_total_samples": total_samples,
                "eval_samples_per_second": (
                    total_samples / total_eval_time if total_eval_time > 0 else 0
                ),
                "eval_batches_processed": len(eval_metrics),
            }
        )

        if hasattr(trainer.policy_model, "expert_tracker"):
            expert_usage = trainer.policy_model.expert_tracker.get_usage_stats()
            eval_expert_usage = {f"eval_{k}": v for k, v in expert_usage.items()}
            aggregated_metrics.update(eval_expert_usage)
            self.logger.info(f"Expert usage during evaluation: {expert_usage}")

        if run_id and self.mlflow_client:
            try:
                mlflow_log_metrics(
                    self.mlflow_client,
                    self.config,
                    aggregated_metrics,
                    step=0,
                    run_id=run_id,
                )
            except Exception as e:
                self.logger.error(f"Failed to log evaluation metrics to MLflow: {e}")

        self.logger.info("A2C Evaluation completed:")
        self.logger.info(f"  - Total samples: {total_samples}")
        self.logger.info(
            f"  - Average reward: {aggregated_metrics.get('avg_reward', 0):.4f}"
        )
        self.logger.info(
            f"  - Average entropy: {aggregated_metrics.get('avg_entropy', 0):.4f}"
        )

        if "avg_eval_bleu" in aggregated_metrics:
            self.logger.info(
                f"  - Average BLEU score: {aggregated_metrics['avg_eval_bleu']:.4f}"
            )
        if "avg_eval_rouge1" in aggregated_metrics:
            self.logger.info(
                f"  - Average ROUGE-1: {aggregated_metrics['avg_eval_rouge1']:.4f}"
            )

        trainer.policy_model.train()

        return aggregated_metrics
