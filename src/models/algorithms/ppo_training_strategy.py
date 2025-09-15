import logging
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

import mlflow
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from models.algorithms.training_strategy import TrainingStrategy
from models.training_monitoring import PerformanceMonitor
from models.utils.distributed_manager import DistributedManager
from models.utils.loss import (
    compute_advantages,
    compute_entropy_loss,
    compute_policy_loss,
    compute_value_loss,
    get_log_probs,
    masked_mean,
)
from models.utils.metrics import compute_nlp_metrics, mlflow_log_metrics
from utils.batch_data import BatchData
from utils.configurations import MoERLConfig
from utils.exceptions import TrainingError
from utils.ressource_manager import TrainingResourceManager

if TYPE_CHECKING:
    from models.trainer import MixtureOfExpertsTrainer


class AdaptiveKLController:

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        """
        Initializes the training strategy with KL coefficient, target value, and horizon.

        Args:
            init_kl_coef (float): Initial KL divergence coefficient.
            target (float): Target value for the training strategy.
            horizon (int): Number of steps or episodes over which to adjust the strategy.
        """
        self.kl_coef = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        """
        Updates the KL coefficient based on the current KL divergence and the number of steps taken.

        Args:
            current_kl (float): The current KL divergence value.
            n_steps (int): The number of steps taken since the last update.

        Modifies:
            self.kl_coef: Scales the KL coefficient by a factor determined by the proportional error
                between current and target KL, clipped to [-0.2, 0.2], and adjusted by the number of steps
                and the horizon.

        """
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        factor = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= factor


class PPOTrainingStrategy(TrainingStrategy):

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
            resource_manager=resource_manager,
            performance_monitor=performance_monitor,
            logger=logger,
            config=config,
            mlflow_client=mlflow_client,
            distributed_manager=distributed_manager,
        )

        self.kl_controller = AdaptiveKLController(
            init_kl_coef=0.2,
            target=6.0,
            horizon=10000,
        )

    def _collect_rollout(
        self, trainer: "MixtureOfExpertsTrainer", batch_data: BatchData
    ) -> Tuple[Dict, Dict]:
        if batch_data is None:
            raise TrainingError("Failed to prepare batch.")

        device = trainer.device
        query_tensors = batch_data.input_ids.to(device)

        with torch.inference_mode():
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=trainer.config.training_params.use_mixed_precision,
                dtype=self.float16_type,
            ):
                response_tensors = trainer._generate_responses(query_tensors)

        base_rewards = trainer._compute_rewards_batch(
            batch_data.queries, response_tensors, batch_data.ground_truths
        )

        eos_id = trainer.tokenizer.eos_token_id

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
                ref_logits, _, _ = trainer.reference_model(
                    full_tensors, attention_mask=attention_mask
                )
                ref_log_probs = get_log_probs(ref_logits, full_tensors)

        B, T = full_tensors.shape
        qlen = query_tensors.shape[1]

        resp_mask = torch.zeros_like(full_tensors, dtype=torch.bool)
        resp_mask[:, qlen:] = True
        resp_mask &= attention_mask.bool()

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

        per_token_kl = log_probs - ref_log_probs

        resp_mask_f = resp_mask.float()
        denom = resp_mask_f.sum(dim=1).clamp_min(1.0)
        kl_seq = (per_token_kl * resp_mask_f).sum(dim=1) / denom

        self.kl_controller.update(kl_seq.mean().item(), n_steps=1)

        kl_penalty = self.kl_controller.kl_coef * kl_seq
        rewards = base_rewards - kl_penalty

        metrics = {
            "kl_coef": self.kl_controller.kl_coef,
            "kl_div": kl_seq.mean().item(),
            "reward": rewards.mean().item(),
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
            "log_probs": log_probs.detach(),
            "values": values.detach(),
            "rewards": rewards.detach(),
            "ref_log_probs": ref_log_probs.detach(),
            "resp_mask": resp_mask,
            "attn_mask": attention_mask,
        }
        return rollout_data, metrics

    def _update_policy(
        self, trainer: "MixtureOfExpertsTrainer", rollout_data: Dict
    ) -> Dict[str, float]:
        """
        Performs policy update using PPO (Proximal Policy Optimization) on the provided rollout data.

        This method computes advantages and returns, normalizes advantages, and iteratively updates the policy
        using mini-batches over multiple PPO epochs. It collects loss metrics from each update and returns their mean.

        Args:
            trainer (PPOMoETrainer): The PPO trainer instance containing model, tokenizer, and configuration.
            rollout_data (Dict): A dictionary containing rollout tensors such as rewards, values, queries, responses,
                masks, log probabilities, and reference log probabilities.

        Returns:
            Dict[str, float]: A dictionary mapping loss metric names to their averaged values across all mini-batches.
        """
        advantages, returns = compute_advantages(
            rollout_data["rewards"],
            rollout_data["values"],
            torch.zeros_like(rollout_data["rewards"], dtype=torch.bool),
            trainer.config.training_params,
        )

        full_tensors = torch.cat(
            [rollout_data["queries"], rollout_data["responses"]], dim=1
        )
        loss_mask = rollout_data["resp_mask"] & rollout_data["attn_mask"].bool()

        adv = advantages.clone()
        mask_f = loss_mask.float()
        count = mask_f.sum().clamp_min(1.0)
        mean = (adv * mask_f).sum() / count
        var = ((adv - mean) * mask_f).pow(2).sum() / count
        std = var.sqrt().clamp_min(1e-6)
        advantages = (adv - mean) / std
        advantages = advantages * mask_f

        batch_size = full_tensors.size(0)
        mini_batch_size = trainer.config.training_params.mini_batch_size

        losses = []
        nb_ppo_epochs = trainer.config.training_params.ppo_epochs

        old_log_probs = rollout_data["log_probs"]
        ref_log_probs = rollout_data["ref_log_probs"]
        old_values = rollout_data["values"]

        for _ in range(nb_ppo_epochs):
            indices = torch.randperm(batch_size, device=full_tensors.device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_full = full_tensors[mb_idx]
                mb_attn = (mb_full != trainer.tokenizer.pad_token_id).long()
                mb_old_logp = old_log_probs[mb_idx]
                mb_ref_logp = ref_log_probs[mb_idx]
                mb_old_v = old_values[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_mask = loss_mask[mb_idx]

                self._update(
                    trainer,
                    mb_full,
                    mb_attn,
                    mb_old_logp,
                    mb_ref_logp,
                    mb_adv,
                    mb_returns,
                    losses,
                    mb_mask,
                    mb_old_v,
                )

        if not losses:
            return {}
        out = {}
        keys = losses[0].keys()
        for k in keys:
            out[k] = float(np.mean([d[k] for d in losses if k in d]))
        return out

    def _update(
        self,
        trainer: "MixtureOfExpertsTrainer",
        mb_full_tensors,
        mb_attention_mask,
        mb_old_log_probs,
        mb_ref_log_probs,
        mb_advantages,
        mb_returns,
        losses: List[Dict[str, float]],
        mb_loss_mask: torch.Tensor,
        old_values: torch.Tensor,
    ) -> bool:
        """
        Performs a single update step for the PPO-MoE training loop, including forward pass, loss computation,
        backward pass with mixed precision, gradient clipping, optimizer step, and loss logging.

        Args:
            trainer (PPOMoETrainer): The trainer object containing the policy model, optimizer, and config.
            mb_full_tensors: Mini-batch input tensors for the policy model.
            mb_attention_mask: Attention mask for the input tensors.
            mb_old_log_probs: Log probabilities from the previous policy.
            mb_ref_log_probs: Reference log probabilities for KL divergence computation.
            mb_advantages: Advantage estimates for the mini-batch.
            mb_returns: Return values for the mini-batch.
            losses (List[Dict[str, float]]): List to append loss statistics for logging.
            mb_loss_mask (torch.Tensor): Mask indicating valid loss elements in the mini-batch.
            old_values (torch.Tensor): Value predictions from the previous policy.

        Returns:
            bool: True if the update was successful (no gradient overflow), False otherwise.
        """
        trainer.optimizer.zero_grad()

        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=trainer.config.training_params.use_mixed_precision,
            dtype=self.float16_type,
        ):
            logits, _, new_values = trainer.policy_model(
                mb_full_tensors, attention_mask=mb_attention_mask
            )
            new_log_probs = get_log_probs(logits, mb_full_tensors)

            policy_loss = compute_policy_loss(
                mb_advantages,
                new_log_probs,
                mb_old_log_probs,
                mb_loss_mask,
                trainer.config.training_params,
            )

            value_loss = compute_value_loss(
                new_values=new_values,
                old_values=old_values,
                returns=mb_returns,
                loss_mask=mb_loss_mask,
                training_params=trainer.config.training_params,
            )

            entropy_mean = compute_entropy_loss(logits, mb_loss_mask)

            kl_divergence = (new_log_probs - mb_ref_log_probs) * mb_loss_mask
            kl_mean = masked_mean(kl_divergence, mb_loss_mask)

            total_loss = (
                policy_loss
                + trainer.config.training_params.value_coeff * value_loss
                - trainer.config.training_params.entropy_coeff * entropy_mean
            )

            entropy_loss = -entropy_mean

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

        if not overflow_occurred:
            losses.append(
                {
                    "policy_loss": float(policy_loss.detach().item()),
                    "value_loss": float(value_loss.detach().item()),
                    "entropy_loss": float(entropy_loss.detach().item()),
                    "kl_div": float(kl_mean.detach().item()),
                    "total_loss": float(total_loss.detach().item()),
                    "grad_scale": float(scale_after),
                }
            )

        return not overflow_occurred

    def evaluate(
        self, trainer: "MixtureOfExpertsTrainer", test_dataloader, run_id: str = None
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on the test dataset.

        Args:
            trainer: The PPOMoETrainer instance
            test_dataloader: DataLoader containing test batches
            run_id: MLflow run ID for logging metrics

        Returns:
            Dictionary containing aggregated evaluation metrics
        """
        self.logger.info("Starting model evaluation...")

        trainer.policy_model.eval()
        trainer.reference_model.eval()

        if hasattr(trainer.policy_model, "expert_tracker"):
            trainer.policy_model.expert_tracker.reset_stats()

        eval_metrics = []
        total_samples = 0

        progress_bar = tqdm(
            enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"
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
                    base_rewards = trainer._compute_rewards_batch(
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
                        policy_logits, _, policy_values = trainer.policy_model(
                            full_tensors,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )
                        policy_log_probs = get_log_probs(policy_logits, full_tensors)

                        ref_logits, _, _ = trainer.reference_model(
                            full_tensors, attention_mask=attention_mask
                        )
                        ref_log_probs = get_log_probs(ref_logits, full_tensors)

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

                    per_token_kl = policy_log_probs - ref_log_probs
                    resp_mask_f = resp_mask.float()
                    denom = resp_mask_f.sum(dim=1).clamp_min(1.0)
                    kl_seq = (per_token_kl * resp_mask_f).sum(dim=1) / denom

                    kl_penalty = self.kl_controller.kl_coef * kl_seq
                    final_rewards = base_rewards - kl_penalty

                    entropy = compute_entropy_loss(policy_logits, resp_mask)

                    nlp_metrics = compute_nlp_metrics(
                        self.bleu_metric,
                        self.rouge_metric,
                        query_tensors=query_tensors,
                        logits=policy_logits,
                        response_tensors=response_tensors,
                        batch_data=batch_data,
                        tokenizer=trainer.tokenizer,
                        prefix="eval_",
                    )

                    batch_metrics = {
                        "reward": final_rewards.mean().item(),
                        "base_reward": base_rewards.mean().item(),
                        "kl_divergence": kl_seq.mean().item(),
                        "kl_penalty": kl_penalty.mean().item(),
                        "entropy": entropy.item(),
                        "value_estimate": policy_values.mean().item(),
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
                        "kl_div": f"{batch_metrics['kl_divergence']:.4f}",
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

        self.logger.info("Evaluation completed:")
        self.logger.info(f"  - Total samples: {total_samples}")
        self.logger.info(
            f"  - Average reward: {aggregated_metrics.get('avg_reward', 0):.4f}"
        )
        self.logger.info(
            f"  - Average KL divergence: {aggregated_metrics.get('avg_kl_divergence', 0):.4f}"
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
        trainer.reference_model.train()

        return aggregated_metrics
