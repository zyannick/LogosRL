from typing import Tuple

import torch
import torch.nn.functional as F

from utils.configurations import TrainingParams


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8):
    """
    Computes the mean of tensor `x` over the specified dimension(s), considering only the elements where `mask` is nonzero.

    Args:
        x (torch.Tensor): The input tensor.
        mask (torch.Tensor): A tensor of the same shape as `x` indicating which elements to include in the mean (nonzero values are included).
        dim (int or tuple of ints, optional): The dimension(s) over which to compute the mean. If None, computes over all elements.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: The masked mean of `x` over the specified dimension(s).
    """
    m = mask.float()
    if dim is None:
        return (x * m).sum() / (m.sum().clamp_min(eps))
    return (x * m).sum(dim=dim) / (m.sum(dim=dim).clamp_min(eps))


def entropy_and_kl_from_logits(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mean entropy and mean KL divergence between two categorical distributions
    defined by their logits, applying a mask to select relevant elements.

    Args:
        policy_logits (torch.Tensor): Logits for the policy distribution (shape: [batch_size, num_classes, ...]).
        ref_logits (torch.Tensor): Logits for the reference distribution (shape: [batch_size, num_classes, ...]).
        loss_mask (torch.Tensor): Boolean or float mask tensor indicating which elements to include in the mean.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Mean entropy of the policy distribution over masked elements.
            - Mean KL divergence between the policy and reference distributions over masked elements.
    """

    policy_dist = torch.distributions.Categorical(logits=policy_logits)
    ref_dist = torch.distributions.Categorical(logits=ref_logits)

    entropy_per_tok = policy_dist.entropy()
    kl_per_tok = torch.distributions.kl.kl_divergence(policy_dist, ref_dist)

    entropy_mean = masked_mean(entropy_per_tok, loss_mask)
    kl_mean = masked_mean(kl_per_tok, loss_mask)

    return entropy_mean, kl_mean


def compute_policy_loss(
    advantages: torch.Tensor,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask: torch.Tensor,
    params: TrainingParams,
):
    """
    Computes the clipped surrogate policy loss for Proximal Policy Optimization (PPO).

    Args:
        advantages (torch.Tensor): Advantage estimates for each action.
        new_log_probs (torch.Tensor): Log probabilities of actions under the current policy.
        old_log_probs (torch.Tensor): Log probabilities of actions under the previous policy.
        mask (torch.Tensor): Boolean mask indicating valid entries for loss computation.
        params (TrainingParams): Training parameters containing the PPO clipping epsilon.

    Returns:
        torch.Tensor: Scalar tensor representing the mean policy loss over the masked entries.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    ratio = torch.clamp(ratio, 1e-6, 1e6)
    surr1 = ratio * advantages
    surr2 = (
        torch.clamp(ratio, 1.0 - params.clip_epsilon, 1.0 + params.clip_epsilon)
        * advantages
    )
    policy_loss = -torch.min(surr1, surr2)[mask].mean()

    return policy_loss


def compute_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: torch.Tensor,
    training_params: TrainingParams,
):
    """
    Computes the value loss for reinforcement learning with optional value clipping.

    This function calculates two versions of the mean squared error (MSE) loss between predicted values and target returns:
    one using clipped predictions and one using unclipped predictions. The final loss is the maximum of the two, following
    the PPO-style value loss clipping. Only elements specified by `loss_mask` are considered in the loss computation.

    Args:
        new_values (torch.Tensor): Predicted value estimates from the current policy.
        old_values (torch.Tensor): Value estimates from the previous policy.
        returns (torch.Tensor): Target returns for each sample.
        loss_mask (torch.Tensor): Boolean mask indicating which elements to include in the loss.
        training_params (TrainingParams): Training parameters containing the value clipping range (`cliprange_value`).

    Returns:
        torch.Tensor: The computed value loss.
    """
    v_pred_clipped = torch.clamp(
        new_values,
        old_values - training_params.cliprange_value,
        old_values + training_params.cliprange_value,
    )

    value_loss_clipped = F.mse_loss(v_pred_clipped[loss_mask], returns[loss_mask])
    value_loss_unclipped = F.mse_loss(new_values[loss_mask], returns[loss_mask])
    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

    return value_loss


def compute_entropy_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean entropy loss for the given logits, considering only the masked elements.

    Args:
        logits (torch.Tensor): The input logits tensor of shape (..., num_classes).
        mask (torch.Tensor): A boolean or integer mask tensor of shape (...,) indicating which elements to include in the loss.

    Returns:
        torch.Tensor: The mean entropy loss computed over the masked elements.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    masked_entropy = entropy[mask]
    return masked_entropy.mean()


def get_log_probs(logits, tokens):
    logits = torch.clamp(logits, min=-10, max=10)
    log_probs_dist = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_dist.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


# def compute_kl_div(trainer, mb_full_tensors, mb_attention_mask, old_log_probs, mask):
#     with torch.no_grad():
#         ref_logits, _, _ = trainer.reference_model(
#             mb_full_tensors, attention_mask=mb_attention_mask
#         )
#         ref_log_probs = get_log_probs(ref_logits, mb_full_tensors)

#     kl_div = (ref_log_probs - old_log_probs)[mask].mean()

#     return kl_div


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    training_params: TrainingParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Generalized Advantage Estimation (GAE) and returns for a batch of trajectories.

    Args:
        rewards (torch.Tensor): Tensor of rewards for each trajectory, shape (batch_size,).
        values (torch.Tensor): Tensor of value estimates for each timestep, shape (batch_size, sequence_length).
        dones (torch.Tensor): Tensor indicating episode termination (1 if done, 0 otherwise), shape (batch_size,).
        training_params (TrainingParams): Object containing training hyperparameters, including gamma (discount factor) and gae_lambda (GAE parameter).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - advantages (torch.Tensor): Normalized advantage estimates, shape (batch_size, sequence_length).
            - returns (torch.Tensor): Estimated returns (advantages + values), shape (batch_size, sequence_length).

    Notes:
        - Advantages are normalized unless their standard deviation is too small or NaN.
        - Uses reverse-time computation for GAE.
    """
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0

    sequence_length = values.size(1)
    rewards_tensor = torch.zeros_like(values)

    rewards_tensor[:, -1] = rewards

    dones_tensor = torch.zeros_like(values)
    dones_tensor[:, -1] = dones.float()

    last_gae_lam = 0
    advantages_reversed = []

    for t in reversed(range(sequence_length)):
        next_values = values[:, t + 1] if t < sequence_length - 1 else 0.0

        delta = (
            rewards_tensor[:, t]
            + training_params.gamma * next_values * (1.0 - dones_tensor[:, t])
            - values[:, t]
        )
        last_gae_lam = (
            delta
            + training_params.gamma
            * training_params.gae_lambda
            * (1.0 - dones_tensor[:, t])
            * last_gae_lam
        )
        advantages_reversed.append(last_gae_lam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values

    advantages_std = advantages.std()
    if advantages_std > 1e-8 and not torch.isnan(advantages_std):
        advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)
    else:
        print(
            f"Warning: advantages std too small ({advantages_std}) or NaN, skipping normalization"
        )

    return advantages, returns
