import logging
import threading
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from trl import AutoModelForCausalLMWithValueHead


def gini_coefficient(values):
    """
    Computes the Gini coefficient for a list of values.
    A value of 0 represents perfect equality, 1 represents maximal inequality.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Gini coefficient is not defined for negative values.")
    if np.sum(values) == 0:
        return 0.0  

    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)

    numerator = np.sum((2 * index - n - 1) * values)
    denominator = n * np.sum(values)

    return numerator / denominator


def herfindahl_hirschman_index(counts):
    """
    Computes the Herfindahl-Hirschman Index (HHI) for a list of counts.
    A value of 1 represents a total monopoly (complete collapse).
    A value close to 0 represents a perfectly distributed load.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("HHI is not defined for negative values.")

    total = np.sum(counts)
    if total == 0:
        return 0.0

    percentages = counts / total
    hhi = np.sum(percentages**2)

    return hhi


class PatchedAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    """
    A patched version of AutoModelForCausalLMWithValueHead that fixes a bug
    in the state_dict method, making it compatible with PEFT checkpointing.
    Before that, I was getting an error about dictionary mutation.
    """

    def state_dict(self, *args, **kwargs):
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_head.state_dict(*args)

        if pretrained_model_state_dict is not None:
            for k, v in v_head_state_dict.items():
                pretrained_model_state_dict[f"v_head.{k}"] = v
            return pretrained_model_state_dict

        else:
            destination = kwargs.get("destination")
            if destination is not None:
                prefix = kwargs.get("prefix", "")
                for k, v in v_head_state_dict.items():
                    destination[f"{prefix}v_head.{k}"] = v
            return None


class ExpertUsageTracker:

    def __init__(self):
        """
        Initializes the ExpertUsageTracker instance.

        Sets up data structures to track expert usage statistics:
        - expert_counts: Counts the number of tokens routed to each expert.
        - total_tokens: Tracks the total number of tokens processed.
        - layer_stats: Stores per-layer expert usage statistics.
        - _lock: Threading lock to ensure thread-safe updates.
        """
        self.expert_counts = defaultdict(int)
        self.total_tokens = 0
        self.layer_stats = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

    def reset_stats(self):
        """
        Resets all tracked statistics for expert usage.

        This method clears the expert usage counts, resets the total token count to zero,
        and clears per-layer statistics. Thread-safe via internal locking.
        """
        with self._lock:
            self.expert_counts.clear()
            self.total_tokens = 0
            self.layer_stats.clear()

    def update(self, router_logits: torch.Tensor, layer_name: Optional[str] = None):
        if router_logits is None or router_logits.numel() == 0:
            return

        with torch.no_grad():
            chosen_experts = torch.argmax(router_logits, dim=-1).flatten()
            num_tokens = chosen_experts.numel()

            unique_indices, counts = torch.unique(chosen_experts, return_counts=True)

            unique_indices_list = unique_indices.cpu().tolist()
            counts_list = counts.cpu().tolist()

        with self._lock:
            self.total_tokens += num_tokens
            for idx, count in zip(unique_indices_list, counts_list):
                expert_id = int(idx)
                self.expert_counts[expert_id] += count
                if layer_name:
                    self.layer_stats[layer_name][expert_id] += count

    def get_usage_stats(self) -> Dict[str, float | int]:
        """
        Computes and returns usage statistics for experts.

        Returns:
            Dict[str, float | int]: A dictionary containing:
                - "expert_usage_percent": A mapping of expert names to their usage percentage.
                - "total_tokens": The total number of tokens processed.
                - "layer_stats_raw_counts": Raw counts of layer statistics.
                - "gini_coefficient": The Gini coefficient representing usage inequality among experts.
                - "herfindahl_hirschman_index": The Herfindahl-Hirschman Index representing usage concentration among experts.
            If no tokens have been processed, returns zeroed statistics.
        """
        with self._lock:
            if self.total_tokens == 0:
                return {"expert_usage_percent": {}, "total_tokens": 0}

            usage_percent = {
                f"expert_{expert}": count / self.total_tokens
                for expert, count in self.expert_counts.items()
            }

            # total_percentage = sum(usage_percent.values())
            # print(f"Sum of all expert usage percentages: {total_percentage * 100:.2f}%")

            num_experts = len(self.expert_counts)
            all_counts = [self.expert_counts.get(i, 0) for i in range(num_experts)]
            gini = gini_coefficient(all_counts)
            hhi = herfindahl_hirschman_index(all_counts)

            stats = {
                "expert_usage_percent": usage_percent,
                "total_tokens": self.total_tokens,
                "layer_stats_raw_counts": dict(self.layer_stats),
                "gini_coefficient": gini,
                "herfindahl_hirschman_index": hhi,
            }
            return stats


class MoEModelWithTracking(nn.Module):

    def __init__(
        self,
        model: PatchedAutoModelForCausalLMWithValueHead,
        gate_name_pattern: str = ".mlp.gate",
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.model_with_value_head = model
        self.gate_name_pattern = gate_name_pattern
        self.logger = logger or logging.getLogger(__name__)
        self.expert_tracker = ExpertUsageTracker()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward hooks on modules within `self.model_with_value_head` whose names end with `self.gate_name_pattern`.
        These hooks are used for expert usage tracking in Mixture-of-Experts (MoE) models.

        - Iterates through all named modules in the model.
        - For each module whose name matches the gate pattern, registers a forward hook and stores the handle.
        - Logs a warning if no matching modules are found, disabling expert tracking.
        - Logs an info message indicating the number of hooks registered if successful.
        """
        for name, module in self.model_with_value_head.named_modules():
            if name.endswith(self.gate_name_pattern):
                handle = module.register_forward_hook(self._create_hook(name))
                self._hooks.append(handle)

        if not self._hooks:
            self.logger.warning(
                f"No MoE gate modules found matching pattern '{self.gate_name_pattern}'. "
                "Expert tracking will be disabled."
            )
        else:
            self.logger.info(f"Registered {len(self._hooks)} expert hooks.")

    def _create_hook(self, layer_name: str):
        """
        Creates a forward hook function for a given layer name.

        The returned hook, when registered to a module, will update the expert tracker
        with the output of the module and the specified layer name during the forward pass.

        Args:
            layer_name (str): The name of the layer for which the hook is created.

        Returns:
            Callable: A hook function to be registered to a module.
        """

        def hook(module, inputs, output):
            self.expert_tracker.update(output, layer_name)

        return hook

    def forward(self, *args, **kwargs):
        return self.model_with_value_head(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model_with_value_head, name)

    def cleanup_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.logger.info("Removed all expert tracking hooks.")

    def __del__(self):
        self.cleanup_hooks()
