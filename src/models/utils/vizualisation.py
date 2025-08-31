from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_overall_expert_utilization(
    log_data: List[Dict[str, Any]], saving_path: Path, step: int = 0
):
    """
    Plots a bar chart showing the overall percentage usage of each expert.

    Args:
        log_data (list): A list of log entries, where each entry is a dictionary.
    """
    try:
        latest_log_entry = log_data[-1]
        expert_usage = latest_log_entry["expert_usage_percent"]

        expert_labels = sorted(expert_usage.keys(), key=lambda x: int(x.split("_")[1]))
        usage_percentages = [expert_usage[label] * 100 for label in expert_labels]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(18, 10))

        sns.barplot(x=expert_labels, y=usage_percentages, ax=ax, palette="viridis")

        ax.set_title(
            "Mixture of Experts (MoE) - Overall Expert Utilization",
            fontsize=20,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Expert ID", fontsize=14, fontweight="bold", labelpad=15)
        ax.set_ylabel(
            "Usage Percentage (%)", fontsize=14, fontweight="bold", labelpad=15
        )
        ax.tick_params(axis="x", rotation=75)

        for _, p in enumerate(ax.patches):
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                xytext=(0, 10),
                textcoords="offset points",
            )

        average_usage = np.mean(usage_percentages)
        ax.axhline(
            average_usage,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"Average Usage: {average_usage:.2f}%",
        )
        ax.legend(fontsize=12)

        ax.set_ylim(0, max(usage_percentages) * 1.15)
        plt.tight_layout()
        plt.savefig(
            saving_path / f"overall_expert_utilization_{str(step).zfill(4)}.png",
            dpi=300,
        )

    except (IndexError, KeyError) as e:
        print(f"Could not plot overall utilization due to missing data: {e}")


def plot_layer_wise_expert_heatmap(
    log_data: List[Dict[str, Any]], saving_path: Path, step: int = 0
):
    """
    Plots a heatmap showing the normalized usage of each expert per layer.

    Args:
        log_data (list): A list of log entries, where each entry is a dictionary.
    """
    try:
        layer_expert_counts = defaultdict(lambda: defaultdict(int))
        for entry in log_data:
            for layer, expert_counts in entry["layer_stats_raw_counts"].items():
                layer_num = int(layer.split(".")[5])
                for expert, count in expert_counts.items():
                    expert_num = int(expert)
                    layer_expert_counts[layer_num][expert_num] += count

        layer_expert_df = pd.DataFrame(layer_expert_counts).fillna(0)
        layer_expert_df = layer_expert_df.sort_index(axis=0).sort_index(axis=1)

        layer_expert_norm = layer_expert_df.div(layer_expert_df.sum(axis=0), axis=1)

        plt.figure(figsize=(20, 12))
        sns.heatmap(
            layer_expert_norm,
            cmap="mako",
            cbar_kws={"label": "Normalized Usage"},
            linewidths=0.3,
            linecolor="grey",
        )

        plt.title("Layer-wise Expert Usage Heatmap", fontsize=18, weight="bold", pad=20)
        plt.xlabel("Layer Number", fontsize=14, weight="bold")
        plt.ylabel("Expert ID", fontsize=14, weight="bold")
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            saving_path / f"layer_wise_expert_heatmap_{str(step).zfill(4)}.png", dpi=300
        )
        plt.close()

    except (IndexError, KeyError) as e:
        print(f"Could not plot heatmap due to missing data: {e}")
