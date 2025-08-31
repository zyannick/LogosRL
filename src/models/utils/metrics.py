from typing import Dict

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from utils.batch_data import BatchData


def compute_perplexity(
    query_tensors: torch.Tensor,
    logits: torch.Tensor,
    response_tensors: torch.Tensor,
    tokenizer,
) -> float:
    """
    Computes the perplexity of model predictions for a given response, conditioned on a query.

    Args:
        query_tensors (torch.Tensor): Tensor containing tokenized queries of shape (batch_size, query_length).
        logits (torch.Tensor): Tensor of model output logits of shape (batch_size, sequence_length, vocab_size).
        response_tensors (torch.Tensor): Tensor containing tokenized responses of shape (batch_size, response_length).
        tokenizer: Tokenizer object used to obtain the pad token ID.

    Returns:
        float: The computed perplexity value for the given responses.
    """
    query_len = query_tensors.shape[1]

    response_logits = logits[:, query_len - 1 : -1, :].contiguous()

    response_labels = response_tensors.contiguous()

    shift_logits = response_logits.view(-1, response_logits.size(-1))
    shift_labels = response_labels.view(-1)

    loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id
    )

    perplexity = torch.exp(loss).item()
    return perplexity


def compute_nlp_metrics(
    bleu_metric,
    rouge_metric,
    query_tensors: torch.Tensor,
    logits: torch.Tensor,
    response_tensors: torch.Tensor,
    batch_data: BatchData,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str = "",
):
    """
    Computes NLP evaluation metrics (perplexity, BLEU, ROUGE) for model predictions.

    Args:
        bleu_metric: An object with a `compute` method for calculating BLEU scores.
        rouge_metric: An object with a `compute` method for calculating ROUGE scores.
        query_tensors (torch.Tensor): Input query tensors to the model.
        logits (torch.Tensor): Model output logits.
        response_tensors (torch.Tensor): Model-generated response tensors.
        batch_data (BatchData): Batch data containing ground truth responses.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for decoding tensors to text.
        prefix (str, optional): Prefix to prepend to metric names in the output dictionary.

    Returns:
        dict: A dictionary containing computed metrics with keys:
            - "<prefix>perplexity": Perplexity value.
            - "<prefix>bleu": BLEU score (if computed).
            - "<prefix>rouge*": ROUGE scores (if computed), with metric names prefixed.
    """

    perplexity_value = compute_perplexity(
        query_tensors, logits, response_tensors, tokenizer
    )

    predictions_text = tokenizer.batch_decode(
        response_tensors, skip_special_tokens=True
    )

    references_text_bleu = [[ref] for ref in batch_data.ground_truths]

    bleu_results = bleu_metric.compute(
        predictions=predictions_text, references=references_text_bleu
    )

    rouge_results = rouge_metric.compute(
        predictions=predictions_text, references=batch_data.ground_truths
    )

    metrics_nlp = {f"{prefix}perplexity": perplexity_value}
    if bleu_results:
        metrics_nlp[f"{prefix}bleu"] = bleu_results.get("bleu", 0.0)
    if rouge_results:
        metrics_nlp.update({f"{prefix}{k}": v for k, v in rouge_results.items()})

    return metrics_nlp


def mlflow_log_metrics(
    mlflow_client, config, metrics: Dict[str, float], step: int, run_id: str
):
    """
    Logs metrics to MLflow, supporting nested dictionaries by flattening keys.

    Args:
        mlflow_client: An MLflow client instance used to log metrics.
        config: Configuration object
        metrics (Dict[str, float]): Dictionary of metrics to log. Can contain nested dictionaries.
        step (int): The step at which the metrics are logged.
        run_id (str): The MLflow run ID to which metrics are logged.

    Notes:
        - Nested dictionaries in `metrics` are flattened using dot notation for keys.
        - Only leaf values (non-dict) are logged as metrics.
    """

    def recursive_logger(metrics_dict: Dict[str, float], parent_key: str = ""):
        for key, value in metrics_dict.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                recursive_logger(value, full_key)
            else:
                mlflow_client.log_metric(
                    run_id=run_id, key=full_key, value=value, step=step
                )

    recursive_logger(metrics)
