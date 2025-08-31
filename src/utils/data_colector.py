from typing import Dict, List

from transformers import AutoTokenizer

from .batch_data import BatchData


class CustomDataCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        pad_to_multiple_of: int = 8,
    ):
        """
        Initializes the data collector with the specified tokenizer and configuration.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to be used for processing data.
            max_length (int, optional): The maximum sequence length for tokenization. Defaults to 512.
            pad_to_multiple_of (int, optional): If set, pad sequences to a multiple of this value. Defaults to 8.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> BatchData:
        """
        Processes a batch of feature dictionaries and returns a BatchData object containing tokenized queries and associated data.

        Args:
            features (List[Dict]): A list of dictionaries, each containing at least the keys "query", "question", and "ground_truth".

        Returns:
            BatchData: An object containing tokenized input IDs, attention masks, original queries, questions, and ground truths.

        Notes:
            - Tokenization is performed using the instance's tokenizer with padding, truncation, and maximum length settings.
            - The returned BatchData object aggregates all relevant fields for downstream processing.
        """
        queries_text = [f["query"] for f in features]

        tokenized_batch = self.tokenizer(
            queries_text,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return BatchData(
            input_ids=tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["attention_mask"],
            queries=queries_text,
            questions=[f["question"] for f in features],
            ground_truths=[f["ground_truth"] for f in features],
        )
