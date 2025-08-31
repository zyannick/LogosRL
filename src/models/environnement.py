import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from utils.configurations import MoERLConfig
from utils.data_colector import CustomDataCollator


class GSM8KEnvironment:

    def __init__(
        self,
        config: MoERLConfig,
        tokenizer: AutoTokenizer,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the environment with the given configuration, tokenizer, and optional logger.

        Args:
            config (MoERLConfig): Configuration object for the environment.
            tokenizer (AutoTokenizer): Tokenizer to process input data.
            logger (Optional[logging.Logger], optional): Logger instance for logging messages. Defaults to None.

        Side Effects:
            Loads data and prepares the environment for further operations.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger or logging.getLogger(__name__)
        self.load_data()
        self.prepare_environment()

    def load_data(self):
        """
        Loads training and testing datasets from disk based on the configured data path.

        Raises:
            FileNotFoundError: If either the training or testing dataset files do not exist.

        Side Effects:
            - Sets self.problems with 'train' and 'test' datasets loaded from disk.
            - Logs a message indicating successful data loading.
        """
        train_data_path: Path = self.config.datapath / "train"
        test_data_path: Path = self.config.datapath / "test"
        if not train_data_path.exists() or not test_data_path.exists():
            raise FileNotFoundError("You need to setup the dataset files.")

        self.problems = {
            "train": Dataset.load_from_disk(train_data_path),
            "test": Dataset.load_from_disk(test_data_path),
        }
        self.logger.info("Data loaded successfully.")

    def prepare_environment(self):
        """
        Prepares the environment by formatting and organizing the dataset for training and testing.

        This method processes the problems for each data split ("train" and "test"), formats the prompts,
        and constructs a dataset containing the query, original question, and ground truth answer.
        The resulting datasets are stored in the `self.dataset` dictionary under their respective splits.

        Logs the number of samples prepared for each split.

        Returns:
            None
        """
        self.logger.info("Preparing the environment with dataset...")

        self.dataset = {}

        for split in ["train", "test"]:
            problems = self.problems[split]
            split_dataset = []
            for problem in problems:
                prompt = self.format_prompt(problem["question"])

                split_dataset.append(
                    {
                        "query": prompt,
                        "question": problem["question"],
                        "ground_truth": problem["answer"],
                    }
                )

            self.logger.info(
                f"Prepared dataset with {len(split_dataset)} samples for split {split}."
            )
            self.dataset[split] = Dataset.from_list(split_dataset)

    @property
    def train_dataset(self) -> Dataset:
        return self.dataset["train"]

    @property
    def test_dataset(self) -> Dataset:
        return self.dataset["test"]

    @property
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a DataLoader for the training dataset.

        The DataLoader uses a custom data collator for batching, shuffles the data,
        and utilizes multiple worker processes for loading data efficiently.

        Returns:
            torch.utils.data.DataLoader: DataLoader instance for the training dataset.
        """
        data_collator = CustomDataCollator(tokenizer=self.tokenizer)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.training_params.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4,
        )
        return train_dataloader

    def epoch_dataloader_sub_sampled(self) -> torch.utils.data.DataLoader:
        """
        Creates a DataLoader for a randomly sub-sampled subset of the training dataset for the current epoch.

        The method randomly shuffles the training dataset and selects up to `max_samples_per_epoch` samples.
        It then constructs a DataLoader using the selected subset, applying a custom data collator and batching.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the sub-sampled training dataset for the epoch.
        """
        data_collator = CustomDataCollator(tokenizer=self.tokenizer)

        num_samples = len(self.train_dataset)
        max_samples = self.config.training_params.max_samples_per_epoch

        shuffled_indices = np.random.permutation(num_samples)

        subset_indices = shuffled_indices[: min(max_samples, num_samples)]

        epoch_dataset = self.train_dataset.select(subset_indices)

        train_dataloader = torch.utils.data.DataLoader(
            epoch_dataset,
            batch_size=self.config.training_params.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4,
        )
        return train_dataloader

    @property
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a DataLoader for the test dataset.

        The DataLoader uses a custom data collator for batching and preprocessing,
        and the batch size is specified by the training parameters in the configuration.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the test dataset.
        """
        data_collator = CustomDataCollator(tokenizer=self.tokenizer)

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.training_params.batch_size,
            collate_fn=data_collator,
        )
        return test_dataloader

    def format_prompt(self, question: str) -> str:

        return f"Solve this step by step:\n\nQuestion: {question}\n\nAnswer:"

    def extract_numerical_answer(self, text: str) -> Optional[float]:
        match = re.search(r"####\s*([0-9,.-]+)", text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def compute_reward(
        self, question: str, generated_answer: str, ground_truth: str
    ) -> float:
        """
        Computes a reward score based on the comparison between a generated answer and the ground truth for a given question.

        The reward is determined as follows:
            - Returns -1.0 if the generated answer does not contain a valid numerical value.
            - Returns 0.0 if the ground truth does not contain a valid numerical value (with a warning logged).
            - Returns 1.0 if the predicted and true answers match within a tolerance of 1e-6.
            - Returns 0.5 if the relative error between predicted and true answers is less than 0.1.
            - Returns 0.2 if the relative error is less than 0.5.
            - Returns -0.5 otherwise.

        Args:
            question (str): The question being answered.
            generated_answer (str): The answer generated by the model.
            ground_truth (str): The correct answer.

        Returns:
            float: The computed reward score.
        """
        predicted_answer = self.extract_numerical_answer(generated_answer)
        true_answer = self.extract_numerical_answer(ground_truth)

        if predicted_answer is None:
            return -1.0

        if true_answer is None:
            self.logger.warning(f"Could not extract true answer from: {ground_truth}")
            return 0.0

        if abs(predicted_answer - true_answer) < 1e-6:
            return 1.0
        else:
            relative_error = abs(predicted_answer - true_answer) / max(
                abs(true_answer), 1
            )
            if relative_error < 0.1:
                return 0.5
            elif relative_error < 0.5:
                return 0.2
            else:
                return -0.5
