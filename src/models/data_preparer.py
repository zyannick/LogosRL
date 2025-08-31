from datasets import Dataset, load_dataset

from utils.configurations import MoERLConfig


class DataPreparer:
    def __init__(self, config: MoERLConfig):
        self.config = config

    def load_dataset_and_save(self) -> None:
        """
        Loads the training and test datasets from disk if available; otherwise, downloads them using the specified dataset name,
        saves them to disk, and loads them into memory.

        The method checks if the dataset files exist at the configured paths. If they do, it loads them using `Dataset.load_from_disk`.
        If not, it downloads the datasets using `load_dataset`, saves them to disk, and loads them into memory.

        Raises:
            Exception: If loading or saving the datasets fails.
        """
        train_data_path = self.config.datapath / "train"
        test_data_path = self.config.datapath / "test"

        if train_data_path.exists() and test_data_path.exists():
            try:
                self.problems = {
                    "train": Dataset.load_from_disk(train_data_path),
                    "test": Dataset.load_from_disk(test_data_path),
                }
                return
            except Exception:
                raise

        try:
            dataset_name = self.config.model_params.dataset_name
            # trunk-ignore(bandit/B615)
            train_dataset = load_dataset(dataset_name, "main", split="train")
            # trunk-ignore(bandit/B615)
            test_dataset = load_dataset(dataset_name, "main", split="test")
            train_dataset.save_to_disk(train_data_path)
            test_dataset.save_to_disk(test_data_path)
        except Exception:
            raise
