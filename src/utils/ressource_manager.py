import logging
from contextlib import contextmanager
from typing import Dict

import torch

from utils.exceptions import ResourceError


class TrainingResourceManager:

    def __init__(self, logger: logging.Logger, max_gpu_memory_fraction: float = 0.85):
        """
        Initializes the resource manager with a logger, maximum GPU memory fraction, and device selection.

        Args:
            logger (logging.Logger): Logger instance for logging messages.
            max_gpu_memory_fraction (float, optional): Maximum fraction of GPU memory to use. Defaults to 0.85.

        Attributes:
            logger (logging.Logger): Logger instance.
            max_gpu_memory (float): Maximum fraction of GPU memory to use.
            device (torch.device): Computation device ('cuda' if available, otherwise 'cpu').
        """
        self.logger = logger
        self.max_gpu_memory = max_gpu_memory_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_gpu_memory(self) -> Dict[str, float]:
        """
        Checks and returns the current GPU memory usage statistics.

        Returns:
            Dict[str, float]: A dictionary containing the following keys:
                - "used": Amount of GPU memory currently allocated by tensors (in GB).
                - "cached": Amount of GPU memory currently reserved by the caching allocator (in GB).
                - "allocated": Maximum GPU memory allocated by tensors since the beginning of this program (in GB).
            If CUDA is not available, all values are set to 0.0.
        """
        if not torch.cuda.is_available():
            return {"used": 0.0, "cached": 0.0, "allocated": 0.0}

        return {
            "used": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def cleanup_gpu_memory(self, force: bool = False) -> bool:
        """
        Cleans up GPU memory if usage exceeds a specified threshold or if forced.

        This method checks the current GPU memory usage and compares it to the maximum allowed usage fraction (`self.max_gpu_memory`).
        If the usage exceeds the threshold or if `force` is set to True, it empties the CUDA cache and synchronizes the device.

        Args:
            force (bool, optional): If True, forces GPU memory cleanup regardless of usage. Defaults to False.

        Returns:
            bool: True if GPU memory was cleaned up, False otherwise.
        """
        if not torch.cuda.is_available():
            return False

        memory_stats = self.check_gpu_memory()
        usage_fraction = (
            memory_stats["used"] / memory_stats["allocated"]
            if memory_stats["allocated"] > 0
            else 0
        )

        if usage_fraction > self.max_gpu_memory or force:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        return False

    @contextmanager
    def managed_computation(self, operation: str):
        """
        Context manager for handling resource management during computations.

        Args:
            operation (str): Description or name of the operation being performed.

        Yields:
            None

        Raises:
            ResourceError: If a CUDA out-of-memory error occurs during the operation.
            Exception: Propagates any other exceptions that occur during the operation.

        Side Effects:
            - Cleans up GPU memory if an out-of-memory error is encountered.
            - Logs errors related to out-of-memory and other computation failures.
        """
        try:
            yield
        except torch.cuda.OutOfMemoryError as e:
            self.cleanup_gpu_memory(force=True)
            self.logger.error(f"oom_error: operation {operation}, error {str(e)}")
            raise ResourceError(f"Out of memory during {operation}: {e}") from e
        except Exception as e:
            self.logger.error(
                f"computation_failed: operation {operation}, error {str(e)}"
            )
            raise
        finally:
            pass