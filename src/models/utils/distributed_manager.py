import os
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedManager:
    """
    Manages the setup and utilities for multi-GPU distributed training using DDP.
    This version follows the standard, efficient pattern where each process
    works on its own data and gradients are synchronized.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.is_distributed = dist.is_available() and torch.cuda.device_count() > 1

        if not self.is_distributed:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info("Running in single-GPU or CPU mode.")
            return

        # Initialize based on environment variables set by torchrun
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        self._setup_distributed()
        self.logger.info(f"Distributed setup complete - World size: {self.world_size}, Rank: {self.rank}")

    def _setup_distributed(self, backend: str = "nccl"):
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        torch.cuda.set_device(self.local_rank)

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        if self.is_distributed:
            return DDP(model, device_ids=[self.local_rank])
        return model

    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Averages metrics across all processes to ensure consistent logging.
        Only the rank 0 process will have the final averaged metrics.
        """
        if not self.is_distributed:
            return metrics

        metric_tensor = torch.tensor(list(metrics.values()), device=self.device)
        
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        
        averaged_tensor = metric_tensor / self.world_size
        
        return {key: averaged_tensor[i].item() for i, key in enumerate(metrics.keys())}
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0