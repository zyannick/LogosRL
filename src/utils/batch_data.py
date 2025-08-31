from typing import List

import torch
from pydantic import BaseModel, ValidationInfo, field_validator


class BatchData(BaseModel):
    input_ids: torch.Tensor
    queries: List[str]
    questions: List[str]
    ground_truths: List[str]

    class Config:
        arbitrary_types_allowed = True

    @field_validator("input_ids")
    def validate_tensor(cls, v):
        if not isinstance(v, torch.Tensor) or v.numel() == 0:
            raise ValueError("input_ids must be non-empty tensor")
        if len(v.shape) != 2:
            raise ValueError("input_ids must be 2D tensor (batch_size, seq_length)")
        return v

    @field_validator("queries", "questions", "ground_truths")
    def validate_lists_length(cls, v: List, info: ValidationInfo) -> List:

        if "input_ids" in info.data:
            expected_len = info.data["input_ids"].shape[0]
            if len(v) != expected_len:
                raise ValueError(
                    f"List length {len(v)} doesn't match batch size {expected_len}"
                )
        return v
