import datetime
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_validator


def get_gpu_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NVIDIA GPU not found or nvidia-smi is not installed"


class SystemParams(BaseModel):
    python_version: str = Field(
        default_factory=lambda: sys.version.split()[0], description="Python Version"
    )
    pytorch_version: str = Field(
        default_factory=lambda: torch.__version__, description="PyTorch Version"
    )
    cuda_version: str = Field(
        default_factory=lambda: (
            torch.version.cuda if torch.cuda.is_available() else "N/A"
        ),
        description="CUDA Version",
    )
    platform: str = Field(
        default_factory=lambda: platform.platform(), description="Operating System"
    )
    cpu_info: str = Field(
        default_factory=lambda: platform.processor(), description="CPU Information"
    )
    gpu_info: str = Field(default_factory=get_gpu_info, description="GPU Information")


class BaseParams(BaseModel):
    output_dir: str = Field(
        default="./moe_outputs", description="Directory to save model outputs"
    )
    data_dir: str = Field(default="./data", description="Directory to save data")
    project_name: str = Field(
        default="moe-usage", description="MLflow project name"
    )
    seed: int = Field(default=42, description="Random seed")


class ModelParams(BaseModel):
    pretrained_model_name: str = Field(
        default="allenai/OLMoE-1B-7B-0924",
        description="Pretrained model name or path",
    )
    # Qwen/Qwen1.5-MoE-A2.7B, allenai/OLMoE-1B-7B-0924 , ibm-granite/granite-3.1-3b-a800m-instruct
    dataset_name: str = Field(default="GSM8K", description="Name of the dataset")

    @field_validator("pretrained_model_name")
    @classmethod
    def validate_model_name(cls, value):
        if not value:
            raise ValueError("Model name must not be empty")
        return value

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value):
        if not value:
            raise ValueError("Dataset name must not be empty")
        return value


class LoggingParams(BaseModel):
    use_mlflow: bool = Field(
        default=True, description="Whether to use MLflow for tracking"
    )
    log_dir: str = Field(default="./logs", description="Directory to save logs")
    log_level: str = Field(default="info", description="Logging level")
    experiment_name: str = Field(
        default="moe_experiment", description="MLflow experiment name"
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking URI"
    )
    log_interval: int = Field(default=100, description="Logging interval (in steps)")
    save_interval: int = Field(
        default=1000, description="Model save interval (in steps)"
    )


class TrainingParams(BaseModel):
    algorithm: str = Field(default="ppo", description="Algorithm to use for training")
    learning_rate: float = Field(
        default=1e-5, description="Learning rate for the optimizer"
    )
    batch_size: int = Field(default=16, description="Batch size for training")
    mini_batch_size: int = Field(default=4, description="Mini batch size for training")
    gradient_accumulation_steps: int = Field(
        default=4, description="Gradient accumulation steps"
    )
    cliprange: float = Field(default=0.2, description="Clip range for PPO")
    cliprange_value: float = Field(default=0.2, description="Clip range value for PPO")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")
    max_length: int = Field(default=512, description="Maximum sequence length")
    min_length: int = Field(
        default=10, description="Minimum sequence length for generation"
    )
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    top_p: float = Field(default=0.9, description="Top-p sampling probability")
    top_k: int = Field(default=50, description="Top-k sampling value")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    max_samples_per_epoch: int = Field(
        default=500, description="Maximum number of samples per epoch"
    )
    clip_epsilon: float = Field(default=1e-6, description="Epsilon value for clipping")
    entropy_coeff: float = Field(
        default=0.01, description="Entropy coefficient for exploration"
    )
    value_coeff: float = Field(
        default=0.5, description="Value coefficient for critic loss"
    )
    max_new_tokens: int = Field(
        default=8, description="Maximum number of new tokens to generate"
    )
    use_8bit_optimizer: bool = Field(
        default=True, description="Whether to use 8-bit optimizer"
    )
    num_epochs: int = Field(default=50, description="Number of training epochs")
    use_mixed_precision: bool = Field(
        default=True, description="Whether to use mixed precision training"
    )
    gamma: float = Field(default=0.99, description="Discount factor for rewards")
    gae_lambda: float = Field(
        default=0.95, description="GAE lambda for advantage calculation"
    )
    ppo_epochs: int = Field(default=4, description="Number of PPO epochs")
    use_amp: bool = Field(
        default=True, description="Whether to use mixed precision training"
    )
    amp_init_scale: float = Field(
        default=2.0**16, description="Initial scale for AMP loss scaling"
    )
    amp_growth_factor: float = Field(
        default=2.0, description="Growth factor for AMP loss scaling"
    )
    amp_backoff_factor: float = Field(
        default=0.5, description="Backoff factor for AMP loss scaling"
    )
    amp_growth_interval: int = Field(
        default=2000, description="Interval for AMP growth"
    )
    amp_max_overflow_retries: int = Field(
        default=3, description="Maximum number of overflow retries for AMP"
    )

    kl_penalty_init_coef: float = Field(
        default=0.6, description="Initial KL penalty coefficient"
    )

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value):
        if not 1e-9 < value < 1.0:
            raise ValueError(f"Invalid learning_rate: {value}")
        return value

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value):
        if value <= 0 or value > 1024:
            raise ValueError(f"Invalid batch_size: {value}")
        return value

    @field_validator("mini_batch_size")
    @classmethod
    def validate_mini_batch_size(cls, value):
        if value <= 0 or value > 1024:
            raise ValueError(f"Invalid mini_batch_size: {value}")
        return value

    @field_validator("num_epochs")
    @classmethod
    def validate_num_epochs(cls, value):
        if value <= 0:
            raise ValueError(f"Invalid num_epochs: {value}")
        return value


class MoERLConfig(BaseModel):
    pipeline_stage: str = Field(
        default="full_pipeline", description="The stage of the pipeline"
    )
    inference_prompt: Optional[str] = None
    system_params: SystemParams = Field(default_factory=SystemParams)
    base_params: BaseParams = Field(default_factory=BaseParams)
    model_params: ModelParams = Field(default_factory=ModelParams)
    logging_params: LoggingParams = Field(default_factory=LoggingParams)
    training_params: TrainingParams = Field(default_factory=TrainingParams)

    checkpoint_path: Optional[Path] = None

    _checkpoint_path_generated: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        """
        If a checkpoint_path wasn't provided, this generates a unique,
        timestamped path after the model is initialized.
        """
        if self.checkpoint_path is None:
            run_name = (
                self.model_params.pretrained_model_name.replace("/", "_")
                + "_"
                + self.model_params.dataset_name.lower()
                + "_"
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            self.checkpoint_path = (
                Path(self.base_params.output_dir) / "checkpoints" / run_name
            )
            self._checkpoint_path_generated = True

    @property
    def algorithm(self):
        return self.training_params.algorithm

    @property
    def experiment_name(self):
        return self.logging_params.experiment_name

    @property
    def project_name(self):
        return self.base_params.project_name

    @property
    def seed(self):
        return self.base_params.seed

    @property
    def pretrained_model_name(self):
        return self.model_params.pretrained_model_name

    @property
    def dataset_name(self):
        return self.model_params.dataset_name

    @property
    def log_dir(self):
        return self.logging_params.log_dir

    @property
    def mlflow_tracking_uri(self):
        return self.logging_params.mlflow_tracking_uri

    @property
    def output_dir(self):
        return self.base_params.output_dir

    @computed_field
    @property
    def datapath(self) -> Path:
        dataset_dir = (
            Path(self.base_params.data_dir) / "raw" / self.dataset_name.lower()
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def set_checkpoint_path(self) -> Path:
        return (
            Path(self.base_params.output_dir)
            / "checkpoints"
            / (
                self.pretrained_model_name.replace("/", "_")
                + "_"
                + self.dataset_name.lower()
                + "_"
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        )

    @property
    def use_mlflow(self):
        return self.logging_params.use_mlflow

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    @classmethod
    def from_argparse(cls, cli_dict: Dict[str, Any]) -> "MoERLConfig":
        def get_sub_config(model_class):
            return {
                k: v
                for k, v in cli_dict.items()
                if k in model_class.model_fields and v is not None
            }

        return cls(
            pipeline_stage=cli_dict.get("pipeline_stage"),
            inference_prompt=cli_dict.get("inference_prompt"),
            checkpoint_path=cli_dict.get("checkpoint_path"),
            system_params=SystemParams(),
            base_params=BaseParams(**get_sub_config(BaseParams)),
            model_params=ModelParams(**get_sub_config(ModelParams)),
            logging_params=LoggingParams(**get_sub_config(LoggingParams)),
            training_params=TrainingParams(**get_sub_config(TrainingParams)),
        )

    def save_to_yaml(self, file_path: str):
        config_dict = {
            "pipeline_stage": self.pipeline_stage,
            "inference_prompt": self.inference_prompt,
            "system_params": self.system_params.model_dump(),
            "base_params": self.base_params.model_dump(),
            "model_params": self.model_params.model_dump(),
            "logging_params": self.logging_params.model_dump(),
            "training_params": self.training_params.model_dump(),
        }
        with open(file_path, "w") as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)

    def create_directories(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        Path(self.base_params.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_params.log_dir).mkdir(parents=True, exist_ok=True)
        self.datapath.mkdir(parents=True, exist_ok=True)
