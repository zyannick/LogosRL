import asyncio
import gc
import logging
import os
import random
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import mlflow
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models.data_preparer import DataPreparer
from models.environnement import GSM8KEnvironment
from models.expert_usage_tracker import (
    MoEModelWithTracking,
    PatchedAutoModelForCausalLMWithValueHead,
)
from models.trainer import MixtureOfExpertsTrainer
from models.utils.distributed_manager import DistributedManager
from utils.configurations import MoERLConfig
from utils.exceptions import (
    ModelLoadError,
    PipelineError,
    ResourceError,
    StateTransitionError,
)
from utils.moe_logger import MoELogger

T = TypeVar("T")


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PipelineStage(Enum):
    PREPARE_DATA = "prepare_data"
    SETUP = "setup"
    TRAIN = "train"
    EVALUATE = "evaluate"
    INFER = "infer"
    DEPLOY = "deploy"
    FULL_PIPELINE = "full_pipeline"

    @classmethod
    def str_to_enum(cls, stage_str: str) -> "PipelineStage":
        for stage in cls:
            if stage.value == stage_str:
                return stage
        raise ValueError(f"Unknown pipeline stage: {stage_str}")


class PipelineState(Enum):
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    DATA_PREPARING = auto()
    READY = auto()
    TRAINING = auto()
    EVALUATING = auto()
    INFERRING = auto()
    ERROR = auto()
    CLEANUP = auto()


@dataclass
class PipelineResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class InferenceRequest:
    prompts: List[str]
    max_length: int = 32
    min_length: int = 8
    temperature: float = 0.7
    batch_size: int = 8

    def __post_init__(self):
        if not self.prompts:
            raise ValueError("Prompts cannot be empty")
        if any(len(p.strip()) == 0 for p in self.prompts):
            raise ValueError("Empty prompts are not allowed")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")


class PipelineResourceManager:

    def __init__(self, logger: logging.Logger):
        """
        Initializes the object with a logger and prepares an empty list for resources.

        Args:
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.logger = logger
        self._resources: List[Any] = []

    def register(self, resource: Any):
        """
        Registers a resource by appending it to the internal resources list.

        Args:
            resource (Any): The resource to be registered.
        """
        self._resources.append(resource)

    def cleanup(self):
        """
        Cleans up resources managed by the pipeline.

        This method iterates over all resources in `self._resources` and attempts to:
        - Move resources to CPU if they have a `cpu()` method.
        - Call their `cleanup()` method if available.
        - Delete the resource reference.

        Any exceptions during cleanup are logged.

        After processing all resources:
        - Clears the resource list.
        - Forces garbage collection.
        - Empties the CUDA cache if using PyTorch with CUDA.

        Logs the start and completion of the cleanup process.
        """
        self.logger.info("Starting resource cleanup")

        for resource in self._resources:
            try:
                if hasattr(resource, "cpu"):
                    resource.cpu()
                if hasattr(resource, "cleanup"):
                    resource.cleanup()
                del resource
            except Exception as e:
                self.logger.error(f"Error cleaning up resource: {e}")
                self.logger.debug(traceback.format_exc())

        self._resources.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Resource cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def known_gate_patterns(model_name: str) -> str:
    if model_name in [
        "Qwen/Qwen1.5-MoE-A2.7B",
        "Qwen/Qwen-MoE-14B",
        "allenai/OLMoE-1B-7B-0924",
    ]:
        return ".mlp.gate"
    elif model_name == "ibm-granite/granite-3.1-3b-a800m-instruct":
        return ".block_sparse_moe.router.layer"
    else:
        return ".gate"


class ModelManager:

    def __init__(
        self,
        config: MoERLConfig,
        logger: logging.Logger,
        dist_manager: DistributedManager,
    ):
        """
        Initializes the pipeline with the given configuration and logger.

        Args:
            config (MoERLConfig): Configuration object for the MoE RL pipeline.
            logger (logging.Logger): Logger instance for logging pipeline events.

        Attributes:
            config (MoERLConfig): Stores the pipeline configuration.
            logger (logging.Logger): Stores the logger instance.
            device (torch.device): Device to run models on ('cuda' if available, else 'cpu').
            tokenizer (Optional[AutoTokenizer]): Tokenizer for processing input data.
            policy_model (Optional[MoEModelWithTracking]): Policy model with tracking capabilities.
            reference_model (Optional[PatchedAutoModelForCausalLMWithValueHead]): Reference model for causal LM with value head.
            _pipeline_resource_manager (PipelineResourceManager): Manages pipeline resources.
        """
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.float16_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer: Optional[AutoTokenizer] = None
        self.policy_model: Optional[MoEModelWithTracking] = None
        self.reference_model: Optional[PatchedAutoModelForCausalLMWithValueHead] = None
        self._pipeline_resource_manager = PipelineResourceManager(logger)
        self.dist_manager = dist_manager
        self.device = self.dist_manager.device

    async def initialize(self) -> None:
        """
        Asynchronously initializes the pipeline by loading the tokenizer, base model, and wrapping models.
        Registers models with the pipeline resource manager, prepares checkpoint directory, and logs progress.
        Cleans up resources and raises ModelLoadError if initialization fails.

        Raises:
            ModelLoadError: If any error occurs during model initialization.
        """
        try:
            if self.dist_manager.is_main_process:
                self.logger.info("Main process is downloading models and tokenizer...")
                await self._load_tokenizer()
                await self._load_quantized_base_model()

            if self.dist_manager.is_distributed:
                torch.distributed.barrier()

            self.logger.info(
                f"Rank {self.dist_manager.rank}: Loading models from cache."
            )
            self.tokenizer = await self._load_tokenizer()
            base_model = await self._load_quantized_base_model()
            self._pipeline_resource_manager.register(base_model)

            self.logger.info(f"Rank {self.dist_manager.rank}: Wrapping models.")
            self.policy_model = self._create_peft_model(base_model, "policy_model")
            self.policy_model = MoEModelWithTracking(
                model=self.policy_model,
                logger=self.logger,
                gate_name_pattern=known_gate_patterns(
                    self.config.pretrained_model_name
                ),
            )

            self.reference_model = (
                PatchedAutoModelForCausalLMWithValueHead.from_pretrained(base_model)
            )

            self.policy_model.to(self.device)
            self.reference_model.to(self.device)
            self.reference_model.eval()

            self._pipeline_resource_manager.register(self.policy_model)
            self._pipeline_resource_manager.register(self.reference_model)

            if self.dist_manager.is_main_process:
                self.config.checkpoint_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"Rank {self.dist_manager.rank}: Models initialized successfully."
            )

        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            self.logger.debug(traceback.format_exc())
            self._pipeline_resource_manager.cleanup()
            raise ModelLoadError(f"Failed to initialize models: {e}") from e

    async def load_trained_model_from_checkpoint(
        self, checkpoint_path: str
    ) -> MoEModelWithTracking:
        """
        Asynchronously loads a trained MoEModelWithTracking model from a checkpoint file.

        Args:
            checkpoint_path (str): The file path to the model checkpoint.

        Returns:
            MoEModelWithTracking: The loaded and registered policy model.

        Raises:
            ModelLoadError: If the model fails to load from the checkpoint.
        """
        try:
            # trunk-ignore(bandit/B614)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.policy_model = MoEModelWithTracking.load_state_dict(state_dict)
            self._pipeline_resource_manager.register(self.policy_model)
            return self.policy_model
        except Exception as e:
            raise ModelLoadError(f"Failed to load trained model: {e}") from e

    async def load_best_checkpoint_for_evaluation(self) -> None:
        """
        Asynchronously loads the best checkpoint for evaluation by loading LoRA adapter weights into the policy model.

        Checks if the checkpoint path is specified and exists. If valid, loads the adapter weights from the checkpoint
        into the policy model and logs the process. Raises a ModelLoadError if the checkpoint path is invalid or if
        loading fails.

        Raises:
            ModelLoadError: If the checkpoint path is not found, not specified, or if loading the adapter weights fails.
        """

        try:
            checkpoint_path = self.config.checkpoint_path
            if not checkpoint_path or not Path(checkpoint_path).exists():
                raise ModelLoadError(
                    f"Checkpoint path not found or specified: {checkpoint_path}"
                )

            self.logger.info(
                f"Loading LoRA adapter weights from checkpoint: {checkpoint_path}"
            )

            self.policy_model.load_adapter(checkpoint_path, "default")

            self.logger.info(
                "Successfully loaded adapter weights into the policy model."
            )

        except Exception as e:
            self.logger.error(f"Failed to load trained model checkpoint: {e}")
            self.logger.debug(traceback.format_exc())
            raise ModelLoadError(f"Failed to load trained model: {e}") from e

    async def _load_tokenizer(self) -> AutoTokenizer:
        """
        Asynchronously loads a tokenizer using the pretrained model name specified in the configuration.

        Returns:
            AutoTokenizer: The loaded tokenizer instance with left-side padding. If the pad token is not set,
            it assigns the end-of-sequence (EOS) token as the pad token.

        Raises:
            ModelLoadError: If the tokenizer fails to load for any reason.
        """
        try:
            # trunk-ignore(bandit/B615)
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.pretrained_model_name, padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            raise ModelLoadError(f"Failed to load tokenizer: {e}") from e

    async def _load_quantized_base_model(self) -> Any:
        """
        Asynchronously loads a quantized base causal language model using 4-bit quantization.

        This method configures the quantization settings with BitsAndBytesConfig, loads the model
        from the pretrained checkpoint specified in the configuration, and logs GPU memory usage
        before and after model loading if CUDA is available.

        Returns:
            Any: The loaded quantized base model.

        Raises:
            ResourceError: If GPU runs out of memory during model loading.
            ModelLoadError: If any other error occurs during model loading.
        """
        try:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_cached = torch.cuda.memory_reserved()
                self.logger.info(
                    f"GPU memory before model loading: allocated={memory_allocated}, cached={memory_cached}"
                )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.float16_type,
                bnb_4bit_use_double_quant=True,
            )

            # trunk-ignore(bandit/B615)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrained_model_name,
                quantization_config=quantization_config,
                dtype=self.float16_type,
                device_map=None,
            )

            # print(base_model)

            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                self.logger.info(
                    f"GPU memory after model loading: allocated={memory_allocated}"
                )

            return base_model

        except torch.cuda.OutOfMemoryError as e:
            raise ResourceError(f"GPU out of memory during model loading: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"Failed to load base model: {e}") from e

    def _create_peft_model(self, base_model: Any, model_name: str) -> Any:
        """
        Wraps a base language model with a value head and applies LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

        Args:
            base_model (Any): The base model to be wrapped and fine-tuned.
            model_name (str): The name of the model, used for logging.

        Returns:
            Any: The PEFT (Parameter-Efficient Fine-Tuning) model with value head and LoRA applied.

        Steps:
            1. Logs the wrapping process.
            2. Loads the base model with a value head.
            3. Prepares the model for k-bit training.
            4. Configures LoRA with specified parameters and target modules.
            5. Applies LoRA to the model.
            6. Sets the generation configuration from the base model.
            7. Enables gradient checkpointing for memory efficiency.
            8. Prints the trainable parameters for verification.
        """
        self.logger.info(f"Wrapping {model_name} with ValueHead and LoRA")

        model_with_v_head = PatchedAutoModelForCausalLMWithValueHead.from_pretrained(
            base_model
        )
        model_with_v_head = prepare_model_for_kbit_training(model_with_v_head)

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(model_with_v_head, lora_config)
        peft_model.generation_config = base_model.generation_config

        peft_model.gradient_checkpointing_enable()

        peft_model.print_trainable_parameters()

        return peft_model

    def cleanup(self):
        """
        Releases resources and resets key attributes to None.

        This method calls the cleanup function of the pipeline resource manager
        to release any allocated resources. It then sets the tokenizer, policy_model,
        and reference_model attributes to None to ensure proper cleanup and prevent
        further use of these objects.
        """
        self._pipeline_resource_manager.cleanup()
        self.tokenizer = None
        self.policy_model = None
        self.reference_model = None


class DataManager:
    def __init__(self, config: MoERLConfig, logger: logging.Logger):
        """
        Initializes the pipeline with the given configuration and logger.

        Args:
            config (MoERLConfig): Configuration object for the MoE RL pipeline.
            logger (logging.Logger): Logger instance for logging pipeline events.
        """
        self.config = config
        self.logger = logger

    async def prepare_dataset(self) -> PipelineResult:
        """
        Asynchronously prepares the dataset by invoking the DataPreparer's loading and saving routine in a separate thread.

        Logs the start and completion of the dataset preparation process. If an exception occurs during preparation,
        logs the error and stack trace, and returns a PipelineResult indicating failure.

        Returns:
            PipelineResult: An object indicating the success or failure of the dataset preparation.
        """
        try:
            self.logger.info("Starting dataset preparation")
            data_preparer = DataPreparer(self.config)
            await asyncio.to_thread(data_preparer.load_dataset_and_save)

            self.logger.info("Dataset preparation completed")
            return PipelineResult(success=True)

        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {e}")
            self.logger.debug(traceback.format_exc())
            return PipelineResult(success=False, error=str(e))


class TrainingManager:
    def __init__(
        self,
        model_manager: ModelManager,
        config: MoERLConfig,
        distributed_manager: DistributedManager,
        logger: logging.Logger,
    ):

        self.model_manager = model_manager
        self.config = config
        self.logger = logger
        self.distributed_manager = distributed_manager
        self.moe_trainer: Optional[MixtureOfExpertsTrainer] = None
        self.environment: Optional[GSM8KEnvironment] = None
        self.mlflow_client = mlflow.MlflowClient()

    def initialize(self):
        """
        Initializes the training pipeline components, including the GSM8K environment and PPO MoE trainer.

        Sets up the environment and trainer using the provided configuration, tokenizer, models, logger, and MLflow client.
        Logs the successful initialization of training components. In case of failure, logs the error and traceback,
        then raises a PipelineError with details.

        Raises:
            PipelineError: If initialization of training components fails.
        """
        try:
            self.environment = GSM8KEnvironment(
                self.config, self.model_manager.tokenizer, self.logger
            )
            self.moe_trainer = MixtureOfExpertsTrainer(
                config=self.config,
                tokenizer=self.model_manager.tokenizer,
                policy_model=self.model_manager.policy_model,
                reference_model=self.model_manager.reference_model,
                environment=self.environment,
                logger=self.logger,
                mlflow_client=self.mlflow_client,
                distributed_manager=self.distributed_manager,
            )
            self.logger.info("Training components initialized")

        except Exception as e:
            self.logger.error(f"Training initialization failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise PipelineError(f"Failed to initialize training: {e}") from e

    def _init_mlflow(self):
        """
        Initializes MLflow logging for the current configuration.

        This method logs all parameters from the configuration to MLflow. It handles nested dictionaries
        by recursively logging each parameter with a dot-separated key. If logging a parameter fails,
        an error message is printed indicating which parameter could not be logged.

        Returns:
            None
        """

        def recursive_log_params(config, prefix=""):
            for key, value in config.items():
                if isinstance(value, dict):
                    recursive_log_params(value, prefix + key + ".")
                else:
                    mlflow.log_param(prefix + key, value)

        for key, value in self.config.model_dump().items():
            try:
                if isinstance(value, dict):
                    recursive_log_params(value, key + ".")
                else:
                    mlflow.log_param(key, value)
            except Exception as err:
                print("Couldn't log {} because of {}".format(key, err))

    async def train(self) -> PipelineResult:
        """
        Asynchronously trains the model using PPO and logs the process to MLflow.

        Initializes MLflow experiment and run, sets up anomaly detection for PyTorch autograd,
        and starts the training process in a separate thread. Handles exceptions by logging errors
        and returning a PipelineResult indicating success or failure.

        Returns:
            PipelineResult: An object indicating whether training was successful or if an error occurred.

        Raises:
            StateTransitionError: If the PPO trainer is not initialized.
        """
        try:
            if not self.moe_trainer:
                raise StateTransitionError("Training not initialized")

            mlflow.set_experiment(self.config.experiment_name)
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            experiment_id = experiment.experiment_id
            torch.autograd.set_detect_anomaly(True)

            with mlflow.start_run(experiment_id=experiment_id) as run:
                run_id = run.info.run_id

                self._init_mlflow()
                self.logger.info("Starting model training")
                await asyncio.to_thread(self.moe_trainer.train, run_id)

                self.logger.info("Training completed successfully")
                return PipelineResult(success=True)

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.debug(traceback.format_exc())
            return PipelineResult(success=False, error=str(e))

    async def evaluate(self) -> PipelineResult:
        """
        Asynchronously evaluates the current policy model using PPO trainer.

        Loads the best checkpoint for evaluation, sets the model to evaluation mode,
        and performs evaluation in a separate thread. Returns the evaluation metrics
        wrapped in a PipelineResult object. Handles errors by logging and returning
        a failed PipelineResult.

        Returns:
            PipelineResult: Object containing success status, evaluation metrics (if successful),
                            or error message (if failed).

        Raises:
            StateTransitionError: If the pipeline is not properly initialized for evaluation.
        """
        try:
            if not self.moe_trainer or not self.model_manager.policy_model:
                raise StateTransitionError("Pipeline not initialized for evaluation")

            self.logger.info(
                "Preparing for model evaluation by loading best checkpoint."
            )

            await self.model_manager.load_best_checkpoint_for_evaluation()

            self.model_manager.policy_model.eval()
            self.moe_trainer.policy_model = self.model_manager.policy_model

            self.logger.info("Model is ready. Starting evaluation.")
            metrics = await asyncio.to_thread(self.moe_trainer.evaluate)

            self.logger.info(f"Evaluation completed: metrics = {metrics}")
            return PipelineResult(success=True, data=metrics, metrics=metrics)

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            self.logger.debug(traceback.format_exc())
            return PipelineResult(success=False, error=str(e))


class InferenceEngine:

    def __init__(
        self,
        model_manager: ModelManager,
        logger: logging.Logger,
    ):
        self.model_manager = model_manager
        self.logger = logger
        self.is_ready = False

    def _sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitizes the input prompt by removing leading and trailing whitespace.
        If the sanitized prompt exceeds 2000 characters, it is truncated to 2000 characters.

        Args:
            prompt (str): The input prompt string to sanitize.

        Returns:
            str: The sanitized and possibly truncated prompt string.
        """
        sanitized = prompt.strip()
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000]
        return sanitized

    async def prepare_for_inference(self):
        """
        Asynchronously prepares the inference engine for evaluation.

        Loads the best model checkpoint for evaluation, sets the policy model to evaluation mode,
        and marks the engine as ready. If the engine is already prepared, the method returns immediately.

        Returns:
            None
        """
        if self.is_ready:
            return

        self.logger.info("Preparing inference engine by loading best checkpoint.")
        await self.model_manager.load_best_checkpoint_for_evaluation()
        self.model_manager.policy_model.eval()  # Set to evaluation mode
        self.is_ready = True
        self.logger.info("Inference engine is ready.")

    async def infer_batch(self, request: InferenceRequest) -> PipelineResult:
        """
        Performs asynchronous batch inference on a list of prompts.

        Args:
            request (InferenceRequest): An object containing prompts and batch size for inference.

        Returns:
            PipelineResult: An object containing the success status, results data, or error message.

        Workflow:
            - Prepares the pipeline for inference if not ready.
            - Sanitizes all input prompts.
            - Processes prompts in batches of size `request.batch_size`.
            - Executes batch processing in a separate thread for each batch.
            - Logs progress and errors.
            - Returns results or error information.

        Exceptions:
            Handles and logs any exceptions that occur during inference, returning an error in the result.
        """
        try:
            if not self.is_ready:
                await self.prepare_for_inference()

            sanitized_prompts = [self._sanitize_prompt(p) for p in request.prompts]

            self.logger.info(
                f"Starting batch inference: batch_size={len(sanitized_prompts)}"
            )

            results = []
            for i in range(0, len(sanitized_prompts), request.batch_size):
                batch = sanitized_prompts[i : i + request.batch_size]

                batch_results = await asyncio.to_thread(
                    self._process_batch, batch, request
                )
                results.extend(batch_results)

            self.logger.info(
                f"Batch inference completed: total_processed={len(results)}"
            )
            return PipelineResult(success=True, data=results)

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            self.logger.debug(traceback.format_exc())
            return PipelineResult(success=False, error=str(e))

    def _process_batch(
        self, batch_prompts: List[str], request: InferenceRequest
    ) -> List[str]:
        """
        Processes a batch of prompts by generating model outputs using the specified inference request parameters.

        Args:
            batch_prompts (List[str]): A list of input prompt strings to be processed.
            request (InferenceRequest): An object containing inference parameters such as max_length and temperature.

        Returns:
            List[str]: A list of decoded output strings generated by the model for each input prompt.
        """
        model = self.model_manager.policy_model
        tokenizer = self.model_manager.tokenizer
        device = self.model_manager.device

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )

        prompt_lengths = [len(x) for x in inputs["input_ids"]]
        decoded_outputs = [
            tokenizer.decode(outputs[i][prompt_lengths[i] :], skip_special_tokens=True)
            for i in range(len(outputs))
        ]

        return decoded_outputs


class PipelineStateMachine:

    def __init__(self, logger: logging.Logger):
        """
        Initializes the pipeline with a logger and sets up the initial state and valid state transitions.

        Args:
            logger (logging.Logger): Logger instance for recording pipeline events.

        Attributes:
            state (PipelineState): Current state of the pipeline, initialized to UNINITIALIZED.
            logger (logging.Logger): Logger for pipeline messages.
            _valid_transitions (dict): Mapping of pipeline states to their valid next states.
        """
        self.state = PipelineState.UNINITIALIZED
        self.logger = logger
        self._valid_transitions = {
            PipelineState.UNINITIALIZED: [PipelineState.INITIALIZING],
            PipelineState.INITIALIZING: [
                PipelineState.DATA_PREPARING,
                PipelineState.READY,
                PipelineState.ERROR,
                PipelineState.CLEANUP,
                PipelineState.INFERRING,
            ],
            PipelineState.DATA_PREPARING: [
                PipelineState.READY,
                PipelineState.ERROR,
                PipelineState.CLEANUP,
            ],
            PipelineState.READY: [
                PipelineState.TRAINING,
                PipelineState.EVALUATING,
                PipelineState.INFERRING,
                PipelineState.CLEANUP,
            ],
            PipelineState.TRAINING: [PipelineState.READY, PipelineState.ERROR],
            PipelineState.EVALUATING: [PipelineState.READY, PipelineState.ERROR],
            PipelineState.INFERRING: [PipelineState.READY, PipelineState.ERROR],
            PipelineState.ERROR: [PipelineState.CLEANUP, PipelineState.INITIALIZING],
            PipelineState.CLEANUP: [PipelineState.UNINITIALIZED],
        }

    def transition_to(self, new_state: PipelineState):
        """
        Transitions the pipeline to a new state if the transition is valid.

        Args:
            new_state (PipelineState): The state to transition to.

        Raises:
            StateTransitionError: If the transition from the current state to the new state is not allowed.

        Side Effects:
            Updates the pipeline's current state and logs the state transition.
        """
        if new_state not in self._valid_transitions.get(self.state, []):
            raise StateTransitionError(
                f"Invalid transition from {self.state} to {new_state}"
            )

        old_state = self.state
        self.state = new_state
        self.logger.info(
            f"State transition: from_state={old_state.name}, to_state={new_state.name}"
        )


def fix_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    This function configures the random seed for Python's built-in random module, NumPy, and PyTorch (including CUDA if available).
    It also sets PyTorch's cuDNN backend to deterministic mode and disables benchmarking for reproducibility.

    Args:
        seed (int): The seed value to use for random number generators.

    Note:
        For full reproducibility, additional steps may be required depending on the specific use case and environment.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MoERLPipeline:
    def __init__(self, config: MoERLConfig):
        """
        Initializes the pipeline with the provided configuration.

        Args:
            config (MoERLConfig): Configuration object containing parameters for the pipeline.

        Sets up:
            - Random seed for reproducibility.
            - Logger for pipeline events.
            - Pipeline state machine for managing pipeline states.
            - Model manager for handling model-related operations.
            - Data manager for managing data loading and preprocessing.
            - Training manager for orchestrating training procedures.
            - Inference engine for running inference tasks.

        Logs:
            - Pipeline initialization details including the pretrained model name.
            - Transitions the pipeline state to INITIALIZING.
        """
        self.config = config
        fix_seed(self.config.seed)
        self.logger = MoELogger(
            "moe_pipeline.log", Path(self.config.logging_params.log_dir)
        ).get_logger()

        self.distributed_manager = DistributedManager(self.logger)

        self.state_machine = PipelineStateMachine(self.logger)

        self.model_manager = ModelManager(
            self.config, self.logger, self.distributed_manager
        )
        self.data_manager = DataManager(self.config, self.logger)
        self.training_manager = TrainingManager(
            self.model_manager, self.config, self.distributed_manager, self.logger
        )
        self.inference_engine = InferenceEngine(self.model_manager, self.logger)

        self.logger.info(f"Pipeline initialized: {self.config.pretrained_model_name}")
        self.state_machine.transition_to(PipelineState.INITIALIZING)

    @property
    def is_ready(self) -> bool:
        """
        Checks if the pipeline is in the READY state.

        Returns:
            bool: True if the pipeline's state is READY, False otherwise.
        """
        return self.state_machine.state == PipelineState.READY

    async def setup(self) -> PipelineResult:
        """
        Asynchronously sets up the pipeline by initializing model and training managers,
        transitioning the pipeline state to READY, and logging the process.

        Returns:
            PipelineResult: An object indicating the success or failure of the setup process.
                On success, returns PipelineResult(success=True).
                On failure, transitions the state to ERROR, logs the exception, and returns
                PipelineResult(success=False, error=str(e)).
        """
        try:
            self.logger.info("Starting pipeline setup")
            await self.model_manager.initialize()
            self.training_manager.initialize()
            self.state_machine.transition_to(PipelineState.READY)
            self.logger.info("Pipeline setup completed successfully")
            return PipelineResult(success=True)

        except Exception as e:
            self.state_machine.transition_to(PipelineState.ERROR)
            self.logger.error(f"Pipeline setup failed: {e}")
            self.logger.debug(traceback.format_exc())
            return PipelineResult(success=False, error=str(e))

    async def prepare_dataset(self) -> PipelineResult:
        """
        Asynchronously prepares the dataset using the data manager and transitions the pipeline state to DATA_PREPARING.

        Returns:
            PipelineResult: The result of the dataset preparation process.
        """
        result = await self.data_manager.prepare_dataset()
        self.state_machine.transition_to(PipelineState.DATA_PREPARING)
        return result

    async def train(self) -> PipelineResult:
        """
        Asynchronously trains the pipeline using the training manager.

        Returns:
            PipelineResult: The result of the training process, indicating success or failure.

        Raises:
            Exception: Propagates any exception raised during training.

        Workflow:
            - Checks if the pipeline is ready; returns a failed PipelineResult if not.
            - Transitions the pipeline state to TRAINING.
            - Invokes the training manager's train method asynchronously.
            - On success, transitions the state to READY and returns the result.
            - On failure, transitions the state to ERROR and re-raises the exception.
        """
        if not self.is_ready:
            return PipelineResult(
                success=False,
                error=f"Pipeline not ready. Current state: {self.state_machine.state}",
            )

        self.state_machine.transition_to(PipelineState.TRAINING)
        try:
            result = await self.training_manager.train()
            self.state_machine.transition_to(PipelineState.READY)
            return result
        except Exception:
            self.state_machine.transition_to(PipelineState.ERROR)
            raise

    async def evaluate(self) -> PipelineResult:
        """
        Asynchronously evaluates the pipeline using the training manager.

        Returns:
            PipelineResult: The result of the evaluation, indicating success or failure.

        Raises:
            Exception: Propagates any exception raised during evaluation.

        Notes:
            - If the pipeline is not ready, returns a PipelineResult indicating failure.
            - Transitions the pipeline state to EVALUATING before evaluation, and to READY or ERROR after.
        """
        if not self.is_ready:
            return PipelineResult(
                success=False,
                error=f"Pipeline not ready. Current state: {self.state_machine.state}",
            )

        self.state_machine.transition_to(PipelineState.EVALUATING)
        try:
            result = await self.training_manager.evaluate()
            self.state_machine.transition_to(PipelineState.READY)
            return result
        except Exception:
            self.state_machine.transition_to(PipelineState.ERROR)
            raise

    async def infer(self, prompts: Union[str, List[str]], **kwargs) -> PipelineResult:
        """
        Asynchronously performs inference on the provided prompts using the pipeline.

        Args:
            prompts (Union[str, List[str]]): A single prompt or a list of prompts to be processed.
            **kwargs: Additional keyword arguments to be passed to the inference engine.

        Returns:
            PipelineResult: The result of the inference, including success status and any errors.

        Raises:
            Exception: Propagates any exception raised during inference.

        Notes:
            - If the pipeline is not ready, returns a PipelineResult indicating failure.
            - Transitions the pipeline state to INFERRING before starting inference and back to READY upon completion.
            - If an error occurs during inference, transitions the pipeline state to ERROR.
        """
        self.logger.info("Starting inference")
        await self.setup()
        if not self.is_ready:
            return PipelineResult(
                success=False,
                error=f"Pipeline not ready. Current state: {self.state_machine.state}",
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        self.logger.info(f"Number of prompts to process: {len(prompts)}")

        request = InferenceRequest(prompts=prompts, **kwargs)

        self.state_machine.transition_to(PipelineState.INFERRING)
        try:
            result = await self.inference_engine.infer_batch(request)
            self.state_machine.transition_to(PipelineState.READY)
            return result
        except Exception:
            self.state_machine.transition_to(PipelineState.ERROR)
            raise

    async def deploy(self) -> PipelineResult:
        """
        Deploys the trained model using the pipeline.

        This asynchronous method performs the deployment process, which may include:
            - Saving model artifacts
            - Uploading the model to a registry
            - Deploying the model to serving infrastructure
            - Running validation tests

        Returns:
            PipelineResult: An object indicating the success or failure of the deployment.
                - success (bool): True if deployment was successful, False otherwise.
                - error (str, optional): Error message if deployment failed.

        Raises:
            Exception: Logs and handles any exceptions that occur during deployment.
        """
        if not self.is_ready:
            return PipelineResult(
                success=False, error="Pipeline not ready for deployment"
            )

        try:
            self.logger.info("Starting model deployment")
            # TODO: Implement deployment logic (model registry, serving, etc.)
            # - Saving model artifacts
            # - Uploading to model registry
            # - Deploying to serving infrastructure
            # - Running validation tests

            self.logger.info("Model deployment completed")
            return PipelineResult(success=True)

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.logger.debug(traceback.format_exc())
            return PipelineResult(success=False, error=str(e))

    async def run_stage(self, stage: PipelineStage) -> PipelineResult:
        """
        Asynchronously executes the specified pipeline stage.

        Args:
            stage (PipelineStage): The stage of the pipeline to execute.

        Returns:
            PipelineResult: The result of executing the specified stage.

        Notes:
            - For the INFER stage, instructs the user to use the `infer()` method directly.
            - For the FULL_PIPELINE stage, runs the entire pipeline.
            - For unknown stages, returns an error in the result.
            - For valid stages (PREPARE_DATA, SETUP, TRAIN, EVALUATE, DEPLOY), executes the corresponding method asynchronously.
        """
        stage_map = {
            PipelineStage.PREPARE_DATA: self.prepare_dataset,
            PipelineStage.SETUP: self.setup,
            PipelineStage.TRAIN: self.train,
            PipelineStage.EVALUATE: self.evaluate,
            PipelineStage.DEPLOY: self.deploy,
        }

        if stage == PipelineStage.INFER:
            return PipelineResult(
                success=False, error="Use infer() method directly for inference"
            )

        if stage == PipelineStage.FULL_PIPELINE:
            return await self.run_full_pipeline()

        if stage not in stage_map:
            return PipelineResult(success=False, error=f"Unknown stage: {stage}")

        return await stage_map[stage]()

    async def run_full_pipeline(self) -> PipelineResult:
        """
        Executes the full pipeline asynchronously, running each stage in sequence:
        'prepare_data', 'setup', 'train', 'evaluate', and 'deploy'.

        For each stage:
            - Logs the start of the stage.
            - Awaits the stage function.
            - Stores the result in a dictionary.
            - If a stage fails (result.success is False), logs the error and returns a PipelineResult
              indicating failure, including the error message and partial results.

        If all stages succeed, logs completion and returns a PipelineResult indicating success
        with the results of all stages.

        Returns:
            PipelineResult: An object containing the overall success status, error message (if any),
            and results from each pipeline stage.
        """
        stages = [
            ("prepare_data", self.prepare_dataset),
            ("setup", self.setup),
            ("train", self.train),
            ("evaluate", self.evaluate),
            ("deploy", self.deploy),
        ]

        results = {}
        for stage_name, stage_func in stages:
            self.logger.info(f"Running pipeline stage: {stage_name}")
            result = await stage_func()
            results[stage_name] = result

            if not result.success:
                self.logger.error(f"Pipeline failed at stage: {stage_name}")
                self.logger.debug(traceback.format_exc())
                return PipelineResult(
                    success=False,
                    error=f"Failed at {stage_name}: {result.error}",
                    data=results,
                )

        self.logger.info("Full pipeline completed successfully")
        return PipelineResult(success=True, data=results)

    async def cleanup(self):
        """
        Asynchronously performs cleanup operations for the pipeline.

        Transitions the pipeline state to CLEANUP, logs the start of cleanup,
        invokes the model manager's cleanup method, and then transitions the
        pipeline state to UNINITIALIZED. Logs completion of cleanup. If any
        exception occurs during the process, logs the error and the traceback.

        Raises:
            Exception: If an error occurs during cleanup.
        """
        try:
            self.state_machine.transition_to(PipelineState.CLEANUP)
            self.logger.info("Starting pipeline cleanup")

            self.model_manager.cleanup()

            self.state_machine.transition_to(PipelineState.UNINITIALIZED)
            self.logger.info("Pipeline cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            self.logger.debug(traceback.format_exc())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
