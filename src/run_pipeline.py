import asyncio
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from pathlib import Path

import mlflow

from pipeline import MoERLPipeline, PipelineStage
from utils.configurations import MoERLConfig
from utils.parser import create_argument_parser


async def main():
    """
    Asynchronously runs the main pipeline for the MoE-RL project.

    This function parses command-line arguments, initializes configuration,
    creates necessary directories, saves configuration to a YAML file, sets up MLflow tracking,
    and executes the specified pipeline stage using the MoERLPipeline context manager.

    Logs the outcome of the pipeline stage, reporting success or failure.

    Raises:
        Exception: Propagates any exceptions raised during pipeline execution.

    Returns:
        None
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    config = MoERLConfig.from_argparse(vars(args))
    config.create_directories()

    config.save_to_yaml(config.checkpoint_path / "config.yaml")

    stage_to_run = PipelineStage.str_to_enum(config.pipeline_stage)

    mlflow.set_tracking_uri(f"sqlite:///{Path(config.output_dir) / 'mlflow.db'}")

    async with MoERLPipeline(config) as pipeline:
        if stage_to_run == PipelineStage.INFER:
            print(f"Running inference with prompt: {config.inference_prompt}")
            result = await pipeline.infer(config.inference_prompt)
            print(f"Inference result: {result}")
        else:
            result = await pipeline.run_stage(stage_to_run)

            if not result.success:
                pipeline.logger.error(
                    f"Pipeline failed at stage: {config.pipeline_stage} with error: {result.error}"
                )
            else:
                pipeline.logger.info(
                    f"Pipeline stage '{config.pipeline_stage}' completed successfully."
                )


if __name__ == "__main__":
    asyncio.run(main())
