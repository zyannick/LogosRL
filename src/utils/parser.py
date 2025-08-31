import argparse
import inspect
from pathlib import Path
from typing import Optional, Union, get_args, get_origin

from pydantic import BaseModel

from utils.configurations import MoERLConfig


def _get_base_type(type_hint):
    """
    Extracts the base type from a type hint.
    e.g., Optional[Path] -> Path
    """
    origin = get_origin(type_hint)
    if origin in (Union, Optional):
        args = get_args(type_hint)
        base_type = next((arg for arg in args if arg is not type(None)), None)
        return base_type or type_hint
    return type_hint


def recursive_add_model(parser: argparse.ArgumentParser, model_class: type[BaseModel]):
    """
    Recursively adds arguments from a Pydantic model to an argparse parser,
    correctly handling complex types like Optional[Path].
    """
    fields = model_class.model_fields

    for name, field in fields.items():
        field_type_hint = field.annotation

        base_type = _get_base_type(field_type_hint)

        if base_type is Path:
            parser.add_argument(
                f"--{name}",
                type=str,
                default=field.default,
                help=field.description,
            )
        elif inspect.isclass(base_type) and issubclass(base_type, BaseModel):
            recursive_add_model(parser, base_type)
        else:
            if name == "system_params":
                continue

            parser.add_argument(
                f"--{name}",
                type=base_type,
                default=field.default,
                help=field.description,
            )


def create_argument_parser():
    """
    Creates and returns an argument parser for the Mixture of Experts RL Trainer.

    The parser is configured with program name and description, and recursively adds
    model-specific arguments based on the MoERLConfig configuration.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="MoE RL Trainer",
        description="Fine-tune a Mixture of Experts model using PPO.",
    )

    recursive_add_model(parser, MoERLConfig)

    return parser
