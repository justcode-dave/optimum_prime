"""
This module contains utilities and classes for converting Hugging Face `transformers` models into their optimized 
`BetterTransformer` implementations. BetterTransformer is a fast inference mechanism for transformer models 
introduced by PyTorch in version 1.12.

The conversion enhances model inference speed by replacing standard transformer layers with optimized 
versions supported by PyTorch's `BetterTransformer`. This is particularly useful for reducing latency 
in transformer-based models during inference without significantly impacting performance.

Main components:
- **raise_save_or_push_incompatible**: Prevents the user from saving or pushing models converted to 
  BetterTransformer without first reverting them to their original state.
  
- **replace_to_bettertransformer**: Recursively replaces transformer layers in the model with their 
  corresponding BetterTransformer layers, if supported.

- **set_last_layer**: Marks the final transformer layer in the model by setting the `is_last_layer` attribute, 
  necessary for ensuring proper behavior during the forward pass.

- **BetterTransformer**: A wrapper class that facilitates the conversion of `transformers` models to 
  their BetterTransformer counterparts. It also provides functionality to reverse the transformation, 
  restoring the model to its original state.

BetterTransformer is designed for models with large-scale transformer architectures, enabling faster 
inference by utilizing PyTorch's `scaled_dot_product_attention` mechanism and optimized transformer kernels.

Note:
- Models using `BetterTransformer` are not compatible with 8-bit quantization (`load_in_8bit`).
- The conversion is deprecated for some models (e.g., Falcon, GPT-BigCode) as Hugging Face's Transformers 
  library natively supports the optimizations provided by BetterTransformer.
"""

import logging
import os
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from packaging.version import parse

from ..utils import check_if_pytorch_greater, is_accelerate_available, recurse_getattr, recurse_setattr
from .models import BetterTransformerManager

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

if is_accelerate_available():
    # Importing Accelerate utilities for model dispatch and device mapping
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_module

# Error message for models without BetterTransformer implementation
ERROR_MESSAGE = r"The Better Transformers implementation for the model {model_name} has not been implemented yet. Please open an issue requesting the addition of this model with its `BetterTransformer` implementation."


def raise_save_or_push_incompatible(*_, **__):
    r"""
    Raises an error when a user attempts to save or push a model that is incompatible with BetterTransformer. 
    The model must be reverted to its original state before performing these operations.
    """
    raise ValueError(
        "You are trying to save or push a model that has been converted with `BetterTransformer`.",
        " Please revert the model to its original state before calling `save_pretrained` or `push_to_hub`.",
        " By calling model = BetterTransformer.reverse(model) before saving or pushing.",
    )


def replace_to_bettertransformer(model, config):
    r"""
    Recursively replaces a model's transformer layers with their BetterTransformer counterparts.

    Steps:
    - Step 1: Recursively explore the model's modules.
    - Step 2: Check for the availability of the BetterTransformer implementation.
    - Step 3: If found, replace the transformer layer with the BetterTransformer version.
    - Step 4: Raise an error if the implementation is missing.
    - Step 5: Post-process the converted model to set the `is_last_layer` attribute for the last transformer layer.

    Args:
        model (`torch.nn.Module`): The input model to convert.
        config (`transformers.PreTrainedConfig`): Configuration dictionary for the model.

    Returns:
        The converted model with BetterTransformer layers.
    """
    for name, module in model.named_children():
        if hasattr(module, "SCB"):
            # BetterTransformer is incompatible with 8-bit quantization modules.
            raise ValueError(
                "`load_in_8bit` and `BetterTransformers` are mutually exclusive",
                " please pass a model that is not loaded in 8-bit.",
            )

        # Identify classes eligible for replacement with BetterTransformer
        target_classes = list(BetterTransformerManager.MODEL_MAPPING[config.model_type].keys())

        # Optionally override specific methods in the module
        if config.model_type in BetterTransformerManager.OVERWRITE_METHODS:
            for class_name, method_name_and_replacement in BetterTransformerManager.OVERWRITE_METHODS[config.model_type].items():
                if module.__class__.__name__ == class_name:
                    method_name = method_name_and_replacement[0]
                    new_method = method_name_and_replacement[1]
                    setattr(module, method_name, types.MethodType(new_method, module))

        should_replace_module = False
        for target_class in target_classes:
            should_replace_module = module.__class__.__name__ == target_class
            if should_replace_module:
                bettertransformer_module = BetterTransformerManager.MODEL_MAPPING[config.model_type][target_class](module, config)
                model._modules[name] = bettertransformer_module
                break

        # Recursively process child modules if they are not eligible for replacement
        if len(list(module.children())) > 0 and should_replace_module is False:
            if config.model_type not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM or (
                config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM and name not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[config.model_type]
            ):
                replace_to_bettertransformer(module, config)

    return model


def set_last_layer(model: torch.nn.Module):
    r"""
    Set the `is_last_layer` attribute for the final transformer layer in a converted BetterTransformer model.

    Args:
        model (`torch.nn.Module`): The input model converted to BetterTransformer.

    Raises:
        `NotImplementedError`: Raised if the process fails and the model is not supported.
    """
    dict_named_module = dict(model.named_modules())
    sort_fn = lambda list_modules: [module.__class__.__name__ for module in list_modules]  # Sort function for module list

    modulelist_lengths = []

    # Identify the longest list of transformer layers within the model
    for key in dict_named_module.keys():
        if (
            isinstance(dict_named_module[key], torch.nn.ModuleList)
            and "encoder" in key
            and (
                model.config.model_type not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM
                or (
                    model.config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM
                    and all(name not in key for name in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[model.config.model_type])
                )
            )
        ):
            modulelist_lengths.append((len(dict_named_module[key]), key))

    # Handle special cases where transformer layers are nested (e.g., in Albert)
    if len(modulelist_lengths) > 1:
        _, key = max(modulelist_lengths, key=lambda item: item[0])
        largest_module_list = dict_named_module[key]

        # Set `is_last_layer` for the last transformer layer
        for module in largest_module_list[-1].modules():
            if "LayerBetterTransformer" in module.__class__.__name__:
                setattr(module, "is_last_layer", True)
                return
    else:
        for key in dict_named_module.keys():
            if isinstance(dict_named_module[key], torch.nn.ModuleList) and all(
                "LayerBetterTransformer" in module_name for module_name in sort_fn(dict_named_module[key])
            ):
                setattr(dict_named_module[key][-1], "is_last_layer", True)
                return

    raise Exception(
        f"The transformation of the model {model.__class__.__name__} to BetterTransformer failed while it should not. Please fill"
        " a bug report or open a PR to support this model at https://github.com/huggingface/optimum/"
    )


class BetterTransformer(object):
    r"""
    Conversion wrapper for transforming a `transformers` model to its `BetterTransformer` version.
    Based on PyTorch's "A Better Transformer" released in version 1.12:
    https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

    Original PR: https://github.com/huggingface/transformers/pull/19553 (adapted for Optimum)
    """

    @check_if_pytorch_greater(
        "1.13.99",
        "Please upgrade PyTorch following https://pytorch.org/get-started/locally/ in order to use BetterTransformer.",
    )
    def transform(
        model: torch.nn.Module,
        keep_original_model: bool = False,
        max_memory: Optional[Dict] = None,
        offload_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> torch.nn.Module:
        r"""
        Converts a `transformers` model to the `BetterTransformer` version.

        Args:
            model (`torch.nn.Module`): Original `transformers` model to convert.
            keep_original_model (`bool`, defaults to `False`): Whether to retain the original model.
            max_memory (`Optional[Dict]`, defaults to `None`): Memory limits for device allocation (used with Accelerate).

        Returns:
            The converted model if successful.
        """
        logger.warning(
            "The class `optimum.bettertransformers.transformation.BetterTransformer` is deprecated and will be removed in a future release."
        )

        hf_config = model.config
        if hf_config.model_type in ["falcon", "gpt_bigcode", "llama", "whisper"]:
            raise ValueError(
                f"Transformers natively supports BetterTransformer optimizations (torch.nn.functional.scaled_dot_product_attention) for the model type {hf_config.model_type}. As such, `model.to_bettertransformers()` or `BetterTransformer.transform(model)` is no longer necessary."
                " Please upgrade to transformers>=4.36 and torch>=2.1.1. More details here: https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-and-memory-efficient-attention-through-pytorchs-scaleddotproductattention."
            )

        # Handle Accelerate model loading if necessary
        if hasattr(model, "hf_device_map"):
            load_accelerate = True
            hf_device_map = model.hf_device_map
        else:
            load_accelerate = False

        if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
            raise Exception("`BetterTransform.transform()` was called on a model already using Better Transformer modeling.")

        # Ensure the model is supported
        if BetterTransformerManager.cannot_support(model.config.model_type):
            raise ValueError(
                f"The model type {model.config.model_type} is not supported with BetterTransformer. The reason identified is:"
                f" {BetterTransformerManager.CAN_NOT_BE_SUPPORTED[model.config.model_type]}. Currently supported models are:"
                f" {BetterTransformerManager.MODEL_MAPPING.keys()}."
            )
        if not BetterTransformerManager.supports(model.config.model_type):
            raise NotImplementedError(
                f"The model type {model.config.model_type} is not yet supported with BetterTransformer. Please open an issue at https://github.com/huggingface/optimum/issues for support. Supported models: {BetterTransformerManager.MODEL_MAPPING.keys()}."
            )

        if parse(torch.__version__) <= parse("1.14"):
            raise ValueError(f"BetterTransformer requires torch>=2.0 but {torch.__version__} is installed. Please upgrade PyTorch.")

        if load_accelerate:
            remove_hook_from_module(model, recurse=True)  # Remove hooks from original model

        training_mode = model.training

        if keep_original_model:
            try:
                if not check_if_pytorch_greater(2.0, "Please upgrade PyTorch to >=2.0 to use training mode"):
                    model = model.requires_grad_(False)
                model_fast = deepcopy(model)
            except RuntimeError:
                raise ValueError(
                    f"The model {model.__class__.__name__} does not support `deepcopy`, required for `keep_original_model=True`. Try running with `keep_original_model=False`."
                )
            model_fast = replace_to_bettertransformer(model_fast, hf_config)
        else:
            model_fast = replace_to_bettertransformer(model, hf_config)
            model = None

        if BetterTransformerManager.requires_nested_tensor(model_fast.config.model_type):
            set_last_layer(model_fast)

        setattr(model_fast, "use_bettertransformer", True)  # Indicate that model uses BetterTransformer

        if load_accelerate:
            all_model_tensors = [name for name, _ in model_fast.state_dict().items()]
            for module_name in hf_device_map.keys():
                all_model_tensors = [name for name in all_model_tensors if not name.startswith(module_name)]

            if len(all_model_tensors) > 0:
                bt_device_map = infer_auto_device_map(model_fast, max_memory=max_memory)
            else:
                bt_device_map = hf_device_map

            model_fast = dispatch_model(model_fast, bt_device_map, offload_dir=offload_dir)

            if keep_original_model:
                model = dispatch_model(model, hf_device_map, offload_dir=offload_dir)

        # Warning for training issues due to lack of padding support
        logger.warning(
            "BetterTransformer does not support padding during training due to fused kernels not supporting attention masks."
            " Using padded batched data during training may lead to unexpected outputs. Please refer to the documentation for more information."
        )

        # Overwrite save and push methods to raise an error if attempted
        model_fast._old_save_pretrained = model_fast.save_pretrained
        model_fast._old_push_to_hub = model_fast.push_to_hub
        model_fast.save_pretrained = raise_save_or_push_incompatible
        model_fast.push_to_hub = raise_save_or_push_incompatible

        # Return the model to its original training or evaluation mode
        model_fast = model_fast.train() if training_mode else model_fast.eval()

        return model_fast

    def reverse(bt_model: "PreTrainedModel") -> "PreTrainedModel":
        """
        Converts a BetterTransformer model back to its canonical `transformers` version for saving or sharing.

        Args:
            bt_model (`PreTrainedModel`): The model using BetterTransformer to be reversed.

        Returns:
            `PreTrainedModel`: The original transformers model.
        """
        if getattr(bt_model, "use_bettertransformer", False) is False:
            raise ValueError("BetterTransformer.reverse() should only be used on models already transformed to BetterTransformer.")

        if parse(torch.__version__) <= parse("1.14"):
            raise ValueError(f"BetterTransformer reverse transform requires torch>=2.0 but {torch.__version__} is installed. Please upgrade PyTorch.")

        config = bt_model.config

        # Handle specific memory-heavy models
        if config.model_type not in ["wav2vec2", "hubert", "bark"]:
            with torch.device("meta"):
                reversed_model = bt_model.__class__(config)
        else:
            logger.warning("The reverse transform for architectures such as wav2vec2, hubert, and bark may be memory-heavy due to a PyTorch bug.")
            reversed_model = bt_model.__class__(config)

        # Restore evaluation or training mode for the reversed model
        if bt_model.training is False:
            reversed_model = reversed_model.eval()

        # Revert module replacements from BetterTransformer transformation
        reversed_modules_paths = []
        for path, module in reversed_model.named_modules():
            if path.startswith(tuple(reversed_modules_paths)):
                continue

            if config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM and any(
                subname in path for subname in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[config.model_type]
            ):
                continue

            target_classes = list(BetterTransformerManager.MODEL_MAPPING[config.model_type].keys())
            has_been_replaced = False
            for target_class in target_classes:
                if module.__class__.__name__ == target_class:
                    has_been_replaced = True
                    break

            if has_been_replaced:
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path)._revert(module))
                reversed_modules_paths.append(path + ".")

        # Replace parameters and buffers that were not modified by BetterTransformer
        for path, param in reversed_model.state_dict().items():
            if param.device == torch.device("meta") or not path.startswith(tuple(reversed_modules_paths)):
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path))

        for path, param in reversed_model.named_buffers():
            if param.device == torch.device("meta") or not path.startswith(tuple(reversed_modules_paths)):
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path))

        return reversed_model
