"""
Dummy Input Generation Classes for Frameworks

This module provides classes and functions for generating dummy input data, supporting multiple deep learning 
frameworks including PyTorch, TensorFlow, and NumPy. These inputs can be used for testing and benchmarking 
machine learning models. It also includes utilities to map data types across frameworks.

Functions:
    - check_framework_is_available: Decorator to check whether the requested framework (PyTorch, TensorFlow, or NumPy) is available.
    - random_int_tensor: Generates a tensor of random integers for the requested framework.
    - random_mask_tensor: Generates a padded mask tensor for the requested framework.

Classes:
    - DTYPE_MAPPER: Maps data types (like int, float) to specific frameworks.
    - DummyInputGenerator: Abstract base class for generating dummy inputs for various supported frameworks.
"""

import functools  # For higher-order functions like decorators
import random  # To generate random values
from abc import ABC, abstractmethod  # Abstract base class and methods
from typing import Any, List, Optional, Tuple, Union  # Type hinting support

import numpy as np  # For generating dummy inputs with NumPy
from transformers.utils import is_tf_available, is_torch_available  # To check availability of PyTorch and TensorFlow

from ..utils import check_if_transformers_greater  # Utility function to check the version of Transformers
from .normalized_config import (  # Import normalized configurations for different model types
    NormalizedConfig,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)

# Framework imports conditional on availability
if is_torch_available():
    import torch  # Import PyTorch if available

if is_tf_available():
    import tensorflow as tf  # Import TensorFlow if available


def check_framework_is_available(func):
    """
    Decorator to ensure the requested framework is installed before calling the function.
    
    Args:
        func: The function to wrap.

    Raises:
        RuntimeError: If the requested framework (PyTorch or TensorFlow) is not available.

    Returns:
        A wrapped function that performs the availability check.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        framework = kwargs.get("framework", "pt")  # Default framework is PyTorch
        pt_asked_but_not_available = framework == "pt" and not is_torch_available()
        tf_asked_but_not_available = framework == "tf" and not is_tf_available()
        if (pt_asked_but_not_available or tf_asked_but_not_available) and framework != "np":
            framework_name = "PyTorch" if framework == "pt" else "TensorFlow"
            raise RuntimeError(f"Requested the {framework_name} framework, but it does not seem installed.")
        return func(*args, **kwargs)

    return wrapper


# Default shapes for dummy inputs for different types of data (e.g., text, images, audio)
DEFAULT_DUMMY_SHAPES = {
    "batch_size": 2,
    "sequence_length": 16,
    "num_choices": 4,
    # image data
    "width": 64,
    "height": 64,
    "num_channels": 3,
    "point_batch_size": 3,
    "nb_points_per_image": 2,
    # audio data
    "feature_size": 80,
    "nb_max_frames": 3000,
    "audio_sequence_length": 16000,
}


class DTYPE_MAPPER:
    """
    A utility class to map data types across frameworks (NumPy, PyTorch, TensorFlow).
    
    Each framework has its own way of representing data types (e.g., float32, int64). This class provides a 
    standardized way to retrieve the appropriate data type for a given framework.

    Methods:
        - np: Returns the corresponding NumPy dtype.
        - pt: Returns the corresponding PyTorch dtype.
        - tf: Returns the corresponding TensorFlow dtype.
    """
    
    @classmethod
    def np(cls, dtype):
        """
        Maps data type strings to NumPy dtypes.

        Args:
            dtype (str): Data type to be mapped (e.g., "fp32", "int64").

        Returns:
            The corresponding NumPy dtype.
        """
        mapping = {
            "fp32": np.float32,
            "fp16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int8": np.int8,
            "bool": bool,
        }
        return mapping[dtype]

    @classmethod
    def pt(cls, dtype):
        """
        Maps data type strings to PyTorch dtypes.

        Args:
            dtype (str): Data type to be mapped (e.g., "fp32", "int64").

        Returns:
            The corresponding PyTorch dtype.
        """
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        return mapping[dtype]

    @classmethod
    def tf(cls, dtype):
        """
        Maps data type strings to TensorFlow dtypes.

        Args:
            dtype (str): Data type to be mapped (e.g., "fp32", "int64").

        Returns:
            The corresponding TensorFlow dtype.
        """
        mapping = {
            "fp32": tf.float32,
            "fp16": tf.float16,
            "bf16": tf.bfloat16,
            "int64": tf.int64,
            "int32": tf.int32,
            "int8": tf.int8,
            "bool": tf.bool,
        }
        return mapping[dtype]


class DummyInputGenerator(ABC):
    """
    Abstract Base Class for generating dummy inputs for different frameworks (PyTorch, TensorFlow, NumPy).
    
    This class provides a standardized interface for generating dummy input data across supported frameworks. 
    Subclasses must implement the `generate` method to create specific types of input.
    """

    SUPPORTED_INPUT_NAMES = ()  # Tuple of input names supported by the generator

    def supports_input(self, input_name: str) -> bool:
        """
        Checks whether the `DummyInputGenerator` supports generating the specified input.

        Args:
            input_name (str): The name of the input to check for support.

        Returns:
            bool: True if the input is supported, False otherwise.
        """
        return any(input_name.startswith(supported_input_name) for supported_input_name in self.SUPPORTED_INPUT_NAMES)

    @abstractmethod
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Abstract method to generate a dummy input tensor for the specified framework.

        Args:
            input_name (str): The name of the input to generate.
            framework (str, optional): The deep learning framework to use ("pt" for PyTorch, "tf" for TensorFlow, "np" for NumPy). Defaults to "pt".
            int_dtype (str, optional): The integer data type for the input tensor. Defaults to "int64".
            float_dtype (str, optional): The floating-point data type for the input tensor. Defaults to "fp32".

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    @staticmethod
    @check_framework_is_available
    def random_int_tensor(
        shape: List[int], max_value: int, min_value: int = 0, framework: str = "pt", dtype: str = "int64"
    ):
        """
        Generates a tensor of random integers in the range [min_value, max_value).

        Args:
            shape (List[int]): The shape of the tensor to generate.
            max_value (int): The maximum value for the random integers.
            min_value (int, optional): The minimum value for the random integers. Defaults to 0.
            framework (str, optional): The framework to generate the tensor in ("pt", "tf", or "np"). Defaults to "pt".
            dtype (str, optional): The data type of the generated tensor. Defaults to "int64".

        Returns:
            A random tensor of integers in the specified framework.
        """
        if framework == "pt":
            return torch.randint(low=min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.pt(dtype))
        elif framework == "tf":
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
        else:
            return np.random.randint(min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.np(dtype))

    @staticmethod
    @check_framework_is_available
    def random_mask_tensor(shape: List[int], padding_side: str = "right", framework: str = "pt", dtype: str = "int64"):
        """
        Generates a mask tensor with either right or left padding.

        Args:
            shape (List[int]): The shape of the mask tensor to generate.
            padding_side (str, optional): Whether to apply padding on the "right" or "left". Defaults to "right".
            framework (str, optional): The framework to generate the mask in ("pt", "tf", or "np"). Defaults to "pt".
            dtype (str, optional): The data type of the generated mask tensor. Defaults to "int64".

        Returns:
            A mask tensor with specified padding in the requested framework.
        """
        shape = tuple(shape)
        mask_length = random.randint(1, shape[-1] - 1)  # Randomly decide the padding length
        if framework == "pt":
            mask_tensor = torch.cat(
                [
                    torch.ones(*shape[:-1], shape[-1] - mask_length, dtype=DTYPE_MAPPER.pt(dtype)),
                    torch.zeros(*shape[:-1], mask_length, dtype=DTYPE_MAPPER.pt(dtype)),
                ],
                dim=-1,
            )
            if padding_side == "left":
                mask_tensor = torch.flip(mask_tensor, [-1])  # Flip to left pad
        elif framework == "tf":
            mask_tensor = tf.concat(
                [
                    tf.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.tf(dtype)),
                    tf.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.tf(dtype)),
                ],
                axis=-1,
            )
            if padding_side == "left":
                mask_tensor = tf.reverse(mask_tensor, [-1])  # Reverse for left padding
        else:
            mask_tensor = np.concatenate(
                [
                    np.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.np(dtype)),
                    np.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.np(dtype)),
                ],
                axis=-1,
            )
            if padding_side == "left":
                mask_tensor = np.flip(mask_tensor, [-1])  # Flip for left padding
        return mask_tensor
    @staticmethod
    @check_framework_is_available
    def random_float_tensor(
        shape: List[int], min_value: float = 0, max_value: float = 1, framework: str = "pt", dtype: str = "fp32"
    ):
        """
        Generates a tensor of random floats within the range [min_value, max_value).

        Args:
            shape (`List[int]`): 
                The shape of the random tensor.
            min_value (`float`, defaults to 0): 
                The minimum value for the generated floats.
            max_value (`float`, defaults to 1): 
                The maximum value for the generated floats.
            framework (`str`, defaults to `"pt"`): 
                The deep learning framework to use (PyTorch, TensorFlow, or NumPy). Default is PyTorch ("pt").
            dtype (`str`, defaults to `"fp32"`): 
                The data type for the generated float tensor. Could be "fp32", "fp16", or "bf16".

        Returns:
            A random tensor of floats in the specified framework.
        """
        if framework == "pt":
            tensor = torch.empty(shape, dtype=DTYPE_MAPPER.pt(dtype)).uniform_(min_value, max_value)
            return tensor
        elif framework == "tf":
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
        else:
            return np.random.uniform(low=min_value, high=max_value, size=shape).astype(DTYPE_MAPPER.np(dtype))

    @staticmethod
    @check_framework_is_available
    def constant_tensor(
        shape: List[int], value: Union[int, float] = 1, dtype: Optional[Any] = None, framework: str = "pt"
    ):
        """
        Generates a constant tensor filled with a specified value.

        Args:
            shape (`List[int]`): 
                The shape of the constant tensor.
            value (`Union[int, float]`, defaults to 1): 
                The value to fill the tensor with.
            dtype (`Optional[Any]`, defaults to `None`): 
                The data type of the constant tensor.
            framework (`str`, defaults to `"pt"`): 
                The framework in which the tensor is created (PyTorch, TensorFlow, NumPy). Defaults to PyTorch.

        Returns:
            A constant tensor in the specified framework.
        """
        if framework == "pt":
            return torch.full(shape, value, dtype=dtype)
        elif framework == "tf":
            return tf.constant(value, dtype=dtype, shape=shape)
        else:
            return np.full(shape, value, dtype=dtype)

    @staticmethod
    def _infer_framework_from_input(input_) -> str:
        """
        Infers the framework (PyTorch, TensorFlow, or NumPy) from the input tensor.

        Args:
            input_: 
                The input tensor from which to infer the framework.

        Returns:
            `str`: The name of the detected framework.

        Raises:
            RuntimeError: If the framework cannot be inferred from the input.
        """
        framework = None
        if is_torch_available() and isinstance(input_, torch.Tensor):
            framework = "pt"
        elif is_tf_available() and isinstance(input_, tf.Tensor):
            framework = "tf"
        elif isinstance(input_, np.ndarray):
            framework = "np"
        else:
            raise RuntimeError(f"Could not infer the framework from {input_}")
        return framework

    @classmethod
    def concat_inputs(cls, inputs, dim: int):
        """
        Concatenates input tensors along a specified dimension.

        Args:
            inputs: 
                List of tensors to concatenate.
            dim (`int`): 
                The dimension along which to concatenate the tensors.

        Returns:
            The concatenated tensor.

        Raises:
            ValueError: If no inputs are provided.
        """
        if not inputs:
            raise ValueError("You did not provide any inputs to concatenate")
        framework = cls._infer_framework_from_input(inputs[0])
        if framework == "pt":
            return torch.cat(inputs, dim=dim)
        elif framework == "tf":
            return tf.concat(inputs, axis=dim)
        else:
            return np.concatenate(inputs, axis=dim)

    @classmethod
    def pad_input_on_dim(
        cls,
        input_,
        dim: int,
        desired_length: Optional[int] = None,
        padding_length: Optional[int] = None,
        value: Union[int, float] = 1,
        dtype: Optional[Any] = None,
    ):
        """
        Pads an input tensor along a specific dimension.

        Args:
            input_: 
                The input tensor to pad.
            dim (`int`): 
                The dimension along which to pad.
            desired_length (`Optional[int]`, defaults to `None`): 
                The desired length of the tensor after padding along the dimension.
            padding_length (`Optional[int]`, defaults to `None`): 
                The amount of padding to add along the dimension.
            value (`Union[int, float]`, defaults to 1): 
                The value to use for padding.
            dtype (`Optional[Any]`, defaults to `None`): 
                The data type of the padding.

        Returns:
            The padded tensor.

        Raises:
            ValueError: If neither `desired_length` nor `padding_length` is provided, or if both are provided.
        """
        if (desired_length is None and padding_length is None) or (
            desired_length is not None and padding_length is not None
        ):
            raise ValueError("You need to provide either `desired_length` or `padding_length`")
        
        framework = cls._infer_framework_from_input(input_)
        shape = input_.shape
        padding_shape = list(shape)
        diff = desired_length - shape[dim] if desired_length else padding_length
        if diff <= 0:
            return input_
        
        padding_shape[dim] = diff
        return cls.concat_inputs(
            [input_, cls.constant_tensor(padding_shape, value=value, dtype=dtype, framework=framework)], dim=dim
        )
class DummyTextInputGenerator(DummyInputGenerator):
    """
    Generates dummy encoder text inputs for text-based tasks, such as language modeling or text classification.
    Provides support for different types of inputs, including input IDs, attention masks, and token type IDs.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            A tuple of supported input names like `input_ids`, `attention_mask`, etc.

    Args:
        task (`str`): 
            The task for which dummy inputs are generated (e.g., "multiple-choice").
        normalized_config (`NormalizedTextConfig`): 
            The normalized configuration object containing model-specific attributes.
        batch_size (`int`, defaults to 2): 
            The batch size of the dummy inputs.
        sequence_length (`int`, defaults to 16): 
            The sequence length of the dummy inputs.
        num_choices (`int`, defaults to 4): 
            The number of choices (for multiple-choice tasks).
        random_batch_size_range (`Optional[Tuple[int, int]]`, defaults to `None`): 
            If provided, the batch size will be a random value in the given range.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, defaults to `None`): 
            If provided, the sequence length will be a random value in the given range.
        random_num_choices_range (`Optional[Tuple[int, int]]`, defaults to `None`): 
            If provided, the number of choices will be a random value in the given range.
        padding_side (`str`, defaults to `"right"`): 
            The side on which the padding is applied ("left" or "right").
        kwargs: 
            Additional arguments.
    """

    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "encoder_attention_mask",
        "token_type_ids",
        "position_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        padding_side: str = "right",
        **kwargs,
    ):
        """
        Initializes the DummyTextInputGenerator with the given task, configuration, and input generation settings.

        Args:
            task (`str`):
                The task type (e.g., "multiple-choice").
            normalized_config (`NormalizedTextConfig`):
                The normalized configuration object containing model-specific attributes.
            batch_size (`int`, defaults to 2):
                The batch size of the generated inputs.
            sequence_length (`int`, defaults to 16):
                The sequence length of the generated inputs.
            num_choices (`int`, defaults to 4):
                Number of choices for multiple-choice tasks.
            random_batch_size_range (`Optional[Tuple[int, int]]`, defaults to `None`):
                If provided, batch size will be randomly selected within this range.
            random_sequence_length_range (`Optional[Tuple[int, int]]`, defaults to `None`):
                If provided, sequence length will be randomly selected within this range.
            random_num_choices_range (`Optional[Tuple[int, int]]`, defaults to `None`):
                If provided, number of choices will be randomly selected within this range.
            padding_side (`str`, defaults to `"right"`):
                Determines the side on which the padding is applied.
            kwargs:
                Additional keyword arguments.
        """
        self.task = task

        # Handle vocab size depending on whether it's an encoder-decoder or standard model config.
        self.vocab_size = normalized_config.vocab_size

        # Randomize batch size, sequence length, and number of choices if ranges are provided
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length
        if random_num_choices_range:
            low, high = random_num_choices_range
            self.num_choices = random.randint(low, high)
        else:
            self.num_choices = num_choices

        # Set padding side and normalized config
        self.padding_side = padding_side
        self.normalized_config = normalized_config

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        """
        Generates dummy input based on the input name, task type, and framework (PyTorch, TensorFlow, or NumPy).

        Args:
            input_name (`str`):
                The name of the input (e.g., "input_ids", "attention_mask").
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs like `input_ids` or `attention_mask`.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for floating point inputs.

        Returns:
            A generated tensor with dummy data in the requested framework.
        """
        # Set value ranges depending on whether it's an input ID or mask
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size

        # Determine the shape of the input (multiple-choice tasks have different shape)
        shape = [self.batch_size, self.sequence_length]
        if self.task == "multiple-choice":
            shape = [self.batch_size, self.num_choices, self.sequence_length]

        # Generate mask tensors or input ID tensors depending on input type
        if "mask" in input_name:
            return self.random_mask_tensor(shape, padding_side=self.padding_side, framework=framework, dtype=int_dtype)
        else:
            return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class DummyXPathSeqInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy XPath sequence inputs. This is used for models that require XPath sequences, such as those used in
    structured data tasks.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for XPath sequences, including `xpath_tags_seq` and `xpath_subs_seq`.

    Args:
        task (`str`):
            The task for which dummy inputs are generated (e.g., "classification").
        normalized_config (`NormalizedTextConfig`):
            The normalized configuration object containing model-specific attributes.
        batch_size (`int`, defaults to 2):
            The batch size of the generated inputs.
        sequence_length (`int`, defaults to 16):
            The sequence length of the generated inputs.
        num_choices (`int`, defaults to 4):
            Number of choices (for multiple-choice tasks).
        random_batch_size_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, batch size will be randomly selected within this range.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, sequence length will be randomly selected within this range.
        random_num_choices_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, number of choices will be randomly selected within this range.
        padding_side (`str`, defaults to `"right"`):
            The side on which the padding is applied ("left" or "right").
        kwargs:
            Additional keyword arguments.
    """

    SUPPORTED_INPUT_NAMES = (
        "xpath_tags_seq",
        "xpath_subs_seq",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        padding_side: str = "right",
        **kwargs,
    ):
        """
        Initializes the DummyXPathSeqInputGenerator with the given task, configuration, and input generation settings.

        Args:
            See class-level docstring for argument details.
        """
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
            padding_side=padding_side,
            **kwargs,
        )
        # Max depth of XPath sequences and padding IDs
        self.max_depth = normalized_config.max_depth
        self.tag_pad_id = normalized_config.tag_pad_id
        self.subs_pad_id = normalized_config.subs_pad_id

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        """
        Generates a dummy XPath sequence input.

        Args:
            input_name (`str`):
                The name of the input to generate (e.g., `xpath_tags_seq`).
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs like `xpath_tags_seq`.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for floating point inputs.

        Returns:
            A generated tensor with dummy XPath data in the requested framework.
        """
        # Set the value range based on the input type
        min_value = 0
        max_value = self.tag_pad_id if input_name == "xpath_tags_seq" else self.subs_pad_id

        # Define the shape for XPath sequences, which includes max depth
        shape = [self.batch_size, self.sequence_length, self.max_depth]

        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class DummyDecoderTextInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy decoder text inputs. This is used for models with decoder components, such as Seq2Seq models.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for decoder text inputs, such as `decoder_input_ids` and `decoder_attention_mask`.
    """

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
    )


class DummySeq2SeqDecoderTextInputGenerator(DummyDecoderTextInputGenerator):
    """
    Generates dummy decoder text inputs for Seq2Seq models, supporting additional inputs like encoder outputs and hidden states.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for Seq2Seq models, including `decoder_input_ids`, `decoder_attention_mask`, 
            `encoder_outputs`, and `encoder_hidden_states`.
    """

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initializes the DummySeq2SeqDecoderTextInputGenerator with the given task, configuration, and input generation settings.

        Args:
            See class-level docstring for argument details.
        """
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
        )

        # Handle hidden size depending on whether it's an encoder-decoder or standard model config
        if isinstance(normalized_config, NormalizedEncoderDecoderConfig):
            self.hidden_size = normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS.hidden_size
        else:
            self.hidden_size = normalized_config.hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy Seq2Seq decoder text inputs or encoder outputs/hidden states.

        Args:
            input_name (`str`):
                The name of the input to generate (e.g., `encoder_hidden_states`).
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs like `decoder_input_ids`.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for floating point inputs.

        Returns:
            A generated tensor with dummy data in the requested framework, either encoder outputs/hidden states
            or decoder inputs.
        """
        if input_name in ["encoder_outputs", "encoder_hidden_states"]:
            return (
                self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.hidden_size],
                    min_value=0,
                    max_value=1,
                    framework=framework,
                    dtype=float_dtype,
                ),
                None,
                None,
            )

        return super().generate(input_name, framework=framework, int_dtype=int_dtype)

class DummyPastKeyValuesGenerator(DummyInputGenerator):
    """
    Generates dummy `past_key_values` inputs for models that require past key values, typically used in autoregressive models 
    for caching previous attention states.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for past key values, here `past_key_values`.

    Args:
        task (`str`):
            The task for which dummy inputs are generated (e.g., "causal-lm").
        normalized_config (`NormalizedTextConfig`):
            The normalized configuration object containing model-specific attributes.
        batch_size (`int`, defaults to 2):
            The batch size of the generated inputs.
        sequence_length (`int`, defaults to 16):
            The sequence length of the generated inputs.
        random_batch_size_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, batch size will be randomly selected within this range.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, sequence length will be randomly selected within this range.
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initializes the DummyPastKeyValuesGenerator with the given task, configuration, and input generation settings.

        Args:
            See class-level docstring for argument details.
        """
        self.num_layers = normalized_config.num_layers
        self.num_attention_heads = normalized_config.num_attention_heads
        self.hidden_size = normalized_config.hidden_size
        
        # Determine batch size
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

        # Determine sequence length
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy past key values inputs.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for floating point inputs.

        Returns:
            A list of tuples, where each tuple contains two tensors representing `key` and `value` in the attention mechanism.
        """
        # Define the shape of the key and value tensors
        shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )

        # Generate a list of (key, value) pairs for each layer
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class DummySeq2SeqPastKeyValuesGenerator(DummyInputGenerator):
    """
    Generates dummy `past_key_values` inputs for Seq2Seq architectures, including cache positions for encoder-decoder models.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for Seq2Seq architectures, here `past_key_values` and `cache_position`.

    Args:
        task (`str`):
            The task for which dummy inputs are generated (e.g., "seq2seq-lm").
        normalized_config (`Union[NormalizedSeq2SeqConfig, NormalizedEncoderDecoderConfig]`):
            The normalized configuration object for Seq2Seq or encoder-decoder models.
        batch_size (`int`, defaults to 2):
            The batch size of the generated inputs.
        sequence_length (`int`, defaults to 16):
            The sequence length of the generated inputs.
        encoder_sequence_length (`Optional[int]`, defaults to `None`):
            Sequence length for encoder outputs.
        random_batch_size_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, batch size will be randomly selected within this range.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, defaults to `None`):
            If provided, sequence length will be randomly selected within this range.
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values", "cache_position")

    def __init__(
        self,
        task: str,
        normalized_config: Union[NormalizedSeq2SeqConfig, NormalizedEncoderDecoderConfig],
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        encoder_sequence_length: Optional[int] = None,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initializes the DummySeq2SeqPastKeyValuesGenerator with the given task, configuration, and input generation settings.

        Args:
            See class-level docstring for argument details.
        """
        self.normalized_config = normalized_config

        # Determine batch size
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

        # Determine sequence length
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length

        # Set encoder sequence length
        self.encoder_sequence_length = (
            self.sequence_length if encoder_sequence_length is None else encoder_sequence_length
        )

        # Extract attention heads and hidden size for encoder and decoder
        if isinstance(normalized_config, NormalizedEncoderDecoderConfig):
            # Cross-attention and self-attention head configurations
            self.encoder_num_attention_heads = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.encoder_num_attention_heads
            self.decoder_num_attention_heads = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_attention_heads
            self.encoder_hidden_size = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
            self.decoder_hidden_size = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
            self.decoder_num_layers = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers
        else:
            self.encoder_num_attention_heads = normalized_config.encoder_num_attention_heads
            self.decoder_num_attention_heads = normalized_config.decoder_num_attention_heads
            self.encoder_hidden_size = normalized_config.hidden_size
            self.decoder_hidden_size = normalized_config.hidden_size
            self.decoder_num_layers = normalized_config.decoder_num_layers
class DummySeq2SeqPastKeyValuesGenerator(DummyInputGenerator):
    """
    Generates dummy past_key_values inputs for seq2seq architectures.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs, including "past_key_values" and "cache_position".
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values", "cache_position")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy inputs for `past_key_values` and `cache_position`.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for floating point inputs.

        Returns:
            A tensor or list of tensors representing the past key values or cache position for a seq2seq model.
        """
        if input_name == "past_key_values":
            # Generate past key values for both the encoder and decoder.
            encoder_shape = (
                self.batch_size,
                self.encoder_num_attention_heads,
                self.encoder_sequence_length,
                self.encoder_hidden_size // self.encoder_num_attention_heads,
            )
            decoder_shape = (
                self.batch_size,
                self.decoder_num_attention_heads,
                self.sequence_length,
                self.decoder_hidden_size // self.decoder_num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.decoder_num_layers)
            ]

        elif input_name == "cache_position":
            # Generate cache position as a single integer tensor.
            return self.random_int_tensor(
                shape=[1],
                max_value=self.sequence_length,
                framework=framework,
                dtype=int_dtype,
            )

        raise ValueError(f"Unsupported input name {input_name}")


class DummyBboxInputGenerator(DummyInputGenerator):
    """
    Generates dummy bounding box (bbox) inputs for vision models that require bbox inputs.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            The name of the supported input, here "bbox".
    """

    SUPPORTED_INPUT_NAMES = ("bbox",)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy bbox inputs in the format of `[batch_size, sequence_length, 4]`.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.

        Returns:
            A tensor representing bounding boxes for a batch of images.
        """
        # Generate random bounding box coordinates
        return self.random_int_tensor(
            [self.batch_size, self.sequence_length, 4],
            1,  # Previously, the value `self.max_2d_position_embeddings - 1` was used, but currently set to 1.
            framework=framework,
            dtype=int_dtype,
        )


class DummyVisionInputGenerator(DummyInputGenerator):
    """
    Generates dummy inputs for vision models, such as pixel values and masks.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for vision models, including "pixel_values" and "pixel_mask".
    """

    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        """
        Initializes the DummyVisionInputGenerator with the given task and configuration.

        Args:
            task (`str`):
                The task for which dummy inputs are generated (e.g., image classification).
            normalized_config (`NormalizedVisionConfig`):
                The normalized configuration object for vision models.
            batch_size (`int`, defaults to 2):
                The batch size of the generated inputs.
            num_channels (`int`, defaults to 3):
                The number of channels in the image (e.g., 3 for RGB).
            width (`int`, defaults to 64):
                The width of the generated images.
            height (`int`, defaults to 64):
                The height of the generated images.
        """
        self.task = task

        # Set the number of channels and image size based on the configuration.
        if normalized_config.has_attribute("num_channels"):
            self.num_channels = normalized_config.num_channels
        else:
            self.num_channels = num_channels

        if normalized_config.has_attribute("image_size"):
            self.image_size = normalized_config.image_size
        elif normalized_config.has_attribute("input_size"):
            input_size = normalized_config.input_size
            self.num_channels = input_size[0]
            self.image_size = input_size[1:]
        else:
            self.image_size = (height, width)

        if not isinstance(self.image_size, (tuple, list)):
            self.image_size = (self.image_size, self.image_size)

        self.batch_size = batch_size
        self.height, self.width = self.image_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy vision inputs, either pixel values or pixel masks.

        Args:
            input_name (`str`):
                The name of the input to generate (e.g., "pixel_values").
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for float inputs.

        Returns:
            A tensor representing pixel values or masks for the vision task.
        """
        if input_name == "pixel_mask":
            # Generate pixel mask with values of 0 and 1 (for padding and non-padding regions).
            return self.random_int_tensor(
                shape=[self.batch_size, self.height, self.width],
                max_value=1,
                framework=framework,
                dtype=int_dtype,
            )
        else:
            # Generate pixel values for images.
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_channels, self.height, self.width],
                framework=framework,
                dtype=float_dtype,
            )


class DummyAudioInputGenerator(DummyInputGenerator):
    """
    Generates dummy audio inputs for tasks that use either raw waveforms or extracted features.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            The names of the supported input features, including "input_features" and "input_values".
    """

    SUPPORTED_INPUT_NAMES = ("input_features", "input_values")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        feature_size: int = DEFAULT_DUMMY_SHAPES["feature_size"],
        nb_max_frames: int = DEFAULT_DUMMY_SHAPES["nb_max_frames"],
        audio_sequence_length: int = DEFAULT_DUMMY_SHAPES["audio_sequence_length"],
        **kwargs,
    ):
        """
        Initializes the `DummyAudioInputGenerator` with task and configuration.

        Args:
            task (`str`):
                The task for which audio inputs are generated (e.g., speech recognition).
            normalized_config (`NormalizedConfig`):
                The normalized configuration object for the audio model.
            batch_size (`int`, defaults to 2):
                The batch size of the generated audio inputs.
            feature_size (`int`, defaults to 80):
                The size of the audio features (e.g., number of mel frequency bins).
            nb_max_frames (`int`, defaults to 3000):
                The maximum number of frames in the audio input.
            audio_sequence_length (`int`, defaults to 16000):
                The sequence length of the raw audio waveform.
        """
        self.task = task
        self.normalized_config = normalized_config

        # Extract configuration for feature size or fall back to default.
        if hasattr(self.normalized_config, "feature_size"):
            self.feature_size = self.normalized_config.feature_size
        else:
            self.feature_size = feature_size
        self.nb_max_frames = nb_max_frames
        self.batch_size = batch_size
        self.sequence_length = audio_sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy audio inputs, either raw waveforms or features.

        Args:
            input_name (`str`):
                The name of the input to generate (e.g., "input_values" or "input_features").
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for float inputs.

        Returns:
            A tensor representing audio features or raw audio waveforms.
        """
        if input_name == "input_values":  # Generate raw waveform
            return self.random_float_tensor(
                shape=[self.batch_size, self.sequence_length],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        else:  # Generate audio features
            return self.random_float_tensor(
                shape=[self.batch_size, self.feature_size, self.nb_max_frames],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )


class DummyTimestepInputGenerator(DummyInputGenerator):
    """
    Generates dummy timestep-related inputs, often used in diffusion models or time-conditioned tasks.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            The names of the supported inputs including "timestep", "text_embeds", "time_ids", and "timestep_cond".
    """

    SUPPORTED_INPUT_NAMES = (
        "timestep",
        "text_embeds",
        "time_ids",
        "timestep_cond",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initializes the `DummyTimestepInputGenerator` for timestep-related input generation.

        Args:
            task (`str`):
                The task for which timestep inputs are generated.
            normalized_config (`NormalizedConfig`):
                The normalized configuration for the model.
            batch_size (`int`, defaults to 2):
                The batch size for input generation.
            random_batch_size_range (`Optional[Tuple[int, int]]`, optional):
                If provided, generates a random batch size within this range.
        """
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        self.text_encoder_projection_dim = normalized_config.text_encoder_projection_dim
        self.time_ids = 5 if normalized_config.requires_aesthetics_score else 6
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        self.time_cond_proj_dim = normalized_config.config.time_cond_proj_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy timestep-related inputs.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for float inputs.

        Returns:
            A tensor representing timestep, text embeddings, or time-conditioned inputs.
        """
        if input_name == "timestep":
            return self.random_int_tensor([self.batch_size], max_value=self.vocab_size, framework=framework, dtype=int_dtype)

        if input_name == "text_embeds":
            dim = self.text_encoder_projection_dim
        elif input_name == "timestep_cond":
            dim = self.time_cond_proj_dim
        else:
            dim = self.time_ids

        return self.random_float_tensor([self.batch_size, dim], max_value=self.vocab_size, framework=framework, dtype=float_dtype)


class DummyLabelsGenerator(DummyInputGenerator):
    """
    Generates dummy labels for supervised learning tasks.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            The names of supported inputs including "labels", "start_positions", and "end_positions".
    """

    SUPPORTED_INPUT_NAMES = (
        "labels",
        "start_positions",
        "end_positions",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initializes the `DummyLabelsGenerator` for generating dummy labels for supervised tasks.

        Args:
            task (`str`):
                The task for which labels are generated.
            normalized_config (`NormalizedConfig`):
                The normalized configuration for the model.
            batch_size (`int`, defaults to 2):
                The batch size for label generation.
            random_batch_size_range (`Optional[Tuple[int, int]]`, optional):
                If provided, generates a random batch size within this range.
        """
        self.task = task
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

        self.sequence_length = kwargs.get("sequence_length", None)
        self.num_labels = kwargs.get("num_labels", None)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy label inputs, either single-label or sequence-based.

        Args:
            input_name (`str`):
                The name of the label input to generate.
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.

        Returns:
            A tensor representing labels for classification or sequence-based tasks.
        """
        max_value = self.num_labels if self.num_labels is not None else 0
        shape = [self.batch_size] if self.sequence_length is None else [self.batch_size, self.sequence_length]

        return self.random_int_tensor(shape, max_value=max_value, framework=framework, dtype=int_dtype)

class DummyPointsGenerator(DummyInputGenerator):
    """
    Generates dummy points and corresponding labels for tasks requiring 2D point coordinates, 
    such as object detection or segmentation tasks.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs for points generation: "input_points" and "input_labels".
    """

    SUPPORTED_INPUT_NAMES = ("input_points", "input_labels")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        point_batch_size: int = DEFAULT_DUMMY_SHAPES["point_batch_size"],
        nb_points_per_image: int = DEFAULT_DUMMY_SHAPES["nb_points_per_image"],
        **kwargs,
    ):
        """
        Initializes the `DummyPointsGenerator` for generating 2D points and their labels.

        Args:
            task (`str`):
                The task requiring the point inputs (e.g., image segmentation).
            normalized_config (`NormalizedConfig`):
                The normalized configuration for the task.
            batch_size (`int`, defaults to 2):
                The batch size for the input.
            point_batch_size (`int`, defaults to 3):
                Number of point batches.
            nb_points_per_image (`int`, defaults to 2):
                Number of points per image.
        """
        self.task = task
        self.batch_size = batch_size
        self.point_batch_size = point_batch_size
        self.nb_points_per_image = nb_points_per_image

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy 2D points or their labels.

        Args:
            input_name (`str`):
                The name of the input to generate ("input_points" or "input_labels").
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for float inputs.

        Returns:
            A tensor representing the 2D points or labels.
        """
        if input_name == "input_points":
            shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image, 2]
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        else:  # input_labels
            shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image]
            return self.random_int_tensor(shape, min_value=0, max_value=1, framework=framework, dtype=int_dtype)


class DummyVisionEmbeddingsGenerator(DummyInputGenerator):
    """
    Generates dummy image positional embeddings or image embeddings.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs: "image_positional_embeddings" and "image_embeddings".
    """

    SUPPORTED_INPUT_NAMES = ("image_positional_embeddings", "image_embeddings")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        image_embedding_size: Optional[int] = None,
        output_channels: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the `DummyVisionEmbeddingsGenerator` for generating dummy image embeddings.

        Args:
            task (`str`):
                The task requiring image embeddings.
            normalized_config (`NormalizedConfig`):
                The normalized configuration for the model.
            batch_size (`int`, defaults to 2):
                The batch size for embedding generation.
            image_embedding_size (`Optional[int]`, optional):
                The size of the image embedding.
            output_channels (`Optional[int]`, optional):
                The number of output channels.
        """
        self.task = task
        self.batch_size = batch_size
        self.image_embedding_size = (
            image_embedding_size
            if image_embedding_size is not None
            else normalized_config.prompt_encoder_config.image_embedding_size
        )
        self.output_channels = (
            output_channels if output_channels is not None else normalized_config.vision_config.output_channels
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy image embeddings or positional embeddings.

        Args:
            input_name (`str`):
                The name of the input to generate ("image_positional_embeddings" or "image_embeddings").
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            int_dtype (`str`, defaults to `"int64"`):
                The dtype for integer inputs.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for float inputs.

        Returns:
            A tensor representing image embeddings or positional embeddings.
        """
        shape = [self.batch_size, self.output_channels, self.image_embedding_size, self.image_embedding_size]
        return self.random_float_tensor(shape, framework=framework)


class DummyPix2StructInputGenerator(DummyInputGenerator):
    """
    Generates dummy inputs for Pix2Struct models, using flattened image patches.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs: "flattened_patches".
    """

    SUPPORTED_INPUT_NAMES = ("flattened_patches",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        preprocessors: List[Any],
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        **kwargs,
    ):
        """
        Initializes the `DummyPix2StructInputGenerator` for generating dummy flattened patches.

        Args:
            task (`str`):
                The task requiring Pix2Struct inputs.
            normalized_config (`NormalizedConfig`):
                The normalized configuration for the model.
            preprocessors (`List[Any]`):
                The pre-processors used for the input pipeline.
            batch_size (`int`, defaults to 2):
                The batch size for generating flattened patches.
            num_channels (`int`, defaults to 3):
                Number of channels in the input image.
        """
        self.task = task
        self.batch_size = batch_size

        # Extract patch size from pre-processors
        patch_height = preprocessors[1].image_processor.patch_size["height"]
        patch_width = preprocessors[1].image_processor.patch_size["width"]
        self.flattened_patch_size = 2 + patch_height * patch_width * num_channels
        self.max_patches = preprocessors[1].image_processor.max_patches

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy flattened image patches for Pix2Struct models.

        Args:
            input_name (`str`):
                The name of the input to generate ("flattened_patches").
            framework (`str`, defaults to `"pt"`):
                The framework to generate the input for (PyTorch by default).
            float_dtype (`str`, defaults to `"fp32"`):
                The dtype for float inputs.

        Returns:
            A tensor representing flattened image patches.
        """
        shape = [self.batch_size, self.max_patches, self.flattened_patch_size]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy past_key_values for GPT BigCode models, which have a fused key-value cache.

    Args:
        input_name (`str`):
            The name of the input to generate.
        framework (`str`, defaults to `"pt"`):
            The framework to generate the input for.
        int_dtype (`str`, defaults to `"int64"`):
            The dtype for integer inputs.
        float_dtype (`str`, defaults to `"fp32"`):
            The dtype for float inputs.

    Returns:
        A tensor representing past key-values in the requested framework.
    """
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_value_shape = (
            self.batch_size,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads * 2,  # GPT BigCode uses fused key-value caching.
        )
        return [
            self.random_float_tensor(past_key_value_shape, framework=framework, dtype=float_dtype)
            for _ in range(self.num_layers)
        ]


class BloomDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy past_key_values for the Bloom model architecture.

    If the installed transformers version is 4.44 or above, it uses the base class's generation logic.
    Otherwise, it uses Bloom-specific key-value shapes.
    """

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if check_if_transformers_greater("4.44"):
            return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)
        else:
            past_key_shape = (
                self.batch_size * self.num_attention_heads,
                self.hidden_size // self.num_attention_heads,
                self.sequence_length,
            )
            past_value_shape = (
                self.batch_size * self.num_attention_heads,
                self.sequence_length,
                self.hidden_size // self.num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]


class MistralDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy `past_key_values` for the Mistral model, which uses a specific configuration for key-value heads.

    Args:
        task (`str`): The task that requires the generation of past key values.
        normalized_config (`NormalizedTextConfig`): Configuration with model-specific attributes.
        batch_size (`int`, defaults to 2): The batch size for generating the past key values.
        sequence_length (`int`, defaults to 16): The sequence length for generating the past key values.
        random_batch_size_range (`Optional[Tuple[int, int]]`, optional): If provided, allows randomization of batch size.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, optional): If provided, allows randomization of sequence length.
    """

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads  # Custom number of heads for Mistral

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy past key-value tensors based on Mistral's configuration.

        Args:
            input_name (`str`): The input name requesting past key values.
            framework (`str`, defaults to `"pt"`): The framework to generate the input for (e.g., PyTorch).
            int_dtype (`str`, defaults to `"int64"`): The data type for integer tensors.
            float_dtype (`str`, defaults to `"fp32"`): The data type for floating-point tensors.

        Returns:
            A list of randomly generated past key-value tensors.
        """
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class GemmaDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy `past_key_values` for the Gemma model, similar to Mistral, with additional attributes for key-value heads and head dimensions.

    Args:
        task (`str`): The task that requires the generation of past key values.
        normalized_config (`NormalizedTextConfig`): Configuration with model-specific attributes.
        batch_size (`int`, defaults to 2): The batch size for generating the past key values.
        sequence_length (`int`, defaults to 16): The sequence length for generating the past key values.
        random_batch_size_range (`Optional[Tuple[int, int]]`, optional): Allows randomization of batch size.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, optional): Allows randomization of sequence length.
    """

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.head_dim = normalized_config.head_dim  # Specific head dimension for Gemma

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy past key-value tensors based on Gemma's configuration.

        Args:
            input_name (`str`): The input name requesting past key values.
            framework (`str`, defaults to `"pt"`): The framework to generate the input for (e.g., PyTorch).
            int_dtype (`str`, defaults to `"int64"`): The data type for integer tensors.
            float_dtype (`str`, defaults to `"fp32"`): The data type for floating-point tensors.

        Returns:
            A list of randomly generated past key-value tensors.
        """
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class DummySpeechT5InputGenerator(DummyInputGenerator):
    """
    Generates dummy inputs for the SpeechT5 model, which includes output sequences, speaker embeddings, and spectrograms.

    Attributes:
        SUPPORTED_INPUT_NAMES (`Tuple[str]`):
            Names of the supported inputs: "output_sequence", "speaker_embeddings", "spectrogram".
    """

    SUPPORTED_INPUT_NAMES = ("output_sequence", "speaker_embeddings", "spectrogram")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        """
        Initializes the `DummySpeechT5InputGenerator` for generating dummy speech-related inputs.

        Args:
            task (`str`): The task requiring speech inputs.
            normalized_config (`NormalizedConfig`): Configuration with model-specific attributes.
            sequence_length (`int`, defaults to 16): The sequence length for generating the output sequence.
        """
        self.task = task
        self.batch_size = 1  # SpeechT5 does not support batch inference for now.
        self.sequence_length = sequence_length
        self.speaker_embedding_dim = normalized_config.speaker_embedding_dim
        self.num_mel_bins = normalized_config.num_mel_bins

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy tensors for SpeechT5 inputs based on input type.

        Args:
            input_name (`str`): The input name (e.g., "output_sequence", "speaker_embeddings").
            framework (`str`, defaults to `"pt"`): The framework to generate the input for.
            int_dtype (`str`, defaults to `"int64"`): The data type for integer tensors.
            float_dtype (`str`, defaults to `"fp32"`): The data type for float tensors.

        Returns:
            A tensor representing the requested speech input.
        """
        if input_name == "output_sequence":
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        elif input_name == "speaker_embeddings":
            shape = [self.batch_size, self.speaker_embedding_dim]
        elif input_name == "spectrogram":
            shape = [20, self.num_mel_bins]  # The length of the first axis is dynamic and arbitrary.
        else:
            raise ValueError(f"Unsupported input {input_name} for DummySpeechT5InputGenerator")

        return self.random_float_tensor(shape=shape, min_value=0, max_value=1, framework=framework, dtype=float_dtype)


class DummyVisionEncoderDecoderPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):
    """
    Generates dummy past_key_values inputs for vision-based encoder-decoder models, such as Vision Encoder-Decoder.

    Attributes:
        task (`str`): The task requiring vision encoder-decoder inputs.
        normalized_config (`NormalizedSeq2SeqConfig`): Configuration with model-specific attributes.
    """

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedSeq2SeqConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        encoder_sequence_length: Optional[int] = None,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initializes the `DummyVisionEncoderDecoderPastKeyValuesGenerator` for generating past key-values.

        Args:
            task (`str`): The task requiring past key-values generation.
            normalized_config (`NormalizedSeq2SeqConfig`): Configuration for vision-based encoder-decoder models.
            batch_size (`int`, defaults to 2): The batch size for generating past key values.
            sequence_length (`int`, defaults to 16): The sequence length for generating past key values.
            encoder_sequence_length (`Optional[int]`, optional): If provided, specifies the sequence length for the encoder.
        """
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            encoder_sequence_length=encoder_sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            **kwargs,
        )
        if normalized_config.model_type == "trocr":
            image_size = normalized_config.encoder.image_size
            patch_size = normalized_config.encoder.patch_size
            self.encoder_sequence_length = (image_size // patch_size) ** 2 + 1

        if isinstance(normalized_config.DECODER_NORMALIZED_CONFIG_CLASS, NormalizedSeq2SeqConfig):
            self.num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_layers
            self.use_cross_attention = True
        else:
            self.num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers
            self.use_cross_attention = False

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates past_key_values for vision-based encoder-decoder models.

        Args:
            input_name (`str`): The input name requesting past key values.
            framework (`str`, defaults to `"pt"`): The framework to generate the input for.
            int_dtype (`str`, defaults to `"int64"`): The data type for integer tensors.
            float_dtype (`str`, defaults to `"fp32"`): The data type for floating-point tensors.

        Returns:
            A list of randomly generated past key-value tensors.
        """
        decoder_hidden_size = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
        decoder_num_attention_heads = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_attention_heads
        decoder_shape = (
            self.batch_size,
            decoder_num_attention_heads,
            self.sequence_length,
            decoder_hidden_size // decoder_num_attention_heads,
        )

        if not self.use_cross_attention:
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]
        else:
            encoder_hidden_size = decoder_hidden_size
            encoder_num_attention_heads = decoder_num_attention_heads

            encoder_shape = (
                self.batch_size,
                encoder_num_attention_heads,
                self.encoder_sequence_length,
                encoder_hidden_size // encoder_num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]


class DummyCodegenDecoderTextInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    """
    Generates dummy decoder inputs for Codegen models.

    This class extends the `DummySeq2SeqDecoderTextInputGenerator` to handle specific Codegen configurations, including
    additional attributes like `num_codebooks`.

    Args:
        task (`str`): The task that requires the generation of inputs.
        normalized_config (`NormalizedTextConfig`): Configuration with model-specific attributes.
        batch_size (`int`, defaults to 2): The batch size for generating the input.
        sequence_length (`int`, defaults to 16): The sequence length for generating the input.
        num_choices (`int`, defaults to 4): Number of choices for multiple-choice tasks.
        random_batch_size_range (`Optional[Tuple[int, int]]`, optional): Range to randomize the batch size.
        random_sequence_length_range (`Optional[Tuple[int, int]]`, optional): Range to randomize sequence length.
        random_num_choices_range (`Optional[Tuple[int, int]]`, optional): Range to randomize number of choices.
    """

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
        )
        self.num_codebooks = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_codebooks  # Specific to Codegen

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy input data for Codegen models.

        Args:
            input_name (`str`): Name of the input tensor to generate.
            framework (`str`, defaults to `"pt"`): The framework for tensor generation (e.g., PyTorch).
            int_dtype (`str`, defaults to `"int64"`): Data type for integer tensors.
            float_dtype (`str`, defaults to `"fp32"`): Data type for floating-point tensors.

        Returns:
            A tensor filled with random integer values or calls the parent method for other inputs.
        """
        if input_name in ["decoder_input_ids"]:
            min_value = 0
            max_value = 2 if input_name != "input_ids" else self.vocab_size
            shape = [self.batch_size * self.num_codebooks, self.sequence_length]
            return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)

        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class DummyEncodecInputGenerator(DummyInputGenerator):
    """
    Generates dummy `audio_codes` input for Encodec models, which specialize in audio encoding.

    Args:
        task (`str`): The task that requires the generation of inputs.
        normalized_config (`NormalizedConfig`): Model configuration with attributes like `num_codebooks`.
        sequence_length (`int`, defaults to 16): The sequence length for generating audio codes.
        batch_size (`int`, defaults to 2): The batch size for generating the input.
    """

    SUPPORTED_INPUT_NAMES = ("audio_codes",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.num_codebooks = normalized_config.decoder.num_codebooks  # Specific number of codebooks for Encodec
        self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates dummy `audio_codes` tensor for Encodec models.

        Args:
            input_name (`str`): Name of the input tensor (e.g., "audio_codes").
            framework (`str`, defaults to `"pt"`): The framework for tensor generation.
            int_dtype (`str`, defaults to `"int64"`): Data type for integer tensors.
            float_dtype (`str`, defaults to `"fp32"`): Data type for floating-point tensors.

        Returns:
            A randomly generated tensor for audio codes.
        """
        if input_name == "audio_codes":
            # Sequence length here is mapped to the number of audio codes
            shape = [1, self.batch_size, self.num_codebooks, self.sequence_length]
        else:
            raise ValueError(f"Unsupported input {input_name} for DummyEncodecInputGenerator")

        return self.random_int_tensor(
            shape=shape,
            min_value=0,
            max_value=50,  # Range specific to audio codes.
            framework=framework,
            dtype=int_dtype,
        )


class DummyIntGenerator(DummyInputGenerator):
    """
    Generates dummy integer inputs, such as `pad_token_id` and `max_length`, for models.

    Args:
        task (`str`): The task requiring integer inputs.
        normalized_config (`NormalizedTextConfig`): Model configuration with text-related attributes.
    """

    SUPPORTED_INPUT_NAMES = (
        "pad_token_id",
        "max_length",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        **kwargs,
    ):
        pass  # Initialization can be skipped for these simple inputs

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        """
        Generates simple dummy integer tensors for inputs like `pad_token_id` and `max_length`.

        Args:
            input_name (`str`): Name of the integer input to generate.
            framework (`str`, defaults to `"pt"`): The framework for tensor generation.
            int_dtype (`str`, defaults to `"int64"`): Data type for integer tensors.

        Returns:
            A single integer tensor of shape `(1,)`.
        """
        return self.random_int_tensor(shape=(1,), min_value=20, max_value=22, framework=framework, dtype=int_dtype)
