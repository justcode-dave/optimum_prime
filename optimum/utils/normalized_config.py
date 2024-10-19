"""
normalized_config.py

This module defines normalization classes for model configurations, enabling access to attributes of
`PretrainedConfig` instances in a standardized and general way. These classes help in handling models 
that may have differing attribute names but need to be accessed in a consistent manner. It includes specific 
configuration classes for text, vision, and combined text-vision models, among others.

Key Classes:
------------
- `NormalizedConfig`: A base class for normalizing configuration attributes, providing access to config fields in a 
  general way regardless of the underlying model.
- `NormalizedTextConfig`: Normalizes configuration for text-based models.
- `NormalizedVisionConfig`: Normalizes configuration for vision-based models.
- `NormalizedTextAndVisionConfig`: Handles normalization for models that combine both text and vision inputs.
- `NormalizedEncoderDecoderConfig`: Handles models with both encoder and decoder architectures, such as seq2seq models.
"""

import functools  # Utility for higher-order functions.
from typing import Callable, Dict, Type, Union  # Types for type hinting.

from transformers import PretrainedConfig  # Importing base configuration from Hugging Face transformers.


class NormalizedConfig:
    """
    Handles the normalization of [`PretrainedConfig`] attribute names, allowing attributes to be accessed in a general way
    regardless of the underlying model configuration.

    Attributes:
        config (`PretrainedConfig` or `Dict`):
            The configuration object to normalize.
    
    Args:
        config (`Union[PretrainedConfig, Dict]`):
            The config or dictionary to normalize.
        allow_new (`bool`, optional):
            If True, allows setting new attributes that do not already exist in the class. Defaults to False.
        **kwargs: 
            Additional keyword arguments representing attributes to normalize.
    """

    def __init__(self, config: Union[PretrainedConfig, Dict], allow_new: bool = False, **kwargs):
        self.config = config  # Store the provided configuration.
        # Loop through kwargs and set attributes based on the allow_new flag.
        for key, value in kwargs.items():
            if allow_new or hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                raise AttributeError(
                    f"{self.__class__} has no attribute {key}. Set allow_new=True to add a new attribute."
                )

    @classmethod
    def with_args(cls, allow_new: bool = False, **kwargs) -> Callable[[PretrainedConfig], "NormalizedConfig"]:
        """
        Creates a partially applied version of the class constructor, presetting the `allow_new` and any additional attributes.

        Args:
            allow_new (`bool`, optional): 
                If True, allows setting new attributes that do not already exist in the class. Defaults to False.
            **kwargs: 
                Additional attributes to be preset in the normalized configuration.

        Returns:
            Callable: A callable that creates a `NormalizedConfig` instance with the given arguments.
        """
        return functools.partial(cls, allow_new=allow_new, **kwargs)

    def __getattr__(self, attr_name):
        """
        Recursively retrieves the attribute from the config using normalized attribute names.

        Args:
            attr_name (`str`):
                The attribute name to retrieve, supporting dot notation for nested attributes.
        
        Returns:
            The attribute value if found.
        
        Raises:
            AttributeError: If the attribute cannot be found in the normalized config.
        """
        if attr_name == "config":
            return super().__getattr__(attr_name)

        try:
            attr_name = super().__getattribute__(attr_name.upper())  # Normalize attribute name by converting to uppercase.
        except AttributeError:
            pass

        attr_name = attr_name.split(".")  # Handle dot notation for nested attributes.
        leaf_attr_name = attr_name[-1]  # Get the final attribute name.
        config = self.config  # Start from the base config.

        # Traverse through the nested attributes.
        for attr in attr_name[:-1]:
            config = getattr(config, attr)

        attr = getattr(config, leaf_attr_name, None)  # Retrieve the final attribute.

        # Fallback to attribute map if not found in the config.
        if attr is None:
            attribute_map = getattr(self.config, "attribute_map", {})
            attr = getattr(self.config, attribute_map.get(leaf_attr_name, ""), None)

        if attr is None:
            raise AttributeError(f'Could not find the attribute named "{leaf_attr_name}" in the normalized config.')
        return attr

    def has_attribute(self, attr_name):
        """
        Checks whether the given attribute exists in the normalized configuration.

        Args:
            attr_name (`str`): 
                The attribute name to check.
        
        Returns:
            bool: True if the attribute exists, False otherwise.
        """
        try:
            self.__getattr__(attr_name)
        except AttributeError:
            return False
        return True


class NormalizedTextConfig(NormalizedConfig):
    """
    Normalizes configurations for text-based models, mapping common model attributes to standardized names.
    """
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_LAYERS = "num_hidden_layers"
    NUM_ATTENTION_HEADS = "num_attention_heads"
    EOS_TOKEN_ID = "eos_token_id"


class NormalizedTextConfigWithGQA(NormalizedTextConfig):
    """
    Normalizes configurations for models using Grouped Query Attention (GQA).
    """
    NUM_KEY_VALUE_HEADS = "num_key_value_heads"


class NormalizedSeq2SeqConfig(NormalizedTextConfig):
    """
    Normalizes configurations for Seq2Seq models with distinct encoder and decoder settings.
    """
    ENCODER_NUM_LAYERS = NormalizedTextConfig.NUM_LAYERS
    DECODER_NUM_LAYERS = NormalizedTextConfig.NUM_LAYERS
    ENCODER_NUM_ATTENTION_HEADS = NormalizedTextConfig.NUM_ATTENTION_HEADS
    DECODER_NUM_ATTENTION_HEADS = NormalizedTextConfig.NUM_ATTENTION_HEADS


class NormalizedVisionConfig(NormalizedConfig):
    """
    Normalizes configurations for vision-based models, mapping common vision model attributes.
    """
    IMAGE_SIZE = "image_size"
    NUM_CHANNELS = "num_channels"
    INPUT_SIZE = "input_size"


class NormalizedSegformerConfig(NormalizedVisionConfig):
    """
    Normalizes configurations for Segformer, a vision-based model used for segmentation tasks.
    
    Note:
        If the attribute is a list, the value 0 is returned, meaning the optimizer will infer the value.
    """
    NUM_ATTENTION_HEADS = "num_attention_heads"
    HIDDEN_SIZE = "hidden_sizes"

    def __getattr__(self, attr_name):
        """
        Custom `__getattr__` for Segformer configurations, returns 0 if the attribute is a list.

        Args:
            attr_name (`str`): 
                The attribute name to retrieve.
        
        Returns:
            The attribute value or 0 if it's a list.
        """
        attr_value = super().__getattr__(attr_name)
        if isinstance(attr_value, list):
            attr_value = 0
        return attr_value


class NormalizedTextAndVisionConfig(NormalizedTextConfig, NormalizedVisionConfig):
    """
    Handles models with both text and vision components, like multimodal models.
    """
    TEXT_CONFIG = None
    VISION_CONFIG = None

    def __getattr__(self, attr_name):
        """
        Retrieves attributes from either the text or vision configuration based on which config contains the attribute.

        Args:
            attr_name (`str`):
                The attribute name to retrieve.
        
        Returns:
            The attribute value if found.
        
        Raises:
            AttributeError: If the attribute cannot be found in either the text or vision config.
        """
        if self.TEXT_CONFIG is not None and attr_name.upper() in dir(NormalizedTextConfig):
            attr_name = f"{self.TEXT_CONFIG}.{attr_name}"
        elif self.VISION_CONFIG is not None and attr_name.upper() in dir(NormalizedVisionConfig):
            attr_name = f"{self.VISION_CONFIG}.{attr_name}"
        return super().__getattr__(attr_name)


# Additional specific configurations for various model architectures.
Pix2StructNormalizedTextConfig = NormalizedTextAndVisionConfig.with_args(
    text_config="text_config", vision_config="vision_config"
)

class NormalizedEncoderDecoderConfig(NormalizedConfig):
    """
    Handles models with both encoder and decoder components, such as seq2seq models.
    """
    ENCODER_NORMALIZED_CONFIG_CLASS = None
    DECODER_NORMALIZED_CONFIG_CLASS = None

    def __getattr__(self, attr_name):
        """
        Retrieves attributes from either the encoder or decoder configuration.

        Args:
            attr_name (`str`):
                The attribute name to retrieve.
        
        Returns:
            The attribute value if found.
        """
        if self.ENCODER_NORMALIZED_CONFIG_CLASS is not None and attr_name.upper() in dir(
            self.ENCODER_NORMALIZED_CONFIG_CLASS
        ):
            return self.ENCODER_NORMALIZED_CONFIG_CLASS.__getattr__(attr_name)
        if self.DECODER_NORMALIZED_CONFIG_CLASS is not None and attr_name.upper() in dir(
            self.DECODER_NORMALIZED_CONFIG_CLASS
        ):
            return self.DECODER_NORMALIZED_CONFIG_CLASS.__getattr__(attr_name)

        return super().__getattr__(attr_name)

# Other model-specific normalized configurations
BartLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="encoder_attention_heads",
    hidden_size="d_model",
)

GPT2LikeNormalizedTextConfig = NormalizedTextConfig.with_args(num_attention_heads="n_head", hidden_size="n_embd")
T5LikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="num_heads",
    hidden_size="d_model",
)
MPTNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="n_heads", hidden_size="d_model", num_layers="n_layers"
)
GPTBigCodeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="n_head", hidden_size="n_embd", num_layers="n_layer"
)

# Configuration for Whisper-like models
WhisperLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    hidden_size="d_model",
)

# Configuration for TrOCR-like models
TrOCRLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_layers="decoder_layers",
    num_attention_heads="decoder_attention_heads",
    hidden_size="hidden_size",
)

# Configuration for Speech-to-Text-like models
SpeechToTextLikeNormalizedTextConfig = NormalizedSeq2SeqConfig.with_args(
    decoder_num_layers="decoder_layers",
    num_layers="decoder_layers",
    input_features_per_channel="input_feat_per_channel",
    allow_new=True,
)

"""
normalized_config.py

This module contains classes and utilities for handling the normalization of model configurations for a wide variety
of model types. It provides normalized access to model configuration attributes regardless of the underlying model's
specific configuration names. This is particularly useful in scenarios where models have different attribute names 
(e.g., number of layers or hidden size) but need to be accessed uniformly.

Key Classes:
------------
- `NormalizedConfigManager`: A manager that provides access to the appropriate normalized configuration class for 
  different model types, ensuring standardized access to attributes like the number of attention heads or hidden size.
"""

class NormalizedConfigManager:
    """
    A manager class that contains all the information needed by ONNX Runtime optimization for a given model type.

    Attributes:
        _conf (`Dict[str, Type[NormalizedConfig]]`):
            A dictionary mapping each supported model type to its corresponding normalized configuration class.
            The normalized configuration classes allow consistent access to attributes like the number of attention heads
            and hidden size, even if they differ between models.
    """

    # TODO: missing normalized configs (currently not useful)
    # The following models are currently not included in the configuration manager, but may need support in future:
    # ['beit', 'clip', 'convbert', 'convnext', 'convnextv2', 'data2vec-text', 'data2vec-vision', 'detr', 'flaubert',
    #  'groupvit', 'ibert', 'layoutlm', 'layoutlmv3', 'levit', 'mobilebert', 'mobilevit', 'owlv2', 'owlvit', 'perceiver',
    #  'roformer', 'squeezebert', 'table-transformer']

    # Contribution note: Please add new models in alphabetical order
    _conf = {
        "albert": NormalizedTextConfig,  # Standard text-based model
        "bart": BartLikeNormalizedTextConfig,  # Seq2Seq model similar to BART
        "bert": NormalizedTextConfig,  # Standard BERT model
        "blenderbot": BartLikeNormalizedTextConfig,  # Blenderbot inherits from BART
        "blenderbot-small": BartLikeNormalizedTextConfig,
        "bloom": NormalizedTextConfig.with_args(num_layers="n_layer"),  # BLOOM model with custom layer config
        "falcon": NormalizedTextConfig,
        "camembert": NormalizedTextConfig,
        "codegen": GPT2LikeNormalizedTextConfig,
        "cvt": NormalizedVisionConfig,
        "deberta": NormalizedTextConfig,
        "deberta-v2": NormalizedTextConfig,
        "deit": NormalizedVisionConfig,
        "distilbert": NormalizedTextConfig.with_args(num_attention_heads="n_heads", hidden_size="dim"),
        "donut-swin": NormalizedVisionConfig,
        "electra": NormalizedTextConfig,
        "encoder-decoder": NormalizedEncoderDecoderConfig,  # Encoder-decoder model (Seq2Seq)
        "gemma": NormalizedTextConfigWithGQA,  # Text model with Grouped Query Attention
        "gpt2": GPT2LikeNormalizedTextConfig,  # GPT-2 model
        "gpt-bigcode": GPTBigCodeNormalizedTextConfig,  # GPT-BigCode model
        "gpt-neo": NormalizedTextConfig.with_args(num_attention_heads="num_heads"),
        "gpt-neox": NormalizedTextConfig,
        "gptj": GPT2LikeNormalizedTextConfig,
        "imagegpt": GPT2LikeNormalizedTextConfig,
        "llama": NormalizedTextConfigWithGQA,
        "longt5": T5LikeNormalizedTextConfig,  # Long-form T5 model
        "marian": BartLikeNormalizedTextConfig,
        "markuplm": NormalizedTextConfig,
        "mbart": BartLikeNormalizedTextConfig,
        "mistral": NormalizedTextConfigWithGQA,
        "mixtral": NormalizedTextConfigWithGQA,
        "mpnet": NormalizedTextConfig,
        "mpt": MPTNormalizedTextConfig,  # MosaicML's MPT model
        "mt5": T5LikeNormalizedTextConfig,  # Multilingual T5
        "m2m-100": BartLikeNormalizedTextConfig,  # Many-to-Many multilingual model
        "nystromformer": NormalizedTextConfig,
        "opt": NormalizedTextConfig,  # Open Pre-trained Transformer model
        "pegasus": BartLikeNormalizedTextConfig,  # Pegasus model for text summarization
        "pix2struct": Pix2StructNormalizedTextConfig,  # Pix2Struct multimodal model
        "phi": NormalizedTextConfig,
        "phi3": NormalizedTextConfigWithGQA,
        "phi3small": NormalizedTextConfigWithGQA,
        "poolformer": NormalizedVisionConfig,
        "regnet": NormalizedVisionConfig,
        "resnet": NormalizedVisionConfig,
        "roberta": NormalizedTextConfig,
        "segformer": NormalizedSegformerConfig,  # Segformer segmentation model
        "speech-to-text": SpeechToTextLikeNormalizedTextConfig,  # Speech-to-text model
        "splinter": NormalizedTextConfig,
        "t5": T5LikeNormalizedTextConfig,  # T5 text-to-text model
        "trocr": TrOCRLikeNormalizedTextConfig,  # TrOCR for optical character recognition
        "vision-encoder-decoder": NormalizedEncoderDecoderConfig,
        "vit": NormalizedVisionConfig,  # Vision Transformer model
        "whisper": WhisperLikeNormalizedTextConfig,  # Whisper for speech recognition
        "xlm-roberta": NormalizedTextConfig,
        "yolos": NormalizedVisionConfig,
        "qwen2": NormalizedTextConfig,  # Qwen-2 model configuration
    }

    @classmethod
    def check_supported_model(cls, model_type: str):
        """
        Checks whether the provided model type is supported by the NormalizedConfigManager.

        Args:
            model_type (`str`): 
                The type of model to check.

        Raises:
            KeyError: If the model type is not supported by the NormalizedConfigManager.
        """
        if model_type not in cls._conf:
            model_types = ", ".join(cls._conf.keys())  # Get a list of all supported models.
            raise KeyError(
                f"{model_type} model type is not supported yet in NormalizedConfig. Only {model_types} are supported. "
                f"If you want to support {model_type}, please propose a PR or open up an issue."
            )

    @classmethod
    def get_normalized_config_class(cls, model_type: str) -> Type:
        """
        Retrieves the normalized configuration class for the provided model type.

        Args:
            model_type (`str`): 
                The type of model whose normalized configuration class should be retrieved.

        Returns:
            `Type`: The normalized configuration class for the specified model type.

        Raises:
            KeyError: If the model type is not supported.
        """
        model_type = model_type.replace("_", "-")  # Normalize underscores in model type names.
        cls.check_supported_model(model_type)  # Check if the model is supported.
        return cls._conf[model_type]  # Return the corresponding normalized config class.
