"""
TensorFlow Lite (TFLite) configuration classes for specific model architectures.

This module defines TFLite configuration classes for different types of model architectures, such as text encoders 
and vision models. These classes inherit from the `TFLiteConfig` base class and specialize the configuration 
and input generation process for specific model types.

Key Classes:
- `TextEncoderTFliteConfig`: Handles the configuration for text-based encoder models, specifying necessary axes 
  such as batch size and sequence length. It also sets up the dummy input generators for text-based models.
- `VisionTFLiteConfig`: Manages the configuration for vision-based models, specifying required axes like 
  batch size, number of channels, width, and height. It includes dummy input generators tailored for vision models.

These configuration classes make it easier to define and export models with the correct input/output specifications, 
handling essential parameters such as mandatory input axes and providing suitable input generators for their 
respective architectures.

"""


from ...utils import DummyTextInputGenerator, DummyVisionInputGenerator, logging
from .base import TFLiteConfig


logger = logging.get_logger(__name__)


class TextEncoderTFliteConfig(TFLiteConfig):
    """
    Handles encoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
    MANDATORY_AXES = ("batch_size", "sequence_length", ("multiple-choice", "num_choices"))


class VisionTFLiteConfig(TFLiteConfig):
    """
    Handles vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MANDATORY_AXES = ("batch_size", "num_channels", "width", "height")
