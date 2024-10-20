"""
Model-specific TensorFlow Lite configurations.

This module defines TFLite configuration classes for various models such as BERT, DistilBERT, and ResNet. Each class 
inherits from a base configuration class (e.g., `TextEncoderTFliteConfig`, `VisionTFLiteConfig`) and specifies 
model-specific properties such as input names and supported quantization approaches.

Key Classes:
- `BertTFLiteConfig`: Defines the configuration for exporting BERT-based models to TFLite format, including supported 
   inputs like `input_ids`, `attention_mask`, and `token_type_ids`.
- `DistilBertTFLiteConfig`: Configuration for DistilBERT models, with reduced input requirements (no `token_type_ids`).
- `ResNetTFLiteConfig`: Configuration for exporting ResNet-based vision models, specifying `pixel_values` as the input.
- Additional classes for models such as `Electra`, `RoFormer`, `Deberta`, `XLMRoberta`, and more, each with 
   their unique adjustments and quantization support.

This module also handles limitations with certain quantization approaches (e.g., `INT8x16` not supported for certain 
models due to unsupported operations like `CAST` and `NEG`) and custom input specifications for each model type.
"""



from typing import List

from ...utils.normalized_config import NormalizedConfigManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig


class BertTFLiteConfig(TextEncoderTFliteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    # INT8x16 not supported because of the CAST op.
    SUPPORTED_QUANTIZATION_APPROACHES = (
        QuantizationApproach.INT8_DYNAMIC,
        QuantizationApproach.INT8,
        QuantizationApproach.FP16,
    )

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


class AlbertTFLiteConfig(BertTFLiteConfig):
    pass


class ConvBertTFLiteConfig(BertTFLiteConfig):
    pass


class ElectraTFLiteConfig(BertTFLiteConfig):
    pass


class RoFormerTFLiteConfig(BertTFLiteConfig):
    # INT8x16 not supported because of the CAST and NEG ops.
    pass


class MobileBertTFLiteConfig(BertTFLiteConfig):
    pass


class XLMTFLiteConfig(BertTFLiteConfig):
    pass


class DistilBertTFLiteConfig(BertTFLiteConfig):
    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]


class MPNetTFLiteConfig(DistilBertTFLiteConfig):
    pass


class RobertaTFLiteConfig(DistilBertTFLiteConfig):
    pass


class CamembertTFLiteConfig(DistilBertTFLiteConfig):
    pass


class FlaubertTFLiteConfig(BertTFLiteConfig):
    pass


class XLMRobertaTFLiteConfig(DistilBertTFLiteConfig):
    SUPPORTED_QUANTIZATION_APPROACHES = {
        "default": BertTFLiteConfig.SUPPORTED_QUANTIZATION_APPROACHES,
        # INT8 quantization on question-answering is producing various errors depending on the model size and
        # calibration dataset:
        # - GATHER index out of bound
        # - (CUMSUM) failed to invoke
        # TODO => Needs to be investigated.
        "question-answering": (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.FP16),
    }


# TODO: no TensorFlow implementation, but a Jax implementation is available.
# Support the export once the Jax export to TFLite is more mature.
# class BigBirdTFLiteConfig(DistilBertTFLiteConfig):
#     pass


class DebertaTFLiteConfig(BertTFLiteConfig):
    # INT8 quantization is producing a segfault error.
    SUPPORTED_QUANTIZATION_APPROACHES = (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.FP16)

    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


class DebertaV2TFLiteConfig(DebertaTFLiteConfig):
    pass


class ResNetTFLiteConfig(VisionTFLiteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("resnet")

    @property
    def inputs(self) -> List[str]:
        return ["pixel_values"]
