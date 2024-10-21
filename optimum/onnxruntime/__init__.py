"""
This module initializes the ONNX Runtime (ORT) extension for the Hugging Face Transformers library. 
It provides lazy imports for key components and configurations required for model optimization, 
quantization, and export functionalities using ONNX and ONNX Runtime.

The main components included are:
- Configuration utilities for calibration, quantization, and optimization (`ORTConfig`, `QuantizationConfig`, etc.).
- ONNX Runtime (ORT) model classes for various tasks, such as sequence classification, token classification, image 
  classification, and others, across tasks like NLP, Vision, and Speech (`ORTModel`, `ORTModelForSequenceClassification`, 
  etc.).
- Diffusion model pipelines for Stable Diffusion and similar models if the `diffusers` library is available.
- Optimization and quantization utilities (`ORTOptimizer`, `ORTQuantizer`) for model compression and performance 
  improvements.
- Training utilities including arguments, trainers, and helpers for integrating ONNX Runtime with the Hugging Face 
  training loops (`ORTTrainer`, `ORTTrainingArguments`).

The module uses lazy imports to optimize loading times, only loading components when they are specifically requested. 
It also handles optional dependencies like `diffusers` by providing alternatives if the dependency is unavailable.

Note:
    If `diffusers` is not available, dummy objects are imported to maintain compatibility with codebases that don't 
    require Stable Diffusion pipelines.

Available Components:
----------------------
- Configuration: Handles various configuration needs, including calibration, quantization, and optimization.
- Modeling: ORT model classes for tasks in NLP, Vision, and Speech.
- Optimization: Tools for optimizing models for deployment.
- Quantization: Tools for quantizing models for performance improvements.
- Training: Classes and arguments for training models with ONNX Runtime integration.
- Diffusion Pipelines: ORT-based pipelines for Stable Diffusion and other diffusion models (if `diffusers` is available).
"""

from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule

from ..utils import is_diffusers_available


_import_structure = {
    "configuration": [
        "CalibrationConfig",
        "AutoCalibrationConfig",
        "QuantizationMode",
        "AutoQuantizationConfig",
        "OptimizationConfig",
        "AutoOptimizationConfig",
        "ORTConfig",
        "QuantizationConfig",
    ],
    "modeling_ort": [
        "ORTModel",
        "ORTModelForAudioClassification",
        "ORTModelForAudioFrameClassification",
        "ORTModelForAudioXVector",
        "ORTModelForCustomTasks",
        "ORTModelForCTC",
        "ORTModelForFeatureExtraction",
        "ORTModelForImageClassification",
        "ORTModelForMaskedLM",
        "ORTModelForMultipleChoice",
        "ORTModelForQuestionAnswering",
        "ORTModelForSemanticSegmentation",
        "ORTModelForSequenceClassification",
        "ORTModelForTokenClassification",
        "ORTModelForImageToImage",
    ],
    "modeling_seq2seq": [
        "ORTModelForSeq2SeqLM",
        "ORTModelForSpeechSeq2Seq",
        "ORTModelForVision2Seq",
        "ORTModelForPix2Struct",
    ],
    "modeling_decoder": ["ORTModelForCausalLM"],
    "optimization": ["ORTOptimizer"],
    "quantization": ["ORTQuantizer"],
    "trainer": ["ORTTrainer"],
    "trainer_seq2seq": ["ORTSeq2SeqTrainer"],
    "training_args": ["ORTTrainingArguments"],
    "training_args_seq2seq": ["ORTSeq2SeqTrainingArguments"],
    "utils": [
        "ONNX_DECODER_NAME",
        "ONNX_DECODER_MERGED_NAME",
        "ONNX_DECODER_WITH_PAST_NAME",
        "ONNX_ENCODER_NAME",
        "ONNX_WEIGHTS_NAME",
        "ORTQuantizableOperator",
    ],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure[".utils.dummy_diffusers_objects"] = [
        "ORTStableDiffusionPipeline",
        "ORTStableDiffusionImg2ImgPipeline",
        "ORTStableDiffusionInpaintPipeline",
        "ORTStableDiffusionXLPipeline",
        "ORTStableDiffusionXLImg2ImgPipeline",
        "ORTStableDiffusionXLInpaintPipeline",
        "ORTLatentConsistencyModelPipeline",
        "ORTLatentConsistencyModelImg2ImgPipeline",
        "ORTPipelineForImage2Image",
        "ORTPipelineForInpainting",
        "ORTPipelineForText2Image",
        "ORTDiffusionPipeline",
    ]
else:
    _import_structure["modeling_diffusion"] = [
        "ORTStableDiffusionPipeline",
        "ORTStableDiffusionImg2ImgPipeline",
        "ORTStableDiffusionInpaintPipeline",
        "ORTStableDiffusionXLPipeline",
        "ORTStableDiffusionXLImg2ImgPipeline",
        "ORTStableDiffusionXLInpaintPipeline",
        "ORTLatentConsistencyModelImg2ImgPipeline",
        "ORTLatentConsistencyModelPipeline",
        "ORTPipelineForImage2Image",
        "ORTPipelineForInpainting",
        "ORTPipelineForText2Image",
        "ORTDiffusionPipeline",
    ]


# Direct imports for type-checking
if TYPE_CHECKING:
    from .configuration import ORTConfig, QuantizationConfig
    from .modeling_decoder import ORTModelForCausalLM
    from .modeling_ort import (
        ORTModel,
        ORTModelForAudioClassification,
        ORTModelForAudioFrameClassification,
        ORTModelForAudioXVector,
        ORTModelForCTC,
        ORTModelForCustomTasks,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForImageToImage,
        ORTModelForMaskedLM,
        ORTModelForMultipleChoice,
        ORTModelForQuestionAnswering,
        ORTModelForSemanticSegmentation,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    from .modeling_seq2seq import (
        ORTModelForPix2Struct,
        ORTModelForSeq2SeqLM,
        ORTModelForSpeechSeq2Seq,
        ORTModelForVision2Seq,
    )
    from .optimization import ORTOptimizer
    from .quantization import ORTQuantizer
    from .trainer import ORTTrainer
    from .trainer_seq2seq import ORTSeq2SeqTrainer
    from .training_args import ORTTrainingArguments
    from .training_args_seq2seq import ORTSeq2SeqTrainingArguments
    from .utils import (
        ONNX_DECODER_MERGED_NAME,
        ONNX_DECODER_NAME,
        ONNX_DECODER_WITH_PAST_NAME,
        ONNX_ENCODER_NAME,
        ONNX_WEIGHTS_NAME,
        ORTQuantizableOperator,
    )

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_diffusers_objects import (
            ORTDiffusionPipeline,
            ORTLatentConsistencyModelImg2ImgPipeline,
            ORTLatentConsistencyModelPipeline,
            ORTPipelineForImage2Image,
            ORTPipelineForInpainting,
            ORTPipelineForText2Image,
            ORTStableDiffusionImg2ImgPipeline,
            ORTStableDiffusionInpaintPipeline,
            ORTStableDiffusionPipeline,
            ORTStableDiffusionXLImg2ImgPipeline,
            ORTStableDiffusionXLInpaintPipeline,
            ORTStableDiffusionXLPipeline,
        )
    else:
        from .modeling_diffusion import (
            ORTDiffusionPipeline,
            ORTLatentConsistencyModelImg2ImgPipeline,
            ORTLatentConsistencyModelPipeline,
            ORTPipelineForImage2Image,
            ORTPipelineForInpainting,
            ORTPipelineForText2Image,
            ORTStableDiffusionImg2ImgPipeline,
            ORTStableDiffusionInpaintPipeline,
            ORTStableDiffusionPipeline,
            ORTStableDiffusionXLImg2ImgPipeline,
            ORTStableDiffusionXLInpaintPipeline,
            ORTStableDiffusionXLPipeline,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
