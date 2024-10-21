"""
The `ORTSeq2SeqTrainingArguments` class extends both `Seq2SeqTrainingArguments` from 🤗 Transformers and 
`ORTTrainingArguments` to provide a set of arguments specific to training sequence-to-sequence models 
with ONNX Runtime optimizations.

Main Features:
--------------
- **Sequence-to-Sequence Training**: Inherits all relevant arguments for sequence-to-sequence model training from 
  `Seq2SeqTrainingArguments`, such as arguments related to beam search, maximum generation length, and evaluation 
  metrics.
- **ONNX Runtime Integration**: Adds support for using ONNX Runtime’s optimizers and training methods through 
  `ORTTrainingArguments`, enabling faster training and improved resource utilization, particularly for large models.
- **Custom Optimizers**: Allows the selection of optimizers from both Transformers (e.g., `adamw_hf`, `adafactor`) 
  and ONNX Runtime (e.g., `adamw_ort_fused`).

Attributes:
-----------
- **optim** (`str` or `ORTOptimizerNames` or `OptimizerNames`, optional, defaults to `"adamw_hf"`):
    The optimizer to use for training. This can include optimizers provided by 🤗 Transformers (e.g., `adamw_hf`, 
    `adafactor`) or those implemented by ONNX Runtime (e.g., `adamw_ort_fused`).

Usage Example:
--------------
```python
from optimum.onnxruntime import ORTSeq2SeqTrainingArguments

# Define training arguments
training_args = ORTSeq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    optim="adamw_ort_fused",
)
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments

from .training_args import ORTTrainingArguments


@dataclass
class ORTSeq2SeqTrainingArguments(Seq2SeqTrainingArguments, ORTTrainingArguments):
    """
    Parameters:
        optim (`str` or [`training_args.ORTOptimizerNames`] or [`transformers.training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
            The optimizer to use, including optimizers in Transformers: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor. And optimizers implemented by ONNX Runtime: adamw_ort_fused.
    """

    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
