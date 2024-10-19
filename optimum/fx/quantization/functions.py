"""
This module provides transformers-compatible quantization functionalities built on top of PyTorch's `quantize_fx` API.
It enables various quantization operations, including fusing operations and preparing models for quantization and 
Quantization-Aware Training (QAT).

The key features include:

1. **QuantizationTracer**:
    - A specialized `HFTracer` that adds support for transformers models in quantization workflows.
    - It is responsible for tracing the model's computation graph, ensuring transformers' specific behaviors, 
      and handling skipped modules and classes during the tracing process.
    - It also stores module scope information for each traced node, which is useful during the quantization process.

2. **fuse_fx**:
    - A wrapper around `torch.quantization.quantize_fx.fuse_fx` that ensures compatibility with HuggingFace transformers models.
    - Fuses layers like Convolution + BatchNorm to reduce the number of operations during inference and improve efficiency.
    - If the model is a transformers `PreTrainedModel`, it traces the model first before applying the fusion.

3. **prepare_fx**:
    - A wrapper around `torch.quantization.quantize_fx.prepare_fx` for transformers models.
    - Prepares a model for static quantization by inserting observers into the model, which collect statistics for later quantization.
    - Supports custom configurations, equalization techniques, and backend-specific configurations for quantization.
    - Compatible with both HuggingFace `PreTrainedModel` and `torch.fx.GraphModule`.

4. **prepare_qat_fx**:
    - A wrapper around `torch.quantization.quantize_fx.prepare_qat_fx` for transformers models.
    - Prepares a model for Quantization-Aware Training (QAT), enabling it to simulate quantized inference during the training phase.
    - Supports customization, backend-specific configurations, and input name specifications.
    - Compatible with HuggingFace `PreTrainedModel` and `torch.fx.GraphModule`.

### Key Components:
- **QuantizationTracer**: Traces the transformers models while preserving model-specific behaviors.
- **fuse_fx**: Fuses supported operations (e.g., Conv2d + BatchNorm2d) for more efficient inference.
- **prepare_fx**: Prepares a model for post-training static quantization.
- **prepare_qat_fx**: Prepares a model for Quantization-Aware Training (QAT).

### Usage:
These functions enable the integration of quantization workflows into transformers models, improving their efficiency 
without the need to modify the core architecture. They are instrumental in reducing the computational load 
and memory footprint of models during inference, especially in edge or resource-constrained environments.

"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.fx.node import Argument, Node, Target
from torch.nn.intrinsic import _FusedModule
from torch.quantization.fx.graph_module import GraphModule, ObservedGraphModule
from torch.quantization.quantize_fx import Scope, ScopeContextManager
from torch.quantization.quantize_fx import fuse_fx as orig_fuse_fx
from torch.quantization.quantize_fx import prepare_fx as orig_prepare_fx
from torch.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx
from transformers import PreTrainedModel
from transformers.utils.fx import HFTracer, check_if_model_is_supported, get_concrete_args, symbolic_trace

from ..utils import check_if_available


if TYPE_CHECKING:
    from torch.fx import Graph


class QuantizationTracer(HFTracer):
    """
    Transformers compatible version of torch.quantization.quantize_fx.QuantizationTracer.
    This tracer is used internally to prepare the model for quantization.
    """

    specialized_concrete_args: Optional[Dict[str, Any]] = None

    def __init__(self, skipped_module_names: List[str], skipped_module_classes: List[Callable]):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}

    def is_leaf_module(self, module: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (module.__module__.startswith("torch.nn") and not isinstance(module, torch.nn.Sequential))
            or module_qualified_name in self.skipped_module_names
            or type(module) in self.skipped_module_classes
            or isinstance(module, _FusedModule)
            or super().is_leaf_module(module, module_qualified_name)
        )

    def call_module(
        self, module: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        module_qualified_name = self.path_of_module(module)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, module, module_qualified_name):
            return super().call_module(module, forward, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (self.scope.module_path, self.scope.module_type)
        return node

    def trace(self, root: "PreTrainedModel", concrete_args: Optional[Dict[str, Any]] = None) -> "Graph":
        if concrete_args is None and self.specialized_concrete_args is not None:
            concrete_args = self.specialized_concrete_args
        return super().trace(root, concrete_args=concrete_args)


def specialized_quantization_tracer_creator(concrete_args: Dict[str, Any]) -> Type:
    """Creates a QuantizationTracer-like class specifying concrete_args as a class attribute."""
    return type("QuantizationTracer", (QuantizationTracer,), {"specialized_concrete_args": concrete_args})


@check_if_available
def fuse_fx(
    model: Union[PreTrainedModel, GraphModule],
    fuse_custom_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    check: bool = True,
) -> GraphModule:
    """
    Transformers models compatible version of torch.quantization.quantize_fx.fuse_fx, refer to the
    [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.fuse_fx.html) for
    more details.

    Args:
        model (`PreTrainedModel` or `torch.fx.GraphModule`):
            The model to fuse.
        fuse_custom_config_dict (`Dict[str, Any]`, *optional*):
            Dictionary for custom configurations for fuse_fx, refer to the PyTorch documentation for more details.
        input_names (`List[str]`, *optional*):
            The input names of the model, only used to trace if model is a PreTrainedModel. This is not needed if model
            is already a GraphModule.
        check (`bool`, *optional*, defaults to `True`):
            If True, a check is done to verify that the model can be traced.

    Returns:
        `torch.fx.GraphModule`: A GraphModule with the fused modules.

    Example:

        ```python
        >>> from transformers import BertModel
        >>> from optimum.fx.quantization import fuse_fx

        >>> model = BertModel.from_pretrained("bert-base-uncased")
        >>> model = fuse_fx(model)
        ```
    """
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        model = symbolic_trace(model, input_names, disable_check=not check)
    orig_symbolic_trace = torch.fx.symbolic_trace
    torch.fx.symbolic_trace = lambda x: x
    graph_module = orig_fuse_fx(model, fuse_custom_config_dict=fuse_custom_config_dict)
    torch.fx.symbolic_trace = orig_symbolic_trace
    return graph_module


@check_if_available
def prepare_fx(
    model: Union[PreTrainedModel, GraphModule],
    qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    equalization_qconfig_dict: Optional[Dict[str, Any]] = None,
    backend_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    check: bool = True,
) -> ObservedGraphModule:
    """
    Transformers models compatible version of torch.quantization.quantize_fx.prepare_fx, refer to the [PyTorch
    documentation](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.prepare_fx.html#torch.quantization.quantize_fx.prepare_fx)
    for more details.

    Args:
        model (`PreTrainedModel` or `torch.fx.GraphModule`):
            The model to prepare, must be in eval mode.
        qconfig_dict (`Any`):
             Dictionary specifying how and which modules and operations should be quantized, refer to the PyTorch
             documentation for more details.
        prepare_custom_config_dict (`Dict[str, Any]`, *optional*):
            Customization configuration dictionary for quantization tool, refer to the PyTorch documentation for more
            details.
        equalization_qconfig_dict (`Dict[str, Any]`, *optional*):
            A dictionary with a similar structure as qconfig_dict except it will contain configurations specific to
            equalization techniques such as input-weight equalization.
        backend_config_dict (`Dict[str, Any]`, *optional*):
            A dictionary that specifies how operators are quantized in a backend, this includes how the operators are
            observed, supported fusion patterns, how quantize/dequantize ops are inserted, supported dtypes etc.
            The structure of the dictionary is still WIP and will change in the future, please don't use right now.
        input_names (`List[str]`, *optional*):
            The input names of the model, only used to trace if model is a PreTrainedModel. This is not needed if model
            is already a GraphModule.
        check (`bool`, *optional*, defaults to `True`):
            If True, a check is done to verify that the model can be traced.

    Returns:
        `torch.quantization.fx.graph_module.ObservedGraphModule`: An ObservedGraphModule ready for calibration.

    Example:

        ```python
        >>> import torch
        >>> from torch.ao.quantization import get_default_qconfig
        >>> from transformers import BertForSequenceClassification
        >>> from optimum.fx.quantization import prepare_fx

        >>> model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
        >>> model.eval()  # doctest: +IGNORE_RESULT

        >>> # Prepare the model
        >>> qconfig = get_default_qconfig('fbgemm')
        >>> qconfig_dict = {"": qconfig}
        >>> prepared_model = prepare_fx(model, qconfig_dict)
        ```
    """
    if check:
        check_if_model_is_supported(model)
    tracer_cls = QuantizationTracer
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        tracer_cls = specialized_quantization_tracer_creator(get_concrete_args(model, input_names))
    orig_quantization_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer
    torch.ao.quantization.quantize_fx.QuantizationTracer = tracer_cls
    graph_module = orig_prepare_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        equalization_qconfig_dict=equalization_qconfig_dict,
        backend_config_dict=backend_config_dict,
    )
    torch.ao.quantization.quantize_fx.QuantizationTracer = orig_quantization_tracer
    return graph_module


@check_if_available
def prepare_qat_fx(
    model: Union[PreTrainedModel, GraphModule],
    qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    backend_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    check: bool = True,
) -> ObservedGraphModule:
    """
    Transformers models compatible version of torch.quantization.quantize_fx.prepare_qat_fx, refer to the [PyTorch
    documentation](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.prepare_qat_fx.html#torch.quantization.quantize_fx.prepare_qat_fx)
    for more details.

    Args:
        model (`PreTrainedModel` or `torch.fx.GraphModule`):
            The model to prepare, must be in train mode.
        qconfig_dict (`Any`):
             Dictionary specifying how and which modules and operations should be quantized, refer to the PyTorch
             documentation for more details.
        prepare_custom_config_dict (`Dict[str, Any]`, *optional*):
            Customization configuration dictionary for quantization tool, refer to the PyTorch documentation for more
            details.
        backend_config_dict (`Dict[str, Any]`, *optional*):
            A dictionary that specifies how operators are quantized in a backend, this includes how the operators are
            observed, supported fusion patterns, how quantize/dequantize ops are inserted, supported dtypes etc.
            The structure of the dictionary is still WIP and will change in the future, please don't use right now.
        input_names (`List[str]`, *optional*):
            The input names of the model, only used to trace if model is a PreTrainedModel. This is not needed if model
            is already a GraphModule.
        check (`bool`, *optional*, defaults to `True`):
            If True, a check is done to verify that the model can be traced.

    Returns:
        `torch.quantization.fx.graph_module.ObservedGraphModule`: An ObservedGraphModule ready for QAT.

    Example:

        ```python
        >>> import torch
        >>> from torch.ao.quantization import get_default_qat_qconfig
        >>> from transformers import BertForSequenceClassification
        >>> from optimum.fx.quantization import prepare_qat_fx

        >>> model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
        >>> model.train()  # doctest: +IGNORE_RESULT

        >>> # Prepare the model
        >>> qconfig = get_default_qat_qconfig('fbgemm')
        >>> qconfig_dict = {"": qconfig}
        >>> prepared_model = prepare_qat_fx(model, qconfig_dict)
        ```
    """
    if check:
        check_if_model_is_supported(model)
    tracer_cls = QuantizationTracer
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        tracer_cls = specialized_quantization_tracer_creator(get_concrete_args(model, input_names))
    orig_quantization_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer
    torch.ao.quantization.quantize_fx.QuantizationTracer = tracer_cls
    graph_module = orig_prepare_qat_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        backend_config_dict=backend_config_dict,
    )
    torch.ao.quantization.quantize_fx.QuantizationTracer = orig_quantization_tracer
    return graph_module
