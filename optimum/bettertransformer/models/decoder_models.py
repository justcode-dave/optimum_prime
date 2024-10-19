"""
This module provides BetterTransformer-compatible attention layers for various decoder models such as GPT2, GPTJ, Bloom, Codegen, Bart, and others. 
These layers extend Hugging Face's original transformer model layers and implement optimizations for inference performance using BetterTransformer.

Each class wraps its corresponding decoder attention layer, adjusting attention mechanisms to leverage optimizations like scaled dot-product attention.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.models.codegen.modeling_codegen import CodeGenAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention
from transformers.models.marian.modeling_marian import MarianAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.pegasus.modeling_pegasus import PegasusAttention
from transformers.models.t5.modeling_t5 import T5Attention

from ...utils.import_utils import check_if_transformers_greater

# Conditional import based on the version of transformers
if check_if_transformers_greater("4.31"):
    from transformers.models.bark.modeling_bark import BarkSelfAttention
else:
    from ...utils.dummy_bettertransformer_objects import BarkSelfAttention

# Import wrapped scaled dot-product attention implementations
from .attention import (
    bark_wrapped_scaled_dot_product,
    bart_forward,
    bloom_forward,
    codegen_wrapped_scaled_dot_product,
    gpt2_wrapped_scaled_dot_product,
    gpt_neo_wrapped_scaled_dot_product,
    gptj_wrapped_scaled_dot_product,
    opt_forward,
    t5_forward,
)
from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    from transformers import PretrainedConfig


# GPT2 Attention Layer adapted for BetterTransformer
class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPT2Attention):
    _attn = gpt2_wrapped_scaled_dot_product  # Overriding attention mechanism with optimized implementation

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        # Set submodule attributes from the original layer
        submodules = ["c_proj", "c_attn", "attn_dropout", "resid_dropout", "bias", "masked_bias"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.is_cross_attention:
            setattr(self, "q_attn", getattr(layer, "q_attn"))
            self.original_layers_mapping["q_attn"] = "q_attn"

        self.downcast_qk = False  # Whether to downcast query/key projections
        self.dropout_prob_attn = config.attn_pdrop  # Attention dropout probability

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# GPTJ Attention Layer adapted for BetterTransformer
class GPTJAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTJAttention, nn.Module):
    _attn = gptj_wrapped_scaled_dot_product  # Overriding attention mechanism with optimized implementation

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        
        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        # Set submodule attributes from the original layer
        submodules = [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "attn_dropout",
            "resid_dropout",
            "scale_attn",
        ]

        # Conditional submodules based on transformers version
        if hasattr(layer, "embed_positions"):
            submodules.append("embed_positions")

        if hasattr(layer, "bias"):
            submodules.append("bias")
        if hasattr(layer, "masked_bias"):
            submodules.append("masked_bias")

        if hasattr(layer, "layer_idx"):
            submodules.append("layer_idx")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True  # Downcast query/key projections
        self.dropout_prob_attn = config.attn_pdrop  # Attention dropout probability

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# GPTNeoX Attention Layer adapted for BetterTransformer
class GPTNeoXAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoXAttention, nn.Module):
    _attn = gpt2_wrapped_scaled_dot_product  # Reusing the gpt2 attention wrapping logic

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        
        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.module_mapping = None
        submodules = ["rotary_emb", "query_key_value", "dense", "bias", "masked_bias", "norm_factor"]

        if hasattr(layer, "layer_idx"):
            submodules.append("layer_idx")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True  # Downcast query/key projections
        self.dropout_prob_attn = 0.0  # No dropout for GPT-NeoX

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# GPTNeo Attention Layer adapted for BetterTransformer
class GPTNeoAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoSelfAttention, nn.Module):
    _attn = gpt_neo_wrapped_scaled_dot_product  # Overriding attention mechanism with optimized implementation

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        # Determine if attention is global or local
        if layer.bias[0][0][-1][0] == 1:
            self.attention_type = "global"
        else:
            self.attention_type = "local"

        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, self.attention_type)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "k_proj", "v_proj", "q_proj", "out_proj", "bias", "masked_bias"]

        if hasattr(layer, "layer_id"):
            submodules.append("layer_id")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())  # Scaling factor
        self.dropout_prob_attn = float(config.attention_dropout)  # Attention dropout probability

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# Bark Attention Layer adapted for BetterTransformer
class BarkAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BarkSelfAttention, nn.Module):
    _attn = bark_wrapped_scaled_dot_product  # Overriding attention mechanism with optimized implementation

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig", is_causal: bool = False):
        super().__init__(config)

        # Set configuration parameters from the layer
        is_causal = layer.is_causal
        config.dropout = layer.dropout
        config.hidden_size = layer.embed_dim
        config.num_heads = layer.num_heads
        config.bias = layer.out_proj.bias is not None

        if is_causal:
            config.block_size = layer.bias.shape[-1]

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, is_causal)

        self.module_mapping = None
        submodules = ["dropout", "attn_dropout", "resid_dropout", "att_proj", "out_proj"]

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if is_causal:
            setattr(self, "bias", getattr(layer, "bias"))
            self.original_layers_mapping["bias"] = "bias"

        self.supports_training = False  # Training is not supported for BarkAttentionLayer
        self.dropout_prob_attn = float(config.dropout)  # Attention dropout probability

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# Bloom Attention Layer adapted for BetterTransformer
class BloomAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BloomAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.dropout_prob_attn = config.attention_dropout  # Attention dropout probability

        self.module_mapping = None
        self.layer_idx = getattr(layer, "layer_idx", None)

        submodules = ["query_key_value", "dense", "attention_dropout"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    def forward(self, *args, **kwargs):
        return bloom_forward(self, *args, **kwargs)


# Codegen Attention Layer adapted for BetterTransformer
class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer, CodeGenAttention, nn.Module):
    _attn = codegen_wrapped_scaled_dot_product  # Overriding attention mechanism with optimized implementation

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "qkv_proj", "out_proj", "scale_attn"]

        # Conditional submodules based on transformers version
        if hasattr(layer, "embed_positions"):
            submodules.append("embed_positions")

        if hasattr(layer, "causal_mask"):
            submodules.append("causal_mask")

        if hasattr(layer, "layer_idx"):
            submodules.append("layer_idx")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.dropout_prob_attn = config.attn_pdrop  # Attention dropout probability

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# OPT Attention Layer adapted for BetterTransformer
class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer, OPTAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.is_decoder,
            )

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())  # Scaling factor

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    def forward(self, *args, **kwargs):
        return opt_forward(self, *args, **kwargs)


# T5 Attention Layer adapted for BetterTransformer
class T5AttentionLayerBetterTransformer(BetterTransformerBaseLayer, T5Attention, torch.nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        if hasattr(config, "text_config"):
            config = config.text_config
        super().__init__(config)

        # Initialize the base layer
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.has_relative_attention_bias)

        submodules = ["q", "k", "v", "o"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        head_dim = layer.d_model // layer.n_heads  # Hidden size / number of attention heads
        self.scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32)).to(torch.get_default_dtype())  # Scaling factor

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.has_relative_attention_bias:
            setattr(self, "relative_attention_bias", layer.relative_attention_bias)
            self.original_layers_mapping["relative_attention_bias"] = "relative_attention_bias"

        self.module_mapping = None
        self.is_decoder = layer.is_decoder

    def forward(self, *args, **kwargs):
        return t5_forward(self, *args, **kwargs)


# Bart BetterTransformer initialization function
def bart_bettertransformer_init(self, layer: "nn.Module", config: "PretrainedConfig"):
    with torch.device("meta"):
        super(BetterTransformerBaseLayer, self).__init__(
            layer.embed_dim,
            layer.num_heads,
            layer.dropout,
            layer.is_decoder,
            layer.k_proj.bias is not None,
        )

    self.module_mapping = None
    submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
    for attr in submodules:
        setattr(self, attr, getattr(layer, attr))

    self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    self.is_decoder = layer.is_decoder


# Bart Attention Layer adapted for BetterTransformer
class BartAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BartAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


# Blenderbot Attention Layer adapted for BetterTransformer
class BlenderbotAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BlenderbotAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


# M2M100 Attention Layer adapted for BetterTransformer
class M2M100AttentionLayerBetterTransformer(BetterTransformerBaseLayer, M2M100Attention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


# Marian Attention Layer adapted for BetterTransformer
class MarianAttentionLayerBetterTransformer(BetterTransformerBaseLayer, MarianAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


# Pegasus Attention Layer adapted for BetterTransformer
class PegasusAttentionLayerBetterTransformer(BetterTransformerBaseLayer, PegasusAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)
