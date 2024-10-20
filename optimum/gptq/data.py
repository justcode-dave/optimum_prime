# coding=utf-8
"""
Utilities for Dataset Loading and Preprocessing for GPTQ Quantization.

This module provides a set of utilities to load, prepare, and process commonly used datasets,
as referenced in the GPTQ paper, specifically for use during model quantization.
It includes methods for dataset collation, tokenization, and batching, with specific support for
popular datasets such as WikiText-2 and C4.

Functions:
    - prepare_dataset: Prepares and batches datasets with optional padding.
    - collate_data: Batches tensors into the correct format, applying padding if necessary.
    - get_wikitext2: Loads and tokenizes the WikiText-2 dataset.
    - get_c4: Loads and tokenizes the C4 dataset.
    - get_c4_new: Loads and tokenizes a specific shard of the C4 dataset.
    - get_ptb, get_ptb_new: Raises errors as loading the PTB dataset is deprecated.
    - get_dataset: General method to load the specified dataset, supporting multiple dataset types.

Example usage:
    You can use these utilities to load datasets for quantization workflows as follows:

    ```python
    from data import get_dataset
    dataset = get_dataset("wikitext2", tokenizer, nsamples=128, seqlen=2048, split="train")
    ```

Dataset options:
    - 'wikitext2'
    - 'c4'
    - 'c4-new'

Deprecated:
    - 'ptb'
    - 'ptb-new'
"""
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
