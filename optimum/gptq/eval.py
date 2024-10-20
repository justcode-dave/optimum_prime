# coding=utf-8
"""
Evaluation Utilities for GPTQ Quantized Models.

This module provides functionality to evaluate the performance of models,
specifically in terms of perplexity on the WikiText-2 dataset. The perplexity
metric is commonly used in natural language processing tasks to measure
the uncertainty of a language model's predictions.

Functions:
    - evaluate_perplexity: Evaluates the perplexity of the given model on the WikiText-2 test dataset.
    
Example usage:
    ```python
    from eval import evaluate_perplexity
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"Perplexity: {ppl}")
    ```

Perplexity Calculation:
    The perplexity is computed by dividing the negative log-likelihood (NLL)
    across the test dataset by the sequence length and number of samples,
    then applying the exponential function to retrieve the perplexity score.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def evaluate_perplexity(model, tokenizer):
    """
    Evaluates the perplexity of the given model on the WikiText-2 test dataset.

    Args:
        model (`nn.Module`):
            The pre-trained model (e.g., GPT-style models) to be evaluated.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer associated with the model.

    Returns:
        `float`: The calculated perplexity score of the model on the WikiText-2 test set.
    """
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)

    seqlen = 512
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()
