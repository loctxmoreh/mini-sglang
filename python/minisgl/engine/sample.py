from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | float | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _softmax_with_temperature(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """temperatures: [batch_size]; logits: [batch_size, vocab_size]."""
    return torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)


def _top_k_top_p_filter(
    probs: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply top-k and/or top-p filtering to a probability distribution.
    Returns filtered (and renormalized) probabilities in sorted order along with
    the corresponding original-vocab indices.
    """
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    vocab = probs.size(-1)
    rank = torch.arange(vocab, device=probs.device).unsqueeze(0)  # [1, vocab]

    if top_k is not None:
        k = top_k.long().clamp(1, vocab).unsqueeze(1)  # [batch, 1]
        sorted_probs = sorted_probs.masked_fill(rank >= k, 0.0)

    if top_p is not None:
        cumprobs = sorted_probs.cumsum(dim=-1)
        top_p_t = top_p.unsqueeze(1) if isinstance(top_p, torch.Tensor) else top_p
        # Zero out tokens whose running cumsum already exceeded top_p before them
        sorted_probs = sorted_probs.masked_fill(cumprobs - sorted_probs > top_p_t, 0.0)

    total = sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sorted_probs = sorted_probs / total
    return sorted_probs, sorted_indices


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    probs = _softmax_with_temperature(logits, temperatures)

    if top_k is None and top_p is None:
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    sorted_probs, sorted_indices = _top_k_top_p_filter(probs, top_k, top_p)
    sampled = torch.multinomial(sorted_probs, num_samples=1)  # [batch, 1]
    return sorted_indices.gather(1, sampled).squeeze(1)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
