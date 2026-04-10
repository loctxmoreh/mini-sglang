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


# Maximum candidates to consider before applying softmax.
# torch.softmax on the full vocabulary (~151k tokens) requires a 2 MiB HIP
# workspace allocated via hipMalloc, which fails when physical GPU memory is
# exhausted (shared-GPU scenarios on ROCm).  By pre-truncating to this limit
# before the softmax call the workspace stays below ~1 MiB and is serviced from
# PyTorch's caching allocator instead.
_MAX_CANDIDATES_BEFORE_SOFTMAX = 50_000


def _gumbel_sample(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Sample indices from log-probabilities using the Gumbel-max trick.

    Equivalent to `torch.multinomial(exp(log_probs), 1)`.
    Avoids rocrand / hipMalloc calls that fail when physical GPU memory is
    exhausted (shared-GPU scenario on ROCm).
    """
    # Gumbel(0,1) = -log(-log(U)) = -log(Exponential(1))
    gumbel = log_probs.new_empty(log_probs.shape).exponential_(1.0).log_().neg_()
    return torch.argmax(log_probs + gumbel, dim=-1)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    scaled = logits / temperatures.unsqueeze(1)  # [B, vocab]
    vocab = scaled.size(-1)

    if top_k is None and top_p is None:
        # No filtering: Gumbel-max directly on scaled logits avoids any softmax
        # hipMalloc workspace.  argmax(logits/T + Gumbel) is exactly equivalent
        # to sampling from softmax(logits/T).
        gumbel = scaled.new_empty(scaled.shape).exponential_(1.0).log_().neg_()
        return torch.argmax(scaled + gumbel, dim=-1)

    # --- top-k / top-p path ---
    # Pre-truncate to _MAX_CANDIDATES_BEFORE_SOFTMAX so that the softmax
    # workspace (≈3× tensor bytes) stays well below 2 MiB.
    max_candidates = _MAX_CANDIDATES_BEFORE_SOFTMAX
    if top_k is not None:
        k_max = int(top_k.max().item()) if isinstance(top_k, torch.Tensor) else int(top_k)
        max_candidates = min(max_candidates, k_max)
    max_candidates = min(max_candidates, vocab)

    # topk returns (values, indices) sorted descending
    topk_logits, topk_indices = torch.topk(scaled, max_candidates, dim=-1, sorted=True)

    # Compute probabilities only over the candidate set (small softmax, no OOM)
    topk_probs = torch.softmax(topk_logits, dim=-1)  # [B, max_candidates]

    rank = torch.arange(max_candidates, device=scaled.device).unsqueeze(0)  # [1, C]

    if top_k is not None:
        k = top_k.long().clamp(1, max_candidates).unsqueeze(1)  # [B, 1]
        topk_probs = topk_probs.masked_fill(rank >= k, 0.0)

    if top_p is not None:
        cumprobs = topk_probs.cumsum(dim=-1)
        top_p_t = top_p.unsqueeze(1) if isinstance(top_p, torch.Tensor) else top_p
        topk_probs = topk_probs.masked_fill(cumprobs - topk_probs > top_p_t, 0.0)

    total = topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    topk_probs = topk_probs / total

    sampled = _gumbel_sample(topk_probs.log()).unsqueeze(1)  # [B, 1]
    return topk_indices.gather(1, sampled).squeeze(1)


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
