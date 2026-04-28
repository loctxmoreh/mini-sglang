"""Numerical / behavioral tests for the Triton sampler."""
from __future__ import annotations

import pytest
import torch

from minisgl.kernel.triton.sampling import (
    triton_sampling_from_probs,
    triton_softmax,
    triton_top_k_sampling_from_probs,
    triton_top_k_top_p_sampling_from_probs,
    triton_top_p_sampling_from_probs,
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("V", [128, 32_000, 152_064])
def test_softmax_matches_eager(dtype, V):
    _require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    B = 4
    logits = torch.randn(B, V, dtype=dtype, device=device) * 5.0
    temps = torch.tensor([0.5, 1.0, 2.0, 0.7], device=device, dtype=torch.float32)

    probs = triton_softmax(logits, temps)
    ref = torch.softmax((logits.float() / temps.unsqueeze(-1)), dim=-1).to(dtype)

    torch.testing.assert_close(probs, ref, atol=1e-3, rtol=1e-3)
    # rows sum to 1 (within tolerance)
    sums = probs.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)


def test_softmax_extreme_logits_no_nan():
    """Numerically stable on rows with very large/small logits."""
    _require_gpu()
    device = torch.device("cuda:0")
    logits = torch.tensor(
        [
            [1000.0, -1000.0, 0.0, 0.0],
            [-1000.0, -1001.0, -999.0, -1000.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    temps = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    probs = triton_softmax(logits, temps)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1 + 1e-5).all()
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# sampling_from_probs — distribution check
# ---------------------------------------------------------------------------

def _empirical_distribution(sampler_fn, V: int, n: int = 20_000) -> torch.Tensor:
    """Run `sampler_fn` `n` times against a fixed prob and tally outputs."""
    counts = torch.zeros(V, dtype=torch.float64)
    device = torch.device("cuda:0")
    g = torch.Generator(device=device)
    g.manual_seed(123)
    for _ in range(n // 1000):
        idx = sampler_fn(g, batch=1000)  # [1000]
        counts += torch.bincount(idx.cpu().to(torch.int64), minlength=V).double()
    return counts / counts.sum()


def test_sampling_from_probs_recovers_distribution():
    _require_gpu()
    device = torch.device("cuda:0")
    V = 16
    target = torch.tensor(
        [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.01, 0.005, 0.005,
         0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    def run(g, batch):
        probs = target.unsqueeze(0).expand(batch, V).contiguous()
        return triton_sampling_from_probs(probs, generator=g)

    emp = _empirical_distribution(run, V, n=20_000)
    diff = (emp - target.cpu().double()).abs().max().item()
    assert diff < 0.02, f"empirical distribution far from target: max |Δ| = {diff:.4f}"


def test_sampling_zero_prob_tokens_never_chosen():
    _require_gpu()
    device = torch.device("cuda:0")
    V = 32
    probs = torch.zeros(8, V, device=device)
    probs[:, [3, 17]] = 0.5  # only tokens 3 and 17 have mass
    g = torch.Generator(device=device); g.manual_seed(0)

    seen = set()
    for _ in range(50):
        idx = triton_sampling_from_probs(probs, generator=g)
        seen.update(idx.cpu().tolist())
    assert seen.issubset({3, 17}), f"sampled illegal token: {seen}"


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------

def test_top_k_only_samples_top_k_tokens():
    _require_gpu()
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    B, V, k = 4, 256, 5
    probs = torch.softmax(torch.randn(B, V, device=device), dim=-1)
    g = torch.Generator(device=device); g.manual_seed(7)

    expected_topk = set()
    for b in range(B):
        topk_idx = torch.topk(probs[b], k=k).indices.cpu().tolist()
        expected_topk.update((b, t) for t in topk_idx)

    for _ in range(50):
        out = triton_top_k_sampling_from_probs(probs, k, generator=g).cpu().tolist()
        for b, t in enumerate(out):
            assert (b, t) in expected_topk, f"top-k sampled outside top-{k}: row={b} tok={t}"


def test_top_k_per_row_tensor():
    _require_gpu()
    device = torch.device("cuda:0")
    B, V = 3, 64
    probs = torch.softmax(torch.randn(B, V, device=device), dim=-1)
    top_k = torch.tensor([1, 3, 5], device=device, dtype=torch.int32)
    g = torch.Generator(device=device); g.manual_seed(1)

    # row 0: top-1 → must be argmax
    argmax0 = int(probs[0].argmax().item())
    for _ in range(20):
        out = triton_top_k_sampling_from_probs(probs, top_k, generator=g).cpu().tolist()
        assert out[0] == argmax0, f"row 0 with k=1 should always pick argmax={argmax0}, got {out[0]}"


# ---------------------------------------------------------------------------
# top-p
# ---------------------------------------------------------------------------

def test_top_p_keeps_at_least_one_token():
    """Even with extreme top_p ~ 0, the most-likely token must be reachable."""
    _require_gpu()
    device = torch.device("cuda:0")
    B, V = 2, 32
    probs = torch.zeros(B, V, device=device)
    probs[:, 7] = 0.9
    probs[:, 12] = 0.1
    g = torch.Generator(device=device); g.manual_seed(0)
    # top_p = 0.001 — only the very top should pass
    for _ in range(20):
        out = triton_top_p_sampling_from_probs(probs, 0.001, generator=g).cpu().tolist()
        assert out == [7, 7]


def test_top_p_within_high_mass_subset():
    _require_gpu()
    device = torch.device("cuda:0")
    B, V = 4, 128
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(B, V, device=device), dim=-1)
    g = torch.Generator(device=device); g.manual_seed(0)
    p = 0.5

    # Compute the per-row nucleus set for reference.
    sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    cum_before = cum - sorted_p
    keep = cum_before < p
    keep[..., 0] = True
    nucleus = [
        set(sorted_idx[b][keep[b]].cpu().tolist()) for b in range(B)
    ]

    for _ in range(50):
        out = triton_top_p_sampling_from_probs(probs, p, generator=g).cpu().tolist()
        for b, t in enumerate(out):
            assert t in nucleus[b], f"row {b}: token {t} outside nucleus {nucleus[b]}"


# ---------------------------------------------------------------------------
# top-k + top-p
# ---------------------------------------------------------------------------

def test_top_k_top_p_combined():
    _require_gpu()
    device = torch.device("cuda:0")
    B, V, k = 2, 64, 8
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(B, V, device=device), dim=-1)
    g = torch.Generator(device=device); g.manual_seed(0)

    topk_idx = torch.topk(probs, k=k, dim=-1).indices.cpu()
    legal = [set(topk_idx[b].tolist()) for b in range(B)]

    for _ in range(40):
        out = triton_top_k_top_p_sampling_from_probs(probs, k, 0.9, generator=g).cpu().tolist()
        for b, t in enumerate(out):
            assert t in legal[b], f"row {b}: token {t} outside top-{k} ∩ top-p set"
