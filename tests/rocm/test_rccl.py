"""
ROCm port validation tests for RCCL-based distributed communication.

Covers the single genuine blocker that is NOT just-works and NOT Triton-replaceable:

  NCCL 2.27 symmetric-window APIs absent from RCCL 2.20.5 (ROCm 6.2)
  ────────────────────────────────────────────────────────────────────
  pynccl.cu (NCCLWrapper constructor) unconditionally calls:
    • ncclMemAlloc  / ncclMemFree       — present in RCCL 2.20.5 ✓
    • ncclCommWindowRegister            — NOT in RCCL 2.20.5 → link error
    • ncclCommWindowDeregister          — NOT in RCCL 2.20.5 → link error
    • ncclWindow_t, NCCL_WIN_COLL_SYMMETRIC — NOT in RCCL 2.20.5 → compile error

  Note: ncclMemAlloc IS available in RCCL 2.20.5 (confirmed via header grep),
  so only the window registration APIs need to be removed/guarded.  Guard with:
    #if NCCL_VERSION_CODE >= 22600   (NCCL 2.26 is when window APIs appeared)

  The tests below validate the post-port requirement: AllReduce and AllGather
  must produce correct numerical results on ROCm with max_size_bytes=0 (the
  path that bypasses the symmetric buffer in Python, and must also bypass
  ncclMemAlloc in C++).

Usage
─────
  Run directly for manual multi-GPU validation:
      python tests/rocm/test_rccl.py

  Or via pytest (requires ≥ 2 GPUs; skipped otherwise):
      pytest tests/rocm/test_rccl.py -v
"""
from __future__ import annotations

import os
import time
from typing import Callable

import pytest
import torch
import torch.distributed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_multi_gpu(minimum: int = 2):
    n = torch.cuda.device_count()
    if n < minimum:
        pytest.skip(f"Test requires ≥{minimum} GPUs, found {n}")


# ---------------------------------------------------------------------------
# Core runner (also used by __main__ multi-process entry point)
# ---------------------------------------------------------------------------

def _run_rccl_worker(tp_size: int, tp_rank: int) -> None:
    """Single worker executed in each spawned process."""
    torch.cuda.set_device(tp_rank)
    stream = torch.cuda.Stream(tp_rank)
    torch.cuda.set_stream(stream)

    torch.distributed.init_process_group(
        world_size=tp_size,
        rank=tp_rank,
        backend="gloo",  # CPU coordination only; collective ops use RCCL via pynccl
    )

    from minisgl.distributed import set_tp_info
    from minisgl.kernel import init_pynccl

    set_tp_info(tp_rank, tp_size)
    cpu_group = torch.distributed.group.WORLD

    # max_size_bytes=0 forces the C++ constructor to skip ncclMemAlloc and the
    # symmetric window registration — the critical path under test.
    comm = init_pynccl(
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_cpu_group=cpu_group,
        max_size_bytes=0,
    )

    dtype = torch.float16
    device = f"cuda:{tp_rank}"

    # ------------------------------------------------------------------ #
    # Test A: AllReduce sum — fixed-value tensor
    # ------------------------------------------------------------------ #
    K = 256
    x = torch.ones(K, dtype=dtype, device=device)
    comm.all_reduce(x, "sum")
    torch.cuda.synchronize()
    expected = torch.full((K,), float(tp_size), dtype=dtype, device=device)
    assert torch.allclose(x, expected), (
        f"[rank {tp_rank}] AllReduce(ones) failed: "
        f"expected {tp_size}, got {x[0].item()}"
    )

    # ------------------------------------------------------------------ #
    # Test B: AllReduce sum — rank-valued tensor, tests cross-rank ordering
    # ------------------------------------------------------------------ #
    x = torch.full((K,), float(tp_rank), dtype=dtype, device=device)
    comm.all_reduce(x, "sum")
    torch.cuda.synchronize()
    rank_sum = tp_size * (tp_size - 1) // 2
    expected = torch.full((K,), float(rank_sum), dtype=dtype, device=device)
    assert torch.allclose(x, expected), (
        f"[rank {tp_rank}] AllReduce(rank_values) failed: "
        f"expected {rank_sum}, got {x[0].item()}"
    )

    # ------------------------------------------------------------------ #
    # Test C: AllReduce — one rank intentionally slow (synchronization test)
    # ------------------------------------------------------------------ #
    x = torch.ones(K, dtype=dtype, device=device)
    if tp_rank == 0:
        torch.cuda.synchronize()
        time.sleep(0.5)
    comm.all_reduce(x, "sum")
    torch.cuda.synchronize()
    expected = torch.full((K,), float(tp_size), dtype=dtype, device=device)
    assert torch.allclose(x, expected), (
        f"[rank {tp_rank}] AllReduce sync test failed"
    )

    # ------------------------------------------------------------------ #
    # Test D: AllGather — each rank contributes its rank index
    # ------------------------------------------------------------------ #
    src = torch.full((K,), float(tp_rank), dtype=dtype, device=device)
    dst = torch.empty(K * tp_size, dtype=dtype, device=device)
    comm.all_gather(dst, src)
    torch.cuda.synchronize()

    # expected layout: [0,0,...,0, 1,1,...,1, ..., N-1,N-1,...,N-1]
    expected = torch.arange(tp_size, dtype=dtype, device=device).repeat_interleave(K)
    assert torch.allclose(dst, expected), (
        f"[rank {tp_rank}] AllGather failed"
    )

    # ------------------------------------------------------------------ #
    # Test E: AllReduce over a large tensor (forces the non-symmetric path)
    # ------------------------------------------------------------------ #
    BIG = 8 * 1024  # 8 K elements — larger than the (zero) symmetric buffer
    x = torch.ones(BIG, dtype=dtype, device=device)
    comm.all_reduce(x, "sum")
    torch.cuda.synchronize()
    expected = torch.full((BIG,), float(tp_size), dtype=dtype, device=device)
    assert torch.allclose(x, expected), (
        f"[rank {tp_rank}] AllReduce large tensor failed"
    )

    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# pytest entry-points (fork-based, single-process approach is not possible
# because pynccl requires one process per GPU — we spawn subprocesses)
# ---------------------------------------------------------------------------

def _mp_run(tp_size: int) -> None:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12399")
    procs = [mp.Process(target=_run_rccl_worker, args=(tp_size, r)) for r in range(tp_size)]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            assert p.exitcode == 0, (
                f"Worker process exited with code {p.exitcode}. "
                "See subprocess output for the failure."
            )
    except BaseException:
        for p in procs:
            p.terminate()
        raise


@pytest.mark.parametrize("tp_size", [2])
def test_rccl_allreduce_allgather_2gpu(tp_size: int):
    """
    RCCL AllReduce and AllGather produce correct results when max_size_bytes=0
    (no ncclMemAlloc / ncclCommWindowRegister / ncclWindow_t calls).
    """
    _require_multi_gpu(tp_size)
    _mp_run(tp_size)


@pytest.mark.parametrize("tp_size", [4])
def test_rccl_allreduce_allgather_4gpu(tp_size: int):
    """Same as 2-GPU test but with tp_size=4 (covers a full CDNA2 MI250X node)."""
    _require_multi_gpu(tp_size)
    _mp_run(tp_size)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing as mp

    tp_size = min(torch.cuda.device_count(), 4)
    if tp_size < 2:
        raise SystemExit("Need at least 2 GPUs to run this test")

    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12399")

    procs = [
        mp.Process(target=_run_rccl_worker, args=(tp_size, r))
        for r in range(tp_size)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exited with {p.exitcode}")
    except BaseException:
        for p in procs:
            p.terminate()
        raise

    print(f"All RCCL tests passed on {tp_size} GPUs.")
