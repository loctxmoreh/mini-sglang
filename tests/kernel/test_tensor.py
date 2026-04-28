from __future__ import annotations

from minisgl.kernel import test_tensor as _test_tensor_kernel
from minisgl.utils import call_if_main
import torch


def test_tensor_kernel():
    """Test the tensor AOT kernel: CPU int32 + GPU int64 device-type check."""
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    _test_tensor_kernel(x, y)


@call_if_main()
def main():
    test_tensor_kernel()
