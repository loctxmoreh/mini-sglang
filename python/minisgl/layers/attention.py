from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # Ensure q/k are contiguous before reshape (split may produce non-contiguous views)
        q = q.contiguous().view(-1, self.num_qo_heads, self.head_dim)
        k = k.contiguous().view(-1, self.num_kv_heads, self.head_dim)
        T = q.shape[0]
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q)
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k)
        positions = batch.positions
        if positions.shape[0] != T:
            raise RuntimeError(
                f"[Layer {self.layer_id}] RoPE shape mismatch: "
                f"q has {T} tokens but positions has {positions.shape[0]} elements. "
                f"batch.phase={batch.phase}, batch.size={batch.size}, "
                f"batch.padded_size={batch.padded_size}, "
                f"input_ids.shape={batch.input_ids.shape}, "
                f"qkv.shape={qkv.shape}, "
                f"qo_attn_dim={self.qo_attn_dim}, "
                f"num_qo_heads={self.num_qo_heads}, head_dim={self.head_dim}"
            )
        q, k = self.rotary.forward(positions, q, k)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, batch)
        return o.view(-1, self.qo_attn_dim)
