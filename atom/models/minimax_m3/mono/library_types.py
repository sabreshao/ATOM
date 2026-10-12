# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor contracts for MiniMax-M3 sparse decode, independent of serving engines."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TPContext:
    """An existing TP group; construction and close are collective on cpu_group.

    device must be the current CUDA device on this rank."""

    cpu_group: object
    rank: int
    size: int
    device: torch.device


@dataclass(frozen=True)
class LayerSpec:
    """One TP4 sparse layer. Projections are AITER-shuffled PTPC E4M3 weights;
    expert weights are AITER-shuffled MXFP4 with shuffled E8M0 scales.
    Norm weights use Gemma's (1 + weight) convention and partial NeoX RoPE.
    Router weights are BF16; routing keeps FP32 logits. All tensors are borrowed
    without quantization, casts, shuffling, or copies.
    """

    layer_id: int
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    eps: float
    route_scale: float
    swiglu_limit: float
    shared_weight: float = 1.0
    sm_scale: float = 128**-0.5
    init_blocks: int = 0
    local_blocks: int = 1


@dataclass(frozen=True)
class CacheSpec:
    """Native FP8 main/index storage with independent allocation.

    Two layouts, differing only in where K and V live. The page16 tile interior
    is the same in both and is AITER SHUFFLE, not token-major: a K page is
    [d/16][t][d%16] over 16 tokens and 128 dims, a V page is [d][t]. A caller
    whose pool is token-major cannot be served by either layout.

    ``packed_page16_scalar_fp8``: K and V share one allocation. main is
    [blocks, 2, 128, 128], k starts at main, v starts 8 pages later, and each
    main block spans 16 physical pages. ``main_table`` entries are block * 2.

    ``split_kv_scalar_fp8``: k and v are independently allocated, each
    [blocks * 8] page16 pages; main is unused and may be None.
    ``main_table`` entries are the block itself.

    Scales are positive FP32 scalars in both. Index storage is
    [index_blocks, 128, 128] E4M3. All storage and scales must remain stable
    until captured graphs are destroyed and the runtime has closed.
    """

    layer_id: int
    main: torch.Tensor | None
    k: torch.Tensor
    v: torch.Tensor
    index: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    max_context: int = 16384
    layout: str = "packed_page16_scalar_fp8"


@dataclass(frozen=True)
class StepMetadata:
    """A uniform decode/verify batch, padded to token_count rows.

    Tables have one row per request. A main_table entry is scaled so that
    entry * 8 + sub is the physical page16 page: under
    ``packed_page16_scalar_fp8`` that makes it the main block ID * 2 (sixteen
    pages a block, K then V), under ``split_kv_scalar_fp8`` the block ID itself
    (eight pages a block in each of the two allocations). index_table entries
    are physical 128-token index block IDs in both.

    seq_lens includes all query tokens. Slots have one entry per padded token:
    main_slots are page16 token addresses (page * 16 + token-in-page),
    index_slots independently address index storage; -1 is padding.
    query_len is the uniform tokens per request, including padded requests.
    """

    main_table: torch.Tensor
    index_table: torch.Tensor
    seq_lens: torch.Tensor
    main_slots: torch.Tensor
    index_slots: torch.Tensor
    token_count: int
    query_len: int
