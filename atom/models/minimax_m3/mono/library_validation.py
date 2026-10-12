# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate borrowed weights and caches without converting their storage."""

import torch

from atom.models.minimax_m3.mono.library_types import CacheSpec, LayerSpec


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MiniMax-M3 ATOM mono: {message}")


def checked_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    require(
        tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0,
        f"{name}: expected contiguous aligned {shape} {dtype} on {device}; "
        f"got {tuple(tensor.shape)} {tensor.dtype} {tensor.device} "
        f"stride={tensor.stride()} offset={tensor.storage_offset()}",
    )
    return tensor


def validate_layer(spec: LayerSpec, device: torch.device) -> None:
    """Check a prepared layer before any runtime allocation or IPC handshake."""

    def check(tensor, name, shape, dtype=torch.bfloat16):
        checked_tensor(tensor, f"layer {spec.layer_id} {name}", shape, dtype, device)

    check(spec.w_qkv, "QKV", (2560, 6144), torch.float8_e4m3fn)
    check(spec.s_qkv, "QKV scales", (2560,), torch.float32)
    check(spec.w_o, "O", (6144, 2048), torch.float8_e4m3fn)
    check(spec.s_o, "O scales", (6144,), torch.float32)
    check(spec.gate, "router", (128, 6144))
    check(spec.g_in, "input norm", (6144,))
    for name in ("g_q", "g_k", "g_iq", "g_ik"):
        check(getattr(spec, name), name, (128,))
    check(spec.cos_sin, "cos/sin", (spec.cos_sin.shape[0], 64))
    check(spec.g_post, "post norm", (6144,))
    check(spec.bias, "router bias", (128,), torch.float32)
    check(spec.w13, "w13", (129, 1536, 3072), torch.float4_e2m1fn_x2)
    check(spec.s13, "s13", (129, 1536, 192), torch.uint8)
    check(spec.w2, "w2", (129, 6144, 384), torch.float4_e2m1fn_x2)
    check(spec.s2, "s2", (129, 6144, 24), torch.uint8)


PACKED = "packed_page16_scalar_fp8"
SPLIT = "split_kv_scalar_fp8"
PAGE_BYTES = 16 * 128  # a page16 page: 16 tokens, 128 dims, one byte each


def _validate_packed_kv(cache: CacheSpec, device: torch.device) -> None:
    """K and V in one allocation, V pre-offset by eight pages."""
    main, k, v = cache.main, cache.k, cache.v
    require(
        main is not None
        and main.ndim == 4
        and tuple(main.shape[1:]) == (2, 128, 128)
        and main.shape[0] > 0
        and main.element_size() == 1
        and main.is_contiguous()
        and main.device == device,
        "native packed K/V layout",
    )
    require(
        k.dtype == v.dtype == torch.float8_e4m3fn
        and k.device == v.device == device
        and k.is_contiguous()
        and v.is_contiguous()
        and k.shape[0] == main.shape[0] * 16
        and v.shape[0] == k.shape[0] - 8
        and k.data_ptr() == main.data_ptr()
        and v.data_ptr() - k.data_ptr() == 8 * PAGE_BYTES
        and k.untyped_storage().data_ptr() == main.untyped_storage().data_ptr()
        and v.untyped_storage().data_ptr() == main.untyped_storage().data_ptr(),
        "page16 K/V views must share native storage",
    )


def _validate_split_kv(cache: CacheSpec, device: torch.device) -> None:
    """K and V independently allocated, each a whole number of page16 pages.

    The tile interior is unchecked and unchangeable: a K page is
    [d/16][t][d%16] and a V page is [d][t]. A token-major pool passes every
    check here and then reads the wrong bytes, so the caller owns that
    guarantee -- see the layout note in CacheSpec.
    """
    k, v = cache.k, cache.v
    require(cache.main is None, "split layout has no packed main storage")
    for name, t in (("k", k), ("v", v)):
        require(
            t.dtype == torch.float8_e4m3fn
            and t.device == device
            and t.is_contiguous()
            and t.data_ptr() % 16 == 0
            and t.numel() > 0
            and t.numel() % PAGE_BYTES == 0,
            f"{name} cache must be contiguous aligned E4M3 of whole page16 pages; "
            f"got {tuple(t.shape)} {t.dtype} {t.device}",
        )
    require(
        k.numel() == v.numel(),
        f"k and v must hold the same page count; got {k.numel() // PAGE_BYTES} "
        f"and {v.numel() // PAGE_BYTES}",
    )


def validate_cache(cache: CacheSpec, device: torch.device, max_context: int):
    """``max_context`` is the caller's qualified ceiling, not a kernel limit."""
    require(
        cache.layout in (PACKED, SPLIT),
        f"unsupported cache layout {cache.layout!r}",
    )
    if cache.layout == PACKED:
        _validate_packed_kv(cache, device)
    else:
        _validate_split_kv(cache, device)
    checked_tensor(
        cache.index,
        "index cache",
        (cache.index.shape[0], 128, 128),
        torch.float8_e4m3fn,
        device,
    )
    for scale in (cache.k_scale, cache.v_scale):
        require(
            scale.numel() == 1
            and scale.dtype == torch.float32
            and scale.device == device
            and scale.is_contiguous()
            and bool(torch.isfinite(scale).all())
            and bool((scale > 0).all()),
            "cache scale must be a positive finite FP32 scalar",
        )
    require(
        0 < cache.max_context <= max_context,
        f"context capacity {cache.max_context} exceeds the qualified "
        f"{max_context}",
    )
