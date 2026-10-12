# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Drive the public ATOM library (TP4) on synthetic weights, with no capture.

``minimax_m3_mono_replay.py`` checks numerics against saved native layers and
needs their fixtures. This checks everything that does not need a reference:
construction over a real TP group, the peer handshake, every supported step
width, eager / graph equality, cache remapping, live-row invariance under
poisoned padding, and close(). It is what a serving adapter actually exercises,
so it is the first thing to run on a new kernel stack or cache layout.

Weights are random but tame: E8M0 scale codes are all UNIT (1.0) and the FP8
projections are small, so an output that is not finite is a bug and not an
overflow of nonsense inputs.

    ATOM_DISABLE_VLLM_PLUGIN=1 VLLM_PLUGINS='' \
    python -m torch.distributed.run --master-addr 127.0.0.1 --master-port 29631 \
        --nproc-per-node 4 tests/minimax_m3_mono_smoke.py
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from atom.models.minimax_m3.mono.library import (
    SUPPORTED_TOKENS,
    AtomM3Mono,
    CacheSpec,
    LayerSpec,
    StepMetadata,
    TPContext,
)

HIDDEN = 6144
QKV_ROWS = 2560  # (16 q + 1 k + 1 v + 1 index q + 1 index k) * 128 at TP4
O_K = 2048  # 16 local q heads * 128
EXPERTS = 129  # 128 routed + the fused shared expert
INTER2 = 1536  # 2 * (3072 / 4): gate and up of a TP4 rank
INTER = 768
HEAD_DIM = 128
ROTARY = 64
UNIT_E8M0 = 127  # the scale code of 1.0
LAYER_ID = 3
BLOCKS = 512
PAGE_SLOTS = 256  # a main block holds 128 K and 128 V token slots


def fp8(shape, dev, scale=0.05):
    return (torch.randn(shape, device=dev) * scale).to(torch.float8_e4m3fn)


def fp4x2(shape, dev):
    """Packed FP4 whose nibbles are small codes, so products stay in range."""
    return torch.randint(0, 0x33, shape, dtype=torch.uint8, device=dev).view(
        torch.float4_e2m1fn_x2
    )


def scales(shape, dev):
    return torch.full(shape, UNIT_E8M0, dtype=torch.uint8, device=dev)


def bf16(shape, dev, scale=1.0):
    return (torch.randn(shape, device=dev) * scale).bfloat16()


def make_layer(dev) -> LayerSpec:
    pos = torch.arange(4096, device=dev, dtype=torch.float32)[:, None]
    freq = pos / 10000.0 ** (torch.arange(32, device=dev) / 32.0)[None, :]
    cos_sin = torch.cat([freq.cos(), freq.sin()], dim=1).bfloat16()
    assert cos_sin.shape[1] == ROTARY
    return LayerSpec(
        LAYER_ID,
        bf16((HIDDEN,), dev, 0.02),  # g_in, a Gemma (1 + w) norm
        fp8((QKV_ROWS, HIDDEN), dev),
        torch.full((QKV_ROWS,), 0.01, dtype=torch.float32, device=dev),
        *[bf16((HEAD_DIM,), dev, 0.02) for _ in range(4)],  # g_q g_k g_iq g_ik
        cos_sin,
        fp8((HIDDEN, O_K), dev),
        torch.full((HIDDEN,), 0.01, dtype=torch.float32, device=dev),
        bf16((HIDDEN,), dev, 0.02),  # g_post
        bf16((128, HIDDEN), dev, 0.02),  # router gate
        torch.zeros(128, dtype=torch.float32, device=dev),  # correction bias
        fp4x2((EXPERTS, INTER2, HIDDEN // 2), dev),
        scales((EXPERTS, INTER2, HIDDEN // 32), dev),
        fp4x2((EXPERTS, HIDDEN, INTER // 2), dev),
        scales((EXPERTS, HIDDEN, INTER // 32), dev),
        1e-6,  # eps
        2.0,  # routed_scaling_factor
        7.0,  # swiglu_limit
    )


def make_cache(dev):
    main = torch.zeros((BLOCKS, 2, 128, 128), dtype=torch.float8_e4m3fn, device=dev)
    index = torch.zeros((BLOCKS, 128, 128), dtype=torch.float8_e4m3fn, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    return main, index, CacheSpec(
        LAYER_ID,
        main,
        main.view(-1, 1, 8, 16, 16),
        main.view(-1, 1, 1, 128, 16)[8:],
        index,
        one,
        one.clone(),
    )


def run_width(runtime, dev, main, index, tokens, query_len, context, log):
    """One step width: eager, then a graph replayed over two cache placements,
    then the same graph with the tail rows padded out and poisoned.

    The cache payload is fixed and only its *base* moves between placements --
    randomizing the contents instead would make replay differ from eager for a
    reason that is not a bug."""
    reqs = tokens // query_len
    width = (context + 127) // 128
    # Fixed addresses: a captured graph reads these same buffers on every replay.
    mt = torch.zeros((reqs, 128), dtype=torch.int32, device=dev)
    it = torch.zeros_like(mt)
    lens = torch.full((reqs,), context, dtype=torch.int32, device=dev)
    slots = torch.empty(tokens, dtype=torch.int64, device=dev)
    islots = torch.empty(tokens, dtype=torch.int64, device=dev)
    hidden = bf16((tokens, HIDDEN), dev, 0.1)
    residual = bf16((tokens, HIDDEN), dev, 0.1)
    # Token j of a request is the (seq_len - query_len + j)-th of its sequence.
    j = torch.arange(tokens, device=dev, dtype=torch.int64) % query_len
    tok = context - query_len + j
    positions = tok.clone()
    md = StepMetadata(mt, it, lens, slots, islots, tokens, query_len)

    # Each request gets its own physical range, so no two write the same slot.
    span = reqs * width
    payload_main = torch.randint(
        0, 64, (span, 2, 128, 128), dtype=torch.uint8, device=dev
    ).view(torch.float8_e4m3fn)
    payload_index = torch.randint(
        0, 64, (span, 128, 128), dtype=torch.uint8, device=dev
    ).view(torch.float8_e4m3fn)
    blocks = torch.arange(width, device=dev, dtype=torch.int32)
    req_of = torch.arange(tokens, device=dev, dtype=torch.int64) // query_len

    def place(shift):
        """Put the same payload at a different offset in both caches. 0x7F is
        E4M3 NaN: anything read outside the payload shows up as a NaN output."""
        mbase, ibase = 7 + shift, 101 + shift
        main.view(torch.uint8).fill_(0x7F)
        index.view(torch.uint8).fill_(0x7F)
        main[mbase : mbase + span].copy_(payload_main)
        index[ibase : ibase + span].copy_(payload_index)
        rows = torch.arange(reqs, device=dev, dtype=torch.int32)[:, None] * width
        # packed layout: a main block spans 16 page16 pages, so the table is x2
        mt[:, :width] = (rows + blocks + mbase) * 2
        it[:, :width] = rows + blocks + ibase
        phys = mbase + req_of * width + tok // 128
        slots.copy_(phys * PAGE_SLOTS + tok % 128)
        islots.copy_((ibase + req_of * width + tok // 128) * 128 + tok % 128)

    def forward():
        runtime.prepare_step(md)
        return runtime.forward_layer(LAYER_ID, hidden, residual, positions)

    place(0)
    runtime.prepare_step(md)  # a prior step may have stopped before its first layer
    out, rout = forward()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(out).all()):
        raise AssertionError(f"tokens={tokens} q={query_len}: eager output not finite")
    ref, ref_r = out.clone(), rout.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout, grout = forward()
    for shift in (0, 3):
        place(shift)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(gout, ref, atol=0, rtol=0)
        torch.testing.assert_close(grout, ref_r, atol=0, rtol=0)

    padding = False
    if reqs >= 2:
        live = tokens // 2
        place(0)
        slots[live:] = -1
        islots[live:] = -1
        lens[reqs // 2 :] = 0
        hidden[live:] = 0
        residual[live:] = 0
        graph.replay()
        torch.cuda.synchronize()
        kept, kept_r = gout[:live].clone(), grout[:live].clone()
        hidden[live:] = float("nan")
        residual[live:] = float("nan")
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(gout[:live], kept, atol=0, rtol=0)
        torch.testing.assert_close(grout[:live], kept_r, atol=0, rtol=0)
        padding = True
    graph.reset()
    log.append(
        {
            "tokens": tokens,
            "query_len": query_len,
            "requests": reqs,
            "context": context,
            "graph_equal": True,
            "cache_remap_equal": True,
            "poisoned_padding_equal": padding,
        }
    )


def main():
    p = argparse.ArgumentParser()
    # Small enough that every request's own block range fits in BLOCKS.
    p.add_argument("--context", type=int, default=1024)
    # The library refuses more rows than this ("requires at most four request
    # rows"); raise it together with that bound.
    p.add_argument("--max-requests", type=int, default=4)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(1234 + rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    dev = torch.device("cuda", rank)

    spec = make_layer(dev)
    main, index, cache = make_cache(dev)
    runtime = AtomM3Mono([spec], [cache], TPContext(dist.group.WORLD, rank, 4, dev))
    log = []
    try:
        skipped = []
        for tokens in SUPPORTED_TOKENS:
            for query_len in (1, 4):
                if tokens % query_len:
                    continue
                if tokens // query_len > args.max_requests:
                    skipped.append((tokens, query_len))
                    continue
                run_width(runtime, dev, main, index, tokens, query_len, args.context, log)
                if rank == 0:
                    print(log[-1], flush=True)
    finally:
        runtime.close()
        runtime.close()  # idempotent
    if rank == 0:
        print(f"OK: {len(log)} width/query combinations", flush=True)
        if skipped:
            print(
                f"skipped (over --max-requests {args.max_requests}): {skipped}",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
