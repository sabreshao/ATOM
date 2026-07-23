#!/usr/bin/env bash
# Kimi-K3 (Minimax-m3-xiaobing) serving on MI355 (gfx950) x8, tp=8, CUDA-graph decode.
#
# Validated: gsm8k 0.94 flexible/strict; decode ~20 tok/s (~4x over eager).
#
# CUDA-graph note: capture is enabled simply by NOT passing --enforce-eager while
# keeping --level 0. ATOM treats model_type=kimi_linear as GDN (use_gdn=True) and,
# with cudagraph_mode=FULL, captures the whole eager decode forward per decode batch
# size (<= max-num-seqs). No torch.compile / @support_torch_compile is needed.
# gpu-memory-utilization is 0.93 (not 0.90) so the ~1.5GB cudagraph pool fits
# alongside the KDA per-request state cache; at 0.90 startup fails with
# "Per-request cache tensor exceeds available KV budget".
#
# Correctness flags that MUST stay:
#   -tp 8                              (tp4 OOMs: MoE weights ~175GB/GPU)
#   --no-enable_prefix_caching        SDPA MLA prefill only sees in-batch tokens;
#   --no-enable_chunked_prefill       a cached/chunked prefix would be missed.
#   (MLA layers use the flash KV-cache layout fix — commit 5240fd5 — to dodge the
#    aiter SHUFFLE-read bug at head_dim=192.)
#
# Env overrides: K3_GMU (default 0.93), K3_LEVEL (default 0), K3_PORT (default 8000),
#   K3_ENFORCE_EAGER=1 to fall back to pure eager (disables CUDA graphs).
set -euo pipefail

MODEL="${K3_MODEL:-/shared/data/amd_int/models/xiaobing/Minimax-m3-xiaobing}"
ATOM_DIR="${K3_ATOM_DIR:-/workdir/xiaobing/ATOM-K3}"

export ATOM_LOADER_USE_THREADPOOL=1
export ATOM_LOADER_THREADPOOL_WORKERS=16

export ATOM_SYNC_AFTER_LOAD=1
export ATOM_DIST_TIMEOUT_SECONDS=3600

export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"

# Prewarm the safetensors shards into the OS page cache in parallel.
( cd "$MODEL" 2>/dev/null && ls model-*.safetensors 2>/dev/null \
    | xargs -P12 -I{} dd if={} of=/dev/null bs=4M 2>/dev/null ) || true

# flydsl moe
export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=0
export ATOM_USE_TRITON_MOE=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0
# unified attention
export ATOM_USE_UNIFIED_ATTN=1
export ATOM_FORCE_ATTN_TRITON=1

# runtime
export K3_MAX_MODEL_LEN=16384
export K3_MAX_NUM_SEQS=64
export K3_MAX_NUM_BATCHED_TOKENS=10240
export K3_GMU=0.93
export K3_LEVEL=0

cd "$ATOM_DIR"
exec /opt/venv/bin/python -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-model-len "${K3_MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${K3_MAX_NUM_SEQS:-64}" \
  --max-num-batched-tokens "${K3_MAX_NUM_BATCHED_TOKENS:-16384}" \
  --gpu-memory-utilization "${K3_GMU:-0.93}" \
  --server-port "${K3_PORT:-8000}" \
  --block-size 128 \
  --no-enable_prefix_caching 
