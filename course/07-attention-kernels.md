# Chapter 07 — Attention kernels

## TL;DR

A forward pass is mostly matmuls, and Ch.01 said those are memory-bound at decode. Attention is the one operation whose cost and memory grow with sequence length, so at long context it dominates — and the *naive* implementation is far worse than it needs to be, because it writes the full `S×S` attention-score matrix out to slow GPU memory (HBM) and reads it back. **FlashAttention** is the fix: tile the computation, do it inside the GPU's tiny fast on-chip memory (SRAM), and *never materialize the score matrix in HBM*, using an "online softmax" to stitch tiles together. Same exact math, a fraction of the memory traffic, and `O(S)` memory instead of `O(S²)`. The serving engines don't write this kernel in Python — they *call* a separately-compiled FlashAttention build (each ships its *own* — vLLM's `vllm_flash_attn`, SGLang's `sgl_kernel` — with the third-party `flash-attn` package only as a fallback) and hand it Ch.06's paged block table so it gathers KV from scattered blocks. This chapter is where "the bottleneck is moving bytes, not doing math" becomes a kernel you can name.

---

## Why this matters

Attention is where long context lives or dies. Everything else in the forward pass scales linearly with sequence length; attention's naive form scales quadratically in both compute *and* memory, and the memory is the killer — a 32k-token sequence's score matrix is over a billion entries — a couple GB *per head* in fp16 (more with the fp32 softmax intermediates), and there are dozens of heads per layer — if you materialize it, which you cannot. The difference between an engine that serves 128k context and one that OOMs at 8k is, to a first approximation, whether its attention kernel is IO-aware. And because the kernel is where the paged, batched KV cache you built in Ch.04–06 actually gets *read*, it is the join point of the whole memory story. You don't need to write a kernel, but you need to know why one is fast, so you can pick the right backend and reason about your long-context bill.

---

## The concept

### Attention is the operation that scales with length

The rest of a transformer layer (the MLP, the projections) does the same work per token regardless of how many other tokens exist. Attention does not: token *i* attends over all *j ≤ i*, so the work grows with the sequence. For a sequence of length `S`, the score matrix `QKᵀ` is `S × S`. That quadratic is why attention — not the MLP — is the operation the whole serving field optimizes, and why the *attention kernel* is where performance is won.

### The real bottleneck is HBM traffic, not FLOPs

Naive attention computes it in three HBM round-trips:

```
S = Q @ Kᵀ          # write an S×S score matrix to HBM
P = softmax(S)      # read S from HBM, write P back to HBM
O = P @ V           # read P from HBM
```

The matmuls are cheap (Ch.01: GPUs have FLOPs to spare). The cost is **moving the `S×S` matrix to and from HBM** — the same memory-bound story as decode, one level down. Worse, materializing `S×S` is `O(S²)` *memory*, so long context OOMs before it's even slow. The GPU's compute units sit idle waiting on HBM. This is a memory-movement problem wearing a math costume.

### FlashAttention: tile it, and never write the matrix

FlashAttention (Dao et al., NeurIPS 2022) is an **IO-aware** kernel. The GPU has a tiny amount of very fast on-chip memory (**SRAM**) and a large amount of slow off-chip memory (**HBM**); the whole trick is to minimize HBM round-trips. FlashAttention:

- Splits Q, K, V into **tiles** small enough to fit in SRAM.
- For each tile, loads it into SRAM, computes that block of the attention *there*, and accumulates into the running output — **never writing the `S×S` score matrix to HBM at all.**
- Produces the mathematically *exact* same result as naive attention (it is not an approximation).

The payoff: the memory footprint drops from `O(S²)` to `O(S)` exactly (the score matrix is never stored), HBM traffic drops sharply (by a factor set by how much of a tile fits in SRAM), and attention speeds up several-fold — more at longer context. The compute didn't change; the *data movement* did. That is the entire lesson of Ch.01, cashed out in the single most important kernel in the stack.

### Online softmax is what makes tiling legal

There's a catch that makes this non-obvious: softmax needs the maximum and the sum over the *entire* row (the max for numerical stability, the sum to normalize). If you only ever see one tile of the row at a time, how do you softmax? **Online softmax**: maintain a running max and running sum as tiles arrive, and rescale the partial output each time the max updates. It's the streaming version of softmax, and it is the mathematical heart of FlashAttention — the reason you can compute an exact softmax without ever holding the whole row. Every fast attention kernel since is a variation on this.

### FlashAttention-2 and -3: filling the GPU

The idea is stable; the *implementation* gets re-tuned every GPU generation. **FlashAttention-2** (2023) improved how the work is partitioned across the GPU's warps and thread blocks, pushing utilization much higher. **FlashAttention-3** (2024) is Hopper-specific — asynchronous copies, warp specialization, and fp8 support. This is the fastest-rotting part of the chapter: which kernel is fastest on which chip changes with each release, and it is exactly the kind of current detail to ask your agent for the day you deploy. The *concept* — IO-aware tiling with online softmax — does not move.

### The engine calls the kernel and hands it the paged table

Crucially, the serving engines do **not** implement FlashAttention's tiling in Python — they dispatch to a separately-compiled kernel (by default each engine's *own* build, not a third-party package), feeding it Ch.06's block table so the kernel gathers KV from scattered blocks:

```python
# vLLM — dispatch to a COMPILED FlashAttention kernel, pass it the paged block table.
# vllm/v1/attention/backends/flash_attn.py @ ae098ab

if is_flash_attn_varlen_func_available():                  # L36
    from vllm.v1.attention.backends.fa_utils import (       # L37 fa_utils re-exports vLLM's OWN compiled build…
        flash_attn_varlen_func, ...)                        # L39 …from vllm.vllm_flash_attn — compiled, not Python here (3rd-party flash-attn is only the fallback)
...
attn_metadata = FlashAttentionMetadata(
    ...
    block_table=block_table_tensor,                # L613 Ch.06's paged table → the kernel reads KV through it
    slot_mapping=slot_mapping, ...)
```

So the kernel does two IO tricks at once: **tile in SRAM** (FlashAttention) *and* **gather KV from non-contiguous blocks** (PagedAttention). That composition — an IO-aware kernel that also understands paged memory — is what makes the batched, paged cache of Ch.04–06 actually fast to read. Paging without a paged-aware kernel would be correctness with no speed.

### Prefill and decode want different kernels

The two regimes from Ch.01 need different attention shapes, which is why engines carry more than one path:

- **Prefill** attends many query tokens over their (growing) context at once — a large, compute-heavy, variable-length attention. This is the `flash_attn_varlen_func` shape: many queries, packed variable-length sequences.
- **Decode** attends *one* query token over a long cached KV — a memory-bound gather from the paged cache. This is the `flash_attn_with_kvcache` / dedicated decode-kernel shape: one query, huge cached K/V read through the block table.

An engine that used the prefill kernel for decode (or vice versa) would leave a lot of performance on the table; the split mirrors the prefill/decode asymmetry you've now seen at every layer of the stack.

### Two engines, one kernel family

Verified in both. **Agreement (load-bearing):** neither engine writes attention's tiling in Python; both dispatch to a **FlashAttention-family** *compiled* kernel — and by default each engine's *own* build (vLLM's `vllm_flash_attn`, SGLang's `sgl_kernel`), with the third-party `flash-attn` package only as a fallback — feeding it the paged block/page table. vLLM imports `flash_attn_varlen_func` via `fa_utils` and passes `block_table` (`flash_attn.py` L37/L613); SGLang imports both `flash_attn_varlen_func` and `flash_attn_with_kvcache` from `sgl_kernel.flash_attn` and builds a paged `page_table` for the decode kernel (`flashattention_backend.py` L42–43, `_build_pa_page_table` L95). **Divergence (backend zoo, will rot):** each engine has a pluggable *backend* abstraction selecting among FlashAttention, FlashInfer, Triton, and vendor kernels per hardware and workload, with different defaults and their own hand-written page-table/Triton glue. The kernel *family* is the concept you keep; which specific backend is selected on which GPU is the part that changes release to release.

---

## Real-system notes

- **vLLM** — `vllm/v1/attention/backends/` @ `ae098ab` holds a backend per kernel family (`flash_attn.py`, `flashinfer.py`, `triton_attn.py`, …). The FlashAttention backend imports `flash_attn_varlen_func` (via `fa_utils`, from vLLM's own compiled `vllm_flash_attn` build — third-party `flash-attn` is only the fallback) and threads the paged `block_table` (L613) into the kernel; `slot_mapping` tells the kernel where to *write* new K/V.
- **SGLang** — `python/sglang/srt/layers/attention/` @ `52c6e27` mirrors this with `flashattention_backend.py` (importing `flash_attn_varlen_func` + `flash_attn_with_kvcache` from `sgl_kernel.flash_attn`, L42–43) plus its own Triton page-table builders (`_build_pa_page_table`, L95) and several vendor/MLA backends. Same dispatch-to-compiled-kernel pattern, richer backend set.
- **The FlashAttention papers** are the primary source for the algorithm (external, not in the cloned repos): Dao et al., *FlashAttention* (NeurIPS 2022); Dao, *FlashAttention-2* (2023); Shah et al., *FlashAttention-3* (2024). Also **FlashInfer** (2024) for a serving-specialized kernel library. Ask your agent for the current fastest option on your specific GPU — it moves.
- **llama.cpp** implements its own attention kernels (CPU/Metal/CUDA) rather than depending on flash-attn, which makes it a good place to *read* an attention kernel end-to-end, and to feel the naive-vs-tiled difference on hardware you own.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Materializing the score matrix.** A naive or misconfigured attention path that writes `S×S` to HBM OOMs or crawls at long context. *Fix: an IO-aware (FlashAttention-family) kernel that tiles in SRAM and never materializes the matrix (this chapter).*
- **Paged cache, non-paged kernel.** Building the block-paged cache of Ch.06 but pairing it with a kernel that assumes contiguous KV silently corrupts or falls back to slow copies. *Fix: a paged-aware kernel fed the block table — the two are a matched pair (Ch.06, this chapter).*
- **One kernel for both regimes.** Using the prefill (varlen) kernel to decode, or vice versa, wastes throughput. *Fix: distinct prefill vs. decode attention paths (this chapter).*
- **Assuming last year's kernel is still fastest.** FlashAttention-2/3, FlashInfer, and vendor kernels trade places by GPU generation. *Fix: benchmark backends on your actual hardware; treat "which kernel" as a current, measured decision, not a constant (this chapter, Ch.17).*
- **Ignoring the kernel when planning long context.** Context-length limits are as much a kernel/memory question as a model question. *Fix: reason about attention memory (`O(S)` with FlashAttention) when sizing context (Ch.14).*

---

## Pair with your agent

- *"Implement naive attention (materialize `S×S`) and a tiled/online-softmax version for a toy sequence, and show me the HBM traffic and peak memory for each across S = 512 / 4k / 32k. Confirm the naive one goes `O(S²)` and the tiled one `O(S)`."*
- *"Walk me through online softmax: maintain a running max and sum over tiles of one row and show the partial-output rescaling. Prove it equals a full-row softmax."*
- *"Open `references/vllm/vllm/v1/attention/backends/flash_attn.py` and show me where it imports the external kernel and where it passes `block_table` — then find the same two things in `references/sglang/.../flashattention_backend.py`."*
- *"On my GPU, benchmark the available attention backends (FlashAttention-2/3, FlashInfer, Triton) for a decode-shaped and a prefill-shaped workload. Tell me which wins for each and why."*
- *"Explain why decode uses `flash_attn_with_kvcache` while prefill uses `flash_attn_varlen_func`, in terms of the one-query-vs-many-query and gather-from-paged-KV shapes."*

---

## What's next

You've now made a single forward pass as fast as the memory hierarchy allows: the atom (Ch.01), a paged, batched KV cache (Ch.04–06), and an IO-aware kernel to read it (Ch.07). The remaining throughput lever doesn't make each step faster — it makes the loop take *fewer steps*. Ch.08 is **speculative decoding**: use a cheap draft to guess several tokens ahead and verify them in one pass of the big model, breaking the "one forward pass = one token" rule that has held since Ch.01.
