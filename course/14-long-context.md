# Chapter 14 — Long context

## TL;DR

Every cost model in this course assumed a modest sequence length. Long context — tens or hundreds of thousands of tokens — stress-tests all of them at once and breaks the naive implementations. Three walls appear: the **KV cache** (Ch.04) balloons to tens of GB per request because its cost is linear in sequence length; **prefill attention** (Ch.07) is quadratic in length, so the first token of a 128k prompt is enormously expensive; and the model's **position encoding** runs past what it was trained on, so RoPE must be *scaled* (linear / NTK / YaRN) to keep positions in the range the model learned. There's no single fix — long context is served by *composing* what you've already built: paged and quantized KV (Ch.06, Ch.09), GQA (Ch.04), chunked prefill (Ch.11), prefix caching (Ch.12), and bounded-attention variants (sliding window, attention sinks). This chapter is where the whole stack is pushed to the scale that separates real long-context serving from a demo.

---

## Why this matters

Long context is where the bills and the failures concentrate. A 200k-token request can cost more in KV memory than the model weights, take seconds to prefill, and — if the position encoding isn't scaled correctly — quietly produce garbage past the training length with no error. It's also the fastest-growing demand: agents accumulate history, RAG stuffs retrieved documents, codebases are pasted whole. Understanding long context is understanding which of your Ch.04–12 techniques are load-bearing at scale and which quietly stop working. Get it right and you serve 128k affordably; get it wrong and you either OOM, crawl, or hallucinate past the horizon.

---

## The concept

### Three walls appear at once

Short sequences hid three costs that all grow with length:

- **Memory** — the KV cache is linear in sequence length (Ch.04), so it dominates at long context.
- **Time** — prefill attention is quadratic in length (Ch.07), so long prompts are slow to first token.
- **Position** — the model's position encoding was trained to some maximum; past it, behavior degrades unless the encoding is scaled.

Each has its own fix, and long-context serving is getting all three right together.

### The KV cache wall (Ch.04, at scale)

Ch.04's formula — `2 × n_layers × n_kv_heads × head_dim × dtype_bytes` per token — is linear in tokens, which is fine at 4k and ruinous at 128k. The same 32-layer, 8-KV-head, fp16 model that cost ~128 KiB/token means **~16 GiB of KV for a single 128k-token request**. One long request can consume more memory than the model weights, and a handful of them exhaust any GPU. This is why every KV-shrinking technique you've learned matters *most* here: GQA (fewer KV heads, Ch.04), FP8 KV (half the bytes, Ch.09), paging (no fragmentation waste, Ch.06), and KV offloading to CPU/NVMe for the cold part of very long contexts. Long context is the regime where the KV cache stops being a detail and becomes the entire budget.

### The prefill wall (Ch.07, at scale)

Prefill attention is `O(S²)` in prompt length (Ch.07). At 128k tokens that quadratic is enormous — the first token of a long prompt can take *seconds*, and it's pure compute (Ch.01's compute-bound regime). Two tools from earlier chapters are what make it bearable: **chunked prefill** (Ch.11) splits the giant prefill across steps so it doesn't freeze every other request's decode, and **prefix caching** (Ch.12) skips it entirely when the long context is reused. FlashAttention (Ch.07) keeps the *memory* of that prefill `O(S)`, but the *compute* is still quadratic — so time-to-first-token on long prompts is a real, growing cost you schedule around, not one you make disappear.

### The position wall: RoPE scaling

A model is trained with positions up to some maximum (say 8k). Its rotary position embedding (RoPE) encodes each token's position by rotating its query/key vectors at position-dependent frequencies. Feed it position 100,000 and those rotations land far outside anything it saw in training — output degrades. The fix is to **scale the RoPE frequencies** so that long positions map back into the trained range. It's a config-driven dispatch:

```python
# vLLM — serving past the trained context length = scale the RoPE frequencies (here: YaRN).
# vllm/model_executor/layers/rotary_embedding/__init__.py @ ae098ab  (get_rope dispatch)

elif scaling_type == "yarn":                                   # L243 pick a RoPE-scaling method from model config
    scaling_factor = rope_parameters["factor"]                 # L244 factor ≈ target length / original trained length
    original_max_position = rope_parameters["original_max_position_embeddings"]  # L245
    ...                                                        # → constructs YaRNScalingRotaryEmbedding (imported L28)
```

The scaling method is a property of how the model was extended — you don't choose it freely; you use what the model's config declares, or you re-extend the model deliberately.

### The RoPE scaling family

The methods trade simplicity against quality, and they're external algorithms (papers) the engines implement:

- **Linear (position interpolation)** — divide every position by the scaling factor so 128k maps into 8k. Simple; degrades short-context quality noticeably.
- **NTK-aware / dynamic-NTK** — scale the *base frequency* instead of positions, preserving high-frequency detail better; "dynamic" adjusts as the sequence grows.
- **YaRN** — scale different frequency bands differently (interpolate low frequencies, extrapolate high ones), the current default for well-behaved long-context extension.

The point for serving: the model ships with a scaling config, the engine dispatches to the matching implementation, and using the *wrong* scaling (or none) is a silent quality failure past the training length.

### Bounding the KV: sliding window and attention sinks

Some models cap the KV cost by *not attending to everything*:

- **Sliding-window attention** — each token attends only to the last `W` tokens, so the KV cache is bounded by `W` regardless of total length. Trades some long-range recall for a fixed memory cost.
- **Attention sinks (StreamingLLM)** — keep the first few tokens (which act as attention "sinks") plus a recent window, enabling stable generation over effectively unbounded streams without the KV growing without bound.

These cap the `S` term in Ch.04's formula structurally rather than by compression — a different lever than quantization, and the reason some long-context models stay cheap where full-attention models don't.

### Prefix caching is long context's best friend (Ch.12)

The single biggest long-context win is often *not recomputing it*. Long contexts are usually **reused** — a chat re-sends its long history every turn, an agent reuses a big tool/context preamble each step, a document is queried repeatedly. Prefix caching (Ch.12) computes that expensive long prefill *once* and reuses its KV, turning the prefill wall from a per-request cost into a one-time cost. Long context and prefix caching are the natural pairing: the more expensive the prefix, the more caching it pays off.

### There is no single fix — only composition

Long context has no silver bullet; it's the chapter where the whole course composes. A production 128k stack typically runs: GQA + FP8 KV (shrink the Ch.04 cache), paging (Ch.06, no waste), chunked prefill (Ch.11, don't stall decode), prefix caching (Ch.12, reuse the context), correct RoPE scaling (this chapter, stay in-range), and possibly sliding-window/sink attention or KV offload for the extremes. Each technique addresses one wall; only together do they make long context affordable. If you understand why each is present, you can size and price a long-context deployment — and know which corner to cut when you must.

### Two engines, one set of levers

Verified in both. **Agreement (load-bearing):** both extend context by scaling RoPE frequencies via a config-dispatched family (linear / NTK / YaRN) — vLLM in `model_executor/layers/rotary_embedding/` (`get_rope` dispatch on `scaling_type`), SGLang in `srt/layers/rotary_embedding/` (`factory.py` + `yarn.py`) — and both compose the same KV-shrinking and prefill-smoothing tools (paging, FP8 KV, chunked prefill, prefix caching) for scale. **Divergence (which methods & offload, will rot):** the exact set of scaling variants, sliding-window/sink support, and KV-offload strategies (CPU/NVMe tiering) differ and evolve fast. The three-walls framing and the scale-the-position-encoding concept are durable; the specific scaling zoo and offload machinery are what change.

---

## Real-system notes

- **vLLM** — RoPE scaling in `vllm/model_executor/layers/rotary_embedding/` @ `ae098ab`: a `get_rope` dispatch on `scaling_type` (`"yarn"` L243) selecting `LinearScalingRotaryEmbedding`, `YaRNScalingRotaryEmbedding`, `NTKScalingRotaryEmbedding`, `DynamicNTKScalingRotaryEmbedding`, etc. Long context also leans on `--kv-cache-dtype fp8` (Ch.09), chunked prefill, and automatic prefix caching (Ch.12).
- **SGLang** — RoPE scaling in `python/sglang/srt/layers/rotary_embedding/` @ `52c6e27` (`factory.py` dispatch + `yarn.py`), plus KV offload management in `disaggregation/decode_kvcache_offload_manager.py` — reflecting how central KV tiering has become for long context at scale.
- **The scaling algorithms are external** (papers): RoPE (Su et al., RoFormer, 2021); linear position interpolation (Chen et al., 2023); NTK-aware scaling (community, 2023); YaRN (Peng et al., 2023); StreamingLLM / attention sinks (Xiao et al., 2023). Ask your agent which scaling a given model actually shipped with — using the wrong one is silent.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Sizing memory without the KV wall.** Planning a long-context deployment from weight size ignores that one 128k request can out-weigh the model. *Fix: budget KV from the Ch.04 formula at the max context, and apply GQA + FP8 KV + paging before assuming it fits (this chapter, Ch.04, Ch.09).*
- **Wrong or missing RoPE scaling.** Serving past the trained length without the model's declared scaling produces degraded output and no error. *Fix: use the scaling the model shipped with; verify quality at the far end of the context, not just the near end (this chapter).*
- **Long prefill freezing the fleet.** A 128k prompt prefilled in one shot stalls every concurrent decode. *Fix: chunked prefill; and prefix-cache reused contexts so the prefill happens once (Ch.11, Ch.12).*
- **Re-prefilling reused long context.** Recomputing a long shared history/document every turn wastes the most expensive compute in the stack. *Fix: prefix caching — long context's biggest single win (Ch.12).*
- **Assuming full attention scales.** Expecting a full-attention model to serve 1M tokens cheaply. *Fix: consider sliding-window / attention-sink models, or KV offload, for the extreme regime (this chapter).*

---

## Pair with your agent

- *"Compute the KV cache for my model at 8k, 32k, 128k, and 256k context (Ch.04 formula), and show me at what length one request's KV exceeds the model weights. Then show what FP8 KV + GQA do to those numbers."*
- *"Measure TTFT for my model at 1k / 16k / 128k prompts to see the quadratic prefill wall, then turn on prefix caching and re-measure a reused long context."*
- *"Open `references/vllm/vllm/model_executor/layers/rotary_embedding/__init__.py` (the `scaling_type` dispatch) and `references/sglang/.../layers/rotary_embedding/` (`factory.py`, `yarn.py`). Show me where each picks a RoPE scaling method, and explain linear vs. YaRN."*
- *"Serve my model past its trained length with the correct RoPE scaling vs. none, and show me where the output degrades without it — the silent-failure horizon."*
- *"Design a 128k-context deployment for my model on my GPUs: pick GQA/FP8-KV/paging/chunked-prefill/prefix-caching settings and tell me the max concurrency and the per-request cost."*

---

## What's next

You've now built the entire single-node serving path — from one forward pass (Ch.01) through the batched, paged, cached, constrained, long-context loop (Ch.02–14). Long context is where the KV cache becomes the whole budget, which owes you one last question: when it *still* won't fit, can you keep fewer tokens? **Ch.14.5** is a short detour on **KV compression and eviction** — sliding windows, attention sinks, and query-aware sparsity — the last memory lever before you spill to another tier. After that, the next block leaves the single process behind: Ch.15 is **backend infrastructure** — how this engine becomes a *service* (API servers, request queues, worker pools, multi-replica scaling, multi-tenancy, and the gateway that fronts them), where the fast engine you've built meets real traffic and the operational reality of running it around the clock.
