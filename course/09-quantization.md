# Chapter 09 — Quantization

## TL;DR

Every speed-up so far has been *exact*: FlashAttention (Ch.07) and speculative decoding (Ch.08) change the output by nothing. Quantization is the first lever that is genuinely a **trade-off** — store the model's numbers in fewer bits (INT8, INT4, FP8 instead of FP16) and you shrink two costs at once: the weight bytes decode reads every step (Ch.01) and the `dtype_bytes` term in the KV-cache formula (Ch.04). That buys speed *and* capacity, often the cheapest of both you can get. The price is precision: fewer bits means rounding error, and past a point, quality loss. Calibration methods (GPTQ, AWQ) and well-chosen formats (FP8) keep the loss small, but "small" is now a judgment call, not a theorem. This chapter is where you learn what to quantize, how much it saves, and how to reason about the accuracy you're spending.

---

## Why this matters

Quantization is the highest-leverage knob you have that touches *both* halves of the serving cost model. Halving the weight precision roughly halves the memory traffic that bounds decode (Ch.01) and lets a bigger model fit on a smaller GPU; halving the KV precision doubles how many requests fit (Ch.04), directly raising the batch size Ch.05 can sustain. A 70B model that needs two GPUs in FP16 often serves on one in INT4. But it is also the one lever that can silently degrade your model, and the degradation is workload-specific — fine on chat, visible on math or code. So this is the chapter where "make it faster" stops being free and starts requiring you to measure quality, not just latency.

---

## The concept

### The first lossy lever

FlashAttention and speculative decoding were exact optimizations — you could turn them on with no eval. Quantization is different: it changes the numbers the model computes with, so it *can* change outputs. That makes it the first technique in this course where the right answer is "measure it on your workload." Everything below is about making the loss small and knowing where it hides.

### What quantization is

A quantized value is a low-bit integer plus a **scale** (and sometimes a zero-point): `x ≈ q · scale`, where `q` is stored in 8, 4, or even fewer bits, or in a low-range float (FP8). You store the compact `q` and the scale; you recover an approximation of `x` by multiplying. Fewer bits means less memory and less bandwidth to move the value — and, on hardware with low-precision tensor cores, less compute. The error is the gap between `x` and `q · scale`, and the whole art is keeping that gap where it doesn't matter.

### Why it helps serving *twice*

Quantization hits both terms of the cost model you've built:

- **Weight reads (Ch.01).** Decode is memory-bound on streaming every weight from HBM. Store weights in INT4 instead of FP16 and that stream is ~4× smaller, so a memory-bound decode step is ~4× less bandwidth — directly faster.
- **KV cache (Ch.04).** The formula was `2 × n_layers × n_kv_heads × head_dim × dtype_bytes` per token. Quantize the KV to FP8 and `dtype_bytes` goes 2 → 1, halving the cache and *doubling* the token slots — more concurrent requests, bigger batch (Ch.05).

Two levers pulled by one technique. That's why quantization is usually the first thing to reach for when you're memory- or capacity-bound.

### The KV cache is quantized by changing its storage dtype

The KV half is concrete in the source — it's the same buffers from Ch.04, allocated in a smaller dtype:

```python
# SGLang — KV-cache quantization = store the cache in fewer bytes.
# sglang/.../mem_cache/memory_pool.py @ 52c6e27  (KVCache.__init__)

self.dtype = dtype                                          # L1206 the logical KV dtype
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, ...):  # L1208 FP8 KV cache requested →
    self.store_dtype = torch.uint8                          # L1210 store 1 byte/element (vs 2 for fp16)
else:
    self.store_dtype = dtype                                # L1212
# Ch.04's k_buffer / v_buffer are allocated with store_dtype, so FP8 halves dtype_bytes in
#   KV bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes   → 2× the token slots.
```

Set the KV dtype to FP8 and the cache you sized in Ch.04 literally halves. (vLLM exposes the same thing as `--kv-cache-dtype fp8`, backed by `quantization/kv_cache.py`.) The KV cache tolerates FP8 well because attention is fairly robust to KV noise — which is why FP8 KV is one of the safest quantizations to turn on.

### Three things you can quantize, independently

- **Weights** — the big memory win, and the most common. Weight-only schemes (INT4/INT8) shrink the weight stream while keeping activations in FP16.
- **Activations** — quantize the values flowing through the network too (e.g. FP8 or INT8 activations), which lets the *matmul itself* run in low precision on hardware that supports it.
- **KV cache** — shown above; independent of the other two, and often worth doing even when you keep weights in FP16.

You can mix these: FP16 weights + FP8 KV, or INT4 weights + FP16 activations, or FP8 everywhere. Each target has its own quality/speed profile.

### Weight-only vs. weight-and-activation

The distinction decides *which* regime you speed up:

- **Weight-only (e.g. GPTQ, AWQ at INT4).** Store weights in 4 bits; dequantize to FP16 inside the kernel; do the matmul in FP16. This saves *memory bandwidth* (the weight read) but not compute. It shines in the **memory-bound** regime — decode at small batch — and can even *lose* at large batch/prefill, where you're compute-bound and the dequant adds work.
- **Weight-and-activation (e.g. FP8 W8A8).** Quantize both, and run the matmul in the low precision on native FP8 tensor cores (Hopper and later). This saves memory *and* compute, so it helps prefill and large batch too. The catch is you need the hardware and the activations tolerate the format.

This mirrors a pattern you've now seen repeatedly: techniques that only cut memory traffic (weight-only quant, speculative decoding) help the memory-bound decode regime; techniques that also cut compute help everywhere.

### The quality trade-off and calibration

Fewer bits, more rounding error — but *how* you quantize matters enormously. Naive round-to-nearest loses noticeable quality at INT4. The methods that made low-bit practical are smarter:

- **GPTQ** quantizes weights column-by-column, correcting the error introduced so far against a small calibration set — so later weights compensate for earlier rounding.
- **AWQ** observes that a few *salient* weight channels (identified by activation magnitude) carry most of the quality, and protects them with per-channel scaling.
- **FP8** (e4m3 / e5m2) often needs little or no calibration because a float's range handles outliers that break integer schemes — which is why it's frequently near-lossless.

These are external algorithms (papers, below), not engine code; the engines *apply* the resulting quantized weights. The point for you: INT4 with a good calibration method ≈ FP16 quality on most workloads, but "most" is doing work — math, code, and long-context reasoning are where loss shows up first, so those are what you eval.

### Granularity: one scale, or many

A scale can cover a whole tensor (cheapest, most error), a channel, or a **group** of weights (e.g. one scale per 128 weights). Finer granularity tracks the real distribution better — less error — but stores and reads more scales. Group quantization (INT4 weights + per-group FP16 scales) is the common sweet spot: near-FP16 quality at ~4× compression, because the scale overhead is small relative to the packed weights.

### Two engines, one idea

Verified in both. **Agreement (load-bearing):** both quantize weights (FP8, GPTQ, AWQ) and the KV cache, storing compact values plus scales and dequantizing (or computing low-precision) in the kernel. vLLM's `model_executor/layers/quantization/` holds `fp8.py`, `auto_awq.py`, `auto_gptq.py`, and `kv_cache.py`; SGLang's `srt/layers/quantization/` holds `fp8.py`, `awq/`, `gptq/`, and `kv_cache.py`, plus the `store_dtype` KV lever above. **Divergence (kernels/defaults, will rot):** which low-bit matmul kernels each ships (Marlin, Machete, CUTLASS, Triton variants), which formats are default, and how MoE experts are quantized differ and change every release. The *idea* — compact storage + scales + dequant-in-kernel — is stable; the fastest INT4 kernel this month is not.

---

## Real-system notes

- **vLLM** — `vllm/model_executor/layers/quantization/` @ `ae098ab`: a method per scheme (`fp8.py`, `auto_awq.py`, `auto_gptq.py`, `fbgemm_fp8.py`, …) plus `kv_cache.py`; `--kv-cache-dtype fp8` quantizes the cache, and weight schemes plug in as `LinearMethod`s that hold packed weights + scales and dispatch to low-bit matmul kernels.
- **SGLang** — `python/sglang/srt/layers/quantization/` @ `52c6e27` mirrors this (`fp8.py`, `awq/`, `gptq/`, `kv_cache.py`, `fp4_kv_cache_quant_method.py`), and the KV pool's `store_dtype` (L1206–1212) is where FP8 KV becomes a 1-byte buffer.
- **The algorithms are external** (papers, not the cloned repos): GPTQ (Frantar et al., 2022); AWQ (Lin et al., 2023); SmoothQuant (Xiao et al., 2022) for activation quantization; LLM.int8() (Dettmers et al., 2022); and the FP8 formats (e4m3/e5m2). Ask your agent which scheme+kernel is fastest and most accurate for your model and GPU today — it moves quickly.
- **llama.cpp** popularized the **GGUF** k-quant families (Q4_K_M, etc.) for CPU/edge — the easiest place to *feel* the size/quality trade by loading the same model at several bit-widths and comparing.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Quantizing without an eval.** Turning on INT4 and checking only latency hides quality regressions that surface on math/code/long-context. *Fix: eval the quantized model on task-representative prompts before and after — quantization is the first lever that needs it (this chapter, Ch.16).*
- **Weight-only quant at large batch.** Expecting a speed-up in a compute-bound regime where dequant overhead dominates. *Fix: use weight-only for memory-bound decode; use W8A8/FP8 when you also need compute savings (this chapter).*
- **Round-to-nearest at low bits.** Naive INT4 without calibration loses real quality. *Fix: use a calibration method (GPTQ/AWQ) or a float format (FP8); pick granularity (per-group) to match (this chapter).*
- **Forgetting the KV cache is separately quantizable.** Leaving KV in FP16 while chasing capacity elsewhere. *Fix: FP8 KV is one of the safest wins — it halves the Ch.04 formula with little quality cost.*
- **Assuming a kernel exists for your combo.** A format the model was quantized in may lack a fast kernel on your GPU, silently falling back to slow paths. *Fix: confirm a supported low-bit matmul kernel exists for your scheme+hardware, and benchmark it (this chapter, Ch.17).*

---

## Pair with your agent

- *"Compute what FP8 KV cache does to my Ch.04 numbers: recompute KV-bytes-per-token and max concurrent requests at FP8 vs FP16, and confirm it's a 2× on slots."*
- *"Serve my model at FP16, FP8 (W8A8), and INT4 (AWQ or GPTQ). Measure decode tokens/sec, memory footprint, AND quality on a math + code + chat eval set. Show me the full speed/size/quality table."*
- *"Explain GPTQ's error-correction and AWQ's salient-channel protection to me, then show which one my model's published quantization used and why."*
- *"Open `references/sglang/.../memory_pool.py` around L1206–1212 and `references/vllm/vllm/model_executor/layers/quantization/`. Show me where KV becomes 1 byte and where a weight scheme plugs in as a LinearMethod."*
- *"At my batch size, is weight-only INT4 actually faster or is dequant eating the win? Measure decode and prefill separately and tell me whether I'm memory- or compute-bound."*

---

## What's next

Quantization shrinks the model to fit and stream faster on one GPU. But the largest models don't fit on one GPU at *any* precision, and even when they fit, one GPU's memory bandwidth caps decode. Ch.10 is **multi-GPU serving** — tensor, pipeline, and expert parallelism — how a single forward pass is split across devices, why that split looks different for inference than for training, and how it interacts with the KV cache, batching, and kernels you've now built.
