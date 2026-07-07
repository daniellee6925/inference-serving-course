# Chapter 21 — Frontier serving

## TL;DR

This is deliberately the thinnest chapter, and that's the point: the frontier of inference serving moves faster than any written page can track, so this chapter *names the edges* rather than pinning down techniques that will have shifted by the time you read it. The edges in 2026: **prefill/decode disaggregation at fleet scale** (separate prefill and decode pools with KV transfer, now a serving standard); **KV-cache offloading and tiering** (spilling cold KV to CPU and NVMe so context outlives GPU memory); **MoE serving economics** (expert parallelism and the batch dynamics that make giant sparse models cheap or ruinous); **deterministic inference** (making serving bit-reproducible, which matters for RL and for trust); and **cache-aware routing at cluster scale**. For every one of these, the honest instruction is the same as Ch.00's: your AI partner and the reference systems' `main` branches will have fresher, more correct answers than this page. Read this for the *map* of where the field is pushing; get the *details* live.

---

## Why this matters — and why it's thin

Everything durable is in Ch.01–20; this chapter is the frontier, which is by definition the part that ages fastest. The course's whole design (Ch.00) is to be a skeleton whose slow-moving concepts you learn here and whose fast-moving specifics you get from your agent. Nowhere is that split sharper than here: naming these frontiers is durable; the current best implementation of each is a moving target that would be stale in a quarter. So this chapter is short on purpose. Its value is knowing *what to ask about* — and recognizing each frontier as an extension of a concept you already hold from earlier chapters.

---

## The concept

### Disaggregation at fleet scale

Ch.11 introduced prefill/decode disaggregation on a small scale; the frontier is running it as the *default* fleet architecture: large pools of prefill workers and decode workers, each tuned for its regime (prefill for compute throughput, decode for latency), with the KV cache transferred between them over a fast fabric. The engineering is all in the KV transfer — its bandwidth, its overlap with compute, and the routing that decides which decode worker inherits a prefill. SGLang's `disaggregation/` subsystem (with transports like `mooncake/`) and vLLM's `kv_transfer/` are where this lives; the research lineage is DistServe, Splitwise, and Mooncake. It extends Ch.11 from "an option" to "the way large deployments are built."

### KV offloading and tiering

The KV cache (Ch.04) is the binding constraint, and the frontier answer to "it doesn't fit" is a **memory hierarchy for KV**: hot KV on the GPU, warm KV offloaded to CPU RAM, cold KV to NVMe — spilling and fetching blocks as sequences and prefix caches (Ch.12) grow past GPU capacity. SGLang's `disaggregation/decode_kvcache_offload_manager.py` and HiCache-style tiering are examples. It's virtual memory (Ch.06's paging idea) extended across the whole memory hierarchy, and it's what lets a fleet hold far more cached context than GPU memory alone allows.

### MoE serving economics

Ch.01 and Ch.09 flagged that Mixture-of-Experts breaks the dense cost model (`2·N_active` FLOPs, degraded batch weight-reuse). The frontier is serving *giant* MoE models (hundreds of billions to trillions of parameters, few active) economically — which is dominated by **expert parallelism** (Ch.10), the all-to-all routing it requires, and the batch dynamics that determine whether experts are well-utilized or starved. Getting MoE serving right (expert placement, routing, the all-to-all fabric) is one of the largest open cost levers, and it's where much frontier engineering is concentrated as MoE becomes the default frontier architecture.

### Deterministic inference

Standard serving is *non-deterministic* (Ch.01): the same prompt can yield different tokens across runs and batch sizes, because floating-point reductions and batch-variant kernels aren't bit-stable. The frontier of making serving **reproducible** — batch-invariant kernels, deterministic reductions — matters for two reasons: **RL** (where the inference engine must match the trainer's numerics token-for-token, as SGLang's `enable_deterministic_inference` / RL-on-policy path targets) and **trust/debugging** (a reproducible service is auditable). It's a young, fast-moving area (the 2025 "defeating nondeterminism in LLM inference" line of work) and a real frontier where serving meets training.

### Cache-aware routing at cluster scale

Ch.12's prefix cache and Ch.15's router combine into a frontier problem: making the *entire cluster's* KV a coherent, shared, cache-aware resource — routing each request to the replica (or the tier) that already holds its prefix, across many machines, as the cache churns. It's where prefix caching, routing, and disaggregation converge, and it's one of the highest-leverage fleet-level cost wins still being actively worked out.

### The moving edges you'll ask about

Others, named without pretending to pin them: **advances in speculative decoding** (better draft models, multi-token methods, EAGLE-family successors, Ch.08); **encoder-prefill optimization** for the exploding multimodal workloads (Ch.20); **quantization frontiers** (lower bit-widths, better calibration, FP4 and beyond, Ch.09); and **hardware co-design** (new accelerators, memory technologies, and interconnects that reset the roofline of Ch.01). Each is an active research-and-engineering front; each is an extension of a chapter you've read.

---

## Real-system notes

- **The reference systems' `main` branches are the real "frontier" document.** The pinned commits this course cites (vLLM `ae098ab`, SGLang `52c6e27`) already differ from what those projects ship today. For anything in this chapter, the current source — `disaggregation/`, `kv_transfer/`, the MoE and quantization layers, the deterministic-inference paths — is the authoritative, freshest reference. This is the one chapter where cloning the *latest* repo beats reading the pinned one.
- **The research lineage** (external, and itself moving): DistServe, Splitwise, Mooncake (disaggregation); the KV-offload / HiCache line (tiering); DeepSeek-scale MoE serving; the deterministic-inference work (2025). Treat these as pointers, not settled answers.
- **Your AI partner is the live half here.** Ch.00's thesis is most true at the frontier: the durable map is on this page; the current best technique is a question to ask, with today's benchmarks, prices, and source in hand.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Treating a frontier technique as settled.** Adopting a specific disaggregation or offload design as "the answer" when the field is still converging. *Fix: hold the concept, get the current best implementation live (this chapter, Ch.00).*
- **Reaching for the frontier before the fundamentals.** Chasing disaggregation or MoE tricks while the batch size, KV formula, or cache config (Ch.04–12) are untuned. *Fix: exhaust the durable levers first; the frontier is for when they're not enough.*
- **Assuming determinism you don't have.** Building on reproducible outputs without engineering for them. *Fix: treat determinism as a property to engineer and verify, not assume (Ch.01, this chapter).*
- **Reading the pinned commit for frontier questions.** Trusting a snapshot for the fastest-moving topics. *Fix: for this chapter's material, read the reference systems' current `main` (this chapter).*

---

## Pair with your agent

- *"What's the current state of the art in prefill/decode disaggregation — pull the latest from vLLM's `kv_transfer/` and SGLang's `disaggregation/` `main` and tell me what changed since the pinned commits."*
- *"Explain KV-cache tiering (GPU→CPU→NVMe) for my long-context / multi-turn workload: when it pays off, the fetch-latency cost, and what to offload."*
- *"Walk me through the economics of serving a large MoE for my traffic: expert parallelism, the all-to-all cost, and the batch size where experts become well-utilized (Ch.10, Ch.01)."*
- *"How would I make my serving deterministic enough to match an RL trainer's numerics? What breaks bit-reproducibility, and what does batch-invariant inference cost?"*
- *"Given my project, which frontier lever (disaggregation, offload, MoE, routing, determinism) is worth investing in — and which fundamentals (Ch.04–12) should I fix first?"*

---

## What's next

You've walked the whole map — one forward pass (Ch.01) to the frontier (Ch.21). The final chapter stops teaching techniques and helps you *choose*: Ch.22 is a **design canvas** for your own serving stack. Given your model, your traffic, your latency SLO, and your budget, which of the twenty-one chapters' levers are load-bearing for *you*, and which can wait? It's where this course turns from a map you read into a system you design.
