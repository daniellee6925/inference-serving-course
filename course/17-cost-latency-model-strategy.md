# Chapter 17 — Cost, latency, and model strategy

## TL;DR

This chapter turns everything you've built into a decision. Serving has one inescapable tension — **throughput vs. latency** — and a single dial that trades them: batch size (`max_num_seqs`) and the per-step token budget (`max_num_batched_tokens`). Bigger batch means more tokens/sec and worse tail latency; you pick the operating point from your **SLO**, not by maximizing either. On top of that sits the real cost strategy, and its first rule is counterintuitive: **the cheapest LLM call is the one you don't make.** Route easy requests to a small model, cascade from small to large, cache shared prefixes (Ch.12), substitute a deterministic tool for an LLM call (Ch.01's whole thesis), and quantize (Ch.09) so the calls you *do* make are cheap. The cost model from Ch.01's roofline through Ch.16's cost-per-token composes into a strategy you can defend on a budget: pick the smallest model that meets quality, the precision that fits, the hardware that matches, the batch that meets the SLO, and the routing that skips the expensive path.

---

## Why this matters

Inference cost is usually the largest line item in an AI product's bill, and it's the one most often left on the table. Teams reach for a bigger GPU when a smaller model would pass their eval, run everything through a frontier model when 80% of requests are easy, and maximize throughput while missing their latency SLO on the requests that pay. Every lever in this course has a cost, a latency, and a quality consequence; a *strategy* is knowing which to pull for a given SLO and budget. The difference between a naive and a considered strategy is routinely 3–10× in cost per request for the same quality — which is the difference between a product that's economical and one that isn't.

---

## The concept

### The throughput-latency frontier

You've circled this since Ch.05: batching amortizes the weight reads (Ch.01), so throughput rises with batch size — but a bigger running set means each decode step does more work, so per-token latency (TPOT) rises too, and admitting more prefill raises TTFT. **You cannot maximize both.** The achievable (throughput, latency) points form a frontier; every serving config picks one point on it. The entire tuning problem is: *given my SLO, which point maximizes throughput (and minimizes cost) while still meeting it?*

### The dial

That point is set by two knobs, and both engines expose them:

```python
# vLLM — the two knobs that ARE the throughput↔latency dial.  vllm/config/scheduler.py @ ae098ab

max_num_batched_tokens: int = ...   # L49 tokens the scheduler may process per step (the Ch.05 token budget)
max_num_seqs: int = ...             # L63 max concurrent sequences = the running-batch cap
# raise either → higher throughput, worse tail latency (TTFT/TPOT); lower → snappier, less efficient.
# SGLang: the same dial as max_running_requests + chunked_prefill_size (server_args.py L632 / L652).
```

`max_num_seqs` caps how many requests decode together (the throughput/TPOT trade); `max_num_batched_tokens` caps total tokens processed per step — which chiefly governs how much prefill is admitted (the TTFT/decode-interference trade, Ch.11). These are the primary levers, and you set them against a measured SLO — not to a vendor default.

### Pick the operating point from the SLO (goodput, Ch.16)

Here's the discipline from Ch.16: tune for **goodput** (requests meeting the SLO), not raw throughput. Raise the batch size while goodput climbs; stop when goodput starts falling because the SLO is being missed even as tokens/sec keeps rising. The optimal operating point is the largest batch that still meets your TTFT and TPOT targets — found by *measuring*, sweeping the dial under realistic load and watching the goodput curve, not by guessing. This is why Ch.16 came first: you can't set the dial without the metric.

### The cost model composes

Every chapter's lever plugs into one equation — cost per token = GPU $/hr ÷ (tokens/sec × 3600) (Ch.16):

- **Ch.01 roofline** sets the ceiling (memory bandwidth bounds decode).
- **Ch.05 batching** raises tokens/sec toward that ceiling.
- **Ch.09 quantization** shrinks the weight/KV bytes, raising decode throughput *and* fitting a bigger model on cheaper hardware.
- **Ch.10 parallelism** trades more GPUs for latency or capacity.
- **Ch.12 prefix caching** removes prefill work entirely for shared prefixes — free tokens.
- **Ch.08 speculative decoding** cuts latency where there's idle compute.

A serving strategy is choosing the combination that minimizes cost-per-token at your SLO. Because these compose, the wins multiply: FP8 + a good batch size + prefix caching can be several times cheaper than the naive config on the same GPU.

### The cheapest call is the one you don't make

The largest cost wins are usually *not* in making the LLM faster — they're in **not calling the big model at all**:

- **Substitute a deterministic tool** for an LLM call. This is Ch.01's founding thesis: if code can compute it exactly (a lookup, a calculation, a database query), don't pay a model to approximate it. The cheapest inference is no inference.
- **Cache the answer** (prefix caching, Ch.12; or a full response cache for repeated queries).
- **Use a smaller model** for the requests that don't need the big one.

Before tuning the engine, ask whether each call needs to happen and needs the biggest model. That question saves more than any kernel.

### Model routing and cascades

Most workloads have a wide difficulty distribution — many easy requests, a few hard ones — and serving them all through one frontier model overpays for the easy majority. Two patterns fix it:

- **Routing** — a cheap classifier sends each request to the smallest model that can handle it (small model for easy, large for hard).
- **Cascade** — try the small model first; escalate to the large one only when the small one's answer is low-confidence or fails a check. You pay for the big model only on the fraction that needs it.

For workloads where most requests are easy, routing/cascades cut cost several-fold at equal quality — often the single biggest lever after "don't call the model." (FrugalGPT and the model-cascade literature are the external references.)

### Speculative decoding is a latency lever, not a throughput one

A reminder from Ch.08, because it's a common strategy mistake: **speculative decoding buys latency, not throughput, and only where there's idle compute.** Turn it on for low-batch, latency-sensitive traffic (single-stream chat, coding assistants) and it's a big TTFT/TPOT win. Turn it on under heavy batch and it competes for compute the batching already claimed — neutral or negative. Speculation and batching pull on the same idle-compute budget from opposite ends; a coherent strategy picks per traffic profile, not globally.

### Model, precision, and hardware selection

Three choices set your whole cost basis before any tuning:

- **Model** — the smallest model that passes your quality eval (Ch.16) at your SLO. Bigger is not better; it's more expensive. This is the highest-leverage decision.
- **Precision** — quantize (Ch.09) to the lowest bit-width that holds quality on your eval; often the cheapest capacity and speed you can buy.
- **Hardware** — the GPU sets the roofline (Ch.01: bandwidth for decode, FLOPs for prefill) and the $/hr. Match it to your model size and traffic shape; a memory-bandwidth-bound decode workload wants bandwidth, not peak FLOPs.

These are decided *with* an eval and a target SLO, not by reflex.

### The strategy, composed

A defensible serving strategy is a procedure, not a guess:

1. **Model** — smallest that passes the eval (Ch.16) at the SLO.
2. **Precision** — lowest bit-width that holds quality (Ch.09).
3. **Hardware** — matched to the model's bound (Ch.01, Ch.10).
4. **Routing** — cheap path for easy requests; big model only when needed.
5. **Caching** — prefix caching (Ch.12) and response caching for reuse.
6. **Dial** — batch size / token budget set to the SLO by measuring goodput (Ch.16).
7. **Speculation** — on for the latency-sensitive, low-batch traffic (Ch.08).
8. **Measure** — cost-per-token and goodput (Ch.16); iterate.

Every step cites a chapter, because the strategy *is* the course composed. Run it in order and you can defend your cost per request on a budget — which is the whole point of everything from Ch.01 forward.

---

## Real-system notes

- **vLLM** — `vllm/config/scheduler.py` @ `ae098ab`: `max_num_batched_tokens` (L49) and `max_num_seqs` (L63) are the throughput-latency dial (both `Field(..., ge=1)` — bounded config, per LLM-config discipline). Auto-tuning of these against a workload is an active area; the defaults are a starting point, not an answer.
- **SGLang** — `server_args.py` @ `52c6e27`: `max_running_requests` (L632) and `chunked_prefill_size` (L652) are the equivalent dial, plus extensive routing (`sgl-router`, cache-aware) for the multi-replica cost strategy.
- **External** — model cascades / routing for cost: FrugalGPT (Chen et al., 2023) and the broader LLM-cascade literature; goodput-optimized serving: DistServe (Zhong et al., 2024). Ask your agent for current small-model options and a routing recipe for your task mix and budget.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Maximizing throughput, missing the SLO.** Cranking batch size for tokens/sec while blowing TTFT/TPOT on the requests that matter. *Fix: tune the dial for goodput at the SLO, not raw throughput (this chapter, Ch.16).*
- **Frontier model for everything.** Running easy and hard requests through one big model overpays for the easy majority. *Fix: routing/cascades — smallest model that handles each request (this chapter).*
- **Making calls that shouldn't happen.** Paying a model to do what code, a cache, or a lookup could do exactly. *Fix: substitute deterministic tools and caches; the cheapest inference is none (Ch.01, Ch.12).*
- **Speculation on globally.** Enabling speculative decoding under heavy batch, where it competes with batching for compute. *Fix: speculation for low-batch/latency-sensitive traffic only (Ch.08).*
- **Bigger GPU as the first move.** Reaching for more/bigger hardware before quantizing, right-sizing the model, or fixing the dial. *Fix: model → precision → dial → routing first; hardware matched to the bound, not maxed (this chapter, Ch.09).*

---

## Pair with your agent

- *"Sweep `max_num_seqs` / `max_num_batched_tokens` for my model under realistic load and plot throughput, TTFT p95, TPOT p95, and goodput. Recommend the operating point for my SLO and the cost-per-token there."*
- *"For my traffic, estimate the cost of routing easy requests to a small model vs. running everything through the big one — assume a difficulty split and show the savings at equal quality."*
- *"Audit my agent's LLM calls: which could be a deterministic tool, a cache hit, or a smaller model? Estimate the cost removed by each substitution (Ch.01, Ch.12)."*
- *"Compose a full cost strategy for my product: model, precision (Ch.09), hardware, dial, routing, caching, speculation — with the cost-per-token at each step and the total vs. the naive baseline."*
- *"Show me where speculative decoding helps vs. hurts on my batch profile by measuring latency with it on/off across batch sizes (Ch.08)."*

---

## What's next

You can now serve a model fast, at scale, and at a defensible cost. The remaining chapters cover the concerns that decide whether the service is *trustworthy* and *operable*. Ch.18 is **safety and adversarial inputs** — prompt injection, jailbreaks, resource-exhaustion attacks, and the trust boundaries a multi-tenant serving stack (Ch.15) must enforce. Speed and cost mean nothing if the service can be turned against its users or its operator, and Ch.18 is where the serving system meets its threat model.
