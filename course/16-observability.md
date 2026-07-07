# Chapter 16 — Observability

## TL;DR

Everything you tuned in Ch.01–15 is only real if you can measure it — and inference has its *own* metrics, not generic web-service ones. A request has **two** latencies, not one: **TTFT** (time to first token — prefill plus queue) and **TPOT/ITL** (time per output token, the between-token decode latency). Mean latency hides both. Throughput (tokens/sec) is only meaningful next to **goodput** — the fraction of requests that actually met their latency SLO. And the numbers that explain *why* your service behaves as it does are the ones from earlier chapters made observable: KV-cache usage (Ch.04/06), prefix-cache hit-rate (Ch.12), preemption rate (Ch.11), queue depth. Both engines expose this family of metrics (via Prometheus), though which exact ones differ (e.g. preemption count is vLLM-side). This chapter is the metrics that matter, what a healthy trace looks like, and the minimum eval kit — because a fast, scaled, multi-tenant service is only fast if you can prove it for the users who matter.

---

## Why this matters

You cannot operate what you cannot see, and inference fails in ways generic monitoring misses entirely. A dashboard showing "p50 latency: 400ms" tells you nothing — was that 400ms all TTFT (a queueing/prefill problem) or spread across 200 tokens of decode (a batch-size problem)? A "throughput up 40%" headline might mean you cranked the batch size and blew every latency SLO. And the silent regressions — a quantization change (Ch.09) that dropped quality 3 points, a prefix-cache hit-rate that collapsed after a routing change (Ch.12), a preemption storm (Ch.11) eating throughput — are invisible without the *right* metrics. Observability is how you turn all the levers of this course from "I think it's faster" into "here's the number," and how you catch the failure before your users do.

---

## The concept

### A request has two latencies

The single most important idea: an LLM request's latency is **two** numbers, because it has two phases (Ch.01).

- **TTFT — Time To First Token.** How long until the user sees *anything*. Dominated by queue time + prefill (Ch.11, Ch.14). This is what makes a chat feel responsive.
- **TPOT / ITL — Time Per Output Token / Inter-Token Latency.** How fast tokens stream *after* the first. Strictly they differ: **ITL** is the *series* of gaps between consecutive tokens (what vLLM records as `inter_token_latencies_iter`); **TPOT** is the *mean* decode time per token, `(E2E − TTFT) / (output_len − 1)`. People use them loosely, but a benchmark should say which. Dominated by decode and batch size (Ch.05). This is what makes a long answer feel fast or sluggish.

Optimizing one can hurt the other — a bigger batch raises throughput and TPOT while starving TTFT (Ch.05). So you *must* track both separately; a single "latency" number is a lie for LLM serving. End-to-end latency is a third view (TTFT + all TPOTs), useful but not a substitute.

### The metric set the engines actually track

These aren't abstractions — the engine records exactly this set every iteration:

```python
# vLLM — the metrics that matter for inference.  vllm/v1/metrics/stats.py @ ae098ab

class SchedulerStats:                                  # L171 per-iteration engine snapshot
    num_running_reqs: int = 0                          # L174 current batch size (Ch.05)
    num_waiting_reqs: int = 0                          # L176 queue depth
    kv_cache_usage: float = 0.0                         # L183 how full the KV pool is (Ch.04/06)
    prefix_cache_stats: PrefixCacheStats = ...          # L185 prefix-cache hit stats (Ch.12)

class IterationStats:                                  # per-iteration timing collection
    self.num_preempted_reqs = 0                        # L332 preemptions this step (Ch.11 pressure signal)
    self.time_to_first_tokens_iter: list[float] = []   # L336 TTFT — prefill + queue latency
    self.inter_token_latencies_iter: list[float] = []  # L337 ITL — the between-token decode-latency series (TPOT = its mean)
```

Read that as a map of the whole course: batch size (Ch.05), queue depth (Ch.11), KV usage (Ch.04/06), prefix-cache hits (Ch.12), preemptions (Ch.11), TTFT, ITL. Each metric is a lever from an earlier chapter made visible. SGLang tracks the same set from a different place — `last_gen_throughput` / `last_input_throughput` and `cache_hit_rate` in its scheduler's metrics reporter, TTFT in the tokenizer front end.

### Goodput, not just throughput

Throughput (tokens/sec, requests/sec) measures how much work you do; it says nothing about whether that work was *useful*. **Goodput** is the metric that matters: the rate of requests that met their SLO (e.g. "TTFT < 500ms *and* TPOT < 50ms"). A stack that maximizes raw throughput by growing the batch can have *high throughput and low goodput* — it's doing lots of work, all of it too slow to count. Optimize goodput, not throughput, whenever you have a latency target — it's the number that reflects what users actually receive.

### The "why" metrics: cache, preemption, KV usage

TTFT and TPOT tell you *what* is slow; these tell you *why*:

- **Prefix-cache hit-rate (Ch.12)** — a drop explains a TTFT spike (you're re-prefilling shared prefixes). A routing change that hurts hit-rate is invisible on a latency graph but obvious here.
- **Preemption rate (Ch.11)** — nonzero preemption means you're over-admitting for your KV pool; it eats throughput and spikes tail latency. It should be near zero in steady state.
- **KV-cache usage (Ch.04/06)** — how close you are to the capacity wall; near 100% means you're one long request from preemption.
- **Queue depth** — rising queue = you need more replicas (Ch.15) or admission is too loose.

These four are your first-response dashboard: when latency moves, one of them explains it.

### Cost-per-token

The business metric that ties it all together. **Cost per token = (GPU $/hour) / (tokens/sec × 3600)** — your hardware cost divided by your throughput. It's how every optimization in this course cashes out: quantization (Ch.09) that doubles decode throughput halves cost-per-token; a prefix-cache hit (Ch.12) that skips a prefill is free tokens; a preemption storm (Ch.11) that wastes compute raises it. Tracking cost-per-token (and cost-per-request) turns "is this faster" into "is this cheaper," which is the question that ultimately funds the service.

### Tracing the request lifecycle

Aggregate metrics show the fleet; a **trace** shows one request's journey: arrival → queue (how long?) → prefill (how long, how many tokens?) → each decode step → detokenize → done. When a specific request was slow, the trace tells you *which phase* — was it stuck in the queue (admission/capacity), slow to prefill (long prompt, cache miss), or slow to decode (big batch)? Distributed tracing across the front end, scheduler, and engine (the process split of Ch.15) is what makes a single slow request debuggable instead of a mystery.

### Evals: the minimum kit

Latency metrics don't catch *quality* regressions — and this course has a lever that silently trades quality for speed: quantization (Ch.09). So observability for a serving stack must include **evals**: a golden set of task-representative prompts with known-good outputs, run before and after any change (a new quantization, a model update, a template change from Ch.03), scoring correctness/quality — not just "did it return 200." The minimum kit: a few dozen prompts spanning your real tasks (including the hard ones — math, code, long-context), an automated score, and a regression gate. You ship the eval kit *with* the service, because "faster" that's secretly "worse" is the most expensive bug in production.

### What a healthy trace looks like

Know your baseline so you can spot the anomaly. A healthy serving fleet shows: TTFT within SLO at your load, TPOT flat and within SLO, preemption rate ≈ 0, prefix-cache hit-rate at whatever your workload supports (high for chat/agents, low for unique prompts), KV usage high but not pinned at 100%, queue depth low and stable, and goodput near 100%. Deviations map to causes: TTFT up + cache hit-rate down → prefix-caching/routing broke; TPOT up + batch size up → over-batching; preemption > 0 + KV pinned → over-admission; queue climbing → need capacity. The dashboard is only useful if you know what "healthy" reads like.

### Two engines, one metric surface

Verified in both. **Agreement (load-bearing):** both track the inference-specific metric set — TTFT, inter-token latency / throughput, running-batch and queue counts, KV-cache usage, and prefix-cache hit-rate — and export it (Prometheus). vLLM in `v1/metrics/` (`SchedulerStats`, `IterationStats`); SGLang in its scheduler's metrics reporter plus the tokenizer front end. (One asymmetry: preemption count is exported by vLLM (`num_preempted_reqs`); at these SHAs SGLang treats preemption as an opt-in priority-scheduling mechanism, not an exported metric.) **Divergence (surface & structure, will rot):** which exact metrics, histogram buckets, label sets, and where each is computed differ and change every release. The *metric set that matters for inference* — two latencies, goodput, and the cache/preemption/KV "why" metrics — is the durable concept; the specific Prometheus surface is what moves.

---

## Real-system notes

- **vLLM** — `vllm/v1/metrics/` @ `ae098ab`: `SchedulerStats` (per-iteration snapshot: running/waiting counts, KV usage, prefix-cache stats) and `IterationStats` (TTFT `time_to_first_tokens_iter`, ITL `inter_token_latencies_iter`, `num_preempted_reqs`), surfaced through a Prometheus logger and a logging logger. This is the reference for "which metrics an engine should expose."
- **SGLang** — metrics in `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` @ `52c6e27` (`last_gen_throughput`, `last_input_throughput`, `cache_hit_rate`, `num_running_reqs`) plus TTFT tracked in `tokenizer_manager.py` and histogram utilities in `utils/gauge_histogram.py`. Same metric intent, different module layout.
- **External** — TTFT / TPOT / ITL are the standard serving vocabulary; **goodput** as the SLO-attainment metric was sharpened by the DistServe work (Zhong et al., 2024). Tracing typically rides on OpenTelemetry; dashboards on Prometheus + Grafana. Ask your agent for the current metric names and a starter Grafana board for your engine.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Tracking one latency.** A single "latency" number hides the TTFT-vs-TPOT trade and misdiagnoses every slowdown. *Fix: track TTFT and TPOT/ITL separately, plus end-to-end (this chapter).*
- **Optimizing throughput, ignoring goodput.** Maximizing tokens/sec by over-batching wrecks the latency SLO while the throughput graph looks great. *Fix: goodput (SLO-attainment) is the target when you have a latency budget (this chapter, Ch.17).*
- **No quality eval in the pipeline.** Latency dashboards can't see a quantization or template regression. *Fix: ship a golden-set eval and gate changes on it (this chapter, Ch.09).*
- **Blind to the "why" metrics.** Watching only latency leaves you guessing when it moves. *Fix: dashboard cache hit-rate, preemption rate, KV usage, and queue depth alongside latency (Ch.11, Ch.12).*
- **No per-request tracing.** Aggregate metrics can't explain one slow request. *Fix: trace arrival→queue→prefill→decode→done across the Ch.15 process split (this chapter).*

---

## Pair with your agent

- *"Instrument my endpoint to report TTFT and TPOT separately (p50/p95/p99), plus throughput and goodput against a 500ms-TTFT / 50ms-TPOT SLO. Show me a load sweep where throughput rises but goodput falls."*
- *"Open `references/vllm/vllm/v1/metrics/stats.py` (`SchedulerStats`, `IterationStats`) and map each field to the chapter it comes from. Then find the same metrics in `references/sglang/.../metrics_reporter.py`."*
- *"Build my first-response dashboard: TTFT, TPOT, cache hit-rate, preemption rate, KV usage, queue depth. Simulate a prefix-cache-routing regression and show which panel catches it."*
- *"Compute cost-per-token for my deployment (GPU $/hr ÷ tokens/sec), then show how FP8 KV (Ch.09) and prefix caching (Ch.12) each move it."*
- *"Build a minimum eval kit: 30 task-representative prompts (chat/math/code/long-context) with a scorer, and a regression gate I run before and after quantizing (Ch.09)."*

---

## What's next

You can now see what the service is doing. Ch.17 turns that visibility into decisions: **cost, latency, and model strategy** — the throughput-vs-latency frontier you've been circling since Ch.05, how to pick a batch size against an SLO, when to substitute a cheaper model or a deterministic tool for an LLM call, and how the whole cost model (Ch.01's roofline through Ch.16's cost-per-token) composes into a serving strategy you can defend on a budget.
