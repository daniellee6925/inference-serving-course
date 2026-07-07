# Chapter 19 — Operations

## TL;DR

A correct, fast, safe engine is not yet a *service you can run*. Operations is what keeps it up: **health checks** that distinguish a live HTTP process from a wedged GPU loop, **rollouts** that deploy a new model or version behind a canary and roll back fast when the eval or the metrics regress, **capacity planning** that sizes the GPU fleet from the KV formula (Ch.04) and the goodput curve (Ch.16), **autoscaling** that adds replicas (Ch.15) as demand climbs, and an **incident runbook** that maps a symptom to the Ch.16 metric that explains it to the fix. This is the forward-deployed reality of owning inference — where everything you built in Ch.01–18 meets the pager, and where the difference between a demo and a service is measured in how gracefully it survives a bad deploy, a traffic spike, and a 3am incident.

---

## Why this matters

The engine is the easy part; keeping it serving is the job. Production inference fails in operational ways that no amount of kernel tuning prevents: a deploy that silently ships a worse-quality quantization (Ch.09), a traffic spike that outruns a fixed replica count, a GPU-loop hang that a naive health check reports as "healthy," a model update with no rollback path. Operations is the discipline that turns the metrics of Ch.16 and the strategy of Ch.17 into a service that stays up — and it's usually where a fast prototype dies on contact with real, sustained, unattended load. Owning inference means owning the runbook, not just the model.

---

## The concept

### The forward-deployed reality

Running inference in production means it must stay up unattended, survive deploys without dropping requests, scale with demand you don't fully control, and recover from incidents fast. Whether you're a **forward-deployed engineer** shipping a bespoke system with a customer or a platform team running a hosted fleet, the operational surface is the same: health, deploys, capacity, incidents. This chapter is that surface — the part of serving that has nothing to do with the model and everything to do with the pager.

### Health checks: liveness vs. real readiness

The most common operational trap: a health check that only proves the *HTTP process* is alive, while the GPU loop behind it is wedged (a hung kernel, a deadlocked scheduler, an OOM'd worker). The process answers "healthy," the load balancer keeps routing to it, and every request hangs. The fix is a health check that exercises the *real path*:

```python
# SGLang — liveness vs. real readiness.
# sglang/.../entrypoints/http_server.py @ 52c6e27

@app.get("/health")                                       # L578 runs a 1-token generation BY DEFAULT; downgradable to
@app.get("/health_generate")                              # L579   pure process-liveness (SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false)
async def health_generate(request: Request) -> Response:  # L580 /health_generate ALWAYS runs a real generation → proves the GPU loop works
```

`/health_generate` always runs an actual generation end-to-end, confirming the whole pipeline (tokenizer → scheduler → GPU loop → detokenizer, the Ch.15 process chain) is alive — not just that the front-end process responds. (`/health` runs the same one-token generation *by default* too, but can be downgraded to a pure process-liveness check via `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`.) The lesson holds either way: **wire your load balancer to a check that exercises the real path**, so you never route traffic to an engine whose GPU loop has wedged while its HTTP process stays up. (vLLM is **not** analogous here: its `/health` → `check_health()` is a local error-flag read — `if self.errored: raise self.dead_error` in `async_llm.py` — which catches an engine that has *already* errored, but **not** one whose GPU loop is silently wedged. It has no generation-based readiness probe like `/health_generate` at this SHA. So "exercise the real path" is a **SGLang** property here; on vLLM you'd wire readiness to an external probe that issues a tiny real completion.)

### Rollouts: canary and drain

Deploying a new model, a new quantization (Ch.09), or a new engine version is the riskiest routine operation, because a regression can be in *quality* (invisible to latency metrics) or *performance*. The safe pattern:

1. **Canary** — route a small fraction of traffic to the new version.
2. **Watch** — the Ch.16 metrics (goodput, TTFT/TPOT) *and* the quality eval (Ch.09/Ch.16 golden set), not just "did it return 200."
3. **Promote or roll back** — expand to full traffic if clean; revert instantly if not.

New replicas spin up on the new version while old ones **gracefully drain** (Ch.15 — finish in-flight requests before terminating), so no request is dropped mid-deploy. The whole rollout is gated on metrics and evals, because the most dangerous deploy is the one that's faster and worse.

### Rollback: keep the old version warm

The corollary: you must be able to **roll back fast**. A quality regression from a model update, or a latency regression from a config change, needs a one-step revert to the last known-good version — which means keeping it deployable (warm, or at least fast to restore). Teams that can't roll back quickly are forced to debug in production under load; teams that can revert first and diagnose after. Rollback speed is an operational SLO of its own.

### Capacity planning

How many GPUs do you need? The answer composes two earlier chapters: the **KV formula** (Ch.04) tells you how many concurrent requests fit per GPU at your context length and precision, and the **goodput curve** (Ch.16) tells you the request rate each replica sustains within your SLO. Divide expected peak traffic by per-replica goodput, add headroom for spikes and failures, and you have a replica count and a GPU budget. Capacity planning is where Ch.04's memory math and Ch.16's measured throughput become a purchase order — and where "it worked in the demo" becomes "it holds at peak."

### Autoscaling

Traffic isn't constant, so a fixed replica count either wastes money (over-provisioned) or drops requests (under-provisioned). **Autoscaling** adds and removes replicas (Ch.15) based on load signals — queue depth, goodput, or utilization (Ch.16). The subtlety unique to LLM serving: replicas are *expensive and slow to start* (loading tens of GB of weights takes time), so autoscaling must be predictive or well-buffered, not purely reactive — you can't spin up a GPU in the 200ms before the queue overflows. Scale on a leading indicator (rising queue depth) with warm headroom, not on a lagging one (already-missed SLOs).

### The incident runbook

When the pager fires, a runbook turns panic into procedure — and for inference, the runbook *is* the Ch.16 metrics mapped to causes and fixes:

- **TTFT spiking** → check prefix-cache hit-rate (Ch.12: routing broke?), queue depth (need replicas?), and prefill load. → add capacity, fix routing, or chunk prefill (Ch.11).
- **TPOT spiking** → check batch size (over-batching?) and preemption rate (Ch.11: over-admission?). → lower the dial (Ch.17) or add capacity.
- **Preemption > 0 + KV pinned** → over-admission for the pool (Ch.04). → tighten admission, add replicas.
- **Quality complaints** → run the eval (Ch.16); suspect the last deploy (Ch.09 quantization?). → roll back.

Every row is a symptom → a metric from Ch.16 → a cause from an earlier chapter → a fix. The runbook is the whole course, operationalized — which is why observability (Ch.16) had to come first.

### The minimum ops kit for day one

You don't need a mature SRE org to run inference well, but you need a floor: a **readiness health check** wired to the load balancer, a **dashboard** of the Ch.16 metrics, an **eval gate** on deploys (Ch.09/16), a **rollback path** to the last-good version, a **capacity model** (Ch.04 + Ch.16), and a **one-page runbook** mapping the top symptoms to fixes. Ship these with the service, not after the first incident — because the first incident is when you'll wish you had them.

### Two engines, one operational surface

Verified in both (architecturally). **Agreement (load-bearing):** both expose health/liveness endpoints (SGLang `/health` + `/health_generate`; vLLM health instrumentation), support graceful drain of in-flight requests (Ch.15), and emit the metrics (Ch.16) that capacity planning, autoscaling, and incident response run on. **Divergence (deploy tooling, will rot):** the rollout/canary/autoscaling machinery mostly lives *outside* the engine — Kubernetes, Ray Serve, a control plane (Paperclip-style) — and differs by deployment, as do the exact health endpoints and drain semantics. The operational *surface* — health, rollout, capacity, incident — is the durable concept; the tooling that implements it is what changes and what you assemble around the engine.

---

## Real-system notes

- **SGLang** — `python/sglang/srt/entrypoints/http_server.py` @ `52c6e27`: `/health_generate` always runs a real one-token generation (guaranteed readiness); `/health` runs one too by default but is downgradable to pure process-liveness (`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`, `environ.py` L765). The point — check the *real path*, not just the process — is the reference for wiring load balancers correctly.
- **vLLM** — health instrumentation in `vllm/entrypoints/serve/instrumentator/health.py` @ `ae098ab` calls `check_health()`, which at this SHA is a local error-flag check (`if self.errored: raise`), **not** a real-path generation — so pair it with an external readiness probe that issues a tiny completion if you need to catch a *wedged* (not merely errored) loop. The OpenAI server supports graceful shutdown/drain, and metrics (Ch.16) feed external autoscalers; the rollout/scaling machinery is deployment-plane (K8s/Ray), not engine-internal.
- **Paperclip / control planes** — for fleets, the operational layer becomes a control plane: durable state, deploys, budgets, audit logs, on-call integration. That's the reference class for multi-tenant, multi-model operations at scale, and where the runbook becomes automation.
- **External** — this is SRE applied to inference: SLOs, error budgets, canary analysis, runbooks (Google's SRE book is the canon); Kubernetes / Ray Serve / KServe for orchestration and autoscaling. Ask your agent for a starter deploy + autoscaling recipe for your substrate.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Liveness check hiding a wedged engine.** A `/health` that only proves the HTTP process is up keeps routing traffic to a dead GPU loop. *Fix: wire the load balancer to a readiness check that runs a real generation (`/health_generate`) (this chapter).*
- **Deploys gated only on "it returns 200."** A faster, lower-quality quantization ships unnoticed. *Fix: canary + a quality eval gate + metric watch before promotion (Ch.09, Ch.16).*
- **No fast rollback.** A bad deploy forces debugging in production under load. *Fix: keep the last-good version warm and revert in one step (this chapter).*
- **Reactive autoscaling.** Scaling only after SLOs are missed, when replicas take minutes to load weights. *Fix: scale on a leading indicator (queue depth) with warm headroom (this chapter, Ch.15).*
- **No runbook.** Every incident is solved from scratch under pressure. *Fix: a one-page symptom→metric→cause→fix runbook built on the Ch.16 metrics (this chapter).*

---

## Pair with your agent

- *"Set up a readiness health check that runs a real generation, and show me why a liveness-only check would keep routing to a hung engine. Wire it to a mock load balancer."*
- *"Design a canary rollout for a new quantization of my model: traffic split, the goodput + eval metrics to watch (Ch.09/16), and the promote/rollback criteria."*
- *"Build my capacity model: from my model's KV formula (Ch.04) and a measured goodput curve (Ch.16), tell me replicas needed for my peak traffic + 30% headroom, and the GPU budget."*
- *"Write my incident runbook: top 5 symptoms (TTFT/TPOT spikes, preemption, quality complaints, queue growth), the Ch.16 metric to check for each, and the fix."*
- *"Explain why LLM autoscaling must be predictive: how long does loading my model's weights take, and how much warm headroom do I need to avoid dropping requests on a spike?"*

---

## What's next

You can now run the service through deploys, spikes, and incidents. The last two content chapters look forward. Ch.20 is **multimodal and encoder-prefill serving** — what changes when inputs are images, audio, or video: a heavy encoder stage in front of the language model, a prefill-dominated cost profile, and new memory and batching considerations. It's the fastest-growing serving workload and the one where the prefill/decode asymmetry you've tracked since Ch.01 tilts hardest toward prefill.

---

*Ch.15–19 were the production layer: the engine became a service (Ch.15), you made it observable (Ch.16), turned that into a cost strategy (Ch.17), hardened it against adversaries (Ch.18), and learned to operate it (Ch.19). Ch.20–21 push to the frontier; Ch.22 is where you design your own.*
