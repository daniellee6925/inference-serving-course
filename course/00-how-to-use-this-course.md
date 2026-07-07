# Chapter 00 — How to use this course

## TL;DR

You are about to learn how production LLM inference and serving systems actually work — not by reading a million lines of CUDA, but by walking a map of the patterns that every serving stack converges on and pairing with your own AI agent to run them on real hardware. This chapter explains why the course is written the way it is, how to read it with an agent at your side, and how each chapter turns into something you can measure on your own GPU. Everything is about you and the model you want to serve.

---

## Why this course exists

By early 2026 the gap between "I can call an LLM API" and "I understand what happens when I call it" had become the most expensive gap in the field. Teams ship a serving stack, watch their GPU bill, and cannot say why p99 latency spikes at 30 concurrent users, why throughput collapses on long prompts, or whether quantizing would help or just lose accuracy. The frameworks — vLLM, SGLang, TensorRT-LLM, llama.cpp — are extraordinary, but they hide the very mechanisms you need to reason about cost and latency. They are a great place to *run* a model and a poor place to *learn* what a model run costs.

This course is the map underneath those frameworks. After reading the source of the systems everyone benchmarks against, one thing becomes obvious: they share the same dozen ideas. A KV cache. Continuous batching. Paged memory. Speculative decoding. A scheduler that interleaves prefill and decode. Different names, different kernels, the same patterns. This course is those patterns, written down once, in plain language, with the trade-offs that decide your bill.

Learning this way — a sharp question, an AI partner that explains and writes the benchmark script, you reading the numbers — is roughly ten times faster than watching a talk or skimming a README. The bottleneck is no longer information. It is knowing what to measure and what to ask.

## What this course is, and is not

This course is a **spine** — the load-bearing mechanisms, trade-offs, and decisions that show up in every serious serving stack. It has diagrams, vocabulary, failure modes, and notes from real systems. It is opinionated about *what to think about* and almost silent about *which kernel to import*.

This course is **not** a step-by-step tutorial. There is no "first `pip install` this, then paste that." It is not tied to one framework — the patterns are the same whether you end up on vLLM, SGLang, or your own loop. It is not a reference manual; when you need an exact flag or the current price of an H100-hour, ask your agent or read the docs.

The details that age fastest — kernel names, framework flags, GPU SKUs, today's tokens-per-second — are exactly the details your agent has the freshest answers about. The details that age slowest — why decode is memory-bound, why the KV cache dominates your memory budget, why batching trades latency for throughput — are what you will find here. I give you the skeleton; your agent helps you put the muscles on it.

## How to actually use this course

This course assumes you can run code and read a little math (see the audience note below). The difference between readers is which prompts you give your agent. To understand a chapter:

- *"Explain this chapter, defining each term the first time you use it. Then give me one number I should be able to estimate by the end."*
- *"What's the one mechanism I should remember from this chapter, and what breaks if I ignore it?"*
- *"What's a question I should be asking that I haven't?"*

To make it real on your own hardware — the move that matters most in this domain:

- *"I just read about [mechanism X]. Write the smallest script that lets me **measure** it on my GPU — latency, memory, or throughput — and explain each line as you write it."*
- *"Here are the failure modes [Ch.NN] warned about. Look at the serving config we set up earlier and tell me which are already a risk."*

To ground an idea in a real system:

- *"Forget my setup for a moment — show me how vLLM (or SGLang, or llama.cpp) does this, and what we should borrow."*

To test yourself:

- *"Quiz me on this chapter. Five questions, easy to hard, and make me estimate at least one number."*

In this domain, a script that *measures* always beats an explanation that asserts. Your agent writes it; your job is to run it, read the numbers, and ask the next question.

## How this turns into your own serving stack

Personalization happens chapter by chapter. By Ch.02 you have a decode loop that turns one forward pass into generated text. By Ch.04 it has a KV cache and you can see the memory cost in a chart. By Ch.05 it batches many requests and your tokens-per-second climbs. By Ch.16 you can read TTFT, throughput, and goodput off a trace. By Ch.22 — the final chapter — you sit down with your agent, walk a design canvas, and decide what *your* serving stack needs and what it does not.

You can flip the order. If you already know your goal — serve one model cheaply, hit a latency SLO, fit a 70B on the GPUs you have — pick the chapters that match and treat the rest as background. Ch.22 will help you choose.

The goal is not to finish the course. The goal is to serve a model you actually wanted to serve, and to know why every millisecond and every gigabyte goes where it goes.

## Who this is for

This course assumes you are a **comfortable engineer going deep** — at home in Python, with ML basics, willing to read a little math and rent a GPU when a laptop won't do. It is for:

- The **infra/platform engineer** who owns the inference bill and wants to stop guessing.
- The **ML engineer** who can train a model but treats serving as a black box.
- The **founder or PM** technical enough to read code, evaluating whether to self-host or buy.
- The **systems-curious** who wants to understand the most performance-sensitive software being written today.

It does not assume you have written a CUDA kernel. It does assume you will run the scripts your agent writes and read the numbers honestly.

---

## The shape of the rest of the course

Twenty-one more chapters, building from the smallest mechanism up to the largest concern.

- **Ch.01–04** — the mechanism: one forward pass → the decode loop → tokenization as the input contract → the KV cache.
- **Ch.05–08** — throughput: batching → paged KV memory → attention kernels → speculative decoding.
- **Ch.09–10** — fitting the model: quantization → multi-GPU serving (tensor/pipeline/expert parallelism).
- **Ch.11–14** — the serving loop: the scheduler → prefix caching → constrained decoding → long context.
- **Ch.15–17** — the production layer: serving infrastructure → observability → cost, latency, and the SLO frontier.
- **Ch.18–19** — quality and ops: safety and adversarial inputs → operations.
- **Ch.20–21** — the edge: multimodal/encoder-prefill serving → frontier serving.
- **Ch.22** — a design canvas for your own serving stack.

The chapters are ordered so each one only assumes what came before. With a clear goal you can skim what does not apply yet and come back when it does.

## Reference systems used throughout

When a chapter says "real-system note," it points at one of these:

- **vLLM** — the canonical open serving engine: continuous batching, PagedAttention, prefix caching.
- **SGLang** — RadixAttention for prefix reuse and a strong structured-output frontend.
- **llama.cpp** — CPU/edge inference and quantization that runs on the laptop in front of you. This is the one that keeps the course *buildable*.

The source excerpts you'll see are pinned to **vLLM** and **SGLang** (the two engines this course reads line-by-line); **llama.cpp** is the laptop foil for feeling a mechanism on hardware you own. When you want the NVIDIA performance ceiling — TensorRT-LLM's compiled engines and fused kernels — ask your agent; it's the right reference for "how fast can this get," but this course grounds its claims in the two engines above.

These are not templates to copy. They are sanity checks. If a pattern in this course matches what two or more do in production, it is load-bearing. If only one does it, the trade-offs are spelled out.

---

## One last thing before you start

You do not need permission to start, and you do not need a cluster. The first thing you run — one forward pass, one token — fits on a laptop with a small model. Once that token is in front of you and you understand exactly where it came from, the next mechanism becomes obvious.

The people running the cheapest, fastest inference today are not the ones who memorized the most flags. They are the ones who understood the mechanism first and measured everything after. This course exists so that when you are staring at a latency graph, you know what to ask.

Turn the page. Ch.01 is one forward pass.
