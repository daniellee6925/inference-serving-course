# Chapter 13 — Constrained decoding

## TL;DR

Prefix caching (Ch.12) made repeated *input* cheap; this chapter constrains the *output*. Tool-calling, structured extraction, and any "return JSON matching this schema" feature need output that is **guaranteed** to parse — and a raw model produces invalid JSON some fraction of the time. Constrained decoding makes invalid output *impossible* rather than merely unlikely, by masking the sampler: at each step a grammar (compiled from a JSON schema, regex, or context-free grammar) says which tokens are legal next, and every illegal token's logit is set to −∞ before sampling (Ch.01's sampler, now with a constraint). The model can only emit tokens that keep the output valid. Both vLLM and SGLang do this through a per-step lifecycle — advance the grammar state, fill an allowed-token bitmask, apply it to the logits — and both apply that bitmask in the format that originated in `xgrammar`, now a de-facto standard, though each reaches it a different way. This is how structured output is made reliable, and it composes with everything from Ch.01's sampler to Ch.12's cache.

---

## Why this matters

The moment an LLM feeds a downstream system — a function call, a database write, a UI that parses its output — "usually valid" is not good enough. A 2% JSON-parse-failure rate is a 2% outage for that feature, and it grows with output length. Constrained decoding drops that to zero *by construction*: the output cannot violate the schema because the tokens that would violate it are never sampleable. That reliability is why every serious tool-calling and structured-extraction stack uses it. But it isn't free — computing the mask over a 100k+ vocabulary every step can stall the GPU, and aligning a character-level grammar to a token vocabulary is genuinely hard — so understanding the mechanism tells you where it costs and where it can even *speed things up*.

---

## The concept

### The reliability problem

Ask a model for JSON and it will *usually* produce valid JSON — but "usually" fails at scale. It forgets a closing brace, adds a trailing comma, emits prose around the object, or hallucinates a field. Prompting and retries reduce the rate; they don't eliminate it. Constrained decoding eliminates it: the output is valid because no invalid continuation was ever an option.

### The mechanism: mask the sampler (Ch.01, again)

Recall Ch.01's sampler — it turns logits into a token, and one of its first steps was applying a logits mask (`allowed_token_ids_mask`, disallowed tokens set to −∞). Constrained decoding is that mask, driven by a grammar. At each decode step:

1. The grammar, given the tokens emitted so far, defines the **set of legal next tokens**.
2. Every token *not* in that set has its logit set to −∞.
3. The sampler (Ch.01) draws from what remains — necessarily a legal token.

Because −∞ becomes probability zero after softmax, an illegal token has *no chance* of being sampled, at any temperature. The constraint is exact, not a bias.

### The grammar is a state machine

To know the legal next tokens, the schema/regex/grammar is compiled into a **state machine**: a finite-state machine for a regex, a pushdown automaton for a context-free grammar (JSON is context-free — it nests). The machine's current state determines the allowed token set; after a token is sampled, the machine advances. So constrained decoding is a state machine running in lockstep with the decode loop (Ch.02), gating each step.

### The per-step lifecycle

Both engines implement the same three-beat loop. SGLang's grammar backend is the clearest to read:

```python
# SGLang — constrained decoding: advance the grammar, build the allowed-token mask, apply it to logits.
# sglang/.../constrained/llguidance_backend.py @ 52c6e27

def accept_token(self, token: int):                    # L56 advance the grammar matcher by the chosen token
    self.ll_matcher.consume_token(token)               # L62

def fill_vocab_mask(self, vocab_mask, idx):            # L78 write the bitmask of ALLOWED next tokens for this state
    fill_next_token_bitmask(self.ll_matcher, vocab_mask, idx)   # L79

@staticmethod
def apply_vocab_mask(logits, vocab_mask):              # L92 apply the mask to logits IN PLACE →
    apply_token_bitmask_inplace(logits, vocab_mask)    # L93 (llguidance's; xgrammar-format) disallowed → −∞
```

`accept_token` advances the state by the token just chosen; `fill_vocab_mask` writes a **bitmask** over the vocabulary (one bit per token: legal or not) for the new state; `apply_vocab_mask` stamps that bitmask onto the logits. Then Ch.01's sampler runs on the masked logits.

### Both engines converge on the same bitmask *format*

Here is the striking cross-system fact — and it's subtler than "same kernel," which is itself the lesson. Both engines apply a token **bitmask** to the logits, and that bitmask's *format and Triton algorithm originated in one project, `xgrammar`* (mlc-ai), which has become a de-facto standard. But they *reach* it three different ways:

- **vLLM imports xgrammar's function directly** — `xgr.apply_token_bitmask_inplace`:

```python
# vLLM — calls xgrammar's kernel directly, applied in the model runner.
# vllm/v1/structured_output/utils.py @ ae098ab

import xgrammar as xgr                                                          # L27 (LazyLoader-bound at runtime)
xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor) # L160 disallowed tokens → −∞
```

- **SGLang's `xgrammar` backend** dispatches to a compiled `apply_token_bitmask_inplace_cuda` (from `sgl_kernel`) on the CUDA path, falling back to a **vendored** copy of xgrammar's Triton kernel (`apply_token_bitmask_inplace_triton`, carrying the attribution `# Adapt from .../xgrammar/.../apply_token_bitmask_inplace_triton.py`). So on the common CUDA path it runs neither xgrammar's function nor the Triton port — it runs its own compiled kernel.
- **SGLang's `llguidance` backend** (the excerpt above) uses **llguidance's** compatible `apply_token_bitmask_inplace`, imported from `llguidance.torch` — *not* xgrammar's.

Same bitmask *format*, three *implementations* — direct import, vendored port, compatible reimplementation. That a mask-application primitive from one project spread this way across two engines and a third library is the real signal: the format is load-bearing and settled, even though no two of them literally share the binary. (It's a sharper version of Ch.01's FlashInfer story — convergence on a *standard*, not necessarily on a single kernel call. And a warning: the same function *name* in two codebases is not proof of the same *code* — you have to trace the import.)

### The cost: the mask is on the critical path

The mask is a vector over the *entire* vocabulary (100k+ entries), recomputed for every sequence at every step. Done naively, that computation stalls the GPU waiting on the CPU to produce the mask. This is why fast grammar backends exist: they **precompute** per-state masks, cache context-independent token sets, and overlap mask computation (on CPU) with the GPU's forward pass, so the mask is ready when the logits are. The performance difference between a naive and an optimized grammar backend is large — constrained decoding done wrong can halve your throughput, done right costs almost nothing.

### Tokenization vs. grammar: the genuinely hard part

The deep difficulty is the one from Ch.02/Ch.03: grammars are defined over **characters/text**, but the model emits **tokens**, and a token can straddle a grammar boundary (a single token might be `",` — a quote *and* a comma). Compiling a character-level grammar into a correct *token-level* mask — accounting for every way the tokenizer could chunk the valid continuations — is the core engineering of Outlines, XGrammar, and llguidance. A related optimization falls out of it: **jump-forward decoding** — when the grammar forces the next tokens (after a `{` in a fixed schema, the next characters *must* be a specific key), the engine can emit them directly and skip the model entirely for those positions, a real latency win on rigid schemas.

### Valid by construction ≠ correct

One honest caveat. Constrained decoding guarantees the output *parses* and *matches the schema* — it removes an entire class of format errors. It does **not** guarantee the content is right: a schema-valid JSON object can still hold a wrong answer, and forcing the model down a grammar path can even push it toward a worse answer than it would freely give. Constrained decoding is a reliability tool for *structure*, not a correctness tool for *content*. Pair it with evals (Ch.16), don't treat it as a substitute.

### Two engines, one lifecycle

Verified in both. **Agreement (load-bearing):** both compile a schema/regex/grammar to a state machine and run the per-step lifecycle — advance state (`accept_token` / `accept_tokens`), fill an allowed-token bitmask (`fill_vocab_mask` / `fill_bitmask`), and apply that bitmask to the logits (disallowed → −∞) in the xgrammar-originated format — implemented per backend (vLLM imports xgrammar's kernel; SGLang-xgrammar dispatches to a compiled CUDA kernel with a vendored Triton fallback, SGLang-llguidance uses llguidance's compatible function) — before Ch.01's sampler runs. **Divergence (backend menu & placement, will rot):** which grammar backends each ships — SGLang has `xgrammar`, `outlines`, `llguidance`, and a reasoner backend; vLLM has `xgrammar`, `outlines`, `guidance`, and `lm_format_enforcer` — plus where the mask is applied and which optimizations (jump-forward, mask caching) each implements. The mask-the-logits-by-grammar-state *lifecycle* is the durable concept; the backend zoo is what changes.

---

## Real-system notes

- **SGLang** — `python/sglang/srt/constrained/` @ `52c6e27`: a `base_grammar_backend.py` interface with `xgrammar_backend.py`, `outlines_backend.py`, `llguidance_backend.py`, and `outlines_jump_forward.py` (the jump-forward optimization). The lifecycle is `accept_token` → `fill_vocab_mask` → `apply_vocab_mask` (llguidance's `apply_token_bitmask_inplace`, L93); the `xgrammar_backend.py` instead dispatches to a compiled `apply_token_bitmask_inplace_cuda` (`sgl_kernel`) on CUDA, with a vendored xgrammar Triton kernel (`apply_token_bitmask_inplace_triton`) as the non-CUDA fallback.
- **vLLM** — `vllm/v1/structured_output/` @ `ae098ab`: `backend_xgrammar.py` (`compile_grammar` L78, `allocate_token_bitmask` L128, `accept_tokens` L152, `fill_bitmask` L195), plus `backend_outlines.py`, `backend_guidance.py`, `backend_lm_format_enforcer.py`. The mask is applied in the model runner via `apply_grammar_bitmask` → `xgr.apply_token_bitmask_inplace` (`structured_output/utils.py` L160).
- **The grammar libraries are external** (papers/projects, not the engine code): Outlines (Willard & Louf, 2023, regex→FSM guided generation); XGrammar (2024, fast CFG with a precomputed adaptive token mask); guidance / llguidance. Ask your agent which backend is fastest and most complete for your schema type today — it moves.

---

## Common failure cases

*These failures are durable; their fixes evolve fastest — each names the pattern and leaves current specifics to you and your AI partner.*

- **Prompting for JSON instead of constraining it.** "Respond only in JSON" still fails a fraction of the time and worse at length. *Fix: constrained decoding makes invalid output impossible by construction — mask the sampler, don't beg the model (this chapter).*
- **A naive grammar backend stalling the GPU.** Computing a full-vocab mask per step on the critical path can halve throughput. *Fix: an optimized backend (XGrammar/llguidance) that precomputes and overlaps mask compute; measure the throughput hit (this chapter, Ch.16).*
- **Token/grammar-boundary bugs.** A hand-rolled or incomplete grammar misses continuations that span token boundaries, over-constraining or corrupting output. *Fix: use a real grammar backend that compiles to a token-level mask; don't build your own (this chapter).*
- **Confusing valid with correct.** Trusting schema-valid output as right, or over-constraining into worse answers. *Fix: constrain structure, eval content; keep the schema as loose as the task allows (this chapter, Ch.16).*
- **Ignoring jump-forward opportunities.** Sampling tokens the grammar already forces wastes model steps on rigid schemas. *Fix: a backend with jump-forward decoding emits forced tokens directly (this chapter).*

---

## Pair with your agent

- *"Take a JSON schema and run my model on 500 prompts with and without constrained decoding. Show me the parse-failure rate (some % without, 0% with) and the throughput difference."*
- *"Open `references/sglang/.../constrained/llguidance_backend.py` (`accept_token`, `fill_vocab_mask`, `apply_vocab_mask`) and `references/vllm/vllm/v1/structured_output/utils.py` (`xgr.apply_token_bitmask_inplace`). Show me that both apply an xgrammar-*format* token bitmask, and trace where each `apply_token_bitmask_inplace` actually comes from — vLLM's from the `xgrammar` package, SGLang-llguidance's from `llguidance.torch`, SGLang-xgrammar's from its own vendored Triton port."*
- *"Explain the token-vs-character problem: give me a token that straddles a JSON boundary and show why compiling a character grammar into a token mask is hard."*
- *"Demonstrate jump-forward decoding on a fixed schema: show which output positions are forced by the grammar and could skip the model entirely."*
- *"Constrain my model to a schema and then find a case where the output is valid but wrong — proving constrained decoding fixes structure, not content."*

---

## What's next

You've now controlled the loop's inputs (Ch.12) and outputs (Ch.13). The next chapter stretches the loop itself: Ch.14 is **long context** — what changes when prompts reach tens or hundreds of thousands of tokens. The KV cache (Ch.04) and attention kernel (Ch.07) you built assumed a modest sequence; at long context the KV cache dominates memory, attention dominates time, position encodings need extending (RoPE scaling), and prefill becomes a chapter of its own. It's where every cost model in this course gets stress-tested at the scale that breaks naive implementations.
