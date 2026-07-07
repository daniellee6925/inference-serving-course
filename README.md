# Production LLM Inference & Serving — a source-grounded course

A 22-chapter (+ Ch.00) expert course on how production LLM inference and serving
systems actually work — the dozen load-bearing mechanisms every serving stack
converges on (KV cache, continuous batching, paged memory, speculative decoding,
the scheduler, prefix caching, constrained decoding, quantization, multi-GPU
parallelism, …), written as a durable *skeleton* you read once and reason against.

It is designed to be read **with an AI coding agent at your side**: the course
carries the concepts that age slowly; your agent supplies the volatile half —
today's kernels, flags, GPU SKUs, prices, and the benchmark script you run on your
own hardware. See `course/00-how-to-use-this-course.md`.

## Source-grounded (expert) mode

This course does not just describe the patterns — it **embeds real source excerpts**,
pinned to specific commits of two reference engines, and cites files inline:

- **vLLM** @ `ae098ab`
- **SGLang** @ `52c6e27`

Every source-embedding chapter passed an independent, fresh-context adversarial
**verify gate** (excerpt faithfulness, annotations checked against the whole
function, every cross-system "both do X" claim verified in *both* sources, and
quantitative claims recomputed). The review record and the durable failure
taxonomy are in `docs/inference-course-review.md`.

## Audience

A **comfortable engineer going deep**: at home in Python, with ML basics, willing
to read a little math and rent a GPU when a laptop won't do.

## Layout

```
course/    22 chapters + Ch.00. Read in order, or jump by goal (Ch.22 helps you choose).
docs/      Review record and any written deliverables.
references/ (gitignored) The pinned vLLM + SGLang clones the excerpts cite.
```

## Reconstructing the references (optional)

The chapters read fine without them, but to open the cited source yourself:

```sh
mkdir -p references && cd references
git clone https://github.com/vllm-project/vllm.git   && git -C vllm  checkout ae098ab
git clone https://github.com/sgl-project/sglang.git  && git -C sglang checkout 52c6e27
```

Then any `file.py @ ae098ab` citation in a chapter points at a real line you can read.
