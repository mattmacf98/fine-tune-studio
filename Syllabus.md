# Fine-Tune Studio — Week-by-Week Syllabus (Mac-Primary / CUDA-Burst Edition)

**A graduate-level, project-based course in small language model fine-tuning (0.5B–8B), designed to run on a 16GB M4 Mac with disciplined, near-zero cloud GPU bursts.**

15 weeks · 2 sessions/week · 1 lab block/week · ~10–15 hrs/week.

---

## The Operating Model (read this first)

You own the wrong hardware for CUDA and the right hardware for almost everything else. The whole course is built around that fact. Every training week runs in four phases:

- **Phase A — Local dev (free, on M4).** Write the pipeline, chat template, and config. Prove correctness on 8–16 examples via MPS/CPU (or MLX for speed). ~80% of your hours live here and cost nothing.
- **Phase B — Local parity (free).** Prove your from-scratch `core/` implementation matches the framework at 0.5B on CPU/MPS. Parity is a correctness property; it holds at any scale, so you never pay to prove it.
- **Phase C — Cloud burst (cents–dollars).** `rsync` the *already-debugged* repo to a rented GPU, run the real multi-hour job, pull artifacts back, **kill the pod.** Per-second billing + free egress means you pay for minutes, not months.
- **Phase D — Local eval/quantize/serve (free, on M4).** Evaluation, GGUF conversion, llama.cpp/Ollama serving, and analysis all run natively on the Mac.

**The golden rule:** the cloud GPU is the source of truth, but it is a *batch job you submit and kill*, never a machine you leave running. First cost lever, always: shut the pod down the second the job ends.

### Reference prices (verified 2026, RunPod Community Cloud / Vast.ai)

RTX 4090 ≈ **$0.34/hr** (RunPod) / **$0.29–0.31/hr** (Vast interruptible). A100 80GB ≈ **$1.39/hr** (RunPod) / **~$0.67/hr** (Vast). Both bill per-second with free egress. A debugged QLoRA-7B run of ~2h on a 4090 ≈ **$0.70**.

### Two standing scripts you build in Week 0 and reuse all semester

- `make train-remote` — rsync repo → boot pod → run job → rsync artifacts home → **terminate pod**. One command, no babysitting.
- `make cuda-check` — 5-minute smoke pod that confirms an op/behavior on real CUDA when local MPS is ambiguous.



### The MPS/MLX honesty caveat

Dev-on-Mac / train-on-CUDA means your dev and prod environments differ. You *will* occasionally hit an op that works on CUDA but not MPS. That's normal heterogeneous-infra engineering; the fix is `make cuda-check`, and documenting those gotchas is itself portfolio-worthy.

### Cost map at a glance

Weeks 0–4 and 11: **$0.** Serving-to-GGUF and merging: **$0.** Everything else: cents to a few dollars. Irreducible cloud floor: Week 8 (multi-GPU) ~$8, Week 10 (GRPO) ~$6, capstone ~$15–25. **Whole semester ≈ $40–60 if disciplined.**

---



## Week 0 — Environment, the Local/Cloud Split, and the Skeleton

**Objectives:** Stand up a Mac-native dev environment (MLX + PyTorch/MPS + HF stack) *and* the cloud-burst tooling. Understand unified memory vs VRAM. Establish the monorepo, CI, and the `core/` API contract.

**Phase A (local).** Install: `mlx`, `mlx-lm`, PyTorch (MPS build), `transformers`, `datasets`, `tokenizers`, `llama.cpp`, `ollama`. Write `hardware_probe.py` — report chip, unified memory, MPS availability, and (when run on a pod) CUDA/VRAM/bf16/FP8. Load Qwen2.5-0.5B two ways: via MLX and via PyTorch-MPS; greedy-decode both; log tokens/sec + peak memory.

**Phase C (cloud, one-time).** Rent the cheapest 4090 for ~15 min, boot it, run `hardware_probe.py` there, confirm `make train-remote` round-trips a dummy job and **auto-terminates**. This is the single most important thing you build all semester — get it working now.

**Ships:** repo skeleton, CI, `hardware_probe`, `make train-remote`, `make cuda-check`, dual MLX/MPS inference smoke test. **Tag** `v0.1`**.**

**Pitfalls:** leaving the verification pod running overnight (set a billing alarm today); PyTorch installed without MPS; assuming unified memory = free (the OS + GPU share your 16GB).

**Cost:** ~$0.10.

---



## Week 1 — Transformer Internals for Fine-Tuners

**Objectives:** Identify exactly which parameters fine-tuning touches. Explain GQA/MHA, RoPE, RMSNorm, SwiGLU, the residual stream, and where adapters attach. Separate inference memory (KV-cache, activations) from training memory (activations + gradients + optimizer states).

**Phase A (local, all of it).** Implement a decoder block from scratch in PyTorch (RoPE + GQA + RMSNorm + SwiGLU); load real Qwen2.5-0.5B weights on MPS; match HF logits to < 1e-4. Build `memory_estimator.py`. Everything this week runs comfortably on the M4 — 0.5B in bf16 is ~1GB of weights.

**Reading:** *Attention Is All You Need*; Llama & Qwen2 tech reports; GQA paper; RoPE paper; Kipply "Transformer Inference Arithmetic"; `modeling_qwen2.py` source.

**Stretch:** add a KV-cache; benchmark MPS vs the HF path. Note where MPS lacks an op and how you worked around it (first entry in your gotchas log).

**Ships:** `core/nn/` reference blocks + `memory_estimator.py` (used by the orchestrator all semester). **Tag** `v0.2`**.**

**Pitfalls:** conflating param count with training footprint (Adam moments ≈ 2× params); forgetting activation memory dominates at long context.

**Cost:** $0.

---



## Week 2 — Tokenization & Vocabulary Design

**Objectives:** Explain BPE/Unigram mechanics; measure fertility; extend a vocabulary and resize embeddings without corrupting logits.

**Phase A (local, all of it).** Train a domain BPE tokenizer with the `tokenizers` Rust backend (fast on CPU). Compare fertility vs Qwen2.5's tokenizer on your domain. Extend the vocab, resize embeddings, init new rows (mean-of-subtokens), prove no logit corruption on unchanged tokens — all on MPS at 0.5B.

**Reading:** Sennrich 2016 (BPE); SentencePiece paper; HF `tokenizers` docs; Karpathy tokenizer lecture.

**Stretch:** short LoRA fine-tune (local, MLX) to measure the extended vocab's downstream effect.

**Ships:** **Dataset Management v1** — tokenizer-inspector UI (token boundaries, fertility, cost). **Tag** `v0.3`**.**

**Pitfalls:** LM-head tying wrong after resize; new special tokens colliding with chat-template tokens.

**Cost:** $0.

---



## Week 3 — Dataset Engineering, Cleaning & Deduplication

**Objectives:** Turn raw sources into a curated, versioned, sharded set. Implement quality filtering, MinHash-LSH near-dedup, eval decontamination, PII scrubbing. Understand why contamination is the #1 way benchmarks get faked.

**Phase A (local, all of it).** This week is CPU/RAM-bound — ideal for the Mac. Build the pipeline: ingest → normalize → language/quality filter → MinHash-LSH near-dedup → decontamination gate → shard to Parquet with a content-hash manifest. Emit per-stage retention stats.

**Reading:** FineWeb/FineWeb-Edu report; SlimPajama dedup writeup; *Textbooks Are All You Need*; `datatrove` docs.

**Stretch:** semantic dedup with embeddings + FAISS (runs on MPS or CPU) vs MinHash.

**Ships:** **Dataset Management v2** — versioned datasets with lineage, dedup reports, a decontamination gate every run must pass. **Tag** `v0.4`**.**

**Pitfalls:** silent eval contamination; over-filtering that kills diversity; non-deterministic shards breaking reproducibility.

**Cost:** $0.

---



## Week 4 — Chat Templates & Instruction-Tuning Data

**Objectives:** Treat the chat template as a correctness-critical interface. Implement prompt-loss masking by hand and match the framework token-for-token. Design and validate an instruction-data schema.

**Phase A (local, all of it).** Implement chat-template rendering + prompt-loss masking by hand; verify it matches TRL's collator exactly on multi-turn samples (TRL imports and runs on Mac — you just won't train big here). Convert a Tülu 3 / OpenHermes slice into your schema + template.

**Reading:** HF chat-templating docs; ChatML spec; Tülu 3 recipe report; TRL `SFTTrainer` collator source.

**Ships:** **Annotation UI v1** — visual overlay of trained-vs-masked tokens; per-base-model template preview. **Tag** `v0.5`**.**

**Pitfalls:** training on the prompt; wrong BOS/EOS; serve-time template ≠ train-time template.

**Cost:** $0.

---



## Week 5 — LoRA From Scratch, Then PEFT  *(Parity Defense #1)*

**Objectives:** Explain low-rank adaptation mechanically before touching `peft`. Reason about rank/alpha/target modules and merge math. Prove your implementation matches the framework.

**Phase A (local).** Implement `LoRALinear` from scratch; inject into your Week-1 block; train Qwen2.5-0.5B on your SFT set via MPS (small, slow, fine).

**Phase B (local parity).** Swap in `peft`; confirm loss-curve parity at 0.5B on MPS. Implement merge; prove merged logits == adapter-applied logits. **This is the defense — done entirely free on the Mac.**

**Phase C (cloud, tiny).** One real 4090 run of your PEFT path to confirm it behaves identically on CUDA and to produce a rank ∈ {4,8,16,64} × target-module ablation faster than MPS would. ~1h on a 4090.

**Reading:** LoRA paper; Aghajanyan *Intrinsic Dimensionality*; `peft` `LoraModel` source.

**Ships:** **Training Orchestration v1** — config-driven SFT+LoRA service that logs metrics, runnable locally (MPS) or via `make train-remote`. **Tag** `v0.6`**.**

**Pitfalls:** wrong target modules; α/r scaling confusion; expecting LoRA to cut activation memory.

**Cost:** ~$0.50.

---



## Week 6 — Experiment Tracking, Config & QLoRA

**Objectives:** Make every run reproducible (Hydra, dual W&B+MLflow, determinism control). Explain int8/int4/NF4/double-quant and why QLoRA makes 7B trainable on modest hardware. **This is the first week where the CUDA stack genuinely diverges from Mac** — `bitsandbytes` NF4 is CUDA-only, so real QLoRA-7B is a cloud burst.

**Phase A (local).** Wrap all training in Hydra configs with a dual W&B+MLflow logging adapter. Write a from-scratch NF4 quantize/dequantize routine (pure Python/NumPy — demystifies the format, no GPU needed). Dry-run the QLoRA config on 0.5B via MLX's built-in quantized LoRA so you understand the mechanics locally.

**Phase C (cloud).** Run real `bitsandbytes` QLoRA-7B on a 4090 (~2h). Produce the LoRA(bf16) vs QLoRA(nf4) quality+memory ablation. Kill the pod.

**Reading:** QLoRA paper; `bitsandbytes` docs; LLM.int8() paper; Hydra docs.

**Ships:** **Experiment Tracking dashboard** — runs, configs, metrics, artifacts, run-diffs. **Tag** `v0.7`**.**

**Pitfalls:** nondeterminism from flash-attn/TF32; paged-optimizer OOM edges; comparing runs with different configs.

**Cost:** ~$1.50.

---



## Week 7 — DoRA, LoRA+, Full FT & the Memory Wall

**Objectives:** Choose among PEFT variants and full FT with reasons. Use gradient checkpointing, bf16/FP8, and `torch.compile` as independent levers.

**Phase A (local).** Implement DoRA and LoRA+ as `core/` variants; reproduce the DoRA directional claim at 0.5B on MPS. Note: **MLX also supports DoRA natively**, so you can cross-check your impl against two references (mlx-lm and peft) — a nice parity bonus.

**Phase C (cloud).** Full fine-tune ≤1.5B with checkpointing + bf16 + `torch.compile` on an A100 (~2h); report each lever's throughput/VRAM delta independently. FP8 stretch needs Ada/Hopper — a brief A100/L40S burst.

**Reading:** DoRA paper; LoRA+ paper; Transformer Engine FP8 docs; PyTorch AMP + `torch.compile` docs.

**Ships:** PEFT method registry (LoRA/QLoRA/DoRA/LoRA+/full by config). **Tag** `v0.8`**.**

**Pitfalls:** DoRA overhead unjustified at small rank; `torch.compile` recompilation blowups; checkpointing the wrong layers.

**Cost:** ~$3.

---



## Week 8 — Multi-GPU: FSDP, DeepSpeed, Accelerate  *(irreducible cloud week)*

**Objectives:** Scale past one GPU. Distinguish data vs sharded-data parallelism and ZeRO stages; pick FSDP vs DeepSpeed with reasons; launch with Accelerate. **There is no Mac path to this — multi-GPU sharding is the point, so it's cloud by definition.**

**Phase A (local).** Write and fully debug the Accelerate config, the FSDP wrapping policy, and the DeepSpeed ZeRO-3 config against a 0.5B single-process CPU/MPS run. Get every config *correct* before you pay for two GPUs.

**Phase C (cloud).** Rent 2×A100 (~2–3h). Run the same full fine-tune under FSDP and under DeepSpeed ZeRO-3; build the throughput/memory/convergence comparison. Get QLoRA working multi-GPU; log the gotchas. Terminate immediately.

**Reading:** ZeRO paper; PyTorch FSDP paper/docs; DeepSpeed config reference; Accelerate docs.

**Ships:** **Training Orchestration v2** — distributed launch + Modal/Ray remote backend wired into `make train-remote`. **Tag** `v0.9`**.**

**Pitfalls:** FSDP+LoRA state-dict saving; mismatched grad-accum across ranks; NCCL timeouts; burning money debugging config *on* the multi-GPU pod instead of locally first.

**Cost:** ~$8 (the semester's single biggest line — worth doing carefully once).

---



## Week 9 — Preference Data & Direct Alignment (DPO, ORPO)  *(Parity Defense #2)*

**Objectives:** Build preference datasets; explain why direct methods replaced RM→PPO for SFT-scale work; implement the DPO loss and match the framework.

**Phase A (local).** Implement the DPO loss from scratch (reference-model logratios, β). Build a preference-pair schema + importer.

**Phase B (local parity).** Match your DPO loss against TRL at 0.5B on MPS — **and** against MLX (mlx-lm-lora supports DPO/ORPO natively), giving you a two-framework parity check for free. **This is the defense.**

**Phase C (cloud).** Real DPO + ORPO runs on your SFT model with UltraFeedback, 4090 (~2h). Compare on your dashboard.

**Reading:** InstructGPT paper; DPO paper; ORPO paper; TRL `DPOTrainer`/`ORPOTrainer` docs; UltraFeedback card.

**Ships:** **Annotation UI v2** — pairwise A/B preference collection emitting DPO-ready pairs; alignment stage in orchestrator. **Tag** `v0.10`**.**

**Pitfalls:** reference-model leakage; length bias (classic DPO failure); SFT vs DPO template mismatch.

**Cost:** ~$1.

---



## Week 10 — Reward Models, GRPO & RLVR  *(mostly-irreducible cloud week)*

**Objectives:** Decide when a reward model is still needed; explain verifiable rewards (RLVR); implement GRPO's group-relative advantage. RL rollouts are compute-heavy — the real runs are cloud, though MLX lets you prototype GRPO locally at tiny scale.

**Phase A (local).** Train a small pairwise reward model on MPS (0.5B, fine locally). Implement the GRPO advantage (group-normalized) by hand. Prototype the *full GRPO loop* at 0.5B using mlx-lm-lora's GRPO (runs on the M4) to confirm your reward function and loop logic before paying for rollouts.

**Phase C (cloud).** Real GRPO with a verifiable reward (GSM8K-style: exact-match + format) on Qwen2.5-1.5B/3B, A100 (~3–4h). Plot reward and pass-rate curves. Confirm your hand-written advantage against TRL here.

**Reading:** DeepSeekMath (GRPO) paper; DeepSeek-R1 report; Tülu 3 RLVR section; TRL `GRPOTrainer` docs.

**Ships:** RLVR training path + reward-function plugin interface. **Tag** `v0.11`**.**

**Pitfalls:** reward hacking; tiny-batch GRPO instability; verifier bugs that reward wrong answers; forgetting to kill a pod mid-rollout.

**Cost:** ~$6.

---



## Week 11 — Evaluation Pipelines & Benchmark Design

**Objectives:** Produce trustworthy evals. Use `lm-evaluation-harness` correctly; run contamination-aware and LLM-as-judge evals with awareness of pitfalls; design a custom decontaminated domain benchmark; compare models with statistics, not vibes.

**Phase A/D (local, all of it).** `lm-eval-harness` runs against any model or endpoint — point it at your local MLX/llama.cpp-served models and evaluate everything you've trained, free, on the Mac. Build the custom benchmark (held-out, decontaminated, rubric) + a bootstrap-CI model-comparison report. LLM-as-judge can call a hosted API or a local served model.

**Reading:** `lm-evaluation-harness` docs + source; HELM methodology; MT-Bench/Chatbot Arena paper; contamination + eval-variance papers.

**Ships:** **Evaluation Dashboard + Benchmark Runner + Model Comparison** — the analytics core, wired to the registry. **Tag** `v0.12`**.**

**Pitfalls:** judge bias (position/verbosity); eval template ≠ train template; single-seed deltas reported as real.

**Cost:** $0 (API-based judging optional, a few cents).

---



## Week 12 — Quantization for Serving: GGUF, AWQ, GPTQ, llama.cpp/Ollama

**Objectives:** Distinguish serving PTQ from QLoRA-training quant. Compare GPTQ/AWQ/GGUF k-quants on the quality/size/speed frontier; ship a local-serving artifact with the correct template. **GGUF/llama.cpp is Mac-native and excellent; AWQ/GPTQ are CUDA-only, so those two are a short burst.**

**Phase A/D (local).** Convert your best merged model to multiple GGUF k-quants with `llama.cpp`; benchmark perplexity + custom benchmark + tokens/sec on the M4; plot the frontier. Package the winner as an Ollama model with a correct Modelfile + chat template. Verify output parity vs the unquantized model.

**Phase C (cloud, tiny).** Produce GPTQ and AWQ quants on a 4090 (~1h) to complete the three-way comparison. Pull them home to benchmark alongside GGUF.

**Reading:** GPTQ paper; AWQ paper; GGUF format spec + `llama.cpp` quantize docs; Ollama Modelfile docs.

**Ships:** **Model Registry** (versioned models/adapters/quant artifacts, auto model cards, safetensors) + **Inference/Prompt Playground** on local llama.cpp/Ollama. **Tag** `v0.13`**.**

**Pitfalls:** chat template lost in GGUF conversion; calibration-set mismatch; comparing methods at different bit-widths.

**Cost:** ~$0.40.

---



## Week 13 — High-Throughput Serving: Flash Attention, vLLM & Deployment

**Objectives:** Serve in production. Explain PagedAttention/continuous batching and Flash Attention's memory story; deploy behind an OpenAI-compatible API; reason about throughput vs latency. **vLLM is CUDA-only; llama.cpp is your Mac-native baseline to compare against.**

**Phase A/D (local).** Serve your model with llama.cpp/Ollama on the M4 as the latency baseline; build the Playground against it. Write the vLLM Dockerfile and K8s manifests locally (kind/minikube runs on Mac for the K8s mechanics — no GPU needed to learn deployment structure).

**Phase C (cloud).** Serve with vLLM on a 4090 (~2h); load-test (throughput-vs-latency, batch-size sweep) against your llama.cpp baseline. Optionally deploy to a cloud K8s cluster with a GPU node.

**Reading:** Flash Attention 1 & 2 papers; vLLM PagedAttention paper + docs; a continuous-batching explainer; a minimal K8s+GPU deploy guide.

**Ships:** **Deployment Pipeline** — one-click promote-from-registry-to-endpoint. **Cut the** `v1.0` **open-source release** — license, docs, auto-generated model cards.

**Pitfalls:** vLLM template/tokenizer drift from training; OOM from over-large `max_num_seqs`; ignoring K8s cold-start.

**Cost:** ~$0.70.

---



## Week 14 — Merging, MoE & Continual Learning

**Objectives:** Extract more from trained models without more training (SLERP/TIES/DARE); explain MoE at concept + upcycling level; mitigate catastrophic forgetting.

**Phase A/D (local).** `mergekit` is CPU/torch — merge two specialist fine-tunes (e.g. code + math) via TIES/DARE **on the Mac**, evaluate against both parents on your dashboard. All free.

**Phase C (cloud, small).** The continual-learning experiment (sequential fine-tune on two domains, measure forgetting, mitigate with replay) — a 4090 burst (~2h) since it's real training. MoE upcycle stretch: another short burst, or prototype in MLX (which supports MoE LoRA natively).

**Reading:** Model Soups; TIES-Merging; DARE; `mergekit` docs; a sparse-upcycling/MoE overview; a forgetting + replay/EWC survey.

**Ships:** merge + continual-learning experiments tracked and compared. **Tag** `v1.1`**.**

**Pitfalls:** merging models with divergent tokenizers/templates; declaring a merge "better" on one benchmark; replay buffer contaminating eval.

**Cost:** ~$0.70.

---



## Week 15 — Capstone

**Objective:** Integrate everything into one defensible, reproducible result and a report AI companies would take seriously.

**Requirements.** Choose a base model (0.5B–8B) and a domain with a verifiable or judgeable target. In your Studio: data (collected + synthetic, cleaned, decontaminated) → SFT → preference/RLVR alignment → **≥3 fine-tuning approaches compared** (e.g. QLoRA vs DoRA vs full; DPO vs GRPO) → hyperparameter sweeps → eval vs public + custom benchmarks → quantize → serve (vLLM + GGUF) → deploy. Reproducible technical report with methods, ablations, CIs, failure analysis, one-command repro.

**Phase discipline for the capstone.** Do *all* data work, template design, config, eval, quantization-to-GGUF, and serving-baseline work on the Mac (free). Reserve cloud strictly for the real training runs and sweeps. A disciplined capstone is ~$15–25: e.g. an A100 for ~10–15 hours total across SFT + alignment + a modest sweep, killed between runs.

**Deliverables.** Tagged release of Fine-Tune Studio; a model on the Hub with a full model card; the technical report. Bonus: the report's "trained on a MacBook + $50 of spot GPU" reproducibility story is itself a differentiator — it proves you understand the cost/compute frontier, which is exactly what resource-conscious teams value.

**Cost:** ~$15–25.

---



## Standing (Cross-Cutting) Stretch Goals

Triton fused kernel (RMSNorm or LoRA-aware matmul) — **CUDA-only, do it during a cloud burst**; MLX has its own kernel API if you want a Mac-native parallel. Speculative decoding (llama.cpp supports it on Mac — free). Second RM architecture. Ray-based distributed sweep (cloud). Upstream PR to TRL/peft/vLLM/mlx-lm.

## Datasets (recurring)

SFT: Tülu 3 mixture, OpenHermes-2.5, FineWeb-Edu slice. Preference: UltraFeedback, HH-RLHF. RLVR: GSM8K, MATH, an executable code set. Eval: MMLU, IFEval, GSM8K + your custom decontaminated domain set. Decontaminate against all eval sets in Week 3.

## Models (recurring)

Spine: Qwen2.5 (0.5B/1.5B/3B/7B) — clean cross-size ablations and first-class MLX + CUDA support. Local dev sweet spot on the M4: 0.5B–3B. Variety: Llama-3.2-1B/3B, Gemma-2-2B, SmolLM2-1.7B, Phi-3.5-mini. Keep RL/GRPO ≤3B.

## Which framework where

- **Mac-native (MLX / mlx-lm / mlx-lm-lora):** local dev, LoRA/QLoRA/DoRA/full, DPO/ORPO/GRPO prototyping, quantization, fast iteration. A genuine second parity reference.
- **Mac-native (PyTorch-MPS + HF):** from-scratch `core/` work, parity defenses, small SFT, all data/eval/template work.
- **Mac-native (llama.cpp/Ollama, mergekit, lm-eval-harness):** serving baseline, GGUF, merging, evaluation — all free.
- **CUDA-only (cloud burst):** bitsandbytes NF4 QLoRA-7B, FlashAttention, FSDP/DeepSpeed multi-GPU, vLLM serving, GPTQ/AWQ, Triton kernels, heavy GRPO rollouts.



## Cost Summary

Zero-cost weeks: 0(≈), 1, 2, 3, 4, 11. Cents-to-dollars weeks: 5, 6, 7, 9, 12, 13, 14. Irreducible cloud: 8 (~~$8), 10 (~~$6), capstone (~$15–25). **Semester total ≈ $40–60 with discipline; the floor is ~$30 if you skip every optional burst.** Every dollar is a *submitted-and-killed batch job*, never an idle instance.

## Release Timeline (GitHub Portfolio)

`v0.1` skeleton + cloud-burst tooling → `v0.2` transformer core → `v0.3` tokenizer lab → `v0.4` data pipeline → `v0.5` annotation UI → `v0.6` LoRA-from-scratch + parity → `v0.7` reproducible QLoRA-7B + tracking → `v0.8` PEFT registry → `v0.9` distributed training → `v0.10` DPO/ORPO → `v0.11` reward models + GRPO/RLVR → `v0.12` eval dashboard + custom benchmark → `v0.13` quantized + registry + playground → `v1.0` **served + deployed (open-source release)** → `v1.1` merge/continual → capstone release + Hub model + report. The through-line a reviewer sees: *a full fine-tuning platform built on a MacBook, with CUDA-faithful results reproduced for cents.*