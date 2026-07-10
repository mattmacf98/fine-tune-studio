# Week 0 — Environment, the Local/Cloud Split, and the Skeleton

**Goal of the week:** end with a working Mac dev environment, a monorepo skeleton with green CI, a `core/` API contract the rest of the course commits against, and — most importantly — a *proven* cloud-burst loop (`make train-remote`) that rsyncs a job to a rented GPU, runs it, pulls results back, and **kills the pod automatically**. If only one thing works by Friday, it should be that loop.

**Time budget:** ~10–12 hours. **Cost:** ~$0.10 (one ~15-min 4090 verification burst).

**Definition of done:** every checkbox below ticked, `v0.1` tagged on GitHub, and you've watched a dummy training job round-trip to a cloud GPU and back with the pod terminating on its own.

---



## Part 0 — Prerequisites & accounts (30 min)

- [x] macOS updated; you're on Apple Silicon (M4). Confirm: `sysctl -n machdep.cpu.brand_string` and `uname -m` → `arm64`.
- [x] **Xcode Command Line Tools:** `xcode-select --install` (needed for compiling `llama.cpp` and native wheels).
- [x] **Homebrew** installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`).
- [x] **GitHub account** + a new **private** repo `ftstudio` (you'll flip it public at `v1.0`).
- [x] **Hugging Face account** + access token (`huggingface-cli login` later). Request access to any gated models you plan to use (Llama needs a license click; Qwen2.5 is open).
- [ ] **RunPod account** (or Vast.ai). Add ~$10 credit. **Set a billing alert / spending limit today** — this is your insurance against a forgotten pod.
- [ ] **Weights & Biases** account (free tier) + API key. Optional now, used from Week 6.

> **Why RunPod for the tutorial:** its CLI (`runpodctl`) and per-second billing with free egress make the submit-and-kill loop clean. Vast.ai is cheaper per hour but a marketplace with more variability; start on RunPod, price-shop on Vast later once the loop is second nature.

---



## Part 1 — Local Mac environment (2–3 hrs)



### 1a. Python & the toolchain

- [x] Install `uv` (fast, reproducible envs): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [x] Create the project env pinned to Python 3.11: `uv venv --python 3.11 && source .venv/bin/activate`
- [x] Install core libs:
  ```bash
  uv pip install torch torchvision            # MPS build is default on macOS arm64
  uv pip install transformers datasets tokenizers accelerate
  uv pip install mlx mlx-lm                    # Apple-native; fast local training
  uv pip install trl peft                      # import/run on Mac; big training is cloud-only
  uv pip install wandb mlflow hydra-core       # tracking + config (used from W6)
  uv pip install pytest ruff                   # CI: test + lint
  ```



### 1b. Verify PyTorch sees the GPU (MPS)

- [x] Run this and confirm `True`:
  ```python
  import torch; print(torch.backends.mps.is_available(), torch.backends.mps.is_built())
  ```
  If either is `False`, you have a CPU-only torch build — reinstall. **This is the #1 silent Week-0 failure.**



### 1c. MLX sanity check

- [x] `python -c "import mlx.core as mx; print(mx.default_device())"` → should print a GPU device.
- [x] Quick generate: `mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --prompt "hello" --max-tokens 20`
  (Downloads the pre-converted MLX weights from the `mlx-community` HF org and runs on the M4.)



### 1d. llama.cpp + Ollama (serving stack, used from W12–13)

- [x] **Ollama:** `brew install ollama`, then `ollama run qwen2.5:0.5b` to confirm local serving works.
- [ ] **llama.cpp** (build from source so you have the `quantize` and `convert` tools later):
  ```bash
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
  cmake -B build && cmake --build build --config Release
  ```
  Metal acceleration is on by default on macOS — no flags needed.



### 1e. HF auth

- [x] `huggingface-cli login` (paste token). Test: `huggingface-cli whoami`.

---



## Part 2 — The monorepo skeleton (2 hrs)

Create this structure. The `core/` package is the spine — everything you write from scratch lives here, and the parity tests in later weeks import from it.

```
ftstudio/
├── pyproject.toml          # uv-managed deps, ruff config
├── Makefile                # dev / test / train-remote / cuda-check targets
├── README.md
├── .github/workflows/ci.yml
├── configs/                # Hydra configs (grow from W6)
│   └── base.yaml
├── core/                   # ← YOU write this, from scratch, all semester
│   ├── __init__.py
│   ├── nn/                 # reference transformer blocks (W1)
│   ├── data/               # pipeline pieces (W3)
│   ├── peft/               # LoRA/DoRA from scratch (W5,7)
│   ├── align/              # DPO/GRPO losses from scratch (W9,10)
│   └── memory.py           # memory_estimator (W1)
├── studio/                 # the platform: FastAPI backend + React front end (grows weekly)
│   ├── backend/
│   └── frontend/
├── scripts/
│   ├── hardware_probe.py
│   └── remote_train.sh     # the burst loop
└── tests/
    └── test_smoke.py
```

- [x] `git init`, create the tree, commit.
- [x] Write a minimal `pyproject.toml` declaring the package and ruff settings.
- [x] `tests/test_smoke.py`: one test that imports `core` and asserts `torch.backends.mps.is_available()` (skips gracefully in CI where there's no MPS).



### The `core/` API contract (write these signatures now, implement later)

This is the contract the whole course commits against — stub it this week so imports resolve and CI is green:

```python
# core/nn/block.py
class DecoderBlock(nn.Module):
    """Implemented in W1. RoPE + GQA + RMSNorm + SwiGLU."""

# core/memory.py
def estimate_training_memory(n_params, seq_len, batch, dtype, optimizer) -> dict:
    """Implemented in W1. Returns weights/grads/optim/activations breakdown in GB."""

# core/peft/lora.py
class LoRALinear(nn.Module):
    """Implemented in W5."""

# core/align/dpo.py
def dpo_loss(policy_logps, ref_logps, beta) -> torch.Tensor:
    """Implemented in W9."""
```

---



## Part 3 — `hardware_probe.py` (1 hr)

The one script that runs *both* locally and on a pod and tells you what you're working with. Build it to detect its environment.

- [x] It should report:
  - **Everywhere:** platform, Python, torch version, torch device (`mps`/`cuda`/`cpu`).
  - **On Mac:** chip name, total unified memory, MPS available/built.
  - **On a CUDA pod:** GPU name, VRAM, compute capability, whether **bf16** and **FP8** are supported (FP8 needs Ada/Hopper, i.e. sm_89+/sm_90+).
- [x] Include a `--bench` flag that loads Qwen2.5-0.5B and reports tokens/sec + peak memory, so the same command benchmarks any machine.

> **Concept to internalize this week — unified memory ≠ VRAM.** Your 16GB is shared by macOS, your apps, the model weights, *and* training activations. Budget realistically: after the OS you have ~10–12GB of working room. A 0.5B model in bf16 is ~1GB of weights but training it (grads + Adam moments + activations) can still push past that at long context. This is *why* the course keeps local dev at 0.5B–3B and bursts to cloud for 7B+.

---



## Part 4 — The cloud-burst loop (3–4 hrs) — **the heart of Week 0**

This is the capability that makes the whole Mac-primary plan viable. Build it, then prove it works with a throwaway job.

### 4a. Install and configure `runpodctl`

- [x] Install: `brew install runpod/runpodctl/runpodctl` (or `wget` the release binary).
- [x] Configure: `runpodctl config --apiKey=YOUR_KEY` (stored in `~/.runpod/config.toml`).
- [x] Register your SSH key: `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`
  (Generate one first if needed: `ssh-keygen -t ed25519`.)
- [x] Verify: `runpodctl pod list` returns (an empty list is fine).

> **CLI shape (noun-verb):** `runpodctl pod create --image=... --gpu-id=...`, `pod list`, `pod get <id>`, `pod stop <id>`, `pod delete <id>`. Pods come with `runpodctl` pre-installed and a pod-scoped key.



### 4b. Pick a base image

Use RunPod's official PyTorch image so CUDA/cuDNN match:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

Check the current tag on the RunPod PyTorch template before you hardcode it — image tags rotate.

### 4c. Write `scripts/remote_train.sh` (the loop)

The loop is five steps: **create → wait for SSH → rsync up → run → rsync down → delete.** Structure it so `delete` runs in a `trap` that fires even on error or Ctrl-C — you must never leak a running pod.

```bash
#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
IMAGE="${IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
CMD="${1:?usage: remote_train.sh '<command to run on pod>'}"

POD_ID=""
cleanup() {
  if [[ -n "$POD_ID" ]]; then
    echo ">>> Terminating pod $POD_ID"
    runpodctl pod delete "$POD_ID" || echo "WARN: delete failed — CHECK DASHBOARD"
  fi
}
trap cleanup EXIT INT TERM   # <-- pod dies even if the script crashes

echo ">>> Creating pod..."
POD_ID=$(runpodctl pod create --image="$IMAGE" --gpu-id="$GPU_ID" -o json | jq -r '.id')

echo ">>> Waiting for SSH..."
# poll runpodctl pod get "$POD_ID" until it reports RUNNING + an SSH host/port
# (parse the JSON; sleep 5 between tries; time out after ~3 min)

HOST=...   # from pod get
PORT=...   # from pod get

echo ">>> Syncing repo up..."
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'data/raw' \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" ./ root@"$HOST":/workspace/ftstudio/

echo ">>> Running job..."
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" \
  "cd /workspace/ftstudio && pip install -e . && $CMD"

echo ">>> Pulling artifacts home..."
rsync -avz -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  root@"$HOST":/workspace/ftstudio/outputs/ ./outputs/

echo ">>> Done. Pod will be terminated by trap."
```

- [ ] Add `jq` (`brew install jq`) for JSON parsing.
- [ ] Wrap it in a Make target:
  ```makefile
  train-remote:      ## rsync -> run on GPU -> pull -> kill pod
  	./scripts/remote_train.sh "$(CMD)"
  cuda-check:        ## 5-min smoke pod to verify an op on real CUDA
  	./scripts/remote_train.sh "python scripts/hardware_probe.py --bench"
  ```



### 4d. Prove the loop with a throwaway job — **do this before you trust it**

- [ ] `make cuda-check` — this creates a pod, runs `hardware_probe.py --bench` on real CUDA, pulls nothing important, and **deletes the pod.**
- [ ] While it runs, open the RunPod dashboard and *watch the pod appear and then disappear.* Confirm with `runpodctl pod list` that nothing is left running.
- [ ] Deliberately Ctrl-C a run mid-way and confirm the `trap` still deletes the pod. If it doesn't, fix it now — this is the safety mechanism.

> **Failure modes to expect the first time:** SSH not ready yet (add polling + timeout); `StrictHostKeyChecking` prompt hanging the script (disable it as shown); rsync pushing your multi-GB `data/` or `.venv` (exclude them); pod created in a region with no 4090 available (`pod create` errors — retry or widen GPU selection). Budget an hour for these; they're one-time.

---



## Part 5 — CI (30 min)

- [ ] `.github/workflows/ci.yml`: on push, run `ruff check .` and `pytest`. Runners are Linux/CPU with no MPS — make the MPS assertion in `test_smoke.py` a skip, not a failure, when `torch.backends.mps.is_available()` is `False`.
- [ ] Push, confirm the badge goes green.

---



## Part 6 — Ship it (15 min)

- [ ] Fill in `README.md`: one paragraph on what Fine-Tune Studio will become, the local/cloud split philosophy, and a "reproduce this" quickstart.
- [ ] Commit everything. `git tag v0.1 && git push --tags`.
- [ ] **Ships to platform:** repo skeleton, CI, `hardware_probe`, `make train-remote`, `make cuda-check`, dual MLX/MPS inference smoke test.

---



## Readings for Week 0 (with what to extract from each)

Keep these light — Week 0 is mostly hands-on. Read for orientation, not mastery.

1. **MLX-LM docs + the WWDC25 "Explore LLMs on Apple Silicon with MLX" session.** *Extract:* how unified memory changes the mental model vs CUDA; that `mlx_lm.lora` trains adapters directly on quantized weights (QLoRA-style) locally; which model families are supported. This is your local-training foundation for the whole course.
2. **RunPod docs: "Transfer files" + the** `runpodctl` **README.** *Extract:* the exact rsync-over-SSH syntax (`rsync -avz -e "ssh -p PORT" ...`), the noun-verb CLI shape, and that per-second billing means the pod's *uptime* is your bill — hence the trap-on-exit discipline.
3. **PyTorch MPS backend docs (**`torch.backends.mps`**).** *Extract:* how to detect MPS, that some ops fall back to CPU or aren't implemented (you'll hit this in W1), and the `PYTORCH_ENABLE_MPS_FALLBACK=1` escape hatch for missing ops.
4. **Safetensors format README.** *Extract:* why it exists (safe, zero-copy, fast vs pickle), and that it's the default checkpoint format you'll produce and load all semester. Short read.
5. **Kipply, "Transformer Inference Arithmetic"** (skim now, deep-read in W1). *Extract:* the memory/latency intuition that motivates `memory_estimator.py` and the whole "stay small locally, burst for big" strategy.
6. `nvidia-smi` **reference** (you'll only use it on pods). *Extract:* how to read VRAM usage, utilization, and process list — your first move when a cloud run OOMs or hangs.

*(Optional, if you have time:)* skim the **Qwen2.5 technical report** intro — Qwen2.5 is the model spine for the course, so knowing its sizes and architecture family pays off immediately in W1.

---



## Common pitfalls this week (ranked by how often they bite)

1. **CPU-only torch install** — MPS shows `False`. Everything "works" but runs 10× too slow and you don't know why. Verify in step 1b before anything else.
2. **Leaking a running pod** — the expensive mistake. The `trap ... EXIT` and a billing alert are your two independent safeguards. Never rely on remembering to click "terminate."
3. **Treating unified memory as free/infinite** — the OS shares your 16GB. Profile real headroom with `hardware_probe --bench` so your local size limits are empirical, not guessed.
4. **SSH host-key prompt hanging the burst script** — `-o StrictHostKeyChecking=no` for ephemeral pods (fine here; these are throwaway hosts).
5. **rsync pushing giant dirs** — always `--exclude '.git' '.venv' 'data/raw'`; keep large datasets on the HF Hub and `wget`/`hf download` them *on the pod* instead of pushing from home internet.
6. **Hardcoding a stale pod image tag** — image tags rotate; check the current RunPod PyTorch template tag before each burst-heavy week.

---



## Stretch goals (optional, if the core is done early)

- [ ] Add a `--spot` option to the burst script (Vast.ai interruptible or RunPod spot) and handle the interruption case (checkpoint + resume). Saves ~40% on the heavy weeks later.
- [ ] Add a tiny `make watch-cost` that polls `runpodctl pod list` and warns if any pod has been up longer than N minutes.
- [ ] Pre-write a `pod_bootstrap.sh` that installs your exact deps on a fresh pod, so cold-start is one command in later weeks.
- [ ] Wire a second provider (Vast.ai `vastai` CLI) behind the same Make target so you can price-shop without changing your workflow.

---



## End-of-week self-check

You're ready for Week 1 when you can answer *yes* to all of these:

- [ ] Does `torch.backends.mps.is_available()` return `True`?
- [ ] Can you generate text from Qwen2.5-0.5B via **both** MLX and PyTorch-MPS?
- [ ] Does `make cuda-check` create a pod, run the probe on real CUDA, and **auto-terminate**?
- [ ] Did you *watch* the pod disappear from the dashboard, and confirm `pod list` is empty?
- [ ] Is CI green and `v0.1` tagged?
- [ ] Do you have a billing alert set on RunPod?

If yes to all six, the hardest infrastructure work of the entire course is behind you — every later week reuses this exact loop.