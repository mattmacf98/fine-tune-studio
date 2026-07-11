# Fine-Tune Studio

Fine-Tune Studio is a semester-long project to build a full small-language-model fine-tuning platform on a 16GB Apple Silicon Mac: a from-scratch `core/` library (transformer blocks, data pipelines, LoRA/DoRA, DPO/GRPO), a `studio/` orchestration UI, and a serving stack (GGUF, llama.cpp, Ollama). By the end it becomes an end-to-end system where you can prepare data, fine-tune Qwen-scale models, align them, evaluate them, quantize them, and serve them — with every artifact reproducible from config and version control.

The design philosophy is **local-first, cloud-burst**: develop, debug, and prove correctness on the Mac for free (~80% of the work), then `rsync` the already-working repo to a rented GPU for the real training job, pull artifacts home, and **kill the pod immediately**. CUDA is the source of truth for training, but it is always a submitted batch job — never a machine you leave running. Local MPS/MLX handles iteration speed; the cloud handles scale; `make cuda-check` resolves the occasional op that behaves differently on CUDA vs MPS.

## Reproduce this

**Prerequisites:** macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), Python 3.12+, and (for cloud steps) a [RunPod](https://www.runpod.io/) account with `runpodctl` configured (`runpodctl doctor`).

```bash
# Clone and install
git clone https://github.com/mattmacf98/fine-tune-studio fine-tune-studio && cd fine-tune-studio
uv sync --group dev

# Verify local environment (MPS on Mac; skips in CI)
uv run pytest

# Probe local hardware
uv run python scripts/hardware_probe.py

# Optional: benchmark on Mac (MLX on MPS, PyTorch on CUDA)
uv run python scripts/hardware_probe.py --bench
```

**Cloud burst loop** — creates a pod, syncs the repo, runs a job, and auto-terminates:

```bash
# 5-minute CUDA smoke test (hardware probe + Qwen2.5-0.5B benchmark) (requires runpod account)
make cuda-check

# Run any command on a cloud GPU (pod is killed on exit, including Ctrl-C)
make train-remote CMD="python scripts/hardware_probe.py --bench"
```

**Lint and CI locally:**

```bash
uv run ruff check .
uv run pytest
```

CI runs the same checks on every push to `master` (MPS tests skip gracefully on Linux runners).