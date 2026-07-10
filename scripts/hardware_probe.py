import argparse
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BENCH_PROMPT = "Hello, how are you?"
BENCH_MAX_NEW_TOKENS = 512

def detect_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_summary(device: str) -> None:
    print(f"System: {platform.uname().system}")
    print(f"Python version: {platform.python_version()}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch device: {device}")


def print_verbose_details(device: str) -> None:
    if device == "mps":
        chip_name = platform.processor() or platform.machine() or "Unknown"
        print(f"Chip name: {chip_name}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    elif device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory / 1024 / 1024
        print(f"GPU name: {props.name}")
        print(f"VRAM: {vram_mb:.2f} MB")
        print(f"Compute capability: {props.major}.{props.minor}")
        print(f"BF16 supported: {props.major >= 8}")
        print(f"FP8 supported: {props.major >= 8 and props.minor >= 9}")

def benchmark_qwen2_5_0_5b_instruct_4bit_mac() -> None:
    from mlx_lm import generate, load

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    start_time = time.perf_counter()
    generate(model, tokenizer, BENCH_PROMPT, max_tokens=BENCH_MAX_NEW_TOKENS)
    generation_time = time.perf_counter() - start_time
    print(f"Benchmark: {BENCH_MAX_NEW_TOKENS / generation_time:.2f} tokens per second")


def benchmark_qwen2_5_0_5b_instruct_4bit_cuda() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, dtype=dtype).to(device)
    model.eval()

    inputs = tokenizer(BENCH_PROMPT, return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=BENCH_MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize(device)
    generation_time = time.perf_counter() - start_time

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_second = new_tokens / generation_time
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    print(f"Benchmark: {tokens_per_second:.2f} tokens per second")
    print(f"Peak memory: {peak_memory_gb:.2f} GB")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report hardware and PyTorch device info for the current machine.",
    )
    parser.add_argument(
        "-b",
        "--bench",
        action="store_true",
        help="Run a benchmark test to measure the performance of the device.",
    )
    args = parser.parse_args()

    device = detect_torch_device()
    print_summary(device)
    print_verbose_details(device)

    if args.bench:
        if device == "mps":
            benchmark_qwen2_5_0_5b_instruct_4bit_mac()
        elif device == "cuda":
            benchmark_qwen2_5_0_5b_instruct_4bit_cuda()
        else:
            print("No benchmark available for this device.")


if __name__ == "__main__":
    main()
