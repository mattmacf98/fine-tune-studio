import torch
import mlx.core as mx
from mlx_lm import load, generate

if __name__ == "__main__":
    print(torch.backends.mps.is_available(), torch.backends.mps.is_built())
    print(mx.device_info())
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    resp = generate(model, tokenizer, "Hello, how are you?")
    print(resp)