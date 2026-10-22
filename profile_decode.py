"""Profile the MinivLLM decode path with torch.profiler.

Loads Qwen3-0.6B with the engine, warms up, then runs one generate() under the
profiler and prints the CUDA-time ranking of all kernels that ran.

    python profile_decode.py

Set MODEL_PATH=/path/to/Qwen3-0.6B to use a local checkpoint.
"""
import os

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams

MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B")
PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "give me your opinion on the impact of artificial intelligence on society",
]

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': MODEL,
    'enforce_eager': True,
    'vocab_size': 151936,
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768,
    'ffn_bias': False,
    'max_num_batch_tokens': 4096,
    'max_model_length': 256,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,
}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    llm = LLMEngine(config=config)
    sampling = SamplingParams(temperature=0.6, max_tokens=128, max_model_length=256)

    # warmup (JIT compile + lazy init)
    llm.generate(prompts, sampling)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        llm.generate(prompts, sampling)
        torch.cuda.synchronize()

    print("\n===== TOP 40 KERNELS BY CUDA TIME =====\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))


if __name__ == "__main__":
    main()
