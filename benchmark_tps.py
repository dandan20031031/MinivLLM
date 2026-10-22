import time
import os
import torch
import numpy as np

from transformers import AutoTokenizer,AutoModelForCausalLM

# ===== minivllm =====
from myvllm.engine.llm_engine import LLMEngine as MiniLLM
from myvllm.sampling_parameters import SamplingParams as MiniSamplingParams

# ===== vllm (optional, lazily imported) =====



config = {
    'max_num_sequences': 16,
    'max_num_seqs': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    # Model path: a local checkpoint directory or a Hugging Face repo id.
    # Override with MODEL_PATH=/path/to/Qwen3-0.6B to avoid re-downloading.
    'model_name_or_path': os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B"),
    'enforce_eager': os.environ.get("ENFORCE_EAGER", "0") == "1",
    'vocab_size': 151936,  # Fixed: was 151643, HF model uses 151936
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,  # Fixed: was 64, should be 128 (hidden_size / num_heads for GQA output)
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,  # Fixed: was 10000, HF uses rope_theta=1000000
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768, # should be >= max_model_length, max position index allowed in rotary embedding
    'ffn_bias': False,  # Fixed: HF Qwen3 doesn't use MLP bias
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,  # Fixed: should match tokenizer.eos_token_id
}

MODEL_NAME = config['model_name_or_path']
PROMPTS = [
    "introduce yourself" ,
    "list all prime numbers within 100" ,
    "give me your opinion on the impact of artificial intelligence on society" ,
]

WARMUP_STEPS = 2
TIMED_RUNS = 3  # 每个 batch 档计时次数，取中位数
COOLDOWN_SECS = 3  # 档间冷却，避免 GPU 频率漂移影响下一档数据
OUTPUT_TOKENS = 256  # ouput token num
# 批规模扫描：逗号分隔，如 "3,8,16"；默认 "3" 保持与历史基线可比
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "3").split(",") if x.strip()]
# transformers 对照默认关闭（bs=16 时耗时过长且基线已测），需要时置 1
RUN_TRANSFORMERS = os.environ.get("RUN_TRANSFORMERS", "0") == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_prompts(tokenizer, n: int) -> list[str]:
    """构造 n 条 prompt：前 3 条保持原始 PROMPTS（与历史基线严格可比），超出部分循环加变体后缀"""
    chosen = []
    for i in range(n):
        p = PROMPTS[i % len(PROMPTS)]
        if i >= len(PROMPTS):
            p = f"{p} (variation {i // len(PROMPTS) + 1})"
        chosen.append(p)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in chosen
    ]


def run_minivllm(tokenizer):
    # 引擎只初始化一次，多 batch 档复用（权重加载占大头）
    llm = MiniLLM(config=config)
    sampling = MiniSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
        max_model_length=128,
    )

    results = {}
    for bs in BATCH_SIZES:
        prompts = build_prompts(tokenizer, bs)

        # warmup（吸收 CUDA Graph 首次 replay 与 torch.compile 按 shape 重编译）
        for _ in range(WARMUP_STEPS):
            llm.generate(prompts, sampling)
            cuda_sync()

        # 计时 3 次取中位数
        latencies, total_tokens = [], 0
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            cuda_sync()
            end = time.perf_counter()
            latencies.append(end - start)
            total_tokens = sum(len(x) for x in outputs["token_ids"])

        latency = float(np.median(latencies))
        tps = total_tokens / latency
        results[bs] = {
            "batch_size": bs,
            "latency": latency,
            "tokens": total_tokens,
            "tps": tps,
            "per_seq_tps": tps / bs,
        }
        print(
            f"[minivllm bs={bs}] latency={latency:.3f}s tokens={total_tokens} "
            f"aggregate_tps={tps:.1f} per_seq_tps={tps / bs:.1f}"
        )
        if bs == BATCH_SIZES[0]:
            # sanity check：图内/图外 logits 若出错，生成文本会立刻退化为乱码或重复
            print(f"  sample_output={outputs['text'][0][:100]!r}")
        time.sleep(COOLDOWN_SECS)

    return results


def run_vllm(tokenizer):
    # vLLM (lazily imported; skipped when not installed)
    from vllm import LLM as VLLM
    from vllm import SamplingParams as VLLMSamplingParams

    llm = VLLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=False, 
        gpu_memory_utilization=0.75,  
        max_model_len=256, 
        speculative_config=None, 
    )

    sampling = VLLMSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # warmup
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_transformers_test(tokenizer):
    # transformers
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # Prepare attention_mask explicitly
    attention_mask = inputs["attention_mask"]

    # warmup
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=OUTPUT_TOKENS)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=OUTPUT_TOKENS)
    end = time.perf_counter()

    total_tokens = sum(len(output) for output in outputs)
    latency = end - start

    tps = total_tokens / latency

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": tps,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')

    results = {}

    print("Running minivllm benchmark...")
    results["minivllm"] = run_minivllm(tokenizer)

    try:
        import importlib.util
        if importlib.util.find_spec("vllm") is not None:
            print("Running vLLM benchmark...")
            results["vLLM"] = run_vllm(tokenizer)
        else:
            print("vLLM not installed, skipping.")
    except Exception as e:
        print(f"vLLM benchmark skipped: {type(e).__name__}: {str(e).splitlines()[0]}")

    if RUN_TRANSFORMERS:
        print("Running transformers benchmark...")
        results["transformers"] = run_transformers_test(tokenizer)
    else:
        print("transformers benchmark disabled (RUN_TRANSFORMERS=0), skipping.")

    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        print(f"{k}:")
        if k == "minivllm":
            for bs, r in v.items():
                print(
                    f"  bs={bs}: latency={r['latency']:.4f} tokens={r['tokens']} "
                    f"aggregate_tps={r['tps']:.1f} per_seq_tps={r['per_seq_tps']:.1f}"
                )
        else:
            for kk, vv in v.items():
                print(f"  {kk}: {vv:.4f}")



if __name__ == "__main__":
    main()
