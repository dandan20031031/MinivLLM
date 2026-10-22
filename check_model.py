"""check_model.py — 打印 myvllm 引擎中的模型结构与参数形状（以 Qwen3-0.6B 为例）

用法:
    .venv\\Scripts\\python.exe check_model.py

走 ModelRunner 的真实构造路径（init_process_group -> 建模型 -> 加载权重 ->
warmup -> 分配 KV cache），保证打印出来的就是引擎实际运行的结构。
"""
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from myvllm.engine.model_runner import ModelRunner

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
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
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,
}


def fmt_num(n: int) -> str:
    return f"{n:,}"


def own_params_str(module: torch.nn.Module) -> str:
    """模块自身的直接参数（不含子模块），显示为 [name(shape), ...]"""
    parts = []
    for n, p in module.named_parameters(recurse=False):
        parts.append(f"{n}{tuple(p.shape)}")
    return "[" + ", ".join(parts) + "]" if parts else ""


def main():
    runner = ModelRunner(config, rank=0, event=[])
    model = runner.model

    num_layers = config['num_layers']

    print("=" * 78)
    print(f"模型: {config['model_name_or_path']}  |  dtype: "
          f"{next(model.parameters()).dtype}  |  device: {next(model.parameters()).device}")
    print("=" * 78)
    print(model.__class__.__name__)

    for name, mod in model.named_modules():
        if not name:
            continue
        parts = name.split(".")
        # layers.0 完整展开，layers.1 及以后折叠
        if parts[0] == "layers" and int(parts[1]) >= 1:
            continue
        depth = name.count(".")
        indent = "    " * depth
        line = f"{indent}|- {parts[-1]}: {mod.__class__.__name__}"
        ps = own_params_str(mod)
        if ps:
            line += f"  {ps}"
        print(line)

    # tied weight 说明（named_parameters 对共享参数去重，但 recurse=False 下两边都会列出）
    if config['tie_word_embeddings']:
        print(f"注: lm_head.weight 与 embed_tokens.weight 共享 (tie_word_embeddings=True)")
    print(f"    (layers.1 ~ layers.{num_layers - 1} 结构与 layers.0 完全相同，已折叠)")

    print("\n" + "=" * 78)
    print("关键 shape 对应关系（GQA: 16 Q head / 8 KV head, head_dim=128）")
    print("=" * 78)
    print("qkv_projection.weight [4096, 1024] = (16Q + 8K + 8V) x head_dim  x hidden_size")
    print("o_proj.weight         [1024, 2048] = hidden_size x (16 head x head_dim)")
    print("gate_up.weight        [6144, 1024] = (gate 3072 + up 3072) x hidden_size")
    print("down_proj.weight      [1024, 3072] = hidden_size x intermediate_size")
    print("embed_tokens.weight   [151936, 1024] = vocab_size x hidden_size")

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    total = sum(p.numel() for p in model.parameters())
    per_layer = sum(p.numel() for p in model.model.layers[0].parameters())
    embed = model.model.embed_tokens.weight.numel()
    dtype = next(model.parameters()).dtype
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    print(f"总参数量:        {fmt_num(total)}  ({total / 1e6:.2f} M)")
    print(f"每个 decoder 层: {fmt_num(per_layer)}  ({per_layer / 1e6:.2f} M) x {num_layers}")
    print(f"embedding 参数:  {fmt_num(embed)}  ({embed / 1e6:.2f} M, 占总参数 {embed / total:.0%})")
    print(f"权重显存估算:    {total * dtype_bytes / 1024 ** 3:.2f} GiB  (@{dtype})")

    counts = Counter(m.__class__.__name__ for m in model.modules())
    print("模块类型统计:")
    for cls, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        if cnt >= 1 and cls not in ("Module",):
            print(f"  {cls:28s} x{cnt}")

    runner.call("exit")


if __name__ == "__main__":
    main()
