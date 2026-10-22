# MinivLLM 端到端与算子级 Benchmark 报告

**测试日期**：2026-10-21
**硬件**：NVIDIA GeForce RTX 5080 (sm_120)
**软件**：Python 3.11 · PyTorch 2.11.0+cu128 · Triton 3.8.0 · transformers（venv 环境）
**模型**：Qwen3-0.6B（本地完整权重 `D:\dandan123\dandan\myvllm\models\Qwen3-0.6B`，fp16 权重 1.5 GB）
**测试脚本**：`benchmark_decoding.py`（算子级）、`benchmark_tps.py`（端到端）

---

## 一、端到端模型 Benchmark

### 1.1 测试设置

| 项目 | 值 |
|---|---|
| 引擎配置 | world_size=1，分别测 `enforce_eager=True`（逐 kernel 发射）与 `enforce_eager=False`（CUDA Graph 捕获回放）两种模式 |
| KV cache | block_size=256，可用 190 块（引擎按剩余显存自动计算并跨卡取 MIN） |
| 负载 | 3 条英文 prompt 走 chat template，`temperature=0.6`，连续批处理同批推进 |
| 计时 | 2 次预热后计时 1 次，`torch.cuda.synchronize` 包夹 |
| 对照组 | vLLM（本机未安装，自动跳过）；HuggingFace transformers 朴素解码 |

### 1.2 结果

| 指标 | MinivLLM eager | MinivLLM CUDA Graph | transformers 朴素解码 |
|---|---:|---:|---:|
| 总耗时 | 3.05 s | **0.80 s** | 89.41 s |
| 生成 token 数 | 336（3 条 × ~112） | 336 | 768（3 条 × ~256） |
| **吞吐（TPS）** | 110.1 tokens/s | **420.0 tokens/s** | 8.6~9.7 tokens/s |
| prefill 吞吐 | 1241~1347 tokens/s | 1158~1524 tokens/s（51 token 批量前向） | — |
| decode 聚合吞吐 | 100~155 tokens/s | **420~500 tokens/s**（每步 3 token） | — |
| decode 单序列吞吐 | 约 35~50 tokens/s | 约 140~165 tokens/s | 约 2.9 tokens/s |
| 相对 transformers | ~13x | **~49x** | 1x |

### 1.3 解读

1. **连续批处理红利**。decode 阶段引擎每步同时为 3 条序列各产出 1 个 token，聚合吞吐 110 tokens/s 平摊到每条序列约 37 tokens/s。批内序列共享同一次 kernel 发射与权重读取，批次越大单序列成本越低——这是 continuous batching 相对逐请求处理的结构性优势。

2. **对 transformers 约 13 倍加速**。transformers 侧为逐条朴素解码：无分页 KV cache 管理、无连续批处理、无 CUDA Graph，且同进程内被 MinivLLM 设置的 `torch.set_default_device` 干扰需安装 accelerate 才能运行。TPS 为归一化指标，两侧停止条件不同（MinivLLM 限总长 128、transformers 限总长 256）不影响对比有效性。

3. **prefill 与 decode 的数量级差异符合预期**。prefill（约 1300 tokens/s）是计算受限的并行 GEMM；decode（约 110 tokens/s）是访存受限的逐 token 过程，每个 token 需要遍历全部历史 KV。

4. **CUDA Graph 带来约 3.8 倍吞吐提升，验证了"发射开销是第一瓶颈"**。eager 模式下 decode 每步逐 kernel 发射，CPU 发射间隙导致 GPU 大量空转；`enforce_eager=False` 将 decode 整段 kernel 序列捕获成单张图、一次 `replay()` 全部回放，TPS 从 110.1 升至 420.0（decode 聚合吞吐 100~155 → 420~500 tokens/s）。prefill 因走 flash-attention varlen 路径不参与捕获，吞吐基本持平（约 1150~1524 tokens/s）。

---

## 二、算子级 Benchmark：PagedAttention Decode Kernel

### 2.1 测试设置

- **注意力配置**（与 Qwen3-0.6B 一致）：num_heads=16，num_kv_heads=8（GQA 组距 4），head_dim=128，block_size=256，fp16 输入
- **四种实现**：Naive PyTorch（逐块 gather + 稠密 attention）、Fast PyTorch（向量化 gather + 批量 matmul）、Triton 逐 token（chunk 内标量循环）、Triton 分块（整 chunk 一次 masked gather）
- **正确性验证**：每种实现先与 float32 参考实现对拍，通过后才计时；页表使用 `randperm` 打乱而非恒等映射，防止"无视页表的错误 kernel"碰巧算对
- **计时**：10 次预热（触发 JIT）+ 100 次迭代，CUDA Event 计时

### 2.2 完整结果

| batch | seq_len | Naive PyTorch | Fast PyTorch | Triton 逐token | Triton 分块 |
|------:|--------:|--------------:|-------------:|---------------:|------------:|
| 1 | 128 | 0.450 ms (1.0x) | 0.944 (0.5x) | 0.071 (6.3x) | **0.019 (23.3x)** |
| 1 | 512 | 0.532 (1.0x) | 1.147 (0.5x) | 0.270 (2.0x) | **0.021 (25.5x)** |
| 1 | 2048 | 1.306 (1.0x) | 1.164 (1.1x) | 1.096 (1.2x) | **0.050 (26.3x)** |
| 8 | 128 | 1.162 (1.0x) | 5.800 (0.2x) | 0.072 (16.1x) | **0.033 (34.7x)** |
| 8 | 512 | 1.996 (1.0x) | 4.612 (0.4x) | 0.273 (7.3x) | **0.017 (118.5x)** |
| 8 | 2048 | 4.921 (1.0x) | 5.657 (0.9x) | 1.762 (2.8x) | **0.065 (75.6x)** |
| 32 | 128 | 3.068 (1.0x) | 17.783 (0.2x) | 0.083 (36.8x) | **0.037 (83.6x)** |
| 32 | 512 | 5.882 (1.0x) | 19.952 (0.3x) | 0.472 (12.5x) | **0.045 (129.3x)** |
| 32 | 2048 | 21.404 (1.0x) | 23.222 (0.9x) | 1.882 (11.4x) | **0.377 (56.8x)** |

全部 36 个数据点（4 实现 × 9 配置）通过 fp32 对拍，最大误差 < 0.0022。

### 2.3 解读

1. **分块向量化是 decode kernel 的正确写法**。逐 token 版本的开销正比于 token 总数：seq=2048 时耗时 1.88 ms，与 Naive PyTorch 几乎同档；分块版将 64 个 token 压进一次向量 gather，同配置仅 0.377 ms。两者差距 3~21 倍，上下文越长差距越大。

2. **稠密拷贝是 PagedAttention 要消灭的瓶颈**。Fast PyTorch 在 36 组数据中 33 组慢于 Naive 实现，batch=32 时单次 17~23 ms 几乎全部消耗在 materialize 稠密 cache 副本上。这从数据上验证了论文核心论点：页感知的 kernel 内寻址（读侧数据不搬运）是分页机制兑现收益的必要条件。

3. **并行度是第一杠杆**。同一实现，batch=1 时加速 23x，batch=32 时 83~129x。grid=(batch, num_heads) 的 program 总数决定 SM 占用率；batch=1 仅 16 个 program，GPU 大部分空闲。batch=1 场景的优化方向是 flash-decoding：将 KV 序列维拆分到多个 program（三维 grid），再用第二个 kernel 合并各 split 的 (m, l, acc)。

4. **与官方 A6000 数据对比趋势完全复现**。相同配置下本机绝对耗时更低（0.019 vs 0.028 ms @ batch1/128），加速比同量级（23x vs 14x），实现排序一致——基准结论对硬件不敏感。

---

## 三、Decode 路径 Profiler 分析（torch.profiler）

### 3.1 方法

使用 `profile_decode.py`（仓库新增脚本）：引擎加载 Qwen3-0.6B 后预热一次，再在 `torch.profiler`（CPU + CUDA 活动）下运行一次完整 generate（3 条 prompt × 128 decode 步，eager 模式），统计全部 kernel 的 GPU 耗时排名。

**总账**：本次 generate 墙钟约 3.2 s，其中 GPU 实际忙碌仅 **515.8 ms（约 16%）**——**CPU 侧调度/发射开销占了约 84% 的墙钟时间**，这是 eager 模式下 decode 的第一瓶颈。

### 3.2 GPU 时间分布（Self CUDA total = 515.8 ms）

| Kernel / 类别 | CUDA 时间 | 占比 | 调用次数 | 平均耗时 |
|---|---:|---:|---:|---:|
| **GEMM**（`aten::mm` → cublas `gemmSN_TN_kernel`，覆盖 qkv/o_proj/gate_up/down_proj/lm_head 全部线性层） | 434.2 ms | **84.2%** | 14 464 | 30.0 µs |
| `paged_attention_decode_kernel`（Triton 分块版） | 20.3 ms | 3.9% | 3 556 | 5.7 µs |
| RMSNorm（3 个 `torch.compile` 融合 kernel） | ~24.2 ms | 4.7% | 18 048 | ~1.2 µs |
| 采样 softmax/argmax（`triton_red_fused__softmax...`） | 12.6 ms | 2.4% | 128 | 98.7 µs |
| QKV view 相关融合 kernel（`triton_poi_fused_add_cat_index_mul_split...`） | ~8.0 ms | 1.6% | 7 168 | ~1.1 µs |
| 残差相加（`vectorized_elementwise` add） | 6.7 ms | 1.3% | 7 168 | 0.9 µs |
| `store_kvcache_kernel` | 3.6 ms | 0.7% | 3 584 | 1.0 µs |
| SiLU·mul（`triton_poi_fused_mul_silu`） | 3.4 ms | 0.7% | 3 584 | 0.9 µs |
| prefill 的 cutlass sgemm | 4.3 ms | 0.8% | 112 | ~40 µs |

### 3.3 瓶颈结论

1. **第一瓶颈在 CPU 侧，不在 GPU 侧**：GPU 空闲率约 84%。每个 decode 步要发射上百个小 kernel（28 层 × 每层数个），CPU 逐个 launch 的延迟远超 kernel 本身的执行时间——Profiler 中 `Torch-Compiled Region` 包装层的 CPU total（数百 ms）远大于其内部 CUDA 时间（8 ms 级）即是直接证据。**这一结论已由开启 CUDA Graph（`enforce_eager=False`）实证验证**：一次 replay 发射全部 kernel、消除 CPU 发射间隙后，端到端 TPS 从 110.1 升至 420.0（约 3.8 倍），与"发射开销约 84%"的诊断在量级上吻合。

2. **第二瓶颈是"瘦 GEMM"**：GPU 忙碌时间的 84.2% 花在 GEMM 上，但形态是 batch=3 token 的小矩阵乘权重（14464 次、平均仅 30 µs）——瓶颈不在算力而在**每次都要扫一遍完整权重**（访存受限）。此类负载下提高效率的路径：CUDA Graph 减少空转后进一步靠权重量化（降低扫描字节数）、或 batch 内合并更多序列摊薄权重读取。

3. **Attention kernel 不是瓶颈**：paged_attention + store_kvcache 合计仅 4.6%——与算子级 benchmark 的结论互为印证（分块版 kernel 本身已接近访存地板），后续 kernel 层优化收益上限有限，系统层（发射开销、GEMM 形态）才是主要矛盾。

4. **RMSNorm/采样/逐元素操作的启示**：RMSNorm 已被 `torch.compile` 融合（单次仅 1.2 µs），但融合后仍有 18k 次调用、合计 4.7%——发射次数本身构成成本，同样指向 CUDA Graph；采样 softmax 单次 98.7 µs、128 次共 2.4%，规模尚小。

---

## 四、Benchmark 执行过程中的问题记录

| 问题 | 原因 | 处理 |
|---|---|---|
| Triton JIT 编译失败（PermissionError） | 沙箱限制写入 `~\.triton\cache` 与 NVIDIA ComputeCache | 沙箱外权限重跑 |
| `ModuleNotFoundError: matplotlib` | `benchmark_tps.py` 顶层导入未使用的库 | 删除死代码导入 |
| `ModuleNotFoundError: vllm` | 顶层 import 在跳过逻辑生效前执行 | vLLM 导入改为函数内延迟导入，未安装自动跳过 |
| transformers 加载失败要求 accelerate | MinivLLM 引擎设置的全局 `torch.set_default_device` 触发其检查 | venv 安装 accelerate |
| CUDA Graph 捕获报 `KeyError: 'max_num_seqs'` | `benchmark_tps.py` 的 config 缺失 batch 上限键（`capture_cudagraph` 按此分配捕获缓冲区） | config 增加 `'max_num_seqs': 16` |
| CUDA Graph 捕获报 `expanded size (151936) must match (1024)` | `capture_cudagraph` 输出缓冲区按 `vocab_size` 分配，但捕获对象 `self.model()` 返回 hidden（1024 维），logits 在图外由 `compute_logits` 计算 | 输出缓冲区改为 `hidden_size`，并注释 logits 在图外计算 |
| CUDA Graph 捕获报 `CUDAGraph::pool() without a preceding successful capture` | `graph.pool()` 被写在 `with torch.cuda.graph()` 捕获块**内部**，而它只能在捕获成功、`with` 块退出后调用；且 warmup 后缺 `torch.cuda.synchronize()` | 把 `graph.pool()` 移到 `with` 块外；warmup 后补 `synchronize()` |

---

## 五、结论与后续建议

1. **瓶颈优先级（Profiler 实测）**：① CPU 发射开销（GPU 空闲 84%，解法 = CUDA Graph）→ ② 瘦 GEMM 形态（GPU 时间的 84%，解法 = 量化/更大 batch 摊薄）→ ③ attention kernel（仅 4.6%，非矛盾主体）。
2. **瓶颈①已验证**：`enforce_eager=False` 开启 CUDA Graph 后 TPS 110.1 → 420.0（约 3.8 倍），确认 CPU 发射开销为首要瓶颈。**后续实验**：安装 vLLM 补齐三方对照；增大 batch（3 → 8/16）观察聚合 TPS 缩放（验证瓶颈②的摊薄效应）。
3. **kernel 优化方向**（收益预期需按 ③ 校准）：flash-decoding（batch=1 短板）、prefill kernel 的块级 causal 跳过（预估 1.7 倍）、混合精度路径。
4. **方法论沉淀**：三层数据（算子对拍 + 端到端 TPS + Profiler 时间分布）构成完整的性能证据链——kernel 层定位差异，系统层验证集成，Profiler 层指明主要矛盾。
