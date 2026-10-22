# MinivLLM Decode 优化实验报告：批规模扩展与 CUDA Graph 边界

**测试日期**：2026-10-22
**硬件**：NVIDIA GeForce RTX 5080 (sm_120, 16 GB)
**软件**：Python 3.11 · PyTorch 2.11.0+cu128 · Triton 3.8.0
**模型**：Qwen3-0.6B（本地完整权重）。权重文件为 fp16，但加载时经 `param.data.copy_(weight)` 装入参数，而参数 dtype 由 `torch.get_default_dtype()` 决定（出厂默认 **fp32**），故模型实际以 fp32 参与计算——这一点是理解后文 GEMM 占比的关键。
**测试脚本**：`benchmark_tps.py`（批规模扫描）
**关联报告**：`MinivLLM-Benchmark-报告.md`（原始端到端与算子级基准）

---

## 摘要

本报告验证前序 Profiler 结论（decode 的第一瓶颈为 CPU 发射开销，占墙钟约 84%）之后的两项系统层优化，并记录过程中发现并修复的五处 CUDA Graph 缺陷。

| 实验 | 措施 | 结果 | 判定 |
|---|---|---|---|
| ① | 并发序列数 3 → 8 → 16 | 聚合吞吐 590 → 1305 → 1891 tokens/s（3.2 倍） | **成立**，边际摊薄递减 |
| ② | `lm_head` 的 `compute_logits` 捕获进图 | 三档差异 +0.9% ~ −6.2%，复测回到基线 | **不成立**，净收益为零 |

附带修正一项既有结论：修复 padding 缺陷后，CUDA Graph 相对 eager 的提升由原报告的 3.8 倍修正为 **5.4 倍**（110.1 → 590.1 tokens/s，同为 3 条序列）。

---

## 一、实验背景

前序 Profiler 分析给出如下诊断（详见关联报告第三章）：一次完整生成（3 条 prompt × 约 112 token）墙钟约 3.2 s，而 GPU 实际忙碌仅 **515.8 ms（约 16%）**，即约 84% 的时间 CPU 在逐个发射 kernel。GPU 时间内部，GEMM 占 84.2%、paged attention kernel 仅占 3.9%。

据此形成两条优化路径：

1. **消除发射间隙**——启用 CUDA Graph（`enforce_eager=False`）。已验证有效，TPS 由 110 提升至 420 量级。
2. **摊薄访存成本**——GEMM 是 batch=3 的"瘦矩阵乘全量权重"，每次都要扫一遍完整权重，属访存受限。增大并发批可让同一次权重读取服务更多 token。

实验①对应路径 2，实验②则检验路径 1 的边界：CUDA Graph 既然已把主体计算全部收进图内，继续把 `lm_head` 也纳入是否有额外收益。

---

## 二、实验①：批规模扩展

### 2.1 方法

`benchmark_tps.py` 改造为批规模扫描：同一引擎实例只初始化一次（权重加载占启动成本大头），依次对 `BATCH_SIZES` 中各档执行 2 次预热 + 3 次计时（取中位数），档间插入 3 s 冷却以规避 GPU 频率漂移。前 3 条 prompt 与历史基线完全一致，超出部分以变体后缀循环补齐，保证各档负载同质。

```python
for bs in BATCH_SIZES:
    prompts = build_prompts(tokenizer, bs)
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()
    latencies = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        cuda_sync()
        latencies.append(time.perf_counter() - start)
    latency = float(np.median(latencies))
```

### 2.2 结果

生成上限 256 token/序列、总长上限 128，各档每序列实际生成约 108~112 token。CUDA Graph 已启用，`lm_head` 在图外计算。

| batch | 图行数 | 延迟 (s) | 生成 token | 聚合吞吐 (tok/s) | 单序列吞吐 (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 0.569 | 336 | **590.1** | 196.7 |
| 8 | 8 | 0.671 | 875 | **1304.6** | 163.1 |
| 16 | 16 | 0.916 | 1732 | **1890.8** | 118.2 |

对照：同机 eager 模式、3 条序列为 110.1 tok/s。

### 2.3 解读

**1. 聚合吞吐接近线性放大，但边际递减。** 批由 3 增至 16（5.33 倍），聚合吞吐由 590.1 增至 1890.8（3.20 倍）；批由 3 增至 8（2.67 倍）时吞吐为 2.21 倍。说明在 batch≈8 之前摊薄收益最显著，之后每 token 分摊到的权重读取成本下降变缓——与"瘦 GEMM 受限于权重扫描字节数"的判断一致：当批足够大，GEMM 的算术强度上升，瓶颈逐步由访存转向计算。

**2. 单序列吞吐下降是延迟换吞吐的正常代价，不是效率退化。** 批由 3 增至 16，总耗时仅由 0.569 s 升至 0.916 s（1.61 倍），而产出的 token 由 336 增至 1732（5.15 倍），净得 3.20 倍聚合吞吐。代价是每个序列的返回间隔被拉长——这正是连续批处理的核心权衡：**聚合吞吐是服务端指标，单序列 TPS 是用户侧延迟指标**，二者随批增大呈反向变化。

**3. 相对 eager 为 5.4 倍。** `590.1 / 110.1 ≈ 5.36`。原报告给出的 3.8 倍（420/110.1）是在 padding 缺陷尚未修复的状态下测得的，低估了实际收益；修正后 CUDA Graph 的贡献较前序结论更大。

**4. 小批仍存在固定浪费。** batch=3 时最小可用图为 4 行，1 行空算，效率 3/4。若要覆盖任意批，需在 `batch_sizes` 中加入更密的档位（如 3、6、12），代价是每档多一份图与内存池，捕获时间与显存占用随之上升。当前 1/2/4/8/16 的档位对常见批是合理的折中。

---

## 三、实验②：`lm_head` 捕获入图

### 3.1 动机

CUDA Graph 的捕获范围止于 `self.model()` 的输出，`compute_logits`（`lm_head` 的 `[bs, hidden] × [hidden, vocab]` 矩阵乘）仍在图外逐次发射。直观上，将其一并捕获可再省一次 kernel launch。

### 3.2 实现

`model_runner.py` 中引入 `capture_logits` 开关，仅单卡生效（TP>1 时 `compute_logits` 内含 `dist.gather`，集合通信不可捕获）：

```python
self.capture_logits = config.get('capture_logits', True) and self.world_size == 1
...
if self.capture_logits:
    logits_out = torch.zeros(max_bs, self.config['vocab_size'], device=...)
else:
    outputs = torch.zeros(max_bs, self.config['hidden_size'], device=...)
...
with torch.cuda.graph(graph, graph_pool):
    hidden_states = self.model(input_ids[:batch_size])
    if self.capture_logits:
        logits_out[:batch_size] = self.model.compute_logits(hidden_states)
    else:
        outputs[:batch_size] = hidden_states
```

### 3.3 结果

A 组为 `capture_logits=False`（基线），B 组为 `True`。

| batch | A 组 | B 组（首测） | 差异 | B 组（复测） |
|---:|---:|---:|---:|---:|
| 3 | 590.1 | 587.2 | −0.5% | — |
| 8 | 1304.6 | 1224.3 | −6.2% | **1315.9（+0.9%）** |
| 16 | 1890.8 | 1845.9 | −2.4% | — |

复测针对差异最大的 batch=8，采用反向顺序（先 B 后 A）以排除热漂移的单向偏置。复测值 1315.9 高于 A 组基线 1304.6，说明首测的 −6.2% 属于测量噪声（同配置两次运行间波动可达 ±7%）。

### 3.4 反向结果分析

**结论：该优化净收益为零，不应合入。**

收益上限可用数量级估算：

- **可省部分**：图外仅剩一次 launch，约 3 µs。batch=8 时单步 decode 约 5.5 ms，占比 0.05%。
- **新增代价**：图内需把 `compute_logits` 的输出落到预分配缓冲，产生一次 `[8, 151936]` 的 D2D 拷贝（约 4.9 MB 读 + 4.9 MB 写，按 2.5 TB/s 计约 4 µs）。

二者量级相当，互相抵消。这与 CUDA Graph 的**二元性质**有关：一旦开启捕获，主体计算全部入图，剩余图外操作只剩单次 launch 的量级，继续扩展捕获范围的边际收益迅速趋零；而捕获范围扩大反而要求引入持久缓冲，带来额外的写入流量。

该结论对后续优化有方向性意义：**CUDA Graph 的收益已被吃尽，不应再在"捕获更多算子"上投入；剩余提升必须来自计算/访存本身**（增大批、降低权重字节数、减少 kernel 数量），而非发射机制。

---

## 四、过程中发现并修复的五处缺陷

上述实验暴露出上游 `capture_cudagraph` 路径的若干缺陷。五处均已修复，并在干净的上游代码上重新验证（batch=3 走 padding 路径，三条序列生成文本均正常）。

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | `capture_cudagraph` 读取 `config['max_num_seqs']`，而 `main.py` 仅定义 `max_num_sequences` | 将 `enforce_eager` 改为 `False` 即抛 `KeyError` | 改为 `config.get('max_num_seqs', config['max_num_sequences'])` |
| 2 | 输出缓冲区按 `vocab_size` 分配，但 `self.model()` 返回 `hidden_size` 维隐状态 | 捕获期抛 `expanded size (151936) must match (1024)` | 缓冲区改为 `hidden_size`，并注明 logits 由 `compute_logits` 在图外产生 |
| 3 | `graph.pool()` 写在 `with torch.cuda.graph(...)` 捕获块**内部** | 抛 `CUDAGraph::pool() without a preceding successful capture` | 移至 `with` 块之后；warmup 后补 `torch.cuda.synchronize()` |
| 4 | 选图遍历 `self.graphs.keys()`，而图按 `16,8,4,2,1` 逆序插入 | batch=3 命中 16 行图，空算 13 行 | 改为 `next(bs_ for bs_ in sorted(self.graphs) if bs_ >= bs)` |
| 5 | 仅填充 `slot_mapping[:bs]`，pad 行保留上次 replay 的槽位 | `store_kvcache` 可能向已回收的物理块写入垃圾 KV | pad 行填 `-1`（kernel 遇 `-1` 直接返回） |

缺陷 4 的影响可量化。修复前后对比（`lm_head` 均在图外，仅此一项差异）：

| batch | 修复前 | 修复后 | 提升 |
|---:|---:|---:|---:|
| 3 | 425.7 | 590.1 | **+38.6%** |
| 8 | 1063.1 | 1304.6 | **+22.7%** |
| 16 | 1890.5 | 1890.8 | 0.0% |

batch=16 恰好命中 16 行图，不受影响，与缺陷的作用机制完全吻合。这也解释了修复前"单序列吞吐随批增大单调下降"的反常现象——小批一直被放大到 16 行空算。

缺陷 5 属于正确性隐患而非性能问题：`store_kvcache` 在 `-1` 时提前返回，若无该保护，pad 行会沿用上一步的槽位，在物理块被回收并复用后污染其他序列的 KV。

---

## 五、结论与下一步

### 5.1 结论

1. **CUDA Graph 是当前最有效的单点优化**，相对 eager 提升 5.4 倍（110.1 → 590.1 tok/s @ batch 3）。
2. **增大并发批可再获 3.2 倍聚合吞吐**（batch 3 → 16：590.1 → 1890.8 tok/s），且在 batch≈8 之后边际收益开始递减。
3. **继续扩大 CUDA Graph 捕获范围无收益**（`lm_head` 入图三档差异均落在噪声内），说明发射机制已不是剩余瓶颈的来源。
4. 上游 CUDA Graph 路径存在 5 处缺陷，其中 2 处导致捕获必然失败、1 处导致小批空算、1 处构成 KV 缓存污染隐患。

### 5.2 下一步

按预期收益排序：

1. **权重量化（W8A16 / W4A16）**。当前模型以 fp32 驻留，GEMM 每次扫描的字节数是 fp16 的两倍、fp8 的四倍。这是 GPU 时间 84.2% 那一项的根本解法，预期收益远大于此前的系统层优化。
2. **切换 fp16/bf16 权重**。如上文所述，当前模型以 fp32 计算，而权重文件本是 fp16；以 fp16 构造模型即可减半访存字节数并具备启用 Tensor Core 的前提。注意这不是改一行 dtype 就完事：`nn.Linear`、KV cache 缓冲区、Triton attention kernel 的 dtype 需同步适配并重新对拍。
3. **更密的批档位**。在 `batch_sizes` 中补充 3/6/12 等档位，消除小批的 padding 浪费（当前 batch=3 浪费 1 行，25%）。
4. **提高并发上限**。当前 `max_num_seqs=16`，受显存与 KV 块数（188 块）约束，可结合量化释放的空间继续上调。
5. **flash-decoding**。仅影响 batch 极小（1~2）场景，当前负载下 attention 占比不足 4%，优先级最低。

---

## 附：测量条件与数据可信度

- **计时**：每档 2 次预热后 3 次计时取中位数，`torch.cuda.synchronize()` 包夹。
- **漂移控制**：档间 3 s 冷却；关键对照（实验② batch=8）采用反向顺序复测。
- **噪声水平**：同配置重复运行间聚合吞吐波动约 ±7%（如 B 组 batch=8 两次为 1224.3 与 1315.9）。凡差异落在此区间内的结论均按"无显著差异"处理。
- **生成正确性**：各档输出文本均经抽样检查，未出现乱码或重复退化；padding 路径（batch=3）单独验证通过。
- **未纳入**：vLLM 三方对照（本机未安装）；batch=1/2 场景（未测，flash-decoding 收益待验证）。
