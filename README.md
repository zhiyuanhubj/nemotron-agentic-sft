# Nemotron-3.5-Super Agentic SFT

本分支整理了截至 **2026-10-06** 本机最新的 Nemotron-3.5-Super **数据处理与全参数 SFT** 代码。默认配方来自当天已完成的 Opus 4.8 reviewed-123 训练：123 个任务各一条经审查的成功轨迹，128 个不与训练重叠的独立任务用于 teacher-loss 验证。训练通过 NeMo Automodel 的 Omni/VLM 模型入口运行，当前数据是文本、thinking 和工具调用。

代码沿用实际训练使用的原生模板、标签掩码、长轨迹窗口、数据集、早停、token 计数和 NVMe 检查点逻辑；本机路径、作业号、W&B 账号和凭据改为运行参数。模型权重、自定义模型/processor 文件、原始轨迹、审查记录、训练产物与密钥需要另外准备，不在 Git 仓库中。

原来的 Nemotron-3-Ultra LoRA 文档保存在 [docs/ultra-legacy-readme.md](docs/ultra-legacy-readme.md)，旧脚本和配置仍保留。3.5-Super 使用下面的 `super/` 入口。

## 最新配方

| 项目 | 默认设置 |
| --- | --- |
| 模型入口 | `NeMoAutoModelForImageTextToText` / `FinetuneRecipeForVLM` |
| 优化范围 | 全参数语言模型与 embeddings；冻结 vision/audio tower |
| 分布式 | 8 节点 × 8 GPU；FSDP2，CP=8，EP=64，TP=1 |
| 序列长度 | 最多 131072 tokens；超过长度的轨迹拆成滚动窗口 |
| Batch | global 32，local 1；`drop_last: false` |
| Epoch | 最多 20；满足持续过拟合条件可提前停止 |
| 优化器 | AdamW，LR `1e-5`，betas `[0.9, 0.95]`，weight decay `0.1` |
| LR 调度 | cosine，warmup 10 steps，min LR `1e-6` |
| 精度 | 已完成配方加载 BF16 参数；FP32 master 对照配方单独标为草案 |
| 后端 | SDPA attention，torch linear / experts / dispatcher，FP32 RMSNorm |
| MTP | loss scaling `0.1` |
| 验证 / 保存 | 每 2 optimizer steps |
| 检查点保留 | 各节点 NVMe 保留 latest 2 与 best 3 的并集；共享存储保留本 run 最佳的一个完整备份 |

`super/configs/full_sft.yaml` 是最新 reviewed-123 配方；`v41_full_sft.yaml` 是上一版 V4.1 数据配方。`fp32_master_draft.yaml` 是 **尚未启动和验证的实验草案**：FP32 resident/master 参数、BF16 FSDP 计算、FP32 梯度归约。它需要独立 run 和实际 64-GPU 容量、更新及检查点验证，不能视为已完成训练的精度配置。

## 目录

```text
super/
  prepare_native.py           原生模板 tokenization 与 assistant 标签掩码
  prepare_v41.py              V4.1 数据选择、独立验证集和滚动窗口
  prepare_reviewed.py         最新已审查 Opus 成功轨迹处理
  data_gate.py                文件哈希、数组结构、独立任务和源审查校验
  agent_sft_data.py           memory-mapped NumPy 数据集与单次 causal shift
  configure.py               生成独立运行配置；不启动训练
  launch.py                  已有 Slurm allocations 的分发、启动及监控
  run_training.py             数据检查与训练入口
  nvme_checkpoint.py         节点分片保存与完整共享存储备份
  verify_checkpoint.py       完整 64-rank 检查点只读校验
  early_stopping.py          持续验证恶化且训练 loss 下降时早停
  token_budget.py            全局 assistant target token 计数 / 可选精确预算
  configs/                   三个训练配置模板
  tests/                     CPU 数据处理及分布式 token 计数验证
patches/nemotron35-super-automodel.patch
                            实际 checkout 的 Super 兼容性补丁
```

源代码版本及 SHA256 记录在 [docs/super-code-provenance.json](docs/super-code-provenance.json)。发布版保留训练算法，调整运行路径、配置生成和启动入口；不包含原集群的其他实验备份移交、清理或评测调度操作。

## 环境准备

实际运行使用 Python 3.12、PyTorch `2.10.0+cu130`、CUDA toolkit 13.0，以及 Automodel commit `8cb12a35b65eda421d8f1fa485c658eb06d22c76`。其他关键包版本见 [super/requirements-observed.txt](super/requirements-observed.txt)。这份版本记录不替代 CUDA、EFA/NCCL、Mamba 等本机依赖的安装。

```bash
git clone --branch nemotron-3.5-super-20261006 \
  https://github.com/zhiyuanhubj/nemotron-agentic-sft.git
cd nemotron-agentic-sft

git clone https://github.com/NVIDIA-NeMo/Automodel.git third_party/Automodel
git -C third_party/Automodel checkout 8cb12a35b65eda421d8f1fa485c658eb06d22c76
git -C third_party/Automodel apply --check ../../patches/nemotron35-super-automodel.patch
git -C third_party/Automodel apply ../../patches/nemotron35-super-automodel.patch

cd third_party/Automodel
uv sync --extra vlm --extra cuda
cd ../..
export AUTOMODEL="$PWD/third_party/Automodel"
export TRAIN_PYTHON="$AUTOMODEL/.venv/bin/python"
```

依赖安装结果应与上述已观测版本核对，尤其是 PyTorch/CUDA；`uv sync` 在其他机器上不保证选择相同的 GPU wheel。补丁包括 Super 架构注册、RADIO 权重映射、FSDP2/MTP 的 Tensor/DTensor 兼容处理，以及离线 consolidation timeout 参数。

模型目录必须包含完整 safetensors shards、index、tokenizer/chat template、processor/config 和模型自带的 remote-code 文件。代码通过 `trust_remote_code=True`、`local_files_only=True` 加载已在本地准备好的模型。

Slurm 启动依赖共享代码/数据存储、各节点 NVMe、`srun`、`rsync`、`nvidia-smi` 和 `ip`。集群需要为每节点提供 8 张 GPU。默认每节点 96 CPU、1500G 内存、NVMe 可用空间大于 1 TiB；启动参数可调整资源请求，模型实际容量需求需另行满足。

## 数据处理

原生模板完整渲染后一次 tokenize，并验证 token 序列与 `apply_chat_template(tokenize=True)` 完全相同。通过字符 offsets 决定标签，避免分段 tokenize 改变 BPE 边界。

- system、user、tool observation、assistant header、opening/empty think 均设为 `-100`。
- 原始 provider 返回的 thinking、assistant 内容、工具调用和 `<|im_end|>` 是训练目标。
- 工具 arguments 的 JSON 字符串转回字典；harness 的 `exit` 消息不进入训练。
- 超过 128K 的轨迹保留最初 system/task prefix 和最多 16K 最近上下文，优先按 assistant 轮次切割；超长单轮按 token 边界切割。
- 复用上下文标签全部屏蔽；每个原始 target token 在所有窗口中恰好训练一次。
- NumPy 数据在存储时不做 shift；collator 用 `input_ids[:, :-1]`、`labels[:, 1:]` 做一次 causal shift。

### 最新 reviewed-123 流程

需要本地 accepted manifest、当前审查目录、collection protocol、完整 result/task 文件，以及独立验证参考轨迹和 benchmark metadata。路径可以来自现有本机目录，原始内容不需要上传到 GitHub。

```bash
export NEMOTRON_MODEL=/shared/models/nemotron35super
export BENCHMARK_JSONL=/shared/benchmark/swebench_pro_verified.jsonl
export REFERENCE_JSONL=/shared/reference/resolved_once_per_task.jsonl.gz
export PREPARED_DIR=/shared/data/opus48_reviewed123_prepared

"$TRAIN_PYTHON" super/prepare_reviewed.py \
  --accepted-manifest /shared/reviews/accepted.json \
  --integrity-audit /shared/reviews/integrity_audit \
  --collection-protocol /shared/reviews/collection_protocol.json \
  --expected-tasks 123 --validation-tasks 128
```

Accepted manifest 的外层包含 `review_complete: true` 和 `accepted` 列表；每条记录包含 `task_id`、`source`（result.json 绝对路径）、`source_sha256`、`attempt`、`integrity_status: accepted`。Collection protocol 包含 `groups`，每组的 `manifest` 指向带有 `run_id` 和可选 `own_patch_repair` 的采集说明文件。result/task 文件保留原采集目录结构，用于任务身份和采集条件匹配。

处理器核对源 SHA256、当前 held source、任务身份、官方 completed/resolved 和全部 fail-to-pass/pass-to-pass 测试通过；保留真正返回的 thinking。默认每任务一条已审查轨迹，训练集为 `augmented/`；验证集 `monitor/` 按 benchmark 语言比例确定性采样，排除全部训练任务。输出 `input_ids.npy`、`labels.npy`、`offsets.npy`、每窗口 manifest、原始 messages JSONL、`summary.json` 和 `READY.json`。

独立任务 teacher loss 用于训练监控，不是 agent benchmark pass rate。若以后在训练任务所属的 benchmark 上评测，需要记录训练/测试任务重叠。

### V4.1 数据流程

```bash
export SOURCE_ROOT=/shared/v41-export
export PREPARED_DIR=/shared/data/v41_prepared
# NEMOTRON_MODEL、BENCHMARK_JSONL、REFERENCE_JSONL 同上
"$TRAIN_PYTHON" super/prepare_v41.py
```

`SOURCE_ROOT/data/` 下读取 `01_clean_thinking_and_actions`、`02_clean_actions_dirty_thinking_removed`、`05_swe_rebench_v2` 和 `05_swebench_pro_verified` 四组导出文件。按动作去重，每任务最多两条不同动作轨迹，适量减少 JS 的第二次尝试，保留所有任务；独立验证默认 128 个任务。输出 split 为 `train/` 与 `validation/`，训练时选择 `v41_full_sft.yaml`。

## 配置、分发和训练

先生成独立 run。默认不开启 W&B；如需启用，设置 `WANDB_ENTITY` 和 `WANDB_API_KEY`，或 `WANDB_KEY_FILE`（共享存储上的私有凭据文件）。不在配置中写入密钥。

```bash
export RUN_DIR=/shared/runs/nemotron35super_opus123

"$TRAIN_PYTHON" super/configure.py \
  --config super/configs/full_sft.yaml \
  --run-name opus123 \
  --run-dir "$RUN_DIR" \
  --prepared-dir "$PREPARED_DIR" \
  --model "$NEMOTRON_MODEL" \
  --accepted-manifest /shared/reviews/accepted.json \
  --integrity-audit /shared/reviews/integrity_audit

# 以下是占位 job IDs；替换为自己的 8 个以上已有单节点 allocations。
"$TRAIN_PYTHON" super/launch.py \
  --run-dir "$RUN_DIR" --automodel "$AUTOMODEL" \
  --jobs 1001,1002,1003,1004,1005,1006,1007,1008 \
  --wait
```

启动器从给定 allocations 中选择剩余时间最长的 8 个空闲节点，检查显存和 NVMe，分发模型/数据并再次检查 GPU，然后每节点启动 8 个 torchrun workers。默认网络匹配 `10.1.`，可用 `--subnet` 修改；CUDA、NCCL/EFA library 路径需要在启动环境中正确配置，AWS EFA 集群可设置 `FI_PROVIDER=efa`。日志在 `$RUN_DIR/logs/rank*.log`，运行状态在 `$RUN_DIR/state/status.json`。

启动失败会停止本次 launcher 创建的 workers。它不取消其他 allocations，不清理已有服务；已有 launch 状态必须先检查再恢复。默认 fresh launch 要求该 run 的节点检查点目录为空。

也可由自己的调度脚本直接调用训练入口；每节点使用相同配置和 `super/` 模块路径：

```bash
export PYTHONPATH="$PWD/super:$AUTOMODEL:${PYTHONPATH:-}"
"$TRAIN_PYTHON" -m torch.distributed.run \
  --nnodes=8 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29676 \
  "$PWD/super/run_training.py" "$RUN_DIR/train.yaml"
```

直接启动前须按 `state/READY` 中的 `local_model`、`local_data` 路径在每个节点完成分发。精确 token-budget 模式可以在模板中设置 `mode: exact`、`supervised_tokens: 整数`；最新完成配方使用 `track_only`，仅记录实际 optimizer 输入 target tokens，包括 sampler 重复。计数通过 DP group 汇总，不重复计算 CP replicas。

## 保存与恢复

保存模型、optimizer、RNG、dataloader 和 scheduler 状态。节点 coordinator 写入本地 NVMe；需要更新最佳共享备份时，收集全部节点分片，验证 safetensors headers、optimizer storage ranges、rank state 数量、文件尺寸和小文件 SHA256。完整备份发布 `COMPLETE.json` 和 `BEST` 指针后，才清理本 run 的旧完整备份。

```bash
"$TRAIN_PYTHON" super/verify_checkpoint.py "$RUN_DIR/checkpoints/BEST"
```

恢复只接受本 run 共享存储下已验证的完整 checkpoint。用 `configure.py --restore-from /shared/runs/同一run/checkpoints/epoch_N_step_M` 生成恢复配置（先归档原 `train.yaml`），再由自己的 Slurm 调度脚本直接调用 `run_training.py`。当前 `launch.py` 是 fresh-run launcher，已有节点状态恢复不自动重排。节点拓扑仍要求 8×8；恢复模型、optimizer、scheduler、RNG、dataloader 和 token-budget 计数。早停 hook 会保存历史，但当前实现重启时重新初始化 monitor；恢复时需考虑这一限制。

## 验证

```bash
# TEST_MODEL 设为真实本地模型 tokenizer 时，额外运行原生模板测试。
TEST_MODEL="$NEMOTRON_MODEL" "$TRAIN_PYTHON" -m unittest discover \
  -s super/tests -p test_pipeline.py -v
"$TRAIN_PYTHON" super/tests/test_token_budget_distributed.py
```

测试覆盖原生工具参数与 assistant 掩码、超长单轮窗口的目标守恒、单次 shift 与 padding、数据修改拒绝、完整检查点缺 rank state 拒绝、早停，以及实际 4 个 CPU/Gloo ranks 上的 DP2/CP2 token-budget 计数。新发布的可配置启动器尚未在新的 64-GPU 作业上重新跑完整训练。

## License

沿用仓库 Apache-2.0；Automodel 补丁和配置来源见 `NOTICE`。模型和训练数据的访问及许可由各自来源决定。
