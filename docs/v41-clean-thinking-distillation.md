# Nemotron 学生失败任务的 DeepSeek-V4.1 蒸馏数据方案

> 数据快照：2026-09-23 UTC。数据集会持续自动更新，本文数字用于记录当前方法和风险。

## 概述

当前处于蒸馏数据构造阶段：用 DeepSeek-V4.1-Flash 在 Nemotron student 失败的软件工程任务上生成 agent trajectory，再按 verifier reward、action 安全性和 thinking 污染情况分层。当前的统计只说明数据采集和筛选效果，尚不能证明 student 训练后的能力提升。

Hugging Face 数据集：

<https://huggingface.co/datasets/zhiyuanhucs/nemotron-student-fail-v41-clean-thinking>

## 1. 任务选择

- Student 是 Nemotron。
- 当前候选池包含 1,273 个 Nemotron 没有获得 reward=1 的软件工程修复任务。
- Teacher 是 DeepSeek-V4.1-Flash。
- 调度顺序优先考虑：
  1. V4.1 历史上已经 reward=1 且 action-clean 的任务；
  2. verifier 只剩一个测试失败、没有 regression 的 near-miss；
  3. 其他 student-fail 任务。
- 最终目标是约 400 个任务，语言 quota 为 Go 153、Python 146、JavaScript 90、TypeScript 11。

## 2. Teacher rollout

V4.1 通过 mini-swe-agent 在仓库环境里多轮执行：

- 阅读代码；
- 修改产品代码；
- 运行本地测试；
- 提交最终 patch；
- 由 benchmark verifier 给出 reward。

只有 verifier reward=1 的 trajectory 才可能进入训练数据。

## 3. 在线 thinking/action guard

每个 assistant turn 会经过：

- 确定性规则检查；
- Qwen semantic judge；
- action firewall。

主要拦截：

- 从 git history 恢复答案；
- 搜索 upstream PR 或公开源码；
- 访问 benchmark、grader 或 verifier；
- 修改、跳过或禁用测试；
- 读取 agent 或 benchmark 日志；
- 显式重建 canonical solution；
- controller/guard 提示泄漏。

当前 V165 最多重新生成 16 次。如果 thinking 仍有污染，但该 turn 的 tool call/action 单独通过 firewall，则：

- 清空该 turn 的 `content`、`reasoning_content` 和 `reasoning`；
- 保留原始 tool call；
- 标记为 `action_only_bridge`；
- 最终只能进入 hybrid tier。

## 4. 离线审计和数据分层

| 层级 | 处理方式 | 预期用途 |
|---|---|---|
| Strict | 保留完整原始 trajectory；thinking 和 action 都通过审计 | raw-thinking SFT |
| Hybrid | 只清空被定位为污染的 assistant turn，保留其他干净 thinking、tool call 和 patch | action/tool SFT |
| Rejected | action、来源、完成状态有问题，或者无法安全定位污染 turn | 不训练，仅保留审计记录 |

Canonical 文件每个 task 只选一个 trial，并保证 strict/hybrid 的 task 不重叠。全部 reward=1 trials 另外单独保存，允许同一任务有多个 trial。

## 5. 当前效果

最新正式导出口径：

- 共审计：26,936 trials
- reward=1：586 trials / 205 tasks
- Strict：14 trials / 11 tasks
- 可导出的 hybrid：126 trials / 65 tasks
- 无法定位污染 turn：11 trials / 7 tasks
- 最终 canonical：69 个唯一任务
  - strict：11
  - hybrid：58
- 语言分布：JavaScript 30、Python 19、Go 17、TypeScript 3

换算后：

- 全部审计到 reward=1 的比例约 2.18%；
- reward=1 中能够导出 strict/hybrid 的 trial 比例约 23.9%；
- 205 个成功任务中，canonical 最终保留 69 个，覆盖率约 33.7%。

V165 当前样本还很少：已审计 9 个，3 个 reward=1，2 个符合 hybrid 候选，0 个 strict，1 个因超限退出被拒绝。

## 6. 需要重点讨论的问题

### 6.1 Hybrid 定义可能过宽

58 个 canonical hybrid 中：

- 36 个带 `explicit_answer_reconstruction`；
- 11 个带 `semantic_thinking_contaminated`；
- 4 个带 `semantic_thinking_uncertain`；
- 只有 7 个是单纯的 `action_only_bridge`。

因此，58 个 hybrid 中有 51 个涉及答案重建、语义污染或者不确定判断。

污染 thinking 被删除后，action 仍然可能已经受到该 thinking 影响。Action firewall 可以判断命令本身是否直接作弊，但无法证明 patch 没有来自污染信息。这是当前最大的概念风险。

### 6.2 建议将 Hybrid 再拆分

- **hybrid-safe**：只有 action-only bridge、长度问题、格式问题等，不涉及答案重建或 semantic contamination。
- **hybrid-risky**：出现 answer reconstruction、semantic contaminated 或 semantic uncertain。

按这个保守定义，当前高置信数据大约是 11 个 strict + 7 个 safe hybrid，合计约 18 个唯一任务。其余 51 个可以继续保留，但不应默认与 strict 等权训练。

### 6.3 Hybrid loss masking

Hybrid 中污染 turn 的文本字段是空字符串，但 tool call 被保留。训练时需要明确：

- 不对空 thinking 字段计算普通文本 loss；
- 只监督 tool call、action 和最终 patch；
- 保留 `redacted_turns` mask；
- 避免将空 thinking 当成 teacher 的正常输出模式。

### 6.4 当前有两套可用数据口径

- `harvest400-status.json` 使用较窄的旧 hybrid 定义，目前约报 21 个 usable tasks。
- Hugging Face exporter 使用较宽的 thinking-redaction 定义，正式导出 69 个 tasks。

训练前需要统一 acceptance policy，否则“100 条有效数据”会有完全不同的含义。

## 7. 建议的验证实验

使用相同训练预算做三个 ablation：

1. Strict only
2. Strict + safe hybrid
3. Strict + all hybrid

在 task 和 repo 都不重叠的 held-out 集合上比较：

- verifier pass@1；
- hack/污染率；
- test tampering 和 source lookup 率；
- tool-call 格式错误率；
- 普通任务上的能力回退。

## 当前判断

Strict 数据质量高，但数量很少。Safe hybrid 值得使用。包含答案重建或 semantic contamination 的 51 条需要单独做 ablation，不宜仅凭 action 表面干净就直接视为可靠蒸馏数据。
