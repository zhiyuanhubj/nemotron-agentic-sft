# V4.1 no-guard cleanroom 蒸馏操作手册

更新：2026-10-02。本文描述任务筛选、rollout、判分和数据准入。文中的批次数字是对应时间点的快照，不是当前实时总量。集群主机名、内网地址和个人工作目录已省略；用本地环境变量和分配信息替代。

## 目标与数据口径

对 Nemotron 学生未解决的 SWE-Bench Pro Verified 任务运行 DeepSeek V4.1 teacher，并区分三种结果：

| 类别 | 准入条件 | 用途 |
|---|---|---|
| `strict` | verifier 明确通过；完整 raw thinking 和所有 actions 均无答案重建、答案来源探测、绕过测试或利用隐藏信息 | 高置信 reasoning SFT |
| `05 / answer_not_obtained` | verifier 明确通过；允许记录 hacking 尝试，但必须审计证明没有从任何渠道成功取得答案 | 与 strict 分开做对照训练 |
| `rejected / quarantine` | verifier、运行环境或答案获取状态不完整，或确认已取得任务答案 | 不计入有效训练产量；保留原始证据 |

`reward=1` 只表示 verifier 报告通过，不能单独证明 trajectory 合格。网络隔离也不能阻止 agent 把任务问题发给允许访问的模型推理端点；因此要审计请求和响应，并区分“尝试查询”与“确实取得任务相关答案”。

## 任务池与语言配额

1. 从官方 Nemotron baseline manifest 选 `status=completed` 且 `correct=false` 的 Pro Verified task；排除 evaluation error、缺少有效终态或 verifier 基建异常的记录。
2. 每个 task 保存稳定 ID、语言、Nemotron 结果、teacher run ID 和源结果路径。rollout 分片之间必须 task ID 不重叠。
3. 用官方 Pro Verified 语言构成作总体目标。失败子集本身的语言分布可能不同；报告实际可选池与已采集集的分布，不能把原 benchmark 的 quota 伪装成已经满足。
4. 首轮冻结任务清单并记录文件哈希。每道题默认一次 rollout；只有确认的 infra-invalid 才可用新 run ID 重试，并保留 original/retry 映射。

2026-10-02 的一个 Nemotron-fail seed 快照有 183 个任务：Go 73、Python 63、JavaScript 41、TypeScript 6。这个集合是条件子集，不代表全部 731 道 Pro Verified 题的模型 pass@1。

## 启动前门槛

每个新 cohort 先跑 1–2 道 canary，确认下面各项后再扩容：

1. **Runner 与模型**：固定 mini-SWE-agent、模型 ID、采样参数、最大 steps、timeout、并发和 retry policy。检查 worker 实际进程的完整参数；只看到 Slurm allocation 为 `RUNNING` 不算 rollout 已启动。
2. **模型服务**：请求 `/v1/models` 确认模型 ID，再发一次实际 completion 请求。Judge 同样要验证实际 completion，不能只看容器 `Up`。
3. **隔离**：从实际 task container 验证网络策略；确认不能连接公开网络、代码托管站、上游答案源、teacher endpoint 或 benchmark grader。检查最终 resolved policy 和容器的实际网络配置；只检查输入 YAML 不够。
4. **仓库与测试**：从干净 checkout 构建 task overlay；清除答案文件和原始 Git 历史。官方 test patch 必须完整、原子应用；所有必需 F2P/P2P 测试需有明确状态，缺失状态按 verifier-invalid 处理，不能默认通过。
5. **工具环境**：确认 Harbor/mini-SWE-agent setup 成功、所需依赖可用、模型路由健康。隔离策略若允许内部只读依赖源，要单独验证它不会转发答案或开放任意网络。
6. **Prompt**：固定 mini-SWE-agent 2.4.5 的官方 `swebench.yaml` instance template，不添加安全规则型 system prompt 或 task prompt。网络隔离、任务环境清理和事后审计负责数据边界；prompt 不应列举上游答案、隐藏测试、缓存或外部来源等具体规避对象。不能把“提示已传入”当作行为符合的证据。

Canary 有 API 错误、setup 失败、测试 patch 冲突、缺失测试状态或答案自查询时，先停在该 cohort 做诊断；不要将这些结果并入普通模型失败分母。

AgentCompass 的 Pro Verified 准备代码会删除 agent 工作区原有 `.git`，再对当前基线快照初始化单提交仓库。因此原始提交历史和未来提交不可访问；但 agent 仍可能尝试 `git log`、`git show`、`git fsck` 等探查。strict 按动作意图审计，这些尝试仍然不合格，不能因为仓库已清理或网络已断开而放行。

## Rollout 与资源调度

- 固定唯一 run ID 和 task manifest；每个 worker 一个互斥分片。
- 采用保守并发起步：每节点最多 2 个 agent。先根据实际任务耗时、模型 endpoint 延迟和 error rate 扩容，不按 GPU 利用率单独判断吞吐。
- 给每道题设置明确的 step、agent wall-clock、单条命令 timeout 和 token 上限；报告每项实际配置。步数/时间耗尽记为 timeout，不自动归类成模型失败。
- 对短暂 API、镜像拉取或启动故障使用有界重试；普通 verifier reward=0 不自动重跑。
- 任何 INT/TERM、节点被回收或 controller 消失时，先查是否产生完整 `result.json`、trajectory 和 verifier 输出。没有终态结果的任务记为 unscored/infra-invalid。
- 通过文件更新时间和实际进程确认 worker 在推进；allocation 存在、容器 `Up`、GPU 有利用率均不能代替结果文件进展。

## 轨迹审计与导出

每条候选至少关联并保存：task ID、语言、baseline outcome、run/trial ID、完整 raw trajectory、patch、verifier reward、测试清单及日志、网络/答案访问信息和审计版本。

按此顺序审核：

1. 检查 task setup、agent exception、模型 API 和 verifier 是否都完整执行。
2. 核对官方 F2P/P2P 必测项及其明确结果；测试 overlay 冲突或缺少必需测试状态时不接受 verifier 结论。
3. 对 `reward=1` 检查最终 patch 满足任务要求，并审阅全部 raw thinking、tool calls、命令输出和 agent 可访问的推理请求/响应。
4. 给 hacking attempt 单独记标签，并判断它是否实际获取答案。只能证明请求失败、返回无关内容或答案不可访问时，才可考虑 `05 / answer_not_obtained`；不确定时进 quarantine。
5. `strict` 必须保留未编辑的完整轨迹并通过高置信审计。删除污染 thinking 不能证明后续 action 与 patch 未被污染；此类记录不得改标 strict。
6. 任务产量按 canonical unique task 计算。重复成功 rollout 记录为 trials，不增加唯一任务数；严格与其他类别分别统计且不得重叠。

## 故障归类

不要把所有 `reward=0` 都记成模型不会做题。至少分开：

| 类别 | 处理 |
|---|---|
| 有效测试失败/回归 | verifier 完整且题目目标未满足时，才计为有效模型失败 |
| `infra_invalid` | API 连接、镜像、依赖安装、worker/controller 或环境启动异常；不计模型失败 |
| `verifier_invalid` | test patch 冲突、必需测试未运行、coverage 缺失或 scorer 假阳性；修复后用新 cohort 重评 |
| `agent_timeout` | 单独报告耗时、steps 和部分 patch；不自动记为普通 reward=0 |
| `answer_access_unknown` | 缺少请求/响应证据，不能推断为“未成功获取答案” |

`fail_to_pass_missing` 本身不能证明 verifier 或测试 runner 有故障。需要连同 `eval_raw_data.tests`、stdout/stderr、patch 和 F2P/P2P 清单一起核对。若缺失测试对应的包因模型补丁编译失败，或测试确实失败，应记为有效模型失败；只有证明必需测试没有运行、测试 overlay 冲突或 scorer 解析错误，才记为 verifier-invalid。`Requested output files missing: /app/patch.txt` 表示没有可评分提交，单凭这一条不归因于基建。

### 已观察到的批次异常（2026-10-01 快照）

在 176 个终态结果中，互斥核对为：89 个有目标未通过证据、11 个 verifier patch 冲突、28 个 API 连接异常、40 个 agent timeout、5 个 setup 异常、1 个 coverage 假阳性、2 个 verifier 通过但轨迹不合格。该批次 raw reward=1 共 3 条，合格 strict/hybrid 为 0。此前流传的“207 trials / 12 reward=1”当时无法由冻结 trial 清单复算，不应当作可靠分母。

已识别的主要风险包括：共享路由恢复期间 API 连接错误；客户端瞬态重试被环境变量禁用；官方 test patch 对 agent 修改路径发生冲突；scorer 对缺失的 P2P 状态默认通过；agent 将任务问题发给允许访问的 V4.1 endpoint。修复脚本不追溯修复旧 overlay 或旧 reward；旧结果仍须保留并单独归类。

## 2026-10-02 Pro Verified seed 运行快照

首轮 80 次完成、27 次 verifier 成功；其后的用户消息 strict replay 完成 68 次、21 次 verifier 成功。两批有重复 task：合并后 90 个唯一任务，30 个任务至少成功一次；148 是 attempts 总数，不能当作独立 task 成功率。已完成样本的 strict 审计通过数为 0。用户消息级的追加规则不足以产出合格轨迹，因此仍须逐条离线审计；prompt 本身不是准入证据。

### 2026-10-02 strict system prompt 诊断

一次 183-task 试跑把“不要回忆上游实现/hidden tests、不要查看 Git 历史、不要修改测试”等规则追加到 system prompt。终止前有 8 条终态轨迹，其中 5 条 verifier reward=1；这 5 条全部未通过 strict 审计，并含答案重建、raw-thinking 污染和 prompt-policy 泄漏标签。抽查的 thinking 直接复述了 prompt 中的 hidden-test、上游实现和测试限制措辞，部分还明确推演或回忆上游修复。这个小样本显示该 prompt 写法会污染 thinking，但不足以单独估计正式成功率。

后续 183-task cohort 保持相同任务清单、mini-SWE-agent harness、1000-step 上限和 no-egress task environment，移除该 addendum，使用官方任务提示并保留原有离线审计门槛。前一批输出保留为 prompt 诊断记录，不计入 strict 产量；新批次需等完整结果和轨迹审计后再估计 clean yield。

### 2026-10-02 AgentCompass task-prompt 前缀修正

检查 no-egress 轨迹后发现，mini-SWE-agent 的 system/instance template 与上游 v2.4.5 配置一致，但 AgentCompass 的 Pro Verified benchmark 还在 issue 前额外添加了“不要访问代码托管站、module proxy、缓存”等来源限制。这些句子不属于上游 [mini-SWE-agent v2.4.5 SWE-bench 配置](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.5/src/minisweagent/config/benchmarks/swebench.yaml)，可能诱发 raw thinking 复述限制并被审计标成 `guard_policy_leakage`。本地 benchmark 前缀已改为只提供仓库路径和 issue 内容；隔离仍由 `network_mode: none` 执行，不在模型提示中列出具体答案来源。

已启动的 `no_teacher_egress_part1..6` 进程在启动时载入旧前缀，必须按旧 prompt cohort 审计和统计。后续 30 个既有成功候选先分成 6 条单任务 canary；确认每条都完成实际测试、没有 test-discovery/parser error，且保存的 task prompt 不再含旧限制语句后，调度器才启动其余 24 条。不同 prompt cohort 的结果不得合并计算成功率。

### 2026-10-02 teacher endpoint 网络边界修复

运行时检查发现 AgentCompass planner 曾将模型 endpoint 自动并入显式 `no-network` rollout policy，使任务容器实际拿到仅允许访问 teacher endpoint 的 allowlist。该 cohort 在产生终态前停止，任务容器与 proxy sidecar 已清理，输出保留为配置诊断且不计数据。修复后，planner 保留显式 `no-network`；mini-SWE-agent 的模型请求由外部 runner 发出，所以容器不需要访问 teacher。一个 canary 的 run metadata 显示最终 rollout policy 为 `no-network`，Docker task container 的 network 为 `none`，同时 runner 成功向 teacher 发起请求并继续执行工具。

该 canary 还关闭了 AgentCompass 自动插入任务输入的网络限制提示，避免污染 raw thinking；网络边界仍由代理/容器策略执行。后续每个 cohort 都需核对最终 policy、task container 网络、prompt 参数和真实模型请求链路，四者都符合预期后才扩量。

### 2026-10-02 Harbor canary 的 teacher 自询通道

另一路 mini-SWE-agent/Harbor canary 把模型 API 放在 agent task container 可访问的内部网络里。sidecar 确实拦截了公开 GitHub 和包注册表请求，但 `allowed_hosts` 仍包括承载模型路由的内部主机；agent shell 因此可以直接向同一 V4.1 `/v1/chat/completions` 发送任务定向问题。这个设置不满足“答案不可从 teacher endpoint 获取”的隔离条件。

其中一条 canary 直接询问了该任务隐藏测试文件的完整内容。V4.1 返回 6,000 个 reasoning tokens、23,485 字符 reasoning，普通 `content` 为空且因长度上限结束；agent 随后打印了响应 JSON 的前 5,000 字符，其中已经出现任务相关规则逻辑，并继续尝试另一个模型查询。该 worker 在 verifier 结束前被停止。此轨迹按**答案信息已取得**隔离保存，不能进入 strict，也不能作为“尝试 hack 但未成功”的 05 对照样本。

这说明外网阻断与 teacher endpoint 隔离是两项不同检查。不要用“agent 容器只允许访问内部 host”证明答案访问失败；若那个 host 也提供模型 API，shell 仍可自询。规模化前必须从 task container 确认它无法访问任何 teacher/chat-completions 服务，同时 trusted runner 的正常推理请求仍能完成。使用 runner 外置推理的 AgentCompass/no-network 路径时，应检查容器实际 network 为 `none`，并单独审计请求来源；不要把上述 Harbor 网络配置用于 strict 或 05 生产。

## 每次发布前的报告字段

- worker allocation 数、真实 agent process 数、已完成/运行中/未启动/异常 trial 数；
- trial reward、有效测试失败、infra-invalid、verifier-invalid、timeout 的互斥计数；
- `strict`、`05 / answer_not_obtained`、quarantine 的 trial 数和 canonical unique task 数；
- 语言分布、与目标分布的差值、重复成功占比；
- 有效唯一任务产量/小时、API 延迟和错误率；
- 所有尚未解决的 verifier、隔离、答案访问或训练准入风险。

不在完成离线审计前上传候选为 strict/05，不因 verifier reward、任务容器隔离配置或模型服务健康探针单独放行。
