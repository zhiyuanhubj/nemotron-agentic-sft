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
- 给每道题设置明确的 step、agent wall-clock、单条命令 timeout 和 token 上限；报告每项实际配置。`step_limit_exhausted` 与 wall-clock/API `agent_timeout` 分开统计，两者都不自动归类成普通模型失败。
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

### 2026-10-02 no-egress 首轮的步数耗尽与缺 patch

截至 19:41 UTC，Pro Verified Nemotron-fail 子集的六个 no-egress 分片有 63 条终态：57 条有完整 `result.json`，其中 verifier 通过 10 条；另外 6 条以 `Requested output files missing: /app/patch.txt` 结束。离线审计覆盖到的 10 条通过轨迹全部有 `explicit_answer_reconstruction` 和 `raw_thinking_contamination`，strict 合格唯一任务为 0。

6 条缺 patch 结果中有 2 条跑满 1000 步后仍未写出 patch；单条耗时约 109 分钟，且其中一个任务有一次对整个 `/` 目录的递归 grep，在 `command_timeout=2400` 下运行满 2400 秒才终止。另一条 1000-step 缺 patch 任务耗时约 72 分钟。还有一条只跑了 89 步却耗时约 114 分钟，其中数次 shell action 各耗时约 9–16 分钟。其余缺 patch 结果在 14、121、168 步结束。最终状态都是 `run_error`，不是 API 超时、verifier 异常或可评分的 reward=0。应将它们记作 `no_patch_output`，再按轨迹判断是 agent 提前结束、耗时命令还是步数耗尽；不能统一归入基建故障，也不能只因错误字符串中提到输出文件就判为 verifier-invalid。

这批数据说明，1000-step 上限可能让个别无进展轨迹消耗很长时间，增加上限并不能保证产出 patch；过高的 shell command timeout 也会让一次低价值搜索占用数十分钟。当前运行使用 `command_timeout=2400`，只影响未来新启动的 `run_cleanroom_v41_seed.sh` 已改为默认 600 秒并允许通过 `COMMAND_TIMEOUT` 显式覆盖；在扩大 cohort 前，先用 canary 确认典型构建和测试命令可在该预算内完成。批量扩容前还应单独统计步数耗尽率、达到步数上限的耗时、最后若干步的 patch/test 进展和缺 patch 比例；评估可用的无进展提前停止策略后，再调整步数预算。不要把本批 10/57 的 verifier 通过率解读为整个 Pro Verified 的基准 pass@1：样本来自 Nemotron 未解决任务子集，且本 cohort 的 10 条成功轨迹均未通过 strict。

### 2026-10-02 no-egress prompt 来源核对

核对六个 `no_teacher_egress_part1..6` 的 `run_info.json` 后确认，实际运行参数 `inject_network_restriction_notice=false`；抽查保存的原始输入也未出现 AgentCompass 额外注入的网络访问限制提示。因此这六个 worker 使用的是 no-egress cohort，不应再按“旧网络提示 prompt cohort”解释其结果。

原始输入包含锁定的 mini-SWE-agent 2.4.5 官方 [`swebench.yaml`](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.5/src/minisweagent/config/benchmarks/swebench.yaml) instance template。除 patch 提交流程外，它还要求每轮输出 `THOUGHT`、至少执行 shell 命令、只改非测试源码、创建复现脚本并运行测试；这些工作流文字来自官方模板，不是 SWE-bench 题目，也不是 AgentCompass 的网络限制 addendum。它会影响 thinking 的长度和内容，是比较简化 prompt 时必须固定或单独标记的实验变量。审计遇到 `guard_policy_leakage` 标签时，应检查具体 raw evidence，区分模型复述官方工作流与复述自定义网络限制；仍须独立审查答案重建、raw-thinking 污染和 hacking 行为。

截至 2026-10-02 21:54 UTC，六个 no-egress 主分片有 101 条唯一任务终态，verifier-positive 22 条（21.8%），strict 合格 0。静态高召回规则将这 22 条全部标记为 `explicit_answer_reconstruction`、`raw_thinking_contamination`、`git_metadata_or_history_probe` 和 `guard_policy_leakage`；其中另有 15 条触发全局/替代来源搜索和测试修改规则。它们目前是待逐条语义复核的候选，不应把规则标签直接解释成“已成功获得答案”，也不应在没有请求/响应证据时放入 `05 / answer_not_obtained`。新审计输出要分别记录尝试动作、是否观察到答案信息、任务 verifier 结果和 thinking 来源线索。

cleanroom 启动器默认不注入网络限制提示；隔离仍通过最终 rollout policy 和 task container 的 `network=none` 实施。启动后要从 `run_info.json` 和原始第一轮输入复核参数和实际 prompt，不以启动脚本默认值替代运行证据。

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

### 2026-10-03 ReBench V2 canary 的 verifier 依赖隔离问题

`v41_swerebenchv2_verified_nemotron_fail_20261002_no_guard_cleanroom_canary25_worker_2333_all` 是 SWE-ReBench V2 对照批次，不是官方 Pro Verified 731 题 baseline。到 2026-10-03 03:21 UTC，25 题中 22 题已评分，全部 `reward=0`，没有 Harbor exception；另有 2 条运行中、1 条待跑。22 条 verifier 日志里有 18 条出现 DNS、依赖下载或外部服务不可达信号，3 个测试套件没有解析出测试结果。这个 0/22 不能作为模型能力成功率。

根因已在实际生成的 `cleanroom-overlay.json` 与 task overlay 中确认：agent 和 environment 均被设置成 `no-network`，Docker Compose 也将 task container 固定为 `network_mode: none`。一些源 task 的 `[verifier]` 虽声明了 `allowed_hosts`，但该环境覆盖会阻断 verifier/test script 在容器内访问包注册表或测试所需外部服务。对照任务中可见 npm 注册表、Go module proxy、GitHub release 和外部 API/数据源失败。静态网络错误信号不自动证明某条具体 F2P 测试必然因基建失败；必须结合该测试的失败输出判断，并将“模型补丁确实未满足测试”与“所需依赖/服务不可用”分开标记。

后续 no-egress cohort 必须在 agent rollout 前确认 verifier 的必需测试能在实际隔离环境中运行：锁定依赖应由可信 setup 阶段预取并缓存；需要外部数据的测试要准备固定本地 fixture；缺少依赖或必需测试未执行时，先标记 `infra_invalid` / `verifier_invalid`，不计为模型失败，也不纳入 pass@1 分母。agent tool container 仍保持 `network=none`，不得为解决测试依赖而让模型 shell 获得互联网或 teacher endpoint 访问。批次统计需列出已评分、基础设施异常、未运行测试和有效测试失败四类计数。

### 2026-10-03 Pro Verified no-egress 首轮和恢复方式

冻结的 Nemotron 已完成失败子集中筛出 152 个此前没有 V4.1 verifier 成功的唯一 task，语言为 Go 60、Python 53、JavaScript 36、TypeScript 3。no-teacher-egress 首轮每题计划一个 rollout。截至 2026-10-03 03:35 UTC，已启动 147 次，产生 134 个可审计 `result.json`，其中 verifier reward=1 有 29 次（占已启动 rollout 的 19.7%，占有结果记录的 21.6%）；part1 尚未完成。当前静态高召回审计把 29 条成功轨迹都标记为答案重建/thinking 污染，因此 strict 仍为 0。该比例只描述 Nemotron-fail 子集的 V4.1 首轮产出，不是 Pro Verified baseline pass@1；答案访问标签须结合原始 action 和实际信息观察证据逐条复核。

part1 的 worker 被中断后，原 AgentCompass run id 已存在，直接以相同 `--run-id` 重启会在 preflight 报 `Run id already exists`。正确恢复方式是用新的输出 id，并以 `--reuse <原 run id>` 指向旧结果；AgentCompass 随后复制可用 checkpoint 并继续未完成任务。不能把 `--reuse <原 run id>` 和相同的 `--run-id` 混用。恢复后进度需核对 `reused_tasks`、`pending_tasks`、`running_tasks` 和 checkpoint 错误；checkpoint 缺 evaluation record 的任务需要重新执行。

2026-10-03 03:39 UTC 已对首轮中无 verifier 成功的 100 个 task 安排一次补充 rollout，按 4 个互斥 shard 分发到 2332、2334、2391、2392；不重复首轮已成功的 26 个 task，也不与仍在恢复的 part1 重叠。2018 负责恢复 part1，2333 继续完成 SWE-ReBench canary。此时 6 个 distill allocation 中 5 个已观测到 GPU 负载；这些进度数字是运行快照，之后应从各自 `progress.json` 和 GPU 采样重新核对。

### 2026-10-03 guarded 生产线事故与恢复

一组在线 thinking guard 随交互会话结束而退出，造成约 52 条 ReBench 和 39 条 Pro Verified guarded rollout 的推理连接错误。它们是 `infra_invalid`，不能当作 V4.1 解题失败，也不能放进成功率分母。恢复时须把 guard 作为独立持久服务运行，并在每批启动前及运行中检查其版本、健康接口、上游 V4.1 与 Qwen judge 的模型 ID；失败任务使用新的 run ID，保留旧错误轨迹。

恢复批次使用互斥任务清单、每节点 2 个 agent、一次正常 rollout 和有限的瞬态 API 重试。Pro Verified 的 39 个连接错误任务已分成两个恢复批次，另有 20 个未跑过的候选；ReBench 的 36 个配对验证失败任务分成两个恢复批次。另一个节点换机后需重新加载 V4.1 权重，服务就绪才自动启动 20 题 guarded 批次。先前 V4.1 `reward=1` 仅用于选题，原轨迹的 hacking 或 thinking 污染不会因重跑计划而变成合格数据；新轨迹仍须单独通过 verifier 和离线全轨迹审计。

截至 14:45 UTC，严格按完整 Nemotron 失败证据配对的 clean 唯一任务为 **7**，宽口径 clean 为 **17**。报告应始终分开这两个分母。发布流程曾引用已经到期的计算分配，自动同步中断；改为现存分配后已重新开始全量构建，仍须检查上传完成日志和远端文件，再宣称 HF 已更新。

后续进一步定位到登录节点的采集 supervisor 被 `systemd-oomd` 反复杀掉，导致其子进程中的审计、依赖代理和 HF publisher 一起中断；模型计算节点上的 Docker 服务与 rollout 不依赖该 supervisor 的进程生命周期。需查看 user service 的 `NRestarts` 和 journal，而不能只看 watcher 当前有 PID。已经给 supervisor 和独立 guard 服务设置 OOM 选择保护；长时间的 HF 导出应迁出 supervisor 单独运行，并在迁移后确认 Hub revision，避免“构建开始”被误报为“上传完成”。

14:58 UTC 的节点调度：换机节点的 V4.1 服务已就绪，guard 健康检查通过，20 题严格分支正在预取环境镜像。两个 ReBench guarded 分片中，前 4 个终态全部因为 thinking/action 策略拒绝而未评分；保留一个分片作为严格对照，另一个分片停止后用同一冻结任务清单启动**独立**断网 no-guard 分支。新分支采用 mini-SWE-agent 的 low reasoning effort、1000-step 上限和每节点并发 2；任务容器无网络，runner 从宿主机调用 V4.1。此分支的 `reward=1` 只是 05 候选，仍需逐条证明未取得答案。不要把 guarded 的中止结果和 no-guard 的新 rollout 当作同一次评分。

### 2026-10-03 15:31 UTC 节点与导出核查

共享账号 `xgen-mm` 的 27 节点配额已满。当前保留 8 个训练节点（2377–2379、2418–2420、2437–2438）；采集使用 2392 和 2439 跑 05 no-guard、2333 跑 Pro Verified guarded，2401 跑 Qwen 审计。2439 的 V4.1 服务通过 `/v1/models`，18 题 worker 以并发 2 启动，两条轨迹已写入，8 卡 GPU 平均利用率采样约 98%。2391、2018、2284、2355 的低产出 guarded 分支已停止，以便训练作业取得节点。物理空闲节点在共享配额已满时不能视为可用节点。

05 导出曾仅凭旧运行名称判断是否隔绝网络；近期 `isolation_fix` 和 `v41_verified_no_guard_recovery` 的 `cleanroom-overlay.json` 已记录 agent 与环境均为 `no-network`，却会被误列为公网轨迹。现改为读取每个运行的 overlay（缺失时仍按旧规则），并让新 05 run 进入导出扫描。断网证明只解决网络来源问题，每条 `reward=1` 仍要检查本地答案来源、修改测试文件和 verifier 覆盖。2392 首批 3 条终态有 2 条 `reward=1`，但两条均修改测试文件且答案访问风险未排除，不能计入 05 合格集，更不能计入 strict。

HF 远端提交 `edcb50e2f00e82e83050ccbdf99441ffad9caf6c` 已确认：canonical strict 17、hybrid 149；05 全池通过旧版导出门槛的 trial 120 条、唯一任务 105 个。此提交早于上述导出修复；下一轮构建/提交需再次核对远端 revision。严格按完整 Nemotron 失败证据配对且 raw thinking 与 actions 均干净的唯一任务仍为 7 个。
