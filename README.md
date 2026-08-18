# Multi-Agent Negotiation Framework

面向 AscendC 算子代码生成的多智能体协商框架。项目将协商协议与任务执行解耦，支持 `K=N / K=1 / K=2` 拓扑、`init / refine / vote` 多轮协议、加权共识、循环破局，以及真实 NPU 编译反馈自纠错。

> **证据边界**：仓库包含 682 条 AscendC 算子任务语料，其中 5 个代表算子具备配套工程和测试脚本，可接入 Ascend 310B4 真机完成“生成 → 编译 → 运行 → 精度验证 → 错误回灌”。零密钥 Demo 只验证协议和接口，不代表模型质量，不用于计算协商增益。

## Why this project

单 Agent 代码生成容易受单点判断影响；简单增加重试次数又无法引入独立审查。本项目探索一种可评测的协商范式：多个角色提出方案、交叉批判、投票聚合，再将胜出方案交给真实执行器验证。执行失败时，核心错误会回灌模型继续修正，最终以硬件测试框架输出的 `passed Precision` / `test pass` 作为通过条件。

## Architecture

```text
Task
  │
  ├─ init: N roles propose solutions
  ├─ refine: cross-review and revise
  ├─ vote: weighted consensus and deterministic elimination
  │
  └─ winner
       │
       ├─ DemoExecutor: interface-only protocol demo
       └─ AscendCTestExecutor
            ├─ SSH upload
            ├─ CANN compile
            ├─ Ascend 310B4 execution + precision check
            └─ error extraction → LLM repair (up to configured cap)
```

代码分层：

```text
src/negotiation/
├── protocol.py                 # Agent、投票、权重、淘汰、循环破局
├── executors/
│   └── simulated.py            # 模式中立的 Demo 执行器
└── evaluation/
    ├── demo.py                 # 无密钥协议演示，不产出性能结论
    ├── real.py                 # 真实模型 + 310B4 预算上限对照
    └── reporting.py            # 快照、原始结果、置信区间、失败分类

framework.py                    # 旧导入兼容层
simulated_executor.py           # 旧导入兼容层
experiment_ascendc.py           # LLM、SSH、CANN 与反馈修正适配
evaluate.py                     # 评测入口
run.py                          # 统一命令入口
tests/                          # 协议、执行器和数据边界测试
```

## Implemented capabilities

- **可配置协商阈值**：`K` 控制从 Top-K 淘汰阶段切换到 Best-1 终局阶段的候选规模；支持 `K=N / K=1 / K=2` 三组对照配置。
- **多轮协议**：初始化提案、交叉改进、Top-K/Best-1 投票。
- **共识机制**：加权票分、一致性累计、权重更新和确定性淘汰。
- **收敛处理**：检测投票环并按历史一致性破局；测试覆盖三种 K 配置。
- **执行器解耦**：协议层不依赖模型、SSH 或 NPU，执行能力通过统一接口注入。
- **反馈自纠错**：提取最多 50 行核心错误，回灌模型迭代修正；每轮使用独立远程目录。
- **真实判定**：编译成功后在 Ascend 310B4 运行参考用例，精度通过才记为成功。
- **实验可审计**：逐题 JSONL、配置快照、数据集 SHA-256、调用预算、NPU 执行次数、失败类型和 95% Wilson 区间。

## Dataset boundary

| 数据层级 | 数量 | 用途 | 能否声称真机验证 |
|---|---:|---|---|
| AscendC 任务语料 | 682 | 任务描述、提示、参考字段和数据集研究 | 否 |
| 代表算子可执行子集 | 5 | Add / Relu / Sigmoid / Fmod / Asinh 工程化验证 | 可运行验证链路 |
| 当前公开真实结果 | 0 | 仓库尚未提交带原始日志的预算对齐实验 | 不可声称增益 |

运行边界检查：

```bash
python run.py dataset-info
```

预期输出包含 `Task records: 682` 和 `Hardware-executable representative tasks: 5`。

## Quick start

要求 Python 3.10+。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

python run.py test
python run.py demo --num 1
```

Demo 产物位于：

- `artifacts/demo/protocol_results.json`
- `artifacts/demo/protocol_replay.md`

其中明确写入 `evidence_tier=demo_only` 和 `performance_claim_allowed=false`。Demo 不输出 pass rate，也不计算 baseline 与协商差值。

## Real hardware evaluation

完整实验设计、控制变量和报告规则见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)；当前结果发布状态见 [docs/RESULTS.md](docs/RESULTS.md)。

### 1. Configure secrets locally

只使用环境变量，不要将密钥写入代码。参考 `env.example`：

```bash
DASHSCOPE_API_KEY=...
NPU_HOST=...
NPU_PORT=22
NPU_USERNAME=...
NPU_SSH_PASSWORD=...
NPU_REMOTE_WORK_DIR=/home/<user>/negotiation-eval
NPU_OPP_PACKAGES=/home/<user>/opp_packages
```

### 2. Run a homogeneous-model ablation

```bash
python run.py real-eval \
  --num 5 \
  --homogeneous-agent 2 \
  --call-budget 50 \
  --max-fix-rounds 10
```

该实验让同一个模型承担 baseline 与所有协商角色，降低“模型能力不同”对协议对比的干扰。每个“任务 × 模式”共享相同逻辑 LLM 调用上限，同时记录实际逻辑调用量和 NPU 执行次数。供应商客户端内部重试仍可能产生额外 API 请求，不能等同于严格费用对齐。

真实产物：

```text
artifacts/real/
├── experiment_snapshot.json   # 模型、参数、数据哈希、环境，不含密钥
├── raw_results.jsonl          # 每个任务/模式的原始记录
├── summary.json               # 通过率、95% CI、资源量与失败分布
└── runner_artifacts/          # 代码、编译日志和检查点
```

### Interpretation rules

1. 只有 `evidence_tier=real_hardware` 的结果可以用于描述通过率。
2. 必须同时报告样本数、95% 置信区间、平均 LLM 调用和 NPU 执行次数。
3. 少于 30 个可执行任务时只视为探索性结果，不在简历中写“协商显著提升”。
4. 调用次数上限相同不等于 Token/成本完全相同；接入供应商 usage 后才能做 Token 或费用对齐。
5. 如需异构模型系统结果，应与“同模型协议消融”分表展示，不能混为协商机制的因果增益。

## Testing and CI

```bash
python -m pytest -q
```

GitHub Actions 会验证：

- 参数边界与三种 K 拓扑收敛。
- 加权投票计算。
- Demo 执行器不读取协商模式。
- 数据集为 682 条语料、其中 5 条带可执行测试。
- 零密钥 Demo 能从干净环境完成。

## Failure taxonomy

真实评测自动将失败归类为：

- `compile_failure`
- `precision_failure`
- `budget_exhausted`
- `timeout`
- `infrastructure`
- `runtime_exception`
- `verification_failure`

原始日志始终保留，分类只用于汇总，不替代真实编译与运行输出。

## Portfolio description

在公开真实结果补齐前，建议使用：

> 设计可配置多智能体协商框架，支持 K=1/2/N 拓扑及 init/refine/vote 多轮协议；将协商逻辑与执行器解耦，接入 Ascend 310B4 编译、精度验证和错误回灌闭环；构建 682 条 AscendC 任务语料，其中 5 个代表算子完成真机验证链路接入，并实现单/多 Agent 调用预算上限对照评测。

公开项目说明应与仓库中的实现和实验产物保持一致。当前不能使用“682 个算子均完成真机验证”或“协商通过率提升 60%”等未经原始结果支持的描述。

## Security and reproducibility

- 仓库不包含 API Key、SSH 主机、密码或个人目录。
- `artifacts/`、检查点、日志和本地环境文件默认忽略。
- 真实评测拒绝在 LLM 初始化降级为 Mock 时继续写入真实结果。
- 快照只记录模型名、公开参数、数据哈希和环境版本。

## License

作者原创协议、评测和适配代码采用 MIT License。`ascend-ops-dataset/` 及 `sources/` 中的第三方材料不适用本仓库 MIT 授权，具体来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [ascend-ops-dataset/LICENSE_DATA.md](ascend-ops-dataset/LICENSE_DATA.md)。

## Current limitations

- 只有 5 个代表算子具备本仓库可直接调用的真机测试工程，无法支撑大样本统计结论。
- 真机评测依赖外部模型 API、CANN 8.0 和 Ascend 310B4 环境。
- `experiment_ascendc.py` 仍保留为硬件适配主模块，后续可继续拆分 SSH、LLM、Prompt 与检查点组件。
- 当前公平性控制以 LLM 调用上限为主，尚未完整记录供应商 Token usage 和费用。
