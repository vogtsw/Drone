# Drone — 低空网络 AI 方案

低空网络智能化的方案规划与参考实现。

## 文档

- [低空网络 AI 技术方案规划](docs/low-altitude-ai-network-plan.md) — 总体架构、四大模块（轨迹预测 / 干扰区域预测 / 联合功控 / 空域管控）、资源与数据需求、验证方案、业界数据集
- [TCEA Agent 设计 Spec](docs/specs/twin-calibration-agent-spec.md) — 孪生校准与模型进化智能体的完整规格说明

## TCEA 参考实现（`tcea/`）

按 Spec 的 MVP 边界（P0 + P1，自主修复 `network_config_change` 与
`environment_change` 两类根因，其余根因正确归因后升级人工）实现的
可运行闭环：

```
Monitor(CUSUM漂移检测) → Diagnoser(预算受控的工具推理) → Planner(最小代价修复)
  → Executor(staged副本执行) → Validator(冻结回归门禁) → Publisher(金丝雀+回滚)
  → Memory(案例库) + 置信度广播
```

| 目录 | 内容 |
|---|---|
| `tcea/twin/` | 玩具数字孪生（3GPP 风格天线/路损物理模型，truth/twin 双状态，漂移注入） |
| `tcea/monitor/` | P0 漂移监控：误差配对 + CUSUM 检验 + 基线管理 |
| `tcea/tools/` | P1 工具层：误差统计 / 工参 diff / 环境 diff / 探针仿真 / 案例检索 |
| `tcea/agents/` | Diagnoser / Planner 节点，策略可插拔（默认规则策略，可换 LLM） |
| `tcea/orchestrator/` | 六阶段状态机 + 修复执行器（staged 副本，GR1/GR2 隔离） |
| `tcea/validation/` | 回归门禁 Validator + 金丝雀/回滚/置信度广播 Publisher |
| `tcea/memory/` | 案例库（漂移签名相似度检索） |
| `tcea/benchmark/` | 注入式漂移基准（Spec 10.3 评估方法 #1） |

### 快速开始

```bash
pip install -r requirements.txt

# 运行测试（15 个用例：单元 + 端到端闭环 + Guardrail + 基准冒烟）
python3 -m pytest tests/ -q

# 运行注入式漂移基准评估
python3 -m tcea.benchmark.run_benchmark --per-type 5 --seed 1 --verbose
```

### 基准结果（45 场景，对照 Spec 10.1 验收线）

| 指标 | 验收线 | 实测 |
|---|---|---|
| 漂移检测召回率 | ≥ 0.90 | 1.000 |
| 归因 Top-1 准确率 | ≥ 0.75 | 1.000 |
| 修复有效率 | ≥ 0.85 | 0.933 |
| 自主闭环率 | ≥ 0.60 | 0.933 |
| 平均检出窗口数 | — | 2.3 |
