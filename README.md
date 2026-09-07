# HiveClaw

用 Python 从零搭建的 AI 编程助手系统：统一各家 LLM 的调用方式，让 Agent 能自动读代码、写代码、跑命令，最后还能在飞书群里直接跟它对话。

## 想法

> 不同 AI 厂商（Claude / GPT / GLM…）API 各有差异，我想做一个统一的入口，上层 Agent 只需要关心"调用模型"这一个动作；再把"模型调用 → 工具执行 → 再次推理"这套循环抽成通用内核；最上面做成一个可以聊天的编程助手，通过飞书随时用。

计划分几层来实现：

1. **ai** —— 统一 LLM 接口：消息/模型类型、流式调用、厂商 Provider 注册分发
2. **agent_core** —— Agent 编排内核：主循环、工具协议、事件驱动
3. **coding_agent** —— 编程 Agent 应用：内置工具（读写文件/执行命令/搜索）、会话持久化、CLI
4. **im** —— 飞书 IM 桥接：Webhook/长连接接入、会话路由

## 环境要求

- Python >= 3.10
- 一个 LLM 的 API Key（Anthropic / OpenAI / GLM 任选其一）

## 进度

- [x] 工程脚手架
- [ ] ai 统一接口层（进行中）
- [ ] agent_core 编排内核
- [ ] coding_agent 应用层
- [ ] im 飞书桥接
- [ ] 测试与文档

## 目录结构

```
HiveClaw/
├── pyproject.toml
├── dev.ps1 / dev.sh      # 开发启动脚本
├── .env.example          # 环境变量模板
├── src/                  # 源码（按上述分层组织，逐步添加）
└── tests/                # 单元测试
```

## 安全审批与项目记忆

使用 `--human-approval` 启用 Human-in-the-Loop。文件写入、文件编辑和 Shell 命令会在执行前展示风险等级与参数，交互模式下由用户输入 `y/N` 决定；程序化调用可通过 `session.pending_approvals()` 和 `session.decide_approval()` 接入飞书或管理后台。审批请求、决策人、时间和结果会写入当前会话的 `events.jsonl`。

项目长期记忆保存在 `.hiveclaw/memories.jsonl`，可在交互模式使用 `/remember <项目事实>` 添加。默认使用零依赖的本地特征检索；安装 `pip install -e ".[embeddings]"` 并配置 `HIVECLAW_EMBEDDING_ENABLED=true` 后，会使用本地 BGE-small-zh-v1.5 ONNX 模型生成语义向量。模型不可用时自动降级，存储仍使用本地文件，不依赖 Qdrant。每次 prompt 会检索相关记忆并注入上下文，支持 `memory_type`、来源会话、置信度和删除操作。
