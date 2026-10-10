# flameox

**面向编码代理的本地、有界运行时证据层。**

Flameox 协调分析器、基准工具、跟踪处理器和明确指定的本地目标，将原生工件或命令转换为有界证据，并可选择保留证据。代理负责提出假设和选择实验；Flameox 记录观测到的输入、执行来源、类型化证据、覆盖范围和限制。

无需初始化工作区或维护项目配置。分析时传入工件的明确路径；采集时传入 argv 和绝对工作目录。Flameox 不搜索父目录、不修改项目文件，也不是托管服务。

## 快速开始

要连接 MCP 客户端，可运行全局配置向导：

```console
npx flameox@latest setup
```

直接使用 CLI 时，传入工件的绝对路径或明确命令：

```console
uvx flameox analyze artifact.preview /absolute/path/to/artifact.json
uvx flameox capture --provider direct --cwd "$PWD" -- python benchmark.py
```

向导会检测受支持的客户端，并询问要修改哪些配置。自动化时应明确指定客户端，例如
`--client codex --yes` 或 `--all --yes`；`--dry-run` 只报告路径和操作，不会写入配置。
向导会保留客户端配置中的其他内容，也不会修改项目文件。详见
[npm 包说明](npm/README.md)。

## 证据与存储

分析和未保留的采集使用有界会话暂存。缓存淘汰或服务停止后，`analysis_id` 会过期。显式保留证据时，Flameox 才会创建用户级数据目录，并按 SHA-256 保存原生字节和规范化证据包。可用 `FLAMEOX_DATA_DIR` 指定其他位置。保留证据不会自动要求保留完整控制台输出。

Flameox 区分观测、推导和推断。性能剖析可用于探索，但不能单独证明因果关系或性能提升。确认性结论需要有代表性的工作负载、明确的指标和估计量、兼容的身份、保留的样本以及语义预言机。

## MCP

MCP 服务不绑定工作区。它与 CLI 共用能力注册表，提供具名分析和采集工具。分析工具接收明确来源；采集工具接收类型化目标和采集器。能力专属字段直接显示在工具 schema 中。运行 `flameox mcp inspect` 查看目录；添加 `--capability CAPABILITY_ID` 或 `--tool TOOL_NAME` 查看对应 schema 和示例。完整工具目录及契约见[接口文档](docs/interfaces.md)。

工具直接内联返回完整的有界结果，包括失败信息和恢复操作。保留后的证据具有持久的 `evidence_id`；调用 `inspect_evidence` 可内联获取脱敏元数据和重新分析所需的来源。

采集接受 argv，不接受 shell 字符串。直接目标需提供绝对工作目录和有界环境覆盖。结果可能包含 `next_page`；请原样调用其中指定的分析工具和参数。采集后续页读取已生成的工件，不会再次执行目标。MCP 工作属于当前请求，因此取消会作用于该请求，服务重启后也不会留下脱离请求的后台任务。

详细设计见英文文档：[架构](docs/architecture.md)、[存储与证据](docs/storage-and-evidence.md)、[调查与实验](docs/investigations.md)、[适配器](docs/adapters.md)、[运行时安全](docs/runtime-safety.md)、[接口](docs/interfaces.md)和[测试](docs/testing.md)。开发流程见[贡献指南](CONTRIBUTING.md)。
