# flameox

Flameox 是面向编码代理的本地、有界运行时证据层。它协调分析器、基准工具、
跟踪处理器和明确指定的本地命令，但本身不是分析器或托管可观测平台。

0.2 是一次不兼容重构：无需初始化工作区，没有命名工作负载、SQLite 控制面、
持久 DuckDB 目录或可轮询的后台任务。调用方直接传入原生证据的绝对路径，或
包含 argv、项目内 cwd、环境覆盖、提供方参数和限制的类型化目标。

```console
uv run flameox capabilities discover --intent "CPU 热点"
uv run flameox analyze artifact.preview /absolute/path/to/artifact.json
uv run flameox capture --provider direct -- python benchmark.py
uv run flameox mcp serve --project-root "$PWD"
```

`flameox setup` 会输出 MCP 客户端配置。显式传入 `--provider` 时，它可将对应的
Python 扩展安装到持久的 uv 工具环境；NVIDIA 等系统或厂商工具则提供外部安装
指引。setup 不会初始化项目，也不会创建 `.flameox`。

分析和未保存的采集不会在项目中写入 Flameox 状态。只有显式调用
`preserve_evidence` 或 CLI 的 `--preserve` 后，才会延迟创建
`<project>/.flameox`。原生字节和证据清单按 SHA-256 寻址，并通过同一文件系统
上的暂存、校验、fsync 和原子重命名发布。

MCP 只公开六个工具：`discover_capabilities`、`inspect_capabilities`、
`analyze`、`capture_and_analyze`、`preserve_evidence` 和
`query_evidence`。唯一资源模板是
`flameox://evidence/{evidence_id}`，只返回不可变规范清单，不公开原生载荷。

`analysis_id` 仅在当前服务进程内有效；重启后过期。`evidence_id` 是持久的内容
身份。长任务属于当前 MCP 请求，通过 SDK 报告进度并响应取消；不存在脱离请求、
跨重启恢复的任务。

旧版 `.diagnostics` 不迁移也不兼容读取。仍可将其中的原生证据按确切路径和格式
传给 `analyze`。

详细契约见英文文档：
[architecture](docs/architecture.md)、
[storage and evidence](docs/storage-and-evidence.md)、
[interfaces](docs/interfaces.md)、
[runtime safety](docs/runtime-safety.md) 和
[investigations](docs/investigations.md)。
