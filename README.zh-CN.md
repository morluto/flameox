# flameox

Flameox 是面向编码代理的本地、有界运行时证据层。它协调分析器、基准工具、
跟踪处理器和明确指定的本地命令，但本身不是分析器或托管可观测平台。

Flameox 无需初始化工作区，也没有命名工作负载、SQLite 控制面、持久 DuckDB 目录
或可轮询的后台任务。调用方直接传入原生证据的绝对路径，或
包含 argv、绝对 cwd、环境覆盖、提供方参数和限制的类型化目标。服务本身不绑定项目或工作区。

```console
uv run flameox mcp inspect
uv run flameox analyze artifact.preview /absolute/path/to/artifact.json
uv run flameox capture --provider direct --cwd "$PWD" -- python benchmark.py
uv run flameox mcp serve
```

`npx flameox@latest setup` 会输出 MCP 客户端配置。显式传入 `--provider` 时，它会
解析保存启动器所需的、按版本固定的 `uvx` 环境；这不会创建持久的 `uv tool` 安装。
NVIDIA 等系统或厂商工具则提供外部安装指引。setup 不会初始化或修改项目。

分析和未保存的采集不会写入持久状态。显式调用 `preserve_evidence` 或 CLI `--preserve`
才会创建用户级 Flameox 数据目录；`rescue_evidence` 和 CLI `--rescue-to` 则将证据发布到
指定的新目录。`FLAMEOX_DATA_DIR` 可覆盖平台默认位置。原生字节和证据清单按 SHA-256
寻址，并通过同一文件系统上的暂存、
校验、fsync 和原子重命名发布。Flameox 不修改项目的 Git 配置。

MCP 工具目录包含 26 个具名分析工具、20 个具名采集工具和 4 个生命周期工具。工具名
直接表示证据问题，例如 `preview_artifact`、`rank_cpu_hotspots`、
`capture_artifact_preview` 和 `capture_cpu_hotspots`。分析与采集分开注册，以准确表达
只读与执行效果。每个工具的 schema 都公开对应能力的类型化字段；能力字段位于顶层，
采集器字段与 `provider.kind` 一起放在 `provider` 对象中。没有能力选择器、请求包装层
或不透明的 `options` 字典。完整目录和契约见[接口文档](docs/interfaces.md)。

可用 `flameox mcp inspect` 查看紧凑目录，用 `--capability CAPABILITY_ID` 查看直接调用示例、
能力字段和兼容采集器字段 schema，或用 `--tool TOOL_NAME` 查看单个工具的完整 MCP schema。MCP 调用无需先做
目录发现。证据生命周期由 `prepare_providers`、`preserve_evidence`、`rescue_evidence`
和 `query_evidence` 管理。唯一资源模板是
`flameox://evidence/{evidence_id}`，只返回带摘要绑定的脱敏清单视图，不公开原生载荷。

例如，有界预览直接调用 `preview_artifact`，将来源和能力字段放在顶层：

```json
{
  "sources": [{"kind": "path", "path": "/absolute/path/to/output.log", "format": "text"}],
  "text_fragment_chars": 1024,
  "page_size": 100
}
```

采集示例使用 `capture_artifact_preview`：`target` 和 `provider` 与能力字段一样位于顶层；
采集器专属字段与 `kind` 并列放在 `provider` 内。默认执行一次目标。仅支持实验设计的
采集工具才会公开顶层 `experiment`，用于描述随机顺序、重复次数、指标、估计量和可选语义
预言机，无需额外的执行模式开关。若结果包含 `next_page`，应原样调用其中指定的分析工具
和参数。采集的后续页只读取已生成的原生工件，不会再次执行目标。

```json
{
  "target": {
    "argv": ["python", "benchmark.py"],
    "cwd": "/absolute/path/to/project"
  },
  "provider": {"kind": "direct"}
}
```

服务端的字节数、内存、超时等保护上限不是 MCP 调节旋钮；公开的响应范围参数只有
顶层 `page_size`。

`analysis_id` 仅在当前服务进程内有效，可能因缓存淘汰或重启而过期。`evidence_id` 是持久的
内容身份。长任务属于当前 MCP 请求，通过 SDK 报告进度并响应取消；不存在脱离请求、
跨重启恢复的任务。

详细契约见英文文档：
[architecture](docs/architecture.md)、
[storage and evidence](docs/storage-and-evidence.md)、
[interfaces](docs/interfaces.md)、
[runtime safety](docs/runtime-safety.md) 和
[investigations](docs/investigations.md)。
