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

`flameox setup` 会输出 MCP 客户端配置。显式传入 `--provider` 时，它可将对应的
Python 扩展安装到持久的 uv 工具环境；NVIDIA 等系统或厂商工具则提供外部安装
指引。setup 不会初始化或修改项目。

分析和未保存的采集不会写入持久状态。只有显式调用 `preserve_evidence` 或 CLI 的
`--preserve` 后，才会延迟创建用户级 Flameox 数据目录；`FLAMEOX_DATA_DIR` 可覆盖
平台默认位置。原生字节和证据清单按 SHA-256 寻址，并通过同一文件系统上的暂存、
校验、fsync 和原子重命名发布。Flameox 不修改项目的 Git 配置。

MCP 公开 `analyze` 与 `capture_and_analyze` 两个类型化操作；它们通过带判别字段的
能力请求保留每种分析选项、来源数量和兼容采集器的精确验证。证据生命周期由
`prepare_providers`、`preserve_evidence`、`rescue_evidence` 和 `query_evidence`
管理。工具搜索由 MCP 客户端负责。唯一资源模板是
`flameox://evidence/{evidence_id}`，只返回不可变规范清单，不公开原生载荷。

例如，有界预览通过 `analyze` 调用。能力和来源放在 `request` 内，语义分页大小是
顶层参数：

```json
{
  "request": {
    "capability_id": "artifact.preview",
    "sources": [{"kind": "path", "path": "/absolute/path/to/output.log"}]
  },
  "page_size": 100
}
```

`capture_and_analyze` 使用相同外层结构，其 `request` 另外包含 `target`、`provider`。
默认执行一次目标；成对实验直接通过 `request.experiment` 提供实验设计，
无需额外的执行模式开关。若结果包含 `next_page`，
应原样调用其中指定的工具和参数。采集的后续页由 `analyze` 读取，不会再次执行目标。
服务端的字节数、内存、超时等保护上限不是 MCP 调节旋钮；公开的响应范围参数只有
顶层 `page_size`。

`analysis_id` 仅在当前服务进程内有效；重启后过期。`evidence_id` 是持久的内容
身份。长任务属于当前 MCP 请求，通过 SDK 报告进度并响应取消；不存在脱离请求、
跨重启恢复的任务。

详细契约见英文文档：
[architecture](docs/architecture.md)、
[storage and evidence](docs/storage-and-evidence.md)、
[interfaces](docs/interfaces.md)、
[runtime safety](docs/runtime-safety.md) 和
[investigations](docs/investigations.md)。
