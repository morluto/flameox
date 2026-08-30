<h1 align="center">flameox</h1>

<p align="center"><strong>面向编码代理的本地运行时证据层，用于调查性能、内存、执行、并发和可靠性问题。</strong></p>

<p align="center">
  <img
    src="docs/assets/flameox-mascot-flamegraph.png"
    width="420"
    alt="flameox 吉祥物：两只角之间有火焰图的公牛"
  >
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12 或更高版本">
  <img src="https://img.shields.io/badge/Evidence-Stays_Local-F97316?style=flat" alt="证据保留在本地">
  <img src="https://img.shields.io/badge/Interfaces-CLI_%2B_MCP-7C3AED?style=flat" alt="CLI 和 MCP 接口">
</p>

<p align="center"><strong>Language / 语言</strong>: <a href="README.md">English</a> | 简体中文</p>

<!-- mcp-name: io.github.morluto/flameox -->

Flameox 协调受支持的性能分析器、基准测试工具和 trace 处理器，保留它们的
原生产物及来源信息，并向代理提供有边界的证据。代理说明要验证的内容；
Flameox 将有效执行范围记录到运行记录中，采集测量，并保留实验记录供复核。

## 快速开始

使用引导式安装流程接入受支持的 MCP 客户端：

```console
npx flameox@latest setup
```

重启客户端，打开要检查的项目，然后询问它：

> 在此项目中初始化 Flameox，并列出可用的性能分析能力。

声明工作负载后，可以提出具体问题：

> 调查 `<workload>` 中的 `<symptom>`。显示采集范围、最有力的证据、局限，
> 以及深入查看证据的链接。

`setup` 会安装有版本号的本地运行时，只修改已批准的客户端配置，并为所选
代理安装 Flameox skill。该 skill 帮助代理组织证据，但不强制采用固定方案。
项目初始化是独立步骤；只有客户端为固定的项目根目录执行初始化后，才会创建
`.diagnostics/`。

升级运行时不会迁移不兼容的 `.diagnostics/` 工作区。请保留旧工作区，创建新
工作区，并在删除旧工作区前导入仍需使用的原生产物。详见[格式与演进](docs/storage-and-evidence.md#format-and-evolution)（英文）。

如果要进行源码开发：

```console
uv sync --extra dev
uv run flameox init .
uv run flameox status
```

需要 Python 3.12 或更高版本，以及仓库中提交的 `uv.lock`。

## 调查流程

```text
症状 → 采集或导入 → 有边界的证据 → 假设
    → 区分假设的实验 → 支持、反驳或无法定论的结论
```

常见证据来源包括 pyperf、py-spy、pytest-reportlog、coverage.py、Memray、Perfetto、torch.profiler、Nsight Systems、Nsight Compute、ROCprofiler、Compute Sanitizer、NVBench，以及类型化的推理提供商导出结果。实际可用性取决于主机、权限、已安装的可选依赖（extra）和所选适配器（adapter）。如果缺少证据，Flameox 会明确报告，而不是静默换用更弱的来源。

性能分析结果适合探索。要得出性能或正确性结论，还需要具代表性的工作负载、声明过的指标和估计对象（estimand）、兼容的运行身份、保留的样本，以及合适的语义 oracle。

## 命名工作负载

命令以参数数组的形式写在 `flameox.toml` 中。参数只能声明为标量；Flameox 不执行 shell 展开。

```toml
[workloads.scan]
argv = ["python", "bench.py", "--implementation", "{implementation}"]
cwd = "."
timeout_seconds = 60

[workloads.scan.parameters]
implementation = ["baseline", "candidate"]

[workloads.scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "validate.py", "--implementation", "{implementation}"]
receipt_schema = "flameox.oracle-receipt.v1"

[experiments.scan_comparison]
workload = "scan"
design = "randomized_complete_blocks"
blocks = 10
treatment_factor = "implementation"
combination_policy = "cartesian"
primary_metric = "pyperf.workload"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.05
confidence_level = 0.95
random_seed = 1984

[experiments.scan_comparison.factors]
implementation = ["baseline", "candidate"]
```

跨 treatment 的等价性要求每次 oracle 调用都通过 `FLAMEOX_ORACLE_RECEIPT`
写出类型化 receipt。receipt binding 必须关联 pair、共享输入、比较属性、两侧
运行与输出身份、oracle、容差和结论。Flameox 不把相同 stdout 字节当作语义证明；
只有单侧检查的 oracle 会被记录为探索性验证。

MCP 的 `configure_workload` 工具会校验并写入规范定义，但不会执行命令。手动编写的有效定义会立即生效；不存在审批副本或第二份工作负载注册表。

```console
uv run flameox workload show scan --json
uv run flameox capture plan pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
uv run flameox capture run pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
```

规划阶段会为每个可执行文件解析一次。生成的绑定（binding）包含确切的调用路径、规范化目标、信任决策和文件身份。执行阶段会重新校验该绑定，而不是再次搜索 `PATH`。计划是短期、一次性的能力；完整意图会保存在工作区的 SQLite 控制平面中。

## 实验与分析

```console
uv run flameox investigations create \
  '{"question":"Does the candidate remove reverse-scan overhead?"}' --json
uv run flameox hypotheses record @hypothesis.json --json
uv run flameox experiment plan scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
uv run flameox experiment run scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
```

实验会保留随机化的处理顺序、尝试过的 trial、失败、取消、校验凭据和排除项。分析会通过同一个固定的 corpus snapshot 解析全部输入：

```console
uv run flameox analyze hotspots <run-or-artifact>
uv run flameox analyze scaling <experiment-id>
uv run flameox analyze compare @comparison-request.json
uv run flameox analyze memory <run-or-artifact>
uv run flameox analyze execution <run-or-artifact>
uv run flameox analyze pytorch <run-or-artifact>
uv run flameox analyze failures
```

只读分析不会创建持久化结论。如果结果应成为调查历史的一部分，请使用 `analyze record`、`analyze record-comparison` 或 `findings record`。

## 数据与安全边界

`.diagnostics/` 包含：

- `control-plane.sqlite3`：计划、操作、运行、修订、幂等记录及其关系；
- 按内容寻址的原生证据文件；
- 不可变的 Parquet generation 和 corpus commit；
- 可重建的 `catalog.duckdb` 分析缓存。

大体积证据不会存放在 SQLite 中。删除 `catalog.duckdb` 不会删除证据；`flameox catalog rebuild` 会从已提交的 generation 重新创建它。

运行记录保留实际尝试及解释结果所需的有效语义。原生 artifact 保留生产者输出；
不可变 generation 保存规范化证据。CLI 和 MCP 响应只内联当前任务所需的有边界语义和摘要，
并通过类型化引用及 MCP `ResourceLink` 提供更深层的证据读取。

CLI 和 MCP 服务器提供的是有边界、面向任务的操作，不接受 shell 字符串、原始 SQL 或任意证据字节。除非主动的隔离（containment）禁止，工作负载可以访问网络。控制进程只会在明确的 setup、upgrade、已批准的 provider 获取或显式启用的符号服务（symbol service）中进行网络 I/O；普通采集和分析不会进行网络 I/O。

受信任的本地采集路径会记录其没有强制隔离子进程。如果项目需要托管式隔离（containment），可以显式选择；当所需保证不可用时，规划阶段会拒绝继续。

## CLI 和 MCP 发现

```console
uv run flameox --help
uv run flameox mcp serve --project-root .
uv run flameox mcp inspect --project-root . --json
```

`mcp inspect` 是当前安装版本的工具 schema、注解和 resource template 的权威清单。工作流和信任语义请参阅 [CLI 和 MCP 边界](docs/interfaces.md)（英文）。

## 完整性与保留

```console
uv run flameox validate
uv run flameox validate --full
uv run flameox catalog validate
uv run flameox catalog rebuild
uv run flameox recover
uv run flameox gc
uv run flameox gc --apply
```

校验不会修复证据。垃圾回收默认只生成试运行（dry run）；只有提供 `--apply` 才会执行，而且候选对象会先移入可恢复的 trash 目录。永久清除还需要另一个明确指定已过期 trash manifest 的命令。

## 文档

- [架构](docs/architecture.md) — 模块和进程边界（英文）
- [存储与证据](docs/storage-and-evidence.md) — 权威数据、快照和发布（英文）
- [调查](docs/investigations.md) — 实验、分析和结论质量（英文）
- [适配器](docs/adapters.md) — 生产者所有权和兼容性（英文）
- [运行时安全](docs/runtime-safety.md) — 执行、文件系统、取消和保留（英文）
- [CLI 和 MCP](docs/interfaces.md) — 公共工作流和信任边界（英文）
- [测试](docs/testing.md) — 测试套件所有权和 CI 测试分组（英文）
- [贡献指南](CONTRIBUTING.md) — 开发和 pull request 流程（英文）

## 开发

```console
uv sync --extra dev
uv run ruff check src tests tools
uv run mypy src tests tools
uv run pytest -q
```

标记和 provider 命令请参阅[测试指南](docs/testing.md)（英文）。Flameox 使用 [MIT 许可证](LICENSE) 发布。
