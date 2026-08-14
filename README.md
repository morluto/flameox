<h1 align="center">flameox</h1>

<p align="center"><strong>Local runtime evidence for coding agents investigating performance, memory, execution, concurrency, and reliability.</strong></p>

<p align="center">
  <img
    src="docs/assets/flameox-mascot-flamegraph.png"
    width="420"
    alt="flameox mascot: an ox with a flame graph between its horns"
  >
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12 or newer">
  <img src="https://img.shields.io/badge/Data-Stays_Local-F97316?style=flat" alt="Data stays local">
  <img src="https://img.shields.io/badge/Interfaces-CLI_%2B_MCP-7C3AED?style=flat" alt="CLI and MCP interfaces">
</p>

<p align="center"><strong>Language / 语言</strong>: English | <a href="README.zh-CN.md">简体中文</a></p>

<!-- mcp-name: io.github.morluto/flameox -->

Flameox connects profilers, benchmark tools, and trace processors to a local
evidence record. It preserves their native artifacts and provenance, then
exposes bounded evidence to the agent. The agent states what it wants to test;
Flameox captures the measurements and preserves the experiment record for review.

## Quick start

Install the local runtime and connect a supported MCP client through the guided
setup:

```console
npx flameox@latest setup
```

Restart the client, open the project you intend to inspect, and ask it to:

> Initialize Flameox in this project and list the available profiling capabilities.

The setup command installs a versioned local runtime and changes only approved
client configuration. Project initialization is separate and creates
`.diagnostics/` only after the client calls the initialization workflow for its
fixed project root.

For source development:

```console
uv sync --extra dev
uv run flameox init .
uv run flameox status
```

Python 3.12 or newer and the committed `uv.lock` are required.

## Investigation path

```text
symptom → capture or import → bounded evidence → hypothesis
        → discriminating experiment → supported, refuted, or inconclusive finding
```

Evidence sources include pyperf, py-spy, pytest-reportlog, coverage.py, Memray,
Perfetto, torch.profiler, Nsight Systems, Nsight Compute, ROCprofiler, Compute
Sanitizer, NVBench, and typed inference-provider exports. Availability depends
on the host, permissions, installed extras, and selected adapter. Flameox reports
missing evidence instead of silently substituting a weaker source.

A profile helps explore a problem; it does not establish a performance or
correctness conclusion. That requires a representative workload, a declared
metric and estimand, compatible run identity, preserved samples, and an
appropriate semantic oracle.

## Named workloads

Commands live in `flameox.toml` as argument arrays. Parameters are declared
scalars; there is no shell expansion.

```toml
schema_version = 1

[workloads.scan]
argv = ["python", "bench.py", "--implementation", "{implementation}"]
cwd = "."
timeout_seconds = 60

[workloads.scan.parameters]
implementation = ["baseline", "candidate"]

[workloads.scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "validate.py", "--implementation", "{implementation}"]

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

The MCP `configure_workload` tool validates and writes the canonical definition
without executing it. A manually authored valid definition is active
immediately; there is no approval copy or secondary workload registry.

```console
uv run flameox workload show scan --json
uv run flameox capture plan pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
uv run flameox capture run pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
```

Planning resolves every executable once. The resulting binding contains the
exact invocation path, canonical target, trust decision, and file identity.
Execution revalidates that binding instead of searching `PATH` again. Plans are
short-lived, single-use capabilities whose complete intent is retained in the
workspace SQLite control plane.

## Experiments and analysis

```console
uv run flameox investigations create \
  '{"question":"Does the candidate remove reverse-scan overhead?"}' --json
uv run flameox hypotheses record @hypothesis.json --json
uv run flameox experiment plan scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
uv run flameox experiment run scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
```

Experiments retain randomized treatment order, attempted trials, failures,
cancellations, validation receipts, and exclusions. Analyses resolve all input
through one pinned corpus snapshot:

```console
uv run flameox analyze hotspots <run-or-artifact>
uv run flameox analyze scaling <experiment-id>
uv run flameox analyze compare @comparison-request.json
uv run flameox analyze memory <run-or-artifact>
uv run flameox analyze execution <run-or-artifact>
uv run flameox analyze pytorch <run-or-artifact>
uv run flameox analyze failures
```

Read-only analysis does not create a durable claim. Use `analyze record`,
`analyze record-comparison`, or `findings record` when the result should become
part of the investigation history.

## Data and safety boundaries

`.diagnostics/` contains:

- `control-plane.sqlite3` for plans, operations, runs, revisions, idempotency,
  and relationships;
- content-addressed native artifacts;
- immutable Parquet generations and corpus commits;
- a rebuildable `catalog.duckdb` analytical cache.

Large evidence does not live in SQLite. Deleting `catalog.duckdb` does not
delete evidence; `flameox catalog rebuild` recreates it from committed
generations.

The CLI and MCP server expose bounded task-shaped operations, not shell strings,
raw SQL, or arbitrary artifact bytes. Workloads may access the network unless
active containment denies it. The control process performs network I/O only for
explicit setup, upgrade, approved provider acquisition, or explicitly enabled
symbol services—not during ordinary capture or analysis.

The trusted-local capture path does not enforce containment for child processes;
it records that limitation. Projects that require managed containment can select
it explicitly. Planning refuses when the requested guarantee is unavailable.

## CLI and MCP discovery

```console
uv run flameox --help
uv run flameox mcp serve --project-root .
uv run flameox mcp inspect --project-root . --json
```

`mcp inspect` is the authoritative inventory of tool schemas, annotations, and
resource templates for the installed version. See [CLI and MCP
boundaries](docs/interfaces.md) for workflow and trust semantics.

## Integrity and retention

```console
uv run flameox validate
uv run flameox validate --full
uv run flameox catalog validate
uv run flameox catalog rebuild
uv run flameox recover
uv run flameox gc
uv run flameox gc --apply
```

Validation never repairs evidence. Garbage collection is a dry run unless
`--apply` is supplied, and applied candidates first move to recoverable trash.
Permanent purge requires a separate explicit command naming an expired trash
manifest.

## Documentation

- [Architecture](docs/architecture.md) — authoritative module and process boundaries
- [Storage and evidence](docs/storage-and-evidence.md) — authority, snapshots, and publication
- [Investigations](docs/investigations.md) — experiments, analysis, and claim quality
- [Adapters](docs/adapters.md) — producer ownership and compatibility
- [Runtime safety](docs/runtime-safety.md) — execution, filesystem, cancellation, and retention
- [CLI and MCP](docs/interfaces.md) — public workflow and trust boundaries
- [Testing](docs/testing.md) — suite ownership and CI lanes
- [Contributing](CONTRIBUTING.md) — development and pull-request workflow

## Development

```console
uv sync --extra dev
uv run ruff check src tests tools
uv run mypy src tests tools
uv run pytest -q
```

See [the testing guide](docs/testing.md) for marker and provider commands.
Flameox is available under the
[MIT License](LICENSE).
