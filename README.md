# Flamo

### Runtime evidence for coding agents

Give an agent profiler traces, benchmark results, memory captures, and execution
evidence it can query, compare, and audit—without uploading your code or captures.

[Quick start](#quick-start) · [What Flamo investigates](#what-flamo-investigates) ·
[How it works](#how-it-works) · [CLI and MCP](#cli-and-mcp) ·
[Documentation](#documentation)

```console
npx flamo setup
```

Flamo is a permanently local CLI and Model Context Protocol server. It connects
coding agents to maintained tools such as pyperf, py-spy, Perfetto Trace
Processor, coverage.py, Memray, and torch.profiler, then keeps their native
artifacts and the provenance needed to check a conclusion later.

Flamo is not another profiler and it is not an automatic bug finder. It gives
agents a consistent evidence workflow: collect from approved workloads, preserve
what happened, compare compatible runs, and keep observations separate from
inferences.

## Quick start

### Connect an agent

Run the guided setup:

```console
npx flamo setup
```

The wizard detects Claude Code, Cursor, OpenCode, Codex, Gemini CLI, and
Antigravity. Nothing is selected by default. It previews every configuration
file it will change, installs and verifies a versioned local runtime, and
activates only the clients you approve.

Restart the configured client, open a project, and ask:

> Initialize Flamo in this project and show me which profiling capabilities are
> available.

Flamo can initialize its local `.diagnostics/` workspace through MCP. Capturing
code requires one additional safety boundary: declare the command as a named
workload in `flamo.toml`, inspect its canonical form, and approve it once. The
[named workload example](#named-workloads-and-capture) shows that flow.

### Use the CLI from source

Python 3.12 or newer and `uv` are required:

```console
uv sync --extra dev --extra python --extra execution --extra memory --extra trace --extra cpu
uv run flamo init .
uv run flamo status
```

## What Flamo investigates

| Question | Evidence Flamo can connect |
| --- | --- |
| Where does this workload spend CPU time? | Sampled stacks, frames, callers, callees, and trace windows |
| Does runtime grow with input size? | Repeated measurements, scaling fits, uncertainty, and correlated hotspots |
| Why does memory grow? | Allocation records, retained memory, phases, threads, and processes |
| Which execution paths changed? | Coverage contexts, files, functions, branches, and two-run differences |
| What does PyTorch spend time on? | Operators, shapes when captured, CPU or accelerator time, and memory |
| Are failures clustered rather than isolated? | Failed attempts grouped by environment, source, workload, and error |

Profiles point to candidates; they do not prove causality or correctness. A
confirmatory comparison also needs a representative workload, declared metric,
compatible source and environment identities, preserved samples, and a semantic
oracle.

## How it works

1. You declare and approve a repeatable workload. Flamo never exposes arbitrary
   shell commands or SQL to an agent.
2. A maintained profiler or benchmark tool performs the capture. Flamo records
   the exact tool, command, environment, source identity, limits, and outcome.
3. Flamo preserves the native artifact and publishes normalized evidence into
   an immutable local corpus.
4. The CLI and MCP server expose the same bounded operations for hotspots,
   scaling, memory, execution, failures, and comparisons.
5. Findings remain tied to the runs, measurements, validation, and analysis
   that support them. Failed attempts remain visible.

Native artifacts and Parquet evidence are authoritative. DuckDB is a rebuildable
local query layer, and Perfetto Trace Processor handles detailed trace queries.

## Boundaries

Flamo supports local investigation of performance, memory, execution,
concurrency, and reliability behavior. It does not continuously monitor
production, upload evidence, provide accounts or synchronization, modify source
code, install system tools, or delete artifacts automatically.

Flamo also does not reimplement profilers, trace databases, native viewers, or
private format decoders. It reports a workload as contained only when an active
backend enforces that boundary, and it never treats statistical
non-significance as proof of equivalence.

## Setup and installation details

Run setup again to connect or disconnect clients, verify the active runtime,
update to the npm package's matching version, or roll back to a previously
installed version. `npx flamo@latest setup` resolves the newest setup release.
Automation can select clients and inspect the plan explicitly:

```console
npx flamo setup --codex --claude --yes
npx flamo setup --all --dry-run --json
npx flamo setup --verify --yes --json
```

The npm package is only a bootstrap. It delegates to the exactly matching
`flamo-diagnostics` Python release and supplies the maintained `jsonc-parser`
helper needed to preserve comments in OpenCode configuration. MCP clients launch
the installed runtime directly; they do not invoke `npx`, `uvx`, or a
network-dependent installer at startup. Setup itself does not initialize a
project or create `.diagnostics/`.

Optional Python extras are independent:

- `python`: pyperf capture and import
- `cpu`: py-spy capture
- `trace`: Perfetto Python API; a local Trace Processor binary must also be
  configured
- `execution`: coverage.py
- `memory`: Memray
- `torch`: PyTorch capture
- `all`: all runtime integrations

Flamo pins the official Python MCP SDK to `mcp==2.0.0b2`.

## Local data model

Initialize a project-local workspace:

```console
uv run flamo init .
uv run flamo status
```

`.diagnostics/` contains content-addressed native artifacts, immutable JSON
records, append-only Parquet generations, and a rebuildable DuckDB catalog.
Parquet and generation manifests are authoritative; deleting
`catalog.duckdb` is safe because `flamo catalog rebuild` reconstructs it.

Artifacts are deduplicated by SHA-256 without collapsing their contextual
registrations. A run records source and environment identity, workload and
measurement identities, lifecycle state, validation, process evidence, and
artifact roles. Investigations, hypotheses, experiments, variants, trials,
frozen run sets, comparisons, and findings remain separate domain records.

## Named workloads and capture

Repeatable commands live in `flamo.toml`. Templates accept declared scalar
parameters only—there is no shell expansion:

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
variants = ["baseline", "candidate"]
design = "randomized_complete_blocks"
blocks = 10
primary_metric = "pyperf.workload"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.05
confidence_level = 0.95
random_seed = 1984
```

Inspect and approve the exact canonical workload before exposing it through
MCP:

```console
uv run flamo workload show scan --json
uv run flamo workload approve scan
uv run flamo capture plan pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
uv run flamo capture run pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
```

Editing a command, environment, parameter domain, timeout, working directory,
or oracle changes the canonical digest and revokes that approval. Execution
uses argument arrays through one subprocess broker, bounded output, timeout and
cancellation cleanup, and optional Linux bubblewrap containment. Perfetto
parsing also runs in a broker-owned worker so a long trace cannot block or
outlive the MCP request. A truthful `uncontained` result is never relabeled as
sandboxed.

## Investigations and experiments

Create an investigation and optionally attach a falsifiable hypothesis before
running a predeclared experiment:

```console
uv run flamo investigations create \
  '{"question":"Does the candidate remove reverse-scan overhead?"}' --json
uv run flamo hypotheses record @hypothesis.json --json
uv run flamo experiment plan scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
uv run flamo experiment run scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
```

Experiment execution randomizes treatment order within complete blocks,
persists the declared protocol before collection, registers every attempted
trial—including cancellation and failure—freezes one trial-aware run set per
variant, and only runs the
automatic paired comparison when the blocks, measurements, source identity,
environment, and cross-treatment validation support it. Failed trials stay in
the evidence rather than disappearing from the denominator.

Useful read-only analyses include:

```console
uv run flamo analyze hotspots <run-or-artifact>
uv run flamo analyze scaling <experiment-id>
uv run flamo analyze compare @comparison-request.json
uv run flamo analyze memory <run-or-artifact>
uv run flamo analyze execution <run-or-artifact>
uv run flamo analyze pytorch <run-or-artifact>
uv run flamo analyze failures
```

Those commands are deterministic read-only previews. Persist a recipe result
and its typed provenance explicitly:

```console
uv run flamo analyze record \
  '{"recipe":"memory","input_id":"<run-id>"}'
uv run flamo analyze record-comparison @comparison-request.json
```

Hotspots can be followed into normalized trace structure without arbitrary SQL:

```console
uv run flamo stacks callers <run-or-artifact> <frame-id> [--cursor CURSOR]
uv run flamo stacks callees <run-or-artifact> <frame-id>
uv run flamo stacks examples <run-or-artifact> <frame-id>
uv run flamo trace window <artifact-id> --start 0 --end 1000000 [--cursor CURSOR]
uv run flamo open <artifact-id>
```

`flamo open` only prints a native viewer plan. `--launch` is an explicit
consequential action and cannot be combined with `--json`.

## CLI and MCP

Start the permanently local stdio server with a fixed project root:

```console
uv run flamo mcp serve --project-root .
```

The MCP layer is a thin adapter over the same application services as the CLI.
It offers approved named capture and experiment plans, bounded evidence and
drill-down queries, pure analysis previews, explicit `record_analysis` and
`record_comparison` operations, typed records, resources, progress, structured
domain errors, and cancellation cleanup. It does not expose shell strings,
arbitrary SQL, raw artifact bytes, approval mutation, deletion, or viewer
launching.

Plan tokens are 256-bit, in-memory, short-lived, bound to the current workspace,
approval, executable, policy, adapter, parameters, and experiment definition,
and atomically single-use. Restarting the server invalidates them.

Inspect the protocol surface with a real stdio client:

```console
uv run flamo mcp inspect --project-root . --json
```

## Integrity and recovery

```console
uv run flamo validate
uv run flamo validate --full
uv run flamo catalog validate
uv run flamo catalog rebuild
uv run flamo catalog compact
uv run flamo recover
uv run flamo gc
uv run flamo gc --apply
```

Full validation hashes native artifacts and Parquet files. Recovery closes only
runs whose exact boot/PID/process-start lease has disappeared. Garbage
collection is dry-run by default and `--apply` moves eligible objects into
recoverable trash instead of unlinking them.

## Documentation

- [Architecture](docs/architecture.md): process model, package boundaries,
  dependencies, and platform policy
- [Storage and evidence](docs/storage-and-evidence.md): authoritative data,
  identity, provenance, publication, and schemas
- [Investigations and analysis](docs/investigations.md): workloads, experiments,
  recipes, statistics, and evidence quality
- [Adapters and capabilities](docs/adapters.md): profiler integration,
  compatibility, probing, and approval behavior
- [Runtime safety](docs/runtime-safety.md): concurrency, recovery, retention,
  integrity, security, privacy, and local observability
- [CLI and MCP boundaries](docs/interfaces.md): human and agent interfaces and
  their trust boundaries
- [Architectural decisions](docs/architecture-decisions.md): settled choices and
  open design questions
- [Acceptance and verification](docs/acceptance.md): completion criteria and
  representative proof

## Development

```console
uv sync --extra dev --extra python --extra execution --extra memory --extra trace --extra cpu
uv run ruff check src tests
uv run mypy src tests
uv run pytest -q
```
