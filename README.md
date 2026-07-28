<h1 align="center">flameox</h1>

<p align="center"><strong>Runtime evidence for coding agents</strong></p>

<p align="center">
  <img
    src="docs/assets/flameox-mascot-flamegraph.png"
    width="420"
    alt="flameox mascot: a friendly ox with a flame graph between its horns"
  >
</p>

<p align="center">
  Let an agent query, compare, and audit profiler traces, benchmarks, memory captures,
  and execution evidence without uploading your code or data.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12 or newer">
  <img src="https://img.shields.io/badge/Data-Stays_Local-F97316?style=flat" alt="Data stays local">
  <img src="https://img.shields.io/badge/Interfaces-CLI_%2B_MCP-7C3AED?style=flat" alt="CLI and MCP interfaces">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> &nbsp;&middot;&nbsp;
  <a href="#what-flameox-investigates">What flameox investigates</a> &nbsp;&middot;&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;&middot;&nbsp;
  <a href="#cli-and-mcp">CLI and MCP</a> &nbsp;&middot;&nbsp;
  <a href="#documentation">Documentation</a>
</p>

<p align="center"><strong>Connect your agent:</strong> <code>npx flameox setup</code></p>

---

flameox helps coding agents investigate performance, memory, execution,
concurrency, and reliability with evidence you can inspect and reproduce. It
connects agents to maintained tools such as pyperf, py-spy, Perfetto Trace
Processor, coverage.py, Memray, and torch.profiler, then keeps each original
artifact alongside a record of how it was produced.

flameox is not a profiler or an automatic bug finder. It coordinates existing
tools, compares runs collected under compatible conditions, and ties findings
back to the measurements that support them.

## Quick start

### Connect an agent

Run the guided setup:

```console
npx flameox setup
```

The wizard detects Claude Code, Cursor, OpenCode, Codex, Gemini CLI, and
Antigravity. It selects nothing by default and previews every configuration file
it will change. After you approve the plan, it installs and verifies a versioned
local runtime and activates the clients you chose.

Restart the configured client, open a project, and ask:

> Initialize flameox in this project and show me which profiling capabilities are
> available.

flameox can initialize its `.diagnostics/` workspace through MCP. Before an agent
can run your code, you must declare the command as a named workload in
`flameox.toml`, inspect its canonical form, and approve it. See
[Named workloads and capture](#named-workloads-and-capture) for an example.

### Use the CLI from source

Python 3.12 or newer and `uv` are required:

```console
uv sync --extra dev --extra python --extra execution --extra memory --extra trace --extra cpu
uv run flameox init .
uv run flameox status
```

## What flameox investigates

| Question | Evidence |
| --- | --- |
| **Where does this workload spend CPU time?** | Sampled stacks, frames, callers, callees, and trace windows |
| **Does runtime grow with input size?** | Repeated measurements, scaling fits, uncertainty, and correlated hotspots |
| **Why does memory grow?** | Allocation records, retained memory, phases, threads, and processes |
| **Which execution paths changed?** | Coverage contexts, files, functions, branches, and two-run differences |
| **What does PyTorch spend time on?** | Operators, shapes when captured, CPU or accelerator time, and memory |
| **Are failures clustered rather than isolated?** | Failed attempts grouped by environment, source, workload, and error |

Profiles show where to investigate; they do not prove why behavior changed or
whether the program remains correct. To confirm a result, use a representative
workload and declared metric, compare the same source and environment, retain the
samples, and validate the program's output for both the baseline and candidate.

## How it works

1. **Declare.** You name a repeatable workload and approve the exact command an
   agent may run.
2. **Capture.** A maintained profiler or benchmark tool runs while flameox records
   the tool, command, environment, source revision, limits, and outcome.
3. **Preserve.** flameox keeps the original artifact and publishes queryable
   evidence to the project workspace.
4. **Analyze.** The CLI and MCP server provide focused queries for
   hotspots, scaling, memory, execution, failures, and comparisons.
5. **Record.** Findings remain tied to the runs, measurements, validation, and
   analysis that support them. Failed attempts remain visible.

## Safety boundaries

flameox runs on your machine and does not upload code or captures. It does not
monitor production, provide accounts or synchronization, modify source code,
install system tools, or delete artifacts automatically.

Agents can run only the named workloads you approve. MCP does not provide
arbitrary shell or SQL access, change approvals, delete evidence, return raw
artifact bytes, or launch native viewers. When containment is unavailable,
flameox reports the workload as `uncontained`.

## Setup and installation details

Run setup again to connect or disconnect clients, verify that connected clients
launch the active runtime, update to the npm package's matching version, or roll
back to a previously installed version. `npx flameox@latest setup` resolves the
newest setup release. Automation can select clients and inspect the plan
explicitly:

```console
npx flameox setup --codex --claude --yes
npx flameox setup --all --dry-run --json
npx flameox setup --verify --yes --json
```

The npm package installs the matching `flameox` Python release. MCP clients then
launch that installed runtime directly; they do not call `npx`, `uvx`, or a
network-dependent installer at startup. Setup does not initialize a project or
create `.diagnostics/`.

Optional Python extras are independent:

- `python`: pyperf capture and import
- `cpu`: py-spy capture
- `trace`: Perfetto Python API; a local Trace Processor binary must also be
  configured
- `execution`: coverage.py
- `memory`: Memray
- `torch`: PyTorch capture
- `all`: all runtime integrations

## Local data model

Initialize a project-local workspace:

```console
uv run flameox init .
uv run flameox status
```

`.diagnostics/` stores original artifacts, run and investigation records, Parquet
evidence, and a rebuildable DuckDB catalog. Parquet files and generation
manifests are the source of truth; if you delete `catalog.duckdb`,
`flameox catalog rebuild` reconstructs it.

Identical artifacts are stored once, but flameox retains the source, environment,
workload, and measurement details for every run. Hypotheses, trials, comparisons,
and findings remain separate so observations do not blur into conclusions.

## Named workloads and capture

Repeatable commands live in `flameox.toml`. Templates accept declared scalar
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

Inspect and approve the exact workload definition before exposing it through MCP:

```console
uv run flameox workload show scan --json
uv run flameox workload approve scan
uv run flameox capture plan pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
uv run flameox capture run pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
```

Editing a command, environment, parameter domain, timeout, working directory,
or oracle invalidates that approval. Execution uses argument arrays instead of
shell strings, bounds command output, and cleans up after timeouts or
cancellation. Linux users can configure bubblewrap containment; otherwise, the
result is reported as `uncontained`.

## Investigations and experiments

Create an investigation and optionally attach a falsifiable hypothesis before
running a predeclared experiment:

```console
uv run flameox investigations create \
  '{"question":"Does the candidate remove reverse-scan overhead?"}' --json
uv run flameox hypotheses record @hypothesis.json --json
uv run flameox experiment plan scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
uv run flameox experiment run scan_comparison \
  --investigation <investigation-id> --adapter pyperf --json
```

Before collecting data, flameox saves the declared protocol. It randomizes
treatment order within complete blocks and records every attempted trial,
including cancellations and failures. The automatic paired comparison runs only
when the trial blocks are complete and the measurements, source, environment,
and output validation are compatible. Failed trials remain in the evidence
instead of disappearing from the denominator.

Useful read-only analyses include:

```console
uv run flameox analyze hotspots <run-or-artifact>
uv run flameox analyze scaling <experiment-id>
uv run flameox analyze compare @comparison-request.json
uv run flameox analyze memory <run-or-artifact>
uv run flameox analyze execution <run-or-artifact>
uv run flameox analyze pytorch <run-or-artifact>
uv run flameox analyze failures
```

These commands do not modify the workspace. Record a result when you want to
preserve it with the runs that produced it:

```console
uv run flameox analyze record \
  '{"recipe":"memory","input_id":"<run-id>"}'
uv run flameox analyze record-comparison @comparison-request.json
```

From a hotspot, inspect its callers, callees, representative stacks, or
surrounding trace window:

```console
uv run flameox stacks callers <run-or-artifact> <frame-id> [--cursor CURSOR]
uv run flameox stacks callees <run-or-artifact> <frame-id>
uv run flameox stacks examples <run-or-artifact> <frame-id>
uv run flameox trace window <artifact-id> --start 0 --end 1000000 [--cursor CURSOR]
uv run flameox open <artifact-id>
```

`flameox open` only prints a native viewer plan. Pass `--launch` separately to
open the viewer; it cannot be combined with `--json`.

## CLI and MCP

Start the stdio server with a fixed project root:

```console
uv run flameox mcp serve --project-root .
```

Through MCP, agents can plan approved captures and experiments, query evidence,
preview analyses, and record results. Capture and experiment plans are
short-lived and single-use; restarting the server invalidates them.

Inspect the protocol surface with a real stdio client:

```console
uv run flameox mcp inspect --project-root . --json
```

## Integrity and recovery

```console
uv run flameox validate
uv run flameox validate --full
uv run flameox catalog validate
uv run flameox catalog rebuild
uv run flameox catalog compact
uv run flameox recover
uv run flameox gc
uv run flameox gc --apply
```

Full validation hashes native artifacts and Parquet files. Recovery closes a run
only after its boot, PID, and process-start lease has disappeared. Garbage
collection is a dry run by default; `--apply` moves eligible objects to
recoverable trash instead of deleting them immediately.

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
