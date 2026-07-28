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
  <img src="https://img.shields.io/badge/Runtime-Permanently_Local-F97316?style=flat" alt="Permanently local runtime">
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

flameox is a permanently local CLI and Model Context Protocol server. It connects
coding agents to maintained tools such as pyperf, py-spy, Perfetto Trace
Processor, coverage.py, Memray, and torch.profiler. It preserves their native
artifacts and the provenance needed to check an agent's conclusion later.

It is not another profiler or an automatic bug finder. It gives agents a
consistent workflow for collecting approved workloads, preserving what happened,
comparing compatible runs, and keeping observations separate from inferences.

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

Profiles identify candidates; they do not prove causality or correctness. A
confirmatory comparison needs a representative workload, a declared metric,
compatible source and environment identities, preserved samples, and a semantic
oracle.

## How it works

1. **Declare.** You approve a repeatable workload. Agents never receive access to
   arbitrary shell commands or SQL.
2. **Capture.** A maintained profiler or benchmark tool runs while flameox records
   the exact tool, command, environment, source identity, limits, and outcome.
3. **Preserve.** flameox keeps the native artifact and publishes normalized
   evidence into an immutable local corpus.
4. **Analyze.** The CLI and MCP server expose the same bounded operations for
   hotspots, scaling, memory, execution, failures, and comparisons.
5. **Record.** Findings remain tied to the runs, measurements, validation, and
   analysis that support them. Failed attempts remain visible.

Native artifacts and Parquet evidence are authoritative. DuckDB is a rebuildable
local query layer, and Perfetto Trace Processor handles detailed trace queries.

## Boundaries

flameox investigates performance, memory, execution, concurrency, and
reliability on the local machine. It does not monitor production, upload
evidence, provide accounts or synchronization, modify source code, install
system tools, or delete artifacts automatically.

flameox also does not reimplement profilers, trace databases, native viewers, or
private format decoders. It reports a workload as contained only when an active
backend enforces that boundary, and it never treats statistical
non-significance as proof of equivalence.

## Setup and installation details

Run setup again to connect or disconnect clients, verify the active runtime,
update to the npm package's matching version, or roll back to a previously
installed version. `npx flameox@latest setup` resolves the newest setup release.
Automation can select clients and inspect the plan explicitly:

```console
npx flameox setup --codex --claude --yes
npx flameox setup --all --dry-run --json
npx flameox setup --verify --yes --json
```

The npm package is a bootstrap, not the runtime. It installs the matching
`flameox` Python release and includes the `jsonc-parser` helper used to preserve
comments in OpenCode configuration. MCP clients then launch that installed
runtime directly; they do not call `npx`, `uvx`, or a network-dependent installer
at startup. Setup does not initialize a project or create `.diagnostics/`.

Optional Python extras are independent:

- `python`: pyperf capture and import
- `cpu`: py-spy capture
- `trace`: Perfetto Python API; a local Trace Processor binary must also be
  configured
- `execution`: coverage.py
- `memory`: Memray
- `torch`: PyTorch capture
- `all`: all runtime integrations

flameox pins the official Python MCP SDK to `mcp==2.0.0b2`.

## Local data model

Initialize a project-local workspace:

```console
uv run flameox init .
uv run flameox status
```

`.diagnostics/` contains content-addressed native artifacts, immutable JSON
records, append-only Parquet generations, and a rebuildable DuckDB catalog.
Parquet files and generation manifests are authoritative. If you delete
`catalog.duckdb`, `flameox catalog rebuild` reconstructs it.

SHA-256 deduplication avoids storing the same artifact twice without losing the
context in which it was registered. Each run records its source, environment,
workload, measurements, lifecycle, validation, process evidence, and artifact
roles. Investigation records keep hypotheses, experiments, trials, comparisons,
and findings distinct.

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

Inspect and approve the exact canonical workload before exposing it through
MCP:

```console
uv run flameox workload show scan --json
uv run flameox workload approve scan
uv run flameox capture plan pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
uv run flameox capture run pyperf --workload scan \
  --parameters '{"implementation":"baseline"}' --json
```

Editing a command, environment, parameter domain, timeout, working directory,
or oracle changes the canonical digest and revokes that approval. Execution
uses argument arrays through a single subprocess broker. The broker bounds
output, cleans up after timeouts and cancellation, and can use bubblewrap for
containment on Linux. Perfetto parsing runs in a broker-owned worker, so a long
trace cannot block or outlive the MCP request. An `uncontained` result is never
reported as sandboxed.

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
when the blocks, measurements, source identity, environment, and
cross-treatment validation are compatible. Failed trials remain in the evidence
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

Those commands are deterministic read-only previews. Persist a recipe result
and its typed provenance explicitly:

```console
uv run flameox analyze record \
  '{"recipe":"memory","input_id":"<run-id>"}'
uv run flameox analyze record-comparison @comparison-request.json
```

Hotspots can be followed into normalized trace structure without arbitrary SQL:

```console
uv run flameox stacks callers <run-or-artifact> <frame-id> [--cursor CURSOR]
uv run flameox stacks callees <run-or-artifact> <frame-id>
uv run flameox stacks examples <run-or-artifact> <frame-id>
uv run flameox trace window <artifact-id> --start 0 --end 1000000 [--cursor CURSOR]
uv run flameox open <artifact-id>
```

`flameox open` only prints a native viewer plan. `--launch` is an explicit
consequential action and cannot be combined with `--json`.

## CLI and MCP

Start the permanently local stdio server with a fixed project root:

```console
uv run flameox mcp serve --project-root .
```

The MCP layer calls the same application services as the CLI. Agents can plan
approved captures and experiments, run bounded evidence queries, preview
analyses, and explicitly record results. MCP does not expose shell strings,
arbitrary SQL, raw artifact bytes, approval changes, deletion, or viewer
launching.

Capture and experiment plans use 256-bit, in-memory, short-lived tokens. Each
token is single-use and bound to the workspace and the approved plan inputs.
Restarting the server invalidates it.

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
