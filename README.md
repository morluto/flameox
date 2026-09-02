<h1 align="center">flameox</h1>

<p align="center"><strong>Bounded local runtime evidence for coding agents.</strong></p>

<!-- mcp-name: io.github.morluto/flameox -->

Flameox coordinates profilers, benchmark tools, trace processors, and direct
local targets. It gives an agent a short path from an explicit native artifact
or live command to bounded evidence, while keeping preservation optional.

Version 0.2 is a clean break. There is no workspace to initialize, no
`flameox.toml`, no SQLite control plane, no durable job to poll, and no parent
directory discovery. Existing artifacts remain usable by passing their exact
paths and formats to `analyze`; old `.diagnostics` state is not migrated.

## Quick start

```console
uv sync --extra dev --extra memory --extra trace --extra cpu
uv run flameox mcp inspect
uv run flameox analyze artifact.preview /absolute/path/to/artifact.json
uv run flameox capture --provider direct --cwd "$PWD" -- python benchmark.py
```

The MCP server has no workspace or project binding:

```console
uv run flameox mcp serve
```

`flameox setup` prints a version-bound stdio client configuration using Python
3.12 and the exact running Flameox release. It does not change an MCP client
registration; apply the printed command through the client's supported MCP
management interface. Explicit
`--provider` selections prepare the exact version-pinned `uvx` environment in
the printed launcher by resolving it once into uvx's cache; they do not create a
persistent global `uv tool` installation. Each invocation declares the complete
managed provider set for that launcher rather than adding to remembered state.
Use `--timeout-seconds` for a slow cold resolution. System and vendor tools are
diagnosed with external install guidance. Setup never initializes or mutates a project.

## Authority model

```text
explicit artifact paths / typed direct target
                    │
                    ▼
         bounded process-lifespan runtime
             │                │
             ▼                ▼
       inline evidence   session scratch/cache
                              │
                       explicit preservation
                              │
                              ▼
             user Flameox data directory
```

Analysis and unpreserved capture make no durable Flameox writes. Capture
artifacts stay in bounded session scratch until preservation, least-recently-used eviction, or
server shutdown. An evicted `analysis_id` returns `EXPIRED_SESSION_ANALYSIS`; preserve conclusions
before relying on them. The first `preserve_evidence` call creates the user-level Flameox data
directory and stores native bytes and a canonical evidence bundle by SHA-256. `FLAMEOX_DATA_DIR`
overrides the platform default for isolation or another storage location. Flameox never edits
project Git files.

The agent owns hypotheses and narrative findings in its own notes. Flameox owns
only observed inputs, effective requests, execution provenance, typed evidence,
coverage, truncation, limitations, and optional immutable preservation.

## MCP interface

The server exposes actual evidence operations for client-side tool search instead of hiding its
capabilities behind `discover`, `inspect`, or generic `analyze(capability_id, arguments)` calls.
There are 24 read-only analysis tools, 17 executing capture tools, and three lifecycle tools. For
example:

```text
analyze_cpu_hotspots       capture_cpu_hotspots
analyze_gpu_launches       capture_gpu_launches
analyze_benchmark_compare  capture_benchmark_summary
analyze_kernel_validation  capture_sanitizer_failures
prepare_providers          preserve_evidence          query_evidence
```

Each tool advertises its capability-specific options and compatible providers in its input schema.
Analysis and capture have separate names and annotations because reading an artifact and executing a
target are materially different effects. Tool search happens in the MCP client; Flameox does not
require an additional catalog-search call.

It exposes one resource template, `flameox://evidence/{evidence_id}`, for the
digest-bound, redacted projection of the canonical immutable manifest. Full
argv, environment values, working directories, and host paths remain available
only through explicit local manifest inspection. Native artifact bytes are
deliberately not available as MCP resources.

Direct capture accepts an argv array, an explicit absolute cwd, bounded environment overrides, a
typed compatible-provider variant, capability-specific options, an explicit single/experiment
choice, and limits as top-level tool arguments. There is no generic request or arguments envelope.
Shell strings are never accepted. Work remains owned by the live MCP request, so SDK progress and
cancellation apply directly; there are no detached or restart-surviving tasks.

Managed external collectors such as py-spy execute from Flameox's uvx
environment. In-process collectors such as coverage.py and Memray are verified
in, and run with, the workload's declared Python interpreter. Flameox does not
substitute one Python runtime for the other. When a capture reports a missing managed provider,
`prepare_providers` prepares its version-pinned uvx environment and returns that same launcher for
reconnection. The agent supplies the complete provider list it wants in that launcher; Flameox does
not merge it with prior calls. Preparation does not modify the running MCP process. When the client
must reconnect, the result returns a typed `next_action` with `kind: "reconnect_mcp"`, an agent-facing
message, and the launcher to use. The managed provider IDs are `aiperf`, `memray`, `otlp`, `perfetto`,
`py-spy`, and `torch`. Host tools, drivers, and permissions are never installed or changed; the same
result reports their setup guidance.

Comparison is intentionally a two-stage workflow. Flameox captures representative baseline and
candidate summaries separately, optionally preserves them, and then passes both artifacts to an
`analyze_*_compare` tool. There are no `capture_*_compare` tools: experiment capture measures cases
and reports an effect, but it is not a substitute for comparing explicit native artifacts.

## Evidence quality

An investigation still follows:

```text
symptom → capture or explicit artifact → bounded evidence → hypothesis
        → discriminating experiment → supported, refuted, or inconclusive finding
```

A profile supports exploration, not causality. Confirmatory claims require a
representative target, declared metric and estimand, compatible identities,
preserved samples, a practical threshold, and an appropriate semantic oracle.

See [architecture](docs/architecture.md), [storage and evidence](docs/storage-and-evidence.md),
[interfaces](docs/interfaces.md), [runtime safety](docs/runtime-safety.md), and
[investigations](docs/investigations.md) for the contracts.

## Development

Flameox requires Python 3.12 or newer and uses the committed `uv.lock`.

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run lint-imports
uv run pytest -q
```

The project is licensed under the MIT License.
