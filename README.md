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
uv run flameox capabilities discover --intent "CPU hotspots"
uv run flameox analyze artifact.preview /absolute/path/to/artifact.json
uv run flameox capture --provider direct -- python benchmark.py
```

The MCP server fixes its project root at startup. Manual launches default to
the startup working directory:

```console
uv run flameox mcp serve --project-root "$PWD"
```

`flameox setup` prints a version-bound stdio client configuration using Python
3.12 and the exact running Flameox release. It does not change an MCP client
registration; apply the printed command through the client's supported MCP
management interface. Explicit
`--provider` selections add the matching Python extras to a persistent uv tool
environment without removing its existing provider extras. System and vendor
tools are diagnosed with external install guidance. Setup never initializes the
project or creates `.flameox`.

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
                    <project>/.flameox
```

Analysis and unpreserved capture make no durable Flameox writes. Capture
artifacts stay in bounded session scratch until preservation or server
shutdown. The first `preserve_evidence` call creates `.flameox`, stores native
bytes and a canonical evidence bundle by SHA-256, and adds `.flameox/` to the
repository-local `.git/info/exclude` when applicable.

The agent owns hypotheses and narrative findings in its own notes. Flameox owns
only observed inputs, effective requests, execution provenance, typed evidence,
coverage, truncation, limitations, and optional immutable preservation.

## MCP interface

The server exposes exactly seven tools:

- `discover_capabilities`
- `inspect_capabilities`
- `analyze`
- `preflight_capture`
- `capture_and_analyze`
- `preserve_evidence`
- `query_evidence`

It exposes one resource template, `flameox://evidence/{evidence_id}`, for the
digest-bound, redacted projection of the canonical immutable manifest. Full
argv, environment values, working directories, and host paths remain available
only through explicit local manifest inspection. Native artifact bytes are
deliberately not available as MCP resources.

Direct capture accepts an argv array, a project-contained cwd, bounded
environment overrides, provider and analysis arguments, and limits. Shell
strings are never accepted. Work remains owned by the live MCP request, so SDK
progress and cancellation apply directly; there are no detached or
restart-surviving tasks.

Managed external collectors such as py-spy execute from Flameox's uv tool
environment. In-process collectors such as coverage.py and Memray are verified
in, and run with, the workload's declared Python interpreter. Flameox does not
substitute one Python runtime for the other.

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
