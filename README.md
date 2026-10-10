<h1 align="center">flameox</h1>

<p align="center">
  <img
    src="docs/assets/flameox-mascot-flamegraph.png"
    width="420"
    alt="flameox mascot: an ox with a flame graph between its horns"
  >
</p>

<p align="center"><strong>Bounded local runtime evidence for coding agents.</strong></p>

<!-- mcp-name: io.github.morluto/flameox -->

Flameox coordinates profilers, benchmark tools, trace processors, and direct local targets. It
turns an explicit artifact or command into bounded evidence while keeping durable preservation
optional. The agent chooses hypotheses and experiments; Flameox records observed inputs,
execution provenance, typed evidence, coverage, and limitations.

There is no workspace to initialize or project configuration to maintain. Pass exact artifact
paths to analysis and an explicit argv plus absolute working directory to capture. Flameox does not
search parent directories, edit project files, or provide a hosted service.

## Quick start

Configure an MCP client with the global setup command:

```console
npx flameox@latest setup
```

Setup detects supported clients and asks which configurations to update. For automation, select
clients explicitly, for example `--client codex --yes` or `--all --yes`; `--dry-run` reports the
paths and actions without writing. Setup preserves unrelated client configuration and does not
change project files. See [the npm package guide](npm/README.md) for details.

For direct CLI use, pass exact artifact paths or an explicit command:

```console
uvx flameox analyze artifact.preview /absolute/path/to/artifact.json
uvx flameox capture --provider direct --cwd "$PWD" -- python benchmark.py
```

To launch the stdio MCP server manually:

```console
uvx flameox mcp serve
```

For local development, clone the repository, install its development dependencies, and inspect the
available tools:

```console
git clone https://github.com/morluto/flameox.git
cd flameox
uv sync --extra dev --extra memory --extra trace --extra cpu
uv run flameox mcp inspect
```

## Evidence and storage

Analysis and unpreserved capture use bounded session scratch. Session `analysis_id` values expire
when evicted or when the server stops. Explicit preservation creates the user-level Flameox data
directory and stores native bytes and a canonical evidence bundle by SHA-256. Set
`FLAMEOX_DATA_DIR` to choose another location. Preservation is optional; it does not imply full
console-output retention.

Flameox distinguishes observed, derived, and inferred claims. Profiles guide exploration but do
not establish causality or performance improvement. Confirmatory claims require representative
workloads, declared metrics and estimands, compatible identities, preserved samples, and a
semantic oracle.

## MCP

The MCP server has no workspace binding. It exposes named analysis and capture tools through the
same capability registry as the CLI. Analysis tools accept explicit sources; capture tools accept
a typed target and provider. Tool schemas expose capability-specific fields directly. Inspect the
catalog with `flameox mcp inspect`; use `--capability CAPABILITY_ID` or `--tool TOOL_NAME` for
focused schemas and examples. The complete contract and current tool catalog are in
[the interface guide](docs/interfaces.md).

Tools return their full bounded results inline, including failures and recovery actions. Preserved
evidence has a durable `evidence_id`; `inspect_evidence` returns its redacted metadata and replay
sources inline.

Capture takes argv, not a shell string. A direct target supplies an absolute working directory and
bounded environment overrides. Results may include `next_page`; replay its named analysis tool
and arguments unchanged. Capture continuations read collected artifacts and do not rerun the
target. MCP work belongs to the live request, so cancellation applies directly and no detached job
survives a restart.

## Documentation

- [Architecture](docs/architecture.md) describes process and package boundaries.
- [Storage and evidence](docs/storage-and-evidence.md) defines preservation and provenance.
- [Investigations](docs/investigations.md) covers experiments, comparison, and evidence quality.
- [Adapters](docs/adapters.md) documents provider compatibility and integration policy.
- [Runtime safety](docs/runtime-safety.md) covers limits, recovery, integrity, and privacy.
- [Interfaces](docs/interfaces.md) defines CLI and MCP contracts.
- [Testing](docs/testing.md) describes test selections and known proof gaps.

## Development

Flameox requires Python 3.12 or newer and uses the committed `uv.lock`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidance.

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run pytest -q
```

Flameox is licensed under the MIT License.
