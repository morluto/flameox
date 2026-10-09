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

Flameox coordinates profilers, benchmark tools, trace processors, and direct
local targets. It gives an agent a short path from an explicit native artifact
or live command to bounded evidence, while keeping preservation optional.

There is no workspace to initialize, `flameox.toml`, SQLite control plane,
durable job to poll, or parent-directory discovery. Analyze existing artifacts
by passing their exact paths and formats.

## Quick start

```console
uv sync --extra dev --extra memory --extra trace --extra cpu
uv run flameox mcp inspect
uv run flameox mcp inspect --capability cpu.hotspots
uv run flameox analyze artifact.preview /absolute/path/to/artifact.json
uv run flameox capture --provider direct --cwd "$PWD" -- python benchmark.py
```

The MCP server has no workspace or project binding:

```console
uv run flameox mcp serve
```

Run the short global setup wizard through npm:

```console
npx flameox@latest setup
```

Setup detects Claude Code, Cursor, OpenCode, Codex, Gemini CLI, and Google Antigravity, then asks
which clients should use Flameox. It preserves unrelated client configuration and writes a
Python 3.12 `uvx` launcher pinned to the exact Flameox release that ran setup. Restart or reconnect
changed clients afterward. For automation, pass `--client codex --yes`, repeat `--client`, or use
`--all --yes`; `--dry-run` reports the same global paths without writing them. Detection is never
automation consent.

Explicit `--provider` selections prepare the exact version-pinned `uvx` environment in the saved
launcher by resolving it once into uvx's cache; they do not create a persistent global `uv tool`
installation. Each invocation declares the complete managed provider set for that launcher rather
than adding to remembered state. Use `--timeout-seconds` for a slow cold resolution. System and
vendor tools are diagnosed with external install guidance. Setup never initializes or mutates a
project.

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

The console-retention default is bounded diagnostics in memory, with explicit
omission counts. Keep native artifacts when needed; retain full console output on
disk only when it is the evidence, an oracle needs it, or the caller requests it.
Preservation alone does not request full logs. See
[console retention](docs/storage-and-evidence.md#console-retention-and-disk-backing).

Workload time and RSS budgets are optional: use `target.budget` in MCP or
`--workload-budget` in CLI capture. They do not inherit analysis-worker limits;
cancellation and storage protection remain active when no workload budget is set.

The agent owns hypotheses and narrative findings in its own notes. Flameox owns
only observed inputs, effective requests, execution provenance, typed evidence,
coverage, truncation, limitations, and optional immutable preservation.

## MCP interface

The MCP catalog has 26 named analysis tools, 20 named capture tools, and four lifecycle tools.
Analysis and capture remain separate for accurate effect annotations. Tool names identify the
evidence question, and each input schema exposes that capability's typed fields directly:

```text
summarize_trace      rank_cpu_hotspots      preview_artifact
... 23 other analysis tools ...
capture_trace_summary capture_cpu_hotspots  capture_artifact_preview
... 17 other capture tools ...
prepare_providers    preserve_evidence
rescue_evidence      query_evidence
```

`flameox mcp inspect` lists tools compactly; `--capability CAPABILITY_ID` shows direct examples and
capability/provider field schemas, and `--tool TOOL_NAME` shows one tool's full MCP schema. No
discovery call is required during MCP use. The complete named catalog is documented in
[interfaces](docs/interfaces.md).

For example, a bounded artifact preview passes its source and capability-specific setting directly:

```json
{
  "sources": [{"kind": "path", "path": "/absolute/path/to/output.log"}],
  "text_fragment_chars": 1024,
  "page_size": 100
}
```

Capture inputs use `target`, a typed `provider` object, and capability fields at the top level.
Provider settings are siblings of `kind`; for example:

```json
{
  "target": {
    "argv": ["python", "benchmark.py"],
    "cwd": "/absolute/path/to/project"
  },
  "provider": {"kind": "direct"}
}
```

When a result contains `next_page`, call its named analysis tool with its arguments unchanged.
Capture continuations read existing captured artifacts and never execute the target again.

It exposes one resource template, `flameox://evidence/{evidence_id}`, for the
digest-bound, redacted projection of the canonical immutable manifest. Full
argv, environment values, working directories, and host paths remain available
only through explicit local manifest inspection. Native artifact bytes are
deliberately not available as MCP resources.

Direct capture accepts an argv array, an absolute cwd, bounded environment overrides, a compatible
typed provider, capability-specific options, and an optional experiment design. Shell strings are
never accepted. `page_size` is the only caller-facing response bound; process, memory, traversal,
and storage ceilings remain server policy. Work belongs to the live MCP request, so progress and
cancellation apply directly and no detached task survives a restart.

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
candidate summaries separately, optionally preserves them, then passes both artifacts to
`compare_benchmarks`, `compare_inference`, or `compare_kernel_validation`. Comparison tools read
explicit artifact identities; an experiment capture measures cases and reports an effect, which is
different evidence.

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
