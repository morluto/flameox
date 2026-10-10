# Working on Flameox

Flameox is a local runtime-evidence layer for coding agents. It coordinates maintained
measurement tools, extracts bounded evidence, and optionally preserves native artifacts and
provenance. The agent owns hypotheses and conclusions. There is no workspace initialization
or named workload configuration: callers supply exact artifact paths or typed capture targets.

## Constraints that affect implementation

- Keep CLI and MCP thin over `AnalysisRuntime` and the operation registry. Public fields,
  bounds, and descriptions belong in canonical runtime models; transport schemas derive from
  them. Expose named task-shaped tools, without search/execute gateways or aliases.
- Preserve native bytes, provenance, failed attempts, and experiment structure. Keep observed,
  derived, and inferred claims distinct; profiles alone do not prove improvement or correctness.
- Keep work request-owned and scratch ephemeral. Durable evidence uses content-addressed
  artifacts and immutable authoritative manifests. DuckDB is ephemeral only; Flameox never
  creates or imports SQLite, though upstream packages may read their native formats internally.
- Capture takes validated argv, cwd, environment, provider fields, and request-lowerable limits.
  It runs trusted local programs; process bounds are not a sandbox.
- Continuations read saved inputs without rerunning workloads. Session handles and durable
  evidence have different lifetimes; preserve exact continuation and recovery handoffs.

## Find the relevant contract

Consult the owning contract when changing that behavior; routine edits do not require reading
all docs. Module ownership is in [architecture](docs/architecture.md#package-boundaries).

| Change | Contract |
| --- | --- |
| Product scope, process or package boundaries | [Architecture](docs/architecture.md) |
| Preservation, provenance, persisted schemas | [Storage and evidence](docs/storage-and-evidence.md) |
| Experiments, metrics, comparisons | [Investigations](docs/investigations.md) |
| Providers, native formats, compatibility | [Adapters](docs/adapters.md) |
| Cancellation, resources, integrity, privacy | [Runtime safety](docs/runtime-safety.md) |
| CLI, MCP, setup, update, continuation | [Interfaces](docs/interfaces.md) |
| Test selection and proof gaps | [Testing](docs/testing.md) |

## Development

Use Python 3.12+, `uv`, and the committed `uv.lock`. Start with `uv sync --extra dev`;
add provider extras only when needed. Ruff and strict mypy own style and typing checks.
Commands and contribution conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

Match validation to the changed behavior. For tool changes, exercise affected installed CLI and
real MCP stdio workflows, including relevant failure and continuation paths, and check schemas
against accepted inputs and emitted results. Documentation-only changes need link and contract
checks, not collector or performance runs. Record material proof gaps in
[testing](docs/testing.md#evidence-and-limits).
