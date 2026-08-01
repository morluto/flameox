# Repository Guidelines

## Product Direction

flameox is a permanently local evidence layer for coding agents investigating
performance, memory, execution, concurrency, and reliability. It gives an agent
a reproducible path from a runtime symptom to a conclusion that another person
or agent can inspect and try to disprove.

Existing profilers, benchmark tools, debuggers, and trace processors measure
runtime behavior. flameox coordinates those tools, preserves their native
artifacts and provenance, extracts bounded evidence, and compares runs and
experiments. The agent forms hypotheses, chooses discriminating experiments,
and explains the conclusion.

A typical investigation moves through:

```text
symptom → capture or import → bounded evidence → hypothesis
        → discriminating experiment → supported, refuted, or inconclusive finding
```

flameox is not a profiler, a generic bug finder, a hosted observability service,
an unrestricted command or SQL gateway, or an arbitrary source-code modification
system. Agents may create and update `flameox.toml` through the structured
workload configuration tool.
A feature belongs when it improves trustworthy collection, evidence
preservation, cross-run analysis, experimental validity, or bounded agent
drill-down without replacing an upstream tool.

When changing the product:

- preserve native artifacts, provenance, failed attempts, and experimental
  structure;
- distinguish observed, derived, and inferred claims;
- prefer bounded task-shaped operations over arbitrary commands or SQL;
- keep DuckDB rebuildable and Parquet/manifests authoritative;
- keep CLI and MCP as thin transports over the same application services;
- let agents configure validated named workloads and proceed directly to planning;
- expose coverage, limitations, compatibility, and containment truthfully;
- optimize for investigation leverage, not integration count.

Profiles guide discovery but do not prove causality, semantic correctness, or
performance improvement. Confirmatory claims require representative workloads,
declared metrics and estimands, compatible identities, preserved samples, and
a semantic oracle.

A passing test suite does not prove the documented behavior complete. Before
calling work complete, identify the relevant behavioral contracts and state any
remaining proof gaps.

## Project Structure & Module Organization

flameox is a Python 3.12+ package using a `src/` layout. Production code lives in
`src/flameox/`: domain types and errors are in `domain/`, orchestration belongs in
`application/`, persistence is in `storage/`, profiler integrations are in
`adapters/`, and CLI/MCP entry points are in `cli.py` and `mcp/`. Tests mirror
these boundaries under `tests/`, with additional `golden/`, `performance/`, and
`evidence/` suites.

Read the relevant contract before changing product behavior:

- `docs/architecture.md` for process, package, dependency, and platform rules;
- `docs/storage-and-evidence.md` for storage, provenance, publication, and
  schema rules;
- `docs/investigations.md` for experiments, recipes, statistics, and evidence
  quality;
- `docs/adapters.md` for integration, compatibility, probing, and adapter policy
  behavior;
- `docs/runtime-safety.md` for concurrency, recovery, retention, integrity,
  security, privacy, and observability;
- `docs/interfaces.md` for CLI and MCP behavior and trust boundaries;

## Build, Test, and Development Commands

Use `uv` and the committed `uv.lock`:

```console
uv sync --extra dev --extra python --extra execution --extra memory --extra trace --extra cpu
uv run flameox --help
uv run pytest -q
uv run ruff check src tests
uv run mypy src tests
```

The first command installs development tools and supported lightweight
integrations. Run a focused test while iterating, for example
`uv run pytest tests/storage/test_workspace.py -q`. Marked performance checks
can be selected with `uv run pytest -m performance`.

## Coding Style & Naming Conventions

Use four-space indentation, complete type annotations, and Python 3.12 syntax.
Ruff enforces a 100-character line limit, import ordering, modernization, and
common bug patterns; mypy runs in strict mode. Keep modules and functions
`snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Follow
existing architectural boundaries: keep domain models independent, put use-case
coordination in application services, and isolate external formats behind
adapters.

## Testing Guidelines

Pytest is configured with strict markers and configuration. Name files
`test_<area>.py` and tests `test_<observable_behavior>`. Add regression tests
near the affected component and prefer public behavior over assertions about
private implementation. Use Hypothesis where invariants or input ranges matter.
Coverage tracks branches in `flameox`; no fixed percentage is declared, but new
behavior should include meaningful success and failure-path coverage.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as
`feat(storage): add immutable evidence catalog`, `test: ...`, and `docs: ...`.
Keep commits focused and use an optional scope when it clarifies ownership.
Pull requests should explain the problem and chosen approach, link relevant
issues, list commands actually run, and call out compatibility or safety
implications. Include CLI output or protocol examples for user-visible changes;
screenshots are only useful for changes with a visual surface.
