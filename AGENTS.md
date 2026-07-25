# Repository Guidelines

## Product Direction

Flamo is a permanently local runtime-evidence system for coding agents. It
shortens the path from an unexplained performance, memory, concurrency, or
reliability symptom to a reproducible, falsifiable conclusion.

Flamo coordinates existing profilers, benchmark tools, debuggers, and trace
processors. Do not reimplement their collectors, formats, statistics, or
viewers when maintained public APIs exist.

When changing the product:

- preserve native artifacts, provenance, failed attempts, and experimental
  structure;
- distinguish observed, derived, and inferred claims;
- prefer bounded task-shaped operations over arbitrary commands or SQL;
- keep DuckDB rebuildable and Parquet/manifests authoritative;
- keep CLI and MCP as thin transports over the same application services;
- expose coverage, limitations, compatibility, and containment truthfully;
- optimize for investigation leverage, not integration count.

Profiles guide discovery but do not prove causality, semantic correctness, or
performance improvement. Confirmatory claims require representative workloads,
declared metrics and estimands, compatible identities, preserved samples, and
a semantic oracle.

A passing test suite does not prove the entire specification complete. Before
calling work complete, identify the relevant `SPEC.md` acceptance criteria and
state any remaining proof gaps.

## Project Structure & Module Organization

Flamo is a Python 3.12+ package using a `src/` layout. Production code lives in
`src/flamo/`: domain types and errors are in `domain/`, orchestration belongs in
`application/`, persistence is in `storage/`, profiler integrations are in
`adapters/`, and CLI/MCP entry points are in `cli.py` and `mcp/`. Tests mirror
these boundaries under `tests/`, with additional `golden/`, `performance/`, and
`evidence/` suites. Read `SPEC.md` before changing safety, storage, execution, or
protocol behavior; it defines the product contract.

## Build, Test, and Development Commands

Use `uv` and the committed `uv.lock`:

```console
uv sync --extra dev --extra python --extra execution --extra memory --extra trace --extra cpu
uv run flamo --help
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
Coverage tracks branches in `flamo`; no fixed percentage is declared, but new
behavior should include meaningful success and failure-path coverage.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as
`feat(storage): add immutable evidence catalog`, `test: ...`, and `docs: ...`.
Keep commits focused and use an optional scope when it clarifies ownership.
Pull requests should explain the problem and chosen approach, link relevant
issues, list commands actually run, and call out compatibility or safety
implications. Include CLI output or protocol examples for user-visible changes;
screenshots are only useful for changes with a visual surface.
