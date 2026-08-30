# Testing

Pytest markers and test locations are the test contract. There is no separate
ownership database or affected-test planner. Pull requests and merge groups run
the complete deterministic suite; scheduled and manual workflows own optional
providers and performance evidence.

## Commands

Install the development environment and run the ordinary suite:

```console
uv sync --extra dev
uv run pytest -q
uv run pytest -o addopts='' -ra -q -m process
```

The configured default excludes process, optional-provider, and performance
tests for a fast local loop. CI runs all non-optional, non-performance tests,
parallelizing tests not marked `serial` and then running the serial set in one
process.

Static validation is direct:

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run lint-imports
uv run vulture src/flameox --min-confidence 80
uv run deptry src --optional-dependencies-dev-groups dev,test
uv run pip-audit
```

The pull-request, merge-group, and main CI lanes own static validation and the
complete deterministic Python and npm suites. The release workflow does not
repeat those suites after a release commit has merged. It owns the distinct
publication evidence instead: synchronized tag identity, registry metadata,
Python and npm artifact construction, installed-wheel behavior, archive
identity, and post-publication resolution.

## Optional and performance evidence

Install the relevant extra and select its marker explicitly. For example:

```console
uv sync --extra dev --extra trace
uv run pytest -o addopts='' -ra -q -m 'optional and requires_perfetto'
```

The same pattern applies to `requires_coverage`, `requires_memray`,
`requires_pyspy`, and `requires_torch`. GPU, system-tool, and platform markers
remain explicit and require a qualified host. An unavailable prerequisite is a
classified skip, not evidence that the provider works.

Performance tests require an intentional opt-in:

```console
FLAMEOX_RUN_PERFORMANCE=1 uv run pytest -o addopts='' -ra -q -m performance
```

## Test design

- Put tests beside their semantic owner and declare requirements with registered
  module- or test-level markers.
- Keep mutable workspaces, processes, and handles function-scoped.
- Assert observable behavior or stable artifacts, not private helper text.
- Process tests prove cleanup and terminal state.
- Storage tests prove transactions, conflicts, and immutable revision history.
- Analysis tests pin one snapshot before lookup.
- Adapter tests preserve native formats and prove the declared reader contract.
- Security tests include hostile paths, bounds, identities, and cancellation.

A passing suite proves only the exercised environments. Report missing provider,
platform, race, crash, and performance evidence explicitly.
