# Testing

Tests prove the process-lifespan runtime, bounded execution, provider adapters, and
optional immutable repository independently.

## Baseline

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run lint-imports
uv run pytest -q
```

Choose the semantic owner while iterating:

| Owner | Focused command, including its process cases |
| --- | --- |
| Runtime coordination, limits, and capture | `uv run pytest -o addopts='' tests/test_runtime*.py tests/test_capture*.py tests/test_workload_budgets.py -q` |
| Immutable publication and evidence | `uv run pytest -o addopts='' tests/test_repository.py tests/test_evidence*.py tests/storage -q` |
| MCP schemas, transport, and evidence handoffs | `uv run pytest -o addopts='' tests/mcp -q` |
| CLI and setup | `uv run pytest -o addopts='' tests/test_cli.py tests/test_setup.py -q` |
| Execution and cancellation | `uv run pytest -o addopts='' tests/execution tests/test_worker_lifecycle.py -q` |
| Native formats and provider integration | `uv run pytest -o addopts='' tests/providers tests/adapters -q` |

CI discovers the entire `tests/` tree and selects by markers, so these owners
share the deterministic and process jobs without a path registry. Match those
jobs locally with `-o addopts='' -m 'not optional and not performance and not process'`
and `-o addopts='' -m 'not optional and not performance and process'`. Both jobs
are required on pull requests. Optional provider jobs select `requires_memray`
or `requires_torch`; the scheduled scale job selects `performance`.

Provider readiness lives in `tests/support/providers.py`. Process readiness and
liveness probes live in `tests/support/processes.py` and are imported explicitly
by their consumers. Keep mutable runtimes, stores, and clients local to each
test; transport tests must not import fixtures from provider or runtime test modules.

## Required behavioral proof

Contract tests assert exactly six MCP tools, one resource template, no concrete resource list,
capability-discriminated input schemas, compatible-provider discriminators, output schemas,
truthful annotations, and direct structured success content without a universal wrapper.

Runtime tests cover bounded streaming analysis, digest-bound continuation,
provider states, typed capture, progress, cancellation and descendant cleanup,
partial/failed evidence, scratch ceilings, and absence of durable writes without
preservation.

Repository tests cover lazy creation, Git exclusion, input mutation, artifact
reuse, concurrent identical/distinct publication, every publication boundary,
corrupt or incomplete bundles, unsupported versions, abandoned staging cleanup,
stable queries, resource errors, and restart semantics.

Provider tests use explicit native fixture paths without a repository. Optional
tests must state the actual provider/version and skip rather than claim evidence
when the host capability is absent.

Tests for Rich or Typer human-readable output must remove ANSI styling and normalize wrapping
whitespace before asserting a multi-token message. Prefer structured JSON assertions when that is
the supported contract. Recovery commands that must remain readable should also be exercised at a
narrow terminal width matching CI.

Assert semantic schema fields rather than generated definition names, display
titles, or ordering of `required` and `enum` arrays. For compact output, prove
which payloads are omitted and that the full evidence remains recoverable;
an arbitrary serialized byte count does not establish either property.

Process tests should signal readiness before cancellation and check cleanup at
the return boundary. Use explicit events to coordinate blocked work, with a
generous watchdog to catch hangs. A timeout may interrupt a stream before all
intended bytes are collected: check its exact retained prefix, byte counts, and
incomplete status after reopening the evidence.

## Performance evidence

Representative performance cases include 1,000-member identity accumulation,
10,000 immutable manifests, and repeated Nsight continuation reads through the
session conversion cache. Performance claims must report the corpus, command,
host-relevant limits, and result rather than relying on a unit-test timeout.
The scale cases assert complete member identities, correct query pages, and
export reuse. Use `uv run pytest -o addopts='' -m performance --durations=0`
to record timings; compare the same workload and environment before making a
speed or complexity claim. These tests have no hardware-independent five-second
acceptance threshold.

## Proof gaps

A passing default suite does not prove every provider or platform. Report
missing hardware, permissions, vendor tools, cross-platform execution, crash
injection boundaries, or scale runs explicitly. Do not replace behavioral proof
with source-text assertions about private helpers.
