# Testing

Tests prove the stateless runtime, bounded execution, provider adapters, and
optional immutable repository independently.

## Baseline

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run lint-imports
uv run pytest -q
```

Use `uv run pytest tests/test_stateless.py -q` while changing the public runtime
or repository. Process tests use `-o addopts='' -m process`; performance tests
use `-o addopts='' -m performance`.

## Required behavioral proof

Contract tests assert exactly 47 MCP tools, one resource template, no concrete resource list,
capability-specific top-level input schemas, compatible-provider discriminators, output schemas,
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

## Performance evidence

Representative performance cases include 1,000-member identity accumulation,
10,000 immutable manifests, and repeated Nsight continuation reads through the
session conversion cache. Performance claims must report the corpus, command,
host-relevant limits, and result rather than relying on a unit-test timeout.

## Proof gaps

A passing default suite does not prove every provider or platform. Report
missing hardware, permissions, vendor tools, cross-platform execution, crash
injection boundaries, or scale runs explicitly. Do not replace behavioral proof
with source-text assertions about private helpers.
