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

Tests for Rich or Typer human-readable output must remove ANSI styling and normalize wrapping
whitespace before asserting a multi-token message. Prefer structured JSON assertions when that is
the supported contract. Recovery commands that must remain readable should also be exercised at a
narrow terminal width matching CI.

## Cross-boundary regression evidence

Test interacting behaviors, not only each component in isolation. The
[agent contract invariants](agent-contract-invariants.md) define the intended outcomes:

| Combination | Required observable proof |
| --- | --- |
| Failed capture and response truncation | Failure classification survives removal or reordering of diagnostic records. |
| Preservation before pagination | The returned source and continuation still retrieve the next page, or an explicit replacement does. |
| Resource redaction and artifact selection | An agent can select a particular artifact using only the public response, without private paths. |
| Missing-provider recovery and an existing provider set | Following the recovery respects the declared selection semantics and identifies any replacement. |
| Equivalent input formatting and parsing | Valid whitespace variations preserve observations and completeness. |
| Point estimate and uncertainty | Descriptive labels remain distinct from confidence-qualified conclusions. |

Use public serialized producer outputs as consumer inputs. Include the neighboring supported case
and the failure-triggering interaction. A passing helper test or source inspection can establish a
local mechanism, but must not be reported as an end-to-end MCP reproduction. The linked issues
identify coverage still needed; this table does not claim those tests already exist.

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
