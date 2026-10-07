# Testing

Prioritize tests in this order:

1. End-to-end workflows using the installed CLI or real MCP stdio transport,
   real subprocesses, and native evidence, with no substituted services.
2. Integration tests that cross request, provider, worker, filesystem, or
   repository boundaries and assert the resulting evidence or failure.
3. Golden examples with explicit expected projections for native formats,
   canonical serialization, and edge cases. Small constructed examples prove
   normalization; they do not prove compatibility with every upstream producer release.

Avoid tests that mirror methods, count internal calls, restate library validators,
or replace the collector and decoder with fabricated successes. Add a focused
test only for a behavioral contract that a retained workflow cannot establish.
Fault injection remains useful for publication failures, cancellation races,
resource observations, and privacy failures that normal execution cannot reliably
trigger. Test the observable outcome at those boundaries.

There is no unit-test marker. The retained filesystem, executable binding,
repository, and public runtime checks exercise real boundary behavior. Fixed
format and serialization examples use the golden marker. Classification alone
does not justify retaining a test: remove duplicated proof and assertions that
only restate an implementation.

## Baseline

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run lint-imports
uv run pytest -q
```

The default suite includes process tests and the real CLI/MCP workflows. Optional
providers and representative scale workloads remain explicit selections:

```console
uv run pytest -m e2e -q
uv run pytest -o addopts='' -m golden -q
uv run pytest -o addopts='' -m performance --durations=0
```

Choose the semantic owner while iterating:

| Owner | Focused command, including its process cases |
| --- | --- |
| Runtime coordination, limits, and capture | `uv run pytest -o addopts='' tests/test_runtime*.py tests/test_capture*.py tests/test_workload_budgets.py -q` |
| Immutable publication and evidence | `uv run pytest -o addopts='' tests/test_repository.py tests/test_evidence*.py tests/storage -q` |
| MCP schemas, transport, and evidence handoffs | `uv run pytest -o addopts='' tests/mcp -q` |
| CLI and setup | `uv run pytest -o addopts='' tests/test_cli.py tests/test_setup.py -q` |
| Execution and cancellation | `uv run pytest -o addopts='' tests/execution tests/test_worker_lifecycle.py -q` |
| Installed CLI workflows | `uv run pytest tests/e2e -q` |
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

Contract tests assert exactly seven MCP tools, one resource template, no concrete resource list,
compact input envelopes, registry-backed capability/provider discovery, valid examples, output schemas,
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

The scale workload publishes 1,000 real immutable manifests, closes the runtime,
and queries every page through a new runtime. It verifies complete identities and
a stable inventory digest without replacing filesystem traversal or bundle
validation. Performance claims must report the corpus, command, host-relevant
limits, and result. Use `uv run pytest -o addopts='' -m performance --durations=0`
to record timings; compare the same workload and environment before making a
speed or complexity claim. There is no hardware-independent timing threshold.

The XML workload compares Python allocation peaks for 50,000 and 200,000 sibling
elements in real xctrace table-of-contents XML. It verifies that completed siblings
do not accumulate in the parser tree; it does not establish an RSS cap for arbitrary
deep nesting or individual large XML values.

## Proof gaps

A passing default suite does not prove every provider or platform. Report
missing hardware, permissions, vendor tools, cross-platform execution, crash
injection boundaries, or scale runs explicitly. Do not replace behavioral proof
with source-text assertions about private helpers.

The test audit removed fabricated vendor-tool capture and conversion cases.
Real Nsight Systems, ROCprof, xctrace, Nsight Compute, Compute Sanitizer, NVBench,
and Node profile capture/export still require the relevant tools, hardware, and
permissions; native-format projection cases
do not establish those lifecycles. Perfetto's golden integration requires a local
Trace Processor on PATH or `FLAMEOX_TRACE_PROCESSOR`, as well as the trace extra.
The AIPerf comparison and live Torch cases require their respective extras and,
for CUDA measurements, a usable GPU.

Projection-cache byte/entry ceilings and implementation-identity invalidation
need public-workflow proof after their private mock/call-count tests were removed.
Canonical manifest identities are checked for integrity and replay, but there is
no fixed expected-digest golden. MCP managed preparation retains controlled
installer fixtures; a real package-install/reconnect workflow remains unproved.
Readiness derivation for incompatible installed dependency sets, exhaustive
discovery-example validation, Torch profiler exit-failure precedence, and a
successful npm-to-uvx handoff also need boundary proof after their mock-only checks
were removed. The real experiment workflow checks classification against its
observed estimate; it does not establish the exact zero-estimate/wide-interval
case. Stalled subprocess acquisition, worker request-encoding failure cleanup,
catalog responsiveness during blocked analysis, and detailed startup-limit
lowering also lack retained workflow proof.

Two source findings remain outside the local fixes made during the audit:

- Repository metadata and manifests are schema-validated, but their JSON reads
  currently have no independent size ceiling. A size policy needs a documented
  compatibility contract before oversized documents can be rejected safely.
- The Memray worker generates and validates measurement, call-edge, and stack
  tables that the provider does not expose or preserve. The native attribution
  tests establish the published frame projection; they do not justify that extra
  normalization work. Simplifying the worker protocol requires a separate
  capability and evidence contract decision.
