# Testing

Prefer evidence in this order:

1. End-to-end workflows through the installed CLI or real MCP stdio transport,
   using real subprocesses and native evidence.
2. Integration workflows across runtime, provider, worker, process, filesystem,
   or repository boundaries.
3. Golden examples with explicit expected projections for native formats and
   stable serialization.

Keep a narrower test only when these workflows cannot establish a behavioral
contract. Avoid tests that mirror methods, count internal calls, restate library
validators, or manufacture successful collector and decoder results. Fault
injection is useful when a real workflow cannot reliably reach a failure
boundary. Assert the observable outcome and keep the test tied to the contract.
When asserting a termination cause, keep unrelated limits permissive so they cannot win the race.

Keep assertions tied to the behavior under investigation. Prefer a public preservation or replay
call over inspecting a private cache, and check the expected result rather than merely accepting
any valid result. Fold regression checks into an existing workflow when it already reaches the
same boundary; avoid repeating summary fields across equivalent fixtures.

There is no unit-test marker. The default pytest selection excludes optional
provider and performance tests, while process tests remain included. CI divides
the default suite into deterministic and process-boundary jobs and reports their
combined branch coverage without a percentage gate. The default local suite is:

```console
uv run pytest -q
```

Useful explicit selections are:

```console
uv run pytest -m e2e -q
uv run pytest -o addopts='' -m golden -q
uv run pytest -o addopts='' -m performance --durations=0
```

The real MCP stdio workflow checks the catalog against the capability registry,
including advertised schemas, examples, and result validation. Derive expected
tools from that registry rather than hard-coding a count or a removed interface.

A separate real stdio workflow starts with reduced server ceilings and verifies inherited defaults,
pre-execution limit rejection, and exact preservation/replay of lower request limits. Schema
regressions belong in the existing transport workflow, including numeric coercion and query bounds.
The stdio workflow also checks inline JSON against structured results, evidence inspection by ID,
and replay through returned source selectors. Restart, rescue, and provenance redaction workflows
use `inspect_evidence`; MCP resource capabilities are absent. Fault injection verifies that malformed
runtime results produce a typed contract failure before JSON serialization.

## Evidence and limits

The retained scale check publishes 1,000 real immutable manifests, closes the
runtime, and queries every page through a new runtime. Performance claims should
report the corpus, command, host-relevant limits, and result; compare the same
workload and environment. This check has no hardware-independent timing target.
The [performance investigation](performance.md) records measured workloads,
semantic oracles, optimization decisions, and the limits of those measurements.

Golden projections establish behavior for their supplied artifact and expected
fields. They do not establish compatibility with every upstream producer
release or a full provider capture lifecycle. A passing suite also does not
prove every publication boundary, provider, package-install route, or platform.
Name the missing evidence when it matters to a change.

The Node CPU workflow captures a real native profile over MCP stdio, rejects
Python-only metric options before workload execution, and reanalyzes preserved
evidence after an invalid analysis request and a CLI restart without rerunning
the workload. It skips when Node.js is unavailable. V8 import regressions reject
malformed required fields while retaining valid empty sample arrays.
An optional CPU-only PyTorch stdio workflow captures native operator evidence,
then replays summary, caller edges, operator rows, and a time window from that
same saved trace. It requires PyTorch and a local Perfetto Trace Processor;
the workload marker proves replay does not execute the workload again.

The capture preservation workflow checks shared scratch ownership: preserving one
analysis retains native files needed by an unpreserved sibling, and preserving
the last sibling releases them. Decoder fault-injection workflows run real
subprocesses that export XML or Parquet, then fail, omit required output, or time
out. They verify failed output cleanup, retry, and successful conversion reuse;
they do not establish native xctrace or Nsight Systems exporter compatibility.

Public runtime workflows verify analysis-cache eviction and explicit input byte
and file-count bounds. Known proof gaps include:

- Resource-baseline race handling and unavailable-metric branches lack retained
  direct proof.
- Cancellation during subprocess startup, V8 hard traversal ceiling overflow,
  and perf demangled/unknown-frame conversion lack retained direct proof.
- A managed dependency reconnect branch and live package installation remain
  unproved.
- Vendor tools, optional hardware, permissions, and other platforms are not
  exercised by the ordinary CI suite.

Provider tests should identify their actual artifact or host requirement. A
skip because a provider or host capability is unavailable is not provider
evidence. Keep observed, derived, and inferred claims distinct, including in
tests and performance reports.

The subprocess broker uses one execution path. Its workflows cover sampled RSS,
resource and output limits, cancellation with and without a deadline, and child
cleanup after parent exit. The unused native `wait4` execution mode and its
backend-specific assertions were removed; sampled RSS does not claim an exact
native peak. POSIX boundary regressions replace native files and bound executables
with FIFOs in a deadline-bounded subprocess and require typed rejection.

The worker harness likewise has one session implementation. Its lifecycle
workflow enters the synchronous session through `AnalysisRuntime.run_in_request`
and checks cancellation receipts, child settlement, consumer failure, and staging
cleanup. The removed async-only heartbeat path had no production callers;
its backend-specific test was replaced by this production-path proof.

Pytest retry workflows check that interrupted retry classifications agree between
summary metrics and diagnostic rows. Kernel malformed-artifact workflows exercise
both summary and comparison, including null output and metric containers. Native
benchmark comparisons retain reader-specific identities and zero-baseline ratios;
benchmark-sample workflows cover aggregate and ratio overflow. NVBench float32
sidecars cannot represent the float64 overflow fixtures used by those workflows.
