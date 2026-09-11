# Architecture

Flameox 0.2 is a process-lifespan capability runtime with an optional immutable
evidence repository. It is not a workspace application and has no mutable
control plane.

## Authority map

| Concern | Authority |
| --- | --- |
| Artifact location | Exact paths supplied by each analysis request |
| Target location | Exact absolute `cwd` supplied by each capture request |
| Available operations | Process-lifespan capability registry |
| In-progress work | Current MCP request and cancellation scope |
| Unpreserved output | Bounded session scratch |
| Completed preserved evidence | Immutable manifest in the user Flameox data directory |
| Native bytes | Content-addressed artifact bundle |
| Query view | Sorted manifest inventory pinned for one query |
| Analysis/storage limits | Startup defaults, lowerable per request |
| Workload time/RSS | Optional explicit target budget |
| Hypotheses and narrative | Agent-owned notes outside Flameox |

The runtime has no workspace identity and never searches parent directories. Analysis consumes
explicit absolute paths. Capture consumes a typed argv and explicit absolute working directory.
Neither is interpreted relative to server startup.

## Process model

Capture retains bounded, memory-backed console diagnostics by default.
Native artifacts remain useful when an investigation needs their contents; full
console output is retained only when it is the requested evidence, an oracle
requires it, or the caller explicitly requests it. Only that full-output case
needs a disk-backed console sink. Preservation controls durability, not an
implicit expansion of what is collected. See
[workload resources and evidence bounds](workload-resource-policy.md) for the
remaining native-artifact and storage-admission work.

`DirectTarget.budget` owns optional workload time and sampled process-tree RSS
controls. Absent values impose neither a workload deadline nor an RSS cap.
Capture and semantic oracles use that budget; worker and conversion protection
continues to use `RequestLimits`. Storage bounds and cancellation are independent
of both. There is no hidden inheritance from decoder limits to trusted workloads.

`AnalysisRuntime` owns the capability registry, subprocess broker, bounded scratch artifacts
(conversions and materialized evidence), and least-recently-used session analysis cache.
The MCP lifespan creates one runtime, exposes it through the SDK request context, and destroys its
scratch on shutdown. Evicting
a capture analysis removes its native session artifacts; a later preservation attempt reports that
the session handle expired. Long work stays inside the request that started it. Progress is reported
through the SDK context and cancellation unwinds the broker, including descendant cleanup.

MCP shared-state operations enter `AnalysisRuntime.run_in_request`, which owns a
task-group-joined worker thread and a per-runtime lock. The same boundary covers
capture admission/finalization and scratch bookkeeping. It keeps blocking readers
off the transport loop without allowing overlapping cache or protection-set
mutations. Capture subprocesses run outside the lock. Dependency preparation only
joins the state boundary when publishing a verified collector binding.

`analysis_id` is a session handle. It is intentionally meaningless after
restart and can be passed to `preserve_evidence` or the bounded
`rescue_evidence` recovery operation. `evidence_id` is a
durable SHA-256 identity derived from the canonical manifest body.

## Package boundaries

- `runtime_contracts.py` owns public models and the capability/capture-provider registries.
- `runtime.py` owns bounded analysis, capture orchestration, scratch, and the session cache.
- `providers/capture.py` owns provider-specific command construction and expected native outputs;
  `providers/availability.py` owns installation and workload requirements.
- `repository.py` owns lazy repository creation, validation, publication,
  source selection and layout, inventory queries, and immutable resource reads.
- `evidence_models.py` owns typed persisted document shapes and membership invariants;
  `source_files.py` owns shared native-source identities, hashing, and bounded copying.
- `adapters/json_preview.py` owns the streaming JSON preview projection.
- `execution.py` and `command_binding.py` own executable binding, subprocess
  limits, cancellation, output bounds, and descendant cleanup.
- `mcp/server.py` and `cli.py` are thin projections over the same runtime.
- Provider adapters accept resolved explicit inputs and return typed evidence;
  they do not discover source trees or publish evidence.

DuckDB may be used in memory for bounded aggregation. It is never a durable
catalog. Flameox production code must not create or depend on SQLite state.

## Capability boundary

One registry entry owns a capability descriptor, strict argument model, accepted formats,
capture/analysis semantics, and model-visible selection guidance. MCP projects each entry
into a read-only analysis tool and, when a compatible capture provider exists, a separate executing
capture tool. The generated callable closes over the stable capability ID; agents never pass a
capability selector or a free-form analysis argument object.

Capture-provider contracts supply the discriminated provider variants for each compatible capture
tool. Missing packages, executables, permissions, versions, or platforms do not change the catalog;
the attempted tool returns typed remediation. The separately invoked CLI setup command or MCP
`prepare_providers` tool resolves dependencies according to where they execute. CLI setup prepares
the complete version-pinned server environment. MCP preparation can activate a pinned standalone
py-spy collector in the existing session; server-import dependencies are checked against the active
release's complete requested dependency contract. Neither creates project state nor owns a durable
operation or provider inventory. Host profilers, drivers, and permissions remain external.
`providers/preparation.py` owns bounded preparation and verified session executable bindings;
transports do not infer readiness from installation receipts.

Request validation is also a normalization boundary. Runtime admission constructs the typed
capability and provider models once; execution, provenance, limitations, and result descriptions
must consume those validated models. They must not reinterpret truthiness or coercible values from
the caller's original mapping after admission.

Direct capture is trusted local execution, not containment. Typed argv prevents
shell interpretation, while the broker provides process-group cleanup, bounded
output, timeouts, resource observation, and exact collector and workload executable identities.
Those bounds do not provide
a network sandbox or neutralize the target program's own external side effects. MCP effect
annotations are discovery and confirmation hints, not authorization controls.
