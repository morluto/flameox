# Architecture

Flameox 0.2 is a process-lifespan capability runtime with an optional immutable
evidence repository. It is not a workspace application and has no mutable
control plane.

## Authority map

| Concern | Authority |
| --- | --- |
| Project boundary | `project_root` fixed once at server startup |
| Available operations | Process-lifespan capability registry |
| In-progress work | Current MCP request and cancellation scope |
| Unpreserved output | Bounded session scratch |
| Completed preserved evidence | Immutable evidence manifest |
| Native bytes | Content-addressed artifact bundle |
| Query view | Sorted manifest inventory pinned for one query |
| Limits | Startup defaults, lowerable per request |
| Hypotheses and narrative | Agent-owned notes outside Flameox |

The runtime never searches parents. A manual server defaults `project_root` to
its startup cwd. Every capture cwd resolves inside that fixed root; explicit
analysis paths may point elsewhere because the caller names the exact input.

## Process model

`AnalysisRuntime` owns the capability registry, subprocess broker, scratch
directory, conversion cache, and session analysis cache. The MCP lifespan
creates one runtime and destroys its scratch on shutdown. Long work stays inside
the request that started it. Progress is reported through the SDK context and
cancellation unwinds the broker, including descendant cleanup.

`analysis_id` is a session handle. It is intentionally meaningless after
restart and can only be passed to `preserve_evidence`. `evidence_id` is a
durable SHA-256 identity derived from the canonical manifest body.

## Package boundaries

- `runtime_contracts.py` owns public models and the capability/capture-provider registries.
- `stateless.py` owns bounded analysis, capture orchestration, scratch, and the session cache.
- `repository.py` owns lazy repository creation, validation, publication,
  inventory queries, and immutable resource reads.
- `execution.py` and `command_binding.py` own executable binding, subprocess
  limits, cancellation, output bounds, and descendant cleanup.
- `mcp/server.py` and `cli.py` are thin projections over the same runtime.
- Provider adapters accept resolved explicit inputs and return typed evidence;
  they do not discover workspaces or publish evidence.

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
`prepare_providers` tool may prepare the exact version-pinned uvx environment named by an explicit
Python provider set; neither creates project state nor owns a durable operation or provider
inventory. Host profilers, drivers, and permissions remain external and receive guidance only.

Direct capture is trusted local execution, not containment. Typed argv prevents
shell interpretation, while the broker provides process-group cleanup, bounded
output, timeouts, resource observation, and exact executable identity.
