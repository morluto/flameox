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

- `stateless.py` owns public models, capability discovery, bounded analysis,
  capture orchestration, scratch, and the session cache.
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

One registry entry owns a capability descriptor, strict argument model,
accepted formats, provider probes, and capture/analysis semantics. Discovery
performs only bounded suffix/header sniffing. Missing packages, executables,
permissions, versions, or platforms are successful discovery states; Flameox
never installs remediation automatically. The separately invoked CLI setup
command may install an explicitly selected Python provider-extra set into a uv
tool environment; it creates no project state and owns no durable operation.

Direct capture is trusted local execution, not containment. Typed argv prevents
shell interpretation, while the broker provides process-group cleanup, bounded
output, timeouts, resource observation, and exact executable identity.
