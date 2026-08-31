# Interfaces

CLI and MCP are thin transports over `AnalysisRuntime`. They do not own storage,
provider behavior, or lifecycle state.

## MCP catalog

The catalog contains exactly six tools:

| Tool | Behavior |
| --- | --- |
| `discover_capabilities` | Rank by intent and bounded source sniffing; report provider state and external remediation. |
| `inspect_capabilities` | Inspect 1-16 IDs with source modes, strict argument schema, examples, limits, and capture semantics. |
| `analyze` | Analyze 1-32 explicit path/evidence sources and return bounded inline evidence plus a session `analysis_id`. |
| `capture_and_analyze` | Run typed argv through the broker in single or bounded experiment mode, then analyze outputs. |
| `preserve_evidence` | Idempotently publish one session analysis and return an evidence reference plus `ResourceLink`. |
| `query_evidence` | Search a deterministic immutable manifest inventory with bounded pagination. |

There is one resource template:

```text
flameox://evidence/{evidence_id}
```

`resources/list` is empty. `resources/read` returns the canonical manifest with
the versioned media type. A missing resource is a protocol error.

Tools do not advertise output schemas. Success uses structured content directly,
without an `ok/result/error` envelope. Tool failures set `isError=true` and carry
a stable code, message, and details.

## Sources and limits

The strict source union is:

```text
PathSource     {kind: "path", path, format?, producer?, expected_sha256?}
EvidenceSource {kind: "evidence", evidence_id, artifact_role?}
```

Continuations are opaque, session-bound, and cryptographically bound to the
request and exact input digests. A changed input cannot reuse a continuation.

Requests may lower startup row, result-byte, timeout, output-byte, and durable
provenance-byte limits. Durable provenance bounds the captured argv and execution
metadata retained for explicit preservation.
They cannot raise them.

## Capture

A direct target contains an argv array, project-contained cwd, at most 32
bounded environment overrides after experiment-case overrides are merged,
provider ID, provider capture arguments, analysis arguments, and request limits.
Shell command strings are not accepted.

Experiment mode adds 2-16 cases, 1-100 blocks, a seed, metric, estimand,
practical threshold, and optional semantic-oracle argv. Version 0.2 evaluates
`wall_time_ns` with a paired `median_difference` or `mean_difference`, reports
eligible blocks and a deterministic percentile interval when at least three
blocks survive capture/oracle validation, and classifies the effect against the
declared threshold. Work is not detached; the request receives progress and owns
cancellation.

## Stable failure codes

The transport distinguishes invalid input, unknown/unavailable capability,
missing or changed input, unsupported format, decode failure, execution failure,
cancellation, limit exceeded, expired session analysis, missing evidence,
repository I/O failure, repository corruption, and unsupported repository
format. Provider absence during discovery is a successful state.

## CLI

The retained surface is:

```text
flameox setup
flameox mcp serve|inspect
flameox capabilities discover|inspect
flameox analyze [--preserve]
flameox capture [--preserve] -- <argv...>
flameox evidence query|show
```

`setup` prints stdio configuration. Repeated `--provider` options explicitly
select the complete Python provider-extra set installed into a persistent uv
tool environment. System and vendor providers receive external installation
guidance. Setup does not create a project repository, durable operation, or MCP
setup endpoint. Other CLI commands construct the same runtime and project-root
rules used by MCP.
