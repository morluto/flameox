# Storage and evidence

Storage is optional. Discovery, inspection, analysis, and unpreserved capture
must not create `.flameox`, `.diagnostics`, SQLite files, or persistent DuckDB
files.

## Repository layout

The first explicit preservation creates exactly:

```text
.flameox/
├── repository.json
├── artifacts/sha256/<prefix>/<digest>/
│   ├── artifact.json
│   └── payload
├── evidence/sha256/<prefix>/<evidence-id>/
│   ├── manifest.json
│   └── data/...
└── .staging/<process-session>/<publication-id>/
```

`repository.json` contains only `format_version` and `created_at`. It does not
bind the repository to a project path and is not mutable lifecycle state.
Creation adds `.flameox/` idempotently to `.git/info/exclude` when the fixed
project root is a Git repository. Tracked `.gitignore` is never edited.

## Identities

Artifact identity is the lowercase SHA-256 of native bytes. An artifact bundle
contains the exact payload and metadata needed to validate its digest and size.
Multi-file native inputs remain separate content-addressed artifacts whose
manifest roles bind their relative paths. An `EvidenceSource` rebuilds such a
bundle only in session scratch, so NVBench and similar directory formats remain
reanalyzable without introducing a mutable repository checkout.

When an analysis composes independently preserved bundles, their source-local artifact roles may
legitimately collide. Publication assigns deterministic `source-NNNN/` role namespaces only to
colliding artifacts; the analysis inputs retain each original bundle role and evidence identity as
provenance.

Evidence identity is SHA-256 of the RFC 8785 canonical manifest body. The body
contains capability/provider identity, input digests, effective capture and
analysis requests, the evidence-episode timestamp, data-file digests, coverage,
and limitations. Process-local handles such as `analysis_id` and continuation
tokens are excluded from preserved data.

The stored envelope adds `format_version` and `evidence_id`. The canonical manifest remains the
local repository and CLI contract. MCP resources expose a separate redacted projection so durable
provenance is not confused with agent-visible metadata:

```text
application/vnd.flameox.evidence-projection+json;version=1
```

The projection retains immutable identities, digests, capture and analysis status, coverage, and
provider identity. It replaces capture requests with digests and bounded status fields and never
returns argv, environment values, working directories, input paths, or scratch paths.

## Publication

Artifact and evidence directories are assembled beneath the same-filesystem
`.staging` tree. Files are flushed and fsynced, the complete staged bundle is
validated, then its directory is renamed into its content-addressed destination.
The manifest therefore becomes visible only with complete data.

Concurrent identical publications converge on one destination and validate it.
An existing payload or manifest that differs from its content identity is
repository corruption. Repeating `preserve_evidence` for the same session
analysis revalidates the immutable bundle and returns the same evidence reference.

Every bundle is independently valid and retained. There are no generations,
HEAD refs, commits, mutable indexes, catalog locks, trash manifests, or general
GC. Startup may remove another staging owner only when its recorded process ID
is provably dead.

## Queries and resources

`query_evidence` sorts the manifest inventory deterministically and computes an
inventory digest before filtering. A continuation is bound to that inventory;
mutation makes it stale rather than silently changing the page. Filters cover
evidence kind, capability, provider, input digest, and time bounds.

`flameox://evidence/{evidence_id}` validates the canonical manifest and returns its redacted MCP
projection. `flameox evidence show` remains the explicit local administrative view of the full
canonical manifest.
Missing or corrupt resources are MCP resource errors. Native payload bytes are
not exposed through resources; their digest and role remain visible in the
manifest.

## Format evolution

This is repository format `1`. Unsupported repository or manifest versions
fail explicitly before their contents are trusted.
