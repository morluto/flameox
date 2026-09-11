# Storage and evidence

Storage is optional. Analysis and unpreserved capture must not create user data, project files,
`.diagnostics`, SQLite files, or persistent DuckDB files.

## Console retention and disk backing

The default is bounded, memory-backed console diagnostics, not a full-output
file for every execution. Report observed, retained, and omitted byte counts for
each stream, together with whether collection completed without interruption.
Interrupted collection is conservatively incomplete even if cleanup closes the
pipes. After interruption,
the eventual stream length may be unknown; omission counts cover only observed bytes.

Keep native profiles, traces, and other artifacts when the investigation needs
them. Retain full console output only when it is itself the evidence, a semantic
oracle needs it, or the caller explicitly requests it. Use request-owned disk
backing for that full-output case; disk backing is not a universal requirement
for console diagnostics. Native tools may independently require files.

Retention and preservation are separate choices. Preservation makes the selected
evidence durable; it does not enable full console collection or recover discarded
bytes. A preserved bounded excerpt must remain labeled as an excerpt, not a
complete native stream.

`target.console_output` defaults to `diagnostics`; `full` explicitly selects full
retention. Process-output capture and semantic-oracle inputs select full retention
automatically. An oracle's own console output remains diagnostic unless the caller
selects `full`. CLI capture exposes the same choice as `--console-output`.

Diagnostics retain at most 4,096 bytes per stream, lowered further to fit the
request's provenance budget. Text uses UTF-8 replacement decoding; byte counts
describe the original stream, not the encoded size of the decoded excerpt.
Full streams have `output_streams` metadata; excerpts have `console_diagnostics`.
The ordinary evidence resource exposes counts and completeness, not excerpt text.
Failed captures with no native artifacts can still preserve their diagnostics,
execution outcome, and analysis failure without creating placeholder log files.

See [workload resources and evidence bounds](workload-resource-policy.md) for the
remaining native-artifact budget and storage-admission work.

## Repository layout

The first explicit preservation creates exactly:

```text
<user-data>/flameox/
├── repository.json
├── artifacts/sha256/<prefix>/<digest>/
│   ├── artifact.json
│   └── payload
├── evidence/sha256/<prefix>/<evidence-id>/
│   ├── manifest.json
│   └── data/...
└── .staging/<process-session>/<publication-id>/
```

`repository.json` contains only `format_version` and `created_at`. It does not bind the repository
to a project path and is not mutable lifecycle state. The default follows the platform user-data
location; `FLAMEOX_DATA_DIR` overrides it. Flameox never creates or edits project Git files.

## Identities

Artifact identity is the lowercase SHA-256 of native bytes. An artifact bundle
contains the exact payload and metadata needed to validate its digest and size.
Multi-file native inputs remain separate content-addressed artifacts whose
manifest layout binds their relative paths. An `EvidenceSource` rebuilds such a
bundle only in session scratch, so NVBench and similar directory formats remain
reanalyzable without introducing a mutable repository checkout.

When an analysis composes independently preserved bundles, their source-local artifact roles may
legitimately collide, including a bundle composed with one of its own members. Publication checks
both logical and expanded artifact roles. If either collides, it assigns deterministic
`source-NNNN/` namespaces to the source groups together. Relative paths are explicit layout fields,
not parsed from newly published roles. Analysis inputs retain their original roles and identities.

Evidence identity is SHA-256 of the RFC 8785 canonical manifest body. The body
contains capability/provider identity, input digests, effective capture and
analysis requests, the evidence-episode timestamp, data-file digests, coverage,
and limitations. Process-local handles such as `analysis_id` and continuation
tokens are excluded from preserved data.

Content identity does not establish schema validity. Readers parse the complete manifest shape,
including nested capture targets, executions, analysis inputs, offsets, and failure records, before
querying or projecting it. A self-consistent manifest with a correctly recomputed evidence ID but an
invalid nested request is repository corruption.

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
Renaming our validated stage does not require hashing it again. If a concurrent publisher wins
instead, its destination is independently validated before reuse.

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

Cursor offsets are strict non-negative integers and must identify an item inside the bound
inventory. An offset at or beyond the inventory end is invalid rather than an empty successful
page; an empty page therefore means that the valid remaining inventory contains no matches.

`flameox://evidence/{evidence_id}` validates the canonical manifest and returns its redacted MCP
projection. `flameox evidence show` remains the explicit local administrative view of the full
canonical manifest.
Missing or corrupt resources are MCP resource errors. Native payload bytes are
not exposed through resources; their digest and role remain visible in the
manifest.

Agent projections expose an opaque `source` accepted unchanged by analysis tools for each artifact,
plus `logical_sources` for directory bundles and ordered `analysis_sources` for the original
analysis. Selectors address immutable manifest positions, not hashes of private roles that could
be checked against guessed filenames. File selectors always select exact members, even when a
filename contains the directory-role delimiter. New manifests include `source_layout`: each source
declares its file/directory kind, exact artifact indices, relative member paths, and identity,
with an ordered mapping for
the original analysis inputs. Empty directories retain their metadata without inventing a native
payload, and re-preserving a selected member retains its file identity. Readers validate membership,
digests, sizes, and analysis mappings. Layouts and relative paths are required; readers never infer
membership from role strings. Missing or ambiguous selectors point back to the evidence resource
for enumeration.

Member paths are normalized relative POSIX paths, with no drive, root, backslash, or parent
traversal. Members must be distinct and prefix-free: a bundle cannot contain both a file `a` and
another file `a/b`. Readers reject impossible layouts as repository corruption before materialization.

Analysis first selects manifest metadata and admits the aggregate input and scratch budgets.
Only then does it verify the selected payloads and materialize directory members. Unselected
payloads and derived analysis data are not read by source selection; full evidence/resource reads
still verify the complete bundle. Materialization uses bounded copies into session-owned staging
and never writes more native bytes than admitted.

Corruption remains fail-closed. Runtime errors include the selected configuration source, a
path-free store identifier, and recovery instructions. `flameox evidence location` prints the
resolved directory locally without reading or initializing the repository. Restore the original
store from a known-good backup, or select a distinct empty store with `FLAMEOX_DATA_DIR` and
restart/reconnect. Before restarting a live process that still owns needed session evidence, call
`rescue_evidence` with that analysis handle and the distinct empty store. Rescue uses normal bounded,
validated, atomic evidence publication without changing the active configured repository. Switching
stores does not recover evidence that was not preserved or rescued. Do not delete existing data or
synthesize replacement metadata.

## Format evolution

This is repository format `3`. Capture executions record collector and workload executable digests
separately; wrapped captures pin and revalidate both identities. Unsupported repository, artifact, or
manifest versions fail explicitly before their contents are trusted. Formats `1` and `2` are not
supported. Existing stores are never rewritten automatically: inspect or export them with a compatible
older release, and select a separate empty directory for a format-3 store.
Changing the version field does not migrate evidence and would invalidate its contract.
