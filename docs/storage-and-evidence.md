# Storage and evidence

Storage is optional. Analysis and unpreserved capture must not create user data, project files,
SQLite files, or persistent DuckDB files.

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
`inspect_evidence` exposes counts and completeness, not excerpt text.
Failed captures with no native artifacts can still preserve their diagnostics,
execution outcome, and analysis failure without creating placeholder log files.

If inline preservation fails because the repository is corrupt, unsupported, or
unwritable, the completed capture remains in the session. The typed failure includes
`details.analysis_id`; use it with `rescue_evidence` and a new evidence directory
before restarting. Retrying publication does not require another workload execution.

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

Derived identity keys retain wide native integers using typed decimal tags,
without rounding them or conflating them with strings. Native JSON objects that
contain the reserved integer or object tag keys are escaped, so they cannot
impersonate those encodings. Ordinary I-JSON identity bytes remain unchanged;
canonical persisted manifests and native artifact hashes use their existing
representations.

When an analysis composes independently preserved bundles, their source-local artifact roles may
legitimately collide, including a bundle composed with one of its own members. Publication checks
both logical and expanded artifact roles. If either collides, it assigns deterministic
`source-NNNN/` namespaces to the source groups together. Relative paths are explicit layout fields,
not parsed from newly published roles. Analysis inputs retain their original roles and identities.

Evidence identity is SHA-256 of the RFC 8785 canonical manifest body. The body
contains operation/provider identity, input digests, effective capture and
analysis requests, the evidence-episode timestamp, data-file digests, coverage,
and limitations. Process-local handles such as `analysis_id` and continuation
tokens are excluded from preserved data.

Content identity does not establish schema validity. Readers parse the complete manifest shape,
including nested capture targets, executions, analysis inputs, offsets, and failure records, before
querying or projecting it. A self-consistent manifest with a correctly recomputed evidence ID but an
invalid nested request is repository corruption.

The stored envelope adds `format_version` and `evidence_id`. The canonical manifest remains the
local repository and CLI contract. The MCP `inspect_evidence` tool exposes a separate redacted
projection inline so durable provenance is not confused with agent-visible metadata.

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

## Queries and inspection

`query_evidence` sorts the manifest inventory deterministically and computes an
inventory digest before filtering. A continuation is bound to that inventory;
mutation makes it stale rather than silently changing the page. Filters cover
evidence kind, operation, provider, input digest, and time bounds. The provider
filter matches either the analysis provider recorded in the manifest or, for
capture evidence, the capture collector ID recorded in its target provenance.

Cursor offsets are strict non-negative integers and must identify an item inside the bound
inventory. An offset at or beyond the inventory end is invalid rather than an empty successful
page; an empty page therefore means that the valid remaining inventory contains no matches.

`inspect_evidence` validates the canonical manifest for an exact evidence ID and returns its
redacted projection inline. `flameox evidence show` remains the explicit local administrative view
of the full canonical manifest. Missing or corrupt evidence returns a structured tool failure.
Native payload bytes remain local; their digest and source selector are visible in the projection.
Evidence references contain IDs rather than resource URIs.

Agent projections expose an opaque `source` accepted unchanged by analysis tools for each artifact,
plus `logical_sources` for directory bundles and ordered `analysis_sources` for the original
analysis. Selectors address immutable manifest positions, not hashes of private roles that could
be checked against guessed filenames. File selectors always select exact members, even when a
filename contains the directory-role delimiter. Manifests include `source_layout`: each source
declares its file/directory kind, exact artifact indices, relative member paths, and identity,
with an ordered mapping for
the original analysis inputs. Empty directories retain their metadata without inventing a native
payload, and re-preserving a selected member retains its file identity. Readers validate membership,
digests, sizes, and analysis mappings. Layouts and relative paths are required; readers never infer
membership from role strings. Missing or ambiguous selectors identify the evidence ID for enumeration
through `inspect_evidence`.

Member paths are normalized relative POSIX paths, with no drive, root, backslash, or parent
traversal. Members must be distinct and prefix-free: a bundle cannot contain both a file `a` and
another file `a/b`. Readers reject impossible layouts as repository corruption before materialization.

Analysis first selects manifest metadata and admits the aggregate input and scratch budgets.
Only then does it verify the selected payloads and materialize directory members. Unselected
payloads and derived analysis data are not read by source selection; full evidence/inspection reads
still verify the complete bundle. Materialization uses bounded copies into session-owned staging
and never writes more native bytes than admitted.

Corruption remains fail-closed. Runtime errors include the selected configuration source, a
path-free store identifier, and recovery instructions. `flameox evidence location` prints the
resolved directory locally without reading or initializing the repository. Restore the original
store from a known-good backup, or select a distinct empty store with `FLAMEOX_DATA_DIR` and
restart/reconnect. Before restarting a live process that still owns needed session evidence, call
`rescue_evidence` with that analysis handle and the distinct empty store. Rescue uses normal bounded,
validated, atomic evidence publication without changing the active configured repository. A live
session can also rescue an already-preserved bundle whose store metadata becomes unusable: its
recorded evidence ID pins the independently verified manifest, derived data, and native artifacts.
Corrupt bundle contents still fail closed. Repeated rescues validate the destination without
requiring the original store. Switching
stores does not recover evidence that was not preserved or rescued. Do not delete existing data or
synthesize replacement metadata.

## Repository format

Repository format `4` records the same task name used by runtime, CLI, and MCP in `operation`.
Analysis inputs bind their file or directory kind as well as their native digest. Unsupported
repository, artifact, and manifest versions fail before contents are trusted.
