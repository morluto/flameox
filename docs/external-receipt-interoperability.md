# External profiler receipt interoperability

The fixture at
`docs/fixtures/gitcontribute-external-validation-v1.json` is a synthetic,
versioned handoff for consumers of `gitcontribute.external-validation.v1`. It
contains no host paths, private repository identity, secret, or captured user
data, and tests never execute a profiler or depend on GitContribute at runtime.

The downstream schema is strict and has no `producer_version` field. Flameox
therefore uses `producer = "flameox"` and records its version as
`environment["flameox.version"]`. That mapping is a compatibility constraint,
not a claim that the downstream contract has a separate version field.

| External receipt | Flameox source |
| --- | --- |
| `producer` and `environment["flameox.version"]` | Flameox package identity and version |
| `external_run_id` | `RunManifest.run_id` |
| `revision` | source-state commit or digest |
| `artifact_sha256` and `artifacts` | content-addressed artifact registrations |
| `argv` and `working_dir` | resolved command specification and a portable workspace label |
| `environment` | bounded environment identity fields |
| timestamps and exit code | run lifecycle and process result |
| `classification` | execution, capture, and validation states |
| `limitations` and `incomplete` | run, extraction, and validation qualification |

`receipt_sha256` follows the downstream contract: SHA-256 over its Go JSON
projection with `receipt_sha256` set to the empty string. Fields tagged
`omitempty`, including the checked-in false `truncated` value, are absent from
that digest projection. The primary and named artifact digests are raw
64-character lowercase hexadecimal strings because that is the downstream
wire representation; Flameox stores the corresponding local identities with a
`sha256:` prefix.

The fixture is deliberately incomplete. A consumer must preserve its concrete
kernel call-chain limitation and must not silently promote it to complete
evidence merely because the process and validation classification passed.
