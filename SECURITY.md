# Security

## Reporting a vulnerability

If you discover a security issue in flameox, please report it privately through
[GitHub Security Advisories](https://github.com/morluto/flameox/security/advisories/new).
Do not open a public issue.

## Secrets management

flameox is a local CLI tool and MCP server. Analysis and capture do not make
control-process network requests. `flameox setup` and MCP `prepare_providers`
may invoke package installers that access the network; directly executed targets
may use the network according to their own behavior. Capture is trusted local
execution; Flameox reports the available containment but does not claim a sandbox.
Configuration that could contain sensitive values should be provided through
environment variables, not committed to the repository.

- Export environment settings into the invoking process; Flameox does not load
  `.env` files automatically. See `.env.example` for the Trace Processor setting.
- Never commit credentials, tokens, or private keys.

## Dependency security

- [pip-audit](https://github.com/pypa/pip-audit) runs in CI to flag known
  vulnerabilities in Python dependencies.
- [Renovate](https://docs.renovatebot.com/) waits three days before proposing
  dependency updates. This delay does not establish a release's safety.

## Evidence safety

Analysis is bounded and session-local unless `preserve_evidence` or `--preserve`
is requested. Native artifacts and captured output can contain workload data;
preserving them copies those bytes and their provenance into the user-level
content-addressed evidence directory selected by `FLAMEOX_DATA_DIR` or the platform default.
Use `flameox evidence location` to inspect that location. Treat that directory with the same
sensitivity as the measured application.
