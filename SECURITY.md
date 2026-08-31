# Security

## Reporting a vulnerability

If you discover a security issue in flameox, please report it privately through
[GitHub Security Advisories](https://github.com/morluto/flameox/security/advisories/new).
Do not open a public issue.

## Secrets management

flameox is a local CLI tool and MCP server. Analysis and capture do not make
control-process network requests. `flameox setup` may invoke the selected Python
package installer, and a directly executed target may use the network according
to its own behavior. Capture is trusted local execution; Flameox reports the
available containment but does not claim a sandbox. Configuration that could
contain sensitive values should be provided through environment variables, not
committed to the repository.

- Copy `.env.example` to `.env` for local development and keep `.env` in
  `.gitignore` (already configured).
- Never commit credentials, tokens, or private keys.

## Dependency security

- [pip-audit](https://github.com/pypa/pip-audit) runs in CI to flag known
  vulnerabilities in Python dependencies.
- [Renovate](https://docs.renovatebot.com/) is configured with a 3-day minimum
  release age to reduce supply-chain risk from compromised new releases.

## Evidence safety

Analysis is bounded and session-local unless `preserve_evidence` or `--preserve`
is requested. Native artifacts and captured output can contain workload data;
preserving them copies those bytes and their provenance into the project's
content-addressed `.flameox` repository. Treat that repository with the same
sensitivity as the measured application.
