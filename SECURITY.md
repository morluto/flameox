# Security

## Reporting a vulnerability

If you discover a security issue in flameox, please report it privately through
[GitHub Security Advisories](https://github.com/morluto/flameox/security/advisories/new).
Do not open a public issue.

## Secrets management

flameox is a local CLI tool and MCP server. Ordinary capture, import,
extraction, and analysis do not make control-process network requests. Explicit
setup, upgrade, approved provider acquisition, and enabled symbol services may
use the network; declared workloads may use it unless containment denies it.
Configuration that could contain sensitive values should be provided through
environment variables, not committed to the repository.

- Copy `.env.example` to `.env` for local development and keep `.env` in
  `.gitignore` (already configured).
- Never commit credentials, tokens, or private keys.

## Dependency security

- [pip-audit](https://github.com/pypa/pip-audit) runs in CI to flag known
  vulnerabilities in Python dependencies.
- [Renovate](https://docs.renovatebot.com/) is configured with a 3-day minimum
  release age to reduce supply-chain risk from compromised new releases.

## Log safety

flameox's structured operation logger (`observability.py`) enforces a bounded
event schema. Logs exclude raw environments, artifact contents, core memory,
and unrestricted child output. Native artifacts and explicitly preserved
bounded output may still contain workload data and retain their sensitivity
classification.
