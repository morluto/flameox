# Security

## Reporting a vulnerability

If you discover a security issue in flameox, please report it by emailing the
maintainer at the address listed in the commit history. Do not open a public
issue.

## Secrets management

flameox is a local CLI tool and MCP server. It does not transmit data over the
network. Configuration that could contain sensitive values (API keys, tokens,
paths) should be provided through environment variables, not hardcoded in
source or committed to the repository.

- Copy `.env.example` to `.env` for local development and keep `.env` in
  `.gitignore` (already configured).
- Never commit credentials, tokens, or private keys.
- The CI pipeline runs [gitleaks](https://github.com/gitleaks/gitleaks) on
  every PR to detect accidentally committed secrets.

## Dependency security

- [pip-audit](https://github.com/pypa/pip-audit) runs in CI to flag known
  vulnerabilities in Python dependencies.
- [Renovate](https://docs.renovatebot.com/) is configured with a 3-day minimum
  release age to reduce supply-chain risk from compromised new releases.

## Log safety

flameox's structured operation logger (`observability.py`) enforces a bounded
event schema that forbids arbitrary payloads. String fields are sanitized to
redact email addresses and IP addresses before writing. No user data, process
arguments, or file contents are ever logged.
