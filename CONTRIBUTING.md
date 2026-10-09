# Contributing to flameox

Contributions should improve how an investigator collects, preserves, compares, or inspects runtime
evidence. Flameox coordinates existing profilers and trace processors; it is not a profiler,
hosted observability service, unrestricted command or SQL gateway, or source-code modification
system. Read the [architecture authority map](docs/architecture.md#authority-map) before proposing
a change that crosses product boundaries.

## Before you start

Use the issue templates for bugs, feature requests, and design discussions. Small fixes and
documentation changes can go directly to a pull request. For a substantial feature, new
integration, or public or persisted contract change, open an issue first to align on behavior and
scope. Search existing issues and pull requests. Report security vulnerabilities privately as
described in [SECURITY.md](SECURITY.md).

## Development

Flameox requires Python 3.12 or newer and uses `uv` with the committed lockfile:

```console
git clone https://github.com/morluto/flameox.git
cd flameox
uv sync --extra dev
uv run flameox --help
```

Install optional extras only for the providers needed in your work. See
[docs/testing.md](docs/testing.md) for test markers, provider requirements, CI selections, and
known proof gaps.

Read the contract that owns the behavior before changing it:

- [Architecture](docs/architecture.md) for process model and package boundaries.
- [Storage and evidence](docs/storage-and-evidence.md) for provenance and preservation.
- [Investigations](docs/investigations.md) for experiments and comparisons.
- [Adapters](docs/adapters.md) for providers and compatibility.
- [Runtime safety](docs/runtime-safety.md) for concurrency, recovery, and privacy.
- [Interfaces](docs/interfaces.md) for CLI and MCP behavior.

Preserve native artifacts and provenance, including failed attempts. Keep observed, derived, and
inferred claims distinct, and report limitations instead of hiding them behind fallbacks. Keep CLI
and MCP as thin transports over the shared runtime. Prefer an existing repository helper or a
maintained public interface over a new abstraction.

Use complete type annotations and Python 3.12 syntax. Ruff enforces formatting, import ordering,
and lint rules; mypy runs in strict mode. Prefer end-to-end workflows through the CLI or real MCP
stdio transport, then integration tests across the affected boundary, then focused golden examples.
Retain narrower tests only for behavior those workflows cannot prove.

Run focused checks for the code you changed, followed by the relevant project checks:

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run pytest -q
```

`pytest -q` runs the default suite. Select optional-provider, performance, or other marked tests
using the guidance in [docs/testing.md](docs/testing.md). For changes under `npm/`, run its checks:

```console
cd npm
npm ci
npm run lint
npm run format:check
npm test
```

Update the owning contract when behavior changes. Update user guides and examples when their
commands or claims are affected. Describe proof gaps when the available validation cannot establish
the behavior.

## Commits and pull requests

Keep commits focused and reviewable. Use Conventional Commit subjects such as
`fix(storage): preserve provenance during artifact deduplication` or
`docs: explain comparison compatibility`.

Open pull requests against `main` and complete the repository template. Explain the concrete
problem and chosen approach, link related issues, list commands actually run, and describe relevant
compatibility or safety effects and proof gaps. Include representative output for user-visible CLI
or protocol changes. Before submitting, review the complete diff against `main` and check that
documentation and validation claims match the final tree.

The project is available under the [MIT License](LICENSE).
