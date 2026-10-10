# Contributing to Flameox

Contributions should improve collection, preservation, comparison, or inspection of runtime
evidence. See [architecture](docs/architecture.md) for product boundaries and
[AGENTS.md](AGENTS.md#find-the-relevant-contract) for the contract relevant to your change.

## Proposing changes

Small fixes and documentation changes can go directly to a pull request. For substantial
features or public or persisted contract changes, discuss behavior and scope in an issue unless
already agreed. Search existing issues and pull requests to avoid duplicate work. Report security
vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Development

Flameox requires Python 3.12 or newer and uses `uv` with the committed lockfile:

```console
git clone https://github.com/morluto/flameox.git
cd flameox
uv sync --extra dev
uv run flameox --help
```

Install optional extras only for providers needed in your work. Test selections and proof gaps
are in [testing](docs/testing.md); the complete automated checks are in
[CI](.github/workflows/ci.yml).

Use complete type annotations. Ruff owns formatting and lint rules; mypy runs in strict mode:

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run pytest -q
```

Choose checks appropriate to the change. For changes under `npm/`:

```console
cd npm
npm ci
npm run lint
npm run format:check
npm test
```

Update the owning contract and affected user examples when behavior changes. Document validation
limits when the available checks cannot establish a claim.

## Commits and pull requests

Keep commits focused. Use Conventional Commit subjects such as
`fix(storage): preserve provenance during artifact deduplication` or
`docs: explain comparison compatibility`.

Open pull requests against `main` and complete the repository template. Explain the concrete
problem and resulting behavior, related issues, commands actually run, and material compatibility
or safety effects. Include representative output for user-visible CLI or protocol changes.

The project is available under the [MIT License](LICENSE).
