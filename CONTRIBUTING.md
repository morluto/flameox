# Contributing to flameox

Thanks for helping improve flameox. Contributions are most useful when they
strengthen the path from a runtime symptom to evidence that another person or
agent can inspect and try to disprove.

Before proposing a large change, read the [authority
map](docs/architecture.md#authority-map). flameox coordinates existing
profilers and trace processors; it is not a new profiler, a hosted observability
service, an unrestricted command or SQL gateway, or a generic source-code
modification system.

## Before you start

Use the repository's issue templates for bugs, feature requests, and design
discussions. Small fixes and documentation improvements can usually go straight
to a pull request. For a substantial feature, new integration, or change to a
public or persisted contract, open an issue first so the intended behavior and
contribution fit can be agreed before implementation.

Search existing issues and pull requests before starting. If you discover a
security vulnerability, follow [SECURITY.md](SECURITY.md) and report it privately
instead of opening a public issue.

## Development setup

flameox requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/) with the committed `uv.lock`:

```console
git clone https://github.com/morluto/flameox.git
cd flameox
uv sync --extra dev
uv run flameox --help
```

Install only the optional providers needed for the area you are changing. The
[testing guide](docs/testing.md#optional-and-performance-evidence) lists the
available extras and their markers. To install every supported integration,
run:

```console
uv sync --extra dev --extra memory --extra trace --extra cpu --extra torch
```

## Understand the contract you are changing

Production code uses a `src/` layout. `runtime_contracts.py` owns public contracts and registries,
`stateless.py` owns request-local orchestration, `repository.py` owns optional immutable
preservation, and `execution.py` owns bounded subprocess work. Provider integrations live in
`providers/`, reusable native-format parsing in `adapters/`, isolated protocols in `workers/`, and
transport code in `cli.py` and `mcp/`. Tests mirror these semantic owners under `tests/`.

Read the contract that owns the behavior before editing it:

| Area | Contract |
| --- | --- |
| Process model, dependencies, and package boundaries | [Architecture](docs/architecture.md) |
| Storage, provenance, publication, and schemas | [Storage and evidence](docs/storage-and-evidence.md) |
| Experiments, comparisons, statistics, and evidence quality | [Investigations](docs/investigations.md) |
| Profiler integrations, compatibility, and adapter policy | [Adapters](docs/adapters.md) |
| Concurrency, recovery, integrity, security, and privacy | [Runtime safety](docs/runtime-safety.md) |
| CLI and MCP behavior and trust boundaries | [Interfaces](docs/interfaces.md) |
| Test markers, provider requirements, and CI | [Testing](docs/testing.md) |

Keep the CLI and MCP server as thin transports over the same application
services. Preserve native artifacts, provenance, failed attempts, and
experimental structure. Observed, derived, and inferred claims must remain
distinct, and limitations must be reported rather than hidden behind a fallback.

Prefer a maintained public interface or an existing repository helper over a
custom abstraction. Fix the condition that caused a defect rather than adding a
fixture-specific workaround.

For provider and projection changes, check these invariants before implementation:

- apply semantic filtering before row, byte, or worker limits;
- make `rows_observed`, `coverage.complete`, truncation, and the returned table describe the same
  semantic population;
- retain every dimension that distinguishes an evidence series, except an explicitly selected
  analysis axis;
- consume validated request models after admission rather than rereading raw mappings; and
- exercise neighboring projections and a realistic native artifact so a new branch cannot turn
  existing evidence into an incorrect complete result with zero rows.

## Make and test the change

Use complete type annotations and Python 3.12 syntax. Ruff enforces formatting,
import ordering, a 100-character line limit, and the configured lint rules; mypy
runs in strict mode.

Add tests near the behavior's semantic owner. Name test files `test_<area>.py`
and tests `test_<observable_behavior>`. Prefer observable behavior or stable
artifacts over assertions about private helper names or source text. Cover the
meaningful failure path as well as the success path, and use Hypothesis when the
contract is an invariant over a useful input range.

Run a focused test while iterating:

```console
uv run pytest tests/test_stateless.py -q
```

Then run validation proportional to the change. The usual baseline is:

```console
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run pytest -q
```

`pytest -q` runs the fast deterministic default suite without hidden retries.
Use paths and registered markers from [docs/testing.md](docs/testing.md) for
process, optional-provider, and performance behavior. In particular:

- Run `uv run lint-imports` when changing package boundaries.
- Run the matching optional-provider marker when changing an integration; a skip
  because the provider is unavailable is not provider evidence.
- Run `uv run pytest -o addopts='' -m performance`
  only for changes whose claims depend on the declared performance budgets.

For changes under `npm/`, use the package's own checks:

```console
cd npm
npm ci
npm run lint
npm run format:check
npm test
```

A passing default suite is not sufficient for every behavioral change. Provide
the representative crash, concurrency, containment, protocol, golden, or scale
proof that the behavior requires. If a proportionate proof is not feasible,
describe the gap rather than substituting a test that mirrors the implementation.

Update the owning contract when behavior changes. Also update the README, CLI or
MCP examples, and compatibility notes when they are affected.

## Commits and pull requests

Keep commits focused, reviewable, and buildable. Commit subjects follow the
Conventional Commit style used in the repository, for example:

```text
fix(storage): preserve provenance during artifact deduplication
feat(adapters): add bounded provider readiness probe
docs: explain comparison compatibility
```

Pull request titles follow the same `type(optional-scope): imperative outcome`
format. GitHub uses the title as the squash-merge commit subject, and `git-cliff`
uses that subject to place and describe the change in the generated changelog.

Open the pull request against `main` and complete the pull request template.
Explain the concrete problem, the chosen approach, and why it fits flameox's
architecture. Link related issues and list only commands that actually ran.
Call out compatibility, platform, persistence, security, or containment effects,
along with any meaningful proof gaps.

For user-visible CLI or protocol changes, include representative output. Keep the
change focused on one outcome and avoid unrelated cleanup or formatting churn.
Before submitting, review the complete diff against `main` and confirm that the
documentation and test claims match the final tree.

By participating, please keep discussion technical, specific, and collaborative.
The project is available under the [MIT License](LICENSE).
