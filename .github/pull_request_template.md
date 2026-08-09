<!--
PR title: type(optional-scope): imperative outcome
Example: fix(storage): preserve comparison evidence

The squash-merge commit and generated changelog use the PR title.
-->

## Description

<!-- What problem does this PR solve? Link related issues. -->

## Approach

<!-- Briefly explain the chosen approach. -->

## Commands run

<!-- Commands you used to verify the change (lint, type-check, test, etc.) -->
```console
uv run ruff check src tests tools
uv run mypy src tests tools
uv run python tools/test.py core
uv run python tools/test.py process
uv run python tools/test.py collection
```

## Compatibility

<!-- Call out any breaking changes, platform sensitivity, or adapter impact. -->
