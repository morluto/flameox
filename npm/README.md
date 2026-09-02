# flameox

This npm package starts the matching Python 0.2 setup command through `uvx`:

```console
npx flameox@latest setup
```

Setup detects supported coding agents, asks which global MCP clients to configure, and writes a
Python 3.12 `uvx` launcher pinned to the exact Flameox release resolved by npm. It preserves
unrelated configuration and tells you which clients must restart or reconnect. Use
`--client codex --yes`, repeated `--client` options, or `--all --yes` for automation, and
`--dry-run` to inspect the resolved paths without writing them.

Without explicit `--provider` options, setup does not install profilers, optional packages, a
persistent managed runtime, or project state. The Python server has no workspace binding. Explicit
preservation writes to Flameox's user-level data directory.

For direct CLI use, install or run the Python package with `uv`/`uvx`.
