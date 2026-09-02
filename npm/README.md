# flameox

This npm package starts the matching Python 0.2 setup command through `uvx`:

```console
npx flameox@latest setup
```

Setup prints a global stdio MCP launch configuration pinned to the exact
Flameox release resolved by npm and to Python 3.12. It does not change an MCP
client registration. Apply the printed command through the client's supported
MCP management interface. Without explicit `--provider` options, setup does not
install profilers, optional packages, a persistent managed runtime, or project state. The Python
server has no workspace binding. Explicit preservation writes to Flameox's user-level data
directory.

For direct CLI use, install or run the Python package with `uv`/`uvx`.
