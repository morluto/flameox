# flameox

This npm package starts the matching Python 0.2 setup command through `uvx`:

```console
npx flameox@latest setup
```

Setup prints the repo-local stdio MCP launch configuration. It does not install
profilers, optional packages, a persistent managed runtime, or project state.
The Python server fixes `project_root` at startup and creates `.flameox` only
after an explicit evidence-preservation request.

For direct CLI use, install or run the Python package with `uv`/`uvx`.
