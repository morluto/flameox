# flameox

This is the small npm bootstrap for flameox's local MCP setup wizard:

```console
npx flameox setup
```

For a non-interactive update of the detected MCP clients and their managed
runtime, run:

```console
npx flameox upgrade
```

It launches the matching `flameox` Python package with `uvx`. The
wizard installs a persistent, versioned local runtime and writes only the MCP
client configurations you approve. Run `npx flameox@latest setup` when you want
to force npm to resolve the newest setup release.
