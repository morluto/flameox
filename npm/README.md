# flamo

This is the small npm bootstrap for Flamo's local MCP setup wizard:

```console
npx flamo setup
```

It launches the matching `flamo-diagnostics` Python package with `uvx`. The
wizard installs a persistent, versioned local runtime and writes only the MCP
client configurations you approve. Run `npx flamo@latest setup` when you want
to force npm to resolve the newest setup release.
