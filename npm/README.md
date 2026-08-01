# flameox

This is the small npm bootstrap for flameox's local MCP setup wizard:

```console
npx flameox@latest setup
```

For a non-interactive update of the detected MCP clients and their managed
runtime, run:

```console
npx flameox@latest upgrade
```

It launches the matching `flameox` Python package with `uvx`. The
wizard installs a persistent, versioned local runtime and writes only the MCP
client configurations you approve. Keep `@latest` in the command: an
unqualified `npx flameox` invocation may reuse an older cached bootstrap. The
bootstrap refreshes uv metadata for the pinned Python package before resolving
it, so a newly published runtime is visible even when uv has cached an older
package index.
