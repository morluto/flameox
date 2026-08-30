# flameox

This is the small npm bootstrap for flameox's local MCP setup wizard:

```console
npx flameox@latest setup
```

For a non-interactive update of the detected MCP clients and their managed
runtime, run:

```console
npx flameox upgrade
```

`upgrade` first resolves `flameox@latest`, then launches its matching Python
package with `uvx`. The wizard installs a persistent, versioned local runtime
and writes only the MCP client configurations you approve. For the interactive
setup flow, keep `@latest` in the command: an unqualified `npx flameox setup`
invocation may reuse an older cached bootstrap. The bootstrap refreshes uv
metadata for the pinned Python package before resolving it, so a newly
published runtime is visible even when uv has cached an older package index.

Runtime upgrades do not migrate incompatible `.diagnostics/` workspaces. Keep
the old workspace, create a new one, and import any native artifacts you still
need before deleting it.
