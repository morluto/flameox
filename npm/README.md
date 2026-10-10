# flameox

The npm package provides `setup` and `update` for managing Flameox as a local MCP server:

```console
npx flameox@latest setup
```

It requires Node.js 18 or newer and `uv`. The launcher runs the matching version of the Python
package on Python 3.12 through `uvx`; it does not install a persistent global `uv` tool. Setup
detects supported MCP clients, asks which configurations to update, and preserves unrelated
settings. Changed clients need a restart or reconnect.

For non-interactive use, select clients explicitly with `--client codex --yes`, repeat `--client`,
or use `--all --yes`. `--dry-run` reports the selected paths and planned actions without writing.
Detection does not select clients automatically in non-interactive mode. Optional `--provider`
arguments prepare the complete managed provider set for the launcher. Host tools and drivers
remain separately managed.

Update already configured clients with `npx flameox@latest update`. It checks PyPI for the latest
stable Python release, preserves each client's provider extras and settings, and verifies all
selected environments before updating their version pins. `--client codex` selects one configured
client; without it, update targets all existing Flameox registrations and creates none.
`--check` and `--dry-run` report planned changes without preparing environments or writing files.
Use `--version 0.2.8` for a specific release or rollback. Restart or reconnect changed clients.

The npm command exposes setup and update. For analysis, capture, evidence management, and MCP server
commands, use the Python CLI through `uv` or `uvx`. See the
[interface guide](https://github.com/morluto/flameox/blob/main/docs/interfaces.md) for setup and
CLI contracts.
