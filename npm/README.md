# flameox

The npm package provides the `setup` command for configuring Flameox as a local MCP server:

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

The npm command only exposes setup. For analysis, capture, evidence management, and MCP server
commands, use the Python CLI through `uv` or `uvx`. See the
[interface guide](https://github.com/morluto/flameox/blob/main/docs/interfaces.md) for setup and
CLI contracts.
