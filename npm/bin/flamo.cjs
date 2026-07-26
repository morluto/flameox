#!/usr/bin/env node

"use strict";

const path = require("node:path");
const { spawn } = require("node:child_process");
const packageJson = require("../package.json");

const args = process.argv.slice(2);
if (args.length === 1 && (args[0] === "--version" || args[0] === "-V")) {
  process.stdout.write(`${packageJson.version}\n`);
  process.exit(0);
}
if (args[0] !== "setup") {
  process.stderr.write(
    "The npm package is the setup bootstrap. Run `npx flamo setup`.\n" +
      "After setup, use Flamo through a connected MCP client or the Python CLI.\n",
  );
  process.exit(2);
}

const helper = path.resolve(__dirname, "../lib/jsonc-edit.cjs");
const environment = {
  ...process.env,
  FLAMO_SETUP_JSONC_HELPER: helper,
};
const pythonPackage = `flamo-diagnostics==${packageJson.version}`;
const child = spawn(
  process.env.FLAMO_UV_EXECUTABLE || "uvx",
  [
    "--no-config",
    "--no-sources",
    "--from",
    pythonPackage,
    "flamo",
    ...args,
  ],
  { env: environment, stdio: "inherit" },
);

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("error", (error) => {
  if (error.code === "ENOENT") {
    process.stderr.write(
      "Flamo setup requires uv. Install it from https://docs.astral.sh/uv/ and retry.\n",
    );
  } else {
    process.stderr.write(`Could not start Flamo setup: ${error.message}\n`);
  }
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exitCode = code ?? 1;
});
