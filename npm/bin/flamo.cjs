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
    "--python",
    "3.12",
    "--from",
    pythonPackage,
    "flamo",
    ...args,
  ],
  { env: environment, stdio: "inherit" },
);

const signals = ["SIGINT", "SIGTERM", "SIGHUP"];
const signalHandlers = new Map();
for (const signal of signals) {
  const handler = () => child.kill(signal);
  signalHandlers.set(signal, handler);
  process.on(signal, handler);
}

function removeSignalHandlers() {
  for (const [signal, handler] of signalHandlers) {
    process.removeListener(signal, handler);
  }
}

child.on("error", (error) => {
  removeSignalHandlers();
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
  removeSignalHandlers();
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exitCode = code ?? 1;
});
