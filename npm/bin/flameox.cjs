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
const command = args[0];
if (command !== "setup" && command !== "upgrade") {
  process.stderr.write(
    "The npm package is the setup bootstrap. Run `npx flameox@latest setup` or `npx flameox@latest upgrade`.\n" +
      "After setup, use flameox through a connected MCP client or the Python CLI.\n",
  );
  process.exit(2);
}

const helper = path.resolve(__dirname, "../lib/jsonc-edit.cjs");
const environment = {
  ...process.env,
  FLAMEOX_SETUP_JSONC_HELPER: helper,
  FLAMEOX_NPM_BOOTSTRAP: "1",
};
const pythonPackage = `flameox==${packageJson.version}`;
const pythonArgs = command === "upgrade" ? ["setup", "--refresh", "--yes", ...args.slice(1)] : args;
const resolveLatest = command === "upgrade" && process.env.FLAMEOX_NPM_UPGRADE_HANDOFF !== "1";
const executable = resolveLatest
  ? process.env.FLAMEOX_NPX_EXECUTABLE || "npx"
  : process.env.FLAMEOX_UV_EXECUTABLE || "uvx";
const childArgs = resolveLatest
  ? ["--yes", "--prefer-online", "flameox@latest", "upgrade", ...args.slice(1)]
  : [
      "--no-config",
      "--no-sources",
      "--refresh-package",
      "flameox",
      "--prerelease",
      "allow",
      "--python",
      "3.12",
      "--from",
      pythonPackage,
      "flameox",
      ...pythonArgs,
    ];
const childEnvironment = resolveLatest
  ? { ...environment, FLAMEOX_NPM_UPGRADE_HANDOFF: "1" }
  : environment;
process.stderr.write(
  resolveLatest
    ? "Resolving the latest flameox bootstrap before upgrading the managed runtime.\n"
    : "Preparing flameox's cached managed Python runtime; this does not add packages to your project.\n",
);
const child = spawn(executable, childArgs, { env: childEnvironment, stdio: "inherit" });

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
      "flameox setup requires uv. Install it from https://docs.astral.sh/uv/ and then " +
        "run `npx flameox@latest setup`.\n",
    );
  } else {
    process.stderr.write(`Could not start flameox setup: ${error.message}\n`);
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
