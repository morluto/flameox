#!/usr/bin/env node

"use strict";

const { spawn } = require("node:child_process");
const packageJson = require("../package.json");

const args = process.argv.slice(2);
if (args.length === 1 && (args[0] === "--version" || args[0] === "-V")) {
  process.stdout.write(`${packageJson.version}\n`);
  process.exit(0);
}
const command = args[0];
if (command !== "setup") {
  process.stderr.write(
    "The npm package exposes setup only. Run `npx flameox@latest setup`.\n" +
      "Use uvx or the Python package for Flameox CLI commands.\n",
  );
  process.exit(2);
}

const environment = {
  ...process.env,
  FLAMEOX_NPM_BOOTSTRAP: "1",
};
const pythonPackage = `flameox==${packageJson.version}`;
const executable = process.env.FLAMEOX_UV_EXECUTABLE || "uvx";
const childArgs = [
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
  ...args,
];
process.stderr.write("Preparing the matching ephemeral Flameox Python runtime.\n");
const child = spawn(executable, childArgs, { env: environment, stdio: "inherit" });

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
