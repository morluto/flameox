"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const test = require("node:test");

const bootstrap = path.resolve(__dirname, "../bin/flameox.cjs");
const packageJson = require("../package.json");

test("bootstrap reports its matching package version", () => {
  const result = spawnSync(process.execPath, [bootstrap, "--version"], { encoding: "utf8" });
  assert.equal(result.status, 0);
  assert.equal(result.stdout, `${packageJson.version}\n`);
});

test("setup reports a missing uv executable with actionable guidance", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "flameox-bootstrap-"));
  try {
    const result = spawnSync(process.execPath, [bootstrap, "setup"], {
      encoding: "utf8",
      env: { ...process.env, FLAMEOX_UV_EXECUTABLE: path.join(directory, "missing-uvx") },
    });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /setup requires uv/);
    assert.match(result.stderr, /npx flameox@latest setup/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("bootstrap rejects the removed upgrade command", () => {
  const result = spawnSync(process.execPath, [bootstrap, "upgrade"], { encoding: "utf8" });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /exposes setup only/);
});
