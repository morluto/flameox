"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const test = require("node:test");

const bootstrap = path.resolve(__dirname, "../bin/flameox.cjs");

function fakeUvx(directory) {
  const executable = path.join(directory, "uvx");
  fs.writeFileSync(
    executable,
    "#!/usr/bin/env node\nprocess.stdout.write(JSON.stringify(process.argv.slice(2)));\n",
    { mode: 0o755 },
  );
  return executable;
}

test("bootstrap reports its matching package version", () => {
  const result = spawnSync(process.execPath, [bootstrap, "--version"], { encoding: "utf8" });
  assert.equal(result.status, 0);
  assert.match(result.stdout, /^0\.2\.0\n$/);
});

test("setup hands off only the 0.2 setup command to uvx", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "flameox-bootstrap-"));
  const result = spawnSync(process.execPath, [bootstrap, "setup"], {
    encoding: "utf8",
    env: { ...process.env, FLAMEOX_UV_EXECUTABLE: fakeUvx(directory) },
  });

  assert.equal(result.status, 0, result.stderr);
  const args = JSON.parse(result.stdout);
  assert.ok(args.includes("flameox==0.2.0"));
  assert.deepEqual(args.slice(-2), ["flameox", "setup"]);
  assert.match(result.stderr, /ephemeral Flameox Python runtime/);
});

test("bootstrap rejects the removed upgrade command", () => {
  const result = spawnSync(process.execPath, [bootstrap, "upgrade"], { encoding: "utf8" });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /exposes setup only/);
});
