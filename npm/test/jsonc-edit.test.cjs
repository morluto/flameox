"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { spawnSync } = require("node:child_process");

const helper = path.resolve(__dirname, "../lib/jsonc-edit.cjs");

function edit(request) {
  const result = spawnSync(process.execPath, [helper], {
    input: JSON.stringify(request),
    encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

test("modifies one nested property without removing comments", () => {
  const source = '{\n  // keep this note\n  "mcp": {},\n}\n';
  const result = edit({
    operation: "modify",
    text: source,
    path: ["mcp", "flameox"],
    remove: false,
    value: { type: "local", command: ["/managed/flameox", "mcp", "serve"] },
  });

  assert.match(result.text, /keep this note/);
  const parsed = edit({ operation: "parse", text: result.text });
  assert.equal(parsed.value.mcp.flameox.type, "local");
});

test("removes only the flameox entry", () => {
  const source =
    '{\n  "mcp": {\n    "other": {"enabled": true},\n    "flameox": {"enabled": true}\n  }\n}\n';
  const result = edit({
    operation: "modify",
    text: source,
    path: ["mcp", "flameox"],
    remove: true,
  });
  const parsed = edit({ operation: "parse", text: result.text });

  assert.deepEqual(parsed.value.mcp, { other: { enabled: true } });
});

test("bootstrap launches the exactly matching Python release", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "flameox-bootstrap-"));
  const capture = path.join(directory, "args.json");
  const fakeUvx = path.join(directory, "uvx");
  fs.writeFileSync(
    fakeUvx,
    [
      "#!/usr/bin/env node",
      '"use strict";',
      'require("node:fs").writeFileSync(process.env.FLAMEOX_CAPTURE, JSON.stringify(process.argv.slice(2)));',
    ].join("\n"),
    { mode: 0o700 },
  );
  const bootstrap = path.resolve(__dirname, "../bin/flameox.cjs");
  const result = spawnSync(process.execPath, [bootstrap, "setup", "--codex", "--dry-run"], {
    encoding: "utf8",
    env: { ...process.env, FLAMEOX_UV_EXECUTABLE: fakeUvx, FLAMEOX_CAPTURE: capture },
  });

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(capture, "utf8")), [
    "--no-config",
    "--no-sources",
    "--prerelease",
    "allow",
    "--python",
    "3.12",
    "--from",
    "flameox==0.1.1",
    "flameox",
    "setup",
    "--codex",
    "--dry-run",
  ]);
});

test(
  "bootstrap terminates when its child exits from a signal",
  { skip: process.platform === "win32" },
  () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "flameox-signal-"));
    const fakeUvx = path.join(directory, "uvx");
    fs.writeFileSync(
      fakeUvx,
      [
        "#!/usr/bin/env node",
        '"use strict";',
        'setTimeout(() => process.kill(process.pid, "SIGTERM"), 20);',
      ].join("\n"),
      { mode: 0o700 },
    );
    const bootstrap = path.resolve(__dirname, "../bin/flameox.cjs");
    const result = spawnSync(process.execPath, [bootstrap, "setup"], {
      encoding: "utf8",
      env: { ...process.env, FLAMEOX_UV_EXECUTABLE: fakeUvx },
      timeout: 5000,
    });

    assert.equal(result.error, undefined);
    assert.equal(result.status, null);
    assert.equal(result.signal, "SIGTERM");
  },
);
