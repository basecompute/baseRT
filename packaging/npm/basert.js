#!/usr/bin/env node
// The `basert` command installed by `npm install -g @basecompute/basert`.
//
// Copied into the package as bin/basert.js by packaging/npm/build-packages.mjs.
// npm puts the prebuilt engine in whichever optional dependency matches this
// machine; this runs the `basert` launcher from that package's engine/ folder,
// where it finds the `basert-*` runtime tools and kernel files beside it.
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ENGINE_PACKAGES = {
  "darwin-arm64": "@basecompute/basert-darwin-arm64",
  "linux-arm64": "@basecompute/basert-linux-arm64-cuda",
};

const platform = `${process.platform}-${process.arch}`;
const pkg = ENGINE_PACKAGES[platform];
if (!pkg) {
  console.error(
    `basert: no prebuilt engine for ${platform}. BaseRT ships macOS/arm64 (Apple Silicon) and Linux/arm64 (NVIDIA CUDA).`
  );
  process.exit(1);
}

let exe;
try {
  exe = path.join(path.dirname(require.resolve(`${pkg}/package.json`)), "engine", "basert");
} catch {
  // A release lists only the engine packages it publishes (see
  // build-packages.mjs --engines); this machine's may not be one of them.
  const published = pkg in (require("../package.json").optionalDependencies || {});
  console.error(
    published
      ? `basert: the engine package ${pkg} is not installed. Reinstall without --no-optional / --omit=optional.`
      : `basert: this release has no npm engine package for ${platform}. Install the engine with install.sh or Homebrew; the Node.js bindings find it there.`
  );
  process.exit(1);
}

// Some package managers drop the executable bit when unpacking; restore it on
// every program in the folder. Programs are known by their Mach-O or ELF
// header, not their name, so one the bundle gains later is covered too.
function isProgram(file) {
  let fd;
  try {
    fd = fs.openSync(file, "r");
    const head = Buffer.alloc(4);
    if (fs.readSync(fd, head, 0, 4, 0) < 4) return false;
    // 64-bit Mach-O, universal Mach-O, ELF.
    return head.readUInt32LE(0) === 0xfeedfacf || head.readUInt32BE(0) === 0xcafebabe || head.readUInt32BE(0) === 0x7f454c46;
  } catch {
    return false; // a directory, or unreadable
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}
try {
  fs.accessSync(exe, fs.constants.X_OK);
} catch {
  const dir = path.dirname(exe);
  for (const f of fs.readdirSync(dir)) {
    if (isProgram(path.join(dir, f))) fs.chmodSync(path.join(dir, f), 0o755);
  }
}

const result = spawnSync(exe, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
  console.error(`basert: failed to start ${exe}: ${result.error.message}`);
  process.exit(1);
}
if (result.signal) process.kill(process.pid, result.signal);
process.exit(result.status ?? 1);
