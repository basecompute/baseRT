#!/usr/bin/env node
// Build the npm packages for one release.
//
//   node packaging/npm/build-packages.mjs <bundles-dir> <out-dir> [--engines <names>]
//
// <bundles-dir> holds both release engine bundles for the version in
// bindings/node/package.json. Writes three tarballs to <out-dir>, ready for
// `npm publish <tarball>`:
//
//   basecompute-basert-darwin-arm64-<v>.tgz      engine for Apple Silicon
//   basecompute-basert-linux-arm64-cuda-<v>.tgz  engine for Linux/arm64 CUDA
//   basecompute-basert-<v>.tgz                   bindings + `basert` command
//
// The engine is split out so each machine downloads only its own (esbuild and
// biome ship the same way): the main package lists both engine packages as
// optionalDependencies, and npm skips the one whose os/cpu don't match.
//
// --engines names the engine packages this release publishes, comma-separated
// (default: all, `basert-darwin-arm64,basert-linux-arm64-cuda`). Every engine
// tarball is still built, for testing, but the main package depends only on
// the named ones: one that never reaches the registry must not be among its
// optionalDependencies. publish.yml publishes all of them.
//
// bindings/node is compiled in a staging copy and its package.json rewritten
// there, so the published package gets its own name and metadata while the
// source folder stays as it is for development.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
// Only the public source tree is packaged. The engine's private checkout,
// which has tools/sync_public.py, makes one with packaging/public-tree.sh.
if (fs.existsSync(path.join(root, "tools/sync_public.py"))) {
  console.error("run this from the tree packaging/public-tree.sh builds, not the internal checkout");
  process.exit(1);
}
const argv = process.argv.slice(2);
const flag = argv.indexOf("--engines");
const engines = flag === -1 ? null : (argv.splice(flag, 2)[1] || "").split(",").filter(Boolean);
const [bundlesDir, outDir] = argv.map((p) => path.resolve(p));
if (argv.length !== 2 || (engines && engines.length === 0)) {
  console.error("usage: build-packages.mjs <bundles-dir> <out-dir> [--engines <names>]");
  process.exit(2);
}

const SCOPE = "@basecompute";
const REPOSITORY = { type: "git", url: "git+https://github.com/basecompute/baseRT.git" };
const HOMEPAGE = "https://docs.basecompute.co";

// One entry per release bundle. `name` must match ENGINE_PACKAGES in
// bindings/node/src/index.ts and packaging/npm/basert.js.
const PLATFORMS = [
  {
    bundle: "basert-engine-macos-arm64",
    name: `${SCOPE}/basert-darwin-arm64`,
    description: "BaseRT engine for Apple Silicon (Metal)",
    os: ["darwin"],
    cpu: ["arm64"],
  },
  {
    bundle: "basert-engine-linux-arm64-cuda",
    name: `${SCOPE}/basert-linux-arm64-cuda`,
    description: "BaseRT engine for Linux/arm64 with NVIDIA CUDA (GB10 / DGX Spark)",
    os: ["linux"],
    cpu: ["arm64"],
    libc: ["glibc"],
  },
];

const published = PLATFORMS.filter((p) => !engines || engines.includes(p.name.split("/")[1]));
for (const name of engines || []) {
  if (!PLATFORMS.some((p) => p.name === `${SCOPE}/${name}`)) {
    console.error(`--engines: no engine package ${name}`);
    process.exit(2);
  }
}

const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, stdio: ["ignore", "pipe", "inherit"] }).toString();
const writeJson = (file, obj) => fs.writeFileSync(file, JSON.stringify(obj, null, 2) + "\n");

const source = JSON.parse(fs.readFileSync(path.join(root, "bindings/node/package.json"), "utf8"));
const version = source.version;
const stage = fs.mkdtempSync(path.join(os.tmpdir(), "basert-npm-"));
fs.mkdirSync(outDir, { recursive: true });

function pack(dir) {
  const [{ filename }] = JSON.parse(run("npm", ["pack", "--json", "--pack-destination", outDir], dir));
  console.log(`packed ${filename}`);
}

try {
  // --- engine packages -------------------------------------------------------
  for (const p of PLATFORMS) {
    const bundle = path.join(bundlesDir, `${p.bundle}-${version}.tar.gz`);
    if (!fs.existsSync(bundle)) {
      throw new Error(`missing ${bundle} (bindings/node is at ${version}; has release v${version} been published?)`);
    }
    const dir = path.join(stage, p.name.split("/")[1]);
    run(path.join(root, "packaging/stage-engine.sh"), [bundle, path.join(dir, "engine")]);
    fs.copyFileSync(path.join(dir, "engine/LICENSE"), path.join(dir, "LICENSE"));
    fs.writeFileSync(
      path.join(dir, "README.md"),
      `# ${p.name}\n\n${p.description}. Installed automatically by ` +
        `[\`${SCOPE}/basert\`](https://www.npmjs.com/package/${SCOPE}/basert); ` +
        `don't depend on it directly.\n`
    );
    writeJson(path.join(dir, "package.json"), {
      name: p.name,
      version,
      description: p.description,
      license: "SEE LICENSE IN LICENSE",
      repository: REPOSITORY,
      homepage: HOMEPAGE,
      os: p.os,
      cpu: p.cpu,
      ...(p.libc && { libc: p.libc }),
      files: ["engine/", "LICENSE", "README.md"],
      preferUnplugged: true,
    });
    pack(dir);
  }

  // --- main package: compile the bindings in a staging copy -----------------
  // The copy's package.json leaves out os, cpu and libc. Those say where the
  // bindings install (darwin/arm64), and npm ci refuses a project whose own
  // platforms exclude the machine it runs on, such as publish.yml's Linux x64
  // build runner. Compiling needs none of them; the main package below states
  // its own.
  const build = path.join(stage, "build");
  fs.mkdirSync(build);
  for (const f of ["src", "tsconfig.json", "package-lock.json"]) {
    fs.cpSync(path.join(root, "bindings/node", f), path.join(build, f), { recursive: true });
  }
  const buildable = { ...source };
  for (const key of ["os", "cpu", "libc"]) delete buildable[key];
  writeJson(path.join(build, "package.json"), buildable);
  run("npm", ["ci", "--ignore-scripts", "--no-audit", "--no-fund"], build);
  run("npx", ["tsc"], build);

  const main = path.join(stage, "basert");
  fs.mkdirSync(path.join(main, "bin"), { recursive: true });
  fs.cpSync(path.join(build, "dist"), path.join(main, "dist"), {
    recursive: true,
    filter: (f) => !f.includes("__tests__"),
  });
  fs.copyFileSync(path.join(here, "basert.js"), path.join(main, "bin/basert.js"));
  fs.chmodSync(path.join(main, "bin/basert.js"), 0o755);
  fs.copyFileSync(path.join(here, "README.md"), path.join(main, "README.md"));
  fs.copyFileSync(path.join(root, "LICENSE"), path.join(main, "LICENSE"));
  fs.copyFileSync(path.join(root, "NOTICE"), path.join(main, "NOTICE"));
  writeJson(path.join(main, "package.json"), {
    name: `${SCOPE}/basert`,
    version,
    description: "Fast local LLM inference on Apple Silicon and NVIDIA GB10: the basert CLI and Node.js bindings",
    license: "Apache-2.0",
    repository: { ...REPOSITORY, directory: "bindings/node" },
    homepage: HOMEPAGE,
    keywords: ["llm", "inference", "apple-silicon", "metal", "cuda", "openai"],
    main: source.main,
    types: source.types,
    bin: { basert: "bin/basert.js" },
    engines: source.engines,
    os: ["darwin", "linux"],
    cpu: ["arm64"],
    dependencies: source.dependencies,
    optionalDependencies: Object.fromEntries(published.map((p) => [p.name, version])),
    files: ["dist/**/*.js", "dist/**/*.d.ts", "bin/", "NOTICE"],
  });
  pack(main);
} finally {
  fs.rmSync(stage, { recursive: true, force: true });
}
