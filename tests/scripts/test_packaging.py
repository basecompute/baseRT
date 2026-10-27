#!/usr/bin/env python3
"""Unit tests for packaging/stage-engine.sh, packaging/check_engine_links.py and
the npm launcher and package build (packaging/npm/).

The bundles here are synthetic: a library is a few bytes of text, and the
Mach-O and ELF programs carry only the load commands the link check reads.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "packaging" / "stage-engine.sh"
BASERT_JS = ROOT / "packaging" / "npm" / "basert.js"
sys.path.insert(0, str(ROOT / "packaging"))
from check_engine_links import missing_links, missing_own_names  # noqa: E402


def macho(deps: list[str], own_id: str | None = None) -> bytes:
    """A 64-bit Mach-O with an LC_LOAD_DYLIB per dep (and LC_ID_DYLIB when given)."""
    cmds = [(0xD, own_id)] if own_id else []
    cmds += [(0xC, d) for d in deps]
    body = b""
    for cmd, name in cmds:
        s = name.encode() + b"\0"
        size = (24 + len(s) + 7) // 8 * 8
        body += struct.pack("<6I", cmd, size, 24, 2, 0x10000, 0x10000) + s.ljust(size - 24, b"\0")
    header = struct.pack("<I2i5I", 0xFEEDFACF, 0x0100000C, 0, 2, len(cmds), len(body), 0, 0)
    return header + body


def elf(needed: list[str], soname: str | None = None) -> bytes:
    """A 64-bit little-endian ELF with one PT_LOAD and a PT_DYNAMIC naming `needed` (and DT_SONAME when given)."""
    strtab, offsets = b"\0", []
    for n in needed + ([soname] if soname else []):
        offsets.append(len(strtab))
        strtab += n.encode() + b"\0"
    str_off = 64 + 2 * 56
    dyn_off = (str_off + len(strtab) + 7) // 8 * 8
    dyn = b"".join(struct.pack("<qQ", 1, o) for o in offsets[: len(needed)])  # DT_NEEDED
    if soname:
        dyn += struct.pack("<qQ", 14, offsets[-1])  # DT_SONAME
    dyn += struct.pack("<qQ", 5, str_off) + struct.pack("<qQ", 0, 0)  # DT_STRTAB, DT_NULL
    size = dyn_off + len(dyn)
    ident = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    ehdr = ident + struct.pack("<HHIQQQIHHHHHH", 3, 0xB7, 1, 0, 64, 0, 0, 64, 56, 2, 64, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, size, size, 0x1000)
    dynamic = struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dyn), len(dyn), 8)
    return (ehdr + load + dynamic + strtab).ljust(dyn_off, b"\0") + dyn


def bundle(path: Path, files: dict[str, bytes], links: dict[str, str]) -> None:
    """Write a flat release-style tarball: `files` (in order) and symlinks."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        for name, target in links.items():
            info = tarfile.TarInfo(f"./{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)


class StageEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def stage(self, files: dict[str, bytes], links: dict[str, str]) -> tuple[subprocess.CompletedProcess, Path]:
        tarball = self.tmp / "bundle.tar.gz"
        bundle(tarball, files, links)
        dest = self.tmp / "engine"
        run = subprocess.run(["sh", str(STAGE), str(tarball), str(dest)], capture_output=True, text=True)
        return run, dest

    def libs(self, dest: Path) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in dest.iterdir() if p.name.startswith("libbaseRT.")}

    def test_the_library_the_unversioned_link_names_ships_alone(self) -> None:
        # The 0.4.1 CUDA bundle: two older libraries from a warm build
        # directory sit beside the release's, written before it.
        run, dest = self.stage(
            {
                "libbaseRT.so.0.2.0": b"stale 0.2.0",
                "libbaseRT.so.0.4.0": b"stale 0.4.0",
                "libbaseRT.so.0.4.1": b"release",
                "basert": elf([]),
            },
            {"libbaseRT.so.0": "libbaseRT.so.0.4.1", "libbaseRT.so": "libbaseRT.so.0"},
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.libs(dest), {"libbaseRT.so": b"release"})

    def test_the_macos_chain_collapses_to_libbaseRT_dylib(self) -> None:
        run, dest = self.stage(
            {"libbaseRT.0.4.1.dylib": b"release", "basert": macho([])},
            {"libbaseRT.0.dylib": "libbaseRT.0.4.1.dylib", "libbaseRT.dylib": "libbaseRT.0.dylib"},
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.libs(dest), {"libbaseRT.dylib": b"release"})
        self.assertFalse((dest / "include").exists())

    def test_the_library_also_ships_under_the_name_it_records(self) -> None:
        # A program linked against the library asks for its install name
        # (SONAME on Linux) at startup, so that name is a second real file.
        lib = macho([], own_id="@rpath/libbaseRT.0.dylib")
        run, dest = self.stage(
            {"libbaseRT.0.4.1.dylib": lib, "basert": macho([])},
            {"libbaseRT.0.dylib": "libbaseRT.0.4.1.dylib", "libbaseRT.dylib": "libbaseRT.0.dylib"},
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.libs(dest), {"libbaseRT.dylib": lib, "libbaseRT.0.dylib": lib})
        self.assertTrue(all(not p.is_symlink() for p in dest.iterdir()))

    def test_the_linux_library_also_ships_under_its_soname(self) -> None:
        so = elf(["libcuda.so.1"], soname="libbaseRT.so.0")
        run, dest = self.stage(
            {"libbaseRT.so.0.2.0": b"stale 0.2.0", "libbaseRT.so.0.4.1": so, "basert": elf([])},
            {"libbaseRT.so.0": "libbaseRT.so.0.4.1", "libbaseRT.so": "libbaseRT.so.0"},
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.libs(dest), {"libbaseRT.so": so, "libbaseRT.so.0": so})

    def test_a_program_needing_a_dropped_name_is_refused(self) -> None:
        # The fully versioned name is dropped; a program that asks for it
        # would not start from a wheel.
        run, _ = self.stage(
            {
                "libbaseRT.0.4.1.dylib": macho([], own_id="@rpath/libbaseRT.0.dylib"),
                "basert": macho([]),
                "basert-tool": macho(["@rpath/libbaseRT.0.4.1.dylib", "/usr/lib/libSystem.B.dylib"]),
            },
            {"libbaseRT.0.dylib": "libbaseRT.0.4.1.dylib", "libbaseRT.dylib": "libbaseRT.0.dylib"},
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("basert-tool needs libbaseRT.0.4.1.dylib", run.stderr)

    def test_a_library_that_calls_itself_something_else_is_refused(self) -> None:
        run, _ = self.stage(
            {"libbaseRT.0.4.1.dylib": macho([], own_id="@rpath/libother.dylib"), "basert": macho([])},
            {"libbaseRT.dylib": "libbaseRT.0.4.1.dylib"},
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("calls itself libother.dylib", run.stderr)

    def test_a_bundle_without_the_unversioned_name_is_refused(self) -> None:
        run, _ = self.stage({"libbaseRT.so.0.4.1": b"release", "basert": elf([])}, {})
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("libbaseRT.dylib or libbaseRT.so", run.stderr)


class EngineLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def write(self, name: str, data: bytes) -> None:
        (self.dir / name).write_bytes(data)

    def test_macho_dependency_on_a_present_name_passes(self) -> None:
        self.write("libbaseRT.dylib", macho([], own_id="@rpath/libbaseRT.0.dylib"))
        self.write("basert-chat", macho(["@rpath/libbaseRT.dylib", "/usr/lib/libc++.1.dylib"]))
        self.assertEqual(missing_links(self.dir), [])

    def test_macho_dependency_on_a_missing_name_fails(self) -> None:
        self.write("libbaseRT.dylib", macho([], own_id="@rpath/libbaseRT.0.dylib"))
        self.write("basert-tool", macho(["@rpath/libbaseRT.0.dylib"]))
        self.assertEqual(missing_links(self.dir), [("basert-tool", "libbaseRT.0.dylib")])

    def test_a_library_id_is_not_a_dependency(self) -> None:
        self.write("libbaseRT.dylib", macho([], own_id="@rpath/libbaseRT.0.dylib"))
        self.assertEqual(missing_links(self.dir), [])

    def test_elf_dependency_on_a_missing_soname_fails(self) -> None:
        self.write("libbaseRT.so", elf(["libcuda.so.1", "libc.so.6"]))
        self.write("basert-tool", elf(["libbaseRT.so.0", "libcublas.so.13"]))
        self.assertEqual(missing_links(self.dir), [("basert-tool", "libbaseRT.so.0")])

    def test_a_library_whose_own_name_is_missing_is_reported(self) -> None:
        # Linked through libbaseRT.dylib, a consumer records libbaseRT.0.dylib.
        self.write("libbaseRT.dylib", macho([], own_id="@rpath/libbaseRT.0.dylib"))
        self.write("libbaseRT.so", elf(["libc.so.6"], soname="libbaseRT.so.0"))
        self.assertEqual(missing_own_names(self.dir), [("libbaseRT.dylib", "libbaseRT.0.dylib"), ("libbaseRT.so", "libbaseRT.so.0")])
        run = subprocess.run([sys.executable, str(ROOT / "packaging" / "check_engine_links.py"), str(self.dir)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertIn("libbaseRT.dylib calls itself libbaseRT.0.dylib", run.stderr)
        self.write("libbaseRT.0.dylib", macho([], own_id="@rpath/libbaseRT.0.dylib"))
        self.write("libbaseRT.so.0", elf(["libc.so.6"], soname="libbaseRT.so.0"))
        self.assertEqual(missing_own_names(self.dir), [])

    def test_own_name_prints_what_a_library_records(self) -> None:
        self.write("libbaseRT.dylib", macho([], own_id="@rpath/libbaseRT.0.dylib"))
        self.write("basert", macho(["@rpath/libbaseRT.dylib"]))
        own = lambda f: subprocess.run(  # noqa: E731
            [sys.executable, str(ROOT / "packaging" / "check_engine_links.py"), "--own-name", str(self.dir / f)],
            capture_output=True, text=True, check=True).stdout
        self.assertEqual(own("libbaseRT.dylib"), "libbaseRT.0.dylib\n")
        self.assertEqual(own("basert"), "")

    def test_elf_system_libraries_are_not_ours_to_ship(self) -> None:
        self.write("basert-chat", elf(["libcuda.so.1", "libcublas.so.13", "libstdc++.so.6"]))
        self.write("LICENSE", b"text")
        self.assertEqual(missing_links(self.dir), [])


class NpmLauncherTests(unittest.TestCase):
    """packaging/npm/basert.js, run by node over a node_modules laid out as npm would."""

    ENGINES = {"darwin-arm64": "basert-darwin-arm64", "linux-arm64": "basert-linux-arm64-cuda"}

    def setUp(self) -> None:
        node = shutil.which("node")
        engine = self.ENGINES.get(f"{sys.platform}-{os.uname().machine.replace('aarch64', 'arm64')}")
        if not node or not engine:
            self.skipTest("needs node on macOS/arm64 or Linux/arm64")
        self.node = node
        self.tmp = Path(tempfile.mkdtemp())
        self.modules = self.tmp / "node_modules" / "@basecompute"
        self.engine_pkg = engine

    def install(self, listed: bool, with_engine: bool = True) -> Path:
        main = self.modules / "basert"
        (main / "bin").mkdir(parents=True)
        shutil.copy(BASERT_JS, main / "bin" / "basert.js")
        deps = {f"@basecompute/{self.engine_pkg}": "0.0.0"} if listed else {}
        (main / "package.json").write_text(json.dumps({"name": "@basecompute/basert", "optionalDependencies": deps}))
        engine = self.modules / self.engine_pkg / "engine"
        if with_engine:
            engine.mkdir(parents=True)
            (engine.parent / "package.json").write_text(json.dumps({"name": f"@basecompute/{self.engine_pkg}"}))
            # Unpacked without executable bits, as some package managers do.
            # A program is known by its header: one named unlike the
            # basert-* tools gets its bit back too.
            for name in ("basert", "basert-chat", "helper"):
                shutil.copy("/usr/bin/true", engine / name)
                os.chmod(engine / name, 0o644)
            (engine / "LICENSE").write_text("text")
            os.chmod(engine / "LICENSE", 0o644)
        return engine

    def launch(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.node, str(self.modules / "basert" / "bin" / "basert.js")], capture_output=True, text=True
        )

    def test_every_program_gets_its_executable_bit_back(self) -> None:
        engine = self.install(listed=True)
        run = self.launch()
        self.assertEqual(run.returncode, 0, run.stderr)
        for name in ("basert", "basert-chat", "helper"):
            self.assertTrue(os.access(engine / name, os.X_OK), name)
        self.assertFalse(os.access(engine / "LICENSE", os.X_OK))

    def test_a_listed_engine_package_missing_says_to_reinstall(self) -> None:
        self.install(listed=True, with_engine=False)
        run = self.launch()
        self.assertEqual(run.returncode, 1)
        self.assertIn("--omit=optional", run.stderr)

    def test_an_engine_this_release_did_not_publish_points_at_install_sh(self) -> None:
        self.install(listed=False, with_engine=False)
        run = self.launch()
        self.assertEqual(run.returncode, 1)
        self.assertIn("this release has no npm engine package", run.stderr)


class NpmBuildTests(unittest.TestCase):
    """packaging/npm/build-packages.mjs, run with npm over synthetic bundles.

    The bindings it compiles say which platforms they install on, and
    publish.yml builds on one they leave out (Linux x64). Here they name a
    platform no test machine is, which `npm ci` refuses unless the build
    compiles a copy without it. The bindings have no dependencies and `npx tsc`
    is stubbed, so nothing is fetched.
    """

    VERSION = "1.2.3"
    FOREIGN = {"os": ["win32"], "cpu": ["ia32"]}

    def setUp(self) -> None:
        node = shutil.which("node")
        if not node or not shutil.which("npm"):
            self.skipTest("needs node and npm")
        self.node = node
        self.tmp = Path(tempfile.mkdtemp())
        v = self.VERSION
        # A public tree with just what the build reads.
        self.root = self.tmp / "tree"
        (self.root / "packaging" / "npm").mkdir(parents=True)
        for name in ("npm/build-packages.mjs", "npm/basert.js", "npm/README.md", "stage-engine.sh", "check_engine_links.py"):
            shutil.copy(ROOT / "packaging" / name, self.root / "packaging" / name)
        (self.root / "LICENSE").write_text("licence")
        (self.root / "NOTICE").write_text("notice")
        bindings = self.root / "bindings" / "node"
        (bindings / "src").mkdir(parents=True)
        (bindings / "src" / "index.ts").write_text("export {};\n")
        (bindings / "tsconfig.json").write_text("{}\n")
        (bindings / "package.json").write_text(
            json.dumps({"name": "@baseRT/node", "version": v, "main": "dist/index.js", **self.FOREIGN})
        )
        (bindings / "package-lock.json").write_text(
            json.dumps(
                {
                    "name": "@baseRT/node",
                    "version": v,
                    "lockfileVersion": 3,
                    "requires": True,
                    "packages": {"": {"name": "@baseRT/node", "version": v, **self.FOREIGN}},
                }
            )
        )
        # Stands in for `npx tsc` compiling src/index.ts.
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        (self.bin / "npx").write_text(
            '#!/bin/sh\n[ "$1" = tsc ] || exit 1\nmkdir -p dist\necho "exports.v = 1;" > dist/index.js\n'
        )
        os.chmod(self.bin / "npx", 0o755)
        self.bundles = self.tmp / "bundles"
        self.bundles.mkdir()
        bundle(
            self.bundles / f"basert-engine-macos-arm64-{v}.tar.gz",
            {f"libbaseRT.{v}.dylib": macho([], own_id="@rpath/libbaseRT.0.dylib"), "basert": macho([]), "LICENSE": b"engine"},
            {"libbaseRT.0.dylib": f"libbaseRT.{v}.dylib", "libbaseRT.dylib": "libbaseRT.0.dylib"},
        )
        bundle(
            self.bundles / f"basert-engine-linux-arm64-cuda-{v}.tar.gz",
            {f"libbaseRT.so.{v}": elf([], soname="libbaseRT.so.0"), "basert": elf([]), "LICENSE": b"engine"},
            {"libbaseRT.so.0": f"libbaseRT.so.{v}", "libbaseRT.so": "libbaseRT.so.0"},
        )

    def build(self, *args: str) -> tuple[subprocess.CompletedProcess, Path]:
        out = self.tmp / "out"
        env = dict(
            os.environ,
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            npm_config_cache=str(self.tmp / "npm-cache"),
            npm_config_update_notifier="false",
        )
        script = self.root / "packaging" / "npm" / "build-packages.mjs"
        run = subprocess.run(
            [self.node, str(script), str(self.bundles), str(out), *args], capture_output=True, text=True, env=env
        )
        return run, out

    def test_the_bindings_compile_on_a_machine_they_do_not_install_on(self) -> None:
        run, out = self.build("--engines", "basert-darwin-arm64")
        self.assertEqual(run.returncode, 0, run.stderr)
        v = self.VERSION
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            [f"basecompute-basert-{v}.tgz", f"basecompute-basert-darwin-arm64-{v}.tgz", f"basecompute-basert-linux-arm64-cuda-{v}.tgz"],
        )
        with tarfile.open(out / f"basecompute-basert-{v}.tgz") as tar:
            main = json.load(tar.extractfile("package/package.json"))
        self.assertEqual((main["os"], main["cpu"]), (["darwin", "linux"], ["arm64"]))
        # Only the engine package this release publishes.
        self.assertEqual(main["optionalDependencies"], {"@basecompute/basert-darwin-arm64": v})
        # The bindings' own folder is left as it was.
        source = json.loads((self.root / "bindings" / "node" / "package.json").read_text())
        self.assertEqual(source["os"], self.FOREIGN["os"])


if __name__ == "__main__":
    unittest.main()
