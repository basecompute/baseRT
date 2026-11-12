#!/usr/bin/env python3
"""Unit tests for packaging/stage-engine.sh, packaging/check_engine_links.py,
the wheel platform tag and build (packaging/python/), the npm launcher and
package build (packaging/npm/) and the Homebrew formula
(packaging/update-formula.sh); in the open repo also for install.sh and the
publish workflow's lists of what a release builds.

The bundles here are synthetic: a library is a few bytes of text, and the
Mach-O and ELF programs carry only the load commands, CPU and version markers
the scripts read.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "packaging" / "stage-engine.sh"
BASERT_JS = ROOT / "packaging" / "npm" / "basert.js"
PLATFORM_TAG = ROOT / "packaging" / "python" / "platform_tag.py"
BUILD_WHEEL = ROOT / "packaging" / "python" / "build-wheel.sh"
FORMULA = ROOT / "packaging" / "update-formula.sh"
# The open repo's own files: install.sh and the workflows are not in the
# engine repository's copy of this tree.
INSTALL_SH = ROOT / "install.sh"
WORKFLOWS = ROOT / ".github" / "workflows"
IN_ENGINE_REPO = (ROOT / "tools" / "sync_public.py").exists()
sys.path.insert(0, str(ROOT / "packaging"))
from check_engine_links import missing_links, missing_own_names  # noqa: E402


AARCH64, X86_64, AMDGPU = 0xB7, 0x3E, 0xE0  # ELF e_machine
# A ROCm kernel file for two GPUs: a clang offload bundle, not an ELF file.
OFFLOAD_BUNDLE = b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", 0)
FATBIN = struct.pack("<I", 0xBA55ED50) + b"cuda kernels"


def glibc(minor: int) -> bytes:
    """The symbol version string an ELF file needing glibc 2.<minor> carries."""
    return f"GLIBC_2.{minor}".encode() + b"\0"


def macho(deps: list[str], own_id: str | None = None, minos: tuple[int, int] | None = None) -> bytes:
    """A 64-bit Mach-O with an LC_LOAD_DYLIB per dep (and LC_ID_DYLIB, LC_BUILD_VERSION when given)."""
    body = b""
    if minos:
        body += struct.pack("<6I", 0x32, 24, 1, minos[0] << 16 | minos[1] << 8, 0, 0)
    cmds = [(0xD, own_id)] if own_id else []
    cmds += [(0xC, d) for d in deps]
    for cmd, name in cmds:
        s = name.encode() + b"\0"
        size = (24 + len(s) + 7) // 8 * 8
        body += struct.pack("<6I", cmd, size, 24, 2, 0x10000, 0x10000) + s.ljust(size - 24, b"\0")
    header = struct.pack("<I2i5I", 0xFEEDFACF, 0x0100000C, 0, 2, len(cmds) + bool(minos), len(body), 0, 0)
    return header + body


def elf(needed: list[str], soname: str | None = None, machine: int = AARCH64, tail: bytes = b"") -> bytes:
    """A 64-bit little-endian ELF for `machine` with one PT_LOAD and a PT_DYNAMIC
    naming `needed` (and DT_SONAME when given), followed by `tail`."""
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
    ehdr = ident + struct.pack("<HHIQQQIHHHHHH", 3, machine, 1, 0, 64, 0, 0, 64, 56, 2, 64, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, size, size, 0x1000)
    dynamic = struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dyn), len(dyn), 8)
    return (ehdr + load + dynamic + strtab).ljust(dyn_off, b"\0") + dyn + tail


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


# What each release bundle is called and which wheel its engine makes.
BUNDLES = {
    "basert-engine-macos-arm64": "macosx_15_0_arm64",
    "basert-engine-linux-arm64-cuda": "manylinux_2_38_aarch64",
    "basert-engine-linux-x64-rocm": "manylinux_2_39_x86_64",
}


def release(folder: Path, v: str, rocm: bool = True) -> None:
    """Write a release's engine bundles (and their .sha256 files) to `folder`,
    shaped like the real ones: the versioned library chain, the launcher, a
    tool, the kernel file, the headers, LICENSE and NOTICE."""
    folder.mkdir(parents=True, exist_ok=True)
    common = {"LICENSE": b"engine licence", "NOTICE": b"engine notice", "include/baseRT/baseRT.h": b"/* api */"}
    mac = {
        f"libbaseRT.{v}.dylib": macho([], own_id="@rpath/libbaseRT.0.dylib", minos=(15, 0)),
        "basert": macho([], minos=(14, 0)),
        "basert-chat": macho(["@rpath/libbaseRT.0.dylib", "/usr/lib/libc++.1.dylib"], minos=(15, 0)),
        "baseRT.metallib": b"MTLB",
    }
    so = {"libbaseRT.so.0": f"libbaseRT.so.{v}", "libbaseRT.so": "libbaseRT.so.0"}
    cuda = {
        f"libbaseRT.so.{v}": elf(["libcublas.so.13", "libcuda.so.1"], soname="libbaseRT.so.0", tail=glibc(38)),
        "basert": elf(["libc.so.6"], tail=glibc(34)),
        "basert-chat": elf(["libbaseRT.so.0", "libc.so.6"], tail=glibc(38)),
        "baseRT_cuda_kernels.fatbin": FATBIN,
    }
    hip = ["libamdhip64.so.7", "libhipblas.so.3", "libhipblaslt.so.1"]
    rocm_files = {
        f"libbaseRT.so.{v}": elf(hip, soname="libbaseRT.so.0", machine=X86_64, tail=glibc(38)),
        "basert": elf(["libc.so.6"], machine=X86_64, tail=glibc(39)),
        "basert-chat": elf(["libbaseRT.so.0", "libc.so.6"], machine=X86_64, tail=glibc(38)),
        "baseRT_rocm_kernels.hsaco": OFFLOAD_BUNDLE,
    }
    contents = {
        "basert-engine-macos-arm64": ({**mac, **common}, {"libbaseRT.0.dylib": f"libbaseRT.{v}.dylib", "libbaseRT.dylib": "libbaseRT.0.dylib"}),
        "basert-engine-linux-arm64-cuda": ({**cuda, **common}, so),
        "basert-engine-linux-x64-rocm": ({**rocm_files, **common}, so),
    }
    for name, (files, links) in contents.items():
        if name.endswith("rocm") and not rocm:
            continue
        tarball = folder / f"{name}-{v}.tar.gz"
        bundle(tarball, files, links)
        sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
        (folder / f"{tarball.name}.sha256").write_text(f"{sha}  {tarball.name}\n")


class StageEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def stage(
        self, files: dict[str, bytes], links: dict[str, str], *extra: str
    ) -> tuple[subprocess.CompletedProcess, Path]:
        tarball = self.tmp / "bundle.tar.gz"
        bundle(tarball, files, links)
        dest = self.tmp / "engine"
        run = subprocess.run(["sh", str(STAGE), str(tarball), str(dest), *extra], capture_output=True, text=True)
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

    def test_the_package_manager_that_installs_it_is_named_beside_the_launcher(self) -> None:
        # `basert self update` reads it to leave the copy to that manager.
        files = {"libbaseRT.0.4.1.dylib": b"release", "basert": macho([])}
        links = {"libbaseRT.dylib": "libbaseRT.0.4.1.dylib"}
        run, dest = self.stage(files, links, "pip: pip install --upgrade basert")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual((dest / "installed-by").read_text(), "pip: pip install --upgrade basert\n")
        shutil.rmtree(dest)
        run, dest = self.stage(files, links)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse((dest / "installed-by").exists())
        shutil.rmtree(dest)
        run, _ = self.stage(files, links, "pip install --upgrade basert")
        self.assertEqual(run.returncode, 2)
        self.assertIn("<manager>: <command>", run.stderr)

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

    ENGINES = {
        "darwin-arm64": "basert-darwin-arm64",
        "linux-arm64": "basert-linux-arm64-cuda",
        "linux-x64": "basert-linux-x64-rocm",
    }

    def setUp(self) -> None:
        node = shutil.which("node")
        machine = {"aarch64": "arm64", "x86_64": "x64"}.get(os.uname().machine, os.uname().machine)
        engine = self.ENGINES.get(f"{sys.platform}-{machine}")
        if not node or not engine:
            self.skipTest("needs node on macOS/arm64, Linux/arm64 or Linux/x64")
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
        release(self.bundles, v)

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

    def manifest(self, tgz: Path) -> dict:
        with tarfile.open(tgz) as tar:
            return json.load(tar.extractfile("package/package.json"))

    def test_the_bindings_compile_on_a_machine_they_do_not_install_on(self) -> None:
        run, out = self.build("--engines", "basert-darwin-arm64")
        self.assertEqual(run.returncode, 0, run.stderr)
        v = self.VERSION
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            [
                f"basecompute-basert-{v}.tgz",
                f"basecompute-basert-darwin-arm64-{v}.tgz",
                f"basecompute-basert-linux-arm64-cuda-{v}.tgz",
                f"basecompute-basert-linux-x64-rocm-{v}.tgz",
            ],
        )
        main = self.manifest(out / f"basecompute-basert-{v}.tgz")
        # Every platform this release has an engine for: one left out of
        # --engines is still there through install.sh or Homebrew.
        self.assertEqual((main["os"], main["cpu"]), (["darwin", "linux"], ["arm64", "x64"]))
        # Only the engine package this release publishes.
        self.assertEqual(main["optionalDependencies"], {"@basecompute/basert-darwin-arm64": v})
        # The engine names npm as what updates it, for `basert self update`.
        with tarfile.open(out / f"basecompute-basert-darwin-arm64-{v}.tgz") as tar:
            said = tar.extractfile("package/engine/installed-by").read().decode()
        self.assertEqual(said, "npm: npm install --global @basecompute/basert@latest\n")
        # The bindings' own folder is left as it was.
        source = json.loads((self.root / "bindings" / "node" / "package.json").read_text())
        self.assertEqual(source["os"], self.FOREIGN["os"])

    def test_the_rocm_engine_is_an_x64_linux_package(self) -> None:
        run, out = self.build()
        self.assertEqual(run.returncode, 0, run.stderr)
        v = self.VERSION
        tgz = out / f"basecompute-basert-linux-x64-rocm-{v}.tgz"
        rocm = self.manifest(tgz)
        self.assertEqual(
            (rocm["name"], rocm["os"], rocm["cpu"], rocm["libc"]),
            ("@basecompute/basert-linux-x64-rocm", ["linux"], ["x64"], ["glibc"]),
        )
        with tarfile.open(tgz) as tar:
            engine = {m.name.split("/", 2)[2] for m in tar.getmembers() if m.name.startswith("package/engine/")}
        self.assertTrue({"baseRT_rocm_kernels.hsaco", "libbaseRT.so", "libbaseRT.so.0", "basert"} <= engine, engine)
        main = self.manifest(out / f"basecompute-basert-{v}.tgz")
        self.assertEqual(main["cpu"], ["arm64", "x64"])
        self.assertEqual(
            main["optionalDependencies"],
            {f"@basecompute/{name}": v for name in ("basert-darwin-arm64", "basert-linux-arm64-cuda", "basert-linux-x64-rocm")},
        )

    def test_a_release_without_the_rocm_bundle_has_no_x64_package(self) -> None:
        # release.yml ships a release without it when its AMD runner is off.
        (self.bundles / f"basert-engine-linux-x64-rocm-{self.VERSION}.tar.gz").unlink()
        run, out = self.build()
        self.assertEqual(run.returncode, 0, run.stderr)
        v = self.VERSION
        self.assertIn("basert-linux-x64-rocm is not built", run.stdout)
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            [f"basecompute-basert-{v}.tgz", f"basecompute-basert-darwin-arm64-{v}.tgz", f"basecompute-basert-linux-arm64-cuda-{v}.tgz"],
        )
        main = self.manifest(out / f"basecompute-basert-{v}.tgz")
        self.assertEqual((main["os"], main["cpu"]), (["darwin", "linux"], ["arm64"]))
        self.assertNotIn("@basecompute/basert-linux-x64-rocm", main["optionalDependencies"])

    def test_the_cuda_bundle_is_never_optional(self) -> None:
        (self.bundles / f"basert-engine-linux-arm64-cuda-{self.VERSION}.tar.gz").unlink()
        run, _ = self.build()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("missing", run.stderr)

    def test_naming_the_rocm_package_needs_its_bundle(self) -> None:
        (self.bundles / f"basert-engine-linux-x64-rocm-{self.VERSION}.tar.gz").unlink()
        run, _ = self.build("--engines", "basert-darwin-arm64,basert-linux-x64-rocm")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(f"basert-engine-linux-x64-rocm-{self.VERSION}.tar.gz", run.stderr)



class PlatformTagTests(unittest.TestCase):
    """packaging/python/platform_tag.py: the wheel tag a staged engine folder gets."""

    def tag(self, files: dict[str, bytes]) -> subprocess.CompletedProcess:
        engine = Path(tempfile.mkdtemp())
        for name, data in files.items():
            (engine / name).write_bytes(data)
        return subprocess.run([sys.executable, str(PLATFORM_TAG), str(engine)], capture_output=True, text=True)

    def test_the_cuda_engine_is_aarch64(self) -> None:
        run = self.tag({
            "libbaseRT.so": elf(["libcublas.so.13"], soname="libbaseRT.so.0", tail=glibc(38)),
            "basert": elf([], tail=glibc(34)),
            "baseRT_cuda_kernels.fatbin": FATBIN,
        })
        self.assertEqual((run.returncode, run.stdout), (0, "manylinux_2_38_aarch64\n"), run.stderr)

    def test_the_rocm_engine_is_x86_64(self) -> None:
        # Until 0.5.0 every ELF engine was tagged aarch64, so this one made a
        # second aarch64 wheel, and publish.yml's Linux test installed both.
        run = self.tag({
            "libbaseRT.so": elf(["libamdhip64.so.7"], soname="libbaseRT.so.0", machine=X86_64, tail=glibc(38)),
            "basert": elf([], machine=X86_64, tail=glibc(39)),
            "baseRT_rocm_kernels.hsaco": OFFLOAD_BUNDLE,
        })
        self.assertEqual((run.returncode, run.stdout), (0, "manylinux_2_39_x86_64\n"), run.stderr)

    def test_a_gpu_code_object_says_nothing_about_the_cpu(self) -> None:
        # A kernel file built for one GPU is an AMDGPU ELF code object.
        run = self.tag({
            "libbaseRT.so": elf([], machine=X86_64, tail=glibc(39)),
            "baseRT_rocm_kernels.hsaco": elf([], machine=AMDGPU),
        })
        self.assertEqual((run.returncode, run.stdout), (0, "manylinux_2_39_x86_64\n"), run.stderr)

    def test_binaries_for_two_cpus_are_refused(self) -> None:
        run = self.tag({"libbaseRT.so": elf([], tail=glibc(38)), "basert": elf([], machine=X86_64, tail=glibc(39))})
        self.assertEqual(run.returncode, 1)
        self.assertIn("more than one CPU (libbaseRT.so for aarch64, basert for x86_64)", run.stderr)

    def test_an_elf_file_for_another_cpu_is_refused(self) -> None:
        run = self.tag({"libbaseRT.so": elf([], machine=0x28, tail=glibc(38))})  # 32-bit Arm
        self.assertEqual(run.returncode, 1)
        self.assertIn("machine 0x28", run.stderr)

    def test_the_macos_engine_takes_the_highest_minimum_os(self) -> None:
        run = self.tag({"libbaseRT.dylib": macho([], minos=(15, 0)), "basert": macho([], minos=(14, 0))})
        self.assertEqual((run.returncode, run.stdout), (0, "macosx_15_0_arm64\n"), run.stderr)


class FormulaTests(unittest.TestCase):
    """packaging/update-formula.sh, from a release's .sha256 files."""

    V = "1.2.3"

    def formula(self, rocm: bool) -> tuple[subprocess.CompletedProcess, str]:
        tmp = Path(tempfile.mkdtemp())
        release(tmp / "bundles", self.V, rocm=rocm)
        for tarball in (tmp / "bundles").glob("*.tar.gz"):
            tarball.unlink()  # the formula needs only the checksums
        run = subprocess.run(["sh", str(FORMULA), self.V, str(tmp / "bundles"), str(tmp / "out")], capture_output=True, text=True)
        rb = tmp / "out" / "basert.rb"
        return run, rb.read_text() if rb.exists() else ""

    @staticmethod
    def block(rb: str, opening: str) -> str:
        """The lines of a top-level `opening do` block of the class."""
        start = rb.index(f"\n  {opening} do\n")
        return rb[start : rb.index("\n  end\n", start)]

    def url(self, bundle_name: str) -> str:
        return f'url "https://github.com/basecompute/baseRT/releases/download/v{self.V}/{bundle_name}-{self.V}.tar.gz"'

    def test_linux_on_intel_gets_the_rocm_bundle(self) -> None:
        run, rb = self.formula(rocm=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        linux = self.block(rb, "on_linux")
        self.assertIn(f"    if Hardware::CPU.intel?\n      {self.url('basert-engine-linux-x64-rocm')}\n", linux)
        self.assertIn(f"    if Hardware::CPU.arm?\n      {self.url('basert-engine-linux-arm64-cuda')}\n", linux)
        self.assertNotIn("depends_on arch:", linux)
        # macOS stays Apple Silicon only.
        self.assertIn("depends_on arch: :arm64", self.block(rb, "on_macos"))
        self.assertNotIn("\n  depends_on arch:", rb)
        self.assertIn("AMD Strix", re.search(r'^  desc "(.*)"$', rb, re.M).group(1))
        for gfx in ("gfx1151", "gfx1150"):
            self.assertIn(f"sudo apt install amdrocm-blas10.0-{gfx}", rb)
        self.assertIn("CUDA 13", rb)

    def test_without_the_rocm_bundle_it_stays_arm64_only(self) -> None:
        run, rb = self.formula(rocm=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("depends_on arch: :arm64", self.block(rb, "on_linux"))
        self.assertIn("depends_on arch: :arm64", self.block(rb, "on_macos"))
        self.assertNotIn("Hardware::CPU.intel?", rb)
        self.assertNotIn("rocm", rb.lower())
        self.assertNotIn("AMD", rb)

    def test_each_checksum_is_the_one_its_sha256_file_names(self) -> None:
        run, rb = self.formula(rocm=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        urls = re.findall(r'url "[^"]*/([^/"]+)"\n\s+sha256 "([0-9a-f]{64})"', rb)
        tmp_shas = {}
        tmp = Path(tempfile.mkdtemp())
        release(tmp, self.V)
        for f in tmp.glob("*.sha256"):
            sha, name = f.read_text().split()
            tmp_shas[name] = sha
        self.assertEqual(len(urls), 3)
        self.assertEqual(dict(urls), tmp_shas)

    def test_a_bad_rocm_checksum_is_refused(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        release(tmp, self.V)
        (tmp / f"basert-engine-linux-x64-rocm-{self.V}.tar.gz.sha256").write_text("not-a-checksum  x\n")
        run = subprocess.run(["sh", str(FORMULA), self.V, str(tmp), str(tmp / "out")], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("bad checksum", run.stderr)


@unittest.skipIf(IN_ENGINE_REPO, "wheels are built from the public tree (packaging/public-tree.sh)")
class WheelBuildTests(unittest.TestCase):
    """packaging/python/build-wheel.sh over a whole release: one wheel per
    bundle, each tagged for the platform its engine runs on. Needs `build` and
    `wheel` (ci.yml and publish.yml install them) and the network: the build
    fetches its setuptools."""

    def setUp(self) -> None:
        missing = [m for m in ("build", "wheel") if importlib.util.find_spec(m) is None]
        if missing:
            self.skipTest(f"needs {' and '.join(missing)} (pip install build wheel)")
        self.tmp = Path(tempfile.mkdtemp())
        init = (ROOT / "bindings" / "python" / "baseRT" / "__init__.py").read_text()
        self.v = re.search(r'^__version__ = "(.*)"$', init, re.M).group(1)

    def test_each_bundle_makes_the_wheel_for_its_platform(self) -> None:
        release(self.tmp / "bundles", self.v)
        out = self.tmp / "wheels"
        # build-wheel.sh runs python3: this one, which has build and wheel.
        env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
        for name in BUNDLES:
            run = subprocess.run(
                ["sh", str(BUILD_WHEEL), str(self.tmp / "bundles" / f"{name}-{self.v}.tar.gz"), str(out)],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            sorted(f"basert-{self.v}-py3-none-{tag}.whl" for tag in BUNDLES.values()),
        )
        with zipfile.ZipFile(out / f"basert-{self.v}-py3-none-manylinux_2_39_x86_64.whl") as whl:
            engine = {n.split("/", 2)[2] for n in whl.namelist() if n.startswith("baseRT/_engine/")}
        self.assertTrue({"baseRT_rocm_kernels.hsaco", "libbaseRT.so", "libbaseRT.so.0", "basert", "installed-by"} <= engine, engine)


@unittest.skipUnless(INSTALL_SH.is_file(), "install.sh is in the open repo only")
class InstallScriptTests(unittest.TestCase):
    """install.sh, run with stand-ins for uname and curl and a folder standing
    in for / (BASERT_INSTALL_TEST_ROOT) where the AMD GPU and ROCm checks look."""

    V = "1.2.3"
    UNAME = '#!/bin/sh\ncase "$1" in -s) echo "$STUB_OS" ;; -m) echo "$STUB_ARCH" ;; *) exit 1 ;; esac\n'
    # The release API answers from a file; a download copies the asset.
    CURL = """#!/bin/sh
out="" url=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out=$2; shift 2 ;; -*) shift ;; *) url=$1; shift ;; esac
done
echo "$url" >> "$STUB_LOG"
case "$url" in
  https://api.github.com/*) cat "$STUB_RELEASE" ;;
  *) cp "$STUB_ASSETS/${url##*/}" "$out" ;;
esac
"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, text in (("uname", self.UNAME), ("curl", self.CURL)):
            (self.bin / name).write_text(text)
            os.chmod(self.bin / name, 0o755)
        # Bundles whose programs are scripts, so the installer can run them
        # here: `self install` and `superfluid` are refused, as by a launcher
        # older than 0.4.3, and the bundle is unpacked as it is.
        self.assets = self.tmp / "assets"
        self.assets.mkdir()
        refuse = b"#!/bin/sh\nexit 1\n"
        urls = []
        for name in BUNDLES:
            tarball = self.assets / f"{name}-{self.V}.tar.gz"
            bundle(tarball, {"basert": refuse, "basert-chat": refuse, f"lib-{name}": b"engine", "LICENSE": b"licence"}, {})
            for f in (tarball.name, f"{tarball.name}.sha256"):
                urls.append(f"https://github.com/basecompute/baseRT/releases/download/v{self.V}/{f}")
        self.api = self.tmp / "release.json"
        self.api.write_text(json.dumps({"tag_name": f"v{self.V}", "assets": [{"browser_download_url": u} for u in urls]}))
        self.log = self.tmp / "curl.log"
        self.log.write_text("")
        self.dest = self.tmp / "basert"

    def amd(self, *gfx_target_versions: int, kfd: bool = True) -> None:
        """An amdgpu driver's view: /dev/kfd and one topology node per version (0 = a CPU)."""
        if kfd:
            (self.root / "dev").mkdir(exist_ok=True)
            (self.root / "dev" / "kfd").write_text("")
        for i, v in enumerate(gfx_target_versions):
            node = self.root / "sys/class/kfd/kfd/topology/nodes" / str(i)
            node.mkdir(parents=True)
            (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {v}\nvendor_id 4098\n")

    def rocm(self, gfx: str) -> None:
        """What `apt install amdrocm-blas10.0-<gfx>` puts in /opt/rocm/core-10.0."""
        lib = self.root / "opt/rocm/core-10.0/lib"
        (lib / "rocblas/library" / gfx).mkdir(parents=True)
        (lib / "libamdhip64.so.7.15.26333-0000000").write_text("hip")
        os.symlink("libamdhip64.so.7.15.26333-0000000", lib / "libamdhip64.so.7")

    def install(self, os_name: str = "Linux", arch: str = "x86_64", **extra: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(self.tmp / "home"),
            "BASERT_INSTALL_DIR": str(self.dest),
            "BASERT_NO_MODIFY_PATH": "1",
            "BASERT_INSTALL_TEST_ROOT": str(self.root),
            "STUB_OS": os_name,
            "STUB_ARCH": arch,
            "STUB_RELEASE": str(self.api),
            "STUB_ASSETS": str(self.assets),
            "STUB_LOG": str(self.log),
            **extra,
        }
        return subprocess.run(["sh", str(INSTALL_SH)], capture_output=True, text=True, env=env)

    def downloaded(self) -> list[str]:
        return [u.rsplit("/", 1)[-1] for u in self.log.read_text().split() if not u.startswith("https://api.")]

    def test_strix_halo_with_rocm_gets_the_rocm_bundle(self) -> None:
        self.amd(0, 110501)
        self.rocm("gfx1151")
        run = self.install()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Linux/x86_64 → basert-engine-linux-x64-rocm", run.stdout)
        self.assertEqual(self.downloaded(), [f"basert-engine-linux-x64-rocm-{self.V}.tar.gz"])
        self.assertTrue((self.dest / "lib-basert-engine-linux-x64-rocm").is_file())
        self.assertNotIn("apt install", run.stdout + run.stderr)

    def test_strix_point_without_rocm_is_told_its_apt_line(self) -> None:
        self.amd(0, 110500)
        run = self.install()
        # Installed anyway: the CLI works, and the engine once ROCm is there.
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue((self.dest / "lib-basert-engine-linux-x64-rocm").is_file())
        self.assertIn("sudo apt install amdrocm-blas10.0-gfx1150", run.stderr)
        self.assertIn("sudo apt install amdrocm-blas10.0-gfx1150", run.stdout)  # again at the end

    def test_blas_kernels_for_another_chip_are_not_enough(self) -> None:
        self.amd(0, 110500)
        self.rocm("gfx1151")
        run = self.install()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("sudo apt install amdrocm-blas10.0-gfx1150", run.stderr)

    def test_a_machine_without_an_amd_gpu_is_refused(self) -> None:
        for kfd, nodes in ((False, (0, 110501)), (True, (0,))):
            with self.subTest(kfd=kfd, nodes=nodes):
                shutil.rmtree(self.root)
                self.root.mkdir()
                self.amd(*nodes, kfd=kfd)
                run = self.install()
                self.assertEqual(run.returncode, 1)
                self.assertIn("needs an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU", run.stderr)
                self.assertEqual(self.downloaded(), [])

    def test_another_amd_gpu_is_named_and_refused(self) -> None:
        for nodes, name in (((0, 110000), "gfx1100"), ((90010,), "gfx90a")):
            with self.subTest(name=name):
                shutil.rmtree(self.root)
                self.root.mkdir()
                self.amd(*nodes)
                run = self.install()
                self.assertEqual(run.returncode, 1)
                self.assertIn(f"Strix Point (gfx1150) GPUs, not {name}.", run.stderr)

    def test_a_strix_gpu_behind_another_is_named_with_its_device_number(self) -> None:
        # The engine opens HIP device 0: a Strix GPU that is device 1 needs
        # HIP_VISIBLE_DEVICES, and the installer says so (and installs anyway).
        self.amd(0, 110000, 110501)
        self.rocm("gfx1151")
        run = self.install()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("→ basert-engine-linux-x64-rocm", run.stdout)
        self.assertIn("HIP device 0, which is gfx1100 here; the gfx1151 is HIP device 1", run.stderr)
        self.assertIn("export HIP_VISIBLE_DEVICES=1", run.stdout)  # again at the end

    def test_device_0_decides(self) -> None:
        self.amd(0, 110501, 110000)
        self.rocm("gfx1151")
        run = self.install()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("HIP_VISIBLE_DEVICES", run.stdout + run.stderr)

    def test_the_visibility_variables_choose_device_0(self) -> None:
        self.amd(0, 110000, 110501)
        self.rocm("gfx1151")
        for env in ({"HIP_VISIBLE_DEVICES": "1"}, {"ROCR_VISIBLE_DEVICES": "1"}, {"CUDA_VISIBLE_DEVICES": "1,0"}):
            with self.subTest(env=env):
                run = self.install(**env)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertNotIn("HIP device", run.stderr)
        run = self.install(HIP_VISIBLE_DEVICES="0")  # only the gfx1100 is left
        self.assertEqual(run.returncode, 1)
        self.assertIn("Strix Point (gfx1150) GPUs, not gfx1100.", run.stderr)

    def fresh(self, *nodes: int) -> None:
        """A new machine: these AMD GPUs (after a CPU node), ROCm for the gfx1151, nothing installed yet."""
        shutil.rmtree(self.root)
        self.root.mkdir()
        shutil.rmtree(self.dest, ignore_errors=True)
        self.log.write_text("")
        self.amd(0, *nodes)
        self.rocm("gfx1151")

    def test_a_uuid_selector_leaves_device_0_to_the_engine(self) -> None:
        # ROCr and HIP match UUIDs differently and the installer does not
        # reproduce either: it neither refuses nor offers a selector.
        uuid = "GPU-1a2b3c4d5e6f7a8b"
        for env in ({"ROCR_VISIBLE_DEVICES": uuid, "HIP_VISIBLE_DEVICES": "0"},
                    {"ROCR_VISIBLE_DEVICES": uuid}, {"HIP_VISIBLE_DEVICES": uuid}):
            with self.subTest(env=env):
                self.fresh(110000, 110501)
                run = self.install(**env)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertNotIn("HIP_VISIBLE_DEVICES=", run.stdout + run.stderr)
                self.assertIn("by UUID", run.stderr)

    def test_lists_that_leave_no_gpu_are_refused(self) -> None:
        # ROCr: set but empty, or an index out of range from the first entry;
        # HIP: an entry that is not a plain decimal ends the list.
        for env in ({"ROCR_VISIBLE_DEVICES": ""}, {"ROCR_VISIBLE_DEVICES": "5"},
                    {"HIP_VISIBLE_DEVICES": "-1"}, {"HIP_VISIBLE_DEVICES": "01"}):
            with self.subTest(env=env):
                self.fresh(110501)
                run = self.install(**env)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertIn("leave no AMD GPU", run.stderr)
                self.assertEqual(self.downloaded(), [])

    def test_the_offered_selector_makes_the_strix_gpu_device_0(self) -> None:
        # HIP_VISIBLE_DEVICES counts the GPUs ROCR_VISIBLE_DEVICES leaves, not
        # the order an earlier HIP list made: run the installer again with the
        # value it offered and the Strix GPU is device 0.
        cases = (
            ((110000, 110501), {}, "1"),
            ((110501, 110000), {"HIP_VISIBLE_DEVICES": "1,0"}, "0"),
            ((110501, 110000, 110500), {"ROCR_VISIBLE_DEVICES": "1,0,2", "HIP_VISIBLE_DEVICES": "0,1"}, "1"),
        )
        for nodes, env, want in cases:
            with self.subTest(env=env):
                self.fresh(*nodes)
                run = self.install(**env)
                self.assertEqual(run.returncode, 0, run.stderr)
                offered = re.findall(r"export HIP_VISIBLE_DEVICES=(\d+)", run.stdout)
                self.assertEqual(offered, [want], run.stdout)
                again = self.install(**{**env, "HIP_VISIBLE_DEVICES": want})
                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertNotIn("HIP device", again.stderr)

    def test_topology_nodes_are_read_in_number_order(self) -> None:
        # Node 10 sorts before node 2 as text; HIP numbers the GPUs by node number.
        self.amd(0, 0, 110501, 0, 0, 0, 0, 0, 0, 0, 110000)
        self.rocm("gfx1151")
        run = self.install()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("HIP device", run.stderr)

    @unittest.skipIf(os.geteuid() == 0, "root opens any file")
    def test_a_kfd_this_user_cannot_open_is_pointed_out(self) -> None:
        self.amd(0, 110501)
        self.rocm("gfx1151")
        os.chmod(self.root / "dev" / "kfd", 0)
        run = self.install()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("render", run.stderr)

    def test_the_other_platforms_are_unchanged(self) -> None:
        run = self.install("Darwin", "arm64")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.downloaded(), [f"basert-engine-macos-arm64-{self.V}.tar.gz"])
        for os_name, arch in (("Darwin", "x86_64"), ("Linux", "riscv64")):
            with self.subTest(platform=f"{os_name}/{arch}"):
                run = self.install(os_name, arch)
                self.assertEqual(run.returncode, 1)
                self.assertIn(
                    f"unsupported platform: {os_name}/{arch}. BaseRT ships macOS/arm64 (Metal), "
                    "Linux/arm64 (NVIDIA CUDA) and Linux/x86_64 (AMD ROCm).",
                    run.stderr,
                )

    def test_it_parses_and_passes_shellcheck(self) -> None:
        for sh in ("sh", "dash", "bash"):
            if shutil.which(sh):
                run = subprocess.run([sh, "-n", str(INSTALL_SH)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, f"{sh}: {run.stderr}")
        if not shutil.which("shellcheck"):
            self.skipTest("shellcheck is not installed")
        run = subprocess.run(["shellcheck", str(INSTALL_SH)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout)


@unittest.skipUnless((WORKFLOWS / "publish.yml").is_file() and not IN_ENGINE_REPO, "the open repo's workflows")
class WorkflowTests(unittest.TestCase):
    """What .github/workflows/publish.yml expects a release to build, against
    what the packaging scripts build; and where the lab's runners may be used."""

    V = "1.2.3"
    # Self-hosted runners: the GB10 and the Strix Halo.
    LAB_LABELS = ("gb10", "rocm-gfx1151")
    # Events a pull request from a fork can start a workflow with.
    PR_EVENTS = {"pull_request", "pull_request_target", "pull_request_review", "pull_request_review_comment",
                 "issue_comment", "workflow_run", "merge_group"}

    def setUp(self) -> None:
        self.publish = (WORKFLOWS / "publish.yml").read_text()

    @staticmethod
    def triggers(workflow: str) -> set[str]:
        block = re.search(r"^on:\n((?:[ \t]+.*\n|\n)*)", workflow, re.M)
        if block is None:
            inline = re.search(r"^on:\s*(.+)$", workflow, re.M)
            return set(re.findall(r"[\w-]+", inline.group(1))) if inline else set()
        return set(re.findall(r"^  (\w+):", block.group(1), re.M))

    def test_lab_runners_never_run_pull_requests(self) -> None:
        # A fork's pull request runs the workflow as the fork wrote it, so
        # only a workflow pull requests cannot start may name these runners.
        for path in sorted(WORKFLOWS.glob("*.y*ml")):
            text = path.read_text()
            labels = [label for label in self.LAB_LABELS if re.search(rf"\b{re.escape(label)}\b", text)]
            if labels:
                with self.subTest(workflow=path.name):
                    self.assertEqual(self.triggers(text) & self.PR_EVENTS, set(), f"{path.name} names {labels}")
        self.assertEqual(self.triggers(self.publish), {"release", "workflow_dispatch"})
        self.assertIn("pull_request", self.triggers((WORKFLOWS / "ci.yml").read_text()))

    def test_publish_names_every_engine_package_the_build_makes(self) -> None:
        mjs = (ROOT / "packaging" / "npm" / "build-packages.mjs").read_text()
        built = set(re.findall(r"name: `\$\{SCOPE\}/(basert-[\w-]+)`", mjs))
        listed = set()
        for value in re.findall(r'^\s*engines="([^"]*)"$', self.publish, re.M):
            listed |= set(value.replace("$engines", "").split())
        self.assertEqual(listed, built)
        self.assertEqual(built, {"basert-darwin-arm64", "basert-linux-arm64-cuda", "basert-linux-x64-rocm"})

    def test_each_wheel_pattern_matches_exactly_one_wheel(self) -> None:
        wheels = [f"basert-{self.V}-py3-none-{tag}.whl" for tag in BUNDLES.values()]
        # The plan's list of the wheels PyPI must have.
        patterns = []
        for value in re.findall(r'^\s*wheels="([^"]*)"$', self.publish, re.M):
            patterns += value.replace("$wheels", "").replace("$V", self.V).split()
        # The wheels each test job installs.
        patterns += re.findall(r"pip install --quiet out/wheels/(\S+\.whl)", self.publish)
        self.assertEqual(len(patterns), 6, patterns)
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertEqual(len(fnmatch.filter(wheels, pattern)), 1, fnmatch.filter(wheels, pattern))

    def test_each_npm_install_takes_one_engine_and_the_main_package(self) -> None:
        tarballs = [f"basecompute-basert-{name}-{self.V}.tgz" for name in ("darwin-arm64", "linux-arm64-cuda", "linux-x64-rocm")]
        tarballs.append(f"basecompute-basert-{self.V}.tgz")
        installs = re.findall(r"npm install --no-audit --no-fund (.+)$", self.publish, re.M)
        self.assertEqual(len(installs), 3, installs)
        engines = set()
        for line in installs:
            with self.subTest(install=line):
                matched = [fnmatch.filter(tarballs, arg.rsplit("/", 1)[-1]) for arg in line.split()]
                self.assertEqual([len(m) for m in matched], [1, 1])
                self.assertEqual(matched[1], [f"basecompute-basert-{self.V}.tgz"])
                engines.add(matched[0][0])
        self.assertEqual(len(engines), 3)


if __name__ == "__main__":
    unittest.main()
