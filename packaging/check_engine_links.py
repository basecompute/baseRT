"""Check that a staged engine folder loads on its own.

    python3 packaging/check_engine_links.py <engine-dir>
    python3 packaging/check_engine_links.py --own-name <library>

Wheels and npm tarballs cannot carry symlinks, so stage-engine.sh ships the
engine library as real files: under the name the bindings load
(libbaseRT.dylib / libbaseRT.so), and under the name the library records for
itself (its install name, @rpath/libbaseRT.0.dylib, or its SONAME,
libbaseRT.so.0), which is what a program linked against it asks for at run
time. Every other libbaseRT name is dropped. This fails, naming each, if

* any Mach-O or ELF file in the folder needs a library of ours the folder
  does not hold: a libbaseRT of any name, or on macOS anything loaded through
  @rpath, @loader_path or @executable_path; or
* a library in the folder records a name for itself the folder does not
  hold, so a program linked against it after installation (a Rust or C
  consumer pointed at the folder with BASERT_LIB_DIR) would build and then
  not start.

System libraries (libSystem, libcuda, libcublas) are the machine's to
provide. --own-name prints the name a library records for itself (nothing if
it records none); stage-engine.sh uses it to place that copy.

The load commands are read from the files themselves, not through otool or
ldd, so one runner checks both platforms' bundles.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
# LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_LAZY_LOAD_DYLIB,
# LC_LOAD_UPWARD_DYLIB: what a file needs. LC_ID_DYLIB is a library's own name.
MACHO_LOADS = {0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023}
LC_ID_DYLIB = 0xD
ELF_MAGIC = b"\x7fELF"
PT_LOAD, PT_DYNAMIC = 1, 2
DT_NULL, DT_NEEDED, DT_STRTAB, DT_SONAME = 0, 1, 5, 14


def _cstr(data: bytes, off: int) -> str:
    return data[off : data.index(b"\0", off)].decode()


def macho_names(data: bytes) -> tuple[list[str], str | None]:
    """(the libraries a Mach-O file loads, its own install name or None)."""
    ncmds = struct.unpack_from("<I", data, 16)[0]
    deps, own, off = [], None, 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd in MACHO_LOADS or cmd == LC_ID_DYLIB:
            name = _cstr(data, off + struct.unpack_from("<I", data, off + 8)[0])
            if cmd == LC_ID_DYLIB:
                own = name
            else:
                deps.append(name)
        off += size
    return deps, own


def elf_names(data: bytes) -> tuple[list[str], str | None]:
    """(an ELF file's DT_NEEDED libraries, its DT_SONAME or None)."""
    if data[4] != 2 or data[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF")
    phoff = struct.unpack_from("<Q", data, 0x20)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    loads, dynamic = [], None
    for i in range(phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz = struct.unpack_from("<IIQQQQ", data, phoff + i * phentsize)
        if p_type == PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        return [], None  # statically linked
    needed, soname, strtab = [], None, None
    for off in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, val = struct.unpack_from("<qQ", data, off)
        if tag == DT_NULL:
            break
        if tag == DT_NEEDED:
            needed.append(val)
        elif tag == DT_SONAME:
            soname = val
        elif tag == DT_STRTAB:
            strtab = val
    if not needed and soname is None:
        return [], None
    # DT_STRTAB is an address; the segment that maps it gives its file offset.
    base = next((off + strtab - vaddr for vaddr, off, size in loads if vaddr <= strtab < vaddr + size), None)
    if base is None:
        raise ValueError("DT_STRTAB is outside every PT_LOAD segment")
    return [_cstr(data, base + n) for n in needed], None if soname is None else _cstr(data, base + soname)


def link_names(f: Path) -> tuple[list[str], str | None] | None:
    """What a Mach-O or ELF file loads and the name it records for itself; None for any other file."""
    data = f.read_bytes()
    if len(data) >= 32 and struct.unpack_from("<I", data)[0] == MH_MAGIC_64:
        return macho_names(data)
    if data[:4] == ELF_MAGIC:
        return elf_names(data)
    return None


def ours(dep: str) -> bool:
    return dep.rsplit("/", 1)[-1].startswith("libbaseRT") or dep.startswith("@")


def missing_links(engine: Path) -> list[tuple[str, str]]:
    """(file, library) for each library of ours a file needs that `engine` lacks."""
    missing = []
    for f in sorted(engine.iterdir()):
        if not f.is_file() or f.is_symlink():
            continue
        names = link_names(f)
        for dep in names[0] if names else []:
            name = dep.rsplit("/", 1)[-1]
            if ours(dep) and not (engine / name).is_file():
                missing.append((f.name, name))
    return missing


def missing_own_names(engine: Path) -> list[tuple[str, str]]:
    """(library, name) for each library of ours whose own recorded name `engine` lacks."""
    missing = []
    for f in sorted(engine.iterdir()):
        if not f.is_file() or f.is_symlink():
            continue
        names = link_names(f)
        own = names[1] if names else None
        if own is not None and ours(own) and not (engine / own.rsplit("/", 1)[-1]).is_file():
            missing.append((f.name, own.rsplit("/", 1)[-1]))
    return missing


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--own-name":
        names = link_names(Path(sys.argv[2]))
        if names and names[1]:
            print(names[1].rsplit("/", 1)[-1])
        return
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    engine = Path(sys.argv[1])
    missing = missing_links(engine)
    for f, name in missing:
        print(f"{engine}: {f} needs {name}, which is not in the folder", file=sys.stderr)
    unnamed = missing_own_names(engine)
    for f, name in unnamed:
        print(f"{engine}: {f} calls itself {name}, which is not in the folder: a program linked against it would not start", file=sys.stderr)
    if missing or unnamed:
        sys.exit(1)


if __name__ == "__main__":
    main()
