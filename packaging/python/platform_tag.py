"""Print the wheel platform tag a staged engine folder needs.

    python packaging/python/platform_tag.py <engine-dir>

The tag is read from the binaries, not hard-coded, so it follows the engine
build: on macOS the highest minimum-OS version any Mach-O file declares
(0.2.6 needs macOS 15, so macosx_15_0_arm64); on Linux the highest glibc symbol
version any ELF file needs (0.2.6 needs 2.38, so manylinux_2_38_aarch64). pip
then refuses the wheel on a machine the engine could not run on, instead of
installing it and failing at load time.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
LC_BUILD_VERSION = 0x32
LC_VERSION_MIN_MACOSX = 0x24
ELF_MAGIC = b"\x7fELF"
GLIBC_RE = re.compile(rb"GLIBC_2\.(\d+)(?:\.\d+)?\x00")


def macho_min_os(data: bytes) -> tuple[int, int] | None:
    if len(data) < 32 or struct.unpack_from("<I", data)[0] != MH_MAGIC_64:
        return None
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == LC_BUILD_VERSION:
            minos = struct.unpack_from("<I", data, off + 12)[0]
            return minos >> 16, (minos >> 8) & 0xFF
        if cmd == LC_VERSION_MIN_MACOSX:
            v = struct.unpack_from("<I", data, off + 8)[0]
            return v >> 16, (v >> 8) & 0xFF
        off += size
    return None


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    engine = Path(sys.argv[1])
    macos: list[tuple[int, int]] = []
    glibc: list[int] = []
    for f in sorted(engine.iterdir()):
        if not f.is_file():
            continue
        data = f.read_bytes()
        if (v := macho_min_os(data)) is not None:
            macos.append(v)
        elif data[:4] == ELF_MAGIC:
            glibc.extend(int(m) for m in GLIBC_RE.findall(data))

    if macos and not glibc:
        major, minor = max(macos)
        print(f"macosx_{major}_{minor}_arm64")
    elif glibc and not macos:
        print(f"manylinux_2_{max(glibc)}_aarch64")
    else:
        sys.exit(f"{engine}: expected only Mach-O or only ELF binaries")


if __name__ == "__main__":
    main()
