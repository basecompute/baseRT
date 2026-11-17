"""Print the wheel platform tag a staged engine folder needs.

    python packaging/python/platform_tag.py <engine-dir>

The tag is read from the binaries, not hard-coded, so it follows the engine
build: on macOS the highest minimum-OS version any Mach-O file declares
(0.2.6 needs macOS 15, so macosx_15_0_arm64); on Linux the CPU the ELF files
are built for and the highest glibc symbol version any of them needs (0.2.6's
CUDA engine needs 2.38 on arm64, so manylinux_2_38_aarch64; the ROCm engine is
x86-64, so manylinux_2_<n>_x86_64). pip then refuses the wheel on a machine the
engine could not run on, instead of installing it and failing at load time.
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
# e_machine of the CPUs an engine is built for, by wheel architecture name.
ELF_CPUS = {0xB7: "aarch64", 0x3E: "x86_64"}
# GPU code objects are ELF files too (an AMDGPU code object, a CUDA cubin).
# They run on the GPU: they say nothing about the CPU and need no glibc.
ELF_GPUS = {0xE0: "AMDGPU", 0xBE: "CUDA"}


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


def elf_machine(data: bytes) -> int:
    """e_machine from an ELF header, in the byte order the header declares."""
    return struct.unpack_from("<H" if data[5] == 1 else ">H", data, 18)[0]


def platform_tag(engine: Path) -> str:
    macos: list[tuple[int, int]] = []
    glibc: list[int] = []
    cpus: dict[str, str] = {}  # CPU -> the first file built for it
    for f in sorted(engine.iterdir()):
        if not f.is_file():
            continue
        data = f.read_bytes()
        if (v := macho_min_os(data)) is not None:
            macos.append(v)
        elif data[:4] == ELF_MAGIC and len(data) >= 20:
            machine = elf_machine(data)
            if machine in ELF_GPUS:
                continue
            if machine not in ELF_CPUS:
                raise ValueError(f"{f}: an ELF file for machine {machine:#x}, not {' or '.join(ELF_CPUS.values())}")
            cpus.setdefault(ELF_CPUS[machine], f.name)
            glibc.extend(int(m) for m in GLIBC_RE.findall(data))

    if macos and not cpus:
        major, minor = max(macos)
        return f"macosx_{major}_{minor}_arm64"
    if cpus and not macos:
        if len(cpus) > 1:
            built = ", ".join(f"{name} for {cpu}" for cpu, name in sorted(cpus.items()))
            raise ValueError(f"{engine}: ELF files for more than one CPU ({built})")
        if not glibc:
            raise ValueError(f"{engine}: no ELF file names a glibc version")
        (cpu,) = cpus
        return f"manylinux_2_{max(glibc)}_{cpu}"
    raise ValueError(f"{engine}: expected only Mach-O or only ELF binaries")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        print(platform_tag(Path(sys.argv[1])))
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
