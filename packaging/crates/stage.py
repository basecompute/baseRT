"""Stage the Rust crates for crates.io in a scratch folder.

    python3 packaging/crates/stage.py <out-dir>

Writes two cargo workspaces, each publishable with
`cargo publish --workspace` (cargo 1.90+, which also orders the crates):

  <out-dir>/cli       the basert CLI (base-convert/) and its 7 library crates
  <out-dir>/bindings  the Rust bindings (bindings/rust/): baseRT-sys, baseRT

Both source folders keep their manifests as they are for development, so the
changes crates.io needs are made to copies here:

* Crate names get a `basert-` prefix (`base-format` -> `basert-format`) so they
  don't claim generic names on the shared registry. The CLI package becomes
  `basert-cli`, since crates.io treats `basert` and the bindings' `baseRT` as
  the same name. Dependency keys are unchanged (`package = "basert-format"`),
  and each library keeps its name (`[lib] name = "base_format"`), so the code,
  each crate's own tests and doctests included, still says `base_format::`.
* Every crate gets the `description` crates.io requires and a repository link,
  and each path dependency between them gets the exact version it is published at.
* crates.io packs each crate on its own, so the CLI can't `include_str!` the
  quant profiles from ../../../profiles. The profiles it embeds are copied into
  the crate and the paths shortened. The public profiles and the engine header
  that tests read from the workspace are copied in too, into the crates whose
  tests read them (`cargo test` on a crate from crates.io); those tests look in
  the crate first.
* baseRT-sys carries the engine's C headers in its own include/baseRT/, which
  its build script hands to dependents' build scripts as DEP_BASERT_INCLUDE.
* Every crate carries the LICENSE and NOTICE.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/basecompute/baseRT"
HOMEPAGE = "https://docs.basecompute.co"

# Source crate name -> (crates.io name, description).
CLI_CRATES = {
    "base-format": ("basert-format", "Reader and writer for the BaseRT .base model format"),
    "base-quant": ("basert-quant", "Quantization packing routines for BaseRT .base models"),
    "base-awq": ("basert-awq", "AWQ (activation-aware weight quantization) calibration for BaseRT models"),
    "base-readers": ("basert-readers", "GGUF, MLX-safetensors and HF-safetensors readers for the BaseRT converter"),
    "base-arch": ("basert-arch", "Per-architecture tensor-name normalization for the BaseRT converter"),
    "base-sign": ("basert-sign", "ed25519 signing and verification of BaseRT .base manifests"),
    "base-hub": ("basert-hub", "Model hub for BaseRT: resolve, download and catalog .base models"),
    "base-convert": ("basert-cli", "The basert CLI: pull, convert and run LLMs locally with BaseRT"),
}
BINDING_CRATES = ["baseRT-sys", "baseRT"]
IGNORE = shutil.ignore_patterns("target", ".DS_Store")


def fail(msg: str) -> None:
    sys.exit(f"stage.py: {msg}")


def sub_once(pattern: str, repl: str, text: str, what: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        fail(f"could not {what}")
    return new


def add_package_fields(manifest: Path, fields: dict[str, str]) -> None:
    """Insert `key = value` lines right after [package] name, skipping keys already set."""
    text = manifest.read_text()
    pkg = text[: text.find("\n[", text.find("[package]") + 1)]
    lines = "".join(
        f"{k} = {v}\n" for k, v in fields.items() if not re.search(rf"^{re.escape(k)}(\.workspace)?\s*=", pkg, re.M)
    )
    text = sub_once(r"^(name = .*\n)", lambda m: m.group(1) + lines, text, f"find name in {manifest}")
    manifest.write_text(text)


def add_license(crate: Path) -> None:
    for name in ("LICENSE", "NOTICE"):
        shutil.copy(ROOT / name, crate / name)


def stage_cli(out: Path) -> str:
    ws = out / "cli"
    shutil.copytree(ROOT / "base-convert", ws, ignore=IGNORE)
    root_toml = ws / "Cargo.toml"
    text = root_toml.read_text()
    version = re.search(r'^\[workspace\.package\][^\[]*?^version = "([^"]+)"', text, re.M | re.S).group(1)

    for old, (new, _) in CLI_CRATES.items():
        if old == "base-convert":
            continue  # the CLI itself; nothing depends on it
        text = sub_once(
            rf'^{re.escape(old)} = \{{ path = "(crates/{re.escape(old)})" \}}$',
            rf'{old} = {{ package = "{new}", version = "={version}", path = "\1" }}',
            text,
            f"rewrite workspace dependency {old}",
        )
    root_toml.write_text(text)

    for old, (new, description) in CLI_CRATES.items():
        crate = ws / "crates" / old
        manifest = crate / "Cargo.toml"
        m = sub_once(rf'^name = "{re.escape(old)}"$', f'name = "{new}"', manifest.read_text(), f"rename {old}")
        if (crate / "src" / "lib.rs").exists():
            if re.search(r"^\[lib\]", m, re.M):
                fail(f"{old} already has a [lib] section")
            m += f'\n[lib]\nname = "{old.replace("-", "_")}"\n'
        manifest.write_text(m)
        add_package_fields(
            manifest,
            {"description": f'"{description}"', "repository": f'"{REPOSITORY}"', "homepage": f'"{HOMEPAGE}"'},
        )
        add_license(crate)

    # The CLI's crate page gets the workspace README (what `basert` is).
    cli = ws / "crates" / "base-convert"
    shutil.copy(ws / "README.md", cli / "README.md")
    add_package_fields(cli / "Cargo.toml", {"readme": '"README.md"'})

    # Embedded profiles: copy each into the crate, then point include_str! at the copy.
    for src in (cli / "src").rglob("*.rs"):
        code = src.read_text()
        for rel in sorted(set(re.findall(r'include_str!\("\.\./\.\./\.\./(profiles/[^"]+)"\)', code))):
            dest = cli / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ws / rel, dest)
        src.write_text(code.replace('include_str!("../../../profiles/', 'include_str!("../profiles/'))

    # What tests read from the workspace: base-quant's and the CLI's read the
    # public profiles, and the CLI's the engine header (its ABI number).
    for crate in ("base-quant", "base-convert"):
        shutil.copytree(ws / "profiles", ws / "crates" / crate / "profiles", dirs_exist_ok=True)
    (cli / "include" / "baseRT").mkdir(parents=True)
    shutil.copy(ROOT / "include" / "baseRT" / "baseRT.h", cli / "include" / "baseRT" / "baseRT.h")

    for src in (ws / "crates").rglob("*.rs"):
        if re.search(r'include_(str|bytes)!\("\.\./\.\./\.\.', src.read_text()):
            fail(f"{src.relative_to(ws)} still embeds a file from outside its crate")
    return version


def stage_bindings(out: Path, version: str) -> None:
    ws = out / "bindings"
    shutil.copytree(ROOT / "bindings" / "rust", ws, ignore=IGNORE)
    (ws / "Cargo.toml").write_text(
        "# Generated by packaging/crates/stage.py so `cargo publish --workspace`\n"
        "# publishes baseRT-sys before baseRT.\n"
        '[workspace]\nresolver = "2"\nmembers = ["baseRT-sys", "baseRT"]\n'
    )
    shutil.copytree(ROOT / "include" / "baseRT", ws / "baseRT-sys" / "include" / "baseRT")
    for crate in BINDING_CRATES:
        manifest = ws / crate / "Cargo.toml"
        found = re.search(r'^version = "([^"]+)"', manifest.read_text(), re.M).group(1)
        if found != version:
            fail(f"{crate} is at {found} but the CLI workspace is at {version}")
        shutil.copy(ws / "README.md", ws / crate / "README.md")
        add_license(ws / crate)
        add_package_fields(
            manifest, {"repository": f'"{REPOSITORY}"', "homepage": f'"{HOMEPAGE}"', "readme": '"README.md"'}
        )
    manifest = ws / "baseRT" / "Cargo.toml"
    manifest.write_text(
        sub_once(
            r'^baseRT-sys = \{ path = "\.\./baseRT-sys" \}$',
            f'baseRT-sys = {{ version = "={version}", path = "../baseRT-sys" }}',
            manifest.read_text(),
            "pin baseRT-sys",
        )
    )


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    # Only the public source tree is published. The engine's private checkout,
    # which has tools/sync_public.py, makes one with packaging/public-tree.sh.
    if (ROOT / "tools" / "sync_public.py").exists():
        fail("run this from the tree packaging/public-tree.sh builds, not the internal checkout")
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        fail(f"{out} already exists")
    out.mkdir(parents=True)
    version = stage_cli(out)
    stage_bindings(out, version)
    print(f"staged {version}: {out / 'cli'} and {out / 'bindings'}")


if __name__ == "__main__":
    main()
