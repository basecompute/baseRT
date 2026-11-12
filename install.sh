#!/bin/sh
# BaseRT one-shot installer.
#
#   curl -LsSf https://basecompute.co/install.sh | sh
#
# Downloads the latest prebuilt engine bundle for this platform — macOS/arm64
# (Metal), Linux/arm64 (NVIDIA CUDA) or Linux/x86_64 (AMD ROCm, for Strix Halo
# and Strix Point) — the engine library + the basert CLI and runtime tools,
# installs it to ~/.basert, then adds it to your PATH. Each
# release goes to ~/.basert/versions/<v>/ (from 0.4.3 on), and ~/.basert holds
# links to the one in use, so `basert self update` updates it later and
# `basert self use <v>` goes back. Re-running installs the target release the
# same way; it skips the download when you're already on it, and a copy an
# earlier run placed directly in ~/.basert is kept as the version to go back
# to. The server `basert serve` runs, superfluid, is not in the bundle:
# the installed basert downloads the release it was tested with beside itself
# (from 0.4.0 on), and does the same at its first `basert serve` when this
# step was skipped.
#
# Environment overrides:
#   BASERT_INSTALL_DIR   install location           (default: $HOME/.basert)
#   BASERT_VERSION       release tag, e.g. v0.5.0   (default: latest)
#   BASERT_FORCE         set to 1 to reinstall even if already up to date
#   BASERT_NO_MODIFY_PATH  set to 1 to skip editing shell profiles
#
# For the installer's tests only: BASERT_INSTALL_TEST_ROOT, a folder that
# stands in for / where the AMD GPU and ROCm checks look.
set -eu

REPO="basecompute/baseRT"
INSTALL_DIR="${BASERT_INSTALL_DIR:-$HOME/.basert}"
STAMP="$INSTALL_DIR/.release"

say()  { printf '\033[1;34mbasert\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mbasert\033[0m %s\n' "$1" >&2; }
err()  { printf '\033[1;31mbasert\033[0m %s\n' "$1" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || err "required tool not found: $1"; }

need curl
need tar
need uname

# --- platform detection + asset selection ------------------------------------
# BaseRT ships three prebuilt engine bundles: macOS/arm64 (Metal), Linux/arm64
# (NVIDIA CUDA) and Linux/x86_64 (AMD ROCm). Pick the one matching this host;
# the installed CLI is built for that backend and `basert pull` resolves the
# matching catalog variant.
#
# The Linux/x86_64 engine runs on the integrated GPU of AMD Strix Halo
# (gfx1151) and Strix Point (gfx1150), through the ROCm 10 runtime that AMD's
# apt repository installs in /opt/rocm/core-10.0. This installer checks for it
# and says what to install; it does not run apt itself.
sysroot="${BASERT_INSTALL_TEST_ROOT:-}"
ROCM_LIB="/opt/rocm/core-10.0/lib"
rocm_missing=""

device_hint=""

# The AMD GPUs' LLVM targets (gfx1151, ...), one per line, from the amdgpu
# driver's topology in node-number order: the order HIP numbers them in.
# Nothing without the driver.
kfd_gpus() {
  [ -e "$sysroot/dev/kfd" ] || return 0
  nodes="$sysroot/sys/class/kfd/kfd/topology/nodes"
  for dir in "$nodes"/*; do
    [ -f "$dir/properties" ] && printf '%s\n' "${dir##*/}"
  done | sort -n | while read -r node; do
    v="$(sed -n 's/^gfx_target_version \([0-9][0-9]*\)$/\1/p' "$nodes/$node/properties")"
    case "$v" in
      "" | 0) ;; # a CPU node
      *) printf 'gfx%d%x%x\n' $((v / 10000)) $((v / 100 % 100)) $((v % 100)) ;;
    esac
  done
}

# A comma list's entries ($1), one per line, as C reads them (ROCr's getline,
# HIP's do { } while (pos < size)): no entry after a final comma.
entries() {
  printf '%s\n' "$1" | sed 's/,$//' | tr ',' '\n'
}

# C's strtol($1, &end, 0) when it reads all of $1: the value (hex after 0x,
# octal after 0), or failure.
strtol0() {
  s=$1 sign=""
  case "$s" in -*) sign=- s=${s#-} ;; +*) s=${s#+} ;; esac
  case "$s" in
    0[xX]*) body=${s#??} base=16; case "$body" in "" | *[!0-9a-fA-F]*) return 1 ;; esac ;;
    0?*) body=${s#0} base=8; case "$body" in *[!0-7]*) return 1 ;; esac ;;
    "" | *[!0-9]*) return 1 ;;
    *) body=$s base=10 ;;
  esac
  # Past any GPU count, and safe from the shell's integer width.
  if [ "${#body}" -gt 9 ]; then echo "${sign}999999999"; return 0; fi
  case $base in 16) v=$((0x$body)) ;; 8) v=$((0$body)) ;; *) v=$((body)) ;; esac
  echo "$sign$v"
}

# The GPUs (stdin, one per line) that ROCR_VISIBLE_DEVICES leaves, in its order,
# as ROCr reads it (amd_filter_device.cpp): set but empty leaves none; entries
# are trimmed and read up to the number of GPUs, each a strtol index, and one
# that does not parse, is out of range or repeats ends the list. A single "?"
# when it reaches an entry that names a GPU by UUID (or an empty one).
rocr_visible() {
  lines="$(cat)"
  if [ "${ROCR_VISIBLE_DEVICES+set}" != set ]; then
    [ -z "$lines" ] || printf '%s\n' "$lines"
    return 0
  fi
  [ -n "$ROCR_VISIBLE_DEVICES" ] || return 0
  count="$(printf '%s\n' "$lines" | grep -c . || true)"
  entries "$ROCR_VISIBLE_DEVICES" | {
    out="" seen="," n=0
    while IFS= read -r entry; do
      [ "$n" -lt "$count" ] || break
      n=$((n + 1))
      entry="$(printf '%s' "$entry" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
      case "$entry" in "" | [Gg]*) echo "?"; exit 0 ;; esac
      i="$(strtol0 "$entry")" || break
      { [ "$i" -ge 0 ] && [ "$i" -lt "$count" ]; } || break
      case "$seen" in *",$i,"*) break ;; esac
      seen="$seen$i,"
      out="$out$(printf '%s\n' "$lines" | sed -n "$((i + 1))p")
"
    done
    printf '%s' "$out"
  }
}

# The GPUs (stdin, as rocr_visible leaves them) that HIP's own list leaves, as
# "index gpu" lines in HIP's order, where index is the number HIP_VISIBLE_DEVICES
# names that GPU by. HIP reads HIP_VISIBLE_DEVICES, or CUDA_VISIBLE_DEVICES when
# that is empty (rocdevice.cpp): only plain decimal entries count, a repeat is
# skipped, and any other entry ends the list. A single "?" when the order is
# unknown: ROCr's list was, or this one reaches an entry naming a GPU by UUID.
hip_visible() {
  lines="$(cat)"
  if [ "$lines" = "?" ]; then echo "?"; return 0; fi
  list="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  if [ -z "$list" ]; then
    [ -z "$lines" ] || printf '%s\n' "$lines" | awk '{ print NR - 1, $0 }'
    return 0
  fi
  count="$(printf '%s\n' "$lines" | grep -c . || true)"
  entries "$list" | {
    out="" seen=","
    while IFS= read -r entry; do
      case "$entry" in *GPU-*) echo "?"; exit 0 ;; esac
      case "$entry" in 0 | [1-9] | [1-9]*[0-9]) ;; *) break ;; esac
      case "$entry" in *[!0-9]*) break ;; esac
      { [ "${#entry}" -le 9 ] && [ "$entry" -lt "$count" ]; } || break
      case "$seen" in *",$entry,"*) continue ;; esac
      seen="$seen$entry,"
      out="$out$entry $(printf '%s\n' "$lines" | sed -n "$((entry + 1))p")
"
    done
    printf '%s' "$out"
  }
}

os="$(uname -s)"
arch="$(uname -m)"
case "$os $arch" in
  "Darwin arm64")
    ASSET_PREFIX="basert-engine-macos-arm64" ;;
  "Linux aarch64" | "Linux arm64")
    # The Linux bundle is CUDA-only — require an NVIDIA GPU / driver.
    if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidia0 ] || [ -d /proc/driver/nvidia ]; then
      ASSET_PREFIX="basert-engine-linux-arm64-cuda"
    else
      err "BaseRT on Linux/arm64 needs an NVIDIA CUDA GPU (none detected)."
    fi ;;
  "Linux x86_64")
    # The x86_64 bundle is ROCm-only, built for these two chips. The engine
    # runs on HIP device 0, so that is the GPU that decides; a Strix GPU behind
    # another one is installed for, with how to make it device 0.
    raw="$(kfd_gpus)"
    seen="$(printf '%s\n' "$raw" | rocr_visible | hip_visible)"
    gpu="" device=0 selector=0 first=""
    if [ "$seen" = "?" ]; then
      # A UUID selector: neither runtime's UUID matching is reproduced here, so
      # which GPU is device 0 is left to the engine's own check at start.
      first="$(printf '%s\n' "$raw" | sed -n '1p')"
      for g in $raw; do
        case "$g" in gfx1151 | gfx1150) gpu="$g"; break ;; esac
      done
      if [ -z "$gpu" ]; then
        [ -n "$first" ] || err "BaseRT on Linux/x86_64 needs an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU with the amdgpu driver (none detected)."
        err "BaseRT on Linux/x86_64 runs on AMD Strix Halo (gfx1151) and Strix Point (gfx1150) GPUs, not $first."
      fi
      warn "ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES names a GPU by UUID, so which GPU is HIP device 0"
      warn "is not checked here; the engine checks that it is a Strix GPU when it starts."
    else
      first="$(printf '%s\n' "$seen" | sed -n '1s/^[0-9]* //p')"
      count="$(printf '%s\n' "$seen" | grep -c . || true)"
      i=0
      while [ "$i" -lt "$count" ]; do
        line="$(printf '%s\n' "$seen" | sed -n "$((i + 1))p")"
        case "${line#* }" in gfx1151 | gfx1150) gpu="${line#* }" device=$i selector="${line%% *}"; break ;; esac
        i=$((i + 1))
      done
      if [ -z "$gpu" ]; then
        [ -n "$raw" ] || err "BaseRT on Linux/x86_64 needs an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU with the amdgpu driver (none detected)."
        [ -n "$first" ] || err "BaseRT on Linux/x86_64 needs an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU, and ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES leave no AMD GPU visible."
        err "BaseRT on Linux/x86_64 runs on AMD Strix Halo (gfx1151) and Strix Point (gfx1150) GPUs, not $first."
      fi
      if [ "$device" -ne 0 ]; then
        # HIP_VISIBLE_DEVICES counts the GPUs ROCR_VISIBLE_DEVICES leaves, not
        # the order an earlier HIP list made: offer that number.
        device_hint="export HIP_VISIBLE_DEVICES=$selector"
        warn "the engine runs on HIP device 0, which is $first here; the $gpu is HIP device $device."
        warn "make it device 0 wherever you run basert:  $device_hint"
      fi
    fi
    if [ ! -r "$sysroot/dev/kfd" ] || [ ! -w "$sysroot/dev/kfd" ]; then
      warn "you cannot open /dev/kfd, so the engine cannot use the GPU: add yourself to the render"
      warn "group (sudo usermod -aG render \"\$USER\") and log in again."
    fi
    # One package brings the runtime and the BLAS kernels for this chip.
    hip=""
    for f in "$sysroot$ROCM_LIB"/libamdhip64.so*; do [ -e "$f" ] && hip="$f"; done
    if [ -z "$hip" ] || [ ! -d "$sysroot$ROCM_LIB/rocblas/library/$gpu" ]; then
      rocm_missing="sudo apt install amdrocm-blas10.0-$gpu"
      warn "the engine needs the ROCm 10 runtime and the $gpu BLAS kernels in /opt/rocm/core-10.0, which"
      warn "are not there. Install them from AMD's ROCm 10 apt repository (stable.repo.amd.com/rocm/core):"
      warn "  $rocm_missing"
    fi
    ASSET_PREFIX="basert-engine-linux-x64-rocm" ;;
  *)
    err "unsupported platform: $os/$arch. BaseRT ships macOS/arm64 (Metal), Linux/arm64 (NVIDIA CUDA) and Linux/x86_64 (AMD ROCm)." ;;
esac
say "platform $os/$arch → $ASSET_PREFIX"

# --- resolve the release (one API call) --------------------------------------
if [ -n "${BASERT_VERSION:-}" ]; then
  api="https://api.github.com/repos/$REPO/releases/tags/$BASERT_VERSION"
  say "resolving release $BASERT_VERSION"
else
  api="https://api.github.com/repos/$REPO/releases/latest"
  say "resolving latest release"
fi
json="$(curl -fsSL "$api")" || err "could not reach the GitHub release API."

# tag_name of the resolved release (no jq dependency).
tag="$(printf '%s' "$json" | grep -o '"tag_name"[^,]*' | grep -o '"[^"]*"$' | tr -d '"' | head -n1)"
[ -n "$tag" ] || tag="${BASERT_VERSION:-unknown}"

# --- skip if already up to date ----------------------------------------------
if [ "${BASERT_FORCE:-0}" != "1" ] && [ -x "$INSTALL_DIR/basert" ] \
   && [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null || true)" = "$tag" ]; then
  say "already on $tag at $INSTALL_DIR — nothing to do (set BASERT_FORCE=1 to reinstall)."
  exit 0
fi

# Find the browser_download_url for this platform's tarball.
url="$(printf '%s' "$json" \
  | grep -o '"browser_download_url"[^,]*' \
  | grep -o 'https://[^"]*'"$ASSET_PREFIX"'[^"]*\.tar\.gz' \
  | head -n1 || true)"
[ -n "$url" ] || err "no \"$ASSET_PREFIX-*.tar.gz\" asset found in release $tag. \
A newer release may be required."

# --- download ----------------------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
say "downloading $(basename "$url")"
curl -fSL --progress-bar "$url" -o "$tmp/bundle.tar.gz"

# --- install -----------------------------------------------------------------
mkdir -p "$INSTALL_DIR/versions/.staging"
stage="$INSTALL_DIR/versions/.staging/install-$$"
rm -rf "$stage"
mkdir "$stage"
trap 'rm -rf "$tmp" "$stage"' EXIT
tar -xzf "$tmp/bundle.tar.gz" -C "$stage"
chmod +x "$stage"/basert* 2>/dev/null || true

# From 0.4.3 the launcher places a release itself: under versions/, the links in
# $INSTALL_DIR turned to it in one step, the copy there before kept. A release
# whose launcher cannot is placed by the basert already installed, when that
# one can; with neither, it is unpacked directly into $INSTALL_DIR as before.
placer=""
if [ -x "$stage/basert" ] && "$stage/basert" self install --help >/dev/null 2>&1; then
  placer="$stage/basert"
elif [ -x "$INSTALL_DIR/basert" ] && "$INSTALL_DIR/basert" self install --help >/dev/null 2>&1; then
  placer="$INSTALL_DIR/basert"
fi
if [ -n "$placer" ]; then
  "$placer" self install --from "$stage" --root "$INSTALL_DIR" \
    || err "placing $tag in $INSTALL_DIR failed (see above)."
  say "installed $tag to $INSTALL_DIR/versions"
else
  install_flat=1
fi

# --- clean install (releases before 0.4.3) ---------------------------------
# Remove the previous bundle's artifacts first so a file dropped in a newer
# release can't linger as an orphan. Only the known bundle contents are touched
# (the bundle is flat: libbaseRT.{dylib,so}, baseRT.metallib on macOS, basert,
# basert-*, include/), so an unrelated file in a custom INSTALL_DIR is left alone.
if [ "${install_flat:-0}" = 1 ]; then
  rm -rf "$stage"
  rmdir "$INSTALL_DIR/versions/.staging" "$INSTALL_DIR/versions" 2>/dev/null || true
  # libbaseRT*.dylib, not libbaseRT.dylib: since 0.2.4 the macOS bundle ships a
  # VERSIONED library plus two symlinks (libbaseRT.0.2.4.dylib <- libbaseRT.0.dylib
  # <- libbaseRT.dylib), and the bare name matched only the last of those. Every
  # upgrade then left the previous release's real library behind — an orphan of
  # the full binary size that nothing points at, and a loaded gun for any future
  # change to the install name. (The .so pattern was already globbed.)
  rm -f  "$INSTALL_DIR"/basert "$INSTALL_DIR"/basert-* "$INSTALL_DIR"/baseRT_* \
         "$INSTALL_DIR"/libbaseRT*.dylib "$INSTALL_DIR"/libbaseRT.so* "$INSTALL_DIR"/baseRT.metallib
  rm -rf "$INSTALL_DIR"/include

  tar -xzf "$tmp/bundle.tar.gz" -C "$INSTALL_DIR"
  chmod +x "$INSTALL_DIR"/basert* 2>/dev/null || true
  printf '%s\n' "$tag" > "$STAMP"
  say "installed $tag to $INSTALL_DIR"
fi

if [ ! -x "$INSTALL_DIR/basert" ]; then
  warn "this release bundle does not include the 'basert' launcher; only the"
  warn "runtime tools (basert-serve, basert-chat, …) were installed. Pull/convert"
  warn "need a newer release. Build the launcher meanwhile: see the docs."
fi

# --- superfluid, the server `basert serve` runs ------------------------------
# basert 0.4.0 and later do not bundle it: the launcher downloads the release
# it was tested with, checks its SHA-256 against the published one and installs
# it beside itself, where `basert serve` looks first. Not fatal here: a start
# installs it too, and `basert superfluid install` does it by hand. A release
# before 0.4.0 has no such command and nothing to install.
if [ -x "$INSTALL_DIR/basert" ] && "$INSTALL_DIR/basert" superfluid --help >/dev/null 2>&1; then
  say "installing superfluid, the server basert serve runs"
  if ! "$INSTALL_DIR/basert" superfluid install; then
    warn "superfluid could not be installed now; 'basert serve' installs it at its first start,"
    warn "or run: $INSTALL_DIR/basert superfluid install"
  fi
fi

# --- PATH wiring -------------------------------------------------------------
# shellcheck disable=SC2016 # $PATH is written to the profile unexpanded
add_path_line='export PATH="'"$INSTALL_DIR"':$PATH"'
already_on_path=0
case ":$PATH:" in *":$INSTALL_DIR:"*) already_on_path=1 ;; esac

if [ "${BASERT_NO_MODIFY_PATH:-0}" != "1" ] && [ "$already_on_path" -eq 0 ]; then
  for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
    [ -f "$rc" ] || continue
    if ! grep -qsF "$INSTALL_DIR" "$rc"; then
      printf '\n# Added by the BaseRT installer\n%s\n' "$add_path_line" >> "$rc"
      say "added $INSTALL_DIR to PATH in $rc"
    fi
  done
fi

# --- done --------------------------------------------------------------------
say "done."
if [ "$already_on_path" -eq 0 ]; then
  echo
  echo "  Restart your shell, or run:  export PATH=\"$INSTALL_DIR:\$PATH\""
fi
if [ -n "$rocm_missing" ]; then
  echo
  echo "  First install the ROCm 10 runtime the engine needs (from AMD's apt repository):"
  echo "    $rocm_missing"
fi
if [ -n "$device_hint" ]; then
  echo
  echo "  The engine runs on HIP device 0, which is not your $gpu; make the $gpu device 0:"
  echo "    $device_hint"
fi
echo
echo "  Try it:"
echo "    basert pull basecompute/Qwen3-0.6B"
echo "    basert chat basecompute/Qwen3-0.6B"
echo "    basert serve basecompute/Qwen3-0.6B   # an OpenAI-compatible server"
if [ -n "$placer" ]; then
  echo
  echo "  Later: basert self update   # the newest release, and the superfluid it needs"
fi
echo
echo "  Docs: https://github.com/$REPO"
