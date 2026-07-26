#!/bin/sh
# BaseRT one-shot installer.
#
#   curl -LsSf https://basecompute.co/install.sh | sh
#
# Downloads the latest prebuilt engine bundle for this platform — macOS/arm64
# (Metal) or Linux/arm64 (CUDA) — the engine library + the basert CLI and
# runtime tools, installs it to ~/.basert, then adds it to your PATH. Re-running
# upgrades in place; it skips the download when you're already on the target
# release.
#
# Environment overrides:
#   BASERT_INSTALL_DIR   install location           (default: $HOME/.basert)
#   BASERT_VERSION       release tag, e.g. v0.5.0   (default: latest)
#   BASERT_FORCE         set to 1 to reinstall even if already up to date
#   BASERT_NO_MODIFY_PATH  set to 1 to skip editing shell profiles
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
# BaseRT ships two prebuilt engine bundles: macOS/arm64 (Metal) and Linux/arm64
# (CUDA). Pick the one matching this host; the installed CLI is built for that
# backend and `basert pull` resolves the matching catalog variant.
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
  *)
    err "unsupported platform: $os/$arch. BaseRT ships macOS/arm64 (Metal) and Linux/arm64 (CUDA)." ;;
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

# Find the browser_download_url for the macOS arm64 tarball.
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

# --- clean install -----------------------------------------------------------
# Remove the previous bundle's artifacts first so a file dropped in a newer
# release can't linger as an orphan. Only the known bundle contents are touched
# (the bundle is flat: libbaseRT.{dylib,so}, baseRT.metallib on macOS, basert,
# basert-*, include/), so an unrelated file in a custom INSTALL_DIR is left alone.
mkdir -p "$INSTALL_DIR"
rm -f  "$INSTALL_DIR"/basert "$INSTALL_DIR"/basert-* "$INSTALL_DIR"/baseRT_* \
       "$INSTALL_DIR"/libbaseRT.dylib "$INSTALL_DIR"/libbaseRT.so* "$INSTALL_DIR"/baseRT.metallib
rm -rf "$INSTALL_DIR"/include

tar -xzf "$tmp/bundle.tar.gz" -C "$INSTALL_DIR"
chmod +x "$INSTALL_DIR"/basert* 2>/dev/null || true
printf '%s\n' "$tag" > "$STAMP"
say "installed $tag to $INSTALL_DIR"

if [ ! -x "$INSTALL_DIR/basert" ]; then
  warn "this release bundle does not include the 'basert' launcher; only the"
  warn "runtime tools (basert-serve, basert-chat, …) were installed. Pull/convert"
  warn "need a newer release. Build the launcher meanwhile: see the docs."
fi

# --- PATH wiring -------------------------------------------------------------
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
echo
echo "  Try it:"
echo "    basert pull basecompute/Qwen3-0.6B"
echo "    basert chat basecompute/Qwen3-0.6B"
echo
echo "  Docs: https://github.com/$REPO"
