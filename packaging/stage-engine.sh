#!/bin/sh
# Unpack a release engine bundle into DEST as the flat folder the PyPI wheel
# and the npm engine packages ship: every runtime file side by side, exactly as
# install.sh lays out ~/.basert, minus the C headers.
#
#   packaging/stage-engine.sh basert-engine-macos-arm64-0.4.1.tar.gz out/engine ['<manager>: <command>']
#
# The third argument, when given, is written beside the launcher as
# `installed-by`: `basert self update` then leaves the copy to that package
# manager and names the command, and the notice of a newer release names it too.
#
# Keeping the release layout matters: the `basert` launcher finds the
# `basert-*` runtime tools next to itself, and those find their kernel files
# (baseRT*.metallib / baseRT_cuda_kernels.fatbin) the same way.
set -eu

[ $# -eq 2 ] || [ $# -eq 3 ] || { echo "usage: $0 <bundle.tar.gz> <dest-dir> ['<manager>: <command>']" >&2; exit 2; }
bundle=$1
dest=$2
installed_by=${3:-}

mkdir -p "$dest"
tar -xzf "$bundle" -C "$dest"
rm -rf "$dest/include"

# Wheels (zip) and npm tarballs cannot carry symlinks, so the versioned library
# chain (libbaseRT.dylib -> libbaseRT.0.dylib -> libbaseRT.0.4.1.dylib, and the
# .so equivalent) collapses to the file it leads to, under the name the
# bindings load. Every other libbaseRT file is dropped: the 0.4.1 CUDA bundle,
# built in a warm build directory, carries libbaseRT.so.0.2.0 and .so.0.4.0
# beside the release's own.
if [ -e "$dest/libbaseRT.dylib" ]; then
  name=libbaseRT.dylib
elif [ -e "$dest/libbaseRT.so" ]; then
  name=libbaseRT.so
else
  echo "$bundle has no libbaseRT.dylib or libbaseRT.so" >&2; exit 1
fi
real=$name
hops=0
while [ -L "$dest/$real" ]; do
  hops=$((hops + 1))
  [ $hops -le 8 ] || { echo "$name in $bundle is a symlink loop" >&2; exit 1; }
  real=$(readlink "$dest/$real")
  case "$real" in */*) echo "$name in $bundle leads outside the bundle ($real)" >&2; exit 1 ;; esac
done
[ -f "$dest/$real" ] || { echo "$name in $bundle leads to $real, which is not in it" >&2; exit 1; }
mv "$dest/$real" "$dest/.engine-library"
for f in "$dest"/libbaseRT.*; do
  [ -e "$f" ] || [ -L "$f" ] || continue
  [ -L "$f" ] || echo "dropping $(basename "$f"): $name leads to $real" >&2
  rm -f "$f"
done
mv "$dest/.engine-library" "$dest/$name"

# A program linked against the library records the name the library gives
# itself (install name @rpath/libbaseRT.0.dylib, SONAME libbaseRT.so.0), not
# the file it was linked through, and asks for that name when it starts. So a
# Rust or C consumer pointed at this folder with BASERT_LIB_DIR needs a file by
# that name too: a second copy, as no symlink survives the packaging.
own=$(python3 "$(dirname "$0")/check_engine_links.py" --own-name "$dest/$name")
case "$own" in
  "" | "$name") ;;
  libbaseRT.*) cp -p "$dest/$name" "$dest/$own" ;;
  *) echo "$name in $bundle calls itself $own, not a libbaseRT name" >&2; exit 1 ;;
esac
# Nothing in the folder may need a name that was dropped, and the library's
# own name must be here.
python3 "$(dirname "$0")/check_engine_links.py" "$dest"

[ -x "$dest/basert" ] || { echo "bundle has no executable basert launcher" >&2; exit 1; }
if [ -n "$installed_by" ]; then
  case "$installed_by" in *": "*) ;; *) echo "installed-by must read '<manager>: <command>', not '$installed_by'" >&2; exit 2 ;; esac
  printf '%s\n' "$installed_by" > "$dest/installed-by"
fi
