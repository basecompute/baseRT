#!/bin/sh
# Build the PyPI wheel for one release engine bundle.
#
#   packaging/python/build-wheel.sh basert-engine-macos-arm64-0.2.6.tar.gz dist/
#
# Produces basert-<version>-py3-none-<platform>.whl in the output folder: the
# Python bindings from bindings/python, the `basert` console command, and the
# engine under baseRT/_engine/. Needs python3 with `build` and `wheel`
# (pip install build wheel). See pyproject.toml for why this stages a copy.
set -eu

[ $# -eq 2 ] || { echo "usage: $0 <bundle.tar.gz> <out-dir>" >&2; exit 2; }
bundle=$1
out=$2
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# Only the public source tree is packaged. The engine's private checkout,
# which has tools/sync_public.py, makes one with packaging/public-tree.sh.
[ ! -e "$root/tools/sync_public.py" ] || {
  echo "run this from the tree packaging/public-tree.sh builds, not the internal checkout" >&2; exit 1; }

# The bundle name carries the release version: basert-engine-<plat>-<ver>.tar.gz
version=$(basename "$bundle" .tar.gz | sed 's/.*-\([0-9][^-]*\)$/\1/')
src_version=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$root/bindings/python/baseRT/__init__.py")
if [ "$version" != "$src_version" ]; then
  echo "version mismatch: bundle is $version but bindings/python is $src_version" >&2
  echo "(bindings/python has not been bumped to this release yet)" >&2
  exit 1
fi

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/baseRT"
cp "$root"/bindings/python/baseRT/*.py "$stage/baseRT/"
cp "$here/_cli.py" "$stage/baseRT/_cli.py"
cp "$here/pyproject.toml" "$here/README.md" "$stage/"
cp "$root/LICENSE" "$root/NOTICE" "$stage/"
"$root/packaging/stage-engine.sh" "$bundle" "$stage/baseRT/_engine" "pip: pip install --upgrade basert"

tag=$(python3 "$here/platform_tag.py" "$stage/baseRT/_engine")

python3 -m build --wheel --outdir "$stage/dist" "$stage"
python3 -m wheel tags --remove --platform-tag "$tag" "$stage"/dist/*.whl >/dev/null
mkdir -p "$out"
mv "$stage"/dist/*.whl "$out/"
echo "built $(ls "$out"/*"$tag".whl)"
