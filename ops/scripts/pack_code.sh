#!/usr/bin/env bash
# Pack the code for Kaggle/Colab (SPEC 4.16): artifacts/dist/lifemate-code-<shortsha>[-dirty].zip.
# VERSION holds only the full commit hash (read by lifemate.core.envinfo.code_version);
# VERSION_STATUS holds "clean" or "dirty". Honours DIST_DIR (default: <repo>/artifacts/dist).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/artifacts/dist}"
mkdir -p "$DIST_DIR"
DIST_DIR="$(cd "$DIST_DIR" && pwd)"

cd "$REPO_ROOT"
SHA="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short=10 HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then STATUS=dirty; SUFFIX=-dirty; else STATUS=clean; SUFFIX=; fi
ZIP="$DIST_DIR/lifemate-code-${SHORT}${SUFFIX}.zip"
rm -f "$ZIP"

python3 - "$ZIP" "$SHA" "$STATUS" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path, sha, status = sys.argv[1:4]
dirs = ["lifemate", "configs", "ops/requirements", "tests"]
files = ["pyproject.toml", "README.md", "docs/SPEC.md"]
skip_parts = {"__pycache__", "runs", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
fixed = (1980, 1, 1, 0, 0, 0)  # fixed timestamps: same tree -> same zip


def add(zf: zipfile.ZipFile, path: Path) -> None:
    info = zipfile.ZipInfo(path.as_posix(), fixed)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, path.read_bytes())


with zipfile.ZipFile(zip_path, "w") as zf:
    paths = [Path(f) for f in files if Path(f).is_file()]
    for d in dirs:
        if Path(d).is_dir():
            paths += [p for p in Path(d).rglob("*") if p.is_file()]
    for p in sorted(set(paths)):
        if skip_parts & set(p.parts) or p.suffix == ".pyc":
            continue
        add(zf, p)
    for name, text in (("VERSION", sha + "\n"), ("VERSION_STATUS", status + "\n")):
        info = zipfile.ZipInfo(name, fixed)
        info.external_attr = 0o644 << 16
        zf.writestr(info, text)
PY

echo "$ZIP"
sha256sum "$ZIP" | cut -d' ' -f1
