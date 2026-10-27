#!/usr/bin/env bash
# Đóng gói mã nguồn cho Kaggle/Colab (SPEC 4.16): artifacts/dist/lifemate-code-<shortsha>[-dirty].zip.
# VERSION chỉ chứa mã hash đầy đủ của commit (được lifemate.core.envinfo.code_version đọc);
# VERSION_STATUS chứa "clean" hoặc "dirty". Tôn trọng biến DIST_DIR (mặc định: <repo>/artifacts/dist).
#
# Cách hoạt động:
#   1. Xác định gốc repo và thư mục đích, tạo thư mục đích nếu chưa có.
#   2. Lấy hash commit và kiểm tra cây làm việc có thay đổi chưa commit hay không (clean/dirty).
#   3. Đoạn Python nhúng ghi file zip với timestamp cố định (cùng cây mã -> cùng file zip),
#      bỏ cache/venv/runs, rồi thêm VERSION và VERSION_STATUS.
#   4. In đường dẫn file zip và mã sha256 của file.
set -euo pipefail

# Gốc repo tính từ vị trí script (ops/scripts/ -> lên hai cấp).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/artifacts/dist}"
mkdir -p "$DIST_DIR"
# Chuẩn hoá DIST_DIR thành đường dẫn tuyệt đối.
DIST_DIR="$(cd "$DIST_DIR" && pwd)"

cd "$REPO_ROOT"
# Danh sách id trong configs/splits/ không có trong git nhưng notebook cần chúng: thiếu thì dừng ngay ở đây,
# thay vì để notebook trên Kaggle/Colab hỏng giữa chừng.
if [[ ! -f configs/splits/meta.json || ! -f configs/splits/train.txt ]]; then
    echo "configs/splits/ is empty: run 'make splits' first (needs the data package)" >&2
    exit 1
fi
SHA="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short=10 HEAD)"
# Có thay đổi chưa commit thì đánh dấu dirty và thêm hậu tố -dirty vào tên file.
if [[ -n "$(git status --porcelain)" ]]; then STATUS=dirty; SUFFIX=-dirty; else STATUS=clean; SUFFIX=; fi
ZIP="$DIST_DIR/lifemate-code-${SHORT}${SUFFIX}.zip"
# Xoá file zip cũ cùng tên (nếu có) trước khi ghi lại.
rm -f "$ZIP"

# Python nhúng nhận ba tham số: đường dẫn zip, hash commit, trạng thái clean/dirty.
python3 - "$ZIP" "$SHA" "$STATUS" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path, sha, status = sys.argv[1:4]
# Thư mục và file đưa vào gói; các phần trong skip_parts bị bỏ qua.
dirs = ["lifemate", "configs", "ops/requirements", "tests"]
files = ["pyproject.toml", "README.md", "docs/SPEC.md"]
skip_parts = {"__pycache__", "runs", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
fixed = (1980, 1, 1, 0, 0, 0)  # timestamp cố định: cùng cây mã -> cùng file zip


def add(zf: zipfile.ZipFile, path: Path) -> None:
    """Thêm một file vào zip với timestamp cố định, nén deflate và quyền 0644."""
    info = zipfile.ZipInfo(path.as_posix(), fixed)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, path.read_bytes())


with zipfile.ZipFile(zip_path, "w") as zf:
    paths = [Path(f) for f in files if Path(f).is_file()]
    for d in dirs:
        if Path(d).is_dir():
            paths += [p for p in Path(d).rglob("*") if p.is_file()]
    # Sắp xếp để thứ tự trong zip luôn giống nhau.
    for p in sorted(set(paths)):
        if skip_parts & set(p.parts) or p.suffix == ".pyc":
            continue
        add(zf, p)
    # Ghi hai file phiên bản ở gốc zip.
    for name, text in (("VERSION", sha + "\n"), ("VERSION_STATUS", status + "\n")):
        info = zipfile.ZipInfo(name, fixed)
        info.external_attr = 0o644 << 16
        zf.writestr(info, text)
PY

# In đường dẫn zip và mã sha256 để đối chiếu khi tải lên.
echo "$ZIP"
sha256sum "$ZIP" | cut -d' ' -f1
