"""Khai báo thư mục gốc của repo ở một chỗ, để các module không phải tự đếm `Path(__file__).parent.parent...`.

Nội dung chính:
- `REPO_ROOT`: đường dẫn tuyệt đối tới thư mục gốc của repo.
- `repo_path`: quy tắc chung để đổi đường dẫn trong config/profile thành đường dẫn tuyệt đối.
- `local_data_root`: thư mục dữ liệu mặc định trên máy, lấy từ profile `local`.

Lưu ý:
- Đường dẫn được tính từ vị trí của chính file này, không phụ thuộc thư mục làm việc hiện tại. Vì vậy giá trị
  vẫn đúng khi gói được cài ở chế độ editable hoặc khi chạy thẳng từ mã nguồn.
"""

from __future__ import annotations

from pathlib import Path

# Đi lên ba cấp: lifemate/core/paths.py -> lifemate/core -> lifemate -> repo.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Mở rộng `~` và tính đường dẫn tương đối từ thư mục gốc repo, không theo thư mục làm việc hiện tại.

    Tham số:
        path: đường dẫn tuyệt đối, hoặc tương đối so với gốc repo (như `data_root` của profile `local`).
        repo_root: thư mục gốc repo.

    Trả về:
        Đường dẫn tuyệt đối.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def local_data_root(repo_root: Path = REPO_ROOT) -> Path:
    """Trả về thư mục dữ liệu của profile môi trường `local` (`configs/env/local.toml`), đã qua `repo_path`."""
    import tomllib

    with open(repo_root / "configs" / "env" / "local.toml", "rb") as f:
        raw = tomllib.load(f)
    return repo_path(raw["data_root"], repo_root)
