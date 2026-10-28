"""Khai báo thư mục gốc của repo ở một chỗ, để các module không phải tự đếm `Path(__file__).parent.parent...`.

Nội dung chính:
- `REPO_ROOT`: đường dẫn tuyệt đối tới thư mục gốc của repo.

Lưu ý:
- Đường dẫn được tính từ vị trí của chính file này, không phụ thuộc thư mục làm việc hiện tại. Vì vậy giá trị
  vẫn đúng khi gói được cài ở chế độ editable hoặc khi chạy thẳng từ mã nguồn.
"""

from __future__ import annotations

from pathlib import Path

# Đi lên ba cấp: lifemate/core/paths.py -> lifemate/core -> lifemate -> repo.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
