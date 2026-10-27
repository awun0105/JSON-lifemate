"""Thư mục gốc của repo: định nghĩa một lần thay cho việc đếm `Path(__file__).parent.parent...` ở từng module.

Cách xây dựng: tính từ vị trí của chính file này, nên đúng cả khi gói được cài ở chế độ editable hay chạy
trực tiếp từ mã nguồn, không phụ thuộc thư mục làm việc hiện tại.
"""

from __future__ import annotations

from pathlib import Path

# lifemate/core/paths.py -> lifemate/core -> lifemate -> repo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
