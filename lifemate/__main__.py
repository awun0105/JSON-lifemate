"""Cho phép chạy CLI bằng lệnh `python -m lifemate`.

Lưu ý:
- File chỉ chuyển tiếp sang `lifemate.run.cli.main` và trả mã thoát của hàm đó cho hệ điều hành.
"""

from lifemate.run.cli import main

raise SystemExit(main())
