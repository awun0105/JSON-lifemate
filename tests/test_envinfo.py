"""Kiểm tra T20 `resolve_dtype` và T22 cách lấy `code_version` khi không có git.

`EnvInfo` được dựng tay qua helper `_info` (GPU giả theo compute capability). Không cần dữ liệu cuộc thi hay tokenizer.
"""

from pathlib import Path

import pytest
import structlog

from lifemate.core.config import ConfigError
from lifemate.core.envinfo import EnvInfo, code_version, detect, resolve_dtype


def _info(cc: tuple[int, int] | None) -> EnvInfo:
    """Tạo `EnvInfo` giả; `cc=None` nghĩa là máy không có GPU."""
    return EnvInfo("fake" if cc else None, cc, 16.0 if cc else None, "x", False, {})


def test_resolve_dtype() -> None:
    """T20. Bảo vệ: chọn dtype theo GPU. Bắt lỗi: sai dtype hoặc không báo lỗi khi ép bf16 trên sm75."""
    assert resolve_dtype("auto", _info((7, 5))) == "float16"
    assert resolve_dtype("auto", _info((8, 0))) == "bfloat16"
    assert resolve_dtype("auto", _info(None)) == "float16"
    assert resolve_dtype("float16", _info((8, 0))) == "float16"
    assert resolve_dtype("bfloat16", _info((8, 6))) == "bfloat16"
    with pytest.raises(ConfigError):
        resolve_dtype("bfloat16", _info((7, 5)))


def test_code_version_fallback(tmp_path: Path) -> None:
    """T22. Bảo vệ: không có .git thì đọc file VERSION; không có cả hai thì trả "unknown" kèm cảnh báo."""
    with structlog.testing.capture_logs() as logs:
        assert code_version(tmp_path) == ("unknown", None)
    assert any(e["event"] == "code_version_unknown" for e in logs)
    (tmp_path / "VERSION").write_text("abc123 dirty\n")
    assert code_version(tmp_path) == ("abc123 dirty", None)
    assert detect(tmp_path).code_version == "abc123 dirty"
