"""T20 resolve_dtype, T22 code_version fallback."""

from pathlib import Path

import pytest
import structlog

from lifemate.core.config import ConfigError
from lifemate.core.envinfo import EnvInfo, code_version, detect, resolve_dtype


def _info(cc: tuple[int, int] | None) -> EnvInfo:
    return EnvInfo("fake" if cc else None, cc, 16.0 if cc else None, "x", False, {})


def test_resolve_dtype() -> None:
    """T20. Protects: dtype per GPU. Catches: wrong dtype or no raise for bf16 on sm75."""
    assert resolve_dtype("auto", _info((7, 5))) == "float16"
    assert resolve_dtype("auto", _info((8, 0))) == "bfloat16"
    assert resolve_dtype("auto", _info(None)) == "float16"
    assert resolve_dtype("float16", _info((8, 0))) == "float16"
    assert resolve_dtype("bfloat16", _info((8, 6))) == "bfloat16"
    with pytest.raises(ConfigError):
        resolve_dtype("bfloat16", _info((7, 5)))


def test_code_version_fallback(tmp_path: Path) -> None:
    """T22. Protects: VERSION file read without .git; else "unknown" with a warning."""
    with structlog.testing.capture_logs() as logs:
        assert code_version(tmp_path) == ("unknown", None)
    assert any(e["event"] == "code_version_unknown" for e in logs)
    (tmp_path / "VERSION").write_text("abc123 dirty\n")
    assert code_version(tmp_path) == ("abc123 dirty", None)
    assert detect(tmp_path).code_version == "abc123 dirty"
