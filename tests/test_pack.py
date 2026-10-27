"""Packaging checks (SPEC 4.16): code zip, notebook requirements, notebooks hold no logic."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [REPO / "ops" / "notebooks" / "kaggle_run.ipynb", REPO / "ops" / "notebooks" / "colab_run.ipynb"]


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs a git checkout")
def test_pack_code_zip(tmp_path: Path) -> None:
    out = subprocess.run(
        ["bash", str(REPO / "ops" / "scripts" / "pack_code.sh")],
        env={"DIST_DIR": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    zip_path = Path(out[0])
    assert zip_path.parent == tmp_path
    assert re.fullmatch(r"lifemate-code-[0-9a-f]+(-dirty)?\.zip", zip_path.name)
    assert re.fullmatch(r"[0-9a-f]{64}", out[1])
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {"lifemate/core/types.py", "configs/v0.toml", "VERSION", "VERSION_STATUS"} <= names
        assert not any("__pycache__" in n or n.startswith(("runs/", ".venv/")) for n in names)
        version = zf.read("VERSION").decode().splitlines()
        assert len(version) == 1 and re.fullmatch(r"[0-9a-f]{40}", version[0])
        assert zf.read("VERSION_STATUS").decode().strip() in {"clean", "dirty"}


def test_notebook_requirements_pinned_sorted_no_gpu() -> None:
    lines = [
        ln for ln in (REPO / "ops" / "requirements" / "notebook.txt").read_text().splitlines() if ln and ln[0] != "#"
    ]
    assert lines
    names = [ln.split("==")[0] for ln in lines]
    assert names == sorted(names)
    for ln in lines:
        assert re.fullmatch(r"[A-Za-z0-9._-]+==[^=\s]+", ln), ln
        name = ln.split("==")[0].lower()
        assert not name.startswith(("torch", "nvidia", "vllm", "triton")), ln


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_holds_no_logic(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for src in code:
        for line in src.splitlines():
            if re.search(r"\b(import|from)\s+lifemate", line):
                assert line.lstrip().startswith("!python -c"), line
    joined = "\n".join(code)
    assert "lifemate.run.cli" in joined and "configs/splits/smoke.txt" in joined
