"""T2, T3: tool docs are complete and deterministic (SPEC 4.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lifemate.tooldocs import ASK_TOOL, CatalogIndex, render_tools

CATALOGS = ("train/train_tools_catalog.json", "public_test/public_test_tools_catalog.json")


def _load(root: Path, rel: str) -> dict[str, Any]:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _all_tools(index: CatalogIndex, catalog: dict[str, Any]) -> tuple[str, ...]:
    return tuple(t["name"] for a in catalog["apps"] for t in a["tools"])


@pytest.mark.data
@pytest.mark.parametrize("rel", CATALOGS)
def test_tooldocs_complete(data_root: Path, rel: str) -> None:
    """Protects: every catalog field reaches the docs. Catches: a dropped param/enum/x-* field."""
    catalog = _load(data_root, rel)
    index = CatalogIndex(catalog)
    text = render_tools(index, _all_tools(index, catalog))
    for app in catalog["apps"]:
        assert f"## {app['id']} ({app['name_vi']}): {app['description']}" in text
        for tool in app["tools"]:
            head = f"- {tool['name']} [{tool['kind']}]" + (f" → {tool['returns']}" if tool["returns"] else "")
            assert head + "\n" in text
            block = text.split(head + "\n", 1)[1].split("\n- ", 1)[0].split("\n## ", 1)[0]
            assert tool["description"] in block
            params = tool["parameters"]
            required = set(params["required"])
            for name, prop in params["properties"].items():
                marker = f"{name}: " if name in required else f"{name}?: "
                assert marker in block, (tool["name"], name)
                for v in prop.get("enum", ()):
                    assert str(v) in block
                if "description" in prop:
                    assert f"({prop['description']})" in block
                if "x-ref" in prop:
                    assert f"(id lấy từ {prop['x-ref']})" in block
                if prop.get("x-match") == "fuzzy":
                    assert "(văn bản tự do)" in block
                if "format" in prop:
                    assert f"[format={prop['format']}]" in block
                for key, sym in (("minimum", ">="), ("exclusiveMinimum", ">"), ("maximum", "<=")):
                    if key in prop:
                        assert f"[{sym}{prop[key]}]" in block
                if prop.get("type") == "array":
                    assert "array<" in block
                    for sub in prop["items"].get("properties") or {}:
                        assert sub in block
            for group in params.get("x-one-of", ()):
                assert "[" + ", ".join(group) + "]" in block
            if params.get("x-one-of"):
                assert "cần một trong: " in block


@pytest.mark.data
def test_tooldocs_drop_descriptions_keeps_structure(data_root: Path) -> None:
    """Shrink level 2 drops param descriptions but keeps names, enums and x-* markers."""
    catalog = _load(data_root, CATALOGS[0])
    index = CatalogIndex(catalog)
    full = render_tools(index, _all_tools(index, catalog))
    short = render_tools(index, _all_tools(index, catalog), drop_param_descriptions=True)
    assert len(short) < len(full)
    assert short.count("(id lấy từ") == full.count("(id lấy từ")
    assert short.count("(văn bản tự do)") == full.count("(văn bản tự do)")


SYN: dict[str, Any] = {
    "apps": [
        {
            "id": "b",
            "name_vi": "Bê",
            "description": "app b",
            "tools": [
                {
                    "name": "b.x",
                    "kind": "read",
                    "description": "đọc",
                    "returns": "",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "app": "b",
                },
            ],
        },
        {
            "id": "a",
            "name_vi": "A",
            "description": "app a",
            "tools": [
                {
                    "name": "a.w",
                    "kind": "write",
                    "description": "ghi",
                    "returns": "ok",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "integer", "minimum": 1}},
                        "required": ["q"],
                    },
                    "app": "a",
                },
                {
                    "name": "a.r",
                    "kind": "read",
                    "description": "đọc",
                    "returns": "",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "app": "a",
                },
            ],
        },
        {
            "id": "sys",
            "name_vi": "Hệ thống",
            "description": "sys",
            "tools": [
                {
                    "name": ASK_TOOL,
                    "kind": "meta",
                    "description": "hỏi",
                    "returns": "",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "app": "sys",
                },
            ],
        },
    ]
}


def test_tooldocs_deterministic() -> None:
    """T3: same input -> same string, independent of allowed-set iteration order; catalog order kept."""
    index = CatalogIndex(SYN)
    one = render_tools(index, ["a.r", "b.x", "a.w", ASK_TOOL])
    two = render_tools(CatalogIndex(json.loads(json.dumps(SYN))), (ASK_TOOL, "a.w", "b.x", "a.r"))
    assert one == two
    assert one.index("## b") < one.index("## a") < one.index("## sys")
    assert one.index("a.w") < one.index("a.r")
    assert "q: integer [>=1]" in one


def test_allowed_tools_order_and_unknown() -> None:
    index = CatalogIndex(SYN)
    got = index.allowed_tools({"a": {"tools": ["a.r", "a.w", "a.ghost"]}})
    assert got == ("a.w", "a.r", ASK_TOOL)
    assert index.app_of("a.w") == "a" and index.tool_kind("a.w") == "write"
    assert index.apps == ("b", "a", "sys")
