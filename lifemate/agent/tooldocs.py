"""Sinh phần mô tả tool dạng gọn từ catalog để chèn vào system prompt (SPEC 4.5).

Nội dung chính:
- `ASK_TOOL`: tên tool hỏi lại người dùng, `sys.ask_missing_info`.
- `CatalogIndex`: chỉ mục catalog, tra tool theo tên để lấy spec, loại và app.
- `render_tools`: sinh văn bản mô tả các tool được phép, chèn vào chỗ `{tool_docs}` của system prompt.

Cách hoạt động:
- Catalog được lập chỉ mục một lần cho cả lần chạy. Thứ tự app và tool giữ đúng như trong catalog, nên văn bản
  sinh ra luôn giống nhau.
- Mỗi app thành một khối `##`. Mỗi tool có ba dòng: tiêu đề, mô tả, và tham số rút gọn (kiểu, ràng buộc,
  `x-ref`, `x-match`, `x-one-of`).
- Kết quả được cache theo bộ (index, tập tool, cờ bỏ mô tả tham số). `CatalogIndex` băm theo định danh đối
  tượng để làm được khoá cache.

Lưu ý:
- Các chuỗi tiếng Việt trong file này là một phần của prompt; sửa chúng là đổi prompt.
- Cờ `drop_param_descriptions` dùng cho mức rút gọn prompt thứ 2 trong `prompt.py`.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from lifemate.core.jsonutil import canonical_json

log = structlog.get_logger(__name__)

ASK_TOOL = "sys.ask_missing_info"


class CatalogIndex:
    """Chỉ mục catalog, lập một lần cho mỗi lần chạy: tên tool tới spec, loại và app.

    Các app giữ đúng thứ tự trong catalog.

    Tham số:
        catalog: dict có khoá `apps`; mỗi app có `id` và danh sách `tools`, mỗi tool có `name`.
    """

    def __init__(self, catalog: Mapping[str, Any]) -> None:
        self._apps: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._app_of: dict[str, str] = {}
        self._order: dict[str, int] = {}
        for app in catalog["apps"]:
            self._apps[app["id"]] = app
            for tool in app["tools"]:
                name = tool["name"]
                self._tools[name] = tool
                self._app_of[name] = app["id"]
                # Vị trí của tool trong toàn catalog, dùng để sắp xếp tập tool được phép.
                self._order[name] = len(self._order)

    # Băm và so sánh theo định danh đối tượng: index chứa dict nên không băm được theo nội dung, nhưng vẫn cần làm
    # khoá cho `lru_cache`.
    def __hash__(self) -> int:
        """Băm theo định danh đối tượng."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Chỉ bằng chính đối tượng này (so sánh theo định danh)."""
        return self is other

    @property
    def apps(self) -> tuple[str, ...]:
        """Các id app theo thứ tự trong catalog."""
        return tuple(self._apps)

    def app_spec(self, app: str) -> dict[str, Any]:
        """Trả về mục catalog của app `app`; báo `KeyError` khi không có app đó."""
        return self._apps[app]

    def has_tool(self, name: str) -> bool:
        """Trả về `True` khi `name` có trong catalog."""
        return name in self._tools

    def tool_spec(self, name: str) -> dict[str, Any]:
        """Trả về mục catalog của tool `name`; báo `KeyError` khi không có tool đó."""
        return self._tools[name]

    def tool_kind(self, name: str) -> str:
        """Trả về loại của tool: `read`, `write` hoặc `meta`."""
        return str(self._tools[name]["kind"])

    def app_of(self, name: str) -> str:
        """Trả về id app chứa tool `name`; tool không có trong catalog thì lấy phần trước dấu chấm đầu tiên."""
        return self._app_of.get(name, name.split(".", 1)[0])

    def tool_kinds(self) -> dict[str, str]:
        """Trả về dict tên tool tới loại tool, cho mọi tool trong catalog."""
        return {n: str(t["kind"]) for n, t in self._tools.items()}

    def allowed_tools(self, installed_apps: Mapping[str, Any]) -> tuple[str, ...]:
        """Liệt kê các tool đã cài và có trong catalog, cộng thêm `sys.ask_missing_info`.

        Tham số:
            installed_apps: `context.installed_apps` của mẫu; mỗi app có danh sách `tools`.

        Trả về:
            Tuple tên tool theo thứ tự catalog. Tool đã cài nhưng không có trong catalog bị bỏ qua và ghi cảnh báo.
        """
        wanted = {ASK_TOOL}
        for app, entry in installed_apps.items():
            for name in entry.get("tools", ()):
                if name in self._tools:
                    wanted.add(name)
                else:
                    log.warning("installed_tool_not_in_catalog", app=app, tool=name)
        # Tool không có trong `_order` (ví dụ `ASK_TOOL` khi catalog không khai báo) được xếp cuối.
        return tuple(sorted(wanted, key=lambda n: self._order.get(n, len(self._order))))


def _type_text(prop: Mapping[str, Any]) -> str:
    """Viết gọn kiểu của một thuộc tính JSON schema: enum `a|b`, `array<...>`, `object{...}` hoặc tên kiểu."""
    if "enum" in prop:
        return "|".join(v if isinstance(v, str) else canonical_json(v) for v in prop["enum"])
    typ = prop.get("type", "any")
    if typ == "array":
        items = prop.get("items")
        return f"array<{_type_text(items)}>" if isinstance(items, Mapping) else "array"
    if typ == "object" and prop.get("properties"):
        # Thuộc tính không bắt buộc có dấu `?`, giống quy ước ở dòng tham số.
        required = set(prop.get("required", ()))
        inner = ", ".join(f"{n}{'' if n in required else '?'}: {_type_text(p)}" for n, p in prop["properties"].items())
        return f"object{{{inner}}}"
    return str(typ)


def _param_text(name: str, prop: Mapping[str, Any], required: bool, drop_description: bool) -> str:
    """Mô tả một tham số trên một dòng: tên, kiểu, mô tả (trừ khi bị bỏ) và các ràng buộc.

    Tham số:
        name: tên tham số.
        prop: JSON schema của tham số.
        required: `False` thì thêm dấu `?` sau tên.
        drop_description: `True` thì bỏ phần mô tả.

    Trả về:
        Một dòng văn bản, các phần cách nhau bằng dấu cách.
    """
    parts = [f"{name}{'' if required else '?'}: {_type_text(prop)}"]
    if prop.get("description") and not drop_description:
        parts.append(f"({prop['description']})")
    if "format" in prop:
        parts.append(f"[format={prop['format']}]")
    if "minimum" in prop:
        parts.append(f"[>={prop['minimum']}]")
    if "exclusiveMinimum" in prop:
        parts.append(f"[>{prop['exclusiveMinimum']}]")
    if "maximum" in prop:
        parts.append(f"[<={prop['maximum']}]")
    if "exclusiveMaximum" in prop:
        parts.append(f"[<{prop['exclusiveMaximum']}]")
    # `x-ref`: id phải lấy từ kết quả của tool khác, không được tự bịa.
    if "x-ref" in prop:
        parts.append(f"(id lấy từ {prop['x-ref']})")
    # `x-match`: cách bộ chấm so khớp giá trị; `fuzzy` nghĩa là văn bản tự do.
    if prop.get("x-match") == "fuzzy":
        parts.append("(văn bản tự do)")
    elif "x-match" in prop:
        parts.append(f"(so khớp: {prop['x-match']})")
    return " ".join(parts)


def _tool_text(tool: Mapping[str, Any], drop_description: bool) -> str:
    """Sinh khối mô tả một tool: dòng tiêu đề, mô tả, dòng tham số và nhóm `x-one-of` nếu có."""
    head = f"- {tool['name']} [{tool['kind']}]"
    if tool.get("returns"):
        head += f" → {tool['returns']}"
    lines = [head]
    if tool.get("description"):
        lines.append(f"  {tool['description']}")
    params = tool.get("parameters") or {}
    required = set(params.get("required", ()))
    props = params.get("properties") or {}
    if props:
        lines.append("  " + " | ".join(_param_text(n, p, n in required, drop_description) for n, p in props.items()))
    else:
        lines.append("  (không có tham số)")
    # `x-one-of`: cần ít nhất một trong các nhóm tham số (ví dụ id hoặc tên).
    if params.get("x-one-of"):
        groups = " hoặc ".join("[" + ", ".join(g) + "]" for g in params["x-one-of"])
        lines.append(f"  cần một trong: {groups}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=256)
def _render_cached(index: CatalogIndex, allowed: frozenset[str], drop_param_descriptions: bool) -> str:
    """Bản có cache của `render_tools`; khoá cache là bộ (index, tập tool, cờ bỏ mô tả tham số)."""
    blocks: list[str] = []
    for app_id in index.apps:
        app = index.app_spec(app_id)
        tools = [t for t in app["tools"] if t["name"] in allowed]
        # App không có tool nào được phép thì không đưa vào prompt.
        if not tools:
            continue
        head = f"## {app_id} ({app.get('name_vi', app_id)})"
        lines = [f"{head}: {app['description']}" if app.get("description") else head]
        lines.extend(_tool_text(t, drop_param_descriptions) for t in tools)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_tools(index: CatalogIndex, allowed: Iterable[str], *, drop_param_descriptions: bool = False) -> str:
    """Sinh mô tả của các tool được phép, mỗi app một khối `##` theo thứ tự catalog; kết quả có cache.

    Tham số:
        index: chỉ mục catalog.
        allowed: tên các tool được phép; thứ tự không quan trọng vì sẽ đổi thành frozenset.
        drop_param_descriptions: `True` để bỏ mô tả tham số (mức rút gọn prompt thứ 2).

    Trả về:
        Văn bản markdown, các khối app cách nhau một dòng trống.
    """
    return _render_cached(index, frozenset(allowed), drop_param_descriptions)
