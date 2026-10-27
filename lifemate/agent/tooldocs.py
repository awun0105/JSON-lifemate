"""Compact tool documentation rendered from the catalog (SPEC 4.5)."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from lifemate.core.jsonutil import canonical_json

log = structlog.get_logger(__name__)

ASK_TOOL = "sys.ask_missing_info"


class CatalogIndex:
    """Catalog indexed once per run: tool name -> spec, kind, app; apps in catalog order."""

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
                self._order[name] = len(self._order)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def apps(self) -> tuple[str, ...]:
        """App ids in catalog order."""
        return tuple(self._apps)

    def app_spec(self, app: str) -> dict[str, Any]:
        """Catalog entry of one app."""
        return self._apps[app]

    def has_tool(self, name: str) -> bool:
        """True when `name` is in the catalog."""
        return name in self._tools

    def tool_spec(self, name: str) -> dict[str, Any]:
        """Catalog entry of one tool."""
        return self._tools[name]

    def tool_kind(self, name: str) -> str:
        """`read`, `write` or `meta`."""
        return str(self._tools[name]["kind"])

    def app_of(self, name: str) -> str:
        """App id owning the tool; falls back to the name prefix for unknown tools."""
        return self._app_of.get(name, name.split(".", 1)[0])

    def tool_kinds(self) -> dict[str, str]:
        """Mapping tool name -> kind for every catalog tool."""
        return {n: str(t["kind"]) for n, t in self._tools.items()}

    def allowed_tools(self, installed_apps: Mapping[str, Any]) -> tuple[str, ...]:
        """Installed tools that exist in the catalog, catalog order, plus `sys.ask_missing_info`."""
        wanted = {ASK_TOOL}
        for app, entry in installed_apps.items():
            for name in entry.get("tools", ()):
                if name in self._tools:
                    wanted.add(name)
                else:
                    log.warning("installed_tool_not_in_catalog", app=app, tool=name)
        return tuple(sorted(wanted, key=lambda n: self._order.get(n, len(self._order))))


def _type_text(prop: Mapping[str, Any]) -> str:
    if "enum" in prop:
        return "|".join(v if isinstance(v, str) else canonical_json(v) for v in prop["enum"])
    typ = prop.get("type", "any")
    if typ == "array":
        items = prop.get("items")
        return f"array<{_type_text(items)}>" if isinstance(items, Mapping) else "array"
    if typ == "object" and prop.get("properties"):
        required = set(prop.get("required", ()))
        inner = ", ".join(f"{n}{'' if n in required else '?'}: {_type_text(p)}" for n, p in prop["properties"].items())
        return f"object{{{inner}}}"
    return str(typ)


def _param_text(name: str, prop: Mapping[str, Any], required: bool, drop_description: bool) -> str:
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
    if "x-ref" in prop:
        parts.append(f"(id lấy từ {prop['x-ref']})")
    if prop.get("x-match") == "fuzzy":
        parts.append("(văn bản tự do)")
    elif "x-match" in prop:
        parts.append(f"(so khớp: {prop['x-match']})")
    return " ".join(parts)


def _tool_text(tool: Mapping[str, Any], drop_description: bool) -> str:
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
    if params.get("x-one-of"):
        groups = " hoặc ".join("[" + ", ".join(g) + "]" for g in params["x-one-of"])
        lines.append(f"  cần một trong: {groups}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=256)
def _render_cached(index: CatalogIndex, allowed: frozenset[str], drop_param_descriptions: bool) -> str:
    blocks: list[str] = []
    for app_id in index.apps:
        app = index.app_spec(app_id)
        tools = [t for t in app["tools"] if t["name"] in allowed]
        if not tools:
            continue
        head = f"## {app_id} ({app.get('name_vi', app_id)})"
        lines = [f"{head}: {app['description']}" if app.get("description") else head]
        lines.extend(_tool_text(t, drop_param_descriptions) for t in tools)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_tools(index: CatalogIndex, allowed: Iterable[str], *, drop_param_descriptions: bool = False) -> str:
    """Render docs of the allowed tools, one `##` block per app in catalog order (cached)."""
    return _render_cached(index, frozenset(allowed), drop_param_descriptions)
