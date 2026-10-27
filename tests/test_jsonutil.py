"""canonical_json and truncate."""

from lifemate.jsonutil import canonical_json, truncate


def test_canonical_json_sorted_compact_unicode() -> None:
    """Protects: deterministic prompt/schema strings. Catches: key order or spacing drift."""
    assert canonical_json({"b": 1, "a": {"d": [1, 2], "c": "phòng"}}) == '{"a":{"c":"phòng","d":[1,2]},"b":1}'
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_truncate() -> None:
    """Protects: tool results are capped. Catches: off-by-one or suffix on short text."""
    assert truncate("abc", 3) == "abc"
    assert truncate("abcdef", 3) == "abc…(đã cắt)"
    assert truncate("abcdef", 2, suffix="..") == "ab.."
