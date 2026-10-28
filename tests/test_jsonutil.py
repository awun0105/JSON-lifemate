"""Test `canonical_json` và `truncate` trong `lifemate.core.jsonutil`.

Lưu ý:
- Hàm thuần, không dùng fixture; không cần dữ liệu cuộc thi hay tokenizer.
"""

from lifemate.core.jsonutil import canonical_json, truncate


def test_canonical_json_sorted_compact_unicode() -> None:
    """JSON chuẩn tắc sắp khoá, không khoảng trắng, giữ tiếng Việt, nên chuỗi prompt và schema luôn ổn định."""
    assert canonical_json({"b": 1, "a": {"d": [1, 2], "c": "phòng"}}) == '{"a":{"c":"phòng","d":[1,2]},"b":1}'
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_truncate() -> None:
    """Văn bản bị cắt đúng giới hạn, không lệch một ký tự, và chuỗi ngắn không bị thêm hậu tố."""
    assert truncate("abc", 3) == "abc"
    assert truncate("abcdef", 3) == "abc…(đã cắt)"
    assert truncate("abcdef", 2, suffix="..") == "ab.."
