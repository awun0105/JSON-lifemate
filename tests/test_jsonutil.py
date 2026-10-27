"""Kiểm tra `canonical_json` và `truncate` trong `lifemate.core.jsonutil`.

Hàm thuần, không dùng fixture; không cần dữ liệu cuộc thi hay tokenizer.
"""

from lifemate.core.jsonutil import canonical_json, truncate


def test_canonical_json_sorted_compact_unicode() -> None:
    """Bảo vệ: chuỗi prompt/schema ổn định. Bắt lỗi: thứ tự khoá hoặc khoảng trắng bị thay đổi."""
    assert canonical_json({"b": 1, "a": {"d": [1, 2], "c": "phòng"}}) == '{"a":{"c":"phòng","d":[1,2]},"b":1}'
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_truncate() -> None:
    """Bảo vệ: kết quả tool bị cắt theo giới hạn. Bắt lỗi: lệch một ký tự hoặc thêm hậu tố cho chuỗi ngắn."""
    assert truncate("abc", 3) == "abc"
    assert truncate("abcdef", 3) == "abc…(đã cắt)"
    assert truncate("abcdef", 2, suffix="..") == "ab.."
