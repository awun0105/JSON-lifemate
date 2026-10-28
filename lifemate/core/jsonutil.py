"""Tạo chuỗi JSON chuẩn tắc và cắt ngắn văn bản, dùng chung cho trace, prompt và file kết quả (SPEC 4.0.1).

Nội dung chính:
- `canonical_json`: chuyển đối tượng thành chuỗi JSON luôn giống nhau cho cùng một dữ liệu.
- `truncate`: cắt văn bản quá dài trước khi đưa vào prompt.

Cách hoạt động:
- Cả hai là hàm thuần, không giữ trạng thái.
- `canonical_json` sắp xếp khoá và bỏ khoảng trắng, nên cùng một đối tượng luôn ra cùng một chuỗi. Nhờ vậy có thể
  so sánh hoặc băm chuỗi để kiểm tra hai đối tượng có giống nhau không.

Lưu ý:
- Ký tự tiếng Việt được giữ nguyên, không đổi thành `\\uXXXX`, để dễ đọc và tốn ít token hơn.
"""

import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Chuyển `obj` thành JSON tất định: khoá theo thứ tự, không khoảng trắng thừa, giữ ký tự ngoài ASCII."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def truncate(text: str, max_chars: int, suffix: str = "…(đã cắt)") -> str:
    """Cắt `text` còn tối đa `max_chars` ký tự và nối `suffix` vào cuối nếu văn bản dài hơn.

    Tham số:
        text: văn bản cần cắt.
        max_chars: số ký tự tối đa giữ lại, chưa tính `suffix`.
        suffix: chuỗi báo hiệu văn bản đã bị cắt.

    Trả về:
        `text` nguyên vẹn nếu không dài quá `max_chars`, ngược lại là phần đầu của `text` cộng `suffix`.
    """
    return text if len(text) <= max_chars else text[:max_chars] + suffix
