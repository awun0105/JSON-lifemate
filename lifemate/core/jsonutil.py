"""JSON dạng chuẩn tắc và cắt ngắn văn bản (SPEC 4.0.1).

Nội dung file: `canonical_json` cho chuỗi JSON tất định (dùng khi ghi trace và khi so sánh, băm),
`truncate` để giới hạn độ dài văn bản đưa vào prompt.

Cách xây dựng: hai hàm thuần, không trạng thái. JSON chuẩn tắc sắp xếp khóa và bỏ khoảng trắng, nên cùng một
đối tượng luôn cho cùng một chuỗi; ký tự tiếng Việt giữ nguyên thay vì thoát thành `\\uXXXX` để dễ đọc và
tiết kiệm token.
"""

import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Tuần tự hóa tất định: khóa được sắp xếp, không khoảng trắng thừa, giữ nguyên ký tự ngoài ASCII."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def truncate(text: str, max_chars: int, suffix: str = "…(đã cắt)") -> str:
    """Cắt `text` còn tối đa `max_chars` ký tự rồi nối thêm `suffix` khi văn bản dài hơn mức đó.

    Văn bản không vượt quá `max_chars` được trả về nguyên vẹn, không có `suffix`.
    """
    return text if len(text) <= max_chars else text[:max_chars] + suffix
