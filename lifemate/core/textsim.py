"""Gom nhóm các câu yêu cầu gần trùng nhau, dùng chung cho script chia fold và bộ dựng dữ liệu SFT.

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f.
Chuyển thể từ repo của trưởng nhóm, giữ nguyên cách chuẩn hóa và ngưỡng so khớp.

Cách xây dựng:
- Văn bản được chuẩn hóa NFC và gộp khoảng trắng trước khi so sánh.
- Gom nhóm tham lam theo thứ tự id: mỗi mẫu chưa có nhóm mở một nhóm mới và kéo vào mọi mẫu đứng sau đủ giống.
  Thứ tự cố định nên kết quả tất định giữa các lần chạy.
- Mục đích: các yêu cầu gần trùng phải nằm cùng một fold, tránh rò rỉ giữa tập huấn luyện và tập đánh giá.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _norm(text: str) -> str:
    """Chuẩn hóa NFC và gộp khoảng trắng (tương ứng normalize_input_text của trưởng nhóm)."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def cluster_near_duplicates(items: Sequence[Mapping[str, Any]], threshold: float) -> list[list[str]]:
    """Gom các id có audio_text giống hệt hoặc giống ít nhất `threshold` (tỉ lệ difflib) vào cùng nhóm.

    Nhờ vậy các yêu cầu gần trùng rơi vào cùng một fold. Thuật toán tham lam, duyệt theo thứ tự id, nên tất định.

    Tham số:
        items: các bản ghi có khóa `id` và (tùy chọn) `audio_text`.
        threshold: ngưỡng tỉ lệ giống nhau trong [0, 1].

    Trả về danh sách nhóm; mỗi nhóm là danh sách id, mỗi id thuộc đúng một nhóm.
    """
    texts = [(it["id"], _norm(it.get("audio_text") or "")) for it in sorted(items, key=lambda x: x["id"])]
    owner: dict[str, int] = {}
    clusters: list[list[str]] = []
    for i, (sid, t1) in enumerate(texts):
        if sid in owner:
            continue
        owner[sid] = len(clusters)
        group = [sid]
        for sid2, t2 in texts[i + 1 :]:
            if sid2 in owner:
                continue
            same = t1 == t2
            # Chỉ so khớp mờ với câu đủ dài (>10 ký tự) và độ dài chênh nhau ít (<15): câu ngắn dễ giống nhau
            # một cách tình cờ, còn độ dài chênh nhiều thì gần như không thể đạt ngưỡng nên bỏ qua cho nhanh.
            if not same and len(t1) > 10 and len(t2) > 10 and abs(len(t1) - len(t2)) < 15:
                m = difflib.SequenceMatcher(None, t1, t2)
                # quick_ratio là cận trên rẻ của ratio, loại sớm cặp chắc chắn không đạt.
                same = m.quick_ratio() >= threshold and m.ratio() >= threshold
            if same:
                owner[sid2] = len(clusters)
                group.append(sid2)
        clusters.append(group)
    return clusters
