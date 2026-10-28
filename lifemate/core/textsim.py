"""Gom các câu yêu cầu gần trùng nhau thành nhóm, dùng chung cho script chia fold và bộ dựng dữ liệu SFT.

Nội dung chính:
- `cluster_near_duplicates`: gom id có văn bản giống hệt hoặc gần giống vào cùng một nhóm.

Cách hoạt động:
- Chuẩn hoá văn bản theo NFC và gộp khoảng trắng trước khi so sánh.
- Duyệt mẫu theo thứ tự id. Mỗi mẫu chưa có nhóm mở một nhóm mới và kéo vào mọi mẫu đứng sau đủ giống mẫu đó.
- Thứ tự duyệt cố định, nên lần chạy nào cũng ra cùng một kết quả.

Lưu ý:
- Mục đích là xếp các yêu cầu gần trùng vào cùng một fold, để tập train không lộ đáp án sang tập đánh giá.
- Nguồn: Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f.
  Cách chuẩn hoá và ngưỡng so khớp giữ nguyên như bản gốc.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _norm(text: str) -> str:
    """Chuẩn hoá NFC và gộp khoảng trắng, giống `normalize_input_text` trong repo của trưởng nhóm."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def cluster_near_duplicates(items: Sequence[Mapping[str, Any]], threshold: float) -> list[list[str]]:
    """Gom các id có `audio_text` giống hệt hoặc giống ít nhất `threshold` vào cùng một nhóm.

    Độ giống đo bằng tỉ lệ của `difflib.SequenceMatcher`. Thuật toán là gom tham lam theo thứ tự id, nên kết quả
    không đổi giữa các lần chạy.

    Tham số:
        items: các bản ghi có khoá `id` và có thể có khoá `audio_text`.
        threshold: ngưỡng tỉ lệ giống nhau, trong khoảng [0, 1].

    Trả về:
        Danh sách nhóm. Mỗi nhóm là danh sách id, và mỗi id thuộc đúng một nhóm.
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
            # Chỉ so khớp mờ khi cả hai câu dài hơn 10 ký tự và độ dài chênh nhau dưới 15 ký tự.
            # Câu ngắn dễ giống nhau do tình cờ. Còn hai câu chênh độ dài nhiều thì gần như không thể đạt ngưỡng, bỏ qua
            # cho nhanh.
            if not same and len(t1) > 10 and len(t2) > 10 and abs(len(t1) - len(t2)) < 15:
                m = difflib.SequenceMatcher(None, t1, t2)
                # `quick_ratio` là cận trên tính nhanh của `ratio`: cặp nào trượt ở đây thì chắc chắn trượt, không cần
                # tính `ratio`.
                same = m.quick_ratio() >= threshold and m.ratio() >= threshold
            if same:
                owner[sid2] = len(clusters)
                group.append(sid2)
        clusters.append(group)
    return clusters
