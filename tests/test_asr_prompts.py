"""Test các file câu mồi ASR (`configs/asr_prompts/`): đọc được qua config, và không trùng lời nói thật.

Lưu ý:
- Test có marker `data` chấm trên các câu đáp án của `configs/splits/train.txt` (3 593 câu); skip khi thiếu gói dữ
  liệu hoặc chưa chạy `make splits`.
- Chỉ đọc `train`, không đọc `val_seen`, `val_unseen`: viết lại câu mồi theo kết quả test là một bước chọn, nên không
  được nhìn tập kiểm tra.
"""

import json
from pathlib import Path

import pytest

from lifemate.backends.asr_guard import echoes_prompt
from lifemate.eval.asr_tune import PROMPT_FILES

# Câu mồi lần 1 được giữ để so sánh, dù trùng nhiều lời nói thật ("Nhắn Zalo cho …").
LEGACY = {"short"}


@pytest.mark.data
def test_prompts_do_not_echo_real_speech(data_root: Path, repo_root: Path) -> None:
    """Nếu ASR chép đúng hoàn toàn, lớp bảo vệ không được báo nhầm: cụm 4 từ trùng 0 câu, cụm 3 từ dưới 0,5% số câu.

    Câu mồi giống lời nói thật thì câu nói đúng cũng bị coi là "chép lại câu mồi" và bị chép lại không câu mồi.
    """
    train_ids_file = repo_root / "configs" / "splits" / "train.txt"
    if not train_ids_file.is_file():
        pytest.skip("configs/splits/train.txt missing: run `make splits`")
    train_ids = set(train_ids_file.read_text(encoding="utf-8").split())
    refs: list[str] = []
    with open(data_root / "train" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in train_ids:
                continue
            refs.append(r["audio_text"])
            refs += [h["audio_text"] for h in r["history"] if h.get("role") == "user" and h.get("audio_text")]
    for name, path in PROMPT_FILES.items():
        if name in LEGACY:
            continue
        prompt = (repo_root / path).read_text(encoding="utf-8")
        assert not [x for x in refs if echoes_prompt(x, prompt, 4)], name
        assert sum(echoes_prompt(x, prompt, 3) for x in refs) < 0.005 * len(refs), name
