"""Test engine ghép `lifemate/backends/asr_merged.py` (`MergedASR`) với hai backend giả.

Nội dung chính:
- Trọng số `configs/asr_merge/h_v1.json` nạp được và khớp thứ tự đặc trưng hiện hành.
- Với mô hình H "chỉ nhìn điểm âm thanh" (trọng số tự đặt), engine lấy large-v3 ở vùng có điểm B cao hơn, giữ RNNT ở
  vùng có điểm B thấp hơn, đổi số đọc thành chữ số, và xếp `history` đúng vị trí lượt người dùng.
- Lớp bảo vệ câu mồi: câu large-v3 lặp câu mồi được chép lại không câu mồi; câu bị bộ lọc câu bịa cắt thì bỏ.
- Trên dữ liệu thật (đánh dấu `data`): phát lại chữ và điểm của các phiên K2, K3 qua hai backend giả, kết quả phải
  trùng từng byte với `asr_v4_report` (`digitized(apply_decider(...))`) trên các mẫu `val_seen` đầu tiên.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from lifemate.backends.asr_merged import MergedASR, MergeWeights
from lifemate.core.config import ASRConfig
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample, Turn
from lifemate.eval.asr_v3_analysis import FEATURES

WEIGHTS = REPO_ROOT / "configs" / "asr_merge" / "h_v1.json"
FIXED_PROMPT = (REPO_ROOT / "configs" / "asr_prompts" / "en_full_v1.txt").read_text(encoding="utf-8").strip()


class FakePrimary:
    """Mô hình chính giả: chữ RNNT cho từng tên file."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        return [self.texts[p.name] for p in wav_paths]

    def warmup(self) -> None:
        pass


class FakeSecondary:
    """Mô hình phụ giả: chữ large-v3 theo (tên file, có câu mồi hay không) và điểm âm thanh theo câu."""

    def __init__(self, texts: dict[tuple[str, bool], str], scores: dict[str, float]) -> None:
        self.texts, self.scores = texts, scores
        self.prompts: list[str] = []

    def transcribe_one(self, audio: Any, prompt: str | None = None) -> tuple[str, dict[str, float]]:
        self.prompts.append(prompt or "")
        return self.texts[(str(audio), bool(prompt))], {"avg_logprob": -0.2, "no_speech_prob": 0.01}

    def score_texts(self, audio: Any, texts: Sequence[str]) -> list[tuple[float, int]]:
        return [(self.scores.get(t, -50.0), len(t.split())) for t in texts]

    def warmup(self) -> None:
        pass


def _turn(role: str, audio: str | None, text: str | None = None) -> Turn:
    return Turn(role=role, audio=audio, audio_text=None, text=text, tool_calls=(), tool_results=())  # type: ignore[arg-type]


SAMPLE = Sample(
    id="m1",
    split="train",
    audio="audio/m1/m1.wav",
    audio_text=None,
    context={"installed_apps": ["messaging"]},
    history=(_turn("user", "audio/m1/m1_u1.wav"), _turn("assistant", None, "Đã gửi tin nhắn.")),
    label=None,
)
CATALOG = {"messaging": {"id": "messaging", "name_vi": "Tin nhắn", "tools": []}}


def _write_weights(tmp_path: Path, acoustic_only: bool) -> Path:
    """File trọng số: bản thật, hoặc bản chỉ dùng `delta_acoustic` (trọng số 1, các đặc trưng khác 0, chặn 0)."""
    raw = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    if acoustic_only:
        n = len(FEATURES)
        raw["model"] = {"mean": [0.0] * n, "std": [1.0] * n, "weights": [0.0] * (n - 1) + [1.0, 0.0], "l2": 1.0}
    out = tmp_path / "h_test.json"
    out.write_text(json.dumps(raw), encoding="utf-8")
    return out


def _cfg(weights: Path) -> ASRConfig:
    return ASRConfig(
        backend="asr",
        model="rnnt",
        revision="",
        engine="chunkformer",
        sanitize=True,
        merge_weights=str(weights.relative_to(REPO_ROOT)) if weights.is_relative_to(REPO_ROOT) else str(weights),
        secondary={"engine": "faster_whisper", "model": "whisper", "revision": ""},
    )


def test_weights_file_loads_and_matches_feature_order() -> None:
    w = MergeWeights.load(WEIGHTS)
    assert w.version == "h_v1" and w.prompt_variant == "E2" and w.guard_ngram == 4 and w.digitize
    assert len(w.model.weights) == len(FEATURES) + 1 and w.reliability


def test_merge_takes_region_with_better_acoustic_score_and_digitizes(tmp_path: Path) -> None:
    primary = FakePrimary({"m1_u1.wav": "nhắn tin cho an qua da lô", "m1.wav": "đặt báo thức năm giờ sáng"})
    secondary = FakeSecondary(
        {
            ("m1_u1.wav", True): "Nhắn tin cho An qua Zalo",
            ("m1.wav", True): "Đặt báo thức 5 giờ sáng Email",
        },
        {
            "nhắn tin cho an qua da lô": -30.0,
            "nhắn tin cho an qua zalo": -10.0,  # B tốt hơn: lấy large-v3
            "đặt báo thức năm giờ sáng": -10.0,
            "đặt báo thức 5 giờ sáng email": -40.0,  # B tệ hơn: giữ RNNT
        },
    )
    engine = MergedASR(
        _cfg(_write_weights(tmp_path, acoustic_only=True)),
        data_root=Path("/data"),
        split="train",
        primary=primary,
        secondary=secondary,
        catalog_apps=CATALOG,
        read_audio=lambda path, _sr: path.name,
    )
    t = engine.transcribe(SAMPLE)
    assert t.history == ("nhắn tin cho an qua Zalo", None)  # chỉ vùng khác nhau "da lô" được thay
    assert t.final == "đặt báo thức 5 giờ sáng"  # giữ RNNT, rồi đổi số
    assert engine.stats.last["regions"] == 2 and engine.stats.last["taken"] == 1
    # Câu mồi E2: phần cố định, tên app đã cài, rồi lời trợ lý liền trước (chỉ ở lượt cuối).
    assert secondary.prompts[0] == f"{FIXED_PROMPT} Ứng dụng đã cài: Tin nhắn."
    assert secondary.prompts[1] == f"{FIXED_PROMPT} Ứng dụng đã cài: Tin nhắn. Đã gửi tin nhắn."


def test_guard_retries_and_filtered_secondary_is_dropped(tmp_path: Path) -> None:
    echo = " ".join(FIXED_PROMPT.split()[:6])  # lặp 6 từ đầu của câu mồi cố định
    primary = FakePrimary({"m1_u1.wav": "mở nhạc lên", "m1.wav": "tắt đèn phòng khách"})
    secondary = FakeSecondary(
        {
            ("m1_u1.wav", True): echo,
            ("m1_u1.wav", False): "Mở nhạc lên Spotify",
            # Câu bịa kiểu phụ đề: bộ lọc cắt nên lớp bảo vệ cho chép lại; lần hai vẫn bịa thì bỏ câu large-v3.
            ("m1.wav", True): "Tắt đèn phòng khách. Hãy subscribe cho kênh để không bỏ lỡ video mới.",
            ("m1.wav", False): "Tắt đèn phòng khách. Hãy subscribe cho kênh để không bỏ lỡ video mới.",
        },
        {"mở nhạc lên": -20.0, "mở nhạc lên spotify": -5.0},
    )
    engine = MergedASR(
        _cfg(_write_weights(tmp_path, acoustic_only=True)),
        data_root=Path("/data"),
        split="train",
        primary=primary,
        secondary=secondary,
        catalog_apps=CATALOG,
        read_audio=lambda path, _sr: path.name,
    )
    t = engine.transcribe(SAMPLE)
    assert t.history[0] == "mở nhạc lên Spotify"
    assert t.final == "tắt đèn phòng khách"
    assert engine.stats.last["retries"] == 2 and engine.stats.last["dropped"] == 1
    assert secondary.prompts[1] == "" and secondary.prompts[3] == ""


# ---------- Phát lại dữ liệu thật ----------

K = REPO_ROOT / "artifacts" / "runs" / "colab" / "asr" / "10-30_v3"
DATA_ROOT = REPO_ROOT / "data" / "train+public_test"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class ReplayPrimary:
    """Phát lại chữ RNNT **trước bộ lọc** (`raw`) theo (id, lượt); `ASRTranscriber` lọc lại như lúc chạy."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.raw = {(r["id"], str(r["turn"])): str(r["raw"]) for r in rows}
        self.keys: list[tuple[str, str]] = []

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        out = [self.raw[k] for k in self.keys[: len(wav_paths)]]
        self.keys = self.keys[len(wav_paths) :]
        return out

    def warmup(self) -> None:
        pass


class ReplaySecondary:
    """Phát lại chữ large-v3 trước bộ lọc và điểm âm thanh của phiên K3 theo (id, lượt)."""

    def __init__(self, v3: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]]) -> None:
        self.v3 = {(r["id"], str(r["turn"])): r for r in v3}
        self.scores: dict[tuple[str, str], dict[str, float]] = {}
        for r in scores:
            d = self.scores.setdefault((r["id"], str(r["turn"])), {})
            d[str(r["a"])] = float(r["score_a"])
            d[str(r["b"])] = float(r["score_b"])
        self.key: tuple[str, str] = ("", "")

    def transcribe_one(self, audio: Any, prompt: str | None = None) -> tuple[str, dict[str, float]]:
        r = self.v3[self.key]
        return str(r["raw"]), {k: float(v) for k, v in (r.get("scores") or {}).items()}

    def score_texts(self, audio: Any, texts: Sequence[str]) -> list[tuple[float, int]]:
        table = self.scores.get(self.key, {})
        return [(table.get(t, 0.0), len(t.split())) for t in texts]

    def warmup(self) -> None:
        pass


@pytest.mark.data
def test_replay_matches_offline_merge() -> None:
    """Trên 40 mẫu `val_seen` đầu: chữ của `MergedASR` (phát lại K2, K3) trùng `digitized(apply_decider(H))`."""
    if not (K / "k2_val_seen").is_dir() or not DATA_ROOT.is_dir():
        pytest.skip("artifacts of round 3 or data package missing")
    from lifemate.eval import asr_v3_analysis as A  # noqa: N812
    from lifemate.eval.asr_v4_report import digitized
    from lifemate.io.data import load_split, read_ids
    from lifemate.io.zaic import load_catalog

    ids = read_ids(REPO_ROOT / "configs" / "splits" / "val_seen.txt")[:40]
    samples = load_split(DATA_ROOT, "train", ids)
    catalog = {str(a["id"]): a for a in load_catalog(DATA_ROOT, "train")["apps"]}
    rnnt = [r for r in _rows(K / "k2_val_seen" / "rnnt.hyps.jsonl") if r["id"] in ids]
    v3 = [r for r in _rows(K / "k2_val_seen" / "v3_E2.hyps.jsonl") if r["id"] in ids]
    scores = [r for r in _rows(K / "k3" / "scores_val_seen.jsonl") if r["id"] in ids]
    weights = MergeWeights.load(WEIGHTS)
    context = A.context_words_by_sample(samples, catalog)
    from lifemate.eval.foreign_words import find_foreign_words

    prompt_words = frozenset(w.folded for w in find_foreign_words(FIXED_PROMPT))
    utts = A.utterance_regions(rnnt, v3, scores)
    decide = A.h_decider(weights.model, context, prompt_words, weights.reliability)
    expected = {(r["id"], str(r["turn"])): r["hyp"] for r in digitized(A.apply_decider(rnnt, utts, decide))}

    primary, secondary = ReplayPrimary(rnnt), ReplaySecondary(v3, scores)
    engine = MergedASR(
        _cfg(WEIGHTS), data_root=DATA_ROOT, split="train", primary=primary, secondary=secondary, catalog_apps=catalog
    )
    original_read = engine._read_audio

    def read_and_mark(path: Path, sr: int) -> Any:
        # Đặt khoá (id, lượt) cho backend phụ theo thứ tự file; đọc WAV thật để giữ cờ "câu dài hơn 30 giây".
        secondary.key = next_keys.pop(0)
        return original_read(path, sr)

    engine._read_audio = read_and_mark  # type: ignore[method-assign]
    mismatches = []
    for s in samples:
        keys = [(s.id, str(i)) for i, t in enumerate(s.history) if t.role == "user"] + [(s.id, "final")]
        primary.keys = list(keys)
        next_keys = list(keys)
        t = engine.transcribe(s)
        got = [h for h, turn in zip(t.history, s.history, strict=True) if turn.role == "user"] + [t.final]
        for key, text in zip(keys, got, strict=True):
            if text != expected[key]:
                mismatches.append((key, text, expected[key]))
    assert not mismatches, mismatches[:5]
