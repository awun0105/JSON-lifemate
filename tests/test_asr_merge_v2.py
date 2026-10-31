"""Test bộ ghép bản 2 (`lifemate/eval/asr_merge_v2.py`), bộ dữ liệu chuẩn hoá (`prepared_dataset.py`) và đường chạy
bản 2 của `MergedASR`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lifemate.backends.asr_merged import MergedASR
from lifemate.eval.asr_merge import Region
from lifemate.eval.asr_merge_v2 import (
    VI_FEATURES,
    candidate_pairs_v2,
    prepare,
    region_indices,
    split_regions_v2,
    vi_features,
)
from lifemate.eval.asr_v3_analysis import FEATURES
from lifemate.eval.prepared_dataset import prepared_record
from tests.test_asr_merged import CATALOG, SAMPLE, FakePrimary, FakeSecondary, _cfg


def _regions(a: str, b: str) -> list[tuple[str, str]]:
    return [(" ".join(p.primary), " ".join(p.secondary)) for p in split_regions_v2(prepare(a), prepare(b))
            if isinstance(p, Region)]  # fmt: skip


def test_prepare_lowercases_strips_punctuation_and_digitizes() -> None:
    assert prepare("Bỏ cái tắt máy lạnh lúc Năm rưỡi sáng đi, trời mát rồi.") == (
        "bỏ cái tắt máy lạnh lúc 5 rưỡi sáng đi trời mát rồi"
    )


def test_digits_no_longer_make_fake_regions_and_tone_only_differences_do() -> None:
    """"năm giờ" khớp "5 giờ" sau khi đổi số; "tắt"/"tất" (chỉ khác dấu) thành vùng; vùng tiếng Anh vẫn là `en`."""
    assert _regions("đặt báo thức năm giờ", "Đặt báo thức 5 giờ.") == []
    assert _regions("tất đèn bếp", "Tắt đèn bếp.") == [("tất", "tắt")]
    parts = split_regions_v2(
        prepare("nhắn da lô cho lan là tóc máy lạnh"), prepare("Nhắn Zalo cho Lan là tắt máy lạnh")
    )
    idx = region_indices(parts)
    assert [(" ".join(parts[k].primary), " ".join(parts[k].secondary)) for k in idx["en"]] == [("da lô", "zalo")]
    assert [(" ".join(parts[k].primary), " ".join(parts[k].secondary)) for k in idx["vi"]] == [("tóc", "tắt")]
    pairs = candidate_pairs_v2(parts)
    assert [p.kind for p in pairs] == ["en", "vi"]
    assert pairs[1].text_b == "nhắn da lô cho lan là tắt máy lạnh"


def test_vi_features_shape_and_values() -> None:
    parts = split_regions_v2(prepare("bỏ cái tóc máy lạnh trời mát gọi"), prepare("bỏ cái tắt máy lạnh trời mát rồi"))
    ks = region_indices(parts)["vi"]
    f_mid = vi_features(parts, ks[0], (-20.0, -15.0, 10, 10, False), frozenset(), {"avg_logprob": -0.2}, True)
    f_end = vi_features(parts, ks[1], None, frozenset({"roi"}), {}, False)
    assert len(f_mid) == len(VI_FEATURES)
    d = dict(zip(VI_FEATURES, f_mid, strict=True))
    assert d["delta_acoustic"] == 5.0 and d["delta_tokens"] == 0.0 and d["at_end"] == 0.0 and d["is_final"] == 1.0
    e = dict(zip(VI_FEATURES, f_end, strict=True))
    assert e["at_end"] == 1.0 and e["delta_acoustic"] == 0.0 and e["in_context"] == 1.0


def test_prepared_record_adds_fields_without_changing_originals() -> None:
    rec = {
        "id": "s",
        "audio_text": "Năm giờ nhé.",
        "history": [{"role": "user", "audio_text": "Hai sáu độ"}, {"role": "assistant"}],
    }
    out = prepared_record(rec)
    assert out["audio_text"] == "Năm giờ nhé." and out["audio_text_prepared"] == "5 giờ nhé"
    assert out["history"][0]["audio_text_prepared"] == "26 độ" and "audio_text_prepared" not in out["history"][1]
    assert "audio_text_prepared" not in rec


def _weights_v2(tmp_path: Path) -> Path:
    """File bản 2: H-Anh và H-Việt chỉ dựa vào chênh điểm âm thanh."""
    from tests.test_asr_merged import WEIGHTS

    raw: dict[str, Any] = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    n = len(FEATURES)
    raw["model"] = {"mean": [0.0] * n, "std": [1.0] * n, "weights": [0.0] * (n - 1) + [1.0, 0.0], "l2": 1.0}
    m = len(VI_FEATURES)
    raw.update(
        pipeline=2,
        vi_features=list(VI_FEATURES),
        vi_threshold=0.5,
        vi_model={"mean": [0.0] * m, "std": [1.0] * m, "weights": [1.0] + [0.0] * (m - 1) + [0.0], "l2": 1.0},
    )
    out = tmp_path / "h_v2_test.json"
    out.write_text(json.dumps(raw), encoding="utf-8")
    return out


def test_merged_v2_decides_vietnamese_regions_and_digitizes_before_merging(tmp_path: Path) -> None:
    """Bản 2: vùng chữ Việt "tóc/tắt" lấy large-v3 khi điểm âm thanh tốt hơn; số đã đổi trước khi gióng."""
    primary = FakePrimary({"m1_u1.wav": "mở nhạc lên", "m1.wav": "tóc máy lạnh lúc năm giờ"})
    secondary = FakeSecondary(
        {("m1_u1.wav", True): "Mở nhạc lên.", ("m1.wav", True): "Tắt máy lạnh lúc 5 giờ."},
        {"tóc máy lạnh lúc 5 giờ": -30.0, "tắt máy lạnh lúc 5 giờ": -10.0},
    )
    engine = MergedASR(
        _cfg(_weights_v2(tmp_path)), data_root=Path("/data"), split="train", primary=primary, secondary=secondary,
        catalog_apps=CATALOG, read_audio=lambda path, _sr: path.name,
    )  # fmt: skip
    t = engine.transcribe(SAMPLE)
    assert t.final == "tắt máy lạnh lúc 5 giờ"
    assert t.history[0] == "mở nhạc lên"
