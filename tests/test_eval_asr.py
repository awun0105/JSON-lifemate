"""Test các chỉ số của `eval_asr`: WER, CER, tỉ lệ bỏ sót thực thể, phân vị (SPEC 6.4.2).

Nội dung chính:
- `_EmptyFinal`, `_RawASR`: transcriber giả cho các ca câu rỗng và câu bịa.

Lưu ý:
- `Sample` dựng tay qua `_sample`; `evaluate` chạy với `GoldTranscriber` (trả đúng văn bản gốc).
- Không cần dữ liệu cuộc thi hay tokenizer.
"""

from typing import Any

import pytest

from lifemate.backends.asr import GoldTranscriber
from lifemate.core.types import Sample, Transcript, Turn
from lifemate.eval.eval_asr import cer, entity_miss_rate, evaluate, normalize_text, percentile, wer


def _sample(sid: str, text: str, calls: list[dict[str, Any]], history: tuple[Turn, ...] = ()) -> Sample:
    """Tạo `Sample` có văn bản gốc `text` và nhãn là danh sách lời gọi `calls`."""
    return Sample(
        id=sid,
        split="train",
        audio=f"audio/{sid}/{sid}.wav",
        audio_text=text,
        context={},
        history=history,
        label={"tool_calls": calls},
    )


def test_normalize() -> None:
    """Chuẩn hoá văn bản: viết thường, bỏ dấu câu, gộp khoảng trắng, giữ dấu tiếng Việt."""
    assert normalize_text("  Chào, BẠN!  nhé. ") == "chào bạn nhé"


def test_wer_cer_known() -> None:
    """WER và CER khớp giá trị tính tay; hai chuỗi rỗng cho 0 thay vì lỗi chia cho 0."""
    assert wer("a b c d", "a x c") == pytest.approx(0.5)
    assert wer("Xin chào", "xin chào!") == 0.0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert wer("", "") == 0.0


def test_entity_miss() -> None:
    """Thực thể trong tham số nhãn ("Huy") bị ASR nghe sai thì tính là bỏ sót, còn số 150000 vẫn khớp."""
    s = _sample(
        "e", "chuyển cho Huy 150000", [{"name": "bank.transfer", "arguments": {"contact": "Huy", "amount": 150000}}]
    )
    missed, total, rate, details = entity_miss_rate([s], [Transcript("chuyển cho hùng 150000", (), 0.0)])
    assert (missed, total, rate) == (1, 2, 0.5)
    assert details[0].missed == ("huy",)


def test_entity_skips_ask_and_absent() -> None:
    """Bỏ qua tham số của tool hỏi lại và thực thể không xuất hiện trong văn bản gốc."""
    s = _sample(
        "e",
        "đặt bàn",
        [{"name": "sys.ask_missing_info", "arguments": {"q": "đặt"}}, {"name": "x", "arguments": {"n": "Hà Nội"}}],
    )
    assert entity_miss_rate([s], [Transcript("", (), 0.0)])[:2] == (0, 0)


def test_percentile() -> None:
    """Phân vị làm tròn theo hạng gần nhất; danh sách rỗng cho 0."""
    xs = [float(i) for i in range(1, 21)]
    assert percentile(xs, 95) == 19.0
    assert percentile(xs, 50) == 10.0
    assert percentile([], 95) == 0.0


def test_evaluate_gold() -> None:
    """Với transcript gold: WER, CER bằng 0, có đếm cả lượt user trong lịch sử, và báo cáo đủ các khoá."""
    u = Turn(role="user", audio=None, audio_text="trước đó", text=None, tool_calls=(), tool_results=())
    a = Turn(role="assistant", audio=None, audio_text=None, text="ok", tool_calls=(), tool_results=())
    samples = [
        _sample("a", "chuyển cho Huy 150000", [{"name": "t", "arguments": {"contact": "Huy", "amount": 150000}}]),
        _sample("b", "thời tiết Hà Nội", [{"name": "w", "arguments": {"city": "Hà Nội"}}], (u, a)),
        _sample("c", "bật đèn", []),
    ]
    rep = evaluate(samples, GoldTranscriber())
    assert rep.wer == 0.0 and rep.cer == 0.0
    assert (rep.entity_missed, rep.entity_total) == (0, 3)
    # 3 câu cuối cộng 1 lượt user trong lịch sử của mẫu "b".
    assert rep.n_utterances == 4
    text = rep.to_text()
    for key in ("WER", "CER", "entity_miss_rate", "p95"):
        assert key in text
    assert rep.to_dict()["entity_misses"] == []


class _EmptyFinal:
    """Transcriber giả luôn trả câu cuối rỗng (như khi VAD cắt mất giọng hoặc bộ lọc xoá nhầm)."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Trả câu cuối rỗng, không có lịch sử."""
        return Transcript(final="", history=tuple(None for _ in sample.history), seconds=0.1)

    def warmup(self) -> None:
        """Không cần làm gì."""


def test_evaluate_counts_empty_and_records_hyps() -> None:
    """Câu ASR rỗng được đếm riêng (`n_empty`), và `hyps` ghi đủ ref/hyp từng câu để soi lỗi."""
    samples = [_sample("s1", "bật đèn", []), _sample("s2", "tắt quạt", [])]
    hyps: list[dict[str, Any]] = []
    rep = evaluate(samples, _EmptyFinal(), hyps)
    assert rep.n_empty == 2 and rep.wer == 1.0
    assert hyps == [
        {"id": "s1", "turn": "final", "ref": "bật đèn", "hyp": ""},
        {"id": "s2", "turn": "final", "ref": "tắt quạt", "hyp": ""},
    ]
    assert evaluate(samples, GoldTranscriber()).n_empty == 0


class _RawASR:
    """Transcriber giả có chữ trước khi lọc: lượt cuối bị bịa (bộ lọc xoá hết), số viết bằng chữ số."""

    def warmup(self) -> None:
        """Không cần làm gì."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Mẫu s1: nghe đúng nhưng viết số bằng chữ số; mẫu s2: câu phụ đề bịa, sau khi lọc thành rỗng."""
        if sample.id == "s1":
            return Transcript(final="nhắn zalo lúc 8 giờ", history=(), seconds=0.1, raw=("nhắn zalo lúc 8 giờ",))
        return Transcript(final="", history=(), seconds=0.1, raw=("Hãy subscribe cho kênh lalaschool",))


def test_evaluate_nonum_wer_hallucination_and_terms() -> None:
    """`wer_nonum` bỏ khác biệt cách viết số, câu bịa đếm trên chữ trước khi lọc, `terms` đếm tên riêng viết đúng."""
    samples = [_sample("s1", "Nhắn Zalo lúc tám giờ", []), _sample("s2", "Gọi Zalo cho mẹ", [])]
    hyps: list[dict[str, Any]] = []
    rep = evaluate(samples, _RawASR(), hyps)
    assert rep.n_hallucinated == 1 and rep.n_empty == 1
    # s1 chỉ khác cách viết số nên không có lỗi; s2 rỗng: 4 từ bị xoá trên tổng 4 + 4 từ không phải số.
    assert rep.wer_nonum == 4 / 8
    assert rep.terms == {"zalo": [2, 1]}
    assert hyps[1]["raw"] == "Hãy subscribe cho kênh lalaschool"


class _Fixed:
    """Transcriber giả trả chữ cố định cho lượt cuối của từng mẫu."""

    def __init__(self, texts: dict[str, str]) -> None:
        self._texts = texts

    def warmup(self) -> None:
        """Không cần làm gì."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Trả chữ đã định cho mẫu `sample.id`."""
        return Transcript(final=self._texts[sample.id], history=(), seconds=0.1, raw=(self._texts[sample.id],))


def test_unk_only_counted_and_bad_ids() -> None:
    """Câu chỉ gồm `⁇` hoặc `<unk>` được đếm vào `n_unk` (và `n_empty`); `bad_ids` lấy đúng mẫu có câu hỏng."""
    from lifemate.eval.asr_bad_ids import bad_ids
    from lifemate.eval.eval_asr import is_unk_only

    assert is_unk_only("⁇") and is_unk_only(" ⁇ ⁇ ") and is_unk_only("<unk>")
    assert not is_unk_only("") and not is_unk_only("bật đèn ⁇")
    samples = [_sample("a", "bật đèn", []), _sample("b", "tắt đèn", [])]
    rep = evaluate(samples, _Fixed({"a": "⁇", "b": "tắt đèn"}))
    assert (rep.n_unk, rep.n_empty) == (1, 1)
    # `<unk>` cũng là câu rỗng: không thành từ "unk" khi chuẩn hoá, nên vẫn nằm trong `n_empty`.
    rep = evaluate(samples, _Fixed({"a": "<unk> <unk>", "b": "tắt đèn"}))
    assert (rep.n_unk, rep.n_empty, rep.wer) == (1, 1, 0.5)
    assert normalize_text("bật <UNK> đèn ⁇") == "bật đèn"
    rows = [
        {"id": "a", "ref": "x", "hyp": "⁇"},
        {"id": "a", "ref": "y", "hyp": "⁇"},
        {"id": "b", "ref": "z", "hyp": ""},
        {"id": "c", "ref": "w", "hyp": "w"},
    ]
    assert bad_ids(rows) == ["a"]
    assert bad_ids(rows, "empty") == ["a", "b"]


def test_evaluate_english_metrics() -> None:
    """Chỉ số tiếng Anh: chỉ câu có từ ngoại mới vào `en_mixed_wer`; recall theo nhóm và theo có/không trong câu mồi;
    từ của câu mồi bị viết ra mà không được nói thì đếm vào `prompt_insertions`.
    """
    samples = [
        _sample("s1", "Check mail rồi nhắn Zalo nha", []),
        _sample("s2", "Bật đèn phòng khách", []),
        _sample("s3", "Giá vàng SJC hôm nay", []),
    ]
    asr = _Fixed({"s1": "chắc mail rồi nhắn zalo nha", "s2": "bật đèn phòng khách zalo", "s3": "giá vàng sgc hôm nay"})
    rep = evaluate(samples, asr, prompt="Nhắn Zalo cho Lan.")
    assert rep.en_utterances == 2
    # s1: 1 lỗi / 6 từ; s3: 1 lỗi / 5 từ. Câu s2 không có từ ngoại nên không tính.
    assert rep.en_mixed_wer == pytest.approx(2 / 11)
    assert (rep.en_words, rep.en_words_hit) == (4, 2) and rep.en_word_recall == 0.5
    # "Check" đứng đầu câu và không gặp viết thường ở đâu trong tập nên xếp nhóm tên.
    assert rep.en_groups == {"common": [1, 1], "name": [2, 1], "acronym": [1, 0]}
    assert rep.en_prompt == {"in": [1, 1], "out": [3, 1]}
    assert rep.prompt_insertions == 1  # "zalo" ở s2
    assert rep.n_prompt_retries == 0
    assert evaluate(samples, asr).en_prompt == {}
    # Đáp án viết "ti vi", chữ ASR viết "tivi" như câu mồi: cùng một từ, không phải từ câu mồi chèn vào.
    tv = [_sample("t1", "Tắt cái ti vi đi", [])]
    assert evaluate(tv, _Fixed({"t1": "tắt cái tivi đi"}), prompt="Mở tivi lên.").prompt_insertions == 0


def test_rescore_replay_matches_hyps() -> None:
    """`ReplayTranscriber` dựng lại transcript từ `hyps`: chấm lại cho cùng số với lần chấm gốc."""
    from lifemate.eval.rescore_asr import ReplayTranscriber

    hist = (Turn(role="user", audio=None, audio_text="Check mail", text=None, tool_calls=(), tool_results=()),)
    samples = [_sample("s1", "Nhắn Zalo cho Lan", [], history=hist), _sample("s2", "Giá vàng SJC", [])]
    hyps: list[dict[str, Any]] = []
    asr = _Fixed({"s1": "nhắn dalo cho lan", "s2": "giá vàng sjc"})
    first = evaluate(samples, GoldTranscriber(), hyps)
    hyps = [
        {**h, "hyp": asr.transcribe(next(s for s in samples if s.id == h["id"])).final} if h["turn"] == "final" else h
        for h in hyps
    ]
    again = evaluate(samples, ReplayTranscriber(hyps))
    # Check, mail, Zalo, SJC; chỉ "Zalo" bị chép sai thành "dalo".
    assert first.en_words == again.en_words == 4
    assert again.en_words_hit == 3 and again.n_utterances == 3


def test_compare_counts_add_up_and_ci_contains_value() -> None:
    """Tử số và mẫu số chấm theo từng mẫu cộng lại bằng số của cả tập; khoảng tin cậy chứa giá trị đo được.

    Chênh lệch của hệ thống mốc với chính nó là 0.
    """
    from lifemate.eval.asr_compare import compare

    samples = [_sample(f"s{i}", t, []) for i, t in enumerate(["Check mail nha", "Bật đèn", "Giá vàng SJC", "Book xe"])]
    asr = _Fixed({"s0": "chắc mail nha", "s1": "bật đèn", "s2": "giá vàng sjc", "s3": "búc xe"})
    hyps: list[dict[str, Any]] = []
    evaluate(samples, asr, hyps)
    full = evaluate(samples, asr)
    res = compare(samples, {"a": (hyps, ""), "b": (hyps, "")}, rounds=50)
    assert res["a"]["report"]["wer"] == pytest.approx(full.wer)
    assert res["a"]["report"]["en_word_recall"] == pytest.approx(full.en_word_recall)
    lo, hi = res["a"]["ci"]["en_word_recall"]
    assert lo <= full.en_word_recall <= hi
    assert res["b"]["diff_ci"]["wer"] == [0.0, 0.0]


def test_en_word_precision_and_f1() -> None:
    """Từ tiếng Anh ASR viết thêm, kể cả ở câu thuần Việt, hạ độ chính xác; F1 là trung bình điều hoà."""
    from lifemate.eval.eval_asr import f1

    samples = [_sample("s0", "Check mail nha", []), _sample("s1", "Bật đèn phòng khách", [])]
    # s0: đúng "check", sai "mail" thành "meo"; s1: "phòng" bị viết thành "phone".
    rep = evaluate(samples, _Fixed({"s0": "check meo nha", "s1": "bật đèn phone khách"}))
    assert (rep.en_words, rep.en_words_hit) == (2, 1)
    assert (rep.en_hyp_words, rep.en_hyp_hits) == (2, 1)
    assert rep.en_word_precision == pytest.approx(0.5) and rep.en_word_f1 == pytest.approx(0.5)
    assert f1(0.0, 0.0) == 0.0 and f1(1.0, 0.5) == pytest.approx(2 / 3)


def test_compare_f1_from_pooled_counts_and_prepare() -> None:
    """F1 trong `compare` tính từ độ phủ và độ chính xác đã cộng dồn; `prepared` đổi số đọc thành chữ số."""
    from lifemate.eval.asr_compare import compare, prepared

    samples = [_sample("s0", "Check mail nha", []), _sample("s1", "Bật đèn", [])]
    asr = _Fixed({"s0": "check meo nha", "s1": "bật đèn phone"})
    hyps: list[dict[str, Any]] = []
    full = evaluate(samples, asr, hyps)
    res = compare(samples, {"a": (hyps, "")}, rounds=20)
    assert res["a"]["report"]["en_word_f1"] == pytest.approx(full.en_word_f1)
    lo, hi = res["a"]["ci"]["en_word_f1"]
    assert lo <= hi
    row = prepared([{"id": "s0", "turn": "final", "hyp": "Hẹn lúc hai giờ."}])[0]
    assert row["hyp"] == "hẹn lúc 2 giờ" and row["raw"] == row["hyp"]


def test_digitized_wer_number_f1_and_entities_all_turns() -> None:
    """Đổi số ở cả hai phía bỏ lỗi cách viết số; F1 số so từ có chữ số; giá trị nói ở lượt trước cũng được xét."""
    hist = (
        Turn(role="user", audio=None, audio_text="Đặt bàn ở Đà Nẵng", text=None, tool_calls=(), tool_results=()),
        Turn(role="assistant", audio=None, audio_text=None, text="Vâng", tool_calls=(), tool_results=()),
    )
    s = _sample("s0", "Lúc tám giờ nhé", [{"name": "x.book", "arguments": {"city": "Đà Nẵng", "hour": 8}}], hist)

    class _Hist(_Fixed):
        def transcribe(self, sample: Sample) -> Transcript:
            """Lượt cuối nghe đúng số, lượt trước nghe sai tên thành phố."""
            return Transcript(final="lúc 8 giờ nhé", history=("đặt bàn ở đà lạt", None), seconds=0.1)

    rep = evaluate([s], _Hist({}))
    assert rep.final_wer_digits == 0.0 and rep.wer > 0.0
    assert (rep.number_recall, rep.number_precision, rep.number_f1) == (1.0, 1.0, 1.0)
    # "8" có trong đáp án đã đổi số và trong chữ ASR; "da nang" nói ở lượt trước, ASR viết "đà lạt".
    assert rep.counts["entity_all_total"] == 2 and rep.counts["entity_all_missed"] == 1


def test_sample_seconds_sums_parts(tmp_path: Any) -> None:
    """Thời gian của một hệ thống là tổng các phần theo mẫu; mẫu thiếu ở một phần bị bỏ."""
    from lifemate.eval.asr_compare import sample_seconds

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"id": "x", "w": {"E2": 1.0}}\n{"id": "y", "w": {"E2": 2.0}}\n', encoding="utf-8")
    b.write_text('{"id": "x", "s": 0.5}\n', encoding="utf-8")
    out = sample_seconds([f"m={a}:w.E2", f"m={b}:s"])
    assert out["m"]["n"] == 1 and out["m"]["max"] == 1.5


def test_read_number_round_trips_and_canonical_numbers() -> None:
    """`read_number` đọc ngược đúng với `parse_number`; mọi cách viết cùng một số cho cùng một dạng chung."""
    from lifemate.backends.asr_numbers import parse_number
    from lifemate.eval.eval_asr import digitized_text, read_number

    for n in [*range(0, 1200), 2005, 10_000, 15_500, 21_000, 105_000, 500_000, 1_000_000, 2_500_000, 1_234_567]:
        assert parse_number(read_number(str(n)).split()) == n, n
    same = {digitized_text(x) for x in ("giá 50.000 đồng", "giá 50 nghìn đồng", "giá năm mươi nghìn đồng")}
    assert len(same) == 1
    assert digitized_text("lúc 9h30") == digitized_text("lúc chín giờ ba mươi")
    assert digitized_text("số 095-0536-0367") == digitized_text("số không chín năm không năm ba sáu không ba sáu bảy")
    assert digitized_text("mua 2 cái") == digitized_text("mua hai cái")
