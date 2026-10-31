"""Engine ASR ghép hai mô hình cho hệ thống thật: ChunkFormer RNNT làm nền, whisper-large-v3 bổ sung từ tiếng Anh,
bộ quyết định H chọn ở từng vùng khác nhau, rồi đổi số đọc thành chữ số (benchmark ASR lần 3 mục 7.2 và lần 4 mục 3).

Nội dung chính:
- `MergeWeights`: nội dung file `configs/asr_merge/<phiên bản>.json` (xuất bởi `asr_v4_report export-h`).
- `MergedASR`: `Transcriber` chạy đúng chuỗi xử lý đã chốt cho một mẫu; thay `ASRTranscriber` khi
  `[asr].merge_weights` khác rỗng.

Cách hoạt động (mỗi mẫu, mỗi câu người dùng theo thứ tự lịch sử rồi lượt cuối):
1. Mô hình chính (`[asr]`, ChunkFormer RNNT) chép mọi câu qua `ASRTranscriber` (bộ lọc câu bịa như thường).
2. Mô hình phụ (`[asr].secondary`, whisper-large-v3, tham lam) chép từng câu với câu mồi E2 của câu đó
   (`asr_prompt.build_sample_prompts`: câu mồi cố định, tên tiếng Việt của app đã cài, lời trợ lý liền trước). Lớp
   bảo vệ g4 so với **phần cố định**: câu chép lại không câu mồi khi lặp 4 từ của câu mồi hoặc lặp vô hạn
   (`asr_guard.needs_retry`). Câu bị bộ lọc câu bịa cắt hoặc rỗng thì bỏ, giữ nguyên chữ RNNT.
3. Hai câu được đặt cạnh nhau (`asr_merge.split_regions`); các vùng mà large-v3 có từ tiếng Anh mới là vùng cần
   quyết định (`candidate_regions`). Nếu có, large-v3 không câu mồi chấm điểm âm thanh câu A (giữ RNNT) và từng câu B
   (lấy large-v3 ở đúng vùng đó) bằng `score_texts` (một lượt encoder, mỗi câu một lượt decoder ép chuỗi).
4. Bộ phân loại H (`asr_v3_analysis.h_decider`, đặc trưng `region_features`) quyết định từng vùng; câu được ghép
   lại bằng `assemble`, đúng như `apply_decider` của phần phân tích offline.
5. `asr_numbers.digitize` đổi số đọc thành chữ số (L1).

Kết quả của engine này trên chữ đã lưu phải trùng từng byte với `asr_v4_report` (`digitized(apply_decider(...))`);
test `tests/test_asr_merged.py` kiểm tra điều đó bằng hai backend giả phát lại chữ và điểm của các phiên K2, K3.

Lưu ý:
- Mô hình phụ phải có `transcribe_one(audio, prompt)` và `score_texts(audio, texts)` (`FasterWhisperASR`).
- Từ điển ngữ cảnh của mẫu (`asr_merge.sample_context_words`) chỉ dùng chữ có sẵn trong đầu vào lúc thi.
- Thời gian từng bước cộng dồn ở `stats` để ghi vào trace và báo cáo thời gian.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from lifemate.backends.asr import ASRBackend, ASRTranscriber, make_backend, read_wav
from lifemate.backends.asr_guard import needs_retry
from lifemate.backends.asr_numbers import digitize
from lifemate.backends.asr_prompt import build_sample_prompts
from lifemate.backends.asr_sanitizer import sanitize
from lifemate.core.config import ASRConfig, secondary_config
from lifemate.core.paths import repo_path
from lifemate.core.types import Sample, Transcript
from lifemate.eval.asr_merge import assemble, candidate_regions, sample_context_words, split_regions
from lifemate.eval.asr_merge_v2 import candidate_pairs_v2, prepare, region_indices, split_regions_v2, vi_features
from lifemate.eval.asr_v3_analysis import FEATURES, LogisticModel, UtteranceRegions, h_decider, predict
from lifemate.eval.candidate_scoring import candidate_pairs
from lifemate.eval.foreign_words import find_foreign_words

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MergeWeights:
    """File trọng số của engine ghép.

    Thuộc tính:
        version: tên phiên bản (tên file không đuôi).
        prompt_file: file câu mồi cố định (đường dẫn tính từ gốc repo).
        prompt_variant: biến thể câu mồi theo mẫu (`asr_prompt.PROMPT_VARIANTS`).
        guard_ngram, guard_max_compression: tham số lớp bảo vệ câu mồi.
        model: bộ phân loại H đã học.
        reliability: độ tin cậy từng từ của câu mồi (đặc trưng `min_reliability`).
        threshold: ngưỡng xác suất lấy large-v3.
        digitize: có đổi số đọc thành chữ số không.
        pipeline: 1 là bản 1 (vùng tiếng Anh, đổi số sau khi ghép); 2 là bản 2 (`asr_merge_v2`: đổi số trước khi gióng,
            vùng chỉ khác dấu thanh, H-Việt cho vùng chữ Việt).
        vi_model, vi_threshold: H-Việt và ngưỡng (chỉ bản 2).
    """

    version: str
    prompt_file: str
    prompt_variant: str
    guard_ngram: int
    guard_max_compression: float
    model: LogisticModel
    reliability: dict[str, float]
    threshold: float = 0.5
    digitize: bool = True
    pipeline: int = 1
    vi_model: LogisticModel | None = None
    vi_threshold: float = 0.5

    @classmethod
    def load(cls, path: Path) -> MergeWeights:
        """Đọc file JSON do `asr_v4_report export-h` ghi; báo lỗi khi thứ tự đặc trưng khác `FEATURES` hiện hành."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if list(raw["features"]) != list(FEATURES):
            raise ValueError(f"{path}: features {raw['features']} differ from current {list(FEATURES)}")
        model = LogisticModel(**raw["model"])
        if len(model.weights) != len(FEATURES) + 1:
            raise ValueError(f"{path}: expected {len(FEATURES) + 1} weights, got {len(model.weights)}")
        return cls(
            version=str(raw.get("version") or Path(path).stem),
            prompt_file=str(raw["prompt_file"]),
            prompt_variant=str(raw.get("prompt_variant", "E2")),
            guard_ngram=int(raw["guard"]["ngram"]),
            guard_max_compression=float(raw["guard"]["max_compression"]),
            model=model,
            reliability={str(k): float(v) for k, v in raw["reliability"].items()},
            threshold=float(raw.get("threshold", 0.5)),
            digitize=bool(raw.get("digitize", True)),
            pipeline=int(raw.get("pipeline", 1)),
            vi_model=_vi_model(raw, path),
            vi_threshold=float(raw.get("vi_threshold", 0.5)),
        )


def _vi_model(raw: Mapping[str, Any], path: Path) -> LogisticModel | None:
    """H-Việt của file bản 2; báo lỗi khi thứ tự tín hiệu khác `asr_merge_v2.VI_FEATURES`."""
    if int(raw.get("pipeline", 1)) != 2:
        return None
    from lifemate.eval.asr_merge_v2 import VI_FEATURES

    if list(raw["vi_features"]) != list(VI_FEATURES):
        raise ValueError(f"{path}: vi_features {raw['vi_features']} differ from current {list(VI_FEATURES)}")
    return LogisticModel(**raw["vi_model"])


@dataclass
class MergeStats:
    """Số đếm cộng dồn của engine ghép, để ghi trace và báo cáo thời gian."""

    samples: int = 0
    utterances: int = 0
    prompt_retries: int = 0
    secondary_dropped: int = 0  # câu large-v3 rỗng hoặc bị bộ lọc câu bịa cắt
    regions: int = 0
    regions_taken: int = 0
    primary_s: float = 0.0
    secondary_s: float = 0.0
    score_s: float = 0.0
    last: dict[str, Any] = field(default_factory=dict)  # số đo của mẫu gần nhất


class MergedASR:
    """`Transcriber` ghép RNNT và large-v3 theo cấu hình đã chốt (xem docstring module).

    Tham số:
        cfg: mục `[asr]` (mô hình chính), có `merge_weights` và `secondary`.
        data_root: thư mục dữ liệu cuộc thi.
        split: tên split; file âm thanh và catalog nằm trong `data_root/split`.
        primary: backend mô hình chính có sẵn (test); `None` thì tạo từ `cfg`.
        secondary: backend mô hình phụ có sẵn (test); `None` thì tạo từ `cfg.secondary`.
        catalog_apps: mã app -> mục catalog; `None` thì đọc `load_catalog(data_root, split)`.
        read_audio: hàm đọc file WAV `(đường dẫn, tần số) -> mảng`; mặc định `asr.read_wav` (test truyền hàm giả).

    Thuộc tính:
        source: luôn `"asr"`.
        weights: trọng số đã nạp.
        stats: số đếm cộng dồn.

    Lỗi:
        ValueError: khi `cfg.merge_weights` rỗng hoặc mô hình phụ không chấm điểm được.
    """

    source = "asr"

    def __init__(
        self,
        cfg: ASRConfig,
        *,
        data_root: Path,
        split: str,
        primary: ASRBackend | None = None,
        secondary: Any | None = None,
        catalog_apps: Mapping[str, Mapping[str, Any]] | None = None,
        read_audio: Callable[[Path, int], Any] | None = None,
    ) -> None:
        if not cfg.merge_weights:
            raise ValueError("MergedASR needs asr.merge_weights")
        self._cfg = cfg
        self.weights = MergeWeights.load(repo_path(cfg.merge_weights))
        self._fixed_prompt = repo_path(self.weights.prompt_file).read_text(encoding="utf-8").strip()
        self._prompt_words = frozenset(w.folded for w in find_foreign_words(self._fixed_prompt))
        self._primary = ASRTranscriber(cfg, data_root=data_root, split=split, backend=primary)
        self._secondary: Any = secondary if secondary is not None else make_backend(secondary_config(cfg))
        for name in ("transcribe_one", "score_texts"):
            if not hasattr(self._secondary, name):
                raise ValueError(f"asr.secondary: engine must provide {name}() (faster_whisper)")
        if catalog_apps is None:
            from lifemate.io.zaic import load_catalog

            catalog_apps = {str(a["id"]): a for a in load_catalog(data_root, split)["apps"]}
        self._catalog = catalog_apps
        self._read_audio = read_audio if read_audio is not None else read_wav
        self.stats = MergeStats()
        self._warmed = False
        self._context: dict[str, frozenset[str]] = {}

    def transcribe(self, sample: Sample) -> Transcript:
        """Chép mọi câu người dùng của `sample` theo chuỗi xử lý đã chốt; `raw` là chữ RNNT trước bộ lọc."""
        start = time.perf_counter()
        primary = self._primary.transcribe(sample)
        t_primary = time.perf_counter() - start
        rnnt_texts = [h for h, turn in zip(primary.history, sample.history, strict=True) if turn.role == "user"]
        texts = [str(t) for t in rnnt_texts] + [primary.final]
        keys = [str(i) for i, turn in enumerate(sample.history) if turn.role == "user"] + ["final"]
        paths = self._primary.wav_paths(sample)
        prompts = build_sample_prompts(sample, self._catalog, self._fixed_prompt, self.weights.prompt_variant)
        context = {sample.id: sample_context_words(sample, self._catalog)}
        self._context = context
        decide = h_decider(self.weights.model, context, self._prompt_words, self.weights.reliability)
        sr = self._cfg.sample_rate
        counts = {"retries": 0, "dropped": 0, "regions": 0, "taken": 0}
        t_secondary = t_score = 0.0
        out: list[str] = []
        for i, (key, text, path, prompt) in enumerate(zip(keys, texts, paths, prompts, strict=True)):
            is_history = i < len(texts) - 1
            audio = self._read_audio(path, sr)
            t0 = time.perf_counter()
            raw, scores = self._secondary.transcribe_one(audio, prompt)
            hyp = sanitize(raw, is_history=is_history)
            if needs_retry(
                raw, hyp, self._fixed_prompt, self.weights.guard_ngram, self.weights.guard_max_compression
            ):
                raw, scores = self._secondary.transcribe_one(audio, "")
                hyp = sanitize(raw, is_history=is_history)
                counts["retries"] += 1
            t_secondary += time.perf_counter() - t0
            merged = text
            if not hyp.strip() or " ".join(raw.split()) != " ".join(hyp.split()):
                counts["dropped"] += 1
            else:
                t1 = time.perf_counter()
                merge = self._merge_v2 if self.weights.pipeline == 2 else self._merge
                merged, n_regions, n_taken = merge(sample.id, key, text, hyp, audio, scores, decide)
                t_score += time.perf_counter() - t1
                counts["regions"] += n_regions
                counts["taken"] += n_taken
            if self.weights.pipeline == 2:
                # Bản 2 đổi số trước khi gióng; câu không ghép (large-v3 bị bỏ) vẫn phải qua cùng bước chuẩn bị.
                out.append(merged if merged is not text else prepare(text))
            else:
                out.append(digitize(merged) if self.weights.digitize else merged)
        it = iter(out[:-1])
        history = tuple(next(it) if t.role == "user" else None for t in sample.history)
        self._account(sample.id, len(texts), counts, t_primary, t_secondary, t_score)
        return Transcript(final=out[-1], history=history, seconds=time.perf_counter() - start, raw=primary.raw)

    def _merge(
        self,
        sample_id: str,
        key: str,
        primary_text: str,
        secondary_text: str,
        audio: Any,
        scores: Mapping[str, float],
        decide: Any,
    ) -> tuple[str, int, int]:
        """Ghép một câu: (chữ sau ghép, số vùng cần quyết định, số vùng lấy large-v3)."""
        parts = split_regions(primary_text, secondary_text)
        cands = candidate_regions(parts)
        if not cands:
            return primary_text, 0, 0
        pairs = candidate_pairs(primary_text, secondary_text)
        scored = self._secondary.score_texts(audio, [pairs[0].text_a] + [p.text_b for p in pairs])
        long = len(audio) > 30 * self._cfg.sample_rate
        acoustic = {
            p.part_index: (float(scored[0][0]), float(sb), long) for p, (sb, _n) in zip(pairs, scored[1:], strict=True)
        }
        u = UtteranceRegions(
            sample_id,
            key,
            "",
            primary_text,
            secondary_text,
            parts,
            cands,
            [None] * len(cands),
            {k: float(v) for k, v in scores.items()},
            acoustic,
        )
        cand = set(cands)
        taken: set[int] = set()

        def take(k: int, region: Any) -> bool:
            ok = k in cand and bool(decide(u, k, region))
            if ok:
                taken.add(k)
            return ok

        return assemble(parts, take), len(cands), len(taken)

    def _merge_v2(
        self,
        sample_id: str,
        key: str,
        primary_text: str,
        secondary_text: str,
        audio: Any,
        scores: Mapping[str, float],
        decide: Any,
    ) -> tuple[str, int, int]:
        """Ghép bản 2: `prepare` hai bản, tách vùng có giữ dấu thanh, chấm điểm âm thanh mọi vùng, vùng `en` theo H-Anh,
        vùng `vi` theo H-Việt. Trả (chữ sau ghép, số vùng, số vùng lấy large-v3)."""
        a, b = prepare(primary_text), prepare(secondary_text)
        parts = split_regions_v2(a, b)
        idx = region_indices(parts)
        if not idx["en"] and not idx["vi"]:
            return a, 0, 0
        pairs = candidate_pairs_v2(parts)
        scored = self._secondary.score_texts(audio, [pairs[0].text_a] + [p.text_b for p in pairs])
        long = len(audio) > 30 * self._cfg.sample_rate
        full = {
            p.part_index: (float(scored[0][0]), float(sb), int(scored[0][1]), int(nb), long)
            for p, (sb, nb) in zip(pairs, scored[1:], strict=True)
        }
        u = UtteranceRegions(
            sample_id, key, "", a, b, parts, idx["en"], [None] * len(idx["en"]),
            {k: float(v) for k, v in scores.items()},
            {k: (v[0], v[1], v[4]) for k, v in full.items() if k in idx["en"]},
        )  # fmt: skip
        context = self._context.get(sample_id, frozenset())
        en_set, vi_set = set(idx["en"]), set(idx["vi"])
        vi_model = self.weights.vi_model
        taken: set[int] = set()

        def take(k: int, region: Any) -> bool:
            if k in en_set:
                ok = bool(decide(u, k, region))
            elif k in vi_set and vi_model is not None:
                feats = vi_features(parts, k, full.get(k), context, u.v3_scores, key == "final")
                ok = predict(vi_model, feats) >= self.weights.vi_threshold
            else:
                ok = False
            if ok:
                taken.add(k)
            return ok

        return assemble(parts, take), len(idx["en"]) + len(idx["vi"]), len(taken)

    def _account(
        self, sample_id: str, n: int, counts: Mapping[str, int], t_primary: float, t_secondary: float, t_score: float
    ) -> None:
        """Cộng dồn số đếm và ghi log một dòng cho mẫu."""
        s = self.stats
        s.samples += 1
        s.utterances += n
        s.prompt_retries += counts["retries"]
        s.secondary_dropped += counts["dropped"]
        s.regions += counts["regions"]
        s.regions_taken += counts["taken"]
        s.primary_s += t_primary
        s.secondary_s += t_secondary
        s.score_s += t_score
        s.last = {
            "utterances": n,
            **counts,
            "primary_s": round(t_primary, 3),
            "secondary_s": round(t_secondary, 3),
            "score_s": round(t_score, 3),
        }
        log.info("asr_merged", sample_id=sample_id, **s.last)

    def warmup(self) -> None:
        """Khởi động cả hai mô hình, chỉ một lần."""
        if not self._warmed:
            self._primary.warmup()
            self._secondary.warmup()
            self._warmed = True
