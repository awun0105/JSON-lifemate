"""Backend VieGem-ASR (`5CD-AI/VieGem-ASR`): mô hình CTC 0,36B trên encoder âm thanh của EmbeddingGemma 2.

Nội dung chính:
- `ViegemASR`: engine `viegem` của `[asr].engine`; giải mã tham lam hoặc tìm kiếm chùm có KenLM.
- `split_quiet`: cắt câu dài thành đoạn ngắn tại chỗ im lặng nhất, như app demo của tác giả.
- `load_viegem`: nạp code và trọng số của repo mô hình ở đúng commit, kiểm tra khoá thiếu và thừa.

Cách hoạt động:
- Repo mô hình chứa code riêng (`asr/*.py`, đã đọc lại ở commit `95eae70b`: chỉ có PyTorch và transformers, không có
  `eval`, `exec` hay gọi mạng). Code được nạp như một gói tên `viegem_asr_code` từ thư mục snapshot, để không trùng tên
  với gói nào khác.
- `viegem_asr.pt` chỉ chứa phần đã fine-tune; các tầng encoder còn lại lấy từ `google/embeddinggemma-2`. Mô hình gốc
  được tải ở đúng revision (`generate_kwargs.encoder_revision`) và đường dẫn snapshot thay cho `encoder_id` của
  checkpoint, nên không có lần tải nào trượt sang bản mới. Checkpoint nạp bằng `torch.load(weights_only=True)`.
- `load_state_dict(strict=False)` của tác giả bỏ qua im lặng khoá thiếu; ở đây mọi khoá thiếu phải thuộc `encoder.`
  (phần lấy từ mô hình gốc) và không được có khoá thừa, không thì báo lỗi.
- Chế độ không streaming (nhìn cả đoạn). Kiểu số: bf16 autocast trên GPU sm80 trở lên, float32 trên GPU cũ hơn (T4
  không có bf16 phần cứng) và CPU; `[asr].dtype` khác `auto` thì theo đó.
- Đầu ra như tác giả mô tả: chữ thường, không dấu câu, số viết bằng chữ, từ tiếng Anh giữ chữ Latin.

`generate_kwargs` nhận: `decoder` (`"greedy"` hoặc `"lm"`), `encoder_id`, `encoder_revision`, `max_segment_s` (20),
`lm_alpha` (0,35), `lm_beta` (1,5), `beam_width` (32), `token_min_logp` (−5), `beam_prune_logp` (−10). Giá trị mặc định
lấy đúng từ app và README của tác giả; khoá lạ gây `ValueError`.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from lifemate.backends.asr import read_wav
from lifemate.core.config import ASRConfig

ENCODER_ID = "google/embeddinggemma-2"
ENCODER_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
_PACKAGE = "viegem_asr_code"


def split_quiet(x: Any, sr: int, max_s: float = 20.0, search_s: float = 5.0) -> list[tuple[int, int]]:
    """Cắt `x` trước mỗi `max_s` giây tại 50 ms im lặng nhất trong `search_s` giây cuối (như `app.py` của tác giả);
    bỏ đoạn ngắn hơn 0,1 s."""
    import numpy as np

    out, s, n, win = [], 0, int(max_s * sr), int(0.05 * sr)
    while len(x) - s > n:
        lo, hi = s + n - int(search_s * sr), s + n
        e = np.convolve(x[lo:hi] ** 2, np.ones(win), mode="valid")
        cut = lo + int(np.argmin(e)) + win // 2
        out.append((s, cut))
        s = cut
    out.append((s, len(x)))
    return [(a, b) for a, b in out if b - a > sr // 10]


def _import_code(repo: Path) -> Any:
    """Nạp `repo/asr` như gói `viegem_asr_code` (giữ được import tương đối `from .features import ...`)."""
    if _PACKAGE in sys.modules:
        return sys.modules[_PACKAGE]
    spec = importlib.util.spec_from_file_location(
        _PACKAGE, repo / "asr" / "__init__.py", submodule_search_locations=[str(repo / "asr")]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load VieGem code from {repo / 'asr'}")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def load_viegem(model: str, revision: str, encoder_id: str, encoder_revision: str) -> dict[str, Any]:
    """Tải snapshot (hoặc dùng thư mục có sẵn), nạp code, checkpoint và mô hình; trả các thành phần cần để giải mã.

    Lỗi:
        RuntimeError: khi có khoá thiếu ngoài `encoder.` hoặc có khoá thừa.
    """
    import torch
    from huggingface_hub import snapshot_download

    repo = Path(model) if Path(model).is_dir() else Path(
        snapshot_download(model, revision=revision or None, allow_patterns=["viegem_asr.pt", "asr/*", "lm/*"])
    )  # fmt: skip
    enc_path = encoder_id if Path(encoder_id).is_dir() else snapshot_download(encoder_id, revision=encoder_revision)
    _import_code(repo)
    m = importlib.import_module(f"{_PACKAGE}.model")
    text = importlib.import_module(f"{_PACKAGE}.text")
    from transformers import AutoTokenizer

    ck = torch.load(repo / "viegem_asr.pt", map_location="cpu", weights_only=True)
    cfg = m.ModelConfig.from_dict({**ck["model_config"], "encoder_id": str(enc_path)})
    units = text.Units(ck["gemma_ids"])
    tok = AutoTokenizer.from_pretrained(str(enc_path))
    net = m.CTCModel(cfg, units.gemma_ids)
    res = net.load_state_dict(ck["model"], strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(
            f"VieGem checkpoint mismatch: missing {bad_missing[:5]} (+{len(bad_missing)}), "
            f"unexpected {list(res.unexpected_keys)[:5]} (+{len(res.unexpected_keys)})"
        )
    return {
        "repo": repo,
        "model": net,
        "units": units,
        "tokenizer": tok,
        "greedy_decode": m.greedy_decode,
        "pieces": ["", "", *tok.convert_ids_to_tokens(units.gemma_ids)],
        "missing_encoder_keys": len(res.missing_keys),
        "n_params": sum(p.numel() for p in net.parameters()),
    }


class ViegemASR:
    """Engine `viegem` (xem docstring module).

    Tham số:
        cfg: mục `[asr]` (`model` là repo hoặc thư mục snapshot của VieGem, `revision` là commit).

    Thuộc tính:
        dtype: `"bfloat16"` khi chạy autocast bf16, còn lại `"float32"`.
        n_params: tổng tham số của mô hình đã nạp (encoder âm thanh và phần đọc).
    """

    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "decoder": "greedy",
        "encoder_id": ENCODER_ID,
        "encoder_revision": ENCODER_REVISION,
        "max_segment_s": 20.0,
        "lm_alpha": 0.35,
        "lm_beta": 1.5,
        "beam_width": 32,
        "token_min_logp": -5.0,
        "beam_prune_logp": -10.0,
    }
    # Các khoá chung của config kiểu Whisper, không áp dụng cho CTC.
    _IGNORED: ClassVar[frozenset[str]] = frozenset(
        {"language", "task", "temperature", "beam_size", "condition_on_previous_text"}
    )

    def __init__(self, cfg: ASRConfig) -> None:
        gk = dict(cfg.generate_kwargs)
        bad = sorted(k for k in gk if k not in self._IGNORED | self._DEFAULTS.keys())
        if bad:
            raise ValueError(f"viegem: unknown generate_kwargs {bad}")
        self._kw = {**self._DEFAULTS, **{k: v for k, v in gk.items() if k in self._DEFAULTS}}
        if self._kw["decoder"] not in ("greedy", "lm"):
            raise ValueError(f"viegem: decoder must be greedy|lm, got {self._kw['decoder']!r}")
        self._cfg = cfg
        import torch  # sau bước kiểm tra tham số, để cấu hình sai báo lỗi cả khi chưa cài torch

        on_gpu = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        self._device = "cuda" if on_gpu else "cpu"
        sm80 = on_gpu and torch.cuda.get_device_capability(0) >= (8, 0)
        self.dtype = cfg.dtype if cfg.dtype != "auto" else ("bfloat16" if sm80 else "float32")
        parts = load_viegem(cfg.model, cfg.revision, self._kw["encoder_id"], self._kw["encoder_revision"])
        self._net = parts["model"].to(self._device).eval()
        self._units, self._tok, self._greedy = parts["units"], parts["tokenizer"], parts["greedy_decode"]
        self._pieces = parts["pieces"]
        self.n_params = parts["n_params"]
        self._lm = None
        if self._kw["decoder"] == "lm":
            lm_mod = importlib.import_module(f"{_PACKAGE}.lm_decode")
            repo = parts["repo"]
            words = (repo / "lm" / "viegem_lm.unigrams.txt").read_text(encoding="utf-8").split()
            self._lm = lm_mod.LMDecoder(
                str(repo / "lm" / "viegem_lm.bin"), unigrams=words, alpha=self._kw["lm_alpha"], beta=self._kw["lm_beta"]
            )

    def _logits(self, wav: Any) -> tuple[Any, Any]:
        import torch

        x = torch.from_numpy(wav)[None].to(self._device)
        lens = torch.tensor([x.shape[1]], device=self._device)
        use_amp = self._device == "cuda" and self.dtype in ("bfloat16", "float16")
        amp_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits, out_lens = self._net(x, lens, streaming=False)
        return logits, out_lens

    def top_logprobs(self, wav: Any, k: int = 32) -> list[dict[str, Any]]:
        """Mỗi đoạn của câu: chữ giải mã tham lam và `k` token có log xác suất cao nhất ở mỗi khung (mảng `ids`, `lp`
        kích thước (T, k)), để giải mã lại bằng KenLM trên CPU mà không chạy lại mô hình."""
        import torch

        out = []
        for a, b in split_quiet(wav, self._cfg.sample_rate, float(self._kw["max_segment_s"])):
            logits, lens = self._logits(wav[a:b])
            lp = torch.log_softmax(logits[0, : int(lens[0])].float(), -1)
            top = lp.topk(k, dim=-1)
            greedy = str(self._units.decode(self._tok, self._greedy(logits, lens)[0]))
            out.append({"greedy": greedy, "ids": top.indices.cpu().numpy(), "lp": top.values.cpu().numpy()})
        return out

    def _segment(self, wav: Any) -> str:
        import torch

        logits, out_lens = self._logits(wav)
        if self._lm is None:
            return str(self._units.decode(self._tok, self._greedy(logits, out_lens)[0]))
        lp = torch.log_softmax(logits[0, : int(out_lens[0])].float(), -1).cpu().numpy()
        best = self._lm.decode(
            lp, self._pieces, beam_width=int(self._kw["beam_width"]), token_min_logp=float(self._kw["token_min_logp"]),
            beam_prune_logp=float(self._kw["beam_prune_logp"]),
        )  # fmt: skip
        return str(self._units.decode(self._tok, best))

    def transcribe_audio(self, wav: Any) -> str:
        """Chép một câu (mảng float32 16 kHz): cắt đoạn ≤ `max_segment_s` giây rồi nối chữ các đoạn."""
        sr = self._cfg.sample_rate
        texts = [self._segment(wav[a:b]) for a, b in split_quiet(wav, sr, float(self._kw["max_segment_s"]))]
        return " ".join(t for t in texts if t).strip()

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Chép lần lượt từng file WAV theo thứ tự."""
        return [self.transcribe_audio(read_wav(p, self._cfg.sample_rate)) for p in wav_paths]

    def warmup(self) -> None:
        """Chạy một lần trên 1 giây im lặng."""
        import numpy as np

        self.transcribe_audio(np.zeros(self._cfg.sample_rate, dtype=np.float32))
