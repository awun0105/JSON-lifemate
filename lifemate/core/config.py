"""Đọc config chạy và profile môi trường từ file TOML, kiểm tra kỹ rồi trả về dataclass (SPEC 4.14, 4.16).

Nội dung chính:
- `LLMConfig`, `AgentConfig`, `ASRConfig`: ba mục `[llm]`, `[agent]`, `[asr]` của config chạy.
- `Config`: gộp ba mục trên thành config của một lần chạy.
- `EnvProfile`: vị trí dữ liệu, thư mục run và cache trên một máy (`configs/env/<tên>.toml`).
- `load_config`, `load_profile`: đọc TOML, kiểm tra khoá và giá trị, trả về dataclass tương ứng.
- `ConfigError`: lỗi báo khi config sai.

Cách hoạt động:
- Config chạy (model, tham số agent, ASR) tách riêng khỏi profile môi trường (đường dẫn, nguồn token). Nhờ vậy
  cùng một config chạy được trên Kaggle, Colab hay máy cá nhân, chỉ cần đổi profile.
- Kiểm tra chặt ngay lúc nạp: khoá lạ hoặc thiếu khoá đều báo `ConfigError`, không để lỗi lộ ra giữa lần chạy.
  Chỉ các khoá trong tập tuỳ chọn (ví dụ `_ASR_OPTIONAL`) mới được bỏ trống và lấy giá trị mặc định.
- Các dataclass đều `frozen=True, slots=True`: config không đổi sau khi nạp, mọi chỗ đọc đều thấy cùng giá trị.

Lưu ý:
- Token Hugging Face (HF) không nằm trong file. Profile chỉ ghi nguồn của token (`env:`, `kaggle_secret:`,
  `colab_userdata:`), còn token được đọc lúc chạy và không bao giờ ghi ra log.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from lifemate.core.paths import repo_path


class ConfigError(ValueError):
    """Lỗi config: khoá bị thiếu, khoá không có trong dataclass, hoặc giá trị không hợp lệ."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Mục `[llm]`: backend, model và tham số sinh văn bản.

    Thuộc tính:
        backend: `"vllm"` cho model thật, `"fake"` cho test.
        model: tên model trên Hugging Face; bắt buộc khi `backend = "vllm"`.
        revision: commit cố định của model.
        dtype: `"auto"`, `"float16"` hoặc `"bfloat16"`; `auto` chọn theo GPU (xem `resolve_dtype`).
        max_model_len: độ dài context tối đa, tính bằng token.
        gpu_memory_utilization: tỉ lệ bộ nhớ GPU vLLM được dùng, trong khoảng (0, 1].
        seed: seed ngẫu nhiên khi sinh văn bản.
        max_tokens: số token tối đa sinh ra trong một bước.
        batch_invariant: bật chế độ của vLLM giúp kết quả không phụ thuộc kích thước batch; cần GPU sm80 trở lên.
        warmup_runs: số lần chạy làm nóng trước khi đo.
    """

    backend: str
    model: str
    revision: str
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    seed: int
    max_tokens: int
    batch_invariant: bool
    warmup_runs: int


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Mục `[agent]`: giới hạn của vòng lặp agent, đường dẫn system prompt và danh sách guard.

    Thuộc tính:
        max_steps: số bước LLM tối đa cho một mẫu.
        max_retries: số lần thử lại tối đa khi đầu ra LLM sai định dạng hoặc bị guard từ chối.
        deadline_s: hạn giờ của một mẫu, tính bằng giây từ lúc bắt đầu mẫu (gồm cả ASR).
        max_result_chars: số ký tự tối đa của kết quả tool trước khi đưa lại vào prompt.
        system_prompt: đường dẫn file system prompt.
        guards: tên các guard được bật, chạy theo thứ tự trong list.
    """

    max_steps: int
    max_retries: int
    deadline_s: float  # giây, tính từ lúc bắt đầu mẫu (gồm cả ASR); tới mốc này bước LLM đang chạy bị ngắt
    max_result_chars: int
    system_prompt: str
    guards: list[str]


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """Mục `[asr]`: backend nhận dạng giọng nói (ASR, speech to text).

    Các trường có giá trị mặc định là tuỳ chọn trong file TOML.

    Thuộc tính:
        backend: `"gold"` dùng transcript có sẵn, `"asr"` chạy model ASR thật.
        model: tên model ASR; bắt buộc khi `backend = "asr"`.
        revision: commit cố định của model ASR.
        engine: cách nạp `model`, một tên trong `ASR_ENGINE_NAMES`.
        device: `"auto"`, `"cpu"` hoặc `"cuda"`.
        dtype: một giá trị trong `ASR_DTYPES`.
        batch_size: số file audio xử lý trong một batch, tối thiểu 1.
        generate_kwargs: tham số giải mã truyền thẳng cho engine (beam, VAD...).
        sample_rate: tần số lấy mẫu audio, đơn vị Hz.
        sanitize: `True` thì lọc câu ảo giác bằng `lifemate.backends.asr_sanitizer`.
        prompt_guard_ngram: lớp bảo vệ câu mồi (`lifemate.backends.asr_guard`); 0 là tắt, từ 2 trở lên là độ dài cụm
            từ dùng để phát hiện chép lại câu mồi. Cần `generate_kwargs.initial_prompt`.
        prompt_guard_max_compression: ngưỡng tỉ lệ nén của lớp bảo vệ; lớn hơn là câu lặp.
        initial_prompt_file: file câu mồi (đường dẫn tính từ gốc repo, ví dụ `configs/asr_prompts/en_top10_v1.txt`);
            `load_config` đọc file và đưa nội dung vào `generate_kwargs.initial_prompt`. Rỗng là không dùng.
    """

    backend: str
    model: str
    revision: str
    engine: str = "hf_seq2seq"  # cách nạp `model` khi backend = "asr"; xem `lifemate.backends.asr.ASR_ENGINES`
    device: str = "auto"
    dtype: str = "auto"  # auto: bfloat16 trên GPU sm80 trở lên, float16 trên GPU cũ hơn, float32 trên CPU
    batch_size: int = 8
    generate_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    sample_rate: int = 16000
    sanitize: bool = False  # bỏ câu ảo giác kiểu phụ đề và đoạn lặp (`lifemate.backends.asr_sanitizer`)
    prompt_guard_ngram: int = 0  # câu hỏng do câu mồi thì chép lại không câu mồi (`lifemate.backends.asr_guard`)
    prompt_guard_max_compression: float = 2.4  # tỉ lệ nén zlib; như ngưỡng câu lặp của faster-whisper
    initial_prompt_file: str = ""  # câu mồi có phiên bản trong configs/asr_prompts/, giống agent.system_prompt


ASR_ENGINE_NAMES = frozenset({"faster_whisper", "hf_seq2seq", "chunkformer", "nemo_ctc"})
ASR_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
# Phải khớp với các trường có giá trị mặc định của `ASRConfig`; thiếu tên nào thì `_build` sẽ bắt buộc khoá đó
# trong TOML.
_ASR_OPTIONAL = frozenset(
    {
        "engine",
        "device",
        "dtype",
        "batch_size",
        "generate_kwargs",
        "sample_rate",
        "sanitize",
        "prompt_guard_ngram",
        "prompt_guard_max_compression",
        "initial_prompt_file",
    }
)


@dataclass(frozen=True, slots=True)
class Config:
    """Toàn bộ config của một lần chạy, gồm ba mục `llm`, `agent` và `asr`.

    Thuộc tính:
        llm: mục `[llm]`.
        agent: mục `[agent]`.
        asr: mục `[asr]`.
    """

    llm: LLMConfig
    agent: AgentConfig
    asr: ASRConfig


@dataclass(frozen=True, slots=True)
class EnvProfile:
    """Vị trí dữ liệu, thư mục run và cache trên một máy cụ thể.

    Thuộc tính:
        data_root: thư mục chứa dữ liệu cuộc thi.
        runs_dir: thư mục ghi kết quả các lần chạy.
        hf_token_source: nguồn đọc token HF, dạng `<loại>:<tên>`.
        persist: cách giữ kết quả sau khi máy tắt (`local`, `drive`, `kaggle_dataset`).
        hf_cache: thư mục cache model của Hugging Face; `None` thì dùng mặc định.
    """

    data_root: str
    runs_dir: str
    hf_token_source: str
    persist: str
    hf_cache: str | None = None

    def with_overrides(self, root: str | None = None, out: str | None = None) -> EnvProfile:
        """Tạo profile mới với tham số dòng lệnh `--root`, `--out` ghi đè lên giá trị trong profile.

        Tham số:
            root: thư mục dữ liệu thay cho `data_root`; `None` thì giữ nguyên.
            out: thư mục run thay cho `runs_dir`; `None` thì giữ nguyên.

        Trả về:
            `EnvProfile` mới; profile gốc không đổi.
        """
        return dataclasses.replace(
            self,
            data_root=root if root is not None else self.data_root,
            runs_dir=out if out is not None else self.runs_dir,
        )

    def resolve_hf_token(self) -> str | None:
        """Đọc token HF từ nguồn đã khai báo, chỉ lúc chạy và không ghi token ra log.

        `hf_token_source` có dạng `<loại>:<tên>`, với loại là `env`, `kaggle_secret` hoặc `colab_userdata`.

        Trả về:
            Token dạng chuỗi, hoặc `None` khi loại nguồn không được hỗ trợ, biến môi trường rỗng, hoặc máy không có
            thư viện của nền tảng tương ứng (Kaggle, Colab).
        """
        kind, _, name = self.hf_token_source.partition(":")
        if kind == "env":
            # Biến môi trường rỗng được coi như không có token.
            return os.environ.get(name) or None
        if kind == "kaggle_secret":
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]  # Kaggle only
            except ImportError:
                return None
            return str(UserSecretsClient().get_secret(name))
        if kind == "colab_userdata":
            try:
                from google.colab import userdata  # type: ignore[import-not-found]  # Colab only
            except ImportError:
                return None
            return str(userdata.get(name))
        return None


_T = TypeVar("_T")


def _build(cls: type[_T], section: str, raw: Any, optional: frozenset[str] = frozenset()) -> _T:
    """Dựng dataclass `cls` từ bảng TOML `raw` của mục `section`, sau khi kiểm tra khoá.

    Tham số:
        cls: lớp dataclass cần dựng.
        section: tên mục, chỉ dùng trong thông báo lỗi.
        raw: giá trị đọc từ TOML cho mục đó.
        optional: các trường được phép vắng mặt và lấy giá trị mặc định.

    Trả về:
        Đối tượng `cls` đã dựng.

    Lỗi:
        ConfigError: khi `raw` không phải bảng, có khoá không thuộc dataclass, hoặc thiếu khoá bắt buộc (mọi trường
            không nằm trong `optional`).
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key in raw:
        if key not in names:
            raise ConfigError(f"unknown key {section}.{key}")
    for name in names - optional:
        if name not in raw:
            raise ConfigError(f"missing key {section}.{name}")
    return cls(**raw)


def _check(cond: bool, msg: str) -> None:
    """Báo `ConfigError` với thông báo `msg` khi `cond` sai."""
    if not cond:
        raise ConfigError(msg)


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Ghi đè khoá trong config thô theo dạng `mục.khoá=giá_trị`, ví dụ `asr.dtype="float32"` hoặc `asr.batch_size=1`.

    Giá trị được đọc như một giá trị TOML (chuỗi cần dấu nháy); không đọc được thì coi là chuỗi trần, nên
    `asr.dtype=float32` cũng được. Khoá phải đã có trong mục, để gõ sai tên khoá thì báo lỗi thay vì lặng lẽ thêm khoá.

    Tham số:
        raw: config thô đọc từ TOML; bị sửa tại chỗ.
        overrides: các chuỗi `mục.khoá=giá_trị`.

    Trả về:
        Chính `raw` sau khi ghi đè.

    Lỗi:
        ConfigError: khi chuỗi sai dạng, hoặc mục hay khoá không có trong config.
    """
    for item in overrides:
        dotted, sep, value = item.partition("=")
        section, dot, key = dotted.strip().partition(".")
        _check(bool(sep and dot and key), f"override {item!r}: expected section.key=value")
        _check(section in raw and key in raw[section], f"override {item!r}: unknown key {section}.{key}")
        try:
            raw[section][key] = tomllib.loads(f"v = {value.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            raw[section][key] = value.strip()
    return raw


def load_config(path: Path, overrides: Sequence[str] = ()) -> Config:
    """Đọc file TOML config chạy và kiểm tra khoá, kiểu, giá trị.

    Tham số:
        path: đường dẫn file TOML có đúng ba mục `[llm]`, `[agent]`, `[asr]`.
        overrides: các ghi đè `mục.khoá=giá_trị` áp trước khi kiểm tra (xem `apply_overrides`).

    Trả về:
        `Config` đã kiểm tra.

    Lỗi:
        ConfigError: khi thiếu hoặc thừa mục, thiếu hoặc thừa khoá, hoặc giá trị nằm ngoài tập cho phép (backend,
            dtype, engine, device, batch_size...).
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    apply_overrides(raw, overrides)
    for key in raw:
        _check(key in {"llm", "agent", "asr"}, f"unknown key {key}")
    for key in ("llm", "agent", "asr"):
        _check(key in raw, f"missing key {key}")
    llm = _build(LLMConfig, "llm", raw["llm"])
    agent = _build(AgentConfig, "agent", raw["agent"])
    asr = _build(ASRConfig, "asr", raw["asr"], optional=_ASR_OPTIONAL)
    # Kiểm tra giá trị. Backend `fake` và `gold` dùng cho test nên không cần tên model.
    _check(llm.backend in {"vllm", "fake"}, f"llm.backend: {llm.backend!r} not in vllm|fake")
    _check(not (llm.backend == "vllm" and llm.model == ""), "llm.model: required when llm.backend = 'vllm'")
    _check(llm.dtype in {"auto", "float16", "bfloat16"}, f"llm.dtype: {llm.dtype!r} not in auto|float16|bfloat16")
    _check(asr.backend in {"gold", "asr"}, f"asr.backend: {asr.backend!r} not in gold|asr")
    _check(asr.engine in ASR_ENGINE_NAMES, f"asr.engine: {asr.engine!r} not in {'|'.join(sorted(ASR_ENGINE_NAMES))}")
    _check(asr.dtype in ASR_DTYPES, f"asr.dtype: {asr.dtype!r} not in {'|'.join(sorted(ASR_DTYPES))}")
    _check(not (asr.backend == "asr" and asr.model == ""), "asr.model: required when asr.backend = 'asr'")
    _check(asr.device in {"auto", "cpu", "cuda"}, f"asr.device: {asr.device!r} not in auto|cpu|cuda")
    # Dataclass không ép kiểu theo annotation, nên kiểu của các trường số và bảng phải kiểm tra tay.
    _check(isinstance(asr.batch_size, int) and asr.batch_size >= 1, "asr.batch_size: must be an int >= 1")
    _check(isinstance(asr.generate_kwargs, dict), "asr.generate_kwargs: must be a table")
    if asr.initial_prompt_file:
        _check(
            "initial_prompt" not in asr.generate_kwargs,
            "asr.initial_prompt_file: set either it or asr.generate_kwargs.initial_prompt, not both",
        )
        prompt_path = repo_path(asr.initial_prompt_file)
        _check(prompt_path.is_file(), f"asr.initial_prompt_file: {asr.initial_prompt_file} not found")
        text = prompt_path.read_text(encoding="utf-8").strip()
        _check(bool(text), f"asr.initial_prompt_file: {asr.initial_prompt_file} is empty")
        asr = dataclasses.replace(asr, generate_kwargs={**asr.generate_kwargs, "initial_prompt": text})
    _check(isinstance(asr.sample_rate, int) and asr.sample_rate > 0, "asr.sample_rate: must be a positive int")
    ngram = asr.prompt_guard_ngram
    _check(
        isinstance(ngram, int) and (ngram == 0 or ngram >= 2), "asr.prompt_guard_ngram: must be 0 (off) or an int >= 2"
    )
    _check(
        isinstance(asr.prompt_guard_max_compression, int | float) and asr.prompt_guard_max_compression > 0,
        "asr.prompt_guard_max_compression: must be a positive number",
    )
    _check(
        not ngram or bool(asr.generate_kwargs.get("initial_prompt")),
        "asr.prompt_guard_ngram: needs asr.generate_kwargs.initial_prompt",
    )
    _check(
        isinstance(agent.guards, list) and all(isinstance(g, str) for g in agent.guards),
        "agent.guards: must be a list of str",
    )
    return Config(llm=llm, agent=agent, asr=asr)


def load_profile(name_or_path: str, repo_root: Path) -> EnvProfile:
    """Đọc profile môi trường theo tên (`configs/env/<tên>.toml`) hoặc theo đường dẫn.

    Tham số:
        name_or_path: tên profile, hoặc đường dẫn tới file có đuôi `.toml`.
        repo_root: thư mục gốc repo, dùng để tìm `configs/env/` khi chỉ có tên.

    Trả về:
        `EnvProfile` đã kiểm tra.

    Lỗi:
        ConfigError: khi có khoá lạ hoặc thiếu khoá bắt buộc (`hf_cache` là tuỳ chọn).
    """
    path = Path(name_or_path)
    # Không có đuôi `.toml` thì coi là tên profile trong `configs/env/`.
    if path.suffix != ".toml":
        path = repo_root / "configs" / "env" / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return _build(EnvProfile, "env", raw, optional=frozenset({"hf_cache"}))
