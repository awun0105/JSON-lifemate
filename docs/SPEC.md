# SPEC: hệ thống agent LifeMate

Tài liệu này mô tả **cần xây cái gì** để người viết code (người hoặc AI) làm được mà không phải hỏi lại. Đề bài xem `overview.md`, dữ liệu xem `eda.md`, lý do của từng quyết định xem `approach.md`.

- **Phạm vi:** toàn hệ thống. Riêng **agent v0 và hạ tầng đo** (mục 4) viết chi tiết đến mức interface, file và test. ASR, fine-tune và Docker (mục 6) viết ở mức interface và tiêu chí nghiệm thu, sẽ viết chi tiết khi có số đo của v0.
- **Kiểm chứng trạng thái:** 27/10/2026, trên gói dữ liệu `train+public_test/` và `zaic_tools` của train.

---

## 1. Bối cảnh

### 1.1. Mục tiêu

Cho một yêu cầu bằng giọng nói tiếng Việt, agent phải gọi đúng chuỗi tool trên các app giả lập (`zaic_tools`). Kết quả là một dòng `submission.jsonl` gồm `id`, `tool_calls` và `state_hash`. Thứ hạng chỉ tính theo `private_test`, bộ này nhiều khả năng có app chưa từng thấy.

### 1.2. Thế nào là xong

| Mốc | Xong khi |
|---|---|
| v0 (30/10) | Chạy hết `val_seen` và `val_unseen`, có điểm từ `eval/score.py` và các tiêu chí ở mục 7 đều đạt |
| Docker (05/11) | Image chạy với `--network none`, ra được `result/submission.jsonl` cho `public_test` |
| Nộp (16/11) | Gói Docker cuối, training code và data chạy lại được |

### 1.3. Trạng thái hiện tại (đã kiểm chứng)

| Thành phần | Trạng thái | Bằng chứng |
|---|---|---|
| Đề bài, EDA, hướng làm | Có | `docs/overview.md`, `docs/eda.md`, `docs/approach.md` |
| Chia dữ liệu | Có | `eval/make_splits.py`, `splits/`: train 1098, val_seen 193, val_unseen 254 |
| Bộ chấm local | Có | `eval/score.py`: gold được 100, toàn `[]` được 7,18 |
| Agent v0 (`lifemate/`), CLI, hạ tầng đo | **Có, 27/10/2026** | 164 test pass; FakeLLM qua CLI được 100,00 (strict và canon) trên `val_seen`, `val_unseen`; `configs/fake.toml` chạy không cần model |
| ASR | Khung có (`asr.py`, `eval_asr.py`), **chưa chọn model, chưa chạy thật** | |
| Docker | Dockerfile, entrypoint, smoke script có; **chưa build** (không có GPU/docker ở máy dev) | `docs/docker.md` |
| Fine-tune | Khung có (`lifemate/train/`), `sft_data` và `asr_data` đã chạy trên dữ liệu thật; train/merge **chưa chạy** | |

### 1.4. Số liệu dữ liệu dùng để thiết kế

| Đại lượng | Giá trị | Ảnh hưởng |
|---|---|---|
| Số call trong gold | 0 – 6, phân bố `0:111, 1:369, 2:428, 3:359, 4:161, 5:107, 6:10` | Giới hạn số bước của vòng lặp |
| Số tool được cài mỗi mẫu | train 8 – 54, public tối đa 59 | Ngân sách token cho phần mô tả tool |
| Số lượt user trong `history` | 0 – 12, 527/1545 mẫu không có history | ASR phải xử lý tới 13 file audio mỗi mẫu |
| Độ dài một kết quả tool (JSON) | p50 165, p95 867, max 2439 ký tự | Ngưỡng cắt bớt kết quả |
| Catalog train | 46 app, 147 tool, 62.853 ký tự | Không thể đưa nguyên catalog vào prompt |
| Số mẫu public_test | 634 | Kiểm tra số dòng khi nộp |

### 1.5. Ràng buộc từ luật thi

| Ràng buộc | Nguồn |
|---|---|
| Tổng tham số < 6B, LLM < 5B, context ≤ 16K | `overview.md` mục 8 |
| Chạy offline, không gọi API | `overview.md` mục 8 |
| Mỗi mẫu ≤ 30 giây trên 1 × A30 | `overview.md` mục 8 |
| Giải mã tất định, cố định seed | `overview.md:318` |
| Đúng một `ToolEnv` mỗi mẫu, không tạo lại để thử | README gói dữ liệu, mục 3 |
| **Mọi call đã gọi đều vào bài nộp, kể cả call bị từ chối** | `zaic_tools/executor.py`, `Session.call` ghi vào `self.calls` trước khi kiểm tra schema |
| Không đọc `state.json` | README gói dữ liệu, mục 2.3 |

Dòng in đậm là ràng buộc quan trọng nhất cho thiết kế. Một call sai schema vẫn bị tính vào `U` (call thừa) và làm giảm điểm. Vì vậy agent **phải gọi `env.validate` trước, chỉ gọi `env.call` khi hợp lệ**. `env.validate` không ghi vào log.

---

## 2. Kiến trúc tổng thể

```
                 ┌──────────────────────── một mẫu ───────────────────────────┐
 <id>.wav, _uN ─►│ [A2] Transcriber ──► chữ (lượt cuối + các lượt user cũ)     │
                 │ [A3] Normalizer (chỉ khi cần, xem 6.1)                      │
 context ───────►│ [A4] PromptBuilder ◄── ToolDocs (catalog ∩ installed_apps)  │
 history ───────►│          │                                                  │
                 │          ▼                                                  │
                 │ [A6] Agent loop ── LLMBackend [A5] (vLLM, JSON ép schema)   │
                 │          │    └─ Guards [A7] ── env.validate ── env.call    │
                 │          ▼                         (zaic_tools [A8])        │
                 │ SampleResult: tool_calls, state_hash, trace                 │
                 └──────────┬──────────────────────────────────────────────────┘
                            ▼
      [A9] submission.jsonl    [A10] trace.jsonl, summary.json   ──► eval/score.py
```

| Khối | Thành phần | Mốc |
|---|---|---|
| A. Chạy khi chấm | A1 data, A2 transcriber, A3 normalizer, A4 prompt, A5 LLM, A6 loop, A7 guards, A8 `zaic_tools` (BTC), A9 writer, A10 tracer | v0: A1, A2 (gold), A4–A10. Sau v0: A2 (ASR). A3 chỉ khi cần, xem 6.1 |
| B. Đánh giá | B1 splits (có), B3 scorer (có), B4 run manager, B5 đo ASR, B6 test | v0: B4, B6. Sau v0: B5 |
| C. Huấn luyện | C1 dữ liệu SFT, C2 QLoRA LLM, C3 fine-tune ASR, C4 versioning | Sau v0 |
| D. Nộp bài | D1 Dockerfile, D2 kiểm tra trước khi nộp, D3 gói nộp | Sau v0 |

---

## 3. Quyết định đã chốt cho SPEC này

| # | Quyết định | Lý do |
|---|---|---|
| S1 | Python 3.11, quản lý bằng `uv` (`pyproject.toml` + `uv.lock`), test bằng pytest | Khóa phiên bản giống nhau giữa Kaggle và Docker |
| S2 | LLM chạy bằng vLLM **trong cùng process** (`vllm.LLM`), gọi qua interface `LLMBackend` | Một process, dễ đóng gói. Có `FakeLLM` để test không cần GPU |
| S3 | Tool call theo **định dạng JSON tự định nghĩa**, vLLM ép đầu ra theo JSON schema (structured outputs). Mô tả tool dạng rút gọn trong prompt | Tiết kiệm token, không bao giờ hỏng JSON, không phụ thuộc chat template của từng model |
| S4 | Mỗi lượt LLM sinh **đúng một** hành động: gọi một tool hoặc báo xong | Đơn giản, dễ ép schema, khớp với dữ liệu fine-tune |
| S5 | Chạy tuần tự từng mẫu, batch = 1 | Tất định. Luật chỉ giới hạn thời gian mỗi mẫu, không giới hạn tổng |
| S6 | Tắt chế độ suy nghĩ của model nếu model có chế độ này (ví dụ `enable_thinking=False` qua `chat_template_kwargs`) | Tốn token và thời gian, khó giữ dưới 30 giây |
| S7 | **Chưa chọn LLM và ASR.** Code không gắn với model cụ thể nào: tên model, revision, dtype chỉ nằm trong config. Khi chọn, model phải đạt điều kiện ở mục 6.0 | Để chọn model sau mà không phải sửa code |
| S8 | Config bằng TOML (`tomllib` có sẵn trong Python 3.11) | Không thêm thư viện |
| S9 | Mọi thành phần thay được đều là `typing.Protocol`, lắp ghép tại **một chỗ duy nhất** (`lifemate/factory.py`), dùng composition thay cho kế thừa | Đổi model, ASR, guard bằng config. Test thay bằng bản giả mà không sửa code chính (mục 4.0) |
| S10 | Prompt sắp từ phần ít đổi đến phần hay đổi. Messages chỉ được thêm vào cuối, không bao giờ sửa | Tận dụng prefix cache của vLLM giữa các mẫu và giữa các bước trong một mẫu (mục 4.6) |
| S11 | Nạp model xong phải **warm-up** trước khi bắt đầu tính giờ mẫu đầu tiên | Lần gọi đầu của vLLM và ASR chậm vì biên dịch kernel, có thể làm mẫu đầu vượt 30 giây |
| S12 | Hạn giờ kiểm tra **trước** mỗi bước bằng ước lượng thời gian của bước đó, không chỉ kiểm tra thời gian đã trôi qua | Tránh bắt đầu một bước lúc 24,9 giây rồi kết thúc lúc 31 giây (mục 4.10) |
| S13 | Dùng lại từ kit `~/PROJECT/AI20K/ai-kit` (nhánh `ai-kit`) **quy ước kỹ thuật**, không dùng lại các layer ứng dụng | Kit là app FastAPI + LangGraph gọi LLM cloud; RAG, MCP, A2A, reliability, cost guard không áp dụng cho agent offline chạy vLLM trong process. Lấy: cấu hình ruff và typecheck, pre-commit (hygiene, detect-secrets, ruff), cấu trúc Makefile, mẫu test với LLM giả có kịch bản (`tests/fakes.py` → `FakeLLM`), Dockerfile hai tầng với `uv --frozen` và user không phải root, quy ước log `structlog` kiểu `sự_kiện_chữ_thường` với kwargs, và dòng ghi nguồn gốc khi chép code. Mỗi chỗ chép ghi nguồn (`# Adapted from ai-kit <path>@<commit>`) |

---

## 4. Chi tiết agent v0

### 4.1. Cấu trúc repo

```
pyproject.toml, uv.lock
lifemate/
├── __init__.py
├── types.py       # value object, Enum StopReason/ErrorCode/GuardVerdict, các Protocol (mục 4.0)
├── jsonutil.py    # canonical_json(): mọi JSON ảnh hưởng tới tính tất định đi qua đây
├── config.py      # đọc TOML thành dataclass Config, kiểm tra giá trị
├── data.py        # A1: đọc <split>.jsonl thành Sample
├── zaic.py        # nạp zaic_tools đúng split, tạo ToolEnv (một lần mỗi mẫu)
├── asr.py         # A2: GoldTranscriber (sau v0: ASRTranscriber cho model ASR sẽ chọn)
├── tooldocs.py    # catalog → mô tả tool rút gọn, có cache
├── prompt.py      # A4: PromptBuilder, Conversation (giữ messages, đếm token tăng dần)
├── protocol.py    # JSON schema đầu ra + parse
├── tokens.py      # TokenCounter dùng tokenizer của model (chạy trên CPU)
├── llm.py         # A5: VLLMBackend, FakeLLM
├── guards.py      # A7: các Guard và GuardChain
├── budget.py      # Budget: hạn giờ dự đoán, số bước, số lần thử lại
├── agent.py       # A6: Agent.run(sample, env) -> SampleResult
├── observers.py   # A10: AgentObserver, Tracer, NullObserver
├── timing.py      # decorator Timed cho LLMBackend và Transcriber
├── store.py       # RunStore: đọc ghi thư mục run (flush, ghi nguyên tử, resume)
├── runner.py      # Runner: warm-up, lặp qua mẫu, cô lập lỗi từng mẫu, gọi chấm
├── factory.py     # nơi duy nhất tạo đối tượng từ Config (registry theo tên backend)
├── envinfo.py     # nhận GPU, chọn dtype, lấy phiên bản code và gói (mục 4.16)
└── cli.py         # chỉ parse tham số rồi gọi Runner; chạy bằng `python -m lifemate.cli`
prompts/system_v0.md
configs/v0.toml
configs/docker.toml   # như v0.toml nhưng model trỏ vào /app/models/* và asr.backend = "asr"; mặc định của image (mục 6.3)
configs/env/       # local.toml, kaggle.toml, colab.toml, rented.toml, docker.toml (mục 4.16)
requirements/notebook.txt   # sinh từ uv.lock, bỏ torch/CUDA, dùng cho Kaggle và Colab
scripts/pack_code.sh
notebooks/         # kaggle_run.ipynb, colab_run.ipynb: chỉ gọi lệnh
lifemate/train/    # sau v0: sft_data, sft, llm_pipeline, asr_data, asr, asr_pipeline, common (mục 6.2, 6.4, 6.5)
configs/train/     # sau v0: llm_vN.toml, asr_vN.toml
data/sft/, data/asr/   # sau v0: dữ liệu train đã dựng + file meta (nộp kèm theo yêu cầu BTC)
models/registry.csv     # sau v0: danh sách model đã train; trọng số không vào git
tests/
eval/              # đã có: make_splits.py, score.py
runs/              # không đưa vào git
```

`vllm` chỉ được import bên trong `VLLMBackend`. Toàn bộ test chạy được trên máy không có GPU và không cài vLLM.

### 4.0. Kiến trúc code

**Mỗi module một việc.** Bảng dưới là ranh giới trách nhiệm. Một module không được làm việc của module khác.

| Module | Chỉ làm | Không làm |
|---|---|---|
| `Agent` | Vòng lặp: hỏi LLM, qua guard, gọi env, phát sự kiện | Đọc file, ghi trace, đo giờ, tạo đối tượng |
| `Conversation` | Giữ messages (chỉ thêm vào cuối), đếm token tăng dần | Quyết định dừng |
| `Budget` | Trả lời "còn được chạy bước tiếp không", trả về `StopReason` nếu không | Gọi LLM |
| `GuardChain` | Chạy lần lượt các `Guard`, trả về kết quả của guard đầu tiên không cho qua | Gọi `env.call` |
| `Tracer` | Nhận sự kiện, ghi `trace.jsonl` | Ảnh hưởng tới hành vi của agent |
| `RunStore` | Đọc ghi thư mục run, resume, ghi nguyên tử | Chạy agent |
| `Runner` | Warm-up, lặp qua mẫu, bắt lỗi ở ranh giới từng mẫu, gọi `score.py` | Logic của agent |
| `factory` | Tạo mọi đối tượng từ `Config` | Logic nghiệp vụ |

**Các interface (`lifemate/types.py`):**

```python
class ToolEnvLike(Protocol):
    def validate(self, name: str, arguments: dict) -> dict: ...
    def call(self, name: str, arguments: dict) -> dict: ...
    def get_tool_calls(self) -> list[dict]: ...
    def get_state_hash(self) -> str: ...


class LLMBackend(Protocol):
    def generate(self, messages: list[dict], schema: dict, *, sample_id: str) -> Generation: ...
    # sample_id chỉ để FakeLLM tra đáp án (A1); VLLMBackend bỏ qua, không đưa vào prompt
    def warmup(self) -> None: ...


class TokenCounter(Protocol):
    def count(self, messages: list[dict]) -> int: ...


class Transcriber(Protocol):
    def transcribe(self, sample: Sample) -> Transcript: ...
    def warmup(self) -> None: ...


class Clock(Protocol):
    def __call__(self) -> float: ...  # time.monotonic lúc chạy thật, đồng hồ giả trong test


class Guard(Protocol):
    def check(self, action: Action, ctx: GuardContext) -> GuardResult: ...


class AgentObserver(Protocol):
    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None: ...
    def on_step(self, step: StepRecord) -> None: ...
    def on_stop(self, result: SampleResult) -> None: ...
```

- `Agent` chỉ phụ thuộc vào các Protocol trên. Unit test dùng `FakeEnv`, `FakeLLM`, `GoldTranscriber`. Chỉ T1 cần `zaic_tools` và dữ liệu thật.
- `TokenCounter` tách khỏi `LLMBackend` (interface nhỏ, đúng việc). `FakeLLM` và `VLLMBackend` đếm token giống hệt nhau vì cùng dùng tokenizer của model, nên test về tràn context khớp với lúc chạy thật.
- Value object (`Action`, `Generation`, `Transcript`, `StepRecord`, `SampleResult`) dùng `@dataclass(frozen=True, slots=True)`.
- Lý do dừng, mã lỗi và kết quả guard là `Enum` (`StopReason`, `ErrorCode`, `GuardVerdict`), không dùng chuỗi tự do. Khi ghi ra JSON thì dùng `.value`.

**Design pattern và lý do dùng:**

| Pattern | Áp dụng vào | Được gì |
|---|---|---|
| Strategy | `LLMBackend`, `Transcriber`, `TokenCounter`, cách render tool docs | Đổi bằng config để so sánh thí nghiệm |
| Chain of Responsibility | `GuardChain` gồm danh sách `Guard` | Thêm hoặc bớt guard không phải sửa `Agent` |
| Observer | `AgentObserver`: `NullObserver`, `CompositeObserver`, `Tracer` (tùy chọn, ghi luồng phụ) | Agent không biết định dạng trace. `trace.jsonl` chính do `RunStore.append` ghi từ `Runner`, vì mẫu bị exception không đi qua `on_stop` |
| Decorator | `Timed(backend)`, `Timed(transcriber)`; `lru_cache` cho tool docs | Đo giờ và cache không phải chép vào từng lớp |
| Factory + registry | `factory.build(config)`, registry `{"vllm": VLLMBackend, "fake": FakeLLM, "gold": GoldTranscriber}` | Một chỗ duy nhất biết cách lắp hệ thống |

Không dùng kế thừa nhiều tầng, không dùng singleton hay biến toàn cục. Đối tượng được truyền vào qua constructor.

### 4.0.1. Quy ước code

- `ruff check` và `ruff format` (độ dài dòng 120), `mypy --strict` cho `lifemate/`. Cả hai chạy trong pre-commit và phải sạch.
- Không có hằng số rải rác. Mọi ngưỡng (giây, ký tự, số bước, số lần thử lại) nằm trong `Config`.
- Chỉ **một** chỗ được bắt mọi exception: ranh giới từng mẫu trong `Runner`, có ghi traceback vào trace. Các chỗ khác bắt đúng loại lỗi cần bắt.
- Mọi JSON đưa vào prompt, schema hay file kết quả đi qua `canonical_json()` (`ensure_ascii=False`, `separators=(",", ":")`, `sort_keys=True`).
- Hàm công khai có type hint và docstring một dòng. Đọc ghi file luôn dùng `with` hoặc `Path.read_text`.

### 4.2. Kiểu dữ liệu (`lifemate/data.py`)

```python
@dataclass(frozen=True, slots=True)
class Turn:
    role: Literal["user", "assistant"]
    audio: str | None                 # user: đường dẫn tương đối trong split
    audio_text: str | None            # user: chỉ train có
    text: str | None                  # assistant
    tool_calls: tuple[dict, ...]      # assistant, có thể rỗng
    tool_results: tuple[dict, ...]    # assistant, có thể rỗng

@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    split: str
    audio: str                        # lượt cuối
    audio_text: str | None            # chỉ train
    context: dict                     # nguyên văn từ jsonl
    history: tuple[Turn, ...]
    label: dict | None                # chỉ train; agent KHÔNG được đọc

def load_split(root: Path, split: str, ids: list[str] | None = None) -> list[Sample]
```

`label` chỉ được đọc bởi `FakeLLM` và bởi bộ chấm. Có test kiểm tra `agent.py` và `prompt.py` không truy cập `label` (mục 5).

### 4.3. Nạp `zaic_tools` (`lifemate/zaic.py`)

Mỗi split có bản `zaic_tools` riêng và cùng tên module. Vì vậy **mỗi process chỉ chạy một split**.

```python
def load_zaic(root: Path, split: str) -> ModuleType
    # chèn <root>/<split> vào đầu sys.path, import zaic_tools,
    # kiểm tra zaic_tools.__file__ nằm trong <root>/<split>/ sau khi Path.resolve() cả hai phía
    # (Docker dùng symlink, mục 6.3), sai thì raise
def open_env(zaic, root: Path, split: str, sample_id: str) -> Session
    # gọi zaic.ToolEnv(split, sample_id, root=root) đúng một lần cho mỗi sample_id
    # trong process; gọi lần hai với cùng id thì raise RuntimeError
```

### 4.4. Transcriber (`lifemate/asr.py`)

Interface `Transcriber` ở mục 4.0.

```python
@dataclass(frozen=True, slots=True)
class Transcript:
    final: str  # lời của lượt cuối
    history: tuple[str | None, ...]  # cùng độ dài với sample.history; None ở các lượt assistant
    seconds: float
```

- **v0: `GoldTranscriber`** trả về `audio_text` của lượt cuối và của các lượt user trong history. Chỉ dùng được với train. Gặp mẫu không có `audio_text` thì raise. `warmup()` không làm gì.
- Sau v0: `ASRTranscriber` (mục 6.1) dùng cùng interface.

### 4.5. Mô tả tool rút gọn (`lifemate/tooldocs.py`)

```python
class CatalogIndex:            # nạp catalog một lần, index {tên tool: spec}, allowed_tools(installed_apps)
def render_tools(index: CatalogIndex, allowed: Iterable[str], *, drop_param_descriptions: bool = False) -> str
```

- Chỉ đưa vào các tool có trong `installed_apps`, cộng thêm `sys.ask_missing_info`.
- Thứ tự app và tool giữ nguyên như trong catalog, để prompt tất định.
- Kết quả được cache bằng `functools.lru_cache`, key là `frozenset` tên các tool được phép. Catalog được nạp một lần cho mỗi lần chạy và đánh index sẵn `{tên tool: spec}`, không duyệt lại catalog cho từng mẫu.
- Định dạng mỗi app, ví dụ:

```
## smart_home (Nhà thông minh): Điều khiển đèn, quạt, điều hoà, rèm, ổ cắm theo phòng/tầng
- smart_home.get_status [read] → devices[]
  Lấy thiết bị và trạng thái; lọc theo phòng/tầng/loại; ...
  room?: string | floor?: integer (Số nguyên, ...) | type?: light|fan|climate|curtain|plug|tv
```

| Thành phần schema | Cách viết |
|---|---|
| Tham số bắt buộc / không bắt buộc | `name` / `name?` |
| Kiểu | `string`, `integer`, `number`, `boolean`, `array<string>`, `object` |
| `enum` | `a|b|c` thay cho kiểu |
| `description` của tham số | trong ngoặc tròn sau kiểu |
| `format`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | `[format=date]`, `[>=1]`, `[>0]` |
| `items` là object | `array<object{a: string, b?: integer}>` |
| Tool không có tham số | dòng `(không có tham số)` |
| `x-one-of` | dòng riêng `cần một trong: [contact_id] hoặc [phone_number]` |
| `x-ref` | `(id lấy từ contacts.contacts)` |
| `x-match: fuzzy` | `(văn bản tự do)` |
| `returns` | `→ ...` trên dòng tên tool, bỏ qua nếu rỗng |

Mọi trường của catalog (`parameters`, `x-*`, `returns`) đều phải xuất hiện. Có test chạy trên toàn bộ catalog train và public để kiểm tra không bỏ sót trường nào (mục 5).

**Quy tắc app bị rút (chế độ `train_mode`):** chỉ áp dụng khi dựng prompt cho mẫu thuộc `splits/train.txt` (few-shot, dữ liệu fine-tune). Khi đó:

1. Xóa 8 app trong `splits/meta.json` → `held_out_apps` khỏi `installed_apps` trước khi gọi `render_tools`.
2. **Bỏ hẳn mẫu** nếu `history` có lượt assistant gọi tool của app bị rút (A4). Đã đếm: 65 mẫu trong `splits/train.txt` như vậy (ví dụ `train_0682`, `train_0732`). Nếu chỉ xóa khỏi tool docs thì model vẫn thấy `[đã gọi] medication.xxx(...)` trong lịch sử. `PromptBuilder.start(..., train_mode=True)` raise `HeldOutLeak`, và `sft_data.py` ghi id vào `data/sft/vN.dropped.txt` với lý do.

Khi chạy đánh giá trên `val_seen` và `val_unseen` thì **không** áp dụng chế độ này.

### 4.6. Dựng prompt (`lifemate/prompt.py`)

```python
class PromptBuilder:
    def __init__(self, system_template: str, renderer: ToolDocRenderer): ...
    def __init__(self, system_template, index: CatalogIndex, counter: TokenCounter, *,
                 max_model_len, max_tokens, max_result_chars, held_out_apps=frozenset()): ...
    def start(self, sample: Sample, transcript: Transcript, *, train_mode: bool = False) -> tuple[Conversation, int]
        # trả về (hội thoại, mức giảm 0-3 đã dùng); train_mode: áp dụng quy tắc app bị rút (mục 4.5),
        # raise HeldOutLeak nếu history gọi app bị rút

class Conversation:
    def __init__(self, messages: list[dict], counter: TokenCounter): ...
    messages: tuple[dict, ...]           # chỉ đọc từ bên ngoài
    def append_assistant(self, text: str, completion_tokens: int) -> None
    def append_user(self, text: str) -> None
    def prompt_tokens(self) -> int       # đếm tăng dần, xem bên dưới
```

**Thứ tự messages: phần ít thay đổi đặt trước, phần hay thay đổi đặt sau (quyết định S10).** vLLM tự tái dùng kết quả tính toán của phần đầu trùng nhau (prefix cache). Vì vậy:

| Vị trí | Nội dung | Giống nhau giữa |
|---|---|---|
| 1. system, phần đầu | Quy tắc và định dạng đầu ra trong `prompts/system_v0.md` | **Mọi mẫu** |
| 2. system, phần cuối | `{tool_docs}` | Các mẫu cài cùng bộ app |
| 3. user, khối đầu | `# Ngữ cảnh`: `current_time`, `timezone`, `location`, `user_profile`, viết thành các dòng `khóa: giá trị`. **Không** viết lại `installed_apps` vì đã có trong tool docs | Chỉ trong một mẫu |
| 4. user, tiếp theo | `# Lịch sử hội thoại` (bỏ qua nếu không có history). Mỗi lượt user là một dòng `Người dùng: <transcript>`. Mỗi lượt assistant là `Trợ lý: <text>`, kèm từng call dạng `  [đã gọi] name(arguments_json) → result_json` | Chỉ trong một mẫu |
| 5. user, cuối | `# Yêu cầu hiện tại`, sau đó là `transcript.final` | Chỉ trong một mẫu |

`system_v0.md` chỉ có một chỗ cần điền là `{tool_docs}`, và chỗ đó phải nằm ở cuối file.

**Messages chỉ được thêm vào cuối, không bao giờ sửa hay cắt message cũ.** Sửa một message cũ làm mất cache của toàn bộ phần phía sau. Kết quả tool được cắt (mục 4.10) **trước** khi thêm vào, không cắt lại sau đó.

**Đếm token tăng dần:** `Conversation` đếm toàn bộ messages bằng `TokenCounter` một lần lúc khởi tạo. Sau đó mỗi lần thêm message chỉ đếm phần mới thêm, cộng số token mẫu hội thoại (chat template) thêm vào. Nhờ vậy không phải tokenize lại cả hội thoại ở mỗi bước. Khi `Generation.prompt_tokens` do vLLM trả về khác với số tự đếm thì lấy số của vLLM, ghi cảnh báo vào trace, và dùng số đó làm mốc cho các bước sau.

**Trong vòng lặp, mỗi bước thêm vào cuối:**

- **assistant:** đúng chuỗi JSON mà LLM đã sinh.
- **user:** `KẾT QUẢ: <result_json>`, hoặc `LỖI (chưa gọi tool): <message>` khi guard chặn (mục 4.9).

**Kết quả tool trong history** cũng được cắt theo `max_result_chars` khi render (cùng hàm `truncate` với vòng lặp). Đã đo: tổng kết quả tool trong history của một mẫu lên tới 4.782 ký tự.

**Thang giảm khi prompt vượt ngân sách token (B1):** `PromptBuilder.start` đếm token ngay sau khi dựng. Nếu `prompt_tokens + max_tokens > max_model_len` thì áp dụng lần lượt các mức sau, dừng ở mức đầu tiên vừa ngân sách, và ghi mức đã dùng vào trace (`prompt_shrink_level`):

| Mức | Làm gì | Mất gì |
|---|---|---|
| 0 | Không giảm | |
| 1 | Cắt kết quả tool trong history còn `max_result_chars / 4` | Chi tiết của các kết quả cũ |
| 2 | Bỏ `description` của từng tham số trong tool docs, giữ tên, kiểu, bắt buộc, enum, `x-*` | Gợi ý cách điền tham số |
| 3 | Bỏ lượt history cũ nhất, lặp cho tới khi vừa; luôn giữ lượt user cuối và ít nhất 2 lượt gần nhất | Ngữ cảnh xa |

Vẫn không vừa sau mức 3 thì `Budget` trả `CONTEXT_OVERFLOW` ở bước 0 như cũ. Mức 2 và 3 đổi phần system/history nên làm giảm trúng cache, nhưng chỉ xảy ra ở số ít mẫu. `summary.json` đếm số mẫu theo từng mức.

`prompts/system_v0.md` phải nêu tối thiểu các quy tắc sau:

- Mỗi lượt chỉ trả về một JSON.
- Tra cứu lấy ID bằng tool loại `read` trước khi gọi tool loại `write`. Không tự bịa ID.
- Thiếu tham số bắt buộc thì gọi `sys.ask_missing_info` với `tool_name` và `params`, rồi báo xong.
- Yêu cầu bị hủy giữa câu, có điều kiện, hoặc không app nào làm được thì báo xong mà không gọi tool. Đây là quy ước của nhãn: câu "nếu mai mưa thì…" có gold `[]` (xem `docs/eda.md`).
- Ngày giờ tính theo `current_time` trong context.
- Số tiền và số đọc bằng chữ đổi thành số.

Tất cả JSON trong prompt đi qua `canonical_json()` (mục 4.0.1) để giữ tất định.

### 4.7. Định dạng đầu ra (`lifemate/protocol.py`)

JSON schema dùng để ép đầu ra, tạo riêng cho từng mẫu:

```json
{
  "anyOf": [
    {"type": "object",
     "properties": {"action": {"const": "call"},
                    "name": {"enum": ["<các tool được phép của mẫu>", "sys.ask_missing_info"]},
                    "arguments": {"type": "object"}},
     "required": ["action", "name", "arguments"], "additionalProperties": false},
    {"type": "object",
     "properties": {"action": {"const": "done"}},
     "required": ["action"], "additionalProperties": false}
  ]
}
```

- `name` bị giới hạn trong danh sách tool được phép, nên lỗi `unknown_tool` không thể xảy ra khi có ép schema.
- `arguments` chỉ ép là object. Kiểm tra từng tham số do guard làm bằng `env.validate`. Ép schema chi tiết theo từng tool là thí nghiệm sau v0.
- Danh sách `name` được sắp xếp, và schema được serialize bằng `canonical_json()`. Nhờ vậy các mẫu cài cùng bộ tool sinh ra **cùng một chuỗi schema**, và grammar đã biên dịch được dùng lại từ cache. `output_schema` cũng được cache bằng `lru_cache` theo `frozenset` tên tool. Ghi thời gian biên dịch grammar của 30 mẫu đầu vào trace để biết có đáng lo không.

```python
def output_schema(allowed_tools: list[str]) -> dict
def parse(text: str) -> Action | ParseError
@dataclass
class Action:
    kind: Literal["call", "done"]
    name: str | None
    arguments: dict | None
```

`parse` vẫn phải xử lý đầu ra không hợp lệ, vì `FakeLLM` và các backend không ép schema có thể sinh ra. Gặp đầu ra hỏng thì trả `ParseError`, không raise.

### 4.8. LLM backend (`lifemate/llm.py`)

Interface `LLMBackend` và `TokenCounter` ở mục 4.0.

```python
@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
```

**`VLLMBackend(config)`**

- Tạo `vllm.LLM` một lần khi khởi tạo, dùng các khóa `model`, `revision`, `dtype` (`float16` trên T4, `bfloat16` trên sm80+), `max_model_len = 16384`, `gpu_memory_utilization`, `seed`, `enable_prefix_caching = true`.
- `generate` dùng `llm.chat(messages, SamplingParams(temperature=0, max_tokens=512, seed=config.seed, <structured outputs theo schema>))`, cộng thêm `chat_template_kwargs={"enable_thinking": false}` nếu model hỗ trợ.
- Tên tham số structured outputs khác nhau giữa các phiên bản vLLM. Dùng đúng API của phiên bản được khóa trong `uv.lock`, và ghi phiên bản đó vào `config.json` của mỗi lần chạy.
- `batch_invariant = true` trong config thì đặt `VLLM_BATCH_INVARIANT=1` trước khi import vllm. Chỉ bật trên GPU sm80 trở lên. Bật trên GPU thấp hơn thì raise ngay khi khởi tạo.
- `warmup()`: gọi `generate` 2 lần với một hội thoại giả có độ dài gần với mẫu thật (khoảng 4K token) và một schema giả. Mục đích là để vLLM biên dịch kernel và CUDA graph trước. Thời gian warm-up ghi vào `config.json`, **không** tính vào mẫu nào.

**`HFTokenCounter(model, revision)`** dùng `transformers.AutoTokenizer` và `apply_chat_template` của đúng model trong config. Chạy được trên CPU, dùng chung cho cả `VLLMBackend` lẫn `FakeLLM`.

**`FakeLLM(samples)`**

Phát lại `label.tool_calls` của từng mẫu theo thứ tự, mỗi lần `generate` trả một call, hết call thì trả `{"action":"done"}`. Mẫu được nhận biết qua tham số `sample_id` của `generate` (A1), vì prompt không chứa id. Có thể nhận thêm `delay_s` (giả lập thời gian mỗi bước, dùng với đồng hồ giả trong T10) và một danh sách đầu ra hỏng để ép theo id (T4, T5). Dùng để test vòng lặp và bộ ghi kết quả từ đầu đến cuối.

### 4.9. Guards (`lifemate/guards.py`)

Thứ tự kiểm tra một `Action` loại `call`. Bước đầu tiên không đạt thì dừng kiểm tra.

| # | Kiểm tra | Không đạt thì |
|---|---|---|
| G1 | `name` nằm trong danh sách tool được phép | Không gọi env, mã `unknown_tool` |
| G2 | `env.validate(name, arguments)` trả `ok: true` | Không gọi env, mã `invalid_args`, đưa `error` của validate lại cho LLM |
| G3 | Không trùng hoàn toàn (cùng `name` và `arguments`) với một call loại `write` đã gọi trong mẫu | **Chỉ cảnh báo**, vẫn gọi env, ghi mã `duplicate_write` |
| G4 | *(không dùng mặc định)* Nếu đã gọi `sys.ask_missing_info` trước đó thì không gọi thêm tool nào | Dừng vòng lặp |

- **G4 không nằm trong danh sách guard mặc định** (A3): vòng lặp đã dừng ngay sau call `sys.ask_missing_info` (`StopReason.ASKED`), nên G4 không bao giờ chạy tới. Lớp `AfterAskGuard` vẫn tồn tại làm ví dụ cho T18 và để dùng nếu sau này đổi quy tắc dừng.
- Bị chặn ở G1–G2 thì không gọi `env.call`, nên call đó **không** vào bài nộp. LLM nhận lại `LỖI (chưa gọi tool): ...` và được thử lại.
- G3 không chặn vì gold có 2 mẫu gọi cùng một call `write` hai lần hợp lệ (`train_0922`, `train_1447`: `loyalty.redeem`). Đã kiểm bằng cách chạy toàn bộ gold train qua G2–G4: 0 call bị G2 chặn, 0 call sau `ask_missing_info`, 2 call trùng. Mã `duplicate_write` dùng để xem trong trace có phải agent bị lặp hay không.
- Mỗi mẫu được thử lại tối đa `max_retries = 2` lần, **tính chung** cho cả lỗi parse lẫn call bị chặn. Quá số đó thì dừng vòng lặp với những call đã gọi.
- Loại tool (`read`, `write`, `meta`) lấy từ trường `kind` trong catalog.

**Cấu trúc code (Chain of Responsibility):**

```python
class GuardVerdict(Enum): PASS, BLOCK, WARN, STOP

@dataclass(frozen=True, slots=True)
class GuardResult:
    verdict: GuardVerdict
    code: ErrorCode | None
    message: str | None          # đưa lại cho LLM khi BLOCK

@dataclass(frozen=True, slots=True)
class GuardContext:
    env: ToolEnvLike
    allowed: frozenset[str]
    tool_kind: Mapping[str, str]
    calls_so_far: tuple[Action, ...]

class GuardChain:
    def __init__(self, guards: Sequence[Guard]): ...
    def check(self, action: Action, ctx: GuardContext) -> tuple[GuardResult, list[GuardResult]]
        # trả về (kết quả quyết định, các cảnh báo WARN đã gặp)
        # chạy lần lượt; gặp BLOCK hoặc STOP thì dừng; WARN thì ghi lại rồi chạy tiếp
```

Bốn lớp `AllowedToolGuard` (G1), `SchemaGuard` (G2), `DuplicateWriteGuard` (G3), `AfterAskGuard` (G4). **Thứ tự chạy là thứ tự trong config** (`[agent].guards`), bảng trên chỉ đánh số để tham chiếu. Thêm guard mới là thêm một lớp vào registry và một dòng config, không sửa `Agent`.

### 4.10. Vòng lặp agent (`lifemate/agent.py`, `lifemate/budget.py`)

```python
class Agent:
    def __init__(self, llm: LLMBackend, transcriber: Transcriber, prompts: PromptBuilder,
                 guards: GuardChain, index: CatalogIndex, cfg: AgentConfig, llm_cfg: LLMConfig,
                 observer: AgentObserver | None = None): ...   # None → NullObserver
    def run(self, sample: Sample, env: ToolEnvLike, *, step_estimate_s: float | None = None,
            clock: Clock = time.monotonic) -> SampleResult
```

`clock` được truyền vào để test hạn giờ mà không phải chờ thật (T10).

```
budget = Budget(cfg, clock)                    # bắt đầu tính giờ từ đây, gồm cả ASR
transcript = transcriber.transcribe(sample)
conv = prompts.start(sample, transcript)
schema = output_schema(allowed_tools)          # có cache
observer.on_sample_start(sample, transcript)
loop:
    reason = budget.before_step(conv.prompt_tokens())   # None nghĩa là được chạy tiếp
    nếu reason: stop(reason)                   # MAX_STEPS | TIMEOUT | CONTEXT_OVERFLOW
    gen = llm.generate(conv.messages, schema, sample_id=sample.id)
    budget.after_step(gen.seconds)
    conv.reconcile(gen.prompt_tokens)          # số của vLLM là mốc, trước khi thêm câu trả lời
    conv.append_assistant(gen.text, gen.completion_tokens)
    action = parse(gen.text)
    ParseError           → budget.retry() hoặc stop(PARSE_ERROR); conv.append_user("LỖI ...")
    done                 → stop(DONE)
    guards: BLOCK        → budget.retry() hoặc stop(RETRIES_EXHAUSTED); conv.append_user("LỖI (chưa gọi tool): ...")
    guards: STOP         → stop(DONE)
    guards: PASS / WARN  → result = env.call(name, arguments)
                           conv.append_user("KẾT QUẢ: " + truncate(canonical_json(result), cfg.max_result_chars))
                           nếu name == "sys.ask_missing_info": stop(ASKED)
    observer.on_step(StepRecord(...))
result = SampleResult(tool_calls=env.get_tool_calls(), state_hash=env.get_state_hash(), ...)
observer.on_stop(result)
```

**`Budget` (hạn giờ dự đoán, quyết định S12):**

```python
class Budget:   # Budget(cfg: AgentConfig, llm_cfg: LLMConfig, clock, *, step_estimate_s)
    def before_step(self, prompt_tokens: int) -> StopReason | None
        # MAX_STEPS        khi đã chạy cfg.max_steps bước
        # CONTEXT_OVERFLOW khi prompt_tokens + llm_cfg.max_tokens > llm_cfg.max_model_len
        # TIMEOUT          khi elapsed + estimate > cfg.hard_limit_s
    def after_step(self, seconds: float) -> None
    def retry(self) -> bool                        # False khi đã dùng hết cfg.max_retries
```

- `estimate` là thời gian dự kiến của bước sắp chạy, bằng giá trị lớn hơn trong hai số: thời gian bước vừa rồi của chính mẫu này, và `step_estimate_s` do Runner truyền vào (p95 thời gian một bước của các mẫu đã chạy). `step_estimate_s` là `None` thì dùng `cfg.initial_step_estimate_s`.
- `hard_limit_s = 28`: chừa 2 giây cho ghi kết quả và sai số. Như vậy sẽ không bắt đầu một bước mà dự kiến kết thúc sau 28 giây.
- Thời gian ASR được tính vào `elapsed` vì `Budget` bắt đầu tính trước khi gọi ASR.

**Quy tắc khác:**

- `Agent.run` không bắt exception. Exception được bắt ở `Runner` (mục 4.11). Mẫu đó vẫn ghi dòng nộp với `env.get_tool_calls()` và `env.get_state_hash()` đang có, cùng mã `exception`. **Không bao giờ dừng cả lần chạy** vì một mẫu lỗi.
- `max_steps = 10` lấy từ: gold tối đa 6 call, cộng một lượt `done`, cộng tối đa 2 lần thử lại, cộng dư 1.
- Ngưỡng cắt 4000 ký tự lớn hơn kết quả dài nhất đã thấy (2439). Bị cắt thì cuối chuỗi thêm `…(đã cắt)`. Cắt **trước** khi thêm vào `Conversation` (quyết định S10).

### 4.11. Runner, RunStore và CLI (`runner.py`, `store.py`, `cli.py`)

```
# Máy local: data_root và runs_dir lấy từ configs/env/local.toml
uv run python -m lifemate.cli --config configs/v0.toml --env local \
    --split train --ids splits/val_seen.txt --name 2026-10-28_v0-<llm>_val-seen [--resume] [--limit N]

# Ghi đè profile khi cần
uv run python -m lifemate.cli --config configs/v0.toml --env kaggle --root /kaggle/input/other/train+public_test ...
```

`--name` là tên thư mục con trong `runs_dir` của profile; `--out` là **chính thư mục run** (ghi đè `runs_dir/name`, khớp với `--out result/` ở Docker). Khi có `--out`, `index.csv` nằm ở thư mục cha của nó. Có thêm profile `local.toml` cho máy phát triển (đường dẫn tương đối trong repo).

`cli.py` chỉ parse tham số, đọc `Config`, gọi `factory.build(config)` rồi `Runner.run(...)`.

**`Runner.run(samples, store)`:**

1. Nạp model, gọi `llm.warmup()` và `transcriber.warmup()` (quyết định S11). Ghi thời gian nạp và warm-up vào `config.json`.
2. Với từng mẫu theo thứ tự id: `open_env` → `agent.run(sample, env, step_estimate_s=...)` trong `try/except Exception`. Đây là **chỗ duy nhất** bắt mọi exception. Lỗi thì tạo `SampleResult` với mã `exception` và traceback. **Nếu lỗi xảy ra trước khi có `env`** (B4), ghi `tool_calls=[]`, `state_hash=""`; dòng nộp vẫn phải có để file đủ id.
3. `store.append(result)` ngay sau mỗi mẫu.
4. Cập nhật p95 thời gian một bước từ các mẫu đã chạy và truyền vào `agent.run` của mẫu sau qua `step_estimate_s` (B6). Mẫu đầu dùng `cfg.initial_step_estimate_s`.
5. Chạy xong: `store.finalize()`, rồi gọi `eval/score.py` **bằng subprocess** nếu split có nhãn (import trong process sẽ nạp `zaic_tools` lần hai).

**`RunStore`:**

- `trace.jsonl`: mở ở chế độ append, `flush()` và `os.fsync()` sau mỗi mẫu. Phiên Kaggle bị ngắt thì chỉ mất tối đa mẫu đang chạy.
- `submission.jsonl` và `summary.json`: ghi ra file `*.tmp` rồi `os.replace` (ghi nguyên tử). Không bao giờ để lại file nộp ghi dở.
- `--resume`: đọc `trace.jsonl`, bỏ qua dòng cuối nếu dòng đó hỏng (bị ngắt giữa chừng), bỏ qua các id đã có. `submission.jsonl` được dựng lại từ `trace.jsonl` ở `finalize()`.
- Thư mục `--out` đã có `trace.jsonl` mà không truyền `--resume` thì báo lỗi và thoát, không ghi đè.
- `--ids` không truyền thì chạy mọi mẫu của split. `--limit` lấy N id đầu tiên, dùng để chạy thử.
- Split có nhãn (train) thì cuối lần chạy tự gọi `eval/score.py submission.jsonl --ids <run>/ids.txt --match both --details details.jsonl` và ghi kết quả ra `score.txt`. `ids.txt` là danh sách id thực sự đã chạy (sau `--limit`), để các id chưa chạy không bị chấm 0. `summary.json.score` = `{total, by_mode: {strict, canon}, by_category}`.
- Thêm một dòng vào `runs/index.csv`: `run, date, git_commit, model, prompt_hash, split, ids_file, n, score, p95_seconds`.
- Split **không có nhãn** (public, private): `finalize()` chạy `check_submission` (B3) thay cho chấm điểm, ghi kết quả ra `check.txt`:
  1. Mọi id của split có mặt đúng một lần, JSON hợp lệ.
  2. `python -m zaic_tools --split <split> --root <root> --validate submission.jsonl` không báo lỗi (subprocess, cwd là thư mục split).
  3. Phân bố: tỉ lệ `[]`, tỉ lệ có `sys.ask_missing_info`, số call trung bình, so với train (7%, 14%, 2,3). Lệch quá 2 lần thì in cảnh báo, không chặn.
  4. Phát lại các call qua `zaic_tools` và đếm `ok: false`; khác 0 thì in cảnh báo kèm id.
- `runs/public_submissions.csv` (ghi tay sau mỗi lần nộp public): `date, run, public_score, val_seen, val_unseen, note`. Dùng để xem điểm val và public có tăng giảm cùng chiều không.
- **Bộ smoke cố định:** `splits/smoke.txt` gồm 30 id lấy phân tầng từ `val_seen` (theo loại đáp án và có/không history), tạo bởi `make_splits.py --smoke 30`. Dùng cho chạy thử sau mỗi lần sửa; `--limit N` chỉ dành cho kiểm tra cài đặt.

### 4.12. Thư mục một lần chạy

```
runs/<name>/
├── config.json        # nội dung config TOML đã đọc, cộng thêm: git commit, git dirty, prompt sha256,
│                      # phiên bản vllm/torch/transformers, tên GPU, compute capability, thời điểm bắt đầu
├── trace.jsonl        # mỗi mẫu một dòng (schema bên dưới)
├── submission.jsonl   # {"id","tool_calls","state_hash"}, đúng định dạng nộp bài
├── score.txt          # chỉ có khi split có nhãn
├── details.jsonl      # chỉ có khi split có nhãn, do score.py ghi
└── summary.json
```

**Một dòng `trace.jsonl`:**

```json
{"id": "train_0874", "stop_reason": "done", "errors": [], "prompt_shrink_level": 0,
 "transcript": {"final": "...", "history": [null, "..."], "source": "gold"},
 "steps": [
   {"i": 1, "prompt_tokens": 3120, "completion_tokens": 41, "llm_s": 0.82,
    "raw": "{\"action\":\"call\",...}", "action": {"kind": "call", "name": "contacts.search", "arguments": {"query": "Huy"}},
    "guard": "pass", "warnings": [], "result": {"ok": true, "contacts": []}, "tool_s": 0.001}
 ],
 "tool_calls": [], "state_hash": "…",
 "timing": {"asr_s": 0.0, "llm_s": 1.6, "tool_s": 0.002, "total_s": 1.7},
 "tokens": {"max_prompt": 3300, "completion_total": 80}}
```

**`summary.json`:**

```json
{"n": 193, "n_exception": 0,
 "stop_reason": {"done": 170, "asked": 20, "max_steps": 1, "timeout": 0, "context_overflow": 0, "parse_error": 0, "retries_exhausted": 2, "exception": 0},
 "prompt_shrink_level": {"0": 190, "1": 3, "2": 0, "3": 0},
 "errors": {"invalid_args": {"count": 5, "ids": ["train_0012", "..."]}},
 "seconds": {"p50": 2.1, "p95": 5.0, "max": 9.3, "over_30": 0, "over_25": 0},
 "tokens": {"max_prompt": 9800, "over_14k": 0},
 "score": {"total": 0.0, "by_category": {"execute": 0.0, "ask_missing_info": 0.0, "out_of_scope": 0.0}}}
```

`errors[*].ids` giữ tối đa 10 id mỗi loại.

### 4.13. Mã lỗi và lý do dừng

Cả hai là `Enum` trong `lifemate/types.py`. Ghi ra JSON bằng `.value`.

**`ErrorCode`:**

| Mã | Ý nghĩa | Nơi ghi |
|---|---|---|
| `parse_error` | Đầu ra LLM không tách được thành `Action` | `steps[i].error` và `errors` |
| `unknown_tool` | Tool không được phép (G1) | `steps[i].guard = "block"`, `steps[i].error`, `errors` |
| `invalid_args` | `env.validate` từ chối (G2) | như trên |
| `duplicate_write` | Gọi lại một call `write` giống hệt (G3, chỉ cảnh báo) | `errors` |
| `exception` | Exception ngoài dự kiến, bắt ở `Runner` | `errors`, kèm traceback trong trace |

**`StopReason`:** `done`, `asked`, `max_steps`, `timeout` (dự kiến vượt `hard_limit_s`), `context_overflow`, `parse_error`, `retries_exhausted`, `exception`.

### 4.14. Config (`configs/v0.toml`)

Đọc vào các dataclass `LLMConfig`, `AgentConfig`, `ASRConfig` trong `lifemate/config.py`. Khóa lạ hoặc thiếu khóa thì báo lỗi ngay khi khởi động.

```toml
[llm]
backend = "vllm"                       # "vllm" | "fake"
model = ""                             # CHƯA CHỌN: tên model trên HF, điền khi chốt (mục 6.0)
revision = ""                          # CHƯA CHỌN: commit hash cố định của model
dtype = "auto"                         # "auto" | "float16" | "bfloat16"; auto chọn theo GPU (mục 4.16)
max_model_len = 16384
gpu_memory_utilization = 0.85          # giảm khi có ASR chạy chung GPU
seed = 0
max_tokens = 512
batch_invariant = false                # true khi chạy trên sm80+
warmup_runs = 2

[agent]
max_steps = 10
max_retries = 2
hard_limit_s = 28                      # không bắt đầu bước nào dự kiến kết thúc sau mốc này
initial_step_estimate_s = 3.0          # ước lượng cho mẫu đầu tiên của lần chạy
max_result_chars = 4000
system_prompt = "prompts/system_v0.md"
guards = ["allowed_tool", "schema", "duplicate_write"]   # "after_ask" có sẵn nhưng không dùng, xem 4.9

[asr]
backend = "gold"                       # "gold" | "asr" (sau v0)
engine = "hf_seq2seq"                  # "hf_seq2seq" | "chunkformer" | "viegem": cách nạp model khi backend = "asr" (mục 6.1)
model = "vinai/PhoWhisper-medium"      # bắt buộc khi backend = "asr" (mục 6.0)
revision = ""
```

### 4.15. Sửa bộ chấm `eval/score.py`

Bộ chấm phải khớp với cách BTC chấm càng sát càng tốt. Ba chỗ cần sửa:

| # | Chỗ | Vấn đề | Sửa |
|---|---|---|---|
| C1 | `score.py:87` `sa[k] != gv` | Python coi `True == 1` và `1 == 1.0`. Agent nộp `true` cho tham số có gold là `1` vẫn được tính là khớp | So sánh có xét kiểu: `bool` chỉ bằng `bool`; `int` và `float` bằng nhau khi cùng giá trị; list và dict so đệ quy theo cùng quy tắc |
| C2 | Cùng chỗ | So khớp phân biệt hoa thường. Nhưng `zaic_tools/utils.py` (`_canon_scalar`, `canon`) cho thấy BTC chuẩn hóa chuỗi khi tính hash: chữ thường, gộp khoảng trắng, `700 cm` = `700cm`. Rất có thể BTC so tham số theo cùng cách | Thêm `--match strict|canon|both` (mặc định `strict`). `canon` dùng đúng `_canon_scalar` của `zaic_tools`. `both` in cả hai điểm để biết điểm thật nằm trong khoảng nào, chi tiết theo `strict` |
| C3 | `score.py:239, 269`; `make_splits.py` (`json.load(open(...))`) | Mở file không có `with` | Dùng `with` hoặc `Path.read_text` |

Giữ nguyên:

- Thuật toán ghép cặp Kuhn (`max_matching`). Mỗi mẫu tối đa 6 call nên đã đủ nhanh.
- Phát lại gold bằng `zaic_tools` để lấy hash. An toàn hơn tính hash từ `final_state`.
- Coi mọi `ok: false` là call bị từ chối. Đã kiểm: các app trong `zaic_tools` chỉ trả `ok: false` khi sai schema, và 0 kết quả gold có `ok: false`.

Kiểm chứng sau khi sửa: gold vẫn được 100,00 ở cả hai chế độ, toàn `[]` vẫn được 7,18.

### 4.16. Môi trường chạy: Kaggle, Colab, máy thuê, Docker

Cùng một code chạy được ở bốn nơi. Mọi khác biệt giữa các nơi nằm trong **profile môi trường**, không nằm trong code.

**Profile (`configs/env/<tên>.toml`), chọn bằng `--env <tên>`:**

```toml
# configs/env/kaggle.toml
data_root = "/kaggle/input/zac2026-lifemate/train+public_test"   # thư mục chứa các split, chỉ đọc
runs_dir = "/kaggle/working/runs"
hf_cache = "/kaggle/working/hf_cache"
hf_token_source = "kaggle_secret:HF_TOKEN"                        # "kaggle_secret:<tên>" | "colab_userdata:<tên>" | "env:<tên>" | "none"
persist = "kaggle_dataset"                                        # cách giữ runs sau phiên, xem bảng dưới
```

| Khóa | Kaggle | Colab | Máy thuê | Docker |
|---|---|---|---|---|
| `data_root` | `/kaggle/input/<dataset>/...` (chỉ đọc) | `/content/drive/MyDrive/zac2026/...` | `~/zac2026/...` | `/tmp/zaic_root` (symlink, mục 6.3) |
| `runs_dir` | `/kaggle/working/runs` | `/content/drive/MyDrive/zac2026/runs` | `~/zac2026/runs` | `result/` |
| `hf_cache` | `/kaggle/working/hf_cache` | `/content/hf_cache` | `~/.cache/huggingface` | trong image, `HF_HUB_OFFLINE=1` |
| `hf_token_source` | `kaggle_secret:HF_TOKEN` | `colab_userdata:HF_TOKEN` | `env:HF_TOKEN` | `none` |
| `persist` | `kaggle_dataset`: cuối phiên lưu `runs/` thành một version của Kaggle Dataset riêng tư | `drive`: ghi thẳng vào Drive, không cần làm gì thêm | `local` | `local` |

- Profile thứ năm `local.toml` cho máy phát triển: `data_root = "train+public_test/train+public_test"`, `runs_dir = "runs"`, `hf_cache` mặc định của HF, `hf_token_source = "env:HF_TOKEN"`, `persist = "local"`. Đường dẫn tương đối tính từ thư mục gốc repo.
- Thứ tự ưu tiên: tham số dòng lệnh (`--root`, `--out`) > profile > giá trị mặc định. Code trong `lifemate/` **không có đường dẫn nào gắn cứng**. `Runner` truyền `--root` tường minh khi gọi `eval/score.py`.
- `data_root` có thể chỉ đọc. Code không ghi gì vào `data_root`. `zaic_tools` được import từ đó, và Python tự bỏ qua việc ghi `__pycache__` khi thư mục chỉ đọc.
- Token HF chỉ được đọc lúc chạy theo `hf_token_source`. Không bao giờ ghi token vào config, notebook, trace hay `config.json`.
- Trước khi tải model, kiểm tra chỗ trống của ổ chứa `hf_cache`. Không đủ khoảng 1,5 lần kích thước model thì báo lỗi rõ ràng.

**Tự nhận GPU (`lifemate/envinfo.py`):**

```python
@dataclass(frozen=True, slots=True)
class EnvInfo:
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    gpu_mem_gb: float | None
    code_version: str           # git commit, hoặc nội dung file VERSION
    code_dirty: bool | None
    packages: dict[str, str]    # phiên bản thật của vllm, torch, transformers, ...

def detect() -> EnvInfo
def resolve_dtype(cfg_dtype: str, info: EnvInfo) -> str
    # "auto": sm80+ thì "bfloat16", thấp hơn thì "float16"; giá trị khác "auto" giữ nguyên
    # config yêu cầu "bfloat16" trên GPU < sm80 thì raise
```

- `config.json` của mỗi lần chạy ghi cả giá trị trong config **và** giá trị thật đã dùng, ví dụ `dtype: auto → float16`.
- `code_version`: lấy từ `git rev-parse HEAD` cộng `git status --porcelain`. Nếu không có `.git` (code đưa lên Kaggle dạng dataset hoặc zip) thì đọc file `VERSION` do `scripts/pack_code.sh` ghi lúc đóng gói. Không có cả hai thì ghi `"unknown"` và in cảnh báo.
- `gpu_memory_utilization` vẫn đặt trong config, vì nó phụ thuộc việc có chạy ASR chung GPU hay không. `config.json` ghi kèm `gpu_mem_gb` để đối chiếu.

**Cài đặt:**

| Nơi | Cách cài | Lý do |
|---|---|---|
| Kaggle, Colab | Giữ torch và CUDA có sẵn. Cài các gói còn lại bằng `pip install -r requirements/notebook.txt`. File này sinh từ `uv.lock` nhưng bỏ torch và các gói CUDA. Chọn phiên bản vLLM khớp với torch có sẵn | Cài lại torch trên Kaggle hay Colab dễ làm hỏng CUDA và tốn thời gian |
| Máy thuê, Docker | `uv sync --frozen` theo `uv.lock` đầy đủ | Môi trường sạch, khóa chặt phiên bản |

Phiên bản thật của các gói được ghi vào `config.json` (`EnvInfo.packages`), để biết hai lần chạy có dùng cùng phiên bản hay không.

**Đưa code lên Kaggle và Colab:** `scripts/pack_code.sh` nén `lifemate/`, `eval/`, `configs/`, `prompts/`, `splits/`, `pyproject.toml` và các file requirements, kèm file `VERSION` (commit và trạng thái dirty). Kaggle dùng file nén này làm Dataset riêng tư. Colab dùng `git clone` hoặc file nén trên Drive.

**Notebook mẫu (`notebooks/kaggle_run.ipynb`, `notebooks/colab_run.ipynb`)** gồm các ô theo thứ tự:

1. Cài đặt.
2. In `EnvInfo`.
3. Chạy `pytest` nhanh (bỏ T1).
4. Chạy `--limit 30`.
5. Chạy đủ `val_seen` và `val_unseen`.
6. Lưu `runs/` theo `persist`.

Notebook chỉ gọi lệnh, không chứa logic. Mọi logic nằm trong `lifemate/`.

**Nơi không chạy được:** Kaggle và Colab không chạy được Docker. Việc kiểm tra Docker (mục 6.3) làm trên máy thuê.

---

## 5. Test (pytest, CPU, không cần vLLM)

Mỗi test ghi rõ ba ý: **bảo vệ** điều gì, **bắt lỗi** khi nào, và **không chứng minh** điều gì.

- Unit test dùng `FakeEnv`, `FakeLLM` và `WhitespaceTokenCounter` (đếm theo khoảng trắng), không cần dữ liệu thi hay tokenizer.
- T1 dùng `zaic_tools` và dữ liệu thật. T15 cần tokenizer của model trong cache HF; không có thì test bị skip kèm lý do, không được coi là đạt.

| # | Test | Bảo vệ | Bắt lỗi khi | Không chứng minh |
|---|---|---|---|---|
| T1 | `test_fake_llm_end_to_end`: FakeLLM chạy toàn bộ `Runner` trên `val_seen` + `val_unseen`, chấm bằng `score.py` | Vòng lặp, guard, `RunStore` ghi đúng call và hash | Điểm khác 100,00 | Chất lượng của LLM thật |
| T2 | `test_tooldocs_complete`: render mọi app của catalog train và public | Không bỏ sót tham số, `required`, `enum`, `x-one-of`, `x-ref`, `x-match`, `returns` | Một trường có trong catalog nhưng không có trong chuỗi render | Mô tả có dễ hiểu với LLM hay không |
| T3 | `test_tooldocs_deterministic` | Prompt giống nhau giữa các lần chạy | Hai lần render cùng đầu vào ra chuỗi khác nhau | |
| T4 | `test_parse_cases`: khoảng 15 đầu ra mẫu (JSON đúng, thiếu ngoặc, thừa chữ trước sau, sai `action`, `arguments` không phải object) | `parse` không bao giờ raise | Raise, hoặc trả `Action` cho đầu ra hỏng | |
| T5 | `test_guards`: tool lạ, thiếu tham số, gọi `write` trùng; vòng lặp dừng ngay sau `ask_missing_info` | Call bị G1–G2 chặn **không** vào `env.get_tool_calls()`. Call `write` trùng vẫn được gọi và có cảnh báo. Sau `ask_missing_info` không có call nào nữa dù FakeLLM còn call | Call bị chặn vẫn có trong log của env, call trùng bị chặn, hoặc có call sau `ask` | Guard có đủ để ngăn mọi lỗi điểm |
| T6 | `test_prompt_no_label`: dựng prompt từ một `Sample` có `label` chứa chuỗi đánh dấu đặc biệt | Prompt không lộ đáp án | Chuỗi đánh dấu xuất hiện trong messages | |
| T7 | `test_held_out_stripped`: dựng prompt `train_mode` cho mẫu có cài app bị rút, và cho mẫu có history gọi app bị rút | Không có tên app bị rút ở bất kỳ đâu trong messages; mẫu có history gọi app bị rút thì raise `HeldOutLeak` | Tên app bị rút xuất hiện, hoặc không raise | |
| T8 | `test_one_env_per_sample` | `open_env` gọi hai lần cùng id thì raise | Không raise | |
| T9 | `test_no_environment_access`: quét mã nguồn `lifemate/` | Code không tự đọc `state.json` | Có chuỗi `_environment` hoặc `state.json` ngoài `zaic.py` | Thư viện bên thứ ba không đọc |
| T10 | `test_budget_predictive`: `clock` giả, mỗi bước tốn 5 giây, `hard_limit_s = 28` | Dừng với `timeout` **trước** bước sẽ vượt mốc (sau bước kết thúc ở giây 25, không chạy bước thứ 6), vẫn ghi được dòng nộp | Bắt đầu một bước dự kiến kết thúc sau 28 giây, hoặc không ghi dòng nộp | Thời gian thật trên A30 |
| T11 | `test_exception_isolated`: một mẫu làm transcriber raise | Lần chạy vẫn đi tiếp, mẫu lỗi có dòng nộp với mã `exception` | Cả lần chạy dừng | |
| T12 | `test_resume`: ngắt sau mẫu thứ 3, thêm một dòng hỏng vào cuối `trace.jsonl`, chạy lại với `--resume` | Không chạy lại mẫu đã xong, bỏ qua dòng hỏng, `submission.jsonl` đủ id và không trùng | Thiếu id, trùng id, chạy lại, hoặc crash vì dòng hỏng | |
| T13 | `test_output_schema`: schema cho một mẫu | `name` chỉ gồm tool được phép cộng `sys.ask_missing_info`, và hai mẫu cùng bộ tool cho ra cùng một chuỗi schema | Có tool ngoài danh sách, hoặc chuỗi schema khác nhau | vLLM có tuân schema hay không (kiểm ở mục 7) |
| T14 | `test_conversation_append_only`: chạy vòng lặp với FakeLLM, chụp lại messages ở mỗi bước | Message cũ không bao giờ bị sửa; prompt gồm phần quy tắc giống hệt nhau giữa mọi mẫu | Một message cũ thay đổi, hoặc phần đầu system khác nhau giữa hai mẫu | Tỉ lệ trúng cache thật của vLLM |
| T15 | `test_token_count_incremental`: so số token đếm tăng dần với số token khi đếm lại toàn bộ, trên 20 mẫu | Đếm tăng dần đúng | Lệch nhau | |
| T16 | `test_atomic_writes`: giả lập lỗi giữa lúc ghi `submission.jsonl` | Không còn file nộp ghi dở; file cũ (nếu có) giữ nguyên | Có file nộp bị cắt cụt | Lỗi ở mức hệ điều hành |
| T17 | `test_score_match_modes` cho `score.py`: `true` với `1`, `1` với `1.0`, `"Phòng Khách"` với `"phòng khách"` | C1, C2 đúng như mục 4.15; gold vẫn được 100,00 ở cả hai chế độ | Sai một trong các cặp trên | BTC chấm đúng như chế độ nào |
| T18 | `test_guard_chain_open_closed`: thêm một guard giả qua config | `Agent` gọi guard mới mà không cần sửa code | Guard mới không được gọi | |
| T19 | `test_env_profiles`: nạp cả 5 profile, chạy với tham số dòng lệnh ghi đè | Thứ tự ưu tiên dòng lệnh > profile > mặc định; mọi profile có đủ khóa | Thiếu khóa, hoặc thứ tự ưu tiên sai | Đường dẫn thật trên Kaggle hay Colab tồn tại |
| T20 | `test_resolve_dtype`: GPU giả sm75 và sm80 | `auto` ra `float16` trên sm75 và `bfloat16` trên sm80; yêu cầu `bfloat16` trên sm75 thì raise | Chọn sai dtype hoặc không raise | |
| T21 | `test_no_hardcoded_paths`: quét `lifemate/` | Không có chuỗi `/kaggle`, `/content`, `/home` hay `train+public_test` | Có | Đường dẫn ghép động lúc chạy |
| T22 | `test_code_version_fallback`: thư mục không có `.git`, có và không có `VERSION` | Đọc được `VERSION`; không có cả hai thì ra `"unknown"` kèm cảnh báo | Crash | |
| T23 | `test_no_token_leak`: đặt token giả, chạy `--limit 2` với FakeLLM | Token không xuất hiện trong `config.json`, `trace.jsonl`, `summary.json` | Xuất hiện | |
| T24 | `test_prompt_shrink_levels`: `WhitespaceTokenCounter` với `max_model_len` nhỏ dần, một mẫu có history dài | Mỗi mức giảm được áp dụng đúng thứ tự 1 → 2 → 3, dừng ở mức đầu tiên vừa; mức 3 luôn giữ lượt user cuối; phần quy tắc của system không đổi ở mọi mức; không vừa sau mức 3 thì `CONTEXT_OVERFLOW` | Sai thứ tự, mất lượt cuối, hoặc phần quy tắc thay đổi | Số token thật của tokenizer model |
| T25 | `test_check_submission`: file nộp giả thiếu id, trùng id, có call sai schema | `check.txt` nêu đúng từng lỗi; file hợp lệ thì không cảnh báo | Bỏ sót một loại lỗi | Cách BTC validate khác với `zaic_tools --validate` |
| T26 | `test_smoke_split`: `make_splits.py --smoke 30` | `smoke.txt` có 30 id, đều thuộc `val_seen`, có đủ 3 loại đáp án, cố định theo seed | Thiếu loại, id ngoài `val_seen`, hoặc đổi giữa hai lần chạy | |

---

## 6. Các khối sau v0 (mức interface)

### 6.0. Điều kiện chọn LLM và ASR (đã chọn 27/10/2026)

| Điều kiện | LLM | ASR |
|---|---|---|
| Số tham số | < 5B | Tổng ASR + LLM + mọi thành phần khác < 6B |
| Context | Hỗ trợ ít nhất 16K token (`max_model_len = 16384`) | |
| Phát hành | Trước 26/10/2026, giấy phép cho phép dùng | Như LLM |
| Chạy được | vLLM bản đã khóa, trên cả T4 (fp16) và sm80+ (bf16); có chat template | Chạy offline trong Docker, giải mã greedy |
| Tiếng Việt | Hiểu tiếng Việt, kể cả câu xen tiếng Anh | Nhận dạng tiếng Việt, kể cả từ tiếng Anh xen kẽ |

**Đã chọn** (lý do và kế hoạch fine-tune ở `approach.md` mục 5.5): LLM `Qwen/Qwen3-4B-Instruct-2507` (4,02B, Apache-2.0, không có chế độ thinking), ASR `vinai/PhoWhisper-medium` (0,77B, BSD-3, `engine = "hf_seq2seq"`). Tổng 4,79B. Đã điền `model` và `revision` vào `configs/v0.toml`, `configs/docker.toml` (đường dẫn `/app/models/*`) và `configs/train/*.toml`. Ứng viên A/B: `Qwen/Qwen3.5-4B`; ASR `chunkformer` và `viegem` (hai engine tùy chọn trong `lifemate/asr.py`, extras `asr-chunkformer` / `asr-viegem`, giấy phép phi thương mại cần BTC xác nhận).

**Trước khi chọn LLM, v0 vẫn làm được:** mọi việc trừ #4 (VLLMBackend chạy thật) và #8 (chạy v0 trên Kaggle) đều không cần model. `config.py` báo lỗi rõ ràng nếu `llm.backend = "vllm"` mà `model` còn trống.

### 6.1. ASR (A2, A3, B5)

- `ASRTranscriber` cài theo interface `Transcriber`. Model được nạp qua `[asr].engine` (`ASR_ENGINES` trong `asr.py`): `hf_seq2seq` (transformers pipeline, dùng cho PhoWhisper), `chunkformer` (gói `chunkformer`, CTC), `viegem` (code trong snapshot `5CD-AI/VieGem-ASR`, CTC). Engine mới chỉ cần một class có `transcribe_batch` và `warmup` rồi đăng ký vào `ASR_ENGINES`.
- Chạy chung GPU với vLLM. **Thứ tự nạp (B2):** nạp vLLM trước với `gpu_memory_utilization` đã chừa chỗ cho ASR (vLLM tính tỉ lệ trên tổng VRAM lúc khởi động), nạp ASR sau, rồi warm-up cả hai. `Runner` đo VRAM còn trống sau khi nạp xong và ghi vào `config.json`; ASR hết chỗ thì báo lỗi ngay, không chạy mẫu nào.
- Chép chữ cho **mọi** lượt user trong history, tối đa 13 file mỗi mẫu. Thời gian ASR tính vào 30 giây.
- Tổng thời lượng audio mỗi mẫu (đo 27/10/2026): train p50 13 s, p95 41 s, max 69 s; public p50 13 s, p95 40 s, max 83 s (11 file).
- **Cách chạy ASR:**
  1. Gom mọi file audio **của cùng một mẫu** (lượt cuối và các lượt user trong history) vào một batch, theo thứ tự cố định: `_u1, _u2, …`, rồi lượt cuối. Batch có thể nhỏ hơn số file nếu GPU không đủ bộ nhớ, nhưng cách chia phải cố định theo thứ tự đó.
  2. **Không** gom file của nhiều mẫu khác nhau, **không** chạy ASR trước cho cả split rồi mới chạy agent. Mỗi mẫu đi trọn ASR → agent → ghi kết quả rồi mới sang mẫu sau (quyết định S5).
  3. Giải mã greedy (`temperature = 0`, `beam_size = 1`), cố định seed.
  4. Chỉ khi đo thấy ASR quá chậm mới tính đến việc bỏ bớt các lượt user cũ. Khi đó vẫn giữ `text` và `tool_calls` của các lượt assistant để giữ ngữ cảnh.
- **Đo trước khi chọn model ASR:** chạy trên 3 mẫu có tổng audio dài nhất của train và public, ghi thời gian ASR và tổng thời gian mỗi mẫu. Trên GPU sm80+, tổng thời gian mỗi mẫu phải ≤ 25 s.
- **Normalizer (A3): không làm ở v0 và chưa chắc làm sau đó (A2).** LLM được fine-tune trên `audio_text` nguyên văn (6.2.1) và ASR nhắm tới `audio_text` nguyên văn (6.4.1), nên đổi chữ ASR thành số trước khi đưa vào LLM sẽ lệch khỏi phân bố đã train. Quy tắc đổi số nằm trong system prompt và do LLM làm. Chỉ cân nhắc Normalizer nếu bảng lỗi cho thấy LLM đổi số sai nhiều, và khi đó phải áp dụng cùng một `normalize` cho transcript lúc dựng dữ liệu SFT (6.2.1) và lúc chạy.
- **Nghiệm thu:**
  - Báo cáo WER và CER trên `val_seen` và `val_unseen`, dùng `audio_text` làm đáp án.
  - Báo cáo điểm agent khi dùng ASR trừ điểm khi dùng `GoldTranscriber`, đây là phần mất do ASR.
  - p95 thời gian ASR mỗi mẫu được ghi trong `summary.json`.
  - Chạy ASR hai lần trên 30 mẫu cho ra chữ giống hệt nhau.

### 6.2. Fine-tune LLM (C1, C2, C4)

Phần này chia hai nhóm:

- **Cố định ngay:** định dạng dữ liệu, cách che loss, quy tắc dữ liệu, cách đưa model đã train vào hệ thống, quản lý phiên bản. Những thứ này ràng buộc code v0 nên phải chốt trước.
- **Chốt sau khi có số đo v0:** rank, learning rate, số epoch, có cần dữ liệu tổng hợp hay không. Các giá trị dưới đây chỉ là mặc định để bắt đầu.

#### 6.2.1. Dữ liệu SFT (`lifemate/train/sft_data.py`)

```
uv run python -m lifemate.train.sft_data --config configs/train/llm_v1.toml --env kaggle \
    --ids splits/train.txt --out data/sft/v1.jsonl
```

**Quy tắc dựng một mẫu, phải giống hệt lúc chạy thật:**

1. Dựng prompt bằng đúng `PromptBuilder` và `system_v0.md` của lúc chạy. Sau đó, **xóa 8 app bị rút** (`splits/meta.json` → `held_out_apps`) khỏi `installed_apps` trước khi render tool docs (mục 4.5).
2. Transcript lấy từ `audio_text` (mặc định, `transcript_source = "gold"`). Tùy chọn `"asr"`: dùng chữ do ASR nghe ra, để model quen với lỗi ASR. Chỉ dùng khi đã có ASR (mục 6.1).
3. Với mỗi call `i` trong `label.tool_calls`:
   - Thêm lượt assistant với nội dung `canonical_json({"action":"call","name":...,"arguments":...})`.
   - Thêm lượt user `KẾT QUẢ: ...`. Kết quả này lấy bằng cách **phát lại call qua `zaic_tools`**, không chép từ `label.tool_results`, để khớp từng byte với lúc chạy thật. Sau đó cắt theo `max_result_chars` giống hệt mục 4.10. Nếu kết quả phát lại khác `label.tool_results[i]` thì ghi cảnh báo kèm id.
4. Kết thúc:
   - Call cuối là `sys.ask_missing_info`: **không** thêm `done`, vì lúc chạy vòng lặp dừng ngay sau call đó (`StopReason.ASKED`).
   - Ngược lại: thêm lượt assistant `{"action":"done"}`.
   - Mẫu gold `[]`: chỉ có một lượt assistant `{"action":"done"}`.
5. Đếm token bằng `HFTokenCounter` của model gốc. Mẫu vượt `max_model_len` thì bỏ, ghi id vào `data/sft/v1.dropped.txt`.

**Định dạng một dòng:**

```json
{"id": "train_0874", "source": "train", "messages": [{"role": "system", "content": "..."}, ...],
 "n_tokens": 4210, "n_assistant_turns": 3}
```

Đi kèm `data/sft/v1.meta.json`: sha256 của file, lệnh đã chạy, `code_version`, sha256 của prompt, danh sách id, số mẫu bị bỏ, số cảnh báo phát lại.

**Dữ liệu chỉ lấy từ:** `splits/train.txt`, và sau này thêm dữ liệu tổng hợp (mục 6.2.5). **Không bao giờ** lấy từ `val_seen` hay `val_unseen` khi còn dùng chúng để chọn model. Không dùng gì từ `public_test`.

#### 6.2.2. Che loss

- Chỉ tính loss trên token của các lượt **assistant**, gồm cả token kết thúc lượt của chat template. Không tính trên system, user, `KẾT QUẢ`.
- Chat template render với cùng tham số như lúc chạy (quyết định S6). Qwen3-4B-Instruct-2507 không có chế độ thinking nên không cần `enable_thinking`; nếu A/B với Qwen3.5-4B thì truyền `enable_thinking=False` ở cả train và chạy.
- Thiếu bước che loss thì model học thuộc mô tả tool và kết quả tool, làm hỏng khả năng đọc app lạ.

#### 6.2.3. Script train (`lifemate/train/sft.py`)

```toml
# configs/train/llm_v1.toml (các số là mặc định ban đầu, chốt sau v0)
[base]
model = "Qwen/Qwen3-4B-Instruct-2507"   # cùng model với [llm] của configs/v0.toml
revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
run_config = "configs/v0.toml"   # lấy prompt, max_result_chars, max_model_len, max_tokens từ đúng config lúc chạy
[data]
train = "data/sft/v1.jsonl"
eval_seen = "data/sft/v1_val_seen.jsonl"      # dựng bằng cùng script, chỉ để đo loss, không train
eval_unseen = "data/sft/v1_val_unseen.jsonl"
[lora]
quant = "nf4"               # QLoRA 4-bit
r = 16
alpha = 32
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
[train]
lr = 2e-4
epochs = 2
batch_size = 1
grad_accum = 8
max_len = 16384
seed = 0
eval_every_steps = 100
save_every_steps = 100
```

- Train bằng Unsloth trên T4, fp16. `seed` cố định cho Python, NumPy, PyTorch và thứ tự trộn dữ liệu.
- **Train tiếp được:** mỗi `save_every_steps` lưu checkpoint (adapter, optimizer, scheduler, trạng thái RNG, bước hiện tại) vào thư mục theo profile môi trường (mục 4.16). `--resume-from <checkpoint>` chạy tiếp đúng từ bước đó. Lý do: một phiên Kaggle chỉ kéo dài khoảng 9–12 giờ.
- Mỗi `eval_every_steps` đo loss trên `eval_seen` và `eval_unseen`, ghi vào `metrics.jsonl`. Checkpoint được lưu trong callback eval, nên `save_every_steps` phải là bội số của `eval_every_steps` (config loader kiểm tra).

**Chọn checkpoint:**

1. Lấy 2 checkpoint có loss `eval_unseen` thấp nhất.
2. Merge từng checkpoint (mục 6.2.4), chạy agent thật trên `val_seen` và `val_unseen` bằng `lifemate.cli`.
3. Chọn checkpoint có điểm `val_unseen` cao nhất. Bằng điểm thì chọn bản có điểm `val_seen` cao hơn.

Chỉ đánh giá bằng agent thật trên 2 checkpoint, để hạn chế việc chọn model theo bộ val quá nhiều lần.

#### 6.2.4. Đưa model đã train vào hệ thống

- **Merge adapter vào model gốc bf16**, lưu thành safetensors đầy đủ. vLLM nạp model đã merge như một model thường, đổi bằng `[llm].model` trong config. Không dùng chế độ LoRA của vLLM, vì merge làm Docker đơn giản hơn và không tốn thêm thời gian chạy.
- Train trên model gốc 4-bit nhưng merge vào bản bf16, nên kết quả có thể lệch nhẹ. Vì vậy **luôn đánh giá model đã merge**, tức đúng thứ sẽ nộp, không đánh giá adapter.
- Merge chạy trên CPU được (T4 không có bf16).

#### 6.2.5. Dữ liệu tổng hợp (chỉ khi bảng lỗi v0 cho thấy cần)

- Sinh bằng LLM lớn trong lúc phát triển (luật cho phép), **chỉ dựa trên catalog train**, không dùng catalog `public_test` (`approach.md` mục 7).
- Mỗi mẫu tổng hợp phải chạy được qua `zaic_tools` của train, và gold của nó phải tái tạo được trạng thái cuối. Mẫu nào không qua thì bỏ.
- Ghi `source = "synthetic"` và phiên bản dữ liệu riêng (`data/sft/synth_vN.jsonl`). Không bao giờ dùng mẫu tổng hợp để đánh giá.
- Một hướng đáng thử: các mẫu dạy model sửa lỗi. Model nhận `LỖI (chưa gọi tool): ...` rồi gọi lại cho đúng, khớp với hành vi của guard ở mục 4.9.

#### 6.2.6. Quản lý phiên bản và chạy lại

- Tên model: `ft-<base>-sft<v>-r<nn>`, ví dụ `ft-base4b-sft1-r01`.
- Mỗi model có thư mục `models/<tên>/` gồm `train_config.json` (toàn bộ config, `code_version`, sha256 dữ liệu, seed, GPU, phiên bản gói, checkpoint được chọn, loss), `metrics.jsonl`, `eval.json` (điểm agent trên hai bộ val, kèm tên run).
- Trọng số lưu trên Kaggle Models hoặc HF private, không vào git. `models/registry.csv` (trong git) gồm `name, base, data_version, checkpoint, val_seen, val_unseen, location`.
- **Một lệnh chạy lại toàn bộ** (yêu cầu nộp training code của BTC):
  ```
  uv run python -m lifemate.train.llm_pipeline --config configs/train/llm_v1.toml --env rented
  ```
  Lệnh này lần lượt: dựng dữ liệu → train → chọn checkpoint → merge → chạy agent trên `--limit 30` để kiểm tra nhanh.

#### 6.2.7. Nghiệm thu

1. Test (CPU, model nhỏ giả hoặc tokenizer thật):
   - Phần prompt của một mẫu SFT giống từng byte với `Conversation.messages` của lúc chạy ở cùng bước.
   - Token được tính loss khi decode ra bằng đúng các chuỗi JSON của assistant cộng token kết thúc lượt.
   - Không có tên app bị rút trong dữ liệu dựng từ `splits/train.txt`.
   - Mẫu `ask_missing_info` không có lượt `done` ở cuối.
   - Không id nào của `val_seen` hay `val_unseen` lọt vào file train.
2. Phát lại toàn bộ dữ liệu SFT qua `FakeLLM` (dùng các lượt assistant làm đầu ra) cho ra 100,00 điểm trên các id của train.
3. Ngắt train giữa chừng rồi `--resume-from` cho ra loss ở bước kế tiếp giống với khi train liền một mạch (cùng GPU).
4. Model đã merge chạy được bằng `lifemate.cli`, và điểm `val_unseen` không thấp hơn model gốc dùng cùng prompt.
5. `llm_pipeline` chạy từ đầu đến cuối trên máy thuê bằng một lệnh.

### 6.3. Docker (D1 – D3)

- BTC mount thư mục test vào `/data`, với cùng cấu trúc như `public_test`. Nhưng `ToolEnv` cần dạng `<root>/<split>/...`. Vì vậy entrypoint phải:
  1. Tìm `/data/*_tools_catalog.json` để biết tên split.
  2. Tạo thư mục `/tmp/zaic_root/<split>` là symlink trỏ tới `/data`.
  3. Chạy `python -m lifemate.cli --config configs/docker.toml --env docker --split <split> --out result/` (`LIFEMATE_CONFIG` đổi được config). `RunStore` ghi `submission.jsonl` cùng `trace.jsonl`, `summary.json`, `config.json`, `check.txt` vào `result/`; BTC chỉ lấy `submission.jsonl`.
  4. `zaic.py` so sánh đường dẫn sau `Path.resolve()` nên symlink không làm hỏng kiểm tra (B5).
- Trọng số model nằm sẵn trong image. Đặt `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`.
- **Nghiệm thu:**
  - `docker run --gpus all --network none -v <public_test>:/data ...` ra `result/submission.jsonl` đủ 634 dòng.
  - `python -m zaic_tools --validate` không báo lỗi.
  - Chạy hai lần trên cùng GPU sm80+ với `batch_invariant = true` cho ra hai file giống hệt.
  - Không mẫu nào vượt 30 giây.

### 6.4. Fine-tune ASR (C3)

**Chỉ làm khi số đo cho thấy cần.** Sau khi có ASR gốc (mục 6.1), so điểm agent khi dùng `GoldTranscriber` và khi dùng ASR trên `val_seen` và `val_unseen`. Fine-tune ASR khi **một trong hai** điều sau đúng:

- Điểm mất do ASR ≥ 3 điểm trên một trong hai bộ val. Ngưỡng 3 điểm này chốt lại sau khi biết độ dao động của điểm giữa các lần chạy.
- Tỉ lệ nghe sai thực thể (bên dưới) ≥ 10%.

#### 6.4.1. Dữ liệu (`lifemate/train/asr_data.py`)

- Cặp `(file wav, audio_text)` gồm lượt cuối và **mọi lượt user trong history**.
- Số cặp theo cách chia hiện tại (đo 27/10/2026): train **3547**, val_seen 624, val_unseen 865.
- Chia theo **id mẫu**, đúng như `splits/`. Mọi file audio của một mẫu thuộc `val_*` không bao giờ được dùng để train.
- **Đích của ASR là `audio_text` nguyên văn**, giữ nguyên cách BTC viết số, tiền, giờ và từ tiếng Anh. Lý do: LLM được fine-tune trên `audio_text` (mục 6.2.1), nên ASR nghe ra giống văn phong đó thì LLM ít bị lệch nhất.
- Dữ liệu ghi ra `data/asr/vN.jsonl` (`{"id","turn","audio","text","split"}`) kèm file meta giống mục 6.2.1. Audio không chép, chỉ ghi đường dẫn tương đối trong split.

#### 6.4.2. Đo lường (`lifemate/eval_asr.py`)

| Chỉ số | Cách tính | Vì sao |
|---|---|---|
| WER, CER | Trên chữ đã chuẩn hóa: chữ thường, NFC, bỏ dấu câu | Chỉ số chung |
| **Tỉ lệ nghe sai thực thể** | Lấy các giá trị tham số chuỗi và số trong `label.tool_calls` mà **có xuất hiện** trong `audio_text`, ví dụ tên người, địa điểm, số tiền. Đếm tỉ lệ không xuất hiện trong chữ ASR, sau khi bỏ dấu và chữ hoa (dùng cùng cách chuẩn hóa `_fold` của `zaic_tools`) | Đây là lỗi làm sai tham số tool, ảnh hưởng điểm trực tiếp |
| Điểm agent mất do ASR | Điểm khi dùng `GoldTranscriber` trừ điểm khi dùng ASR, cùng LLM và prompt | Quyết định có cần fine-tune hay không |
| Thời gian | p50, p95, max thời gian ASR mỗi mẫu | Giới hạn 30 giây |

#### 6.4.3. Script train (`lifemate/train/asr.py`)

```toml
# configs/train/asr_v1.toml (mặc định ban đầu, chốt sau khi chọn model ASR)
[base]
model = "vinai/PhoWhisper-medium"       # cùng model với [asr] của configs/v0.toml
revision = "55a7e3eb6c906de891f8f06a107754427dd3be79"
[data]
train = "data/asr/v1.jsonl"
[method]
kind = "full"               # "lora" | "full"; 0,77B full fine-tune vừa T4 (approach 5.5.2)
freeze_encoder = true       # giai đoạn 1 chỉ train decoder; giai đoạn 2 mở encoder nếu WER còn cao
spec_augment = true
[train]
lr = 1e-5
epochs = 3
batch_size = 8
grad_accum = 2
seed = 0
eval_every_steps = 200
save_every_steps = 200
```

- Dùng chung khung với fine-tune LLM (mục 6.5): seed, checkpoint, train tiếp, ghi metric.
- Chọn checkpoint theo **tỉ lệ nghe sai thực thể** trên `val_unseen`, sau đó WER. Checkpoint tốt nhất được chạy agent thật để đo điểm.
- Merge (nếu dùng LoRA) và nạp vào `ASRTranscriber` qua `[asr].model` trong config.

#### 6.4.4. Dữ liệu tổng hợp bằng TTS (tùy chọn)

- Luật cho phép dùng TTS lúc phát triển, cấm dùng lúc chạy thi.
- Chỉ sinh cho loại lỗi mà bảng đo 6.4.2 chỉ ra, ví dụ tên người, địa danh, số tiền đọc bằng chữ, từ tiếng Anh xen kẽ.
- Câu tổng hợp lấy từ `audio_text` của `splits/train.txt` rồi thay thực thể. Không dùng câu của val hay public.
- Ghi `source = "tts"` và phiên bản riêng. Không bao giờ dùng để đánh giá.

#### 6.4.5. Nghiệm thu

1. Test: không file audio nào của id `val_*` có trong dữ liệu train; số cặp khớp với số đo ở 6.4.1.
2. Model ASR fine-tune có tỉ lệ nghe sai thực thể và WER trên `val_unseen` thấp hơn model gốc.
3. Điểm agent dùng ASR mới không thấp hơn khi dùng ASR gốc, trên cả hai bộ val.
4. p95 thời gian ASR mỗi mẫu không tăng quá 10% so với model gốc.
5. Chạy hai lần trên 30 mẫu ra chữ giống hệt nhau.
6. `asr_pipeline` (dựng dữ liệu → train → chọn checkpoint → merge → đo) chạy từ đầu đến cuối bằng một lệnh.

### 6.5. Khung train dùng chung (`lifemate/train/common.py`)

| Thành phần | Việc |
|---|---|
| `seed_everything(seed)` | Cố định seed cho Python, NumPy, PyTorch, CUDA; bật chế độ tất định của PyTorch khi có thể |
| `CheckpointManager` | Lưu và nạp checkpoint (trọng số hoặc adapter, optimizer, scheduler, RNG, bước); giữ N checkpoint gần nhất cộng N tốt nhất theo chỉ số đã chọn |
| `MetricsLogger` | Ghi `metrics.jsonl` (bước, loss, chỉ số eval, thời gian); flush sau mỗi dòng |
| `write_train_config()` | Ghi `train_config.json` gồm config, `EnvInfo` (mục 4.16), sha256 dữ liệu |
| Đường dẫn | Lấy từ profile môi trường (mục 4.16); checkpoint ghi vào nơi được giữ lại sau phiên |

Cả `sft.py` và `asr.py` dùng các thành phần này, không tự viết lại. Thêm `tests/train/` để kiểm tra `CheckpointManager` (lưu rồi nạp ra đúng bước và trạng thái RNG) và `seed_everything` (hai lần chạy cùng seed ra cùng số ngẫu nhiên).

---

## 7. Tiêu chí nghiệm thu v0

1. `uv run pytest` chạy hết 26 test ở mục 5 và đều đạt (hoặc skip có lý do theo đúng quy định ở mục 5), trên máy không có GPU. `ruff check`, `ruff format --check` và `mypy --strict lifemate/` không báo lỗi.
2. T1 (FakeLLM) cho đúng **100,00** trên `val_seen` và trên `val_unseen`.
3. Chạy v0 với vLLM trên `val_seen` (193 mẫu) và `val_unseen` (254 mẫu):
   - `submission.jsonl` có đúng số dòng bằng số id, không trùng id.
   - `summary.json` có `n_exception = 0`.
4. **0 call bị từ chối** trong mọi `submission.jsonl`. Kiểm bằng cách phát lại các call qua `zaic_tools` và đếm kết quả `ok: false`.
5. 0 mẫu có `stop_reason = "parse_error"` do JSON hỏng khi chạy với vLLM có ép schema. Lỗi `parse_error` chỉ được phép xuất hiện với FakeLLM trong test.
6. `tokens.max_prompt` < 15.872 (16.384 trừ 512) trên cả hai bộ val.
7. Chạy `--limit 30` hai lần liên tiếp trên cùng máy cho ra hai `submission.jsonl` giống hệt từng byte.
8. Điểm `val_seen` và `val_unseen` cao hơn điểm của bài nộp toàn `[]` trên cùng bộ. Điểm toàn `[]` lấy bằng `score.py` với file chỉ chứa `[]`.
9. `summary.json` báo p50, p95, max thời gian và GPU đã dùng. Trên T4 **chưa đặt ngưỡng đạt**, chỉ ghi lại số. Thời gian nạp model và warm-up được ghi riêng trong `config.json`, không cộng vào mẫu đầu tiên.
10. Có một dòng trong bảng nhật ký `approach.md` mục 8 cho mỗi bộ val.
11. `eval/score.py` đã sửa C1–C3 (mục 4.15); `score.txt` in điểm của cả hai chế độ `strict` và `canon`.
12. `notebooks/kaggle_run.ipynb` chạy từ đầu đến cuối trên một phiên Kaggle mới, với code đưa lên bằng `scripts/pack_code.sh`. `config.json` có `code_version` khác `"unknown"`, có `dtype` thật đã dùng, và `runs/` được lưu lại theo `persist`.
13. `summary.json` có `prompt_shrink_level` với mức 3 bằng 0 và `context_overflow` bằng 0 trên cả hai bộ val.

---

## 8. Thứ tự làm và công sức

```
#0 types + jsonutil + config ─► #1 pyproject + data + zaic ──┬─► #2 tooldocs + prompt + protocol + tokens ──┐
                                                              └─► #3 FakeLLM + FakeEnv                       ├─► #5 agent + budget + guards ─► #6 runner + store + observers + cli ─► #7 test T1–T18, T24, T25
                                                                  #4 VLLMBackend + warm-up ──────────────────┘                                                                    │
#10 sửa score.py (C1–C3) + T17, làm bất cứ lúc nào
#11 môi trường chạy (4.16): envinfo + profile làm cùng #0; notebook + pack_code làm trước #8                                                                                                                                 ▼
                                                                                                                                         #8 chạy v0 trên Kaggle ─► #9 ghi nhật ký
```

- #0 làm đầu tiên vì mọi module dùng chung các Protocol, Enum và `canonical_json`.
- #3 (FakeLLM) làm trước #4 để kiểm toàn bộ vòng lặp bằng T1 khi chưa có GPU. Lỗi ở vòng lặp và bộ ghi được bắt sớm, không lẫn với lỗi của model.
- #4 làm song song với #2 và #5.

| Việc | Ước lượng |
|---|---|
| #0 types, jsonutil, config, ruff/mypy/pre-commit | 1,5 giờ |
| #1 khung project, data, zaic | 1 giờ |
| #2 tooldocs (có cache), prompt + Conversation + thang giảm (B1), protocol, tokens, system prompt | 5 giờ |
| #3 FakeLLM, FakeEnv | 1 giờ |
| #4 VLLMBackend, warm-up, chạy thử trên Kaggle | 2,5 giờ |
| #5 agent, budget, guard chain | 3 giờ |
| #6 runner, store (flush, ghi nguyên tử, resume), check_submission (B3), observers, cli, factory | 4 giờ |
| #7 test T1–T18, T24, T25 | 4,5 giờ |
| #8 chạy hai bộ val trên T4 | 1 – 2 giờ máy |
| #10 sửa score.py, `make_splits.py --smoke`, T17, T26 | 2 giờ |
| #11 profile môi trường, envinfo, requirements/notebook.txt, pack_code.sh, 2 notebook, T19–T23 | 4 giờ |
| **Tổng** | **khoảng 28,5 giờ làm, 2 giờ máy** |

---

## 9. Rollback

- Mọi thay đổi nằm trong file mới (`lifemate/`, `prompts/`, `configs/`, `requirements/`, `scripts/`, `notebooks/`, `tests/`, `pyproject.toml`). Bỏ v0 chỉ cần xóa các file đó hoặc revert commit.
- `splits/train.txt`, `val_seen.txt`, `val_unseen.txt` **không đổi nội dung** trong v0; chỉ thêm `smoke.txt`. `eval/score.py` và `eval/make_splits.py` chỉ sửa theo mục 4.15 và để thêm `--smoke`; chế độ mặc định `strict` của `score.py` giữ nguyên cách chấm cũ trừ trường hợp `bool` và số (C1). Muốn quay lại thì revert commit đó.
- `runs/` không vào git. Xóa một lần chạy hỏng là xóa thư mục của nó và dòng tương ứng trong `runs/index.csv`.

## 10. Ngoài phạm vi v0

- ASR thật, normalizer, fine-tune và Docker (mục 6).
- Ép schema chi tiết theo từng tool, few-shot trong prompt, lọc bớt tool khi prompt quá dài.
- Chạy nhiều mẫu song song.
- `compare.py`, `show.py` (`approach.md` mục 4.8, bước 3).
- Kiểm tra lại các nhãn đáng ngờ, ví dụ `train_1606`.

## 11. Câu hỏi còn mở

| # | Câu hỏi | Ảnh hưởng | Cách giải quyết |
|---|---|---|---|
| Q1 | Hướng dẫn Docker chi tiết của BTC: lệnh chạy, đường dẫn kết quả tuyệt đối hay tương đối | Mục 6.3 | Chờ tài liệu riêng của BTC. Entrypoint viết sao cho đổi đường dẫn dễ |
| Q2 | ~~Chọn LLM và ASR~~ Đã chọn 27/10/2026 (mục 6.0, `approach.md` 5.5). Còn lại: BTC đếm tham số Qwen3.5-4B theo tổng checkpoint hay chỉ phần ngôn ngữ; giấy phép phi thương mại của ChunkFormer / VieGem có được chấp nhận không | A/B sau v0 | Hỏi BTC; trong lúc chờ chỉ nộp bằng bộ mặc định |
| Q3 | vLLM phiên bản được khóa có chạy được trên cả T4 (dev) và sm80+ (chấm) không; các tên API mà `VLLMBackend` dùng (`LLM(...)`, `SamplingParams(structured_outputs=StructuredOutputsParams(json=...))` với fallback `guided_decoding=GuidedDecodingParams`, `LLM.chat(..., chat_template_kwargs=)`, `RequestOutput.prompt_token_ids`) mới chỉ được test với engine giả | #4, Docker | Chạy thử `--limit 30` trên cả hai loại GPU ngay khi chọn model; sửa `_structured_kwargs` nếu API khác |
| Q4 | Tổng token của prompt với mẫu nhiều tool nhất (59 tool ở public) | Có cần lọc tool không | Đo bằng `HFTokenCounter` trên mọi mẫu train và public trước khi chạy v0 |
