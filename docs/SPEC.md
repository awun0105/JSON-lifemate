# SPEC: hệ thống agent LifeMate

Tài liệu này mô tả **cần xây cái gì** để người viết code (người hoặc AI) làm được mà không phải hỏi lại. SPEC chỉ mô tả **trạng thái hiện tại**. Đề bài xem `overview.md`, dữ liệu xem `eda.md`, lịch sử quyết định xem `decisions.md`, số đo từng lần chạy xem `experiments.md`, lập luận từ nguyên lý gốc cho fine-tune xem `rationale.md`. Pipeline, cách đánh giá, hạ tầng đo, nộp bài, danh sách model và mốc thời gian nằm ở các phụ lục A–F cuối tài liệu.

- **Phạm vi:** toàn hệ thống. Riêng **agent v0 và hạ tầng đo** (mục 4) viết chi tiết đến mức interface, file và test. ASR, fine-tune và Docker (mục 6) viết ở mức interface và tiêu chí nghiệm thu, sẽ viết chi tiết khi có số đo của v0.
- **Kiểm chứng trạng thái:** 28/10/2026, trên gói dữ liệu `data/train+public_test/` và `zaic_tools` của train.

---

## 1. Bối cảnh

### 1.1. Mục tiêu

Cho một yêu cầu bằng giọng nói tiếng Việt, agent phải gọi đúng chuỗi tool trên các app giả lập (`zaic_tools`). Kết quả là một dòng `submission.jsonl` gồm `id`, `tool_calls` và `state_hash`. Thứ hạng chỉ tính theo `private_test`, bộ này nhiều khả năng có app chưa từng thấy.

### 1.2. Thế nào là xong

| Mốc | Xong khi |
|---|---|
| v0 (30/10) | Chạy hết `val_seen` và `val_unseen`, có điểm từ `lifemate/eval/score.py` và các tiêu chí ở mục 7 đều đạt |
| Docker (05/11) | Image chạy với `--network none`, ra được `result/submission.jsonl` cho `public_test` |
| Nộp (16/11) | Gói Docker cuối, training code và data chạy lại được |

### 1.3. Trạng thái hiện tại (đã kiểm chứng)

| Thành phần | Trạng thái | Bằng chứng |
|---|---|---|
| Đề bài, EDA, hướng làm | Có | `docs/overview.md`, `docs/eda.md`, phụ lục A–F của SPEC; lịch sử ở `docs/decisions.md`, `docs/experiments.md`, `docs/rationale.md` |
| Chia dữ liệu | **Đổi 28/10/2026** theo team lead | `lifemate/eval/make_splits.py`: 6 app giữ lại tách trước, 5 fold theo cụm câu gần trùng; train 1093, val_seen 274, val_unseen 178, kiểm tra rò rỉ bằng 0. `configs/splits/` không có trong git, sinh bằng `make splits` |
| Bộ chấm local | Có | `lifemate/eval/score.py`: gold được 100, toàn `[]` được 7,18 |
| Agent v0 (`lifemate/`), CLI, hạ tầng đo | **Có, 27/10/2026**; đã chạy thật trên Kaggle T4 với hạn 29 s | 218 test pass (28/10); FakeLLM qua CLI được 100,00 (strict và canon) trên `val_seen`, `val_unseen`; `configs/fake.toml` chạy không cần model |
| Chọn model | **Đã chọn** (mục 6.0, phụ lục E.5): Qwen3-4B-Instruct-2507; ASR whisper-large-v3 bản CTranslate2, không câu mồi (lịch sử ở `decisions.md` D17, D25, D30) | Qwen3, PhoWhisper và large-v3 đã chạy trên Kaggle T4 (`experiments.md`) |
| vLLM thật | `VLLMBackend` viết theo API 0.32, **chỉ test với engine giả** (Q3) | |
| ASR | `ASRTranscriber` + 4 engine (`faster_whisper` mặc định, `hf_seq2seq`, `chunkformer`, `nemo_ctc`), bộ lọc ảo giác `asr_sanitizer.py`, lớp bảo vệ câu mồi `asr_guard.py`, `eval_asr.py` (thêm chỉ số câu chêm tiếng Anh, `en_mixed.py`), chọn cấu hình ASR `asr_tune.py`, chấm lại offline `rescore_asr.py` | large-v3 không câu mồi: WER 29,7%, bỏ sót thực thể 31% trên `val_seen` (ASR sweep v1, `experiments.md`); chưa có baseline agent hợp lệ với ASR thật |
| Docker | Dockerfile, entrypoint, `configs/docker.toml`, smoke script; **chưa build** | `docs/docker.md` |
| Fine-tune | LLM train bằng ms-swift 4.6.0 (`swift_sft.py`, 28/10); dữ liệu trộn 70% chữ ASR, 3% mẫu phục hồi lỗi; template ms-swift đã so khớp với chat template trên 60 mẫu thật; `sft_data` và `asr_data` đã chạy trên dữ liệu thật; **train/merge chưa chạy** | Phụ lục E.5.2 ghi kế hoạch và thứ tự |
| Môi trường | Profile local/kaggle/colab/rented/runpod/docker, notebook run và train, `ops/scripts/runpod_setup.sh` | Chỉ profile local đã chạy |
| Báo cáo | `lifemate/run/report.py` (`make report`, `make report-train`) | Test với dữ liệu giả |

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
      [A9] submission.jsonl    [A10] trace.jsonl, summary.json   ──► lifemate/eval/score.py
```

| Khối | Thành phần | Mốc |
|---|---|---|
| A. Chạy khi chấm | A1 data, A2 transcriber, A3 normalizer, A4 prompt, A5 LLM, A6 loop, A7 guards, A8 `zaic_tools` (BTC), A9 writer, A10 tracer | v0: A1, A2 (gold), A4–A10. Sau v0: A2 (ASR). A3 chỉ khi cần, xem 6.1 |
| B. Đánh giá | B1 splits (có), B3 scorer (có), B4 run manager, B5 đo ASR, B6 test | v0: B4, B6. Sau v0: B5 |
| C. Huấn luyện | C1 dữ liệu SFT, C2 QLoRA LLM, C3 fine-tune ASR, C4 versioning | Sau v0 |
| D. Nộp bài | D1 Dockerfile, D2 kiểm tra trước khi nộp, D3 gói nộp | Sau v0 |

---

## 3. Quyết định đã chốt cho SPEC này

Các quyết định thiết kế còn hiệu lực. Ngày chốt, lý do và quyết định bị thay ghi ở `decisions.md`.

| # | Quyết định | Lý do |
|---|---|---|
| S1 | Python 3.11, quản lý bằng `uv` (`pyproject.toml` + `uv.lock`), test bằng pytest | Khóa phiên bản giống nhau giữa Kaggle và Docker |
| S2 | LLM chạy bằng vLLM **trong cùng process** (`vllm.LLM`), gọi qua interface `LLMBackend` | Một process, dễ đóng gói. Có `FakeLLM` để test không cần GPU |
| S3 | Tool call theo **định dạng JSON tự định nghĩa**, vLLM ép đầu ra theo JSON schema (structured outputs). Mô tả tool dạng rút gọn trong prompt | Tiết kiệm token, không bao giờ hỏng JSON, không phụ thuộc chat template của từng model |
| S4 | Mỗi lượt LLM sinh **đúng một** hành động: gọi một tool hoặc báo xong | Đơn giản, dễ ép schema, khớp với dữ liệu fine-tune |
| S5 | Chạy tuần tự từng mẫu, batch = 1 | Tất định. Luật chỉ giới hạn thời gian mỗi mẫu, không giới hạn tổng |
| S6 | Tắt chế độ suy nghĩ của model nếu model có chế độ này (ví dụ `enable_thinking=False` qua `chat_template_kwargs`) | Tốn token và thời gian, khó giữ dưới 30 giây |
| S7 | Code không gắn với model cụ thể nào: tên model, revision, dtype chỉ nằm trong config. Model phải đạt điều kiện ở mục 6.0; model đang dùng ở phụ lục E.5 | Đổi model mà không phải sửa code |
| S8 | Config bằng TOML (`tomllib` có sẵn trong Python 3.11) | Không thêm thư viện |
| S9 | Mọi thành phần thay được đều là `typing.Protocol`, lắp ghép tại **một chỗ duy nhất** (`lifemate/run/factory.py`), dùng composition thay cho kế thừa | Đổi model, ASR, guard bằng config. Test thay bằng bản giả mà không sửa code chính (mục 4.0) |
| S10 | Prompt sắp từ phần ít đổi đến phần hay đổi. Messages chỉ được thêm vào cuối, không bao giờ sửa | Tận dụng prefix cache của vLLM giữa các mẫu và giữa các bước trong một mẫu (mục 4.6) |
| S11 | Nạp model xong phải **warm-up** trước khi bắt đầu tính giờ mẫu đầu tiên | Lần gọi đầu của vLLM và ASR chậm vì biên dịch kernel, có thể làm mẫu đầu vượt 30 giây |
| S12 | Hạn giờ **thật**, không dự đoán (đổi 27/10/2026): bước nào cũng được bắt đầu khi còn giờ, tới `deadline_s` thì bước LLM đang chạy bị ngắt; call đã gọi vẫn nằm trong bài nộp | Bản cũ dự đoán thời gian bước tới bằng p95 mọi bước; bước đầu chậm (đọc cả prompt) kéo dự đoán lên khoảng 14 s nên trên Kaggle T4 18/30 mẫu bị dừng oan ở giây 13–16 (mục 4.10) |
| S13 | Dùng lại từ kit `~/PROJECT/AI20K/ai-kit` (nhánh `ai-kit`) **quy ước kỹ thuật**, không dùng lại các layer ứng dụng | Kit là app FastAPI + LangGraph gọi LLM cloud; RAG, MCP, A2A, reliability, cost guard không áp dụng cho agent offline chạy vLLM trong process. Lấy: cấu hình ruff và typecheck, pre-commit (hygiene, detect-secrets, ruff), cấu trúc Makefile, mẫu test với LLM giả có kịch bản (`tests/fakes.py` → `FakeLLM`), Dockerfile hai tầng với `uv --frozen` và user không phải root, quy ước log `structlog` kiểu `sự_kiện_chữ_thường` với kwargs, và dòng ghi nguồn gốc khi chép code. Mỗi chỗ chép ghi nguồn (`# Adapted from ai-kit <path>@<commit>`) |

---

## 4. Chi tiết agent v0

### 4.1. Cấu trúc repo

Gốc repo chỉ có sáu thư mục nguồn cộng hai thư mục sinh ra; mỗi tên nói đúng một loại nội dung.

```
AGENTS.md  README.md  Makefile  pyproject.toml  uv.lock  .pre-commit-config.yaml
lifemate/                 # code, chia theo vai trò; mỗi gói 2–7 module
├── core/                 # kiểu dữ liệu và hạ tầng chung, không phụ thuộc gói khác
│   ├── types.py          # value object, Enum StopReason/ErrorCode/GuardVerdict, các Protocol (mục 4.0)
│   ├── config.py         # đọc TOML thành dataclass Config và EnvProfile, kiểm tra giá trị (4.14, 4.16)
│   ├── jsonutil.py       # canonical_json(): mọi JSON ảnh hưởng tới tính tất định đi qua đây
│   ├── envinfo.py        # nhận GPU, chọn dtype, lấy phiên bản code và gói (4.16)
│   ├── tokens.py         # TokenCounter dùng tokenizer của model (CPU)
│   ├── textsim.py        # gom câu gần trùng (difflib), dùng cho chia dữ liệu và lọc dữ liệu fine-tune
│   └── paths.py          # REPO_ROOT, định nghĩa một lần
├── agent/                # vòng lặp suy luận, không biết gì về file hay GPU
│   ├── agent.py          # A6: Agent.run(sample, env) -> SampleResult
│   ├── budget.py         # hạn giờ dự đoán, số bước, số lần thử lại
│   ├── guards.py         # A7: các Guard và GuardChain
│   ├── prompt.py         # A4: PromptBuilder, Conversation, thang giảm prompt
│   ├── tooldocs.py       # catalog → mô tả tool rút gọn, có cache
│   ├── protocol.py       # JSON schema đầu ra + parse
│   └── observers.py      # A10: AgentObserver, Tracer, NullObserver
├── backends/             # model thật và giả, sau Protocol LLMBackend / Transcriber
│   ├── llm.py            # A5: VLLMBackend, FakeLLM
│   ├── asr.py            # A2: GoldTranscriber, ASRTranscriber, ASR_ENGINES (faster_whisper, hf_seq2seq, chunkformer, nemo_ctc)
│   ├── asr_sanitizer.py  # lọc câu ảo giác và vòng lặp của Whisper, kiểm tra giữ thực thể (theo team lead)
│   └── asr_guard.py      # lớp bảo vệ câu mồi: phát hiện câu hỏng do initial_prompt để chép lại không câu mồi
├── io/                   # đọc dữ liệu thi, ghi kết quả
│   ├── data.py           # A1: đọc <split>.jsonl thành Sample
│   ├── zaic.py           # nạp zaic_tools đúng split, tạo ToolEnv (một lần mỗi mẫu)
│   ├── store.py          # RunStore: thư mục run (flush, ghi nguyên tử, resume), index.csv
│   └── check_submission.py
├── run/                  # ghép mọi thứ lại và chạy
│   ├── runner.py         # warm-up, lặp qua mẫu, cô lập lỗi từng mẫu, gọi chấm
│   ├── factory.py        # nơi duy nhất tạo đối tượng từ Config (registry theo tên backend)
│   ├── timing.py         # decorator Timed cho LLMBackend và Transcriber
│   ├── cli.py            # parse tham số rồi gọi Runner: `python -m lifemate` hoặc `python -m lifemate.run.cli`
│   └── report.py         # bảng text từ index.csv và metrics.jsonl, PNG nếu có matplotlib (`make report`)
├── eval/                 # chấm điểm và chia dữ liệu (script, ngoài mypy strict)
│   ├── score.py          # `python -m lifemate.eval.score` (mục 4.15)
│   ├── make_splits.py    # sinh configs/splits/ (mục 3)
│   ├── audit.py          # rà từng lời gọi của một lần chạy so với nhãn (theo team lead)
│   ├── eval_asr.py       # WER, CER, tỉ lệ nghe sai thực thể, chỉ số câu chêm tiếng Anh (6.4.2)
│   ├── foreign_words.py  # tìm từ ngoại (tiếng Anh) chêm trong câu tiếng Việt, xếp nhóm
│   ├── asr_tune.py       # chọn mô hình và cấu hình ASR (D36, D40)
│   ├── rescore_asr.py    # chấm lại hyps.jsonl đã có trên một tập con, không chạy model
│   ├── asr_compare.py    # bảng so sánh nhiều hệ thống ASR từ hyps.jsonl, khoảng tin cậy bootstrap
│   ├── foreign_words_audit.py  # đo quy tắc tìm từ ngoại bằng nhãn rà tay
│   └── asr_merge.py      # ghép chữ RNNT với từ tiếng Anh của Whisper, lọc theo từ điển ngữ cảnh (thử offline)
└── train/                # fine-tune: common, sft_data, sft, swift_sft (ms-swift), llm_pipeline, asr_cache, asr_data, asr, asr_pipeline (6.2, 6.4, 6.5)
tests/                    # pytest, CPU; tests/train/ cho phần fine-tune
configs/                  # mọi đầu vào cố định, có phiên bản
├── v0.toml  fake.toml  docker.toml
├── asr_prompts/          # câu mồi ASR có phiên bản (asr.initial_prompt_file)
├── env/                  # local, kaggle, colab, rented, runpod, docker (4.16)
├── train/                # llm_v1.toml, asr_v1.toml
├── prompts/              # system_v0.md, system_v1.md (đang dùng)
└── splits/               # KHÔNG vào git: train, val_seen, val_unseen, smoke, asr_tune, *_en_mixed (.txt), folds/, meta.json; `make splits`
docs/                     # SPEC, decisions, experiments, rationale, overview, eda, CONTRIBUTING, docker, runpod, diagrams/ (png + src); slides/ không vào git
ops/                      # vận hành: không có logic, chỉ gọi lifemate
├── docker/               # Dockerfile, Dockerfile.dockerignore, entrypoint.sh, smoke.sh (6.3)
├── notebooks/            # {kaggle,colab}_run.ipynb, {kaggle,colab}_train_{llm,asr}.ipynb
├── requirements/         # notebook.txt (sinh từ uv.lock), train.txt
└── scripts/              # pack_code.sh, runpod_setup.sh, export_notebook_requirements.py
data/                     # KHÔNG vào git: train+public_test/ (gói của BTC), sft/, asr/ (dữ liệu dựng ra)
artifacts/                # KHÔNG vào git, trừ models/registry.csv và models/README.md
├── runs/                 # mỗi lần chạy một thư mục + index.csv; runs/train/<tên> cho fine-tune
├── models/               # llm/, asr/ snapshot và model đã merge; Docker chép thư mục này
└── dist/                 # zip code cho Kaggle/Colab
```

Quy tắc phụ thuộc giữa các gói: `core` không import gói nào; `agent` chỉ import `core`; `backends`, `io` import `core`; `run` import tất cả; `eval` và `train` import `core`, `io`, `run`. Không có import ngược.

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

**Các interface (`lifemate/core/types.py`):**

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

### 4.2. Kiểu dữ liệu (`lifemate/io/data.py`)

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

### 4.3. Nạp `zaic_tools` (`lifemate/io/zaic.py`)

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

### 4.4. Transcriber (`lifemate/backends/asr.py`)

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

### 4.5. Mô tả tool rút gọn (`lifemate/agent/tooldocs.py`)

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

**Quy tắc app bị rút (chế độ `train_mode`):** chỉ áp dụng khi dựng prompt cho mẫu thuộc `configs/splits/train.txt` (few-shot, dữ liệu fine-tune). Khi đó:

1. Xóa 6 app trong `configs/splits/meta.json` → `held_out_apps` khỏi `installed_apps` trước khi gọi `render_tools`. Thiếu `meta.json` thì `read_held_out` báo lỗi, không coi là danh sách rỗng.
2. **Bỏ hẳn mẫu** nếu `history` có lượt assistant gọi tool của app bị rút (A4), ví dụ `train_0655`. Nếu chỉ xóa khỏi tool docs thì model vẫn thấy `[đã gọi] medication.xxx(...)` trong lịch sử. `PromptBuilder.start(..., train_mode=True)` raise `HeldOutLeak`, và `sft_data.py` ghi id vào `data/sft/vN.dropped.txt` với lý do.

Khi chạy đánh giá trên `val_seen` và `val_unseen` thì **không** áp dụng chế độ này.

### 4.6. Dựng prompt (`lifemate/agent/prompt.py`)

> **Từ 30/10/2026 (D50), mặc định là định dạng ChatML** (`[agent].format = "chatml"`): tin nhắn dựng bằng
> `lifemate/agent/chatml.py`, giống hệt `build_messages` của repo lead (danh sách tool ở trường `tools` của tin `system`,
> lời gọi `tool_calls`, kết quả ở vai `tool`, kết thúc bằng câu trả lời không có `<tool_call>`); vòng lặp
> `lifemate/agent/chatml_agent.py`. Mục 4.6 và 4.7 dưới đây mô tả định dạng JSON cũ, vẫn dùng được với
> `format = "json"` (`configs/v0_json.toml`). Chi tiết: `docs/code_guide.md` mục 3.

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
| 1. system, phần đầu | Quy tắc và định dạng đầu ra trong `configs/prompts/system_v0.md` | **Mọi mẫu** |
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

`configs/prompts/system_v0.md` phải nêu tối thiểu các quy tắc sau:

- Mỗi lượt chỉ trả về một JSON.
- Tra cứu lấy ID bằng tool loại `read` trước khi gọi tool loại `write`. Không tự bịa ID.
- Thiếu tham số bắt buộc thì gọi `sys.ask_missing_info` với `tool_name` và `params`, rồi báo xong.
- Yêu cầu bị hủy giữa câu, có điều kiện, hoặc không app nào làm được thì báo xong mà không gọi tool. Đây là quy ước của nhãn: câu "nếu mai mưa thì…" có gold `[]` (xem `docs/eda.md`).
- Ngày giờ tính theo `current_time` trong context.
- Số tiền và số đọc bằng chữ đổi thành số.

Tất cả JSON trong prompt đi qua `canonical_json()` (mục 4.0.1) để giữ tất định.

### 4.7. Định dạng đầu ra (`lifemate/agent/protocol.py`)

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

### 4.8. LLM backend (`lifemate/backends/llm.py`)

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

### 4.9. Guards (`lifemate/agent/guards.py`)

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

### 4.10. Vòng lặp agent (`lifemate/agent/agent.py`, `lifemate/agent/budget.py`)

```python
class Agent:
    def __init__(self, llm: LLMBackend, transcriber: Transcriber, prompts: PromptBuilder,
                 guards: GuardChain, index: CatalogIndex, cfg: AgentConfig, llm_cfg: LLMConfig,
                 observer: AgentObserver | None = None): ...   # None → NullObserver
    def run(self, sample: Sample, env: ToolEnvLike, *, train_mode: bool = False,
            clock: Clock = time.monotonic) -> SampleResult
```

`clock` được truyền vào để test hạn giờ mà không phải chờ thật (T10).

```
budget = Budget(cfg, llm_cfg, clock)          # bắt đầu tính giờ từ đây, gồm cả ASR
transcript = transcriber.transcribe(sample)
conv = prompts.start(sample, transcript)
schema = output_schema(allowed_tools)          # có cache
observer.on_sample_start(sample, transcript)
loop:
    reason = budget.before_step(conv.prompt_tokens())   # None nghĩa là được chạy tiếp
    nếu reason: stop(reason)                   # MAX_STEPS | TIMEOUT | CONTEXT_OVERFLOW
    gen = llm.generate(conv.messages, schema, sample_id=sample.id, time_left_s=budget.remaining)
    nếu gen.timed_out: stop(TIMEOUT)            # bước bị ngắt ở deadline_s, call cũ vẫn nộp
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

**`Budget` (hạn giờ thật, quyết định S12):**

```python
class Budget:   # Budget(cfg: AgentConfig, llm_cfg: LLMConfig, clock)
    remaining: float                               # cfg.deadline_s - elapsed, không âm
    def before_step(self, prompt_tokens: int) -> StopReason | None
        # MAX_STEPS        khi đã chạy cfg.max_steps bước
        # CONTEXT_OVERFLOW khi prompt_tokens + llm_cfg.max_tokens > llm_cfg.max_model_len
        # TIMEOUT          khi remaining <= 0
    def after_step(self, seconds: float) -> None
    def retry(self) -> bool                        # False khi đã dùng hết cfg.max_retries
```

- Không dự đoán thời gian bước tới. Agent gọi `llm.generate(..., time_left_s=budget.remaining)`. `VLLMBackend` khi có `time_left_s` thì đưa request vào engine (`llm_engine.add_request`) và chạy từng `step()`; tới hạn thì `abort_request` và trả `Generation(timed_out=True, text="")`. Agent ghi bước đó (`action=None`) rồi dừng với `TIMEOUT`; các call đã gọi vào env vẫn có trong `get_tool_calls()` nên vẫn được nộp và chấm.
- Engine thiếu API `add_request`/`step`/`abort_request` thì `VLLMBackend` ghi log lỗi `vllm_abortable_generate_unsupported` và quay về `LLM.chat` (không ngắt được). Phải thấy log này bằng 0 trên máy chấm cùng loại trước khi nộp. Chỉ quay về khi đường step/abort hỏng ngay lần đầu (lúc warmup); đã chạy được rồi mà sau đó lỗi thì lỗi được ném ra và mẫu thành `exception`. Request đã thêm vào engine luôn bị huỷ khi có lỗi giữa chừng.
- `deadline_s = 29`: luật BTC là 30 giây mỗi mẫu, quá giờ thì mẫu 0 điểm. 1 giây còn lại cho việc sau khi ngắt (lấy call và hash từ env, ghi trace, thường dưới 0,1 s) và cho chênh lệch giữa đồng hồ của agent với đồng hồ BTC (BTC có thể tính cả lúc mở env). Chỉnh sau khi đo trên A30.
- Thời gian ASR được tính vào `elapsed` vì `Budget` bắt đầu tính trước khi gọi ASR.
- Warm-up gọi cùng đường engine có ngắt (giới hạn 300 s) để biên dịch đúng đường mà mẫu thật chạy.

**Quy tắc khác:**

- `Agent.run` không bắt exception. Exception được bắt ở `Runner` (mục 4.11). Mẫu đó vẫn ghi dòng nộp với `env.get_tool_calls()` và `env.get_state_hash()` đang có, cùng mã `exception`. **Không bao giờ dừng cả lần chạy** vì một mẫu lỗi.
- `max_steps = 10` lấy từ: gold tối đa 6 call, cộng một lượt `done`, cộng tối đa 2 lần thử lại, cộng dư 1.
- Ngưỡng cắt 4000 ký tự lớn hơn kết quả dài nhất đã thấy (2439). Bị cắt thì cuối chuỗi thêm `…(đã cắt)`. Cắt **trước** khi thêm vào `Conversation` (quyết định S10).

### 4.11. Runner, RunStore và CLI (`runner.py`, `store.py`, `cli.py`)

```
# Máy local: data_root và runs_dir lấy từ configs/env/local.toml
uv run python -m lifemate.run.cli --config configs/v0.toml --env local \
    --split train --ids configs/splits/val_seen.txt --name 2026-10-28_v0-<llm>_val-seen [--resume] [--limit N]

# Ghi đè profile khi cần
uv run python -m lifemate.run.cli --config configs/v0.toml --env kaggle --root /kaggle/input/other/train+public_test ...
```

`--name` là tên thư mục con trong `runs_dir` của profile; `--out` là **chính thư mục run** (ghi đè `runs_dir/name`, khớp với `--out result/` ở Docker). Khi có `--out`, `index.csv` nằm ở thư mục cha của thư mục đó. Có thêm profile `local.toml` cho máy phát triển (đường dẫn tương đối trong repo).

`cli.py` chỉ parse tham số, đọc `Config`, cố định seed của Python, NumPy, PyTorch bằng `[llm].seed` (`lifemate/core/seeding.py` › `seed_everything(seed, deterministic=False)`, ghi vào `config.json` khoá `random_seeds`; thể lệ mục 8), gọi `factory.build(config)` rồi `Runner.run(...)`. Container đặt thêm `PYTHONHASHSEED=0` trong `ops/docker/Dockerfile`, vì biến này chỉ có tác dụng khi đặt trước lúc Python khởi động (D41).

**`Runner.run(samples, store)`:**

1. Nạp model, gọi `llm.warmup()` và `transcriber.warmup()` (quyết định S11). Ghi thời gian nạp và warm-up vào `config.json`.
2. Với từng mẫu theo thứ tự id: `open_env` → `agent.run(sample, env)` trong `try/except Exception`. Đây là **chỗ duy nhất** bắt mọi exception. Lỗi thì tạo `SampleResult` với mã `exception` và traceback. **Nếu lỗi xảy ra trước khi có `env`** (B4), ghi `tool_calls=[]`, `state_hash=""`; dòng nộp vẫn phải có để file đủ id.
3. `store.append(result)` ngay sau mỗi mẫu.
5. Chạy xong: `store.finalize()`, rồi gọi `lifemate/eval/score.py` **bằng subprocess** nếu split có nhãn (import trong process sẽ nạp `zaic_tools` lần hai).

**`RunStore`:**

- `trace.jsonl`: mở ở chế độ append, `flush()` và `os.fsync()` sau mỗi mẫu. Phiên Kaggle bị ngắt thì chỉ mất tối đa mẫu đang chạy.
- `submission.jsonl` và `summary.json`: ghi ra file `*.tmp` rồi `os.replace` (ghi nguyên tử). Không bao giờ để lại file nộp ghi dở.
- `submission.jsonl` được ghi ngay đầu run (trước khi nạp model) và ghi lại sau mỗi mẫu, luôn đủ một dòng cho mỗi id của split: id chưa chạy có dòng giữ chỗ `{"tool_calls": [], "state_hash": ""}`. BTC chấm bằng cách phát lại `tool_calls` và `--validate` không đọc `state_hash`, nên dòng giữ chỗ được điểm như mẫu lỗi (0, hoặc đúng khi đáp án là `[]`); tiến trình chết lúc nào (hết bộ nhớ, bị kill vì quá giờ) thì BTC vẫn có file đủ dòng với các mẫu đã xong (D39).
- `--resume`: đọc `trace.jsonl`, bỏ qua dòng cuối nếu dòng đó hỏng (bị ngắt giữa chừng), bỏ qua các id đã có. `submission.jsonl` được dựng lại từ `trace.jsonl` ở `finalize()`. Config mới phải khớp `config.json` cũ ở `config`, `prompt_sha256`, `split`, `train_mode`, nếu không thì báo lỗi; `dtype` và thông tin máy không được so, để container nộp bài chạy tiếp được trên GPU khác. `config.json` giữ bản ghi gốc; mỗi lần chạy tiếp nối `started_at`, `warmup_s`, `env_info`, `dtype` vào danh sách `resumes`.
- Thư mục `--out` đã có `trace.jsonl` mà không truyền `--resume` thì báo lỗi và thoát, không ghi đè.
- Hai lỗi trên (trùng thư mục, lệch config khi chạy tiếp) được kiểm tra **trước** khi nạp LLM và ASR.
- Mã thoát của CLI: 1 khi `score.py` lỗi hoặc không đọc được điểm (split có nhãn), còn lại là 0. `check_submission` báo lỗi thì CLI in `FAILED (check)` ra stderr nhưng vẫn thoát 0, vì container nộp bài chạy chính CLI này và file nộp đã ghi xong. Mọi exception ở các bước sau `finalize()` (chấm, kiểm tra, ghi `index.csv`) chỉ được ghi log và báo qua `failed`, không làm CLI thoát khác 0.
- `--ids` không truyền thì chạy mọi mẫu của split. `--limit` lấy N id đầu tiên, dùng để chạy thử.
- Split có nhãn (train) thì cuối lần chạy tự gọi `lifemate/eval/score.py submission.jsonl --ids <run>/ids.txt --match both --details details.jsonl` và ghi kết quả ra `score.txt`. `ids.txt` là danh sách id thực sự đã chạy (sau `--limit`), để các id chưa chạy không bị chấm 0. `summary.json.score` = `{total, by_mode: {strict, canon}, by_category}`.
- Thêm một dòng vào `artifacts/runs/index.csv`: `run, date, git_commit, model, prompt_hash, split, ids_file, n, score, p95_seconds`.
- Split **không có nhãn** (public, private): `finalize()` chạy `check_submission` (B3) thay cho chấm điểm, ghi kết quả ra `check.txt`. Trong lần chạy, `Runner` gọi với `open_tool_envs=False`: bước 2 và 4 bị bỏ, vì cả hai mở thêm một `ToolEnv` cho mỗi mẫu (`zaic_tools --validate` cũng mở `ToolEnv` để lấy danh sách tool được phép), mà thể lệ chỉ cho đúng một `ToolEnv` mỗi mẫu (D41). Hai bước này chỉ chạy khi gọi `check_submission` riêng, ngoài lần chạy giải pháp (ví dụ trước khi nộp public):
  1. Mọi id của split có mặt đúng một lần, JSON hợp lệ, mỗi phần tử của `tool_calls` có dạng `{"name": str, "arguments": object}`; dòng sai dạng được báo là dòng không hợp lệ chứ không làm checker crash.
  2. `python -m zaic_tools --split <split> --root <root> --validate submission.jsonl` không báo lỗi (subprocess, cwd là thư mục split).
  3. Phân bố: tỉ lệ `[]`, tỉ lệ có `sys.ask_missing_info`, số call trung bình, so với train (7%, 14%, 2,3). Lệch quá 2 lần thì in cảnh báo, không chặn.
  4. Phát lại các call qua `zaic_tools` và đếm `ok: false`; khác 0 thì in cảnh báo kèm id.
- `runs/public_submissions.csv` (ghi tay sau mỗi lần nộp public): `date, run, public_score, val_seen, val_unseen, note`. Dùng để xem điểm val và public có tăng giảm cùng chiều không.
- **Bộ smoke cố định:** `configs/splits/smoke.txt` gồm 30 id lấy phân tầng từ `train` (theo loại đáp án và có/không history), tạo bởi `make_splits.py --smoke 30`. Dùng cho chạy thử sau mỗi lần sửa; `--limit N` chỉ dành cho kiểm tra cài đặt.

### 4.12. Thư mục một lần chạy

```
artifacts/runs/<name>/
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

Cả hai là `Enum` trong `lifemate/core/types.py`. Ghi ra JSON bằng `.value`.

**`ErrorCode`:**

| Mã | Ý nghĩa | Nơi ghi |
|---|---|---|
| `parse_error` | Đầu ra LLM không tách được thành `Action` | `steps[i].error` và `errors` |
| `unknown_tool` | Tool không được phép (G1) | `steps[i].guard = "block"`, `steps[i].error`, `errors` |
| `invalid_args` | `env.validate` từ chối (G2) | như trên |
| `duplicate_write` | Gọi lại một call `write` giống hệt (G3, chỉ cảnh báo) | `errors` |
| `exception` | Exception ngoài dự kiến, bắt ở `Runner` | `errors`, kèm traceback trong trace |

**`StopReason`:** `done`, `asked`, `max_steps`, `timeout` (tới `deadline_s`; bước đang chạy bị ngắt), `context_overflow`, `parse_error`, `retries_exhausted`, `exception`.

### 4.14. Config (`configs/v0.toml`)

Đọc vào các dataclass `LLMConfig`, `AgentConfig`, `ASRConfig` trong `lifemate/core/config.py`. Khóa lạ hoặc thiếu khóa thì báo lỗi ngay khi khởi động.

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
deadline_s = 29.0                      # giây tính từ đầu mẫu (gồm ASR); tới mốc này bước LLM đang chạy bị ngắt
max_result_chars = 4000
system_prompt = "configs/prompts/system_v0.md"
guards = ["allowed_tool", "schema", "duplicate_write"]   # "after_ask" có sẵn nhưng không dùng, xem 4.9

[asr]
backend = "gold"                       # "gold" | "asr" (sau v0)
engine = "faster_whisper"              # "faster_whisper" | "hf_seq2seq" | "chunkformer" | "nemo_ctc" (mục 6.1)
dtype = "auto"                         # "auto" | "float32" | "float16" | "bfloat16"; auto: bf16 trên sm80+, fp16 trên GPU cũ, fp32 trên CPU
model = "Systran/faster-whisper-large-v3"   # bắt buộc khi backend = "asr" (mục 6.0)
revision = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
sanitize = true                        # lọc câu ảo giác bằng lifemate/backends/asr_sanitizer.py
# generate_kwargs: beam 5, best_of 5, temperature 0, VAD, condition_on_previous_text = false, không initial_prompt,
# không repetition_penalty / no_repeat_ngram_size (WER val_seen 29,7% so với 44,0% khi có; decisions.md D30)
```

### 4.15. Sửa bộ chấm `lifemate/eval/score.py`

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
| `persist` | `kaggle_dataset`: cuối phiên lưu `artifacts/runs/` thành một version của Kaggle Dataset riêng tư | `drive`: ghi thẳng vào Drive, không cần làm gì thêm | `local` | `local` |

- Profile thứ năm `local.toml` cho máy phát triển: `data_root = "data/train+public_test"`, `runs_dir = "runs"`, `hf_cache` mặc định của HF, `hf_token_source = "env:HF_TOKEN"`, `persist = "local"`. Đường dẫn tương đối tính từ thư mục gốc repo.
- Thứ tự ưu tiên: tham số dòng lệnh (`--root`, `--out`) > profile > giá trị mặc định. Code trong `lifemate/` **không có đường dẫn nào gắn cứng**. `Runner` truyền `--root` tường minh khi gọi `lifemate/eval/score.py`.
- `data_root` có thể chỉ đọc. Code không ghi gì vào `data_root`. `zaic_tools` được import từ đó, và Python tự bỏ qua việc ghi `__pycache__` khi thư mục chỉ đọc.
- Token HF chỉ được đọc lúc chạy theo `hf_token_source`. Không bao giờ ghi token vào config, notebook, trace hay `config.json`.
- Trước khi tải model, kiểm tra chỗ trống của ổ chứa `hf_cache`. Không đủ khoảng 1,5 lần kích thước model thì báo lỗi rõ ràng.

**Tự nhận GPU (`lifemate/core/envinfo.py`):**

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
- `code_version`: lấy từ `git rev-parse HEAD` cộng `git status --porcelain`. Nếu không có `.git` (code đưa lên Kaggle dạng dataset hoặc zip) thì đọc file `VERSION` do `ops/scripts/pack_code.sh` ghi lúc đóng gói. Không có cả hai thì ghi `"unknown"` và in cảnh báo.
- `gpu_memory_utilization` vẫn đặt trong config, vì giá trị này phụ thuộc việc có chạy ASR chung GPU hay không. `config.json` ghi kèm `gpu_mem_gb` để đối chiếu.

**Cài đặt:**

| Nơi | Cách cài | Lý do |
|---|---|---|
| Kaggle, Colab | Giữ torch và CUDA có sẵn. Cài các gói còn lại bằng `pip install -r ops/requirements/notebook.txt`. File này sinh từ `uv.lock` nhưng bỏ torch và các gói CUDA. Chọn phiên bản vLLM khớp với torch có sẵn | Cài lại torch trên Kaggle hay Colab dễ làm hỏng CUDA và tốn thời gian |
| Máy thuê, Docker | `uv sync --frozen` theo `uv.lock` đầy đủ | Môi trường sạch, khóa chặt phiên bản |

Phiên bản thật của các gói được ghi vào `config.json` (`EnvInfo.packages`), để biết hai lần chạy có dùng cùng phiên bản hay không.

**Đưa code lên Kaggle và Colab:** `ops/scripts/pack_code.sh` nén `lifemate/`, `lifemate/eval/`, `configs/`, `configs/prompts/`, `configs/splits/`, `pyproject.toml` và các file requirements, kèm file `VERSION` (commit và trạng thái dirty). Kaggle dùng file nén này làm Dataset riêng tư. Colab dùng `git clone` hoặc file nén trên Drive.

**Notebook mẫu (`ops/notebooks/kaggle_run.ipynb`, `ops/notebooks/colab_run.ipynb`)** gồm các ô theo thứ tự:

1. Cài đặt.
2. In `EnvInfo`.
3. Chạy `pytest` nhanh (bỏ T1).
4. Chạy `--limit 30`.
5. Chạy đủ `val_seen` và `val_unseen`.
6. Lưu `artifacts/runs/` theo `persist`.

Notebook chỉ gọi lệnh, không chứa logic. Mọi logic nằm trong `lifemate/`.

**Notebook train (`ops/notebooks/{kaggle,colab}_train_{llm,asr}.ipynb`)** theo cùng mẫu: cài `ops/requirements/notebook.txt` + `ops/requirements/train.txt`, in `EnvInfo`, chạy `pytest tests/train`, khôi phục `artifacts/runs/` của phiên trước (Kaggle: chép từ dataset `lifemate-runs` đã đính kèm; Colab: work dir nằm sẵn trên Drive), gọi `lifemate.train.llm_pipeline` hoặc `asr_pipeline` một lệnh, in `artifacts/models/registry.csv` và metric, lưu `artifacts/runs/` theo `persist`. Pipeline đọc `pipeline_progress.json` nên phiên sau chạy tiếp từ stage dở dang.

**Máy thuê RunPod:** profile `runpod` (`/workspace` là ổ bền), `ops/scripts/runpod_setup.sh` cài uv, lấy code, tải snapshot hai model vào `artifacts/models/`, chạy test và smoke; chi tiết `docs/runpod.md`.

**Nơi không chạy được:** Kaggle và Colab không chạy được Docker. Việc kiểm tra Docker (mục 6.3) làm trên máy thuê có Docker (nhiều template RunPod không có).

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
| T10 | `test_timeout_at_agent_level`, `test_budget.py`, `test_time_left_aborts_running_step`: `clock` giả, mỗi bước 5 giây, `deadline_s = 29` | 5 bước xong ở giây 25, bước 6 được bắt đầu và bị ngắt ở giây 29; 5 call đã gọi vẫn nộp; bước đầu chậm 14 s không chặn bước sau | Dừng trước hạn vì dự đoán, chạy quá `deadline_s`, hoặc mất call đã gọi | Thời gian thật trên A30 |
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
| T26 | `test_smoke_split`: `make_splits.py --smoke 30` | `smoke.txt` có 30 id, đều thuộc `train`, không trùng `val_seen`, có đủ 3 loại đáp án, cố định theo seed | Thiếu loại, id ngoài `train` hoặc trùng `val_seen`, hoặc đổi giữa hai lần chạy | |

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

**Đã chọn** (lý do và kế hoạch fine-tune ở phụ lục E.5): LLM `Qwen/Qwen3-4B-Instruct-2507` (4,02B, Apache-2.0, không có chế độ thinking), ASR `Systran/faster-whisper-large-v3` (1,55B, MIT, `engine = "faster_whisper"`, từ 28/10/2026 theo team lead; trước đó là `vinai/PhoWhisper-medium` 0,77B, BSD-3, `engine = "hf_seq2seq"`). Tổng 5,57B. Đã điền `model` và `revision` vào `configs/v0.toml`, `configs/docker.toml` (đường dẫn `/app/models/*`) và `configs/train/*.toml`. Ứng viên A/B: `Qwen/Qwen3.5-4B`; ASR `chunkformer` và `viegem` (hai engine tùy chọn trong `lifemate/backends/asr.py`, extras `asr-chunkformer` / `asr-viegem`, giấy phép phi thương mại cần BTC xác nhận).

**Trước khi chọn LLM, v0 vẫn làm được:** mọi việc trừ #4 (VLLMBackend chạy thật) và #8 (chạy v0 trên Kaggle) đều không cần model. `config.py` báo lỗi rõ ràng nếu `llm.backend = "vllm"` mà `model` còn trống.

### 6.1. ASR (A2, A3, B5)

- `ASRTranscriber` cài theo interface `Transcriber`. Model được nạp qua `[asr].engine` (`ASR_ENGINES` trong `asr.py`): `faster_whisper` (CTranslate2, mặc định, whisper-large-v3), `hf_seq2seq` (transformers pipeline, dùng cho PhoWhisper; nhận cùng tên khóa giải mã với faster-whisper qua `hf_generate_kwargs`: `beam_size` thành `num_beams`, `initial_prompt` thành `prompt_ids`), `chunkformer` (gói `chunkformer`, CTC tham lam; tải snapshot đúng revision trước khi nạp), `nemo_ctc` (NeMo, dùng cho `nvidia/parakeet-ctc-0.6b-Vietnamese`; CTC tham lam, tự gọi `forward` thay cho `transcribe` vì bộ nạp dữ liệu lhotse lỗi với torch 2.13). Engine mới chỉ cần một class có `transcribe_batch` và `warmup` rồi đăng ký vào `ASR_ENGINES`.
- Chạy chung GPU với vLLM. **Thứ tự nạp (B2):** nạp vLLM trước với `gpu_memory_utilization` đã chừa chỗ cho ASR (vLLM tính tỉ lệ trên tổng VRAM lúc khởi động), nạp ASR sau, rồi warm-up cả hai. `Runner` đo VRAM còn trống sau khi nạp xong và ghi vào `config.json`; ASR hết chỗ thì báo lỗi ngay, không chạy mẫu nào. **Đã đo 28/10/2026:** T4 15 GB không đủ cho large-v3 chạy cạnh vLLM (431/452 mẫu lỗi hết bộ nhớ, `experiments.md` base28 eval v2); eval và baseline có ASR thật cần GPU ≥ 24 GB (L4, A30, A100). `asr.dtype` giữ `"auto"`; int8 không dùng cho baseline vì khác cấu hình A30 lúc chấm.
- Chép chữ cho **mọi** lượt user trong history, tối đa 13 file mỗi mẫu. Thời gian ASR tính vào 30 giây.
- Tổng thời lượng audio mỗi mẫu (đo 27/10/2026): train p50 13 s, p95 41 s, max 69 s; public p50 13 s, p95 40 s, max 83 s (11 file).
- **Cách chạy ASR:**
  1. Gom mọi file audio **của cùng một mẫu** (lượt cuối và các lượt user trong history) vào một batch, theo thứ tự cố định: `_u1, _u2, …`, rồi lượt cuối. Batch có thể nhỏ hơn số file nếu GPU không đủ bộ nhớ, nhưng cách chia phải cố định theo thứ tự đó.
  2. **Không** gom file của nhiều mẫu khác nhau, **không** chạy ASR trước cho cả split rồi mới chạy agent. Mỗi mẫu đi trọn ASR → agent → ghi kết quả rồi mới sang mẫu sau (quyết định S5).
  3. Giải mã tất định (`temperature = 0`), cố định seed. Bản large-v3 dùng beam 5 theo team lead (`generate_kwargs` trong `configs/v0.toml`); cần đo thời gian trên T4.
  3a. **Lọc ảo giác** (`sanitize = true`): cắt câu kiểu phụ đề YouTube và vòng lặp từ; lượt lịch sử chỉ toàn ảo giác thành chuỗi rỗng. Chữ số chỉ bị gộp khi lặp từ 8 lần để giữ số điện thoại. Kiểm tra trên 5036 câu `audio_text`: không câu nào bị đổi.
  3a'. **Câu mồi** (`initial_prompt`): khai trong `asr.initial_prompt_file` (file có phiên bản trong `configs/asr_prompts/`, đường dẫn tính từ gốc repo, giống `agent.system_prompt`); `load_config` đọc file vào `generate_kwargs.initial_prompt`. Không khai cả hai cùng lúc.
  3b. **Bảo vệ câu mồi** (`prompt_guard_ngram`, mặc định 0 = tắt; chỉ engine Whisper, cần `initial_prompt`): câu chép với câu mồi bị coi là hỏng khi rỗng sau lọc, bộ lọc ảo giác có cắt, chứa cụm `prompt_guard_ngram` từ liên tiếp của câu mồi, hoặc tỉ lệ nén zlib lớn hơn `prompt_guard_max_compression` (2,4). Câu hỏng được chép lại không câu mồi (`lifemate/backends/asr_guard.py`). Vẫn tất định.
  4. Chỉ khi đo thấy ASR quá chậm mới tính đến việc bỏ bớt các lượt user cũ. Khi đó vẫn giữ `text` và `tool_calls` của các lượt assistant để giữ ngữ cảnh.
- **Đo trước khi chọn model ASR:** chạy trên 3 mẫu có tổng audio dài nhất của train và public, ghi thời gian ASR và tổng thời gian mỗi mẫu. Trên GPU sm80+, tổng thời gian mỗi mẫu phải ≤ 25 s.
- **Normalizer (A3): không làm (quyết định 27/10/2026).** Đo trên Kaggle (30 mẫu smoke): `audio_text` của BTC viết số bằng chữ số ("8 giờ kém 15", "30 phần trăm", "126 triệu"), còn PhoWhisper viết bằng chữ ("tám giờ kém mười lăm", "ba mươi phần trăm"). BTC không chấm chữ ASR, chỉ chấm tool call và trạng thái cuối, nên **giữ nguyên chữ ASR nghe ra** và cho LLM học trên chính chữ đó (LLM v2, `transcript_source = "asr"`, mục 6.2.1). LLM tự đổi số theo quy tắc trong system prompt. Không đổi chữ ASR về dạng `audio_text` trước khi đưa vào LLM.
- **`asr.dtype`** (mặc định `"auto"`): bfloat16 trên GPU sm80+ (A30), float16 trên GPU cũ hơn (T4), float32 trên CPU (`resolve_asr_dtype`). Không ghi dtype thì `transformers` nạp float32: khoảng 3 GB cho PhoWhisper-medium, hết bộ nhớ khi chạy chung vLLM trên T4 15 GB (đo 27/10/2026).
- **Nghiệm thu:**
  - Báo cáo WER và CER trên `val_seen` và `val_unseen`, dùng `audio_text` làm đáp án.
  - Báo cáo điểm agent khi dùng ASR trừ điểm khi dùng `GoldTranscriber`, đây là phần mất do ASR.
  - p95 thời gian ASR mỗi mẫu được ghi trong `summary.json`.
  - Chạy ASR hai lần trên 30 mẫu cho ra chữ giống hệt nhau.

### 6.1.1. Quy trình benchmark ASR: dữ liệu, thước đo, cách so sánh

Phần này là chuẩn chung cho mọi lần benchmark ASR (đã áp dụng ở `docs/reports/benchmark/asr_v1.md`, `asr_v2.md`, `asr_v3.md`;
siết thêm ở D53). Mỗi báo cáo benchmark phải ghi rõ dùng tập nào cho việc gì, và không được dùng một tập cho hai vai trò.

**Dữ liệu và vai trò** (`lifemate/eval/make_splits.py`, seed 42, chỉ ghi danh sách mã mẫu):

| Tập | Số mẫu | Vai trò | Luật dùng |
|---|---|---|---|
| `asr_learn` (`train` ngoài `asr_tune`) | 793 | **Học**: đếm từ để viết câu mồi, bảng độ tin cậy từng từ, trọng số bộ phân loại | Mọi tham số học từ dữ liệu chỉ được học ở đây |
| `asr_tune` (rút phân tầng từ `train`) | 300 | **Dò**: chọn giữa các cấu hình hoặc cách làm đã cố định tham số | Mỗi bước chọn dùng tập dò một lần; tập dò dùng lại qua nhiều bước thì số trên tập dò lạc quan, phải ghi rõ |
| `train` (= `asr_learn` + `asr_tune`) | 1 093 | **Kiểm định chéo** 5 phần theo mẫu (`fold_of`) cho bước chọn nào chấm offline được | Ưu tiên cách này thay cho tập dò 300 mẫu khi so các cách quyết định (lọc, ghép, bộ phân loại): số đoạn có nhãn gấp ba, CI hẹp hơn |
| `val_seen` | 274 | **Đo** cấu hình thắng, app đã gặp | Đo sau khi đã chọn; được xem khi phân tích giữa chừng, nhưng không chọn gì theo số của tập này |
| `val_unseen` | 178 | **Đo cuối**, app chưa gặp (giả lập private test) | **Khoá**: chỉ đo một lần cho cấu hình sắp nộp (D53) |
| `*_en_mixed` | 165 / 152 / 95 | Lát cắt có câu chêm tiếng Anh của `asr_tune`, `val_seen`, `val_unseen` | Không phải tập riêng: mô hình chép cả tập gốc, chỉ chấm thêm các chỉ số tiếng Anh trên lát cắt |
| `smoke` | 30 (từ `train`) | Chạy thử bắt lỗi | Không dùng để chấm hay chọn |
| `public_test` | 634 | Đo ngoài, một con số tổng | Nộp một lần mỗi mốc lớn với cấu hình đã chốt trên val; ghi cặp (val, public) vào `docs/experiments.md`; không dò tham số trên public |

Quy tắc chung: tập nào đã dùng để **viết** một thứ (câu mồi, bảng từ) thì không được dùng để **chọn** thứ đó; tập nào dùng để chọn thì
không được dùng để báo kết quả. Tiêu chí chọn ghi vào báo cáo **trước khi đo**; chọn ngoài tiêu chí thì ghi rõ là ngoài tiêu chí và
không viện số val làm lý do (bài học D42).

**Thước đo** (`lifemate/eval/eval_asr.py`, tính trên mọi câu người dùng của mẫu: lượt cuối và các lượt trong lịch sử):

| Nhóm | Chỉ số | Dùng khi |
|---|---|---|
| Độ đúng chữ | WER, CER, `wer_nonum` (bỏ số, vì đáp án và Whisper viết số khác cách) | Mọi kịch bản |
| Thực thể | `entity_miss_rate`: giá trị tham số của lời gọi đúng không có trong chữ ASR | Mọi kịch bản; đây là hàm mục tiêu khi dò (cùng với tỉ lệ câu rỗng) |
| Câu hỏng | `n_empty` (câu rỗng), `n_hallucinated` (câu bịa trước khi lọc), `n_unk` (toàn `⁇`), `n_prompt_retries` (chép lại do lớp bảo vệ) | Mọi kịch bản; bắt buộc khi thử câu mồi, guard, sanitizer |
| Tiếng Anh | `en_word_recall` (và bản nới lỏng), theo nhóm từ (`en_groups`), theo có/không trong câu mồi (`en_prompt`) và trong ngữ cảnh (`en_context`); `en_mixed_wer`, `vi_only_wer`; `prompt_insertions` (từ câu mồi chèn vào khi không nói) | Kịch bản chêm tiếng Anh, câu mồi, ghép |
| Ghép | `merge_decision_accuracy` (đoạn chọn đúng phía ít lỗi hơn), `merge_wrong_english`, `merge_missed_english` | Kịch bản ghép, so cách quyết định |
| Thời gian, bộ nhớ | p50/p95/max giây mỗi **mẫu** (không phải mỗi câu), RTF, `load_s`, `gpu_peak_mb` | Mọi phiên GPU, đo kèm, cùng máy cho mọi ứng viên |
| Điểm thi | Điểm agent trên chữ ASR đã lưu (`--transcripts replay`), chênh so với chữ chuẩn (`gold`) | Kịch bản chọn pipeline cuối |

Quy tắc tìm từ tiếng Anh (`foreign_words.py`) đã rà tay; đổi quy tắc thì rà lại và ghi độ chính xác.

**Cách so sánh và kết luận:**

- Hai hệ thống khác nhau khi CI 95% của **chênh lệch** không chứa 0, bằng bootstrap **ghép cặp theo mẫu** (1 000 vòng, seed 0,
  `asr_compare.compare`); không so hai CI riêng lẻ, không bootstrap theo câu.
- Trước khi so, ước lượng độ nhạy: với khoảng 400 từ tiếng Anh, chênh 10 từ đã là 2,5 điểm. So sánh "không có ý nghĩa thống kê" chỉ
  có nghĩa là chưa đo được, không phải "không hơn"; báo cáo phải ghi cả hai cách hiểu và nêu cần thêm bao nhiêu dữ liệu.
- Nhiều ứng viên cho cùng một câu hỏi thì so trong **một bảng chung**, một lần, một quy tắc chọn; không chọn nối tiếp.
- Hoà thì chọn ứng viên **đơn giản hơn** (ít tham số, ít thời gian chạy, không cần mô hình thêm).
- Kiểm tra tất định: chạy lại ít nhất 50 mẫu, chữ phải giống hệt; cấu hình không tất định (nhiệt độ thử lại) bị loại.
- Mọi phiên GPU lưu `hyps.jsonl` (chữ từng câu, câu mồi đã dùng, điểm tin cậy) để chấm lại offline bằng `rescore_asr`; kết quả thô ở
  `artifacts/`, không lên git; báo cáo chỉ trích vài câu ngắn làm ví dụ.

**Kịch bản benchmark và cách áp dụng:**

| Kịch bản | Dữ liệu | Thước đo chính | Cách kết luận |
|---|---|---|---|
| 1. Chạy mô hình gốc, lấy chỉ số nền | `val_seen`, `val_unseen` một lần, cấu hình mặc định của mô hình | WER, `entity_miss_rate`, câu hỏng, p95 | Chỉ làm mốc; không chọn gì |
| 2. Dò cấu hình tốt nhất của một mô hình (`asr_tune.py`) | `asr_tune`; đo thắng trên val | Hàm mục tiêu: `entity_miss_rate` + tỉ lệ câu rỗng, hoà thì `wer_nonum`; p95 vượt ngưỡng thì xếp sau | Dò từng bước theo thứ tự cố định (giải mã, VAD, câu mồi, chống lặp), mỗi bước giữ lựa chọn tốt nhất; cùng công sức dò cho mọi mô hình |
| 3. So các mô hình bằng cấu hình thắng của từng mô hình | val một lần | Như 2, cộng thời gian và giấy phép | Chọn mô hình chính và mô hình dự phòng; chưa đổi config hệ thống nếu chưa có điểm thi |
| 4. Câu tiếng Việt chêm tiếng Anh | Lát cắt `*_en_mixed` của các tập đã có; chấm lại chữ đã lưu | `en_word_recall` tách theo nhóm, `en_mixed_wer`, `vi_only_wer` | Báo cả hai mặt: tiếng Anh được bao nhiêu, tiếng Việt mất bao nhiêu |
| 5. Câu mồi (`initial_prompt`) | Từ chọn trên `asr_learn`; cách dùng chọn trên `asr_tune`; đo trên val | `en_word_recall`, `prompt_insertions`, `n_prompt_retries`, `vi_only_wer`, p95 | Câu mồi tự viết, không chép câu dữ liệu; dò câu mồi cùng lớp bảo vệ; điều kiện: câu thuần Việt không tệ hơn |
| 6. Guard (câu bịa, câu rỗng, chép lại câu mồi) và sanitizer | Như 5; thêm rà tay trên câu đáp án (bộ lọc không được cắt câu thật) | `n_hallucinated`, `n_empty`, `n_prompt_retries`, số câu đúng bị báo nhầm | Lớp bảo vệ không được báo nhầm quá 0,5% câu đáp án; dấu hiệu hỏng phải không cần đáp án |
| 7. Ghép hai mô hình và cách quyết định từng đoạn | Tham số học trên `asr_learn`; chọn cách quyết định bằng kiểm định chéo trên `train` (1 093 mẫu), tập dò chỉ để đối chiếu; đo thắng trên val | `merge_decision_accuracy` (chỉ số chọn), `merge_wrong_english`, `merge_missed_english`, `vi_only_wer`, thời gian chấm thêm | Điều kiện chặn: không tệ hơn mốc có ý nghĩa ở tiếng Anh và câu thuần Việt, thời gian thêm ≤ 1 s p95; hoà thì chọn cách đơn giản hơn |
| 8. Chọn pipeline cuối bằng điểm thi | Chữ đã lưu của từng ứng viên, chạy agent trên val (`--transcripts replay`), kèm chữ chuẩn làm trần | Điểm thi, CI chênh so với mốc, loại lỗi, số mẫu hết giờ | Đưa vào config nộp bài khi không thấp hơn mốc có ý nghĩa; chỉ chạy lại mẫu có chữ khác mốc (agent tất định); nộp public test một lần |
| 9. Thời gian cả hệ thống | Hai tập val, GPU giống máy chấm (24 GB), ASR và LLM chạy cùng | p95/max giây mỗi mẫu, số mẫu quá 30 s, bộ nhớ đỉnh | Làm trước khi đóng Docker; không đổi cấu hình theo val ở bước này |

### 6.2. Fine-tune LLM (C1, C2, C4)

Phần này chia hai nhóm:

- **Cố định ngay:** định dạng dữ liệu, cách che loss, quy tắc dữ liệu, cách đưa model đã train vào hệ thống, quản lý phiên bản. Những thứ này ràng buộc code v0 nên phải chốt trước.
- **Chốt sau khi có số đo v0:** rank, learning rate, số epoch, có cần dữ liệu tổng hợp hay không. Các giá trị dưới đây chỉ là mặc định để bắt đầu.

#### 6.2.1. Dữ liệu SFT (`lifemate/train/sft_data.py`)

```
uv run python -m lifemate.train.sft_data --config configs/train/llm_v1.toml --env kaggle \
    --ids configs/splits/train.txt --out data/sft/v1.jsonl
```

**Quy tắc dựng một mẫu, phải giống hệt lúc chạy thật:**

1. Dựng prompt bằng đúng `PromptBuilder` và prompt hệ thống của lúc chạy (`agent.system_prompt`). Sau đó, **xóa 6 app giữ lại** (`configs/splits/meta.json` → `held_out_apps`) khỏi `installed_apps` trước khi render tool docs (mục 4.5).
2. Transcript: `transcript_source = "mix"` (mặc định của `llm_v1.toml` từ 28/10, theo team lead) lấy chữ ASR cho 70% mẫu (chọn theo hash của id) và `audio_text` cho phần còn lại; mẫu ASR rỗng hoặc làm mất tên, số của đáp án thì dùng `audio_text`. Chữ ASR đọc từ cache dựng một lần bằng `python -m lifemate.train.asr_cache` (cần GPU). Các lựa chọn khác: `"gold"` (chỉ `audio_text`), `"asr"` (chỉ chữ ASR). **Bản LLM dùng để nộp phải học trên `"asr"`** (LLM v2), vì lúc chấm LLM chỉ thấy chữ ASR, và chữ ASR khác `audio_text` cả ở lỗi nghe lẫn cách viết số (mục 6.1). Đổi ASR (model, dtype hay fine-tune) thì phải dựng lại dữ liệu và train lại LLM v2.
3. Với mỗi call `i` trong `label.tool_calls`:
   - Thêm lượt assistant với nội dung `canonical_json({"action":"call","name":...,"arguments":...})`.
   - Thêm lượt user `KẾT QUẢ: ...`. Kết quả này lấy bằng cách **phát lại call qua `zaic_tools`**, không chép từ `label.tool_results`, để khớp từng byte với lúc chạy thật. Sau đó cắt theo `max_result_chars` giống hệt mục 4.10. Nếu kết quả phát lại khác `label.tool_results[i]` thì ghi cảnh báo kèm id.
4. Kết thúc:
   - Call cuối là `sys.ask_missing_info`: **không** thêm `done`, vì lúc chạy vòng lặp dừng ngay sau call đó (`StopReason.ASKED`).
   - Ngược lại: thêm lượt assistant `{"action":"done"}`.
   - Mẫu gold `[]`: chỉ có một lượt assistant `{"action":"done"}`.
5. Đếm token bằng `HFTokenCounter` của model gốc. Mẫu vượt `max_model_len` thì bỏ, ghi id vào `data/sft/v1.dropped.txt`.
6. **Tăng cường, chỉ cho tập train** (`sft_data.Augment`, theo team lead): `error_recovery_ratio` (mặc định 0,03) thêm trước lời gọi đúng một lời gọi thiếu tham số bắt buộc và câu lỗi thật của guard; lượt gọi sai mang `"loss": false`. Tắt mặc định: `oversample_hard` (nhân bản mẫu hỏi lại và nhãn rỗng), `distractor_apps` (thêm app đã cài gây nhiễu, không thêm vào mẫu nhãn rỗng, không bao giờ là app giữ lại), `dedup_max_k` (giữ tối đa k mẫu mỗi cụm câu gần trùng). File val không bị tăng cường.

**Định dạng một dòng:**

```json
{"id": "train_0874", "source": "train", "category": "execute", "error_recovery": false,
 "messages": [{"role": "system", "content": "..."}, ...], "n_tokens": 4210, "n_assistant_turns": 3}
```

Đi kèm `data/sft/v1.meta.json`: sha256 của file, lệnh đã chạy, `code_version`, sha256 của prompt, danh sách id, số mẫu bị bỏ, số cảnh báo phát lại.

**Dữ liệu chỉ lấy từ:** `configs/splits/train.txt`, và sau này thêm dữ liệu tổng hợp (mục 6.2.5). **Không bao giờ** lấy từ `val_seen` hay `val_unseen` khi còn dùng chúng để chọn model. Không dùng gì từ `public_test`.

#### 6.2.2. Che loss

- Chỉ tính loss trên token của các lượt **assistant**, gồm cả token kết thúc lượt của chat template. Không tính trên system, user, `KẾT QUẢ`, và lượt assistant có `"loss": false` (lượt gọi sai của mẫu phục hồi lỗi).
- ms-swift tự che loss theo cùng quy tắc. `sft.build_labels` (che loss bằng chat template của tokenizer, giống vLLM) là chuẩn để `swift_sft.check_template` so trước mỗi lần train.
- Chat template render với cùng tham số như lúc chạy (quyết định S6). Qwen3-4B-Instruct-2507 không có chế độ thinking nên không cần `enable_thinking`; nếu A/B với Qwen3.5-4B thì truyền `enable_thinking=False` ở cả train và chạy.
- Thiếu bước che loss thì model học thuộc mô tả tool và kết quả tool, làm hỏng khả năng đọc app lạ.

#### 6.2.3. Script train (`lifemate/train/swift_sft.py`, ms-swift)

```toml
# configs/train/llm_v1.toml (theo team lead 28/10; chưa đo trên pipeline này)
[base]
model = "Qwen/Qwen3-4B-Instruct-2507"   # cùng model với [llm] của configs/v0.toml
revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
run_config = "configs/v0.toml"   # lấy prompt, max_result_chars, max_model_len, max_tokens từ đúng config lúc chạy
[data]
train = "data/sft/v1.jsonl"
eval_seen = "data/sft/v1_val_seen.jsonl"      # dựng bằng cùng script, chỉ để đo loss, không train
eval_unseen = "data/sft/v1_val_unseen.jsonl"  # tập val duy nhất truyền cho ms-swift
transcript_source = "mix"
asr_ratio = 0.7
asr_cache = "data/asr/train_asr.json"
error_recovery_ratio = 0.03
oversample_hard = 1.0
distractor_apps = 0
dedup_max_k = 0
[lora]
quant = "nf4"               # QLoRA 4-bit; "none" = LoRA 16-bit trên GPU >= 24 GB
r = 32
alpha = 64
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
[train]
lr = 2e-4
lr_scheduler = "cosine"
tuner_backend = "peft"      # ms-swift --tuner_backend: peft | unsloth
warmup_ratio = 0.05
epochs = 3
batch_size = 1
grad_accum = 16
max_len = 10240
seed = 0
eval_every_steps = 50
save_every_steps = 50
```

- Train bằng ms-swift 4.6.0 (`swift sft`, `lifemate/train/swift_sft.py`), cùng công cụ với team lead; fp16 trên T4, bf16 trên sm80. `seed` cố định cho Python, NumPy, PyTorch và thứ tự trộn dữ liệu. Trước khi train, `check_template` so token và nhãn loss của ms-swift với chat template lúc chạy (vLLM) trên tối đa 20 mẫu, khác thì dừng. Tập val truyền cho ms-swift chỉ là `val_unseen`, vì ms-swift gộp nhiều tập val thành một loss. Trình train Unsloth cũ còn trong `sft.py`, đã comment lại.
- **Train tiếp được:** ms-swift lưu `checkpoint-<step>/` mỗi `save_every_steps` (adapter, optimizer, scheduler, trạng thái RNG, `trainer_state.json`) vào thư mục theo profile môi trường (mục 4.16). Chạy lại thì `swift_sft.train` tự nối tiếp từ checkpoint mới nhất (`--no-resume` để chạy lại từ đầu). Lý do: một phiên Kaggle chỉ kéo dài khoảng 9–12 giờ.
- Mỗi `eval_every_steps` đo loss trên `eval_unseen` (ms-swift gộp nhiều tập val thành một loss, nên chỉ truyền một tập). Loss nằm trong `trainer_state.json` của từng checkpoint. `save_every_steps` phải là bội số của `eval_every_steps` (config loader kiểm tra) để mỗi checkpoint có loss.

**Chọn checkpoint:**

1. Lấy 2 checkpoint có loss `eval_unseen` thấp nhất.
2. Merge từng checkpoint (mục 6.2.4), chạy agent thật trên `val_seen` và `val_unseen` bằng `lifemate.run.cli` với chữ ASR thật (`--transcripts asr`), không dùng transcript gold, vì lúc chấm LLM chỉ thấy chữ ASR (`decisions.md` D34).
3. Chọn checkpoint có điểm `val_unseen` cao nhất. Bằng điểm thì chọn bản có điểm `val_seen` cao hơn.

Chỉ đánh giá bằng agent thật trên 2 checkpoint, để hạn chế việc chọn model theo bộ val quá nhiều lần.

#### 6.2.4. Đưa model đã train vào hệ thống

- **Merge adapter vào model gốc bf16**, lưu thành safetensors đầy đủ. vLLM nạp model đã merge như một model thường, đổi bằng `[llm].model` trong config. Không dùng chế độ LoRA của vLLM, vì merge làm Docker đơn giản hơn và không tốn thêm thời gian chạy.
- Train trên model gốc 4-bit nhưng merge vào bản bf16, nên kết quả có thể lệch nhẹ. Vì vậy **luôn đánh giá model đã merge**, tức đúng thứ sẽ nộp, không đánh giá adapter.
- Merge chạy trên CPU được (T4 không có bf16).

#### 6.2.5. Dữ liệu tổng hợp (chỉ khi bảng lỗi v0 cho thấy cần)

- Sinh bằng LLM lớn trong lúc phát triển (luật cho phép), **chỉ dựa trên catalog train**, không dùng catalog `public_test` (`decisions.md` D5).
- Mỗi mẫu tổng hợp phải chạy được qua `zaic_tools` của train, và gold của mẫu phải tái tạo được trạng thái cuối. Mẫu nào không qua thì bỏ.
- Ghi `source = "synthetic"` và phiên bản dữ liệu riêng (`data/sft/synth_vN.jsonl`). Không bao giờ dùng mẫu tổng hợp để đánh giá.
- Một hướng đáng thử: các mẫu dạy model sửa lỗi. Model nhận `LỖI (chưa gọi tool): ...` rồi gọi lại cho đúng, khớp với hành vi của guard ở mục 4.9.

#### 6.2.6. Quản lý phiên bản và chạy lại

- Tên model: `ft-<base>-sft<v>-r<nn>`, ví dụ `ft-base4b-sft1-r01`.
- Mỗi model có thư mục `artifacts/models/<tên>/` gồm `train_config.json` (toàn bộ config, `code_version`, sha256 dữ liệu, seed, GPU, phiên bản gói, checkpoint được chọn, loss), `metrics.jsonl`, `eval.json` (điểm agent trên hai bộ val, kèm tên run).
- Trọng số lưu trên Kaggle Models hoặc HF private, không vào git. `artifacts/models/registry.csv` (trong git) gồm `name, base, data_version, checkpoint, val_seen, val_unseen, location`.
- **Một lệnh chạy lại toàn bộ** (yêu cầu nộp training code của BTC):
  ```
  uv run python -m lifemate.train.llm_pipeline --config configs/train/llm_v1.toml --env rented
  ```
  Lệnh này lần lượt: dựng dữ liệu → train → chọn checkpoint → merge → chạy agent trên `--limit 30` để kiểm tra nhanh.

#### 6.2.7. Nghiệm thu

1. Test (CPU, model nhỏ giả hoặc tokenizer thật):
   - Phần prompt của một mẫu SFT giống từng byte với `Conversation.messages` của lúc chạy ở cùng bước.
   - Token được tính loss khi decode ra bằng đúng các chuỗi JSON của assistant cộng token kết thúc lượt.
   - Không có tên app bị rút trong dữ liệu dựng từ `configs/splits/train.txt`.
   - Mẫu `ask_missing_info` không có lượt `done` ở cuối.
   - Không id nào của `val_seen` hay `val_unseen` lọt vào file train.
2. Phát lại toàn bộ dữ liệu SFT qua `FakeLLM` (dùng các lượt assistant làm đầu ra) cho ra 100,00 điểm trên các id của train.
3. Ngắt train giữa chừng rồi `--resume-from` cho ra loss ở bước kế tiếp giống với khi train liền một mạch (cùng GPU).
4. Model đã merge chạy được bằng `lifemate.run.cli`, và điểm `val_unseen` không thấp hơn model gốc dùng cùng prompt.
5. `llm_pipeline` chạy từ đầu đến cuối trên máy thuê bằng một lệnh.

### 6.3. Docker (D1 – D3)

- BTC mount thư mục test vào `/data`, với cùng cấu trúc như `public_test`. Nhưng `ToolEnv` cần dạng `<root>/<split>/...`. Vì vậy entrypoint phải:
  1. Tìm `/data/*_tools_catalog.json` để biết tên split.
  2. Tạo thư mục `/tmp/zaic_root/<split>` là symlink trỏ tới `/data`.
  3. Chạy `python -m lifemate.run.cli --config configs/docker.toml --env docker --split <split> --out result/` (`LIFEMATE_CONFIG` đổi config, `RESULT_DIR` đổi thư mục kết quả). **Không** truyền `--resume`: mỗi lần BTC chạy là một lần chạy mới, độc lập (thể lệ: lệch kết quả thì chạy 5 lần lấy trung bình). `result/trace.jsonl` còn từ lần trước thì entrypoint đổi tên thành `trace.<thời điểm>.bak`, để không chạy tiếp trên kết quả cũ và không chết vì config cũ lệch config mới (D39). `RunStore` ghi `submission.jsonl` cùng `trace.jsonl`, `summary.json`, `config.json`, `check.txt` vào `result/`; BTC chỉ lấy `submission.jsonl`.
  4. `zaic.py` so sánh đường dẫn sau `Path.resolve()` nên symlink không làm hỏng kiểm tra (B5).
- Trọng số model nằm sẵn trong image. Đặt `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`.
- **Nghiệm thu:**
  - `docker run --gpus all --network none -v <public_test>:/data ...` ra `result/submission.jsonl` đủ 634 dòng.
  - `python -m zaic_tools --validate` không báo lỗi.
  - Chạy hai lần trên cùng GPU sm80+ với `batch_invariant = true` cho ra hai file giống hệt.
  - Không mẫu nào vượt 30 giây.

### 6.4. Fine-tune ASR (C3)

**Chỉ làm khi số đo cho thấy cần, và đo sau LLM v2.** Thứ tự: ASR gốc (mục 6.1) → LLM v1 trên `audio_text` → LLM v2 trên chữ ASR thật (`transcript_source = "asr"`, mục 6.2.1) → so điểm agent khi dùng `GoldTranscriber` và khi dùng ASR trên `val_seen` và `val_unseen` với LLM v2. Lý do ở phụ lục E.5.2 và `rationale.md`: LLM nhìn thấy danh bạ và tên thiết bị nên tự sửa được phần lớn lỗi thực thể của ASR; đo trước v2 sẽ phóng đại phần mất do ASR. Fine-tune ASR khi **một trong hai** điều sau đúng:

- Điểm mất do ASR ≥ 3 điểm trên một trong hai bộ val. Ngưỡng 3 điểm này chốt lại sau khi biết độ dao động của điểm giữa các lần chạy.
- Tỉ lệ nghe sai thực thể (bên dưới) ≥ 10%.

> **Lưu ý 28/10/2026:** ASR lúc chạy đã đổi sang whisper-large-v3 bản CTranslate2, còn `configs/train/asr_v1.toml` vẫn là kế hoạch cho PhoWhisper-medium. Nếu fine-tune large-v3: train trên bản transformers `openai/whisper-large-v3` (1,55B, cần LoRA hoặc GPU ≥ 24 GB), rồi chuyển sang CTranslate2 bằng `ct2-transformers-converter` và trỏ `[asr].model` vào thư mục đó.

#### 6.4.1. Dữ liệu (`lifemate/train/asr_data.py`)

- Cặp `(file wav, audio_text)` gồm lượt cuối và **mọi lượt user trong history**.
- Số cặp theo cách chia hiện tại (28/10/2026): train **3593**, val_seen 916, val_unseen 527.
- Chia theo **id mẫu**, đúng như `configs/splits/`. Mọi file audio của một mẫu thuộc `val_*` không bao giờ được dùng để train.
- **Đích khi fine-tune ASR là `audio_text`**, vì đó là nhãn duy nhất có. Hệ quả: ASR sau fine-tune sẽ viết số theo kiểu BTC, khác ASR gốc. Vì LLM học trên chữ ASR (mục 6.1, 6.2.1), **sau mỗi lần fine-tune ASR phải dựng lại dữ liệu SFT bằng ASR mới và train lại LLM v2**. Đo bằng điểm agent, không chỉ WER, vì WER phạt cả khác biệt cách viết số mà BTC không chấm.
- Dữ liệu ghi ra `data/asr/vN.jsonl` (`{"id","turn","audio","text","split"}`) kèm file meta giống mục 6.2.1. Audio không chép, chỉ ghi đường dẫn tương đối trong split.

#### 6.4.2. Đo lường (`lifemate/eval/eval_asr.py`)

| Chỉ số | Cách tính | Vì sao |
|---|---|---|
| WER, CER | Trên chữ đã chuẩn hóa: chữ thường, NFC, bỏ token không biết (`⁇`, `<unk>`) và dấu câu. Câu chỉ có token không biết vì vậy là câu rỗng | Chỉ số chung |
| **Tỉ lệ nghe sai thực thể** | Lấy các giá trị tham số chuỗi và số trong `label.tool_calls` mà **có xuất hiện** trong `audio_text`, ví dụ tên người, địa điểm, số tiền. Đếm tỉ lệ không xuất hiện trong chữ ASR, sau khi bỏ dấu và chữ hoa (dùng cùng cách chuẩn hóa `_fold` của `zaic_tools`) | Đây là lỗi làm sai tham số tool, ảnh hưởng điểm trực tiếp |
| Điểm agent mất do ASR | Điểm khi dùng `GoldTranscriber` trừ điểm khi dùng ASR, cùng LLM và prompt | Quyết định có cần fine-tune hay không |
| Câu chêm tiếng Anh | Từ ngoại trong `audio_text` tìm bằng quy tắc (`lifemate/eval/foreign_words.py`: không dấu và không ghép được âm tiết Việt, hoặc viết hoa toàn bộ; nhóm từ thường, viết tắt, tên, từ mượn). `en_mixed_wer`: WER trên các câu có từ ngoại. `en_word_recall`: tỉ lệ từ ngoại có trong chữ ASR sau khi chuẩn hoá, theo nhóm và theo có/không trong câu mồi. `prompt_insertions`: từ ngoại của câu mồi xuất hiện trong chữ ASR mà đáp án không có | Benchmark ASR lần 2 (D40) |
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
kind = "full"               # "lora" | "full"; 0,77B full fine-tune vừa T4 (phụ lục E.5.2)
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

#### 6.4.4. Dữ liệu tổng hợp bằng TTS (làm trước khi mở encoder)

- Luật cho phép dùng TTS lúc phát triển, cấm dùng lúc chạy thi.
- Chỉ sinh cho loại lỗi mà bảng đo 6.4.2 chỉ ra, ví dụ tên người, địa danh, số tiền đọc bằng chữ, từ tiếng Anh xen kẽ.
- Câu tổng hợp lấy từ `audio_text` của `configs/splits/train.txt` rồi thay thực thể. Không dùng câu của val hay public.
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
| `seed_everything(seed)` (`lifemate/core/seeding.py`) | Cố định seed cho Python, NumPy, PyTorch, CUDA; bật chế độ tất định của PyTorch khi có thể |
| `CheckpointManager` | Lưu và nạp checkpoint (trọng số hoặc adapter, optimizer, scheduler, RNG, bước); giữ N checkpoint gần nhất cộng N tốt nhất theo chỉ số đã chọn |
| `MetricsLogger` | Ghi `metrics.jsonl` (bước, loss, chỉ số eval, thời gian); flush sau mỗi dòng |
| `write_train_config()` | Ghi `train_config.json` gồm config, `EnvInfo` (mục 4.16), sha256 dữ liệu |
| Đường dẫn | Lấy từ profile môi trường (mục 4.16); checkpoint ghi vào nơi được giữ lại sau phiên |

`asr.py` dùng các thành phần này. Fine-tune LLM dùng cơ chế checkpoint của ms-swift (`swift_sft.checkpoints`); trình train Unsloth cũ trong `sft.py` (đã comment lại) từng dùng chúng. Thêm `tests/train/` để kiểm tra `CheckpointManager` (lưu rồi nạp ra đúng bước và trạng thái RNG) và `seed_everything` (hai lần chạy cùng seed ra cùng số ngẫu nhiên).

---

## 7. Tiêu chí nghiệm thu v0

1. `uv run pytest` chạy hết 26 test ở mục 5 và đều đạt (hoặc skip có lý do theo đúng quy định ở mục 5), trên máy không có GPU. `ruff check`, `ruff format --check` và `mypy --strict lifemate/` không báo lỗi.
2. T1 (FakeLLM) cho đúng **100,00** trên `val_seen` và trên `val_unseen`.
3. Chạy v0 với vLLM trên `val_seen` (274 mẫu) và `val_unseen` (178 mẫu; số theo cách chia 28/10):
   - `submission.jsonl` có đúng số dòng bằng số id, không trùng id.
   - `summary.json` có `n_exception = 0`.
4. **0 call bị từ chối** trong mọi `submission.jsonl`. Kiểm bằng cách phát lại các call qua `zaic_tools` và đếm kết quả `ok: false`.
5. 0 mẫu có `stop_reason = "parse_error"` do JSON hỏng khi chạy với vLLM có ép schema. Lỗi `parse_error` chỉ được phép xuất hiện với FakeLLM trong test.
6. `tokens.max_prompt` < 15.872 (16.384 trừ 512) trên cả hai bộ val.
7. Chạy `--limit 30` hai lần liên tiếp trên cùng máy cho ra hai `submission.jsonl` giống hệt từng byte.
8. Điểm `val_seen` và `val_unseen` cao hơn điểm của bài nộp toàn `[]` trên cùng bộ. Điểm toàn `[]` lấy bằng `score.py` với file chỉ chứa `[]`.
9. `summary.json` báo p50, p95, max thời gian và GPU đã dùng. Trên T4 **chưa đặt ngưỡng đạt**, chỉ ghi lại số. Thời gian nạp model và warm-up được ghi riêng trong `config.json`, không cộng vào mẫu đầu tiên.
10. Có một dòng trong `experiments.md` cho mỗi bộ val.
11. `lifemate/eval/score.py` đã sửa C1–C3 (mục 4.15); `score.txt` in điểm của cả hai chế độ `strict` và `canon`.
12. `ops/notebooks/kaggle_run.ipynb` chạy từ đầu đến cuối trên một phiên Kaggle mới, với code đưa lên bằng `ops/scripts/pack_code.sh`. `config.json` có `code_version` khác `"unknown"`, có `dtype` thật đã dùng, và `artifacts/runs/` được lưu lại theo `persist`.
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
| #11 profile môi trường, envinfo, ops/requirements/notebook.txt, pack_code.sh, 2 notebook, T19–T23 | 4 giờ |
| **Tổng** | **khoảng 28,5 giờ làm, 2 giờ máy** |

---

## 9. Rollback

- Mọi thay đổi nằm trong file mới (`lifemate/`, `configs/prompts/`, `configs/`, `ops/requirements/`, `ops/scripts/`, `ops/notebooks/`, `tests/`, `pyproject.toml`). Bỏ v0 chỉ cần xóa các file đó hoặc revert commit.
- `configs/splits/train.txt`, `val_seen.txt`, `val_unseen.txt` **không đổi nội dung** trong v0; chỉ thêm `smoke.txt`. `lifemate/eval/score.py` và `lifemate/eval/make_splits.py` chỉ sửa theo mục 4.15 và để thêm `--smoke`; chế độ mặc định `strict` của `score.py` giữ nguyên cách chấm cũ trừ trường hợp `bool` và số (C1). Muốn quay lại thì revert commit đó.
- `artifacts/runs/` không vào git. Xóa một lần chạy hỏng là xóa thư mục của lần chạy đó và dòng tương ứng trong `artifacts/runs/index.csv`.

## 10. Ngoài phạm vi v0

- ASR thật, normalizer, fine-tune và Docker (mục 6).
- Ép schema chi tiết theo từng tool, few-shot trong prompt, lọc bớt tool khi prompt quá dài.
- Chạy nhiều mẫu song song.
- `compare.py`, `show.py` (phụ lục D.8, bước 3).
- Kiểm tra lại các nhãn đáng ngờ, ví dụ `train_1606`.

## 11. Câu hỏi còn mở

| # | Câu hỏi | Ảnh hưởng | Cách giải quyết |
|---|---|---|---|
| Q1 | Hướng dẫn Docker chi tiết của BTC: lệnh chạy, đường dẫn kết quả tuyệt đối hay tương đối | Mục 6.3 | Chờ tài liệu riêng của BTC. Entrypoint viết sao cho đổi đường dẫn dễ |
| Q2 | ~~Chọn LLM và ASR~~ Đã chọn 27/10/2026 (mục 6.0, phụ lục E.5). Còn lại: BTC đếm tham số Qwen3.5-4B theo tổng checkpoint hay chỉ phần ngôn ngữ; giấy phép phi thương mại của ChunkFormer / VieGem có được chấp nhận không | A/B sau v0 | Hỏi BTC; trong lúc chờ chỉ nộp bằng bộ mặc định |
| Q3 | vLLM phiên bản được khóa có chạy được trên cả T4 (dev) và sm80+ (chấm) không; các tên API mà `VLLMBackend` dùng (`LLM(...)`, `SamplingParams(structured_outputs=StructuredOutputsParams(json=...))` với fallback `guided_decoding=GuidedDecodingParams`, `LLM.chat(..., chat_template_kwargs=)`, `RequestOutput.prompt_token_ids`) mới chỉ được test với engine giả | #4, Docker | Chạy thử `--limit 30` trên cả hai loại GPU ngay khi chọn model; sửa `_structured_kwargs` nếu API khác |
| Q4 | Tổng token của prompt với mẫu nhiều tool nhất (59 tool ở public) | Có cần lọc tool không | Đo bằng `HFTokenCounter` trên mọi mẫu train và public trước khi chạy v0 |

---

## Phụ lục A. Pipeline

### A.1. Sơ đồ

```
              ┌──────────── làm cho từng mẫu ────────────────────────────────────┐
 file .wav ──►│ [A] ASR ──► "Chuyển cho Huy…"                                    │
              │                    │                                             │
 context ────►│ [B] Dựng prompt ◄──┘   (câu nói + ngữ cảnh + hướng dẫn app)      │
 catalog ────►│         │                                                        │
              │         ▼                                                        │
              │ [C] Vòng lặp agent:  LLM ──đề xuất call──► code ──env.call()──►  │
              │                       ▲                    zaic_tools (app giả)  │
              │                       └──────── kết quả ◄──────────┘             │
              │         │ (LLM nói "xong")                                       │
              │         ▼                                                        │
              │ [D] env.get_tool_calls() + env.get_state_hash()                  │
              └──────────────────────────┬───────────────────────────────────────┘
                                         ▼
                          một dòng trong submission.jsonl
```

- **[A] ASR:** chuyển audio của câu cuối và của các lượt cũ trong history thành chữ.
- **[B] Dựng prompt:** ghép câu nói, ngữ cảnh (giờ, vị trí, hồ sơ người dùng), history và hướng dẫn sử dụng của các app đã cài (lấy từ catalog).
- **[C] Vòng lặp agent:** LLM chỉ sinh ra chữ. Code đọc chữ đó, gọi `env.call()`, rồi đưa kết quả lại cho LLM. Lặp đến khi LLM báo xong.
- **[D] Ghi kết quả:** `env` tự ghi lại mọi call; code lấy danh sách call và `state_hash` để ghi thành một dòng nộp bài.

### A.2. Ví dụ từng lượt (`train_0874`)

Câu nói: _"Chuyển cho Huy một trăm rưỡi qua ví giúp em nha, nội dung ghi là tiền vé bóng đá."_

| Lượt | LLM sinh ra | Code làm gì | Kết quả đưa lại cho LLM |
|---|---|---|---|
| 1 | `{"action": "call", "name": "contacts.search", "arguments": {"query": "Huy"}}` | `env.call("contacts.search", {"query": "Huy"})` | `{"ok": true, "contacts": [{"id": "c_07", "name": "Huy"}]}` |
| 2 | `{"action": "call", "name": "ewallet.transfer", "arguments": {"contact_id": "c_07", "amount": 150000, "note": "tiền vé bóng đá"}}` | `validate` trước, hợp lệ thì `env.call(...)` | `{"ok": true, "done": true}` |
| 3 | `{"action": "done"}` | Dừng vòng lặp | |

Hai tình huống còn lại dùng cùng vòng lặp:

- **Thiếu thông tin:** LLM gọi `sys.ask_missing_info(tool_name, params)`; vòng lặp dừng ngay sau call đó.
- **Ngoài phạm vi:** LLM trả `{"action": "done"}` ngay từ lượt đầu mà không gọi gì, kết quả là `[]`.

LLM chỉ sinh JSON theo đúng một schema do vLLM ép (structured outputs), nên không có lỗi định dạng. Chi tiết ở mục 4.7.

### A.3. Ràng buộc phải giữ khi thiết kế

| Ràng buộc | Ảnh hưởng đến thiết kế |
|---|---|
| Toàn hệ thống < 6B tham số, LLM < 5B | Chọn ASR và LLM nhỏ, ví dụ Whisper/PhoWhisper bản nhỏ + LLM 1,5B – 4B |
| Context LLM ≤ 16K token | Phải nén hướng dẫn tool; mỗi mẫu cài trung bình khoảng 31 tool, nhiều nhất 62 |
| Chạy offline | Mọi model nằm sẵn trong Docker |
| ≤ 30 giây mỗi mẫu trên 1 GPU A30 (24 GB) | Giới hạn số vòng lặp; đo thời gian sớm |
| Giải mã tất định | `temperature = 0` hoặc greedy, cố định seed |
| Chỉ truy cập dữ liệu qua `zaic_tools`, đúng một `ToolEnv` mỗi mẫu | Không đọc `state.json`; không tạo lại `ToolEnv` để thử lại |
| `zaic_tools` khác nhau theo split | Luôn nạp `zaic_tools` từ thư mục của split đang chạy (`/data` khi chạy Docker), không copy ra ngoài |

## Phụ lục B. Bốn chỗ cải thiện

| # | Chỗ cải thiện | Sửa cái gì | Có phải fine-tune không | Chi phí |
|---|---|---|---|---|
| 1 | Code điều khiển | `validate` trước `call`, ép LLM sinh JSON đúng định dạng, chặn gọi lặp, giới hạn số vòng, chuẩn hóa chữ do ASR nghe ra, nén tool | Không | Rẻ |
| 2 | System prompt | Quy tắc, ví dụ mẫu (few-shot), cách trình bày tool | Không: model giữ nguyên | Rẻ |
| 3 | Fine-tune LLM | Các con số bên trong LLM (thường chỉ train miếng dán LoRA) | Có | Đắt hơn |
| 4 | Fine-tune ASR | Các con số bên trong model ASR | Có | Đắt hơn |

### B.1. Thứ tự làm

1. Agent đơn giản nhất chạy được, dùng transcript có sẵn (`audio_text`) để tạm bỏ qua ASR. Mục đích là có điểm khởi đầu.
2. Cải thiện code điều khiển và system prompt.
3. Thử vài LLM có sẵn dưới 5B, giữ bản tốt nhất.
4. Thay transcript bằng chữ do ASR nghe ra, đo mức tụt điểm.
5. Fine-tune LLM nếu bảng lỗi cho thấy quyết định sai hoặc điền sai tham số nhiều.
6. Fine-tune ASR nếu mức tụt điểm ở bước 4 lớn.
7. Đóng gói Docker: bản chạy được đầu tiên trước 05/11/2026, bản cuối trước 16/11/2026 (phụ lục F).

Hạ tầng đo lường (phụ lục D) làm cùng lúc với bước 1, không tách thành việc riêng.

### B.2. Dữ liệu dùng cho fine-tune

- **LLM:** dựng lại mỗi mẫu train thành một cuộc hội thoại nhiều lượt, gồm prompt, call đúng (`label.tool_calls[i]`), kết quả (phát lại call qua `zaic_tools` để khớp từng byte với lúc chạy; `label.tool_results[i]` chỉ dùng để đối chiếu), call tiếp theo, …, `{"action": "done"}`. Chỉ tính loss trên các lượt của trợ lý. Chi tiết ở mục 6.2. Model học được kỹ năng chung (đọc hướng dẫn, tra cứu lấy ID trước, đổi số nói bằng chữ, khi nào hỏi lại, khi nào không làm gì), chứ không học thuộc từng app.
- **ASR:** dùng cặp (file wav, `audio_text`), gồm cả các lượt user trong history.
- **Dữ liệu tổng hợp:** được tạo bằng model lớn hoặc TTS trong lúc phát triển, nhưng không được dùng chúng khi chạy bài thi. Không gán nhãn `public_test` hay `private_test`.

## Phụ lục C. Đánh giá

### C.1. Vì sao phải tự đánh giá trên train

- `public_test` không có đáp án, nên chỉ xem được điểm trên leaderboard, tối đa 5 lần mỗi ngày.
- Thứ hạng chung cuộc chỉ tính theo `private_test`. Bộ này phát ra sau khi giải pháp đã đóng băng và nhiều khả năng có app chưa từng thấy (public đã có 11 app train không có).
- Vì vậy cần một bộ đánh giá local mô phỏng tình huống gặp app lạ.

### C.2. Cách chia train (`lifemate/eval/make_splits.py`)

Cách chia theo team lead (ThanhDanh1510/ZaloAI2026, `split.py`@7033c9f), có sửa lỗi rò rỉ của bản gốc (28/10/2026).

| Phần | Số mẫu | Cách lấy | Đo cái gì |
|---|---|---|---|
| `val_unseen` | 178 | Tách ra **đầu tiên**: mọi mẫu có đáp án gọi tới 6 app giữ lại, cộng 5 mẫu gold `[]` mà yêu cầu nói về các app đó (gắn tay trong `UNSEEN_OUT_OF_SCOPE`) | Khả năng gặp app lạ, gần với điểm private |
| `val_seen` | 274 | Fold 0 trong 5 fold của phần còn lại | Khả năng với app quen |
| `train` | 1.093 | Fold 1 đến 4 của phần còn lại | Dùng để học, lấy ví dụ mẫu cho prompt |
| `smoke` | 30 | Lấy phân tầng từ `train` | Chạy thử nhanh sau mỗi lần sửa |

- **6 app giữ lại:** medication, fitness, flights, bike_rental, library, subscriptions (đại diện các nhóm sức khỏe, du lịch, di chuyển, cá nhân, tài chính).
- **5 fold:** câu gần trùng (giống nhau từ 85% theo difflib) được gom thành cụm và cả cụm nằm chung một fold, để `val_seen` không có câu gần giống `train`. Các fold được cân bằng theo số lời gọi (0, 1, 2-3, từ 4) và có hay không `sys.ask_missing_info`.
- **Sửa lỗi của bản gốc:** bản của lead chia fold trên toàn bộ mẫu rồi mới tách app giữ lại, nên 137/173 mẫu `val_unseen` lọt vào tập train. Ở đây app giữ lại bị tách trước, `meta.json` ghi kết quả kiểm tra rò rỉ (phải bằng 0) và có test bảo vệ.
- **Quy tắc bắt buộc:** nhiều mẫu train có *cài* một trong 6 app giữ lại (dù không dùng). Khi dựng prompt cho các mẫu này phải **xóa 6 app đó khỏi `installed_apps`**, nếu không model sẽ đọc thấy hướng dẫn của chúng và chúng không còn "lạ". Mẫu train có *history* gọi tool của app giữ lại bị bỏ khi dựng dữ liệu fine-tune (mục 4.5).
- **Khi chỉ làm prompt (chưa fine-tune):** không lấy ví dụ mẫu từ 6 app giữ lại và không viết quy tắc riêng cho chúng.
- **Không có trong git:** danh sách id suy ra từ dữ liệu thi nên bị bỏ qua trong `.gitignore`. Lệnh `make splits` sinh lại giống từng byte (seed 42); kernel Kaggle và script RunPod tự chạy lệnh này. Mọi thí nghiệm dùng chung một cách chia để điểm so sánh được với nhau. Điểm đo trên cách chia cũ (trước 28/10) không so được với điểm mới.

### C.3. Bộ chấm local (`lifemate/eval/score.py`)

Bộ chấm chạy lại các call đã nộp bằng `zaic_tools` của split, rồi tính điểm theo công thức chính thức (mục 5 trong `overview.md`).

```
python -m lifemate.eval.score submission.jsonl --ids configs/splits/val_seen.txt
python -m lifemate.eval.score submission.jsonl --ids configs/splits/val_unseen.txt
python -m lifemate.eval.score submission.jsonl --ids configs/splits/val_seen.txt --details details.jsonl
```

**Đã kiểm chứng:** nộp chính đáp án được 100 điểm; nộp toàn `[]` được 7,18 điểm (đúng bằng 111 mẫu ngoài phạm vi / 1.545).

**Hai chỗ luật chấm không nói rõ, bộ chấm phải tự giả định:**

1. **Cách tách từ khi tính token-F1 cho tham số fuzzy:** tách theo chữ và số sau khi chuyển chữ thường và chuẩn hóa NFC, giữ nguyên dấu tiếng Việt, tính F1 trên tập từ có lặp.
2. **"Identical" cho tham số thường:** so giống hệt, phân biệt chữ hoa chữ thường; hai bên phải có cùng bộ tên tham số.

Nếu BTC chấm khác hai điểm này, điểm local có thể lệch nhẹ so với leaderboard.

### C.4. Cách đọc báo cáo

| Bảng | Trả lời câu hỏi | Ví dụ cách dùng |
|---|---|---|
| Điểm theo loại đáp án | Loại câu nào đang mất điểm? | Điểm `ask_missing_info` thấp thì xem lại quy tắc hỏi lại |
| Confusion matrix | Agent chọn đúng "làm / hỏi lại / `[]`" không, nhầm sang đâu? | Ô "gold = hỏi lại, agent = làm" lớn nghĩa là agent hay tự đoán khi danh bạ trùng tên |
| Loại lỗi | Các mẫu cần hành động sai vì sao? | Nhiều "đúng tool nhưng sai tham số" thì xem cách đổi giờ, số tiền, ID |
| Điểm theo app | App nào hay làm sai? | Calendar thấp thì kiểm tra cách xử lý ngày giờ |

**So `val_seen` với `val_unseen`:**

- Hai điểm gần nhau: agent thật sự đọc được hướng dẫn của app lạ.
- `val_unseen` thấp hơn nhiều: agent đang học thuộc app quen. Khi đó so bảng loại lỗi của hai bộ để biết nguyên nhân:
  - Lỗi chủ yếu là "đúng tool nhưng sai tham số": agent hiểu app lạ làm gì nhưng điền sai, nên cải thiện cách trình bày schema.
  - Lỗi chủ yếu là "gọi nhầm tool" hoặc quyết định sai: agent không hiểu app lạ làm gì, nên thêm mô tả app hoặc ví dụ cách đọc catalog.
- Ngưỡng "gần" hay "chênh nhiều" chưa chốt; sẽ đặt sau khi có vài lần chạy thật để biết độ dao động. `val_unseen` chỉ có 178 mẫu nên điểm có thể lệch vài điểm giữa các lần chạy.

### C.5. Kiểm tra trước khi nộp public_test

Không chấm được điểm public_test ở local, nhưng trước khi nộp nên kiểm tra:

1. **Schema:** `python -m zaic_tools --split public_test --root . --validate submission.jsonl`, chạy với `PYTHONPATH=public_test` từ thư mục gói dữ liệu.
2. **Đủ dòng:** đúng 634 dòng, không trùng `id`.
3. **Phân bố so với train:** tỉ lệ `[]` quanh 7%, tỉ lệ hỏi lại quanh 14%, số call trung bình quanh 2,3. Lệch quá nhiều là dấu hiệu agent hành xử bất thường.
4. **Đọc vài mẫu bằng mắt** để tìm lỗi. Không dùng để chấm điểm hay chọn phương án, vì tự gán nhãn public_test để tinh chỉnh là bị cấm.

## Phụ lục D. Hạ tầng đo lường

Mục tiêu: mỗi lần đổi prompt, code hay model đều đo được điểm, thời gian, token và biết sai ở mẫu nào. Mọi thứ lưu thành file, chưa cần MLflow.

### D.1. Mỗi lần chạy là một thư mục

Cấu trúc thư mục `artifacts/runs/<name>/` (`config.json`, `trace.jsonl`, `submission.jsonl`, `score.txt`, `details.jsonl`, `summary.json`) ở mục 4.12. Mỗi lần chạy thêm một dòng vào `artifacts/runs/index.csv` một cách tự động; `experiments.md` chỉ chép lại các lần chạy đáng nhớ. `artifacts/runs/` không đưa vào git vì chứa nội dung dữ liệu thi.

### D.2. Logging (`trace.jsonl`)

Mỗi mẫu ghi:

- Chữ ASR nghe ra (và `audio_text` nếu có).
- Từng lượt LLM: số token prompt, số token sinh ra, thời gian, đầu ra thô, call đã tách được.
- Từng call: tham số, kết quả `env` trả về, có bị từ chối không.
- Lỗi (nếu có), tổng thời gian, `state_hash`.

### D.3. Phân loại lỗi

Mỗi mẫu bọc riêng trong `try/except` ở `Runner`: một mẫu hỏng thì vẫn ghi kết quả cho mẫu đó và chạy tiếp, không bao giờ dừng cả lần chạy. Danh sách mã lỗi (`ErrorCode`) và lý do dừng (`StopReason`) hiện hành ở mục 4.13. Cuối mỗi lần chạy, `summary.json` đếm từng loại lỗi kèm tối đa 10 id mẫu.

### D.4. Thời gian, token và điểm (`summary.json`)

- **Thời gian:** tách ASR, LLM (cộng các lượt), tool và tổng mỗi mẫu; báo p50, p95, max và số mẫu vượt 30 giây. Ghi kèm loại GPU: T4 (Kaggle) chậm hơn A30 khoảng 2 – 3 lần, nên trên T4 lấy ngưỡng cảnh báo khoảng 60 giây.
- **Token:** số token prompt lớn nhất so với giới hạn 16K; cảnh báo khi vượt khoảng 14K.
- **Điểm:** score tổng và score theo loại đáp án, tách riêng `val_seen` và `val_unseen`.

### D.5. Benchmark và so sánh

- **Bộ cố định:**
  - `smoke`: `configs/splits/smoke.txt`, 30 mẫu lấy phân tầng từ `val_seen`, chạy vài phút, dùng sau mỗi lần sửa.
  - `val_seen` và `val_unseen`: chạy đầy đủ.
- **`compare.py runA runB`:** liệt kê mẫu trước đúng nay sai và ngược lại. Một con số trung bình không cho thấy điều này.
- **`show.py <run> <id>`:** in trace của một mẫu dạng hội thoại, dễ đọc hơn JSON thô.
- **Đo riêng ASR:** WER/CER trên các cặp (wav, `audio_text`) của train. So điểm agent khi dùng `audio_text` và khi dùng chữ ASR; khoảng chênh là phần mất do ASR.

### D.6. Test (pytest, chạy trên CPU, không cần model)

- **Scorer:** chấm đáp án ra 100, nộp toàn `[]` ra 7,18.
- **Tách đầu ra LLM:** một bộ đầu ra lỗi mẫu (thiếu ngoặc, thừa chữ, JSON sai).
- **Dựng prompt:** đã xóa 6 app giữ lại khỏi `installed_apps`; không vượt ngân sách token.
- **Vòng lặp với LLM giả (FakeLLM):** LLM giả trả đúng đáp án, điểm phải ra 100. Kiểm tra cả vòng lặp mà không cần GPU.
- **Tính tất định:** chạy cùng 5 mẫu hai lần, kết quả giống hệt nhau.

### D.7. Quản lý phiên bản

| Thành phần | Cách quản lý |
|---|---|
| Code | git; mỗi lần chạy ghi commit hash vào `config.json` |
| Prompt | file trong `configs/prompts/`, nằm trong git; mỗi lần chạy ghi hash prompt |
| Dữ liệu fine-tune | `data/sft/v1.jsonl`, `v2`, … kèm hash và script tạo ra |
| Model fine-tune | tên dạng `ft-qwen4b-sftv1-r01`, kèm `train_config.json` (model gốc, phiên bản dữ liệu, siêu tham số, seed, commit, loss); lưu trên Kaggle Models hoặc kho HF private, không đưa vào git |
| Đường cong loss | tùy chọn W&B, không bắt buộc |
| Lần nộp public | `runs/public_submissions.csv`: ngày, run đã nộp, điểm public. Theo dõi điểm val và điểm public có tăng giảm cùng chiều không; nếu không, bộ val chưa phản ánh đúng bài thi |

### D.8. Thứ tự xây

1. `config` + `trace` + `summary` + phân loại lỗi: viết cùng agent v0.
2. Test scorer và FakeLLM.
3. `compare.py`, `show.py`: khi đã có từ hai lần chạy trở lên.
4. Quản lý phiên bản fine-tune: khi bắt đầu fine-tune.

## Phụ lục E. Nộp bài và tuân thủ

### E.1. Docker

- BTC mount thư mục test (cùng cấu trúc `public_test`) vào **`/data`**, chạy inference script, lấy kết quả tại **`result/submission.jsonl`** (`overview.md` mục 7). Chi tiết còn lại theo hướng dẫn riêng của BTC.
- Trọng số model nằm sẵn trong image; không tải gì lúc chạy.
- Chạy thử bằng `docker run --network none` để chắc không có thư viện nào ngầm tải file (tokenizer, ASR, …).
- Đo thời gian nạp model và thời gian từng mẫu trên GPU 24 GB thuê theo giờ (Kaggle/Colab không chạy được Docker).
- Tạo checksum và link Google Drive công khai theo yêu cầu BTC. Được nộp tối đa hai (02) lần.

### E.2. Training code và data phải chạy lại được

BTC yêu cầu nộp cả training code, inference code và data. Fine-tune không được chỉ nằm trong notebook: phải có script chạy từ đầu đến cuối (tạo dữ liệu → train → merge → model).

### E.3. Tự kiểm tra luật thi

Viết thành kiểm tra tự động trong code chạy:

- Đúng một `ToolEnv` mỗi mẫu.
- Mọi call đã thực hiện trên `ToolEnv` đều có trong submission.
- Không mở file nào trong `*_environment/`.
- File nộp: đủ mọi id, không trùng, JSON hợp lệ, đúng schema (phụ lục C.5).

### E.4. Chốt chặn thời gian

Agent tự theo dõi thời gian: khi một mẫu chạy khoảng 25 giây thì dừng vòng lặp và trả về những call đã có. Như vậy mẫu đó vẫn có thể được một phần điểm, thay vì bị 0 vì quá 30 giây.

### E.5. Danh sách model được dùng

Chọn ngày 27/10/2026 theo điều kiện ở mục 6.0. Số tham số lấy từ metadata safetensors trên HF.

| Model | Vai trò | Số tham số | Ngày phát hành | Giấy phép | Revision |
|---|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | LLM (mặc định) | 4,02B | 05/08/2025 | Apache-2.0 | `cdbee75f` |
| `Systran/faster-whisper-large-v3` | ASR (mặc định từ 28/10) | 1,55B | 11/2023 | MIT | `edaa852e` |
| `vinai/PhoWhisper-medium` | ASR (mặc định tới 27/10, vẫn chạy được với `engine = "hf_seq2seq"`) | 0,77B | 18/02/2024 | BSD-3 | `55a7e3eb` |

Tổng 5,57B < 6B (với PhoWhisper là 4,79B), LLM 4,02B < 5B. Ứng viên A/B (chưa dùng khi nộp): LLM `Qwen/Qwen3.5-4B` (4,66B kể cả vision encoder, Apache-2.0, 02/03/2026); ASR `nvidia/parakeet-ctc-0.6b-Vietnamese` (0,6B, NVIDIA Open Model License, 15/01/2026, engine `nemo_ctc`), `khanhld/chunkformer-ctc-large-vie` (0,11B, CC-BY-NC-4.0, 02/2025, engine `chunkformer`; giấy phép phi thương mại cần BTC xác nhận trước khi nộp) và `khanhld/chunkformer-rnnt-large-vie` (0,11B, CC-BY-4.0, 10/2025, engine `chunkformer`), đang được so cùng phương pháp với large-v3 và PhoWhisper (`lifemate/eval/asr_tune.py`). VieGem-ASR đã bỏ khỏi code ngày 28/10/2026 (`decisions.md` D37).

#### E.5.1. Vì sao chọn như vậy

**LLM.** Không có LLM ≤ 5B nào phát hành năm 2026 được huấn luyện riêng cho tiếng Việt và tool calling (Sailor2 1B/8B, SeaLLMs-v3 1.5B, PhoGPT-4B đều cũ hơn và không có tool calling). Trong họ Qwen, bản Instruct-2507 được chọn thay bản Thinking-2507 (BFCL-v3 61,9 so với 71,2) vì:

- Ngân sách 30 giây/mẫu đã gồm ASR cho tối đa 13 file audio và 3–6 bước gọi tool. Mỗi bước thinking sinh thêm vài trăm đến vài nghìn token không ép được bằng JSON schema, nên thời gian không đoán trước được (quyết định S6).
- Nhãn SFT là chuỗi tool call trực tiếp, không có lời suy nghĩ. Fine-tune bản Thinking bằng dữ liệu không có think sẽ dạy model bỏ think, mất luôn lợi thế của bản đó.
- Instruct-2507 là Transformer chuẩn, chạy trên vLLM ≥ 0,8.5 và fp16 trên T4; Qwen3.5-4B dùng Gated DeltaNet, cần vLLM ≥ 0,17 (bản khóa 0,32 đủ) nhưng chưa rõ fp16/T4, và 4,66B sát trần 5B nếu BTC đếm cả vision encoder. Nên Qwen3.5-4B chỉ là ứng viên A/B sau khi có baseline.

**ASR (hiện hành, 28/10/2026).** ASR mặc định là whisper-large-v3 bản CTranslate2 (faster-whisper): beam 5, best_of 5, VAD, `condition_on_previous_text = false`, **không có câu mồi** (`initial_prompt`) và không có tham số chống lặp (`repetition_penalty`, `no_repeat_ngram_size`); sau đó qua bộ lọc ảo giác `lifemate/backends/asr_sanitizer.py`. Team lead đo 98% lỗi nặng của Whisper là câu ảo giác kiểu phụ đề YouTube, nên bộ lọc là phần quan trọng. Đo trên `val_seen` (274 mẫu, 916 câu, ASR sweep v1): bỏ câu mồi và tham số chống lặp đưa WER từ 44,0% xuống 29,7%, số câu rỗng từ 190 xuống 24, tỉ lệ bỏ sót thực thể từ 52% xuống 31%. PhoWhisper-medium cùng bộ đo: WER 27,9%, tỉ lệ bỏ sót thực thể 38,5%, p95 2,9 s so với 7,6 s của large-v3; large-v3 được giữ vì bỏ sót thực thể ít hơn, và lần chạy baseline đầy đủ sẽ quyết định cuối (`decisions.md`, `experiments.md`). Đoạn dưới là lý do chọn PhoWhisper lúc đầu (27/10).

**ASR (27/10/2026).** PhoWhisper-medium được chọn mặc định vì cắm thẳng vào `HFSeq2SeqASR` (transformers pipeline) không cần code thêm, giấy phép BSD, và đầu ra chữ thường, số viết bằng chữ, cùng văn phong với `audio_text`. Điểm yếu: WER khi xen tiếng Anh cao (PhoWhisper-large 31% trên ViMedCSS) và Whisper hay bịa chữ ở đoạn im lặng. ChunkFormer (N-WER 6,89, tốt nhất trong bài EACL 2026, nhẹ 110M) và VieGem-ASR (WER trung bình 6,34, xen tiếng Anh 14,3%) tốt hơn trên benchmark công khai nhưng cần adapter và vướng giấy phép phi thương mại. Cả ba được đo trên `val_seen` / `val_unseen` theo mục 6.4.2 trước khi chốt.

#### E.5.2. Kế hoạch fine-tune cho từng model

**LLM: Qwen3-4B-Instruct-2507, QLoRA bằng ms-swift (mục 6.2). Cập nhật 28/10/2026 theo team lead.**

| Việc | Cách làm | Vì sao |
|---|---|---|
| Phương pháp | QLoRA: model gốc nf4, LoRA r=32, alpha=64 trên 7 ma trận attention + MLP; train bằng ms-swift 4.6.0 (`swift sft`), fp16 trên T4. Trước mỗi lần train, so token và nhãn loss của ms-swift với chat template của vLLM | Full fine-tune 4B cần > 40 GB VRAM, không vừa T4/Kaggle. LoRA trên cả attention và MLP giữ được kiến thức chung, chỉ dạy định dạng và quy tắc gọi tool. r=32 theo lead; bộ cũ r=16 (ít overfit hơn trên dữ liệu nhỏ) để so ở lượt đầu |
| Dữ liệu | 1093 mẫu `configs/splits/train.txt` dựng bằng đúng `PromptBuilder` lúc chạy, xóa 6 app giữ lại; transcript trộn 70% chữ ASR và 30% `audio_text` (mẫu ASR mất tên hoặc số của đáp án thì dùng `audio_text`); 3% mẫu phục hồi lỗi (lượt gọi sai không tính loss); kết quả tool phát lại qua `zaic_tools` | Prompt lúc train và lúc chạy phải giống từng byte, nếu không model học một phân bố và chạy trên phân bố khác. Xóa app bị rút để `val_unseen` vẫn đo được khả năng đọc app lạ |
| Che loss | Chỉ trên token của lượt assistant | Không che thì model học thuộc mô tả tool và kết quả tool, làm hỏng khả năng đọc tool docs mới |
| Chat template | Render với tham số như lúc chạy; Instruct-2507 không có chế độ thinking nên không cần `enable_thinking` | Giữ đúng token đặc biệt mà vLLM sẽ sinh lúc chạy |
| Siêu tham số ban đầu | lr 2e-4 cosine, warmup 5%, 3 epoch, batch 1 × grad_accum 16, max_len 10240, seed 0 | Theo lead; max_len theo prompt dài nhất lead đo được (9.544 token). Bộ cũ (2 epoch, grad_accum 8, max_len 16384) để so. Chưa đo trên pipeline này |
| Chọn checkpoint | 2 checkpoint có loss `eval_unseen` thấp nhất, merge, chạy agent thật trên chữ ASR thật (`--transcripts asr`), lấy điểm `val_unseen` cao nhất | Loss không phản ánh đúng điểm thi; nhưng chỉ chạy agent thật 2 lần để không chọn model theo val quá nhiều |
| Đưa vào hệ thống | Merge adapter vào bf16, lưu safetensors, trỏ `[llm].model` (Docker: `/app/models/llm`) | vLLM nạp model đã merge như model thường, không tốn thêm thời gian chạy, Docker đơn giản |
| Bước tiếp theo (lộ trình của lead) | Lần lượt bật: nhân bản ca khó 1,5-2 lần (`oversample_hard`), app gây nhiễu (`distractor_apps`), lọc cụm câu trùng (`dedup_max_k`); mỗi kỹ thuật một lượt train riêng | Đo riêng từng kỹ thuật để biết cái nào có ích |

**ASR: full fine-tune hai giai đoạn (mục 6.4). Chỉ làm khi điểm mất do ASR ≥ 3 hoặc tỉ lệ nghe sai thực thể ≥ 10%.** Kế hoạch dưới viết cho PhoWhisper-medium (27/10). Với whisper-large-v3 (1,55B), fine-tune làm trên bản transformers rồi chuyển sang CTranslate2 bằng `ct2-transformers-converter`; full fine-tune 1,55B không vừa T4, nên phải dùng LoRA hoặc GPU lớn hơn.

| Việc | Cách làm | Vì sao |
|---|---|---|
| Phương pháp | Full fine-tune, không LoRA. Giai đoạn 1: đóng băng encoder, train decoder. Giai đoạn 2 (nếu WER còn cao): mở encoder với lr nhỏ hơn 10 lần | 0,77B vừa T4 với fp16 + gradient checkpointing. Decoder quyết định cách viết số, tên riêng, từ tiếng Anh, là thứ cần chỉnh để khớp `audio_text`; encoder đã học 844 giờ tiếng Việt, mở sớm với 3547 cặp dễ làm hỏng |
| Dữ liệu | 3547 cặp (wav, `audio_text`) từ lượt cuối và mọi lượt user trong history của `configs/splits/train.txt`; chia theo id mẫu | `audio_text` là nhãn duy nhất có. ASR sau fine-tune đổi cách viết số, nên phải train lại LLM v2 trên chữ ASR mới |
| Tăng dữ liệu | SpecAugment; TTS chỉ cho loại lỗi mà bảng đo chỉ ra (tên người, địa danh, số tiền, từ tiếng Anh), câu lấy từ `audio_text` train rồi thay thực thể | Dữ liệu ít, SpecAugment chống overfit. TTS được phép lúc phát triển, cấm lúc thi |
| Siêu tham số ban đầu | lr 1e-5, 3 epoch, batch 8 × grad_accum 2, greedy decode, seed 0 | lr nhỏ vì full fine-tune trên model đã tốt; 3 epoch trên 3547 cặp là khoảng 660 bước |
| Chọn checkpoint | Theo tỉ lệ nghe sai thực thể trên `val_unseen`, rồi WER; checkpoint tốt nhất chạy agent thật | Lỗi thực thể là lỗi làm sai tham số tool, ảnh hưởng điểm trực tiếp; WER chung không phản ánh |
| Đưa vào hệ thống | Lưu snapshot transformers, trỏ `[asr].model` (Docker: `/app/models/asr`), giữ `engine = "hf_seq2seq"` | Cùng đường nạp với model gốc |

**Thứ tự làm, rút ra từ phân tích nguyên lý gốc (27/10/2026, lập luận đầy đủ ở `rationale.md`):**

1. Cắm ASR gốc, đo WER, tỉ lệ nghe sai thực thể, thời gian.
2. LLM v1: QLoRA trên `audio_text` gold. *(Cập nhật 28/10: gộp bước 2 và 3 thành một lượt, dữ liệu trộn 70% chữ ASR theo lead.)*
3. **LLM v2: dựng lại dữ liệu với chữ ASR thật** (`transcript_source = "asr"`), nhãn vẫn gold. Lý do: danh bạ, tên thiết bị, app đã cài nằm trong ngữ cảnh mẫu mà chỉ LLM nhìn thấy, ASR thì không. LLM train trên chữ ASR học được quy ước "chọn tên trong danh bạ gần nhất với chữ nghe được", tức sửa lỗi nghe bằng thông tin chỉ LLM có, không tốn thêm thời gian lúc chạy.
4. **Chỉ sau LLM v2** mới đo điểm mất do ASR và tỉ lệ nghe sai thực thể để quyết định fine-tune ASR. Đo sớm hơn thì con số gồm cả phần LLM v2 tự sửa được, dẫn tới tốn 1–2 ngày GPU vào thứ có thể biến mất.
5. Nếu fine-tune ASR: làm dữ liệu TTS nhắm vào ba nhóm thực thể (tên người, thiết bị và địa điểm, số tiền và giờ) với encoder đóng băng **trước** khi nghĩ tới mở encoder, vì đó là lớp rẻ, rủi ro thấp và đo được ngay; mở encoder là bước rủi ro cao nhất với 13 giờ audio.

Điểm yếu của lập luận: với địa chỉ tự do hay tên không có trong danh bạ thì LLM không có gì để đối chiếu, chỉ ASR mới cứu được; bảng lỗi sau LLM v2 sẽ cho biết nhóm này chiếm bao nhiêu.

**Nếu đổi sang một ASR CTC (ví dụ Parakeet):** mô hình CTC, không có decoder tự hồi quy, nên fine-tune là train CTC loss trên cùng 3547 cặp với encoder đóng băng phần lớn . Cần viết thêm script train riêng vì `lifemate/train/asr.py` hiện chỉ hỗ trợ seq2seq của transformers. Chỉ làm khi số đo cho thấy hai model này thắng PhoWhisper trên `val_unseen` **và** BTC chấp nhận giấy phép phi thương mại.

### E.6. Vòng private (20/11 – 23/11)

Đội tự chạy giải pháp đã nộp trên private test trong 3 ngày. Chuẩn bị trước:

- Máy chạy: GPU thuê nào, thuê từ khi nào.
- Ước lượng tổng thời gian chạy toàn bộ private test.
- Một lệnh duy nhất từ dữ liệu ra `submission.jsonl`; chạy tập dượt trên public_test trước ngày 16/11.

### E.7. Bảo mật

- Dữ liệu thi chỉ để ở chỗ riêng tư (Kaggle Dataset ở chế độ Private, Drive cá nhân); không đưa lên git.
- Không ghi HF token hay khóa API vào code hoặc notebook.

## Phụ lục F. Mốc thời gian (đề xuất)

| Thời gian | Việc cần xong |
|---|---|
| 27/10 – 30/10 | Agent v0 dùng `audio_text` + hạ tầng phụ lục D.8 bước 1 – 2; có điểm đầu tiên trên `val_seen`, `val_unseen` |
| 31/10 – 03/11 | Cải thiện prompt và code điều khiển; thêm ASR, đo WER và mức tụt điểm |
| trước 05/11 | Docker chạy được offline trên GPU thuê; nộp public lần đầu từ chính Docker đó |
| 05/11 – 12/11 | Fine-tune LLM (QLoRA), fine-tune ASR nếu cần; script train chạy lại được |
| 13/11 – 15/11 | Chốt phiên bản, kiểm tra Docker lần cuối, tập dượt quy trình vòng private |
| 16/11, 23:59 | Nộp Docker (tối đa hai lần) |
| 20/11 – 23/11 | Chạy private test, nộp kết quả trước 23:59 ngày 23/11 |
