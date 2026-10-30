# Report

Tài liệu này giải thích mã nguồn của repo JSON-lifemate: repo làm gì, một mẫu thi đi qua những đoạn mã nào, mỗi thư mục
và mỗi file giữ việc gì, vì sao mã được viết như vậy, và những gì còn chưa làm. Mục tiêu là đọc xong thì giải thích được
cho người khác trong team mà không phải mở từng file.

Ngày viết: 30/10/2026, theo mã trên nhánh `asr-benchmark-v2`. Các tài liệu liên quan:


| Muốn biết                                  | Đọc                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| Đặc tả đầy đủ, quyết định thiết kế ban đầu | [`docs/SPEC.md`](SPEC.md)                                                            |
| Thể lệ thi                                 | [`docs/overview.md`](overview.md)                                                        |
| Các quyết định đã chốt và lý do            | [`docs/decisions.md`](decisions.md)                                                       |
| Kết quả benchmark ASR                      | [`docs/reports/benchmark/asr_v1.md`](reports/benchmark/asr_v1.md), [`docs/reports/benchmark/asr_v2.md`](reports/benchmark/asr_v2.md), [`docs/reports/benchmark/asr_v3.md`](reports/benchmark/asr_v3.md)              |
| Mô tả ngắn từng thư mục                    | `README.md` trong mỗi thư mục `lifemate/*/`, [`configs/`](../configs/), [`ops/`](../ops/), [`tests/`](../tests/) |


Cách ghi tham chiếu mã trong tài liệu: `đường/dẫn/file.py › TênHàm`, đường dẫn tính từ thư mục gốc repo.

## Mục lục

1. [Bài toán và việc repo làm](#1-bài-toán-và-việc-repo-làm)
2. [Bản đồ repo](#2-bản-đồ-repo)
3. [Hành trình của một mẫu, từng bước](#3-hành-trình-của-một-mẫu-từng-bước)
4. [Lõi chung: `lifemate/core`](#4-lõi-chung-lifematecore)
5. [Đọc và ghi dữ liệu: `lifemate/io`](#5-đọc-và-ghi-dữ-liệu-lifemateio)
6. [ASR: `lifemate/backends/asr*.py`](#6-asr-lifematebackendsasrpy)
7. [LLM: `lifemate/backends/llm.py`](#7-llm-lifematebackendsllmpy)
8. [Agent: `lifemate/agent`](#8-agent-lifemateagent)
9. [Chạy và lắp ráp: `lifemate/run`](#9-chạy-và-lắp-ráp-lifematerun)
10. [Chấm điểm và đo lường: `lifemate/eval`](#10-chấm-điểm-và-đo-lường-lifemateeval)
11. [Huấn luyện: `lifemate/train`](#11-huấn-luyện-lifematetrain)
12. [Config, ops, Docker, test](#12-config-ops-docker-test)
13. [Các nguyên tắc xuyên suốt](#13-các-nguyên-tắc-xuyên-suốt)
14. [Trạng thái hiện tại và việc còn thiếu](#14-trạng-thái-hiện-tại-và-việc-còn-thiếu)
15. [Câu hỏi hay gặp khi giải thích cho team](#15-câu-hỏi-hay-gặp-khi-giải-thích-cho-team)
16. [Bảng tra nhanh](#16-bảng-tra-nhanh)

---

## 1. Bài toán và việc repo làm

**Bài thi LifeMate** (Zalo AI Challenge 2026): người dùng nói một yêu cầu bằng tiếng Việt, đôi khi chêm tiếng Anh, ví dụ
"Đặt báo thức 5 rưỡi sáng mai". Hệ thống phải gọi đúng tool với đúng tham số, ví dụ
`alarm.create(time="2026-11-11T05:30:00+07:00")`. Điểm chấm dựa trên lời gọi tool, không chấm chữ.

Mỗi mẫu thi gồm:


| Phần                         | Dạng                                                        | Ghi chú                     |
| ---------------------------- | ----------------------------------------------------------- | --------------------------- |
| Câu cuối của người dùng      | Chỉ âm thanh (`.wav`)                                       | Yêu cầu cần thực hiện; điểm tính trên lời gọi tool đáp lại câu này |
| Các lượt người dùng trước đó | Chỉ âm thanh, mỗi lượt một file `_u1.wav`, `_u2.wav`…       | Phải tự chép; chữ chép đưa vào lịch sử hội thoại trong prompt của LLM |
| Các lượt trợ lý trước đó     | Chữ: câu trả lời, lời gọi tool, kết quả tool                | Có sẵn                      |
| `context`                    | Chữ: giờ hiện tại, múi giờ, vị trí, hồ sơ, `installed_apps` | Có sẵn                      |
| Catalog tool                 | JSON mô tả mọi app và tool của split                        | Một file cho cả split       |
| Môi trường tool              | Gói `zaic_tools` của BTC, mô phỏng các app                  | Chỉ được dùng qua `ToolEnv` |

**`ToolEnv` là gì.** `ToolEnv` là lớp Python trong gói `zaic_tools` của BTC: một "điện thoại giả" chứa dữ liệu app của
đúng một mẫu (danh bạ, lịch, báo thức, thiết bị nhà…). Mỗi mẫu tạo một `ToolEnv` bằng `ToolEnv(split, mã mẫu)`, rồi:

- `env.call(tên_tool, tham_số)`: kiểm tra lời gọi theo schema, chạy trên dữ liệu app của mẫu, trả về kết quả
  (`{"ok": true, …}`) hoặc lỗi (`{"ok": false, "error": …}`). Agent đọc kết quả này để quyết định bước tiếp.
- `env.validate(tên_tool, tham_số)`: chỉ kiểm tra schema, không chạy.
- `env.get_tool_calls()`: nhật ký mọi lời gọi đã thực hiện, kể cả lời gọi bị từ chối; đây là phần `tool_calls` của bài nộp.
- `env.get_state_hash()`: mã băm trạng thái app cuối cùng; đây là `state_hash` của bài nộp. BTC chấm bằng cách phát lại
  lời gọi trên cùng thư viện, nên chạy trên máy cho cùng kết quả như lúc chấm.

Dữ liệu app nằm trong `<split>_environment/state.json`, nhưng hệ thống không được đọc file đó; mọi thông tin về app
chỉ lấy qua `ToolEnv`.

Tên `ToolEnv` và code của nó là của BTC (`from zaic_tools import ToolEnv`); dự án không viết lại hay đóng gói lại. Phía
dự án chỉ có ba phần bao quanh:

- [`lifemate/io/zaic.py`](../lifemate/io/zaic.py) › `ZaicLoader.open_env`: nạp đúng gói `zaic_tools` của tập đang chạy
  và gọi `ToolEnv(...)` của BTC cho một mẫu; `EnvRegistry` nhớ các mẫu đã mở để mỗi mẫu chỉ có một `ToolEnv`.
- [`lifemate/core/types.py`](../lifemate/core/types.py) › `ToolEnvLike`: mô tả kiểu "thứ có `call`, `get_tool_calls`,
  `get_state_hash`", để code agent không phụ thuộc trực tiếp vào gói của BTC.
- [`tests/fakes.py`](../tests/fakes.py) › `FakeEnv`: bản giả dùng khi test trên CPU, thay cho `ToolEnv` thật.


**Ràng buộc chính của thể lệ** ([`docs/overview.md`](overview.md), mục 8):

- Mỗi mẫu xử lý trong 30 giây, tính cả ASR, LLM và tool. Quá giờ thì mẫu đó 0 điểm.
- Tổng tham số dưới 6B; LLM dưới 5B và context tối đa 16 384 token.
- Không được đọc `state.json` hay bất kỳ file nào trong `<split>_environment/`. Mỗi mẫu đúng một `ToolEnv`, mọi lời gọi
trên `ToolEnv` đó phải có trong file nộp.
- Cố định seed, giải mã tất định. Kết quả chạy lại khác thì BTC chạy 5 lần lấy trung bình.
- Chạy offline trên máy chấm: 1 GPU A30 24 GB, CPU 16 nhân, 64 GB RAM.

**Repo này làm ba việc:**

1. **Hệ thống dự thi:** từ âm thanh ra `submission.jsonl` (gói [`lifemate/agent`](../lifemate/agent), `backends`, `run`, `io`).
2. **Đo lường:** chấm điểm giống BTC, chia tập, benchmark ASR ([`lifemate/eval`](../lifemate/eval)).
3. **Huấn luyện:** dựng dữ liệu SFT và fine-tune LLM, fine-tune ASR ([`lifemate/train`](../lifemate/train)).

Mô hình đang dùng: LLM **Qwen3-4B-Instruct-2507** (4,02B) chạy bằng **vLLM**. ASR trong config hiện tại là
**whisper-large-v3** (1,55B) chạy bằng **faster-whisper**; benchmark đề xuất đổi sang ChunkFormer RNNT ghép với large-v3
(mục 14).

---

## 2. Bản đồ repo

```
JSON-lifemate/
├── lifemate/                 gói Python chính (khoảng 14 000 dòng)
│   ├── core/                 kiểu dữ liệu, config, seed, đếm token, tiện ích; không phụ thuộc phần nào khác
│   ├── io/                   đọc dữ liệu BTC, nạp zaic_tools, ghi thư mục run, kiểm tra file nộp
│   ├── backends/             các khối thay được: ASR (4 engine), LLM (vLLM, giả)
│   ├── agent/                dựng prompt, vòng lặp agent, guard, ngân sách, định dạng đầu ra
│   ├── run/                  CLI, lắp ráp đối tượng (factory), chạy từng mẫu (runner), đo giờ
│   ├── eval/                 chấm điểm, chia tập, benchmark ASR, phân tích lỗi
│   └── train/                dữ liệu SFT, fine-tune LLM (ms-swift), fine-tune ASR
├── configs/                  config chạy (v0.toml), prompt hệ thống, câu mồi ASR, profile máy, danh sách mẫu
├── ops/                      script chạy trên Kaggle, Colab; Dockerfile nộp bài
├── tests/                    khoảng 280 test pytest, chạy trên CPU
├── docs/                     SPEC, quyết định, báo cáo, tài liệu này
├── data/                     dữ liệu thi và dữ liệu sinh ra (git bỏ qua)
└── artifacts/                kết quả các lần chạy (git bỏ qua)
```

**Hướng phụ thuộc giữa các gói** (gói trên được import gói dưới, không có chiều ngược lại):

```
run  ─────►  agent  ─────►  backends  ─────►  core
  │            │                                ▲
  └──► io ─────┴────────────────────────────────┘
eval, train  ─────►  dùng lại agent, backends, io, core (eval và train không import lẫn nhau)
```

[`lifemate/core/types.py`](../lifemate/core/types.py) không import gì khác trong `lifemate`, nên mọi tầng dùng được mà không bị import vòng.

---

## 3. Hành trình của một mẫu, từng bước

**Hai định dạng hội thoại với LLM** (`[agent].format`, quyết định D50 ngày 30/10/2026):

| | `chatml` (mặc định trong [`configs/v0.toml`](../configs/v0.toml)) | `json` ([`configs/v0_json.toml`](../configs/v0_json.toml), định dạng cũ) |
|---|---|---|
| Vòng lặp | [`lifemate/agent/chatml_agent.py`](../lifemate/agent/chatml_agent.py) › `ChatMLAgent` | [`lifemate/agent/agent.py`](../lifemate/agent/agent.py) › `Agent` |
| Dựng tin nhắn | [`lifemate/agent/chatml.py`](../lifemate/agent/chatml.py), giống hệt `build_messages` của repo lead | [`lifemate/agent/prompt.py`](../lifemate/agent/prompt.py) › `PromptBuilder` + [`configs/prompts/system_v1.md`](../configs/prompts/system_v1.md) |
| Danh sách tool | Trường `tools` của tin `system` (JSON schema); chat template của Qwen chèn vào khối `<tools>` | Văn bản gọn trong nội dung tin `system` (`render_tools`) |
| Lời gọi tool | Tin `assistant` có `tool_calls`; mô hình sinh `<tool_call>{"name": ..., "arguments": ...}</tool_call>` | Nội dung tin `assistant` là JSON `{"action":"call",...}`, ép theo schema |
| Kết quả tool, lỗi guard | Tin vai `tool`; lỗi guard là `{"ok": false, "error": ...}` | Tin `user` `KẾT QUẢ: ...` hoặc `LỖI (chưa gọi tool): ...` |
| Kết thúc | Câu trả lời bằng lời (không có `<tool_call>`) | `{"action":"done"}` |
| Dữ liệu huấn luyện | [`src/preprocessing/dataset/builder.py`](https://github.com/ThanhDanh1510/ZaloAI2026/blob/main/src/preprocessing/dataset/builder.py) của repo lead | [`lifemate/train/sft_data.py`](../lifemate/train/sft_data.py) |

Hai vòng lặp có cùng ngân sách, guard, hạn giờ, cách ghi kết quả. Các bước dưới mô tả định dạng `json` (nơi mã được
viết trước); bước 3 và 4 có ghi chú phần khác của `chatml`. Thử nghiệm J của benchmark ASR lần 3 chạy ở định dạng `json`.

Phần này đi theo đúng thứ tự mã chạy khi gõ:

```
python -m lifemate.run.cli --config configs/v0.toml --env local --split train --ids configs/splits/smoke.txt
```

### Bước 0. CLI đọc config và lắp ráp hệ thống

`lifemate/run/cli.py › run`:

1. Đọc [`configs/v0.toml`](../configs/v0.toml) thành `Config` ([`lifemate/core/config.py`](../lifemate/core/config.py) › `load_config`), áp các `--set khoá=giá_trị` nếu có.
2. Đọc profile máy [`configs/env/local.toml`](../configs/env/local.toml) (`load_profile`): thư mục dữ liệu, thư mục run, nguồn token Hugging Face.
3. Đọc các mẫu của split, lọc theo `--ids` ([`lifemate/io/data.py`](../lifemate/io/data.py) › `load_split`, `read_ids`).
4. Mở thư mục run và kiểm tra `--resume` **trước** khi nạp mô hình, để lỗi config lộ ra ngay chứ không sau vài phút nạp
 vLLM.
5. Cố định seed của Python, NumPy, PyTorch ([`lifemate/core/seeding.py`](../lifemate/core/seeding.py) › `seed_everything`).
6. Gọi [`lifemate/run/factory.py`](../lifemate/run/factory.py) › `build` để tạo mọi đối tượng, rồi giao cho [`lifemate/run/runner.py`](../lifemate/run/runner.py) › `Runner`.

`factory.build` là **nơi duy nhất** tạo đối tượng. Theo tên trong config, hàm chọn:


| Khối        | Tên trong config                     | Lựa chọn                                                         |
| ----------- | ------------------------------------ | ---------------------------------------------------------------- |
| LLM         | `[llm].backend`                      | `vllm` (`VLLMBackend`), `fake` (`FakeLLM`, phát lại đáp án)      |
| Transcriber | `[asr].backend` hoặc `--transcripts` | `gold` (chữ đáp án), `asr` (ASR thật), `replay` (chữ ASR đã lưu) |
| Engine ASR  | `[asr].engine`                       | `faster_whisper`, `chunkformer`, `hf_seq2seq`, `nemo_ctc`        |
| Guard       | `[agent].guards`                     | `allowed_tool`, `schema`, `duplicate_write`, `after_ask`         |


LLM và transcriber được bọc trong `TimedLLM`, `TimedTranscriber` ([`lifemate/run/timing.py`](../lifemate/run/timing.py)) để đo giờ mà không sửa lớp bên trong.

### Bước 1. Runner nhận một mẫu

`Runner.run` ghi ngay một `submission.jsonl` có đủ dòng giữ chỗ cho mọi mẫu, khởi động (warm-up) LLM và ASR, rồi duyệt
mẫu theo thứ tự id. Với từng mẫu, `Runner.run_one`:

- mở **đúng một** `ToolEnv` cho mẫu ([`lifemate/io/zaic.py`](../lifemate/io/zaic.py) › `ZaicLoader.open_env`; mở lần hai báo lỗi);
- gọi `Agent.run(sample, env)`;
- nếu có bất kỳ `Exception` nào: trả kết quả `exception`, **vẫn giữ** các lời gọi tool đã gửi và hash trạng thái. Đây là
chỗ duy nhất trong repo bắt mọi lỗi; một mẫu lỗi không làm dừng cả lần chạy.

Sau mỗi mẫu, kết quả được nối vào `trace.jsonl` (có `fsync`) và `submission.jsonl` được ghi lại. Tiến trình chết giữa
chừng vẫn còn file nộp đủ dòng, và `--resume` chạy tiếp từ mẫu chưa xong.

### Bước 2. Bấm giờ và chép âm thanh

[`lifemate/agent/agent.py`](../lifemate/agent/agent.py) › `Agent.run`:

```python
budget = Budget(self.cfg, self.llm_cfg, clock)      # bấm giờ TRƯỚC khi chạy ASR
transcript = self.transcriber.transcribe(sample)    # chữ của câu cuối và các câu trong lịch sử
```

`Budget` lấy mốc giờ ngay khi tạo, nên thời gian ASR cũng trừ vào hạn `deadline_s = 29.0` giây (chừa 1 giây so với luật
30 giây).

Transcriber trả về `Transcript(final=..., history=(...), seconds=...)`. `history` có cùng độ dài với `sample.history`:
lượt người dùng có chữ, lượt trợ lý là `None`.

### Bước 3. Dựng prompt

[`lifemate/agent/prompt.py`](../lifemate/agent/prompt.py) › `PromptBuilder.start` dựng hội thoại gồm hai tin nhắn:

**Tin nhắn `system`** = quy tắc trong [`configs/prompts/system_v1.md`](../configs/prompts/system_v1.md) + mô tả các tool **của app đã cài** (thay vào chỗ
`{tool_docs}`). Mô tả tool do [`lifemate/agent/tooldocs.py`](../lifemate/agent/tooldocs.py) › `render_tools` sinh, mỗi app một khối, mỗi tool ba dòng (ví dụ minh
hoạ dạng):

```
## alarm (Báo thức): Đặt và quản lý báo thức
- alarm.create [write] → {...}
  Tạo báo thức mới
  time: string (Thời điểm báo thức) [format=date-time] | label?: string (văn bản tự do)
```

Dấu `?` là tham số tùy chọn; `(id lấy từ ...)` là tham số `x-ref` phải chép từ kết quả tool khác; `(văn bản tự do)` là
tham số mà bộ chấm so khớp mờ.

**Tin nhắn `user`** gồm ba khối:

```
# Ngữ cảnh
current_time: 2026-11-10T21:00:00+07:00
timezone: Asia/Ho_Chi_Minh
Lịch (bây giờ là thứ Ba 2026-11-10 21:00):
  2026-11-09 thứ Hai (tuần này, hôm qua)
  2026-11-10 thứ Ba (tuần này, hôm nay)
  2026-11-11 thứ Tư (tuần này, mai)
  ... (đủ 15 ngày)
location: 10.77, 106.70
user_profile.home_address: ...

# Lịch sử hội thoại
Người dùng: <chữ ASR của câu người dùng thứ nhất>
Trợ lý: <câu trả lời của trợ lý>
  [đã gọi] contacts.search({"query":"Lan"}) → {"ok":true,...}

# Yêu cầu hiện tại
<chữ ASR của câu cuối>
```

(Ví dụ minh hoạ dạng, không phải mẫu thật.)

Ba điểm cần biết:

- **Bảng lịch 15 ngày** (`render_calendar`): LLM hay tính sai "thứ sáu", "mai", "chủ nhật này". Bảng tính sẵn giúp LLM
chỉ cần tra.
- **`installed_apps` không in ra**: danh sách tool trong `system` đã chỉ gồm tool được phép.
- **Định dạng `chatml`** dựng khác ([`lifemate/agent/chatml.py`](../lifemate/agent/chatml.py) › `build_messages`): tin `system` là lời dặn ngắn của lead kèm thời
  gian, hồ sơ, vị trí, và trường `tools`; lịch sử thành các tin `user`, `assistant` (`tool_calls`), `tool` riêng; không có
  bảng lịch. Rút gọn hai mức: schema rút gọn (`compact_tool_schema`), rồi bỏ dần lượt lịch sử cũ.
- **Rút gọn khi quá dài**: nếu prompt cộng `max_tokens` vượt `max_model_len`, thử lần lượt bốn mức: 0 giữ nguyên; 1 cắt
kết quả tool trong lịch sử còn 1/4; 2 bỏ thêm mô tả tham số; 3 bỏ dần các cặp lượt lịch sử cũ nhất (giữ ít nhất 2
lượt). Mức đã dùng được ghi vào trace (`prompt_shrink_level`).

### Bước 4. Vòng lặp agent

Config mặc định ([`configs/v0.toml`](../configs/v0.toml), `[agent].format = "chatml"`) dùng
[`lifemate/agent/chatml_agent.py`](../lifemate/agent/chatml_agent.py) › `ChatMLAgent`: hội thoại theo định dạng tool
calling có sẵn của Qwen, giống dữ liệu SFT của lead. Sơ đồ một mẫu, từ lúc nhận mẫu tới lúc có bài nộp (các số lấy từ
`[agent]` của config: `max_steps = 10`, `max_retries = 2`, `deadline_s = 29.0`).

**Sơ đồ tổng quát** (chưa đi vào tên hàm):

```
            ┌─────────────────────────────┐
            │  Mẫu thi: âm thanh các câu  │
            │  người dùng + lịch sử chữ   │
            │  của trợ lý + app đã cài    │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │  ASR: chép mọi câu người    │   bắt đầu tính giờ từ đây
            │  dùng thành chữ             │   (hạn 29 giây cho cả mẫu)
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │  Dựng hội thoại cho LLM:    │
            │  luật, danh sách tool,      │
            │  lịch sử, yêu cầu cuối      │
            └──────────────┬──────────────┘
                           ▼
     ┌────────────────────────────────────────────┐
     │  LLM đọc hội thoại và viết câu trả lời     │◄─────────────────────┐
     └──────────────────────┬─────────────────────┘                      │
                            ▼                                            │
              LLM muốn gọi tool hay đã xong?                             │
           ┌────────────────┴─────────────────┐                          │
      đã xong (trả lời bằng lời)        muốn gọi tool                    │
           │                                  ▼                          │
           │              ┌──────────────────────────────────┐           │
           │              │ Kiểm tra lời gọi: tool có được   │           │
           │              │ phép không, tham số có đúng không│           │
           │              └───────────────┬──────────────────┘           │
           │                 sai          │          đúng                │
           │          ┌───────────────────┴──────────────┐               │
           │          ▼                                  ▼               │
           │  Không gọi tool;                  Gọi tool trên "điện thoại │
           │  báo lỗi vào hội thoại            giả" ToolEnv; ghi kết quả │
           │  để LLM sửa                       vào hội thoại             │
           │          │                                  │               │
           │          │                      là câu hỏi lại người dùng?  │
           │          │                        có │              │ không  │
           │          │                           ▼              │        │
           │          │                         dừng             │        │
           │          │                           │              │        │
           │          └───────────────────────────┼──────────────┴────────┘
           │                                      │        (lượt LLM tiếp theo)
           │◄─────────────────────────────────────┘
           ▼
     ┌────────────────────────────────────────────┐
     │  Bài nộp: danh sách lời gọi tool đã thực   │
     │  hiện + mã trạng thái app cuối cùng        │
     └────────────────────────────────────────────┘
```

Vòng lặp cũng dừng khi: đã đủ 10 lượt LLM, hội thoại quá dài so với giới hạn của LLM, hết giờ, hoặc LLM viết sai quá 2
lần. Khi dừng giữa chừng, các lời gọi tool đã thực hiện trước đó vẫn được nộp.

**Sơ đồ chi tiết** (kèm tên hàm và file):

```
Runner.run_one(mẫu)                                         lifemate/run/runner.py
  │  mở ToolEnv của mẫu (đúng một lần)                       lifemate/io/zaic.py › ZaicLoader.open_env
  ▼
ChatMLAgent.run(mẫu, env)                                   lifemate/agent/chatml_agent.py
  1. Budget(...)              bắt đầu bấm giờ (hạn 29 s tính từ đây, gồm cả ASR)
  2. transcriber.transcribe   chép câu cuối và mọi câu người dùng trong lịch sử
  3. prompts.start            dựng tin nhắn: system (luật + danh sách tool của app đã cài),
                              lịch sử hội thoại, câu cuối
  4. tool_call_constraint     luật ép đầu ra: sau <tool_call> chỉ viết được tool đã cài,
                              tham số đúng tên, kiểu, enum
  │
  ▼
┌──────────────────── VÒNG LẶP: mỗi vòng là một lượt LLM ────────────────────┐
│ a. Đếm token prompt. Sắp tràn context thì cắt dần kết quả tool dài nhất     │
│    (_fit_context).                                                         │
│ b. Budget.before_step                                                       │
│      đã đủ 10 lượt?           → dừng MAX_STEPS                              │
│      prompt + đầu ra > context? → dừng CONTEXT_OVERFLOW                     │
│      hết giờ?                 → dừng TIMEOUT                                │
│ c. LLM.generate (vLLM, ép theo luật ở bước 4; tự huỷ khi chạm hạn giờ)      │
│      bị huỷ giữa chừng        → dừng TIMEOUT                                │
│ d. parse_reply(văn bản LLM vừa sinh)                                        │
│      ├─ khối <tool_call> hỏng → nối tin báo lỗi; còn lượt thử lại (tối đa 2)│
│      │                          thì quay lại a, hết thì dừng PARSE_ERROR    │
│      ├─ không có <tool_call>  → LLM trả lời bằng lời: dừng DONE             │
│      └─ có 1 hay nhiều lời gọi → xử lý TỪNG lời gọi:                        │
│           GuardChain: allowed_tool → schema (env.validate) → duplicate_write│
│             ├─ BLOCK  → KHÔNG gửi tới env; nối tin tool                    │
│             │           {"ok": false, "error": "<lý do>"}; tốn 1 lượt thử   │
│             │           lại, hết lượt thì dừng RETRIES_EXHAUSTED            │
│             ├─ STOP   → dừng DONE (lời gọi không được gửi)                  │
│             └─ PASS / WARN → env.call(tên, tham số); nối tin tool = kết quả │
│                   └─ nếu là sys.ask_missing_info → dừng ASKED               │
│ e. Chưa dừng → quay lại a: LLM đọc kết quả tool rồi quyết định bước tiếp    │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
  5. tool_calls = env.get_tool_calls()   mọi lời gọi đã tới env (bài nộp)
     state_hash = env.get_state_hash()   trạng thái app cuối (bài nộp)
     → SampleResult (lý do dừng, các bước, lỗi, thời gian, token)
```

Ví dụ hai lượt với câu "Gọi cho chị Lan": lượt 1, LLM viết `<tool_call>` gọi `contacts.search` với `query = "Lan"`;
guard cho qua, `env.call` trả danh sách liên hệ, kết quả nối vào hội thoại. Lượt 2, LLM đọc kết quả, gọi
`phone.call` với `contact_id` vừa tìm được; env báo gọi thành công. Lượt 3, LLM trả lời bằng lời ("Đã gọi chị Lan"), không có
`<tool_call>` nên mẫu dừng `DONE`. Bài nộp gồm hai lời gọi `contacts.search` và `phone.call` cùng `state_hash`.

Hai lời gọi trong cùng một lượt được xử lý lần lượt; nếu lời gọi đầu làm mẫu dừng (ví dụ `sys.ask_missing_info`) thì lời
gọi sau không chạy. Lời gọi bị BLOCK không tới env nên không có trong bài nộp và không làm đổi trạng thái app.

Định dạng cũ `format = "json"` dùng [`lifemate/agent/agent.py`](../lifemate/agent/agent.py) › `Agent`, cùng ngân sách,
guard, hạn giờ và kết quả, chỉ khác cách LLM viết lời gọi: mỗi lượt đúng một JSON `{"action": …}`. Mỗi vòng (tối đa
`max_steps = 10`):

```
Budget.before_step   → dừng nếu đủ số bước, tràn context, hoặc hết giờ
LLM.generate         → một JSON, bị ép theo schema của mẫu
parse                → Action (call / done) hoặc ParseError
  ├─ ParseError      → nối "LỖI: đầu ra không phải JSON hợp lệ..." ; tốn 1 lượt thử lại
  ├─ done            → dừng (StopReason.DONE)
  └─ call            → GuardChain.check
                         ├─ BLOCK → không gửi; nối "LỖI (chưa gọi tool): <lý do>" ; tốn 1 lượt thử lại
                         ├─ STOP  → dừng như done
                         └─ PASS/WARN → env.call ; nối "KẾT QUẢ: {...}"
                                         nếu là sys.ask_missing_info → dừng (StopReason.ASKED)
```

**Ở định dạng `chatml`**, khi `constrain_tools` bật, vLLM ép phần sau `<tool_call>` theo luật
`chatml.tool_call_constraint` (tool đã cài, tham số đúng tên, kiểu, `enum`); `chatml.parse_reply` đọc các khối
`<tool_call>`, không có khối nào là câu kết (dừng `DONE`). Lời gọi bị guard chặn thành tin `tool` `{"ok": false, "error": "<lý do>"}`, đúng dạng lượt "phục
hồi lỗi" trong dữ liệu của lead. Phần dưới mô tả định dạng `json`.

**Định dạng đầu ra** ([`lifemate/agent/protocol.py`](../lifemate/agent/protocol.py)): mỗi lượt LLM trả đúng một JSON:

```json
{"action":"call","name":"alarm.create","arguments":{"time":"2026-11-11T05:30:00+07:00"}}
{"action":"done"}
```

`output_schema(allowed)` tạo JSON schema trong đó `name` chỉ được là một tên trong danh sách tool cho phép của mẫu (luôn
có thêm `sys.ask_missing_info`). vLLM dùng schema này để **ép** đầu ra (constrained decoding), nên LLM không thể gọi tool
không tồn tại hay sinh JSON sai cấu trúc. `parse` vẫn chịu lỗi (cắt phần thừa ngoài cặp ngoặc nhọn) và không bao giờ ném
lỗi.

**Chuỗi guard** ([`lifemate/agent/guards.py`](../lifemate/agent/guards.py)), chạy theo thứ tự trong config, dừng ở kết quả BLOCK hoặc STOP đầu tiên:


| Guard                          | Kiểm tra                                  | Kết quả khi vi phạm                                  |
| ------------------------------ | ----------------------------------------- | ---------------------------------------------------- |
| `allowed_tool` (G1)            | Tool có trong danh sách được phép         | BLOCK, mã `unknown_tool`                             |
| `schema` (G2)                  | `env.validate(name, arguments)` chấp nhận | BLOCK, mã `invalid_args`, kèm nguyên văn lỗi của BTC |
| `duplicate_write` (G3)         | Lời gọi `write` y hệt đã chạy trước đó    | WARN (vẫn gửi), mã `duplicate_write`                 |
| `after_ask` (G4, mặc định tắt) | Đã hỏi lại người dùng                     | STOP                                                 |


Ý nghĩa của BLOCK: lời gọi sai **không bao giờ tới env**, nên không bị tính vào file nộp và không làm hỏng trạng thái. LLM
đọc lỗi và sửa ở bước sau. Ví dụ LLM quên tham số bắt buộc `time`: G2 chặn, LLM nhận `LỖI (chưa gọi tool): thiếu tham số bắt buộc: time` và gọi lại cho đủ.

**Ngân sách** ([`lifemate/agent/budget.py`](../lifemate/agent/budget.py)): trước mỗi bước kiểm tra ba điều kiện theo thứ tự: số bước, tràn context
(`prompt_tokens + max_tokens > max_model_len`), hết giờ. Số lần thử lại (lỗi parse, BLOCK) đếm riêng, tối đa
`max_retries = 2`.

**Hội thoại chỉ nối thêm** ([`lifemate/agent/prompt.py`](../lifemate/agent/prompt.py) › `Conversation`): tin cũ không bao giờ bị sửa. Hai lợi ích: vLLM dùng lại
phần đầu prompt đã tính (prefix cache), và hội thoại lúc chạy khớp từng byte với dữ liệu huấn luyện (mục 11). Token được
đếm tăng dần; khi số đếm lệch với số vLLM báo, lấy số của vLLM (`reconcile`).

### Bước 5. Hạn giờ được thực thi thật

[`lifemate/backends/llm.py`](../lifemate/backends/llm.py) › `VLLMBackend._generate_until` không gọi `LLM.chat` một lần rồi chờ. Hàm đưa request vào engine của vLLM,
chạy từng bước giải mã (`engine.step()`), và **huỷ request** (`abort_request`) ngay khi chạm hạn. Một mẫu vì vậy không
bao giờ vượt hạn quá một bước giải mã. Bước bị huỷ trả `timed_out=True`, agent dừng với `StopReason.TIMEOUT`, và các lời
gọi tool đã gửi trước đó **vẫn được nộp**.

### Bước 6. Kết quả của mẫu

`Agent.run` trả về `SampleResult`:


| Trường        | Nội dung                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| `stop_reason` | `done`, `asked`, `max_steps`, `timeout`, `context_overflow`, `parse_error`, `retries_exhausted`, `exception` |
| `errors`      | Mã lỗi gặp trong mẫu: `parse_error`, `unknown_tool`, `invalid_args`, `duplicate_write`, `exception`          |
| `steps`       | Từng bước: văn bản thô, hành động, kết luận guard, kết quả tool, số token, số giây                           |
| `tool_calls`  | `env.get_tool_calls()`: các lời gọi đã tới env, chính là nội dung file nộp                                   |
| `state_hash`  | `env.get_state_hash()`                                                                                       |
| `timing`      | `asr_s`, `llm_s`, `tool_s`, `total_s`                                                                        |


### Bước 7. Chốt file và chấm điểm

Hết mẫu, `RunStore.finalize` dựng lại `submission.jsonl` từ `trace.jsonl`. Split có nhãn thì `Runner._score` chạy
`lifemate.eval.score` trong **tiến trình con** (để không nạp `zaic_tools` lần hai), ghi `score.txt` và `details.jsonl`.
Split không nhãn (public test) thì chạy `check_submission`. Cuối cùng nối một dòng vào [`artifacts/runs/index.csv`](../artifacts/runs/index.csv).

Thư mục một lần chạy:


| File                         | Nội dung                                                                   |
| ---------------------------- | -------------------------------------------------------------------------- |
| `config.json`                | Toàn bộ config, thông tin máy, commit mã, SHA-256 của prompt, seed         |
| `trace.jsonl`                | Một dòng mỗi mẫu: mọi bước, lời gọi, thời gian. Nguồn dữ liệu gốc duy nhất |
| `submission.jsonl`           | File nộp, dựng từ trace                                                    |
| `summary.json`               | Thống kê: số mẫu, lý do dừng, p50/p95 thời gian, điểm                      |
| `score.txt`, `details.jsonl` | Điểm tổng, điểm theo loại câu, lỗi; điểm từng mẫu                          |
| `ids.txt`                    | Danh sách mẫu đã chạy                                                      |


---

## 4. Lõi chung: [`lifemate/core`](../lifemate/core)


| File                    | Giữ việc gì                                                                              | Điểm cần nhớ                                                                                                                                                                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`lifemate/core/types.py`](../lifemate/core/types.py)              | Mọi kiểu dữ liệu và giao diện dùng chung                                                 | Dataclass `frozen=True, slots=True` (không sửa được sau khi tạo). Enum `StopReason`, `ErrorCode`, `GuardVerdict` thay cho chuỗi tự do. Các giao diện là `Protocol`: `ToolEnvLike`, `LLMBackend`, `Transcriber`, `TokenCounter`, `Clock`, `Guard`, `AgentObserver` |
| [`lifemate/core/config.py`](../lifemate/core/config.py)             | Đọc `configs/*.toml` thành `Config` (ba mục `[llm]`, `[agent]`, `[asr]`) và `EnvProfile` | Kiểm tra chặt: khoá lạ hoặc thiếu khoá báo `ConfigError` ngay lúc nạp. Token HF không nằm trong file; profile chỉ ghi nguồn (`env:HF_TOKEN`, `kaggle_secret:...`)                                                                                                 |
| [`lifemate/core/envinfo.py`](../lifemate/core/envinfo.py)            | Dò GPU, commit mã, phiên bản thư viện; chọn dtype                                        | `dtype = "auto"`: bfloat16 trên GPU từ Ampere (A30), float16 trên T4                                                                                                                                                                                              |
| [`lifemate/core/seeding.py`](../lifemate/core/seeding.py)            | Cố định seed Python, NumPy, PyTorch                                                      | Theo yêu cầu tái lập của thể lệ                                                                                                                                                                                                                                   |
| [`lifemate/core/tokens.py`](../lifemate/core/tokens.py)             | Đếm token theo chat template của Qwen                                                    | `HFTokenCounter` dùng tokenizer thật; `SimpleTokenCounter` đếm thô cho test                                                                                                                                                                                       |
| [`lifemate/core/textsim.py`](../lifemate/core/textsim.py)            | Gom câu gần trùng (giống nhau từ 85%)                                                    | Dùng khi chia tập và lọc dữ liệu SFT; lấy ý tưởng từ [`src/preprocessing/dataset/split.py`](https://github.com/ThanhDanh1510/ZaloAI2026/blob/main/src/preprocessing/dataset/split.py) của lead                                                                                                                                                                                          |
| [`lifemate/core/jsonutil.py`](../lifemate/core/jsonutil.py)           | `canonical_json`: JSON sắp khoá, không khoảng trắng                                      | Cùng dữ liệu luôn ra cùng chuỗi, nên so sánh và băm được                                                                                                                                                                                                          |
| [`lifemate/core/paths.py`](../lifemate/core/paths.py), [`lifemate/core/tomlio.py`](../lifemate/core/tomlio.py) | Thư mục gốc repo; ghi TOML đơn giản                                                      |                                                                                                                                                                                                                                                                   |


**Vì sao dùng `Protocol`:** `Agent` chỉ biết "một thứ có `generate(...)`", không biết đó là vLLM hay LLM giả. Nhờ vậy
test chạy `Agent` với `FakeLLM` và `FakeEnv` trên CPU, còn lúc thi dùng vLLM và `ToolEnv` thật, mà mã của `Agent` không
đổi.

---

## 5. Đọc và ghi dữ liệu: [`lifemate/io`](../lifemate/io)


| File                  | Giữ việc gì                                                                                                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`lifemate/io/data.py`](../lifemate/io/data.py)             | `load_split(root, split, ids)`: đọc `<root>/<split>/<split>.jsonl` thành danh sách `Sample`. `read_ids`: đọc file mỗi dòng một mã mẫu                                                   |
| [`lifemate/io/zaic.py`](../lifemate/io/zaic.py)             | Nạp gói `zaic_tools` **của đúng split** và mở `ToolEnv` cho từng mẫu. **Đây là module duy nhất được chạm tới thư mục môi trường của BTC** (qua `zaic_tools`); có test kiểm tra điều này |
| [`lifemate/io/store.py`](../lifemate/io/store.py)            | `RunStore`: đọc ghi thư mục run, ghi nguyên tử, chạy tiếp, dựng `submission.jsonl` và `summary.json` từ trace                                                                           |
| [`lifemate/io/check_submission.py`](../lifemate/io/check_submission.py) | Kiểm tra file nộp cho split không nhãn: đủ id, không trùng, tên tool có trong catalog, phân bố câu trả lời giống train                                                                  |


**[`lifemate/io/zaic.py`](../lifemate/io/zaic.py) làm gì cụ thể:** mỗi split có bản `zaic_tools` riêng nhưng cùng tên module. `load_zaic` xoá bản cũ khỏi
`sys.modules`, chèn thư mục split lên đầu `sys.path`, import lại, rồi kiểm tra module thật sự nằm trong thư mục split.
`EnvRegistry` ghi nhớ cặp (split, mã mẫu) đã mở để mỗi mẫu chỉ có một `ToolEnv`, đúng luật.

**[`lifemate/io/store.py`](../lifemate/io/store.py) và chạy tiếp:** `trace.jsonl` là nguồn gốc duy nhất; mỗi mẫu xong ghi một dòng có `flush` và `fsync`. Khi
chạy tiếp, mẫu đã có trong trace được bỏ qua, dòng cuối bị cắt dở (do tắt máy giữa lúc ghi) bị loại, và config mới phải
khớp config cũ ở các khoá quan trọng (`check_resume`), để không trộn kết quả của hai cấu hình.

---

## 6. ASR: `lifemate/backends/asr*.py`

### 6.1. Các transcriber


| Lớp                                         | Dùng khi                                                      | Đầu ra                                  |
| ------------------------------------------- | ------------------------------------------------------------- | --------------------------------------- |
| `GoldTranscriber`                           | Muốn tách lỗi ASR khỏi lỗi LLM                                | Trả lại chữ đáp án `audio_text` của BTC |
| `ASRTranscriber`                            | Chạy ASR thật                                                 | Chép mọi file âm thanh của mẫu          |
| `ReplayTranscriber` ([`lifemate/eval/rescore_asr.py`](../lifemate/eval/rescore_asr.py)) | Đọc lại chữ ASR đã lưu (`--transcripts replay --hyps <file>`) | Chữ từ file `hyps.jsonl`                |


`ASRTranscriber.transcribe` làm bốn việc:

1. `wav_paths`: lập danh sách file, các lượt người dùng trong lịch sử trước, câu cuối sau.
2. Chia batch (`batch_size`), gọi backend, kiểm tra số chữ trả về bằng số file.
3. Nếu bật `prompt_guard_ngram`: câu nào hỏng do câu mồi thì chép lại không câu mồi (`_guard`, mục 6.3).
4. Nếu bật `sanitize`: lọc câu bịa (mục 6.4). Lượt lịch sử chỉ toàn câu bịa thì thành rỗng.

### 6.2. Bốn backend mô hình


| Engine           | Lớp                | Mô hình đã thử                     | Thư viện          |
| ---------------- | ------------------ | ---------------------------------- | ----------------- |
| `faster_whisper` | `FasterWhisperASR` | whisper-large-v3                   | CTranslate2       |
| `chunkformer`    | `ChunkFormerASR`   | ChunkFormer RNNT, CTC (tiếng Việt) | gói `chunkformer` |
| `hf_seq2seq`     | `HFSeq2SeqASR`     | PhoWhisper                         | transformers      |
| `nemo_ctc`       | `NemoCTCASR`       | Parakeet                           | NeMo              |


Thư viện nặng chỉ được import bên trong hàm khởi tạo, nên import module trên máy không GPU vẫn được và test chạy trên
CPU. `FasterWhisperASR` có thêm ba hàm cho benchmark lần 3: `transcribe_one(audio, prompt)` (chép một câu với câu mồi
riêng, trả thêm điểm tin cậy), `transcribe_batch(paths, prompts)`, và `score_texts(audio, texts)` (chấm xác suất của các
câu cho sẵn bằng ép chuỗi).

### 6.3. Lớp bảo vệ câu mồi: [`lifemate/backends/asr_guard.py`](../lifemate/backends/asr_guard.py)

Câu mồi (`initial_prompt`) giúp Whisper viết đúng tên tiếng Anh, nhưng đôi khi làm Whisper chép lại câu mồi hoặc lặp vô
hạn. `needs_retry` coi một câu là hỏng khi có một trong bốn dấu hiệu, không cần đáp án:

1. rỗng sau khi lọc;
2. bộ lọc câu bịa có cắt;
3. có 4 từ liên tiếp trùng câu mồi;
4. tỉ lệ nén zlib lớn hơn 2,4 (dấu hiệu lặp).

Câu hỏng được chép lại không câu mồi.

### 6.4. Bộ lọc câu bịa: [`lifemate/backends/asr_sanitizer.py`](../lifemate/backends/asr_sanitizer.py)

Whisper được huấn luyện trên nhiều phụ đề YouTube, nên đôi khi "nghe" ra "hãy subscribe cho kênh…" trên đoạn im lặng.
`sanitize` cắt các cụm phụ đề đã biết (`HALLUCINATION_PATTERNS`) và gộp đoạn lặp. Lấy từ [`src/preprocessing/asr/sanitizer.py`](https://github.com/ThanhDanh1510/ZaloAI2026/blob/main/src/preprocessing/asr/sanitizer.py) của lead, sửa
một chỗ: mẫu "đăng ký" chỉ khớp khi nói về kênh, để không cắt yêu cầu thật như "hủy gói đăng ký".

### 6.5. Câu mồi theo mẫu: [`lifemate/backends/asr_prompt.py`](../lifemate/backends/asr_prompt.py)

`build_sample_prompts(sample, catalog_apps, fixed_prompt, variant)` trả một câu mồi cho mỗi câu người dùng:

- `E0`: câu mồi cố định [`configs/asr_prompts/en_full_v1.txt`](../configs/asr_prompts/en_full_v1.txt);
- `E1`: thêm câu trả lời của trợ lý ở lượt liền trước;
- `E2`: thêm "Ứng dụng đã cài: &lt;tên tiếng Việt các app&gt;." rồi lời trợ lý liền trước. Đây là biến thể thắng của benchmark
lần 3.

Hàm chỉ đọc `installed_apps`, lượt trợ lý và catalog; không đọc lượt người dùng (lúc thi chỉ có âm thanh).

---

## 7. LLM: [`lifemate/backends/llm.py`](../lifemate/backends/llm.py)


| Lớp           | Dùng khi               | Cách hoạt động                                                                                                                                           |
| ------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FakeLLM`     | Test, kiểm tra cài đặt | Phát lại `label.tool_calls` của mẫu thành JSON chuẩn tắc, rồi `{"action":"done"}`. Chèn được đầu ra hỏng và giả lập thời gian để test nhánh lỗi, hết giờ |
| `VLLMBackend` | Chạy thật              | vLLM offline, giải mã tham lam (temperature 0, seed cố định), đầu ra ép theo JSON schema, prefix cache bật                                               |


Các chi tiết của `VLLMBackend`:

- **Tương thích nhiều phiên bản vLLM:** `_structured_kwargs` dùng `StructuredOutputsParams` (bản mới) hoặc
`GuidedDecodingParams` (bản cũ). Chat template được gọi với `enable_thinking=False` (Qwen3 không suy nghĩ dài); bản
vLLM không nhận tham số này thì tự bỏ.
- **Hạn giờ bằng step/abort** (mục 3, bước 5). Nếu engine không có API này, quay về `LLM.chat` và ghi log lỗi.
- **Warm-up:** chạy 2 lần một prompt khoảng 4 000 token để biên dịch kernel trước mẫu đầu tiên, để mẫu đầu không bị chậm
bất thường.
- **`batch_invariant`:** bật chế độ của vLLM cho kết quả không phụ thuộc kích thước batch; chỉ chạy trên GPU từ Ampere.
- **`gpu_memory_utilization = 0.80`:** chừa khoảng 20% bộ nhớ GPU cho ASR chạy chung.

---

## 8. Agent: [`lifemate/agent`](../lifemate/agent)

Mục 3 đã đi qua luồng. Phần này tóm lại trách nhiệm từng file và các hằng số cần biết.


| File           | Giữ việc gì                                                 | Không làm                                                                   |
| -------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| [`lifemate/agent/agent.py`](../lifemate/agent/agent.py)     | Vòng lặp: hỏi LLM, parse, guard, gọi env, phát sự kiện      | Không đọc file, không ghi trace, không tạo đối tượng, không bắt `Exception` |
| [`lifemate/agent/prompt.py`](../lifemate/agent/prompt.py)    | Dựng prompt, giữ hội thoại chỉ nối thêm, rút gọn khi dài    | Không quyết định dừng                                                       |
| [`lifemate/agent/protocol.py`](../lifemate/agent/protocol.py)  | Schema đầu ra và bộ parse                                   |                                                                             |
| [`lifemate/agent/tooldocs.py`](../lifemate/agent/tooldocs.py)  | Mô tả tool gọn từ catalog; `CatalogIndex` tra tool theo tên |                                                                             |
| [`lifemate/agent/guards.py`](../lifemate/agent/guards.py)    | Chuỗi guard                                                 | Không gọi `env.call`                                                        |
| [`lifemate/agent/budget.py`](../lifemate/agent/budget.py)    | Trả lời "còn chạy được bước tiếp không"                     | Không gọi LLM                                                               |
| [`lifemate/agent/observers.py`](../lifemate/agent/observers.py) | Nhận sự kiện (bắt đầu mẫu, từng bước, kết thúc)             | Không tác động ngược vòng lặp                                               |


**Ba chuỗi cố định LLM đọc** (trong [`lifemate/agent/agent.py`](../lifemate/agent/agent.py)), phải giống từng byte với dữ liệu huấn luyện, không được sửa:

```python
PARSE_ERROR_MSG = "LỖI: đầu ra không phải JSON hợp lệ, hãy trả về đúng một JSON"
BLOCK_PREFIX = "LỖI (chưa gọi tool): "
RESULT_PREFIX = "KẾT QUẢ: "
```

**Prompt hệ thống** ([`configs/prompts/system_v1.md`](../configs/prompts/system_v1.md)) có 14 quy tắc. Các quy tắc quan trọng nhất:

1. Không tự đặt ID; chưa có ID thì gọi tool đọc của chính app đó trước.
2. Gọi tool đọc không kèm bộ lọc, rồi tự chọn trong kết quả.
3. Chỉ điền tham số bắt buộc và tham số người dùng nói rõ.
4. Gọi đủ nhưng không thừa.
5. Câu có điều kiện: kiểm tra rồi làm đúng nhánh.
6. Thiếu hoặc mơ hồ thì hỏi lại bằng `sys.ask_missing_info`.
7. Huỷ ý, không app nào làm được, hay chỉ chào hỏi thì trả `done` ngay.

Các quy tắc còn lại hướng dẫn đọc ngày giờ, số tiền, văn bản tự do, số điện thoại, địa chỉ và lịch sử hội thoại.
[`configs/prompts/system_v1_alternatives.md`](../configs/prompts/system_v1_alternatives.md) thêm quy tắc 15 cho định dạng hai phương án `{A | B}` (benchmark ASR lần 3).

**Train mode** (`--train-mode`): giấu 6 app "held-out" (medication, fitness, flights, bike_rental, library,
subscriptions) khỏi danh sách tool, để đo khả năng làm việc với app chưa gặp. Mẫu có lịch sử gọi tool của app bị giấu
báo `HeldOutLeak`.

---

## 9. Chạy và lắp ráp: [`lifemate/run`](../lifemate/run)


| File         | Giữ việc gì                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| [`lifemate/run/cli.py`](../lifemate/run/cli.py)     | Điểm vào dòng lệnh; đọc tham số, config, mẫu; gọi factory và runner            |
| [`lifemate/run/factory.py`](../lifemate/run/factory.py) | Tạo mọi đối tượng từ `Config`; registry chọn backend theo tên                  |
| [`lifemate/run/runner.py`](../lifemate/run/runner.py)  | Warm-up, chạy từng mẫu, cô lập lỗi theo mẫu, chốt file nộp, chấm hoặc kiểm tra |
| [`lifemate/run/timing.py`](../lifemate/run/timing.py)  | `TimedLLM`, `TimedTranscriber`: đo giờ theo mẫu Decorator                      |
| [`lifemate/run/report.py`](../lifemate/run/report.py)  | In bảng các lần chạy từ [`artifacts/runs/index.csv`](../artifacts/runs/index.csv), bảng loss của lần huấn luyện              |


**Các tham số CLI hay dùng:**


| Tham số                                              | Ý nghĩa                                                      |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `--config configs/v0.toml`                           | Config chạy                                                  |
| `--env local`                                        | Profile máy: `local`, `kaggle`, `colab`, `runpod`, `docker`… |
| `--split train`, `--ids configs/splits/val_seen.txt` | Split và danh sách mẫu                                       |
| `--transcripts gold|asr|replay`, `--hyps <file>`     | Nguồn chữ, ghi đè `[asr].backend`                            |
| `--set agent.system_prompt=configs/prompts/x.md`     | Ghi đè một khoá config, dùng được nhiều lần                  |
| `--out <thư mục>`, `--resume`                        | Thư mục run; chạy tiếp                                       |
| `--limit N`                                          | Chỉ chạy N mẫu đầu (kiểm tra cài đặt)                        |
| `--train-mode`                                       | Giấu app held-out                                            |


**Vì sao `factory` là nơi duy nhất tạo đối tượng:** muốn đổi một khối (LLM, ASR, guard) chỉ cần sửa config; mọi chỗ
khác nhận đối tượng qua tham số hàm khởi tạo. Không có biến toàn cục hay singleton.

---

## 10. Chấm điểm và đo lường: [`lifemate/eval`](../lifemate/eval)

### 10.1. Bộ chấm: [`lifemate/eval/score.py`](../lifemate/eval/score.py)

Mô phỏng đúng luật chấm của BTC (luật gốc chép trong docstring của file):

1. **Phát lại:** chạy lời gọi đáp án và lời gọi nộp trên `ToolEnv` của chính split, từ trạng thái ban đầu của mẫu. Mỗi
 bên ra một hash trạng thái và tập lời gọi bị từ chối.
2. **Đáp án rỗng:** nộp rỗng thì 1 điểm, nộp có lời gọi thì 0 điểm.
3. **Hash khác nhau:** 0 điểm.
4. **Hash khớp:** điểm là F1. Lời gọi nộp "khớp" lời gọi đáp án khi cùng tên tool, cùng tập tham số, tham số thường bằng
 nhau (có xét kiểu: `True` khác `1`), tham số mờ (`x-match: fuzzy`) có token-F1 &gt; 0,6. Số cặp khớp tối đa M tìm bằng
 ghép cặp hai phía (thuật toán Kuhn, bỏ qua thứ tự). P = M / số lời gọi nộp, R = M / số lời gọi đáp án.
5. Điểm tập = 100 × trung bình điểm các mẫu.

Hai chế độ: `strict` (đúng luật) và `canon` (chuẩn hoá chuỗi thêm: chữ thường, gộp khoảng trắng, "700 cm" = "700cm").
Báo cáo in thêm điểm theo loại câu (thực hiện, hỏi lại, ngoài phạm vi), ma trận nhầm lẫn và lỗi theo loại.

### 10.2. Chia tập: [`lifemate/eval/make_splits.py`](../lifemate/eval/make_splits.py)

Ghi danh sách mã mẫu vào [`configs/splits/`](../configs/splits/), không chép dữ liệu. Thứ tự:

1. Tách mẫu dùng 6 app held-out vào `val_unseen` (178 mẫu).
2. Gom câu gần trùng thành nhóm.
3. Chia 5 phần cân bằng số lời gọi và tỉ lệ câu hỏi lại: phần 0 là `val_seen` (274 mẫu), phần 1–4 là `train` (1 093
 mẫu).
4. Rút thêm các tập phụ: `smoke` (30 mẫu chạy thử), `asr_tune` (300), `asr_learn` (793), `*_en_mixed`.

Chi tiết và sơ đồ: [`docs/reports/benchmark/asr_v3.md`](reports/benchmark/asr_v3.md), mục "Dữ liệu cần chuẩn bị".

### 10.3. Đo ASR


| File                                                                                                              | Giữ việc gì                                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`lifemate/eval/eval_asr.py`](../lifemate/eval/eval_asr.py)                                                                                                     | Chạy một transcriber trên một split, tính WER, CER, tỉ lệ bỏ sót thực thể (tên, số trong đáp án), câu bịa, câu rỗng, thời gian; các chỉ số tiếng Anh: tỉ lệ chép đúng từ tiếng Anh, WER câu chêm, WER câu thuần Việt |
| [`lifemate/eval/foreign_words.py`](../lifemate/eval/foreign_words.py)                                                                                                | Nhận ra từ tiếng Anh trong câu tiếng Việt bằng luật (từ không dấu không ghép được thành âm tiết Việt, từ viết hoa toàn bộ…)                                                                                          |
| [`lifemate/eval/asr_tune.py`](../lifemate/eval/asr_tune.py)                                                                                                     | Dò cấu hình từng mô hình ASR trên `asr_tune` theo từng bước (benchmark lần 1, 2)                                                                                                                                     |
| [`lifemate/eval/asr_compare.py`](../lifemate/eval/asr_compare.py)                                                                                                  | So nhiều hệ thống ASR, khoảng tin cậy bootstrap theo mẫu                                                                                                                                                             |
| [`lifemate/eval/rescore_asr.py`](../lifemate/eval/rescore_asr.py)                                                                                                  | Chấm lại chữ ASR đã lưu, không chạy lại mô hình                                                                                                                                                                      |
| [`lifemate/eval/asr_merge.py`](../lifemate/eval/asr_merge.py)                                                                                                    | Ghép chữ RNNT với chữ large-v3: giữ chữ RNNT, chỉ lấy từ large-v3 các đoạn có từ tiếng Anh mới qua bộ lọc ngữ cảnh                                                                                                   |
| [`lifemate/eval/asr_v3.py`](../lifemate/eval/asr_v3.py), [`lifemate/eval/asr_v3_analysis.py`](../lifemate/eval/asr_v3_analysis.py), [`lifemate/eval/asr_v3_report.py`](../lifemate/eval/asr_v3_report.py), [`lifemate/eval/asr_v3_agent.py`](../lifemate/eval/asr_v3_agent.py), [`lifemate/eval/candidate_scoring.py`](../lifemate/eval/candidate_scoring.py), [`lifemate/eval/gpu_memory.py`](../lifemate/eval/gpu_memory.py) | Bộ chạy và phân tích của benchmark ASR lần 3                                                                                                                                                                         |


### 10.4. Phân tích lỗi agent: [`lifemate/eval/audit.py`](../lifemate/eval/audit.py)

Giải thích vì sao một lần chạy mất điểm: thiếu lời gọi, thừa lời gọi, đọc thừa, sai tham số (thiếu, thừa, sai kiểu, chỉ
khác chữ hoa). Lấy ý tưởng từ [`src/evaluation/audit_labels.py`](https://github.com/ThanhDanh1510/ZaloAI2026/blob/main/src/evaluation/audit_labels.py) của lead; ghép cặp theo tên tool thay vì theo vị trí.

---

## 11. Huấn luyện: [`lifemate/train`](../lifemate/train)

### 11.1. Dữ liệu SFT: [`lifemate/train/sft_data.py`](../lifemate/train/sft_data.py)

Mục tiêu: **mỗi dòng dữ liệu huấn luyện giống từng byte với hội thoại agent sẽ thấy lúc thi.** Cách làm của
`build_sample`:

1. Dựng prompt bằng **chính** `PromptBuilder` của agent (train mode: giấu app held-out).
2. Lấy chữ người dùng từ chữ đáp án, chữ ASR đã lưu, hoặc trộn (mặc định 70% ASR). Mẫu ASR làm mất tên hoặc số của đáp
 án thì dùng chữ đáp án.
3. Phát lại từng lời gọi đáp án trên một `ToolEnv` mới, nối lượt trợ lý (`call_text(...)`, cùng hàm agent dùng) và lượt
 kết quả (`RESULT_PREFIX + kết quả thật`). Cuối cùng nối `{"action":"done"}`. Gặp `sys.ask_missing_info` thì dừng,
 giống lúc chạy.
4. Tăng cường (`Augment`, chỉ cho tập train):
   - **phục hồi lỗi:** 3% mẫu được chèn một lời gọi thiếu tham số, kèm đúng thông báo lỗi guard thật, rồi lời gọi đúng;
    lượt sai không tính loss, nên mô hình học cách sửa chứ không học cách sai;
   - nhân bản mẫu khó;
   - thêm app gây nhiễu;
   - lọc cụm câu trùng.
5. Bỏ dòng vượt `max_model_len`, ghi lý do vào file riêng.

[`lifemate/train/asr_cache.py`](../lifemate/train/asr_cache.py) chép âm thanh một lần và lưu chữ ([`data/asr/train_asr.json`](../data/asr/train_asr.json)) để dựng dữ liệu không phải chạy lại ASR.

### 11.2. Fine-tune LLM: [`lifemate/train/swift_sft.py`](../lifemate/train/swift_sft.py), [`lifemate/train/sft.py`](../lifemate/train/sft.py), [`lifemate/train/llm_pipeline.py`](../lifemate/train/llm_pipeline.py)

- Huấn luyện chạy bằng **ms-swift** (`swift sft`), theo lựa chọn của lead.
- Config [`configs/train/llm_v1.toml`](../configs/train/llm_v1.toml):
  - LoRA r = 32, alpha = 64 trên cả 7 lớp tuyến tính;
  - QLoRA 4-bit trên T4;
  - lr 2e-4 cosine, 3 epoch, batch hiệu dụng 16, độ dài tối đa 10 240 token.
- `swift_sft.check_template` mã hoá vài dòng theo cả cách của ms-swift và cách của dự án (`sft.build_labels`), dừng nếu
khác nhau. Mục đích là chắc chắn loss chỉ tính trên lượt trợ lý.
- [`lifemate/train/llm_pipeline.py`](../lifemate/train/llm_pipeline.py) chạy cả quy trình bằng một lệnh: dữ liệu → train → chọn checkpoint theo loss trên `val_unseen` →
gộp LoRA → chạy agent thật để chấm → chọn → đăng ký mô hình. Tiến độ lưu sau mỗi giai đoạn, chạy tiếp được.

### 11.3. Fine-tune ASR: [`lifemate/train/asr_data.py`](../lifemate/train/asr_data.py), [`lifemate/train/asr.py`](../lifemate/train/asr.py), [`lifemate/train/asr_pipeline.py`](../lifemate/train/asr_pipeline.py)

Mã dựng cặp (file wav, chữ đáp án), fine-tune mô hình kiểu Whisper, chọn checkpoint theo tỉ lệ bỏ sót thực thể. Chưa chạy
thật; benchmark ASR hiện dùng mô hình có sẵn.

---

## 12. Config, ops, Docker, test

### 12.1. [`configs/`](../configs/)


| Đường dẫn                  | Nội dung                                                                 |
| -------------------------- | ------------------------------------------------------------------------ |
| [`configs/v0.toml`](../configs/v0.toml)                  | Config chạy chính: `[llm]`, `[agent]`, `[asr]`                           |
| [`configs/fake.toml`](../configs/fake.toml), [`configs/docker.toml`](../configs/docker.toml) | Config cho test với LLM giả; config trong container nộp bài              |
| `configs/env/*.toml`               | Profile máy: thư mục dữ liệu, thư mục run, nguồn token, cách lưu kết quả |
| [`configs/prompts/`](../configs/prompts/)                 | Prompt hệ thống ([`configs/prompts/system_v1.md`](../configs/prompts/system_v1.md) đang dùng)                               |
| [`configs/asr_prompts/`](../configs/asr_prompts/)             | Câu mồi tiếng Anh cho large-v3 ([`configs/asr_prompts/en_full_v1.txt`](../configs/asr_prompts/en_full_v1.txt) tốt nhất)               |
| [`configs/splits/`](../configs/splits/)                  | Danh sách mã mẫu và [`configs/splits/meta.json`](../configs/splits/meta.json)                                          |
| [`configs/train/`](../configs/train/)                   | Config huấn luyện LLM và ASR                                             |


Các giá trị quan trọng trong [`configs/v0.toml`](../configs/v0.toml):


| Khoá                             | Giá trị                                         | Lý do                                        |
| -------------------------------- | ----------------------------------------------- | -------------------------------------------- |
| `llm.model`                      | `Qwen/Qwen3-4B-Instruct-2507`, revision cố định | Dưới 5B, Apache-2.0                          |
| `llm.max_model_len`              | 16 384                                          | Trần của thể lệ                              |
| `llm.max_tokens`                 | 512                                             | Một JSON lời gọi không cần dài hơn           |
| `llm.gpu_memory_utilization`     | 0,80                                            | Chừa chỗ cho ASR                             |
| `agent.max_steps`, `max_retries` | 10, 2                                           |                                              |
| `agent.deadline_s`               | 29,0                                            | Chừa 1 giây so với luật 30 giây, tính cả ASR |
| `agent.max_result_chars`         | 4 000                                           | Cắt kết quả tool dài                         |
| `asr.backend`                    | `gold`                                          | Đổi sang `asr` khi chạy âm thanh thật        |


### 12.2. [`ops/`](../ops/)


| Thư mục         | Nội dung                                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| `ops/kaggle/*/` | Mỗi thư mục là một kernel Kaggle (script + `kernel-metadata.json`): smoke, eval, asr\_tune, asr\_v3… |
| [`ops/colab/`](../ops/colab/)    | Script chuẩn bị máy Colab và chạy nền cho benchmark ASR lần 3                                        |
| [`ops/docker/`](../ops/docker/)   | `Dockerfile`, [`ops/docker/entrypoint.sh`](../ops/docker/entrypoint.sh), [`ops/docker/smoke.sh`](../ops/docker/smoke.sh) cho bài nộp                                                |


**Container nộp bài:**

- `Dockerfile` hai tầng, Python 3.11, cài bằng `uv`, chạy offline (`HF_HUB_OFFLINE=1`), `PYTHONHASHSEED=0`.
- [`ops/docker/entrypoint.sh`](../ops/docker/entrypoint.sh) tìm đúng một `*_tools_catalog.json` trong `/data` để biết tên split, tạo liên kết thư mục cho
`ToolEnv`, đổi tên trace cũ (mỗi lần BTC chạy phải là lần chạy mới, không chạy tiếp), rồi gọi CLI ghi
`result/submission.jsonl`.

### 12.3. [`tests/`](../tests/)

- Khoảng 280 test, chạy bằng `make test` trên CPU, không cần GPU hay vLLM.
- [`tests/fakes.py`](../tests/fakes.py) có `FakeClock` (đồng hồ chỉnh tay), `FakeEnv` (mô phỏng tool trong bộ nhớ), bộ đếm token giả.
- Test cần dữ liệu thi hoặc tokenizer thật thì tự bỏ qua khi thiếu.
- Một số test bảo vệ luật thi:
  - [`tests/test_cli.py`](../tests/test_cli.py): chạy CLI với LLM giả phải được 100 điểm, và token HF không lộ ra log;
  - có test quét mã nguồn để chắc chắn chỉ [`lifemate/io/zaic.py`](../lifemate/io/zaic.py) chạm tới thư mục môi trường.
- Danh sách đầy đủ: [`tests/README.md`](../tests/README.md).

### 12.4. Lệnh `make`


| Lệnh                                                          | Việc                              |
| ------------------------------------------------------------- | --------------------------------- |
| `make install`, `make install-gpu`                            | Cài thư viện (CPU), hoặc kèm vLLM |
| `make check`                                                  | `ruff` + `mypy --strict`          |
| `make test`                                                   | pytest                            |
| `make splits`                                                 | Sinh lại [`configs/splits/`](../configs/splits/)        |
| `make run ENV=local SPLIT=train IDS=configs/splits/smoke.txt` | Chạy agent                        |
| `make score SUB=<submission.jsonl> IDS=<ids>`                 | Chấm một file nộp                 |
| `make eval ENV=colab TAG=base`                                | Đo ASR trên hai tập val           |
| `make report`                                                 | Bảng các lần chạy                 |
| `make docker-build`, `make docker-run DATA_SPLIT_DIR=<split>` | Build và chạy container như BTC   |


---

## 13. Các nguyên tắc xuyên suốt

1. **Mỗi module một việc, phụ thuộc qua giao diện.** Đổi LLM, ASR hay guard không phải sửa vòng lặp agent. Test thay mọi
 khối nặng bằng bản giả.
2. **Lúc huấn luyện giống lúc chạy từng byte.** Prompt, chuỗi phản hồi, định dạng lời gọi đều dùng chung một hàm giữa
 `agent` và `train`. Có SHA-256 của prompt trong mỗi lần chạy để truy vết.
3. **Tất định và tái lập.** Seed cố định, giải mã tham lam, JSON chuẩn tắc, thứ tự mẫu theo id, mọi lần chạy ghi đủ
 config, commit mã và phiên bản thư viện.
4. **Không mất kết quả.** Ghi nguyên tử, trace ghi ngay sau từng mẫu, file nộp luôn đủ dòng, chạy tiếp được.
5. **Lỗi lộ ra sớm.** Config kiểm tra chặt lúc nạp; `--resume` kiểm tra trước khi nạp mô hình; agent không nuốt
 `Exception`, chỉ `Runner.run_one` bắt lỗi theo từng mẫu.
6. **Tuân thủ thể lệ bằng mã, không bằng lời hứa:**
   - một `ToolEnv` mỗi mẫu (`EnvRegistry`);
   - không đọc `state.json` (chỉ [`lifemate/io/zaic.py`](../lifemate/io/zaic.py), có test);
   - hạn giờ thực thi bằng huỷ request;
   - lời gọi sai bị guard chặn trước khi tới env.
7. **Ghi nguồn khi lấy ý tưởng của lead.** Các file lấy từ repo của lead ghi rõ nguồn và điểm khác:
 [`lifemate/backends/asr_sanitizer.py`](../lifemate/backends/asr_sanitizer.py), [`lifemate/core/textsim.py`](../lifemate/core/textsim.py), [`lifemate/eval/make_splits.py`](../lifemate/eval/make_splits.py), [`lifemate/eval/audit.py`](../lifemate/eval/audit.py), [`lifemate/train/sft_data.py`](../lifemate/train/sft_data.py), [`lifemate/train/swift_sft.py`](../lifemate/train/swift_sft.py).

---

## 14. Trạng thái hiện tại và việc còn thiếu


| Phần                                             | Trạng thái                                                                                                                                                                                |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent, prompt, guard, hạn giờ, runner, chấm điểm | Chạy được, có test; chạy thật trên T4 với Qwen3-4B chưa fine-tune                                                                                                                         |
| ASR trong hệ thống chạy thật                     | `ASRTranscriber` chỉ chạy **một** engine. [`configs/v0.toml`](../configs/v0.toml) vẫn là large-v3 beam 5, không câu mồi, và mặc định `backend = "gold"`                                                         |
| Pipeline ASR tốt nhất theo benchmark             | RNNT + large-v3 với câu mồi ba phần + ghép theo luật lần 2. **Chưa có trong hệ thống chạy thật**: thiếu engine ghép `MergedASR` và khoá config chọn câu mồi (việc P8 của benchmark lần 3) |
| Điểm thi hiện tại                                | Trên `val_seen`, agent chưa fine-tune với chữ RNNT được khoảng 15 điểm; chữ ghép cao hơn khoảng 0,4–0,5 điểm, chưa có ý nghĩa thống kê. Điểm bị giới hạn chủ yếu bởi LLM                  |
| Fine-tune LLM                                    | Mã có đủ (dữ liệu, ms-swift, pipeline); chưa có kết quả trong repo này                                                                                                                    |
| Thời gian                                        | ASR p95 khoảng 11,3 giây mỗi mẫu trên T4; agent khoảng 13 giây mỗi mẫu trên T4. Chưa đo cả hệ thống trên GPU giống A30                                                                    |
| Docker                                           | Có Dockerfile và entrypoint; chưa build bản cuối với mô hình đã chọn                                                                                                                      |


**Rủi ro về thời gian cần nhớ:** `deadline_s = 29` tính cả ASR. Với pipeline ASR mới, mẫu dài có thể tốn khoảng 14 giây
cho ASR, nên LLM chỉ còn khoảng 15 giây. Hết giờ thì bước LLM đang chạy bị huỷ, mẫu nộp những lời gọi đã có.

---

## 15. Câu hỏi hay gặp khi giải thích cho team

**Hệ thống dùng định dạng nào?**
Từ 30/10/2026 (D50), mặc định là `chatml`: tool calling gốc của Qwen, tin nhắn giống hệt `build_messages` của repo lead,
nên mô hình fine-tune bằng dữ liệu của lead chạy được ngay trong hệ thống này. [`tests/test_chatml.py`](../tests/test_chatml.py) so tin nhắn với bản
chép nguyên văn hàm của lead trên cả 1 545 mẫu train. Định dạng `json` cũ (mỗi lượt một JSON `{"action": ...}` ép theo
schema) giữ lại sau `format = "json"` để tái lập các lần chạy trước. Ở `chatml`, đầu ra vẫn được ép (D51,
`[agent].constrain_tools`): khi mô hình viết `<tool_call>`, phần sau chỉ được là tên một tool đã cài và tham số đúng tên,
kiểu, `enum` (`chatml.tool_call_constraint`, structural tag của vLLM); câu kết bằng lời tự do; tham số bắt buộc không ép
để guard chặn rồi mô hình hỏi lại. Cái giá còn lại: prompt dài hơn, và tham số bị buộc theo thứ tự khai báo trong schema
(không ảnh hưởng điểm).

**Guard khác gì với `env.validate`?**
Guard `schema` gọi chính `env.validate` của BTC. Khác ở chỗ: nếu không có guard, lời gọi sai vẫn tới `env.call`, bị ghi
vào file nộp và tính vào U (lời gọi thừa) khi chấm. Guard chặn trước, nên lời gọi sai không bao giờ vào file nộp, và LLM
có cơ hội sửa.

**Tại sao không bắt `Exception` trong agent?**
Lỗi lập trình phải lộ ra khi phát triển. Chỉ `Runner.run_one` bắt lỗi, ở ranh giới từng mẫu, để lúc thi một mẫu lỗi không
làm dừng cả lần chạy.

**Làm sao chạy thử mà không có GPU?**
`make test`, hoặc chạy CLI với [`configs/fake.toml`](../configs/fake.toml) (LLM giả phát lại đáp án, chữ đáp án thay ASR): cả hệ thống chạy trên
CPU và phải được 100 điểm trên split có nhãn.

**Muốn thử một prompt hệ thống khác thì sửa gì?**
Tạo file mới trong [`configs/prompts/`](../configs/prompts/), chạy với `--set agent.system_prompt=configs/prompts/<file>.md`. Không sửa
[`configs/prompts/system_v1.md`](../configs/prompts/system_v1.md) khi đang có dữ liệu huấn luyện dựng theo file đó.

**Muốn đổi mô hình ASR thì sửa gì?**
Sửa `[asr]` trong config: `backend = "asr"`, `engine`, `model`, `generate_kwargs`. Mã không đổi. Riêng pipeline ghép hai
mô hình cần viết thêm engine `MergedASR` (mục 14).

**Repo này khác repo của lead thế nào?**
Repo của lead tập trung vào tiền xử lý dữ liệu và fine-tune ([`src/preprocessing/dataset/split.py`](https://github.com/ThanhDanh1510/ZaloAI2026/blob/main/src/preprocessing/dataset/split.py), [`src/preprocessing/dataset/builder.py`](https://github.com/ThanhDanh1510/ZaloAI2026/blob/main/src/preprocessing/dataset/builder.py), ChatML). Repo này là hệ thống chạy
thi đầy đủ (agent, guard, hạn giờ, chấm điểm, Docker) cộng benchmark ASR, và đã lấy lại các ý tưởng của lead ở phần chia
tập, lọc câu bịa, dữ liệu SFT, có ghi nguồn. Hai điểm cần thống nhất khi gộp: cách chia tập (pull request #3 trên repo
của lead), và chữ ASR dùng trong dữ liệu huấn luyện phải theo pipeline ASR sẽ dùng lúc thi.

---

## 16. Bảng tra nhanh


| Muốn tìm                         | Ở đâu                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| Vòng lặp agent                   | `lifemate/agent/agent.py › Agent.run`                                                |
| Prompt gửi cho LLM trông thế nào | `lifemate/agent/prompt.py › PromptBuilder._build`, [`configs/prompts/system_v1.md`](../configs/prompts/system_v1.md)    |
| Mô tả tool trong prompt          | `lifemate/agent/tooldocs.py › render_tools`                                          |
| Schema đầu ra, parse             | [`lifemate/agent/protocol.py`](../lifemate/agent/protocol.py)                                                         |
| Guard                            | [`lifemate/agent/guards.py`](../lifemate/agent/guards.py)                                                           |
| Hạn giờ, số bước                 | [`lifemate/agent/budget.py`](../lifemate/agent/budget.py), `lifemate/backends/llm.py › VLLMBackend._generate_until` |
| Gọi vLLM                         | `lifemate/backends/llm.py › VLLMBackend.generate`                                    |
| Chép âm thanh                    | `lifemate/backends/asr.py › ASRTranscriber.transcribe`                               |
| Lọc câu bịa, bảo vệ câu mồi      | [`lifemate/backends/asr_sanitizer.py`](../lifemate/backends/asr_sanitizer.py), [`lifemate/backends/asr_guard.py`](../lifemate/backends/asr_guard.py)                                 |
| Mở `ToolEnv`                     | [`lifemate/io/zaic.py`](../lifemate/io/zaic.py)                                                                |
| Ghi kết quả, chạy tiếp           | `lifemate/io/store.py › RunStore`                                                    |
| Chạy từng mẫu, chấm điểm         | `lifemate/run/runner.py › Runner`                                                    |
| Tạo đối tượng từ config          | `lifemate/run/factory.py › build`                                                    |
| Tham số dòng lệnh                | `lifemate/run/cli.py › parse_args`                                                   |
| Luật chấm                        | [`lifemate/eval/score.py`](../lifemate/eval/score.py)                                                             |
| Chia tập                         | [`lifemate/eval/make_splits.py`](../lifemate/eval/make_splits.py)                                                       |
| Đo ASR                           | [`lifemate/eval/eval_asr.py`](../lifemate/eval/eval_asr.py)                                                          |
| Ghép hai ASR                     | [`lifemate/eval/asr_merge.py`](../lifemate/eval/asr_merge.py)                                                         |
| Dữ liệu huấn luyện LLM           | `lifemate/train/sft_data.py › build_sample`                                          |
| Fine-tune LLM                    | [`lifemate/train/swift_sft.py`](../lifemate/train/swift_sft.py), [`lifemate/train/llm_pipeline.py`](../lifemate/train/llm_pipeline.py), [`configs/train/llm_v1.toml`](../configs/train/llm_v1.toml)        |
| Kiểu dữ liệu, enum, giao diện    | [`lifemate/core/types.py`](../lifemate/core/types.py)                                                             |
| Config và profile máy            | [`lifemate/core/config.py`](../lifemate/core/config.py), [`configs/v0.toml`](../configs/v0.toml), [`configs/env/`](../configs/env/)                         |
| Container nộp bài                | [`ops/docker/Dockerfile`](../ops/docker/Dockerfile), [`ops/docker/entrypoint.sh`](../ops/docker/entrypoint.sh)                                  |
