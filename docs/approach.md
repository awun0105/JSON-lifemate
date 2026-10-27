# Hướng làm và các quyết định

Tài liệu ghi lại cách giải bài LifeMate, các quyết định đã chốt (kèm lý do) và nhật ký thí nghiệm. Đề bài xem `overview.md`, dữ liệu xem `eda.md`, thiết kế chi tiết để implement xem `SPEC.md`. Khi hai tài liệu khác nhau thì `SPEC.md` là bản đúng.

## 1. Pipeline

### 1.1. Sơ đồ

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

### 1.2. Ví dụ từng lượt (`train_0874`)

Câu nói: _"Chuyển cho Huy một trăm rưỡi qua ví giúp em nha, nội dung ghi là tiền vé bóng đá."_

| Lượt | LLM sinh ra | Code làm gì | Kết quả đưa lại cho LLM |
|---|---|---|---|
| 1 | `{"action": "call", "name": "contacts.search", "arguments": {"query": "Huy"}}` | `env.call("contacts.search", {"query": "Huy"})` | `{"ok": true, "contacts": [{"id": "c_07", "name": "Huy"}]}` |
| 2 | `{"action": "call", "name": "ewallet.transfer", "arguments": {"contact_id": "c_07", "amount": 150000, "note": "tiền vé bóng đá"}}` | `validate` trước, hợp lệ thì `env.call(...)` | `{"ok": true, "done": true}` |
| 3 | `{"action": "done"}` | Dừng vòng lặp | |

Hai tình huống còn lại dùng cùng vòng lặp:

- **Thiếu thông tin:** LLM gọi `sys.ask_missing_info(tool_name, params)`; vòng lặp dừng ngay sau call đó.
- **Ngoài phạm vi:** LLM trả `{"action": "done"}` ngay từ lượt đầu mà không gọi gì, kết quả là `[]`.

LLM chỉ sinh JSON theo đúng một schema do vLLM ép (structured outputs), nên không có lỗi định dạng. Chi tiết ở `SPEC.md` mục 4.7.

### 1.3. Ràng buộc phải giữ khi thiết kế

| Ràng buộc | Ảnh hưởng đến thiết kế |
|---|---|
| Toàn hệ thống < 6B tham số, LLM < 5B | Chọn ASR và LLM nhỏ, ví dụ Whisper/PhoWhisper bản nhỏ + LLM 1,5B – 4B |
| Context LLM ≤ 16K token | Phải nén hướng dẫn tool; mỗi mẫu cài trung bình khoảng 31 tool, nhiều nhất 62 |
| Chạy offline | Mọi model nằm sẵn trong Docker |
| ≤ 30 giây mỗi mẫu trên 1 GPU A30 (24 GB) | Giới hạn số vòng lặp; đo thời gian sớm |
| Giải mã tất định | `temperature = 0` hoặc greedy, cố định seed |
| Chỉ truy cập dữ liệu qua `zaic_tools`, đúng một `ToolEnv` mỗi mẫu | Không đọc `state.json`; không tạo lại `ToolEnv` để thử lại |
| `zaic_tools` khác nhau theo split | Luôn nạp `zaic_tools` từ thư mục của split đang chạy (`/data` khi chạy Docker), không copy ra ngoài |

## 2. Bốn chỗ cải thiện

| # | Chỗ cải thiện | Sửa cái gì | Có phải fine-tune không | Chi phí |
|---|---|---|---|---|
| 1 | Code điều khiển | `validate` trước `call`, ép LLM sinh JSON đúng định dạng, chặn gọi lặp, giới hạn số vòng, chuẩn hóa chữ do ASR nghe ra, nén tool | Không | Rẻ |
| 2 | System prompt | Quy tắc, ví dụ mẫu (few-shot), cách trình bày tool | Không: model giữ nguyên | Rẻ |
| 3 | Fine-tune LLM | Các con số bên trong LLM (thường chỉ train miếng dán LoRA) | Có | Đắt hơn |
| 4 | Fine-tune ASR | Các con số bên trong model ASR | Có | Đắt hơn |

### 2.1. Thứ tự làm

1. Agent đơn giản nhất chạy được, dùng transcript có sẵn (`audio_text`) để tạm bỏ qua ASR. Mục đích là có điểm khởi đầu.
2. Cải thiện code điều khiển và system prompt.
3. Thử vài LLM có sẵn dưới 5B, giữ bản tốt nhất.
4. Thay transcript bằng chữ do ASR nghe ra, đo mức tụt điểm.
5. Fine-tune LLM nếu bảng lỗi cho thấy quyết định sai hoặc điền sai tham số nhiều.
6. Fine-tune ASR nếu mức tụt điểm ở bước 4 lớn.
7. Đóng gói Docker: bản chạy được đầu tiên trước 05/11/2026, bản cuối trước 16/11/2026 (mục 6).

Hạ tầng đo lường (mục 4) làm cùng lúc với bước 1, không tách thành việc riêng.

### 2.2. Dữ liệu dùng cho fine-tune

- **LLM:** dựng lại mỗi mẫu train thành một cuộc hội thoại nhiều lượt, gồm prompt, call đúng (`label.tool_calls[i]`), kết quả (phát lại call qua `zaic_tools` để khớp từng byte với lúc chạy; `label.tool_results[i]` chỉ dùng để đối chiếu), call tiếp theo, …, `{"action": "done"}`. Chỉ tính loss trên các lượt của trợ lý. Chi tiết ở `SPEC.md` mục 6.2. Model học được kỹ năng chung (đọc hướng dẫn, tra cứu lấy ID trước, đổi số nói bằng chữ, khi nào hỏi lại, khi nào không làm gì), chứ không học thuộc từng app.
- **ASR:** dùng cặp (file wav, `audio_text`), gồm cả các lượt user trong history.
- **Dữ liệu tổng hợp:** được tạo bằng model lớn hoặc TTS trong lúc phát triển, nhưng không được dùng chúng khi chạy bài thi. Không gán nhãn `public_test` hay `private_test`.

## 3. Đánh giá

### 3.1. Vì sao phải tự đánh giá trên train

- `public_test` không có đáp án, nên chỉ xem được điểm trên leaderboard, tối đa 5 lần mỗi ngày.
- Thứ hạng chung cuộc chỉ tính theo `private_test`. Bộ này phát ra sau khi giải pháp đã đóng băng và nhiều khả năng có app chưa từng thấy (public đã có 11 app train không có).
- Vì vậy cần một bộ đánh giá local mô phỏng tình huống gặp app lạ.

### 3.2. Cách chia train (`lifemate/eval/make_splits.py`)

| Phần | Số mẫu | Cách lấy | Đo cái gì |
|---|---|---|---|
| `train` | 1.098 | Phần còn lại | Dùng để học, lấy ví dụ mẫu cho prompt |
| `val_seen` | 193 | 15% phần còn lại, chia đều theo loại đáp án và theo có/không có history | Khả năng với app quen |
| `val_unseen` | 254 | Mọi mẫu có đáp án gọi tới 8 app bị rút, cộng 10 mẫu gold `[]` mà yêu cầu nói về app bị rút (gắn tay trong `UNSEEN_OUT_OF_SCOPE`) | Khả năng gặp app lạ, gần với điểm private |
| `smoke` | 30 | Lấy phân tầng từ `val_seen` | Chạy thử nhanh sau mỗi lần sửa |

- **8 app bị rút:** door_lock, fitness, home_security, learning, library, loyalty, medication, robot_vacuum. Đều là app chỉ có ở train, nên có thể không xuất hiện lại. Các app dùng chung (contacts, calendar, …) được giữ lại để học.
- **Quy tắc bắt buộc:** 490 mẫu train có *cài* một trong 8 app bị rút (dù không dùng). Khi dựng prompt cho các mẫu này phải **xóa 8 app đó khỏi `installed_apps`**, nếu không model sẽ đọc thấy hướng dẫn của chúng và chúng không còn "lạ". Việc xóa không làm sai đáp án vì đáp án không gọi tới các app đó. Ngoài ra 65 mẫu train có *history* gọi tool của app bị rút; các mẫu này bị bỏ khi dựng dữ liệu fine-tune hoặc few-shot (`SPEC.md` mục 4.5).
- `val_unseen` chỉ có 10 mẫu `[]`, đủ để phát hiện lỗi lớn (gặp app lạ là gọi bừa) nhưng không đủ để so hai phiên bản chênh ít điểm. Khả năng nhận ra câu ngoài phạm vi đo chủ yếu trên `val_seen` (15 mẫu) và public.
- **Khi chỉ làm prompt (chưa fine-tune):** không lấy ví dụ mẫu từ 8 app bị rút và không viết quy tắc riêng cho chúng.
- Danh sách id nằm ở `configs/splits/`, cố định với seed = 42. Mọi thí nghiệm dùng chung một cách chia để điểm so sánh được với nhau.

### 3.3. Bộ chấm local (`lifemate/eval/score.py`)

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

### 3.4. Cách đọc báo cáo

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
- Ngưỡng "gần" hay "chênh nhiều" chưa chốt; sẽ đặt sau khi có vài lần chạy thật để biết độ dao động. `val_unseen` chỉ có 244 mẫu nên điểm có thể lệch vài điểm giữa các lần chạy.

### 3.5. Kiểm tra trước khi nộp public_test

Không chấm được điểm public_test ở local, nhưng trước khi nộp nên kiểm tra:

1. **Schema:** `python -m zaic_tools --split public_test --root . --validate submission.jsonl`, chạy với `PYTHONPATH=public_test` từ thư mục gói dữ liệu.
2. **Đủ dòng:** đúng 634 dòng, không trùng `id`.
3. **Phân bố so với train:** tỉ lệ `[]` quanh 7%, tỉ lệ hỏi lại quanh 14%, số call trung bình quanh 2,3. Lệch quá nhiều là dấu hiệu agent hành xử bất thường.
4. **Đọc vài mẫu bằng mắt** để tìm lỗi. Không dùng để chấm điểm hay chọn phương án, vì tự gán nhãn public_test để tinh chỉnh là bị cấm.

## 4. Hạ tầng đo lường

Mục tiêu: mỗi lần đổi prompt, code hay model đều đo được điểm, thời gian, token và biết sai ở mẫu nào. Mọi thứ lưu thành file, chưa cần MLflow.

### 4.1. Mỗi lần chạy là một thư mục

```
runs/2026-10-28_v0-qwen4b_val-seen/
├── config.json       model + revision, adapter, hash prompt, tham số decode, split, git commit, seed, loại GPU
├── trace.jsonl       mỗi mẫu một dòng: toàn bộ các bước agent đã làm (mục 4.2)
├── submission.jsonl  đúng định dạng nộp bài
├── score.txt         báo cáo của lifemate/eval/score.py
└── summary.json      số liệu tổng hợp (mục 4.4)
artifacts/runs/index.csv        mỗi lần chạy thêm một dòng, ghi tự động; bảng ở mục 8 chỉ chép lại các lần chạy đáng nhớ
```

`artifacts/runs/` không đưa vào git (chứa nội dung dữ liệu thi).

### 4.2. Logging (`trace.jsonl`)

Mỗi mẫu ghi:

- Chữ ASR nghe ra (và `audio_text` nếu có).
- Từng lượt LLM: số token prompt, số token sinh ra, thời gian, đầu ra thô, call đã tách được.
- Từng call: tham số, kết quả `env` trả về, có bị từ chối không.
- Lỗi (nếu có), tổng thời gian, `state_hash`.

### 4.3. Phân loại lỗi

Mỗi mẫu bọc riêng trong `try/except`: một mẫu hỏng thì vẫn ghi kết quả cho mẫu đó và chạy tiếp, không bao giờ dừng cả lần chạy.

| Mã lỗi | Ý nghĩa |
|---|---|
| `parse_error` | Đầu ra LLM không tách được thành call |
| `unknown_tool` | Gọi tool không thuộc các app đã cài |
| `invalid_args` | `env.validate` không chấp nhận tham số |
| `env_exception` | `zaic_tools` báo lỗi |
| `max_turns` | Vượt số vòng lặp cho phép |
| `context_overflow` | Prompt vượt 16K token |
| `timeout` | Mẫu bị dừng vì hết thời gian (mục 5.4) |

Cuối mỗi lần chạy in bảng đếm từng loại lỗi kèm vài id mẫu.

### 4.4. Thời gian, token và điểm (`summary.json`)

- **Thời gian:** tách ASR, LLM (cộng các lượt), tool và tổng mỗi mẫu; báo p50, p95, max và số mẫu vượt 30 giây. Ghi kèm loại GPU: T4 (Kaggle) chậm hơn A30 khoảng 2 – 3 lần, nên trên T4 lấy ngưỡng cảnh báo khoảng 60 giây.
- **Token:** số token prompt lớn nhất so với giới hạn 16K; cảnh báo khi vượt khoảng 14K.
- **Điểm:** score tổng và score theo loại đáp án, tách riêng `val_seen` và `val_unseen`.

### 4.5. Benchmark và so sánh

- **Bộ cố định:**
  - `smoke`: `configs/splits/smoke.txt`, 30 mẫu lấy phân tầng từ `val_seen`, chạy vài phút, dùng sau mỗi lần sửa.
  - `val_seen` và `val_unseen`: chạy đầy đủ.
- **`compare.py runA runB`:** liệt kê mẫu trước đúng nay sai và ngược lại. Một con số trung bình không cho thấy điều này.
- **`show.py <run> <id>`:** in trace của một mẫu dạng hội thoại, dễ đọc hơn JSON thô.
- **Đo riêng ASR:** WER/CER trên các cặp (wav, `audio_text`) của train. So điểm agent khi dùng `audio_text` và khi dùng chữ ASR; khoảng chênh là phần mất do ASR.

### 4.6. Test (pytest, chạy trên CPU, không cần model)

- **Scorer:** chấm đáp án ra 100, nộp toàn `[]` ra 7,18.
- **Tách đầu ra LLM:** một bộ đầu ra lỗi mẫu (thiếu ngoặc, thừa chữ, JSON sai).
- **Dựng prompt:** đã xóa 8 app bị rút khỏi `installed_apps`; không vượt ngân sách token.
- **Vòng lặp với LLM giả (FakeLLM):** LLM giả trả đúng đáp án, điểm phải ra 100. Kiểm tra cả vòng lặp mà không cần GPU.
- **Tính tất định:** chạy cùng 5 mẫu hai lần, kết quả giống hệt nhau.

### 4.7. Quản lý phiên bản

| Thành phần | Cách quản lý |
|---|---|
| Code | git; mỗi lần chạy ghi commit hash vào `config.json` |
| Prompt | file trong `configs/prompts/`, nằm trong git; mỗi lần chạy ghi hash prompt |
| Dữ liệu fine-tune | `data/sft/v1.jsonl`, `v2`, … kèm hash và script tạo ra |
| Model fine-tune | tên dạng `ft-qwen4b-sftv1-r01`, kèm `train_config.json` (model gốc, phiên bản dữ liệu, siêu tham số, seed, commit, loss); lưu trên Kaggle Models hoặc kho HF private, không đưa vào git |
| Đường cong loss | tùy chọn W&B, không bắt buộc |
| Lần nộp public | `runs/public_submissions.csv`: ngày, run đã nộp, điểm public. Theo dõi điểm val và điểm public có tăng giảm cùng chiều không; nếu không, bộ val chưa phản ánh đúng bài thi |

### 4.8. Thứ tự xây

1. `config` + `trace` + `summary` + phân loại lỗi: viết cùng agent v0.
2. Test scorer và FakeLLM.
3. `compare.py`, `show.py`: khi đã có từ hai lần chạy trở lên.
4. Quản lý phiên bản fine-tune: khi bắt đầu fine-tune.

## 5. Nộp bài và tuân thủ

### 5.1. Docker

- BTC mount thư mục test (cùng cấu trúc `public_test`) vào **`/data`**, chạy inference script, lấy kết quả tại **`result/submission.jsonl`** (`overview.md` mục 7). Chi tiết còn lại theo hướng dẫn riêng của BTC.
- Trọng số model nằm sẵn trong image; không tải gì lúc chạy.
- Chạy thử bằng `docker run --network none` để chắc không có thư viện nào ngầm tải file (tokenizer, ASR, …).
- Đo thời gian nạp model và thời gian từng mẫu trên GPU 24 GB thuê theo giờ (Kaggle/Colab không chạy được Docker).
- Tạo checksum và link Google Drive công khai theo yêu cầu BTC. Được nộp tối đa hai (02) lần.

### 5.2. Training code và data phải chạy lại được

BTC yêu cầu nộp cả training code, inference code và data. Fine-tune không được chỉ nằm trong notebook: phải có script chạy từ đầu đến cuối (tạo dữ liệu → train → merge → model).

### 5.3. Tự kiểm tra luật thi

Viết thành kiểm tra tự động trong code chạy:

- Đúng một `ToolEnv` mỗi mẫu.
- Mọi call đã thực hiện trên `ToolEnv` đều có trong submission.
- Không mở file nào trong `*_environment/`.
- File nộp: đủ mọi id, không trùng, JSON hợp lệ, đúng schema (mục 3.5).

### 5.4. Chốt chặn thời gian

Agent tự theo dõi thời gian: khi một mẫu chạy khoảng 25 giây thì dừng vòng lặp và trả về những call đã có. Như vậy mẫu đó vẫn có thể được một phần điểm, thay vì bị 0 vì quá 30 giây.

### 5.5. Danh sách model được dùng

Chọn ngày 27/10/2026 theo điều kiện ở SPEC 6.0. Số tham số lấy từ metadata safetensors trên HF.

| Model | Vai trò | Số tham số | Ngày phát hành | Giấy phép | Revision |
|---|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | LLM (mặc định) | 4,02B | 05/08/2025 | Apache-2.0 | `cdbee75f` |
| `vinai/PhoWhisper-medium` | ASR (mặc định) | 0,77B | 18/02/2024 | BSD-3 | `55a7e3eb` |

Tổng 4,79B < 6B, LLM 4,02B < 5B. Ứng viên A/B (chưa dùng khi nộp): LLM `Qwen/Qwen3.5-4B` (4,66B kể cả vision encoder, Apache-2.0, 02/03/2026); ASR `khanhld/chunkformer-ctc-large-vie` (0,11B, CC-BY-NC-4.0, 02/2025) và `5CD-AI/VieGem-ASR` (0,36B, CC-BY-NC-SA-4.0, 19/10/2026). Hai ASR này có `engine` riêng trong `lifemate/backends/asr.py`; giấy phép phi thương mại phải được BTC xác nhận trước khi nộp.

#### 5.5.1. Vì sao chọn như vậy

**LLM.** Không có LLM ≤ 5B nào phát hành năm 2026 được huấn luyện riêng cho tiếng Việt và tool calling (Sailor2 1B/8B, SeaLLMs-v3 1.5B, PhoGPT-4B đều cũ hơn và không có tool calling). Trong họ Qwen, bản Instruct-2507 được chọn thay bản Thinking-2507 (BFCL-v3 61,9 so với 71,2) vì:

- Ngân sách 30 giây/mẫu đã gồm ASR cho tối đa 13 file audio và 3–6 bước gọi tool. Mỗi bước thinking sinh thêm vài trăm đến vài nghìn token không ép được bằng JSON schema, nên thời gian không đoán trước được (quyết định S6).
- Nhãn SFT là chuỗi tool call trực tiếp, không có lời suy nghĩ. Fine-tune bản Thinking bằng dữ liệu không có think sẽ dạy nó bỏ think, mất luôn lợi thế của bản đó.
- Instruct-2507 là Transformer chuẩn, chạy trên vLLM ≥ 0,8.5 và fp16 trên T4; Qwen3.5-4B dùng Gated DeltaNet, cần vLLM ≥ 0,17 (bản khóa 0,32 đủ) nhưng chưa rõ fp16/T4, và 4,66B sát trần 5B nếu BTC đếm cả vision encoder. Nên Qwen3.5-4B chỉ là ứng viên A/B sau khi có baseline.

**ASR.** PhoWhisper-medium được chọn mặc định vì cắm thẳng vào `HFSeq2SeqASR` (transformers pipeline) không cần code thêm, giấy phép BSD, và đầu ra chữ thường, số viết bằng chữ, cùng văn phong với `audio_text`. Điểm yếu: WER khi xen tiếng Anh cao (PhoWhisper-large 31% trên ViMedCSS) và Whisper hay bịa chữ ở đoạn im lặng. ChunkFormer (N-WER 6,89, tốt nhất trong bài EACL 2026, nhẹ 110M) và VieGem-ASR (WER trung bình 6,34, xen tiếng Anh 14,3%) tốt hơn trên benchmark công khai nhưng cần adapter và vướng giấy phép phi thương mại. Cả ba được đo trên `val_seen` / `val_unseen` theo SPEC 6.4.2 trước khi chốt.

#### 5.5.2. Kế hoạch fine-tune cho từng model

**LLM: Qwen3-4B-Instruct-2507, QLoRA (SPEC 6.2).**

| Việc | Cách làm | Vì sao |
|---|---|---|
| Phương pháp | QLoRA: model gốc nf4, LoRA r=16, alpha=32 trên 7 ma trận attention + MLP; Unsloth trên T4 fp16 | Full fine-tune 4B cần > 40 GB VRAM, không vừa T4/Kaggle. LoRA trên cả attention và MLP giữ được kiến thức chung, chỉ dạy định dạng và quy tắc gọi tool. 1098 mẫu train là ít, r=16 đủ và ít overfit hơn r lớn |
| Dữ liệu | 1098 mẫu `configs/splits/train.txt` dựng bằng đúng `PromptBuilder` lúc chạy, xóa 8 app bị rút; transcript là `audio_text`; kết quả tool phát lại qua `zaic_tools` | Prompt lúc train và lúc chạy phải giống từng byte, nếu không model học một phân bố và chạy trên phân bố khác. Xóa app bị rút để `val_unseen` vẫn đo được khả năng đọc app lạ |
| Che loss | Chỉ trên token của lượt assistant | Không che thì model học thuộc mô tả tool và kết quả tool, làm hỏng khả năng đọc tool docs mới |
| Chat template | Render với tham số như lúc chạy; Instruct-2507 không có chế độ thinking nên không cần `enable_thinking` | Giữ đúng token đặc biệt mà vLLM sẽ sinh lúc chạy |
| Siêu tham số ban đầu | lr 2e-4, 2 epoch, batch 1 × grad_accum 8, max_len 16384, seed 0 | Mặc định quen thuộc của QLoRA cho SFT nhỏ; chốt lại sau khi có số đo v0 |
| Chọn checkpoint | 2 checkpoint có loss `eval_unseen` thấp nhất, merge, chạy agent thật, lấy điểm `val_unseen` cao nhất | Loss không phản ánh đúng điểm thi; nhưng chỉ chạy agent thật 2 lần để không chọn model theo val quá nhiều |
| Đưa vào hệ thống | Merge adapter vào bf16, lưu safetensors, trỏ `[llm].model` (Docker: `/app/models/llm`) | vLLM nạp model đã merge như model thường, không tốn thêm thời gian chạy, Docker đơn giản |
| Giai đoạn 2 (nếu bảng lỗi cần) | `transcript_source = "asr"`: dựng lại dữ liệu bằng chữ ASR thật; thêm dữ liệu tổng hợp từ catalog train | Model quen lỗi ASR thật; chỉ làm khi điểm mất do ASR lớn |

**ASR: PhoWhisper-medium, full fine-tune hai giai đoạn (SPEC 6.4). Chỉ làm khi điểm mất do ASR ≥ 3 hoặc tỉ lệ nghe sai thực thể ≥ 10%.**

| Việc | Cách làm | Vì sao |
|---|---|---|
| Phương pháp | Full fine-tune, không LoRA. Giai đoạn 1: đóng băng encoder, train decoder. Giai đoạn 2 (nếu WER còn cao): mở encoder với lr nhỏ hơn 10 lần | 0,77B vừa T4 với fp16 + gradient checkpointing. Decoder quyết định cách viết số, tên riêng, từ tiếng Anh, là thứ ta cần chỉnh để khớp `audio_text`; encoder đã học 844 giờ tiếng Việt, mở sớm với 3547 cặp dễ làm hỏng |
| Dữ liệu | 3547 cặp (wav, `audio_text`) từ lượt cuối và mọi lượt user trong history của `configs/splits/train.txt`; chia theo id mẫu | `audio_text` là nhãn duy nhất có. ASR sau fine-tune đổi cách viết số, nên phải train lại LLM v2 trên chữ ASR mới |
| Tăng dữ liệu | SpecAugment; TTS chỉ cho loại lỗi mà bảng đo chỉ ra (tên người, địa danh, số tiền, từ tiếng Anh), câu lấy từ `audio_text` train rồi thay thực thể | Dữ liệu ít, SpecAugment chống overfit. TTS được phép lúc phát triển, cấm lúc thi |
| Siêu tham số ban đầu | lr 1e-5, 3 epoch, batch 8 × grad_accum 2, greedy decode, seed 0 | lr nhỏ vì full fine-tune trên model đã tốt; 3 epoch trên 3547 cặp là khoảng 660 bước |
| Chọn checkpoint | Theo tỉ lệ nghe sai thực thể trên `val_unseen`, rồi WER; checkpoint tốt nhất chạy agent thật | Lỗi thực thể là lỗi làm sai tham số tool, ảnh hưởng điểm trực tiếp; WER chung không phản ánh |
| Đưa vào hệ thống | Lưu snapshot transformers, trỏ `[asr].model` (Docker: `/app/models/asr`), giữ `engine = "hf_seq2seq"` | Cùng đường nạp với model gốc |

**Thứ tự làm, rút ra từ phân tích nguyên lý gốc (27/10/2026, lập luận đầy đủ ở mục 9):**

1. Cắm ASR gốc, đo WER, tỉ lệ nghe sai thực thể, thời gian.
2. LLM v1: QLoRA trên `audio_text` gold.
3. **LLM v2: dựng lại dữ liệu với chữ ASR thật** (`transcript_source = "asr"`), nhãn vẫn gold. Lý do: danh bạ, tên thiết bị, app đã cài nằm trong ngữ cảnh mẫu mà chỉ LLM nhìn thấy, ASR thì không. LLM train trên chữ ASR học được quy ước "chọn tên trong danh bạ gần nhất với chữ nghe được", tức sửa lỗi nghe bằng thông tin chỉ nó có, không tốn thêm thời gian lúc chạy.
4. **Chỉ sau LLM v2** mới đo điểm mất do ASR và tỉ lệ nghe sai thực thể để quyết định fine-tune ASR. Đo sớm hơn thì con số gồm cả phần LLM v2 tự sửa được, dẫn tới tốn 1–2 ngày GPU vào thứ có thể biến mất.
5. Nếu fine-tune ASR: làm dữ liệu TTS nhắm vào ba nhóm thực thể (tên người, thiết bị và địa điểm, số tiền và giờ) với encoder đóng băng **trước** khi nghĩ tới mở encoder, vì đó là lớp rẻ, rủi ro thấp và đo được ngay; mở encoder là bước rủi ro cao nhất với 13 giờ audio.

Điểm yếu của lập luận: với địa chỉ tự do hay tên không có trong danh bạ thì LLM không có gì để đối chiếu, chỉ ASR mới cứu được; bảng lỗi sau LLM v2 sẽ cho biết nhóm này chiếm bao nhiêu.

**Nếu đổi sang ChunkFormer hoặc VieGem-ASR:** cả hai là CTC, không có decoder tự hồi quy, nên fine-tune là train CTC loss trên cùng 3547 cặp với encoder đóng băng phần lớn (VieGem chỉ mở 4 lớp conformer trên cùng như tác giả làm). Cần viết thêm script train riêng vì `lifemate/train/asr.py` hiện chỉ hỗ trợ seq2seq của transformers. Chỉ làm khi số đo cho thấy hai model này thắng PhoWhisper trên `val_unseen` **và** BTC chấp nhận giấy phép phi thương mại.

### 5.6. Vòng private (20/11 – 23/11)

Đội tự chạy giải pháp đã nộp trên private test trong 3 ngày. Chuẩn bị trước:

- Máy chạy: GPU thuê nào, thuê từ khi nào.
- Ước lượng tổng thời gian chạy toàn bộ private test.
- Một lệnh duy nhất từ dữ liệu ra `submission.jsonl`; chạy tập dượt trên public_test trước ngày 16/11.

### 5.7. Bảo mật

- Dữ liệu thi chỉ để ở chỗ riêng tư (Kaggle Dataset ở chế độ Private, Drive cá nhân); không đưa lên git.
- Không ghi HF token hay khóa API vào code hoặc notebook.

## 6. Mốc thời gian (đề xuất)

| Thời gian | Việc cần xong |
|---|---|
| 27/10 – 30/10 | Agent v0 dùng `audio_text` + hạ tầng mục 4.8 bước 1 – 2; có điểm đầu tiên trên `val_seen`, `val_unseen` |
| 31/10 – 03/11 | Cải thiện prompt và code điều khiển; thêm ASR, đo WER và mức tụt điểm |
| trước 05/11 | Docker chạy được offline trên GPU thuê; nộp public lần đầu từ chính Docker đó |
| 05/11 – 12/11 | Fine-tune LLM (QLoRA), fine-tune ASR nếu cần; script train chạy lại được |
| 13/11 – 15/11 | Chốt phiên bản, kiểm tra Docker lần cuối, tập dượt quy trình vòng private |
| 16/11, 23:59 | Nộp Docker (tối đa hai lần) |
| 20/11 – 23/11 | Chạy private test, nộp kết quả trước 23:59 ngày 23/11 |

## 7. Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|---|---|---|
| 26/10/2026 | Dùng LLM đọc hướng dẫn tool, không làm mô hình phân loại theo từng app | Public (và nhiều khả năng private) có app chưa từng thấy |
| 26/10/2026 | Giữ `zaic_tools` nguyên chỗ trong từng split, không copy hay sửa | Mỗi split có bản riêng (`utils.py` tính hash khác nhau); server luôn dùng bản gốc |
| 26/10/2026 | Chia train / val_seen / val_unseen như mục 3.2 | Cần ước lượng điểm khi gặp app lạ; không có đáp án public để tự chấm |
| 26/10/2026 | Chưa làm `evaluate.py` (chấm hai bộ và tự nhận xét) | Chưa có agent; chạy `score.py` hai lần là đủ. Làm khi bắt đầu thử nhiều phiên bản mỗi ngày |
| 26/10/2026 | Không dùng catalog của public_test để tự tạo mẫu huấn luyện | Dễ bị coi là vi phạm luật cấm gán nhãn public_test; nếu cần thì hỏi BTC qua aichallenge@zalo.me trước |
| 27/10/2026 | Phát triển trên Kaggle (T4 16 GB); thuê GPU 24 GB vài giờ để thử Docker | Kaggle đủ chạy model 4B ở fp16 và QLoRA; Kaggle/Colab không chạy được Docker |
| 27/10/2026 | Fine-tune bằng QLoRA (Unsloth), khi nộp thì merge adapter vào model gốc bf16 | LoRA trên T4 16 GB dễ hết bộ nhớ và T4 không có bf16; chất lượng QLoRA gần bằng LoRA |
| 27/10/2026 | Hạ tầng đo lường dạng file (`artifacts/runs/`), không dùng MLflow | Đủ cho một đội nhỏ, không cần dựng server |
| 27/10/2026 | Chạy LLM bằng **vLLM** trong cùng process, không dùng `transformers`, llama.cpp hay Ollama khi nộp | A30 (sm80) hỗ trợ chế độ batch-invariant của vLLM để giữ tất định; prefix cache giảm thời gian prompt dài; nạp thẳng model đã merge LoRA. llama.cpp là dự phòng nếu vLLM không chạy được trong Docker; Ollama thêm một lớp trung gian không cần thiết |
| 27/10/2026 | LLM sinh JSON tự định nghĩa (`{"action": "call"|"done", ...}`), vLLM ép theo JSON schema; mô tả tool dạng rút gọn trong prompt | Tiết kiệm token so với schema OpenAI đầy đủ, không hỏng JSON, không phụ thuộc chat template từng model |
| 27/10/2026 | Không làm lockbox nội bộ; dùng public leaderboard làm bộ khóa, chỉ nộp ở mốc lớn | Public có 600 mẫu, 11 app lạ, không có nhãn, do BTC tạo cùng cách với private. Lockbox tự tách chỉ khoảng 130 mẫu, điểm dao động lớn và làm mất dữ liệu train |
| 27/10/2026 | Thêm 10 mẫu gold `[]` gắn tay vào `val_unseen`; thêm `configs/splits/smoke.txt` | `val_unseen` trước đó không có mẫu ngoài phạm vi nào |
| 27/10/2026 | Chưa chọn LLM và ASR; code không gắn với model nào, chọn theo điều kiện ở `SPEC.md` mục 6.0 | Để chọn sau khi có hạ tầng đo mà không phải sửa code |
| 27/10/2026 | Không làm Normalizer ở v0; LLM tự đổi số theo quy tắc trong system prompt | LLM và ASR đều nhắm tới `audio_text` nguyên văn; chuẩn hóa trước LLM làm lệch phân bố đã train |
| 27/10/2026 (sau đo Kaggle) | Giữ nguyên chữ ASR nghe ra, không chuẩn hóa; LLM nộp bài học trên chữ ASR (v2) | `audio_text` viết số bằng chữ số, PhoWhisper viết bằng chữ; BTC không chấm chữ ASR. Sửa giả định "ASR nhắm tới `audio_text` nguyên văn" ở dòng trên |
| 27/10/2026 | ASR chạy bfloat16/float16 trên GPU (`asr.dtype = "auto"`) | Mặc định float32 tốn khoảng 3 GB, hết bộ nhớ cạnh vLLM trên T4; WER float32 với auto đo bằng kernel Kaggle |
| 27/10/2026 | Python 3.11 + `uv`, ruff + mypy strict, pre-commit; lấy quy ước kỹ thuật từ `ai-kit`, không lấy layer ứng dụng | Kit là app cloud FastAPI + LangGraph, không khớp với agent offline |

## 8. Nhật ký thí nghiệm

Mỗi lần chạy agent ghi một dòng. Điểm lấy từ `lifemate/eval/score.py` với `--ids configs/splits/val_seen.txt` và `--ids configs/splits/val_unseen.txt`.

| Ngày | Phiên bản | Thay đổi | ASR / LLM | val_seen | val_unseen | Public | Nhận xét |
|---|---|---|---|---|---|---|---|
| 27/10/2026 | v0 pipeline check | Agent v0 + hạ tầng đo xong (`c720d5e`) | gold / FakeLLM (phát lại gold) | 100,00 | 100,00 | | Chỉ kiểm vòng lặp, guard, ghi file và bộ chấm; chưa có model thật |

## 9. Phân tích nguyên lý gốc về fine-tune (27/10/2026)

Chuỗi lập luận dẫn tới mục 5.5.2. Đọc mục này khi muốn biết **vì sao** chứ không chỉ **làm gì**.

### 9.1. Bản chất bài toán

Bỏ hết chi tiết kỹ thuật thì bài toán là: một người nói một câu tiếng Việt, hệ thống phải để lại trạng thái các
app đúng như người đó muốn. Điểm chấm có hai phần: hash trạng thái cuối (đúng hết hoặc 0) và F1 tập tool call
(đúng từng phần). Phần hash là "tất cả hoặc không": sai một ký tự trong tên người nhận, sai một số trong số tiền
là mất trọn phần hash của mẫu đó.

Chân lý nền, tức những điều không thể giảm thêm được nữa:

1. Điểm phụ thuộc vào **giá trị tham số chính xác từng ký tự** và **đúng tool, đúng thứ tự**. Không phụ thuộc
   vào câu trả lời hay cách diễn đạt.
2. Mọi thông tin để làm đúng nằm trong đúng ba nguồn: **âm thanh**, **ngữ cảnh mẫu** (danh bạ, thiết bị, app
   đã cài), **tài liệu tool**. Không có nguồn thứ tư.
3. Dữ liệu có nhãn chỉ có 1098 mẫu, khoảng 13 giờ audio. Model gốc đã học từ hàng nghìn tỉ token và hàng trăm
   nghìn giờ audio. Với lượng này ta không "dạy kiến thức" được, chỉ **dạy quy ước**.
4. Lúc thi sẽ gặp app lạ (ta tự rút 8 app ra khỏi train để mô phỏng). Thứ cần học là **cách đọc tài liệu rồi
   hành động**, không phải thuộc lòng app.
5. 30 giây mỗi mẫu trên A30, gồm cả nghe lẫn suy luận.

Ví dụ đời thường: thuê một kế toán giỏi (model gốc) về công ty. Không ai dạy lại kế toán từ đầu; chỉ dạy mẫu
biểu, quy ước đặt tên, quy trình duyệt. Và ta muốn người đó xử lý được cả loại chứng từ mới chỉ bằng cách đọc
hướng dẫn kèm theo.

### 9.2. Giả định bị thách thức

| Giả định | Thách thức | Kết luận |
|---|---|---|
| "Fine-tune để model hiểu tiếng Việt hơn" | 13 giờ và 1098 câu không dạy được tiếng Việt. Qwen đã hiểu ở mức đủ. | Bỏ. Mục tiêu là quy ước đầu ra. |
| "Lỗi tên riêng là lỗi ASR, phải sửa ở ASR" | Chân lý 2: danh bạ nằm trong ngữ cảnh mẫu, LLM thấy, ASR không thấy. | Sửa: phần lớn lỗi tên riêng nên sửa ở LLM (mục 9.4). |
| "Model lớn hơn hoặc có thinking sẽ tốt hơn" | Chân lý 5 và 1: thinking tốn thời gian không đoán được; lỗi của ta là lỗi quy ước, không phải lỗi suy luận sâu. | Giữ quyết định bỏ thinking. |
| "Càng nhiều epoch, loss càng thấp, càng tốt" | Chân lý 4: thuộc catalog train làm hỏng app lạ. | Chốt bằng điểm `val_unseen`, không bằng loss. |
| "WER thấp là ASR tốt" | Chân lý 1: chỉ từ nằm trong tham số tool mới ảnh hưởng điểm. | Đo tỉ lệ nghe sai thực thể. |

### 9.3. LLM: suy từ chân lý lên cách fine-tune

#### 9.3.1. Bản chất

Một LLM decoder là hàm ước lượng xác suất token kế tiếp p(token | mọi token trước). Fine-tune nghĩa là dịch hàm
này một chút để trên đầu vào giống bài của ta, token có xác suất cao nhất là token ta muốn. Nó không "hiểu thêm",
nó **lệch ưu tiên**.

#### 9.3.2. Vì sao LoRA, vì sao QLoRA, vì sao r = 16

Từ chân lý 3 và 4: lượng thông tin cần đưa vào rất nhỏ (một bộ quy ước), thứ cần bảo toàn rất lớn (khả năng đọc
hiểu chung). Hai yêu cầu này dẫn thẳng tới **thay đổi có hạng thấp**. LoRA (Low-Rank Adaptation) đóng băng
trọng số W và học thêm ΔW = B·A với A ∈ ℝ^{r×d}, B ∈ ℝ^{d×r}, r nhỏ. Với d = 2560 và r = 16, ta chỉ cho phép
model thay đổi theo 16 hướng trong 2560 hướng có thể của mỗi ma trận. Đó là ràng buộc có chủ đích, giống việc cho
nhân viên mới chỉ được sửa mẫu biểu, không được sửa sổ cái.

Số tham số học được: 2·r·d cho mỗi ma trận d×d thay vì d². Trên 7 ma trận (q, k, v, o, gate, up, down) × 36 lớp
của Qwen3-4B, tổng vào khoảng 60–70M, tức 1,5% model. Dữ liệu tính loss chỉ khoảng 100–200K token (1098 mẫu ×
~3 lượt assistant × vài chục token), nên ta đã ở chế độ "tham số nhiều hơn dữ liệu". Tăng r lên 64 chỉ thêm khả
năng ghi nhớ, không thêm thông tin, và tăng rủi ro overfit trên `val_unseen`. α/r giữ bằng 2 (α = 32) vì LoRA
nhân đầu ra ΔW·x với α/r; tăng r mà giữ α thì bước cập nhật hiệu dụng nhỏ đi.

Gắn LoRA vào cả attention và MLP vì tri thức dạng "từ → hành vi" ("chuyển khoản" → `bank.transfer`) phần lớn nằm
trong MLP, còn attention là "nhìn vào đâu" (đọc tool docs). Ta cần cả hai.

QLoRA (lượng tử hóa phần đóng băng xuống 4 bit theo phân phối chuẩn, NormalFloat4) là hệ quả của chân lý 5 và
bộ nhớ T4 16 GB: 4,02B tham số bf16 là 8 GB, cộng gradient và hai moment của AdamW cho full fine-tune là 48–64 GB.
Không có cách nào khác trên T4. NF4 chia khối 64 giá trị, chuẩn hóa theo max tuyệt đối, ánh xạ lên 16 mức đặt tại
các phân vị của N(0,1); vì trọng số mạng gần phân phối chuẩn nên 16 mức này tối ưu hơn lượng tử tuyến tính.
Gradient vẫn chảy qua khối đã giải lượng tử về bf16 lúc nhân, nên A và B học ở độ chính xác đầy đủ.

#### 9.3.3. Thứ cần dạy

Liệt kê lỗi có thể xảy ra giữa "nghe được chữ" và "trạng thái đúng":

- Chọn sai tool, hoặc sai thứ tự khi câu có nhiều bước.
- Sai giá trị tham số: số viết bằng chữ ("hai trăm nghìn") phải thành số; tên người phải khớp **đúng chuỗi trong
  danh bạ**; giờ, ngày đúng định dạng của tool.
- Không hỏi lại khi thiếu thông tin, hoặc hỏi lại khi đã đủ.
- Gọi tool không tồn tại, sai schema.

Mỗi loại là một **quy ước ánh xạ** từ (chữ + ngữ cảnh + tài liệu) sang JSON. Đây chính xác là thứ 1098 mẫu dạy
được. Suy ra:

- Dữ liệu SFT phải cho model thấy đúng ba nguồn y như lúc thi (prompt giống từng byte, kết quả tool phát lại qua
  `zaic_tools`). Fine-tune ước lượng p(y | x) trên phân phối x của tập train; lúc chạy, x khác một dấu cách trong
  tool docs là ta đang suy diễn trên phân phối khác.
- Loss chỉ tính trên phần model phải tự sinh (JSON của assistant). Nếu tính trên cả tool docs và kết quả tool
  (hơn 90% token của mẫu), mục tiêu biến thành "thuộc catalog", phá chân lý 4. Che bằng label = −100.

#### 9.3.4. Chọn checkpoint

Loss là cross-entropy trung bình trên token; điểm thi là hàm 0/1 của hash cộng F1. Một token sai trong 50 token là
loss gần như không đổi nhưng hash về 0. Vậy loss chỉ để lọc thô. Điểm thật trên `val_unseen` gần nhất với thứ
cần tối ưu, nhưng nhìn vào nó nhiều lần sẽ tự lừa mình (so sánh nhiều lần làm điểm val lạc quan hơn thực tế), nên
chỉ chạy agent thật trên 2 checkpoint có loss `eval_unseen` thấp nhất.

#### 9.3.5. Merge

W' = W + (α/r)·B·A là một phép cộng ma trận làm một lần; lúc chạy không còn phép nhân thêm, không tốn thời gian
trong 30 giây. W lúc train là bản nf4, W lúc merge là bản bf16 gốc, nên W' khác một chút so với thứ model nhìn
thấy khi train. Sai số nhỏ nhưng khác 0: luôn đánh giá bản đã merge, tức đúng thứ sẽ nộp.

### 9.4. ASR: suy từ chân lý lên cách fine-tune

#### 9.4.1. Bản chất

Whisper là seq2seq hai nửa: encoder biến sóng âm thành biểu diễn "đã nghe được gì"; decoder là một mô hình ngôn
ngữ nhỏ biến biểu diễn đó thành chữ. Lỗi ASR vì thế có hai loại bản chất khác nhau:

- **Lỗi nghe** (encoder): nhiễu, giọng vùng miền, từ hiếm. Sửa cần nhiều giờ audio đa dạng. Ta có 13 giờ so với
  844 giờ encoder đã học; cố sửa là ước lượng hàng trăm triệu tham số từ mẫu nhỏ hơn 60 lần, phương sai tăng,
  model quên giọng và tiếng ồn không có trong 13 giờ đó (bias–variance).
- **Lỗi viết** (decoder): nghe đúng nhưng viết "15" thay vì "mười lăm", thêm dấu câu, viết hoa, từ tiếng Anh sai
  chính tả. Đây là quy ước, 3547 cặp dạy được.

Từ chân lý 1: chữ ASR phải ở **đúng dạng LLM đã được train**. Cập nhật 27/10/2026: PhoWhisper viết số bằng chữ còn
`audio_text` viết bằng chữ số, và BTC không chấm chữ ASR, nên LLM nộp bài học trên chính chữ ASR (v2). Fine-tune
ASR về `audio_text` thì đổi luôn cách viết, nên phải train lại LLM v2 bằng ASR mới. Vậy "giai đoạn 1 chỉ train decoder" là hệ
quả trực tiếp: lỗi cần sửa ở decoder, dữ liệu không đủ chạm encoder. Giai đoạn 2 mở encoder với lr nhỏ hơn 10
lần (discriminative learning rate) chỉ khi WER vẫn cao sau giai đoạn 1, để ΔW của encoder ở trong lân cận nghiệm
đã tốt.

Full fine-tune được vì 0,77B × 16 byte ≈ 12 GB, cộng activation của 30 giây mel với gradient checkpointing vừa
T4. Khi vừa bộ nhớ thì full cho ít ràng buộc hơn LoRA, và với ASR không chắc lỗi cần sửa nằm trong không gian hạng
thấp của decoder.

#### 9.4.2. Thước đo

Chỉ từ sẽ thành tham số tool mới ảnh hưởng điểm. Sai "ừ", "nhé" không làm sai tool call; sai "Lan" thành "Lâm"
hay "năm trăm" thành "năm chăm" làm sai tham số và mất toàn bộ hash. Tỉ lệ nghe sai thực thể là ước lượng trực
tiếp xác suất mẫu bị hỏng vì ASR, gần hàm mất mát thật hơn WER. WER chỉ để theo dõi model có hỏng chung không.

### 9.5. Ba điều chỉnh rút ra từ phân tích (và vì sao)

### Điều chỉnh 1: fine-tune LLM phải có vòng 2 trên chữ ASR thật

**Từ đâu ra.** Chân lý 2 nói mọi thông tin nằm trong ba nguồn, và danh bạ, tên thiết bị, app đã cài thuộc nguồn
"ngữ cảnh mẫu". Thành phần nào nhìn thấy ngữ cảnh mẫu? Chỉ LLM. ASR chỉ nhận sóng âm. Vậy khi ASR nghe "Lan"
thành "Lang", ASR không có gì để đối chiếu, nhưng LLM có danh bạ gồm "Lan" và "Long" ngay trong prompt.

**Nghĩa là gì.** Nếu train LLM trên đúng chữ ASR thật (chạy ASR đã chọn trên audio của train, lấy chữ nó nghe ra
làm transcript) với nhãn vẫn là tool call gold, model sẽ học quy ước "chọn tên trong danh bạ gần nhất với chữ
nghe được", "số tiền phải là số tròn hợp lý", v.v. Đây là sửa lỗi nghe bằng thông tin mà chỉ LLM có, và nó không
tốn thêm thời gian lúc chạy. Vòng 1 (trên `audio_text` gold) vẫn cần trước để học quy ước sạch; vòng 2 là tinh
chỉnh để chịu được nhiễu.

**Ví dụ đời thường.** Thư ký nghe sếp nói qua điện thoại kém "chuyển cho anh Lang", nhìn danh bạ chỉ có "Lan" và
"Long", chọn "Lan". Người sửa lỗi là thư ký (có danh bạ), không phải cái điện thoại.

**Trong code.** `sft_data.py` đã có `transcript_source = "asr"`; vòng 2 là chạy lại pipeline với tùy chọn đó và
dữ liệu `data/sft/v2.jsonl`. Cần chọn ASR trước khi làm vòng 2.

### Điều chỉnh 2: quyết định fine-tune ASR chỉ đưa ra sau LLM vòng 2

**Từ đâu ra.** Điều chỉnh 1 nói phần lớn lỗi thực thể có thể sửa ở LLM rẻ hơn. Nếu đo "điểm mất do ASR" ngay sau
khi cắm ASR vào LLM vòng 1, con số đó gồm cả phần LLM vòng 2 sẽ tự sửa được. Dùng con số đó để quyết định fine-tune
ASR là bỏ 1–2 ngày GPU vào thứ có thể biến mất sau một vòng train LLM rẻ hơn.

**Thứ tự đúng.** ASR gốc → LLM v1 (gold) → LLM v2 (chữ ASR) → đo điểm mất do ASR và tỉ lệ nghe sai thực thể →
chỉ khi vẫn ≥ 3 điểm hoặc ≥ 10% mới fine-tune ASR. Hai ngưỡng này là ước lượng ban đầu của nhiễu giữa hai lần
chạy trên ~200 mẫu val; SPEC yêu cầu đo lại độ dao động rồi chốt.

**Ví dụ đời thường.** Trước khi thay điện thoại mới cho sếp, thử dạy thư ký cách đối chiếu danh bạ đã. Nếu sau
đó vẫn nghe sai nhiều thì mới thay điện thoại.

### Điều chỉnh 3: nếu fine-tune ASR, làm dữ liệu TTS nhắm vào ba nhóm thực thể trước khi nghĩ tới mở encoder

**Từ đâu ra.** Mục 9.4.2: lỗi gây mất điểm tập trung ở ba nhóm có thể liệt kê được: tên người trong danh bạ, tên
thiết bị và địa điểm, số tiền và giờ. Ba nhóm này sinh được bằng TTS (tổng hợp giọng từ chữ): lấy câu mẫu từ
`audio_text` của train, thay thực thể bằng giá trị khác, đọc bằng TTS, nhãn chính là câu đã thay. Dữ liệu như vậy
đánh thẳng vào loại lỗi gây mất điểm, hiệu quả hơn thêm giờ audio chung chung, và vẫn chỉ cần train decoder.

**Vì sao trước mở encoder.** Mở encoder là bước có rủi ro cao nhất (mục 9.4.1, lỗi nghe cần nhiều giờ audio đa
dạng). Dữ liệu TTS có mục tiêu là bước rủi ro thấp, rẻ, và kiểm chứng được ngay bằng tỉ lệ nghe sai thực thể trên
`val_unseen`. Theo nguyên tắc "thêm từng lớp và mỗi lớp phải tự chứng minh", lớp rẻ và đo được đi trước.

**Ràng buộc.** TTS chỉ được dùng lúc phát triển, cấm lúc thi. Không dùng câu của val hay public để sinh. Không
bao giờ dùng dữ liệu tổng hợp để đánh giá.

### 9.6. Điểm yếu còn lại của lập luận

Chân lý 2 giả định ngữ cảnh mẫu luôn chứa đủ ứng viên để LLM đối chiếu. Với địa chỉ tự do, tên không có trong
danh bạ, hay nội dung tin nhắn tự do thì không có gì để đối chiếu, và lúc đó chỉ ASR mới cứu được. Bảng lỗi sau
LLM v2 sẽ cho biết nhóm này chiếm bao nhiêu; nếu lớn thì điều chỉnh 2 phải đẩy fine-tune ASR lên sớm hơn.

### 9.7. Thứ tự làm, tóm tắt

1. Cắm ASR gốc (PhoWhisper-medium), đo WER, tỉ lệ nghe sai thực thể, thời gian.
2. LLM v1: QLoRA trên `audio_text` gold. Chọn checkpoint bằng điểm `val_unseen`.
3. LLM v2: dựng lại dữ liệu với chữ ASR thật, train tiếp từ v1 hoặc từ gốc (thử cả hai), chọn như trên.
4. Đo điểm mất do ASR sau v2. Nếu còn ≥ 3 điểm hoặc ≥ 10% thực thể: fine-tune ASR giai đoạn 1 (decoder) với
   dữ liệu thật + TTS có mục tiêu. Mở encoder chỉ khi WER vẫn cao.
5. Mỗi bước đều đo trên `val_seen` và `val_unseen`, ghi vào `approach.md` mục 8.

### 9.8. Ba điều chỉnh, giải thích lại bằng một mẫu thật

Mẫu `train_0001`. Người dùng nói:

> "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng, nhắn **Sếp Dũng** giờ mới; không mưa thì thôi."

Nhãn gold gồm năm bước, trong đó có `contacts.search(query="Sếp Dũng")` rồi `sms.send(contact_id="c_08", ...)`.

#### 9.8.1. Vì sao LLM cần vòng 2 train trên chữ ASR thật

**Chuyện gì xảy ra lúc thi.** ASR (nhận dạng giọng nói) nghe file audio và viết ra chữ. Giả sử nó nghe nhầm
thành "nhắn **Sếp Dung** giờ mới". LLM chỉ nhận được dòng chữ này, không nghe audio. Nếu LLM được train chỉ trên
chữ gold "Sếp Dũng" (vòng 1), nó chưa bao giờ thấy chữ sai, nên gọi `contacts.search(query="Sếp Dung")`. Tùy tool
có khớp mờ hay không, có thể không tìm ra `c_08`, tin nhắn không gửi, hash trạng thái sai, mất trọn phần điểm hash
của mẫu.

**Ai có thông tin để sửa?** ASR không biết danh bạ của người này có ai. LLM thì có thể biết: với nhiều app (thiết
bị nhà thông minh, app đã cài, lịch) danh sách thực thể nằm ngay trong ngữ cảnh mẫu mà LLM đọc trong prompt; với
danh bạ thì LLM nhìn thấy kết quả `contacts.search` trả về ở bước trước. LLM là thành phần duy nhất vừa thấy chữ
nghe được, vừa thấy danh sách ứng viên.

**Vòng 2 làm gì.** Chạy ASR đã chọn trên chính audio của tập train, lấy chữ nó nghe ra (có lỗi thật), ghép với
nhãn gold không đổi. Train LLM trên cặp (chữ có lỗi → tool call đúng). Qua vài trăm mẫu, LLM học quy ước: chữ nghe
được chỉ là gợi ý, giá trị tham số phải là chuỗi có trong danh sách ứng viên gần nhất với nó. Giống thư ký nghe sếp
nói qua điện thoại rè "nhắn cho Dung", mở danh bạ chỉ thấy "Dũng" và "Dung Nguyễn (kế toán)", nhìn ngữ cảnh "sếp"
thì chọn Dũng. Người sửa lỗi là thư ký vì thư ký cầm danh bạ, không phải cái điện thoại.

**Vì sao không tốn thêm thời gian lúc chạy.** Quy ước nằm trong trọng số đã merge; lúc chạy LLM sinh đúng số token
như trước, không thêm bước nào.

#### 9.8.2. Vì sao chỉ quyết định fine-tune ASR sau LLM vòng 2

**"Điểm mất do ASR" là gì.** Chạy cùng LLM hai lần trên bộ val: một lần đưa chữ gold vào, một lần đưa chữ ASR
vào. Hiệu số điểm là phần mất vì ASR nghe sai.

**Bẫy khi đo sớm.** Giả sử đo với LLM v1 và thấy mất 8 điểm. Trong 8 điểm đó, một phần là lỗi kiểu "Sếp Dung"
mà LLM v2 sẽ tự sửa bằng danh bạ, một phần là lỗi kiểu nghe "chuyển năm trăm nghìn" thành "chuyển năm chăm
nghìn" hay nghe thiếu cả một mệnh đề, thứ không có danh sách nào để đối chiếu. Nếu nhìn con số 8 mà quyết định
fine-tune ASR ngay, ta bỏ 1–2 ngày GPU để sửa cả phần LLM v2 lẽ ra sửa miễn phí. Đo lại sau v2, có thể chỉ còn
mất 2 điểm, dưới ngưỡng 3, và không cần fine-tune ASR nữa.

Ví dụ đời thường: nhân viên mới ghi sai tên khách nhiều. Trước khi thay cả hệ thống điện thoại của công ty, thử
phát cho nhân viên quyển danh bạ và dạy cách đối chiếu. Sau một tuần vẫn sai nhiều thì mới thay điện thoại.

**Thứ tự.** ASR gốc → LLM v1 (chữ gold) → LLM v2 (chữ ASR) → đo điểm mất và tỉ lệ nghe sai thực thể → nếu còn
≥ 3 điểm hoặc ≥ 10% mới fine-tune ASR.

#### 9.8.3. Nếu fine-tune ASR: TTS nhắm vào ba nhóm thực thể trước, mở encoder sau

**Lỗi nào mới làm mất điểm.** Với `train_0001`, ASR nghe "dời chạy bộ" thành "rời chạy bộ" thì LLM vẫn hiểu,
không mất điểm. Nhưng nghe "muộn 1 tiếng" thành "muộn 2 tiếng" thì `start` của sự kiện sai, mất hash. Nên lỗi
đáng sửa gom về ba nhóm: tên người, tên thiết bị và địa điểm, số tiền và giờ. Đó là ba nhóm trở thành tham số tool.

**TTS có mục tiêu là gì.** TTS (text-to-speech, máy đọc chữ thành giọng). Lấy câu mẫu trong train, giữ nguyên
khung, chỉ thay thực thể: "nhắn Sếp Dũng" → "nhắn anh Hoàng", "nhắn chị Thu Hà"; "muộn 1 tiếng" → "muộn 3 tiếng";
"chuyển hai trăm nghìn" → "chuyển một triệu rưỡi". Mỗi câu đọc bằng TTS với vài giọng khác nhau, nhãn chính là
câu chữ đã thay. Được vài nghìn cặp audio-chữ chỉ xoay quanh đúng ba nhóm lỗi gây mất điểm, thay vì thêm giờ audio
chung chung mà phần lớn là từ không ảnh hưởng điểm. Luật cho phép TTS lúc phát triển, cấm lúc thi.

**Vì sao chỉ train decoder với dữ liệu này.** Whisper có hai nửa: encoder "nghe" (sóng âm → biểu diễn), decoder
"viết" (biểu diễn → chữ). Lỗi "năm chăm" thay vì "năm trăm", hay "Hoàng" viết thành "Hoàn", phần lớn là decoder
chọn chữ sai dù encoder đã nghe gần đúng. Dạy decoder bằng dữ liệu TTS là bước rẻ: train nhanh, ít tham số, đo
được ngay bằng tỉ lệ nghe sai thực thể trên `val_unseen`.

**Vì sao mở encoder là bước rủi ro nhất.** Encoder của PhoWhisper học từ 844 giờ audio thật đủ loại giọng, tiếng
ồn. Ta chỉ có 13 giờ thật cộng một ít TTS, mà TTS là giọng máy, sạch, đều. Mở encoder cho nó học trên dữ liệu này
là bắt một người đã nghe tốt mọi giọng miền đi luyện nghe lại chỉ bằng giọng của một máy đọc; dễ "quen tai" với
giọng máy và nghe kém đi với giọng thật trong bài thi. Vì vậy mở encoder để sau cùng, chỉ khi dữ liệu TTS cho
decoder vẫn chưa đủ, và khi đó dùng tốc độ học nhỏ hơn 10 lần để nó không đi xa khỏi cái đã học.

Tóm lại ba điều chỉnh đều xuất phát từ một ý: sửa lỗi ở chỗ có thông tin để sửa, làm bước rẻ và đo được trước,
bước đắt và rủi ro sau.
