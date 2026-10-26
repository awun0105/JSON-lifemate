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

### 3.2. Cách chia train (`eval/make_splits.py`)

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
- Danh sách id nằm ở `splits/`, cố định với seed = 42. Mọi thí nghiệm dùng chung một cách chia để điểm so sánh được với nhau.

### 3.3. Bộ chấm local (`eval/score.py`)

Bộ chấm chạy lại các call đã nộp bằng `zaic_tools` của split, rồi tính điểm theo công thức chính thức (mục 5 trong `overview.md`).

```
python eval/score.py submission.jsonl --ids splits/val_seen.txt
python eval/score.py submission.jsonl --ids splits/val_unseen.txt
python eval/score.py submission.jsonl --ids splits/val_seen.txt --details details.jsonl
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
├── score.txt         báo cáo của eval/score.py
└── summary.json      số liệu tổng hợp (mục 4.4)
runs/index.csv        mỗi lần chạy thêm một dòng, ghi tự động; bảng ở mục 8 chỉ chép lại các lần chạy đáng nhớ
```

`runs/` không đưa vào git (chứa nội dung dữ liệu thi).

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
  - `smoke`: `splits/smoke.txt`, 30 mẫu lấy phân tầng từ `val_seen`, chạy vài phút, dùng sau mỗi lần sửa.
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
| Prompt | file trong `prompts/`, nằm trong git; mỗi lần chạy ghi hash prompt |
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

| Model | Vai trò | Số tham số | Ngày phát hành | Giấy phép |
|---|---|---|---|---|
| | | | | |

Kiểm tra: tổng < 6B, LLM < 5B, phát hành trước 26/10/2026, giấy phép cho phép.

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
| 27/10/2026 | Hạ tầng đo lường dạng file (`runs/`), không dùng MLflow | Đủ cho một đội nhỏ, không cần dựng server |
| 27/10/2026 | Chạy LLM bằng **vLLM** trong cùng process, không dùng `transformers`, llama.cpp hay Ollama khi nộp | A30 (sm80) hỗ trợ chế độ batch-invariant của vLLM để giữ tất định; prefix cache giảm thời gian prompt dài; nạp thẳng model đã merge LoRA. llama.cpp là dự phòng nếu vLLM không chạy được trong Docker; Ollama thêm một lớp trung gian không cần thiết |
| 27/10/2026 | LLM sinh JSON tự định nghĩa (`{"action": "call"|"done", ...}`), vLLM ép theo JSON schema; mô tả tool dạng rút gọn trong prompt | Tiết kiệm token so với schema OpenAI đầy đủ, không hỏng JSON, không phụ thuộc chat template từng model |
| 27/10/2026 | Không làm lockbox nội bộ; dùng public leaderboard làm bộ khóa, chỉ nộp ở mốc lớn | Public có 600 mẫu, 11 app lạ, không có nhãn, do BTC tạo cùng cách với private. Lockbox tự tách chỉ khoảng 130 mẫu, điểm dao động lớn và làm mất dữ liệu train |
| 27/10/2026 | Thêm 10 mẫu gold `[]` gắn tay vào `val_unseen`; thêm `splits/smoke.txt` | `val_unseen` trước đó không có mẫu ngoài phạm vi nào |
| 27/10/2026 | Chưa chọn LLM và ASR; code không gắn với model nào, chọn theo điều kiện ở `SPEC.md` mục 6.0 | Để chọn sau khi có hạ tầng đo mà không phải sửa code |
| 27/10/2026 | Không làm Normalizer ở v0; LLM tự đổi số theo quy tắc trong system prompt | LLM và ASR đều nhắm tới `audio_text` nguyên văn; chuẩn hóa trước LLM làm lệch phân bố đã train |
| 27/10/2026 | Python 3.11 + `uv`, ruff + mypy strict, pre-commit; lấy quy ước kỹ thuật từ `ai-kit`, không lấy layer ứng dụng | Kit là app cloud FastAPI + LangGraph, không khớp với agent offline |

## 8. Nhật ký thí nghiệm

Mỗi lần chạy agent ghi một dòng. Điểm lấy từ `eval/score.py` với `--ids splits/val_seen.txt` và `--ids splits/val_unseen.txt`.

| Ngày | Phiên bản | Thay đổi | ASR / LLM | val_seen | val_unseen | Public | Nhận xét |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
