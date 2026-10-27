# Khám phá dữ liệu LifeMate

Ghi chép khi khám phá gói dữ liệu `train+public_test` (bản `v3.2.0`, build ngày 24/10/2026). Mọi con số dưới đây được tính trực tiếp từ dữ liệu; các kết luận về hành vi tool được kiểm chứng bằng cách chạy lại đáp án qua thư viện `zaic_tools` của Ban Tổ chức.

Tài liệu gồm hai phần:

- **Phần I (mục 1 – 6): Hiểu bài toán.** Giải thích bằng lời dễ hiểu: cần làm gì, dữ liệu gồm những gì và dùng để làm gì.
- **Phần II (mục 7 – 12): Kết quả khám phá dữ liệu.** Các con số và phát hiện cụ thể.

---

# Phần I. Hiểu bài toán

## 1. Bài toán là gì

Cần làm một "trợ lý ảo bằng giọng nói". Người dùng nói một câu tiếng Việt, chương trình hiểu câu đó rồi tự bấm đúng các nút trong đúng các app thay cho họ. Các app này là app giả lập do BTC cung cấp, không phải app thật.

### 1.1. Một ví dụ từ đầu đến cuối

Đây là mẫu `train_0874` có thật trong dữ liệu.

**Bước 1. Người dùng nói** (file `train_0874.wav`): _"Chuyển cho Huy một trăm rưỡi qua ví giúp em nha, nội dung ghi là tiền vé bóng đá."_

**Bước 2. Chương trình biết thêm ngữ cảnh:** bây giờ là mấy giờ, người dùng ở đâu, và điện thoại đang cài những app nào (danh bạ, ví điện tử, ngân hàng, …).

**Bước 3. Chương trình phải tự suy nghĩ:**

- "Huy là ai? Mình chưa biết số hay ID của Huy, phải tra danh bạ đã."
  - Gọi `contacts.search({"query": "Huy"})`.
  - App giả lập trả về: Huy có ID là `c_07`.
- "Giờ chuyển tiền qua ví được rồi. 'Một trăm rưỡi' là 150000."
  - Gọi `ewallet.transfer({"contact_id": "c_07", "amount": 150000, "note": "tiền vé bóng đá"})`.

**Bước 4. Kết quả được chấm:** danh sách các thao tác trên, cùng một "mã vân tay" (`state_hash`) của dữ liệu các app sau khi làm xong. Mã này cho biết ví đã thực sự bị trừ đúng 150000 cho đúng người hay chưa.

Toàn bộ cuộc thi xoay quanh việc làm đúng như vậy cho hàng trăm câu nói khác nhau.

## 2. Ba cách trả lời được chấm điểm

| Tình huống | Chương trình phải làm | Ví dụ |
|---|---|---|
| Làm được (79%) | Gọi đúng các thao tác | Ví dụ chuyển tiền cho Huy ở trên |
| Thiếu thông tin (14%) | Gọi `sys.ask_missing_info` để hỏi lại | "Gọi cho Lan": danh bạ có 2 người tên Lan, nên phải hỏi là Lan nào |
| Không làm được hoặc không nên làm (7%) | Không làm gì, trả về `[]` | "Đặt khách sạn ở Đà Lạt" khi không có app khách sạn; "Tắt đèn… à thôi khỏi" |

**Cách tính điểm:** chương trình được điểm khi dữ liệu app cuối cùng khớp với đáp án, và điểm càng cao khi các thao tác càng giống đáp án. Làm thừa hoặc làm sai đều bị trừ điểm.

## 3. Dataset cung cấp gì, dùng để làm gì

Thư mục `train+public_test/` có hai phần.

**`train/` (1.545 mẫu, có đáp án)** là sách bài tập có lời giải. Dùng nó để:

- **Học:** cho model xem nhiều ví dụ "câu nói → các thao tác đúng" để bắt chước, hoặc dùng làm ví dụ mẫu trong prompt.
- **Tự chấm điểm:** chạy chương trình trên train rồi so với đáp án bằng `eval/score.py`. Biết ngay được bao nhiêu điểm và sai ở đâu, không cần nộp lên web.

**`public_test/` (634 mẫu, không có đáp án)** là đề thi thử. Chạy chương trình trên đây, nộp file kết quả lên web, rồi xem điểm trên bảng xếp hạng. BTC cố ý không phát đáp án public_test (đáp án nằm trên server của BTC), nên không thể chấm điểm public_test bằng `eval/score.py`.

Trong mỗi phần có các thành phần sau:

| Thành phần | Nội dung | Ý nghĩa |
|---|---|---|
| `audio/<id>/*.wav` | Giọng nói của người dùng | Đầu vào chính, phải chuyển thành chữ |
| `<split>.jsonl` | Mỗi dòng là một mẫu: đường dẫn audio, giờ, địa chỉ nhà, app đã cài, lịch sử hội thoại; riêng train có thêm transcript và đáp án | "Đề bài" của từng câu |
| `<split>_tools_catalog.json` | Hướng dẫn sử dụng của mọi app: có thao tác nào, cần tham số gì | Chương trình đọc file này để biết cách "bấm nút" |
| `<split>_environment/<id>/state.json` | Dữ liệu trong các app của người dùng: danh bạ, số dư, đèn nào đang bật, … | Chỉ thư viện giả lập được đọc, **chương trình bị cấm đọc trực tiếp** |
| `zaic_tools/` | Thư viện giả lập các app | Chương trình gọi `env.call(...)` để thao tác, giống như bấm nút trên app thật |
| `sample_submission.jsonl` | 3 dòng mẫu của file nộp | Xem định dạng file nộp |

## 4. Cần xây những gì

Chương trình có hai phần chính:

```
 file .wav  ──►  [1] ASR  ──►  chữ tiếng Việt  ──►  [2] LLM agent  ◄──►  zaic_tools (app giả lập)
                (nghe)                              (suy nghĩ, gọi tool)        │
                                                                                ▼
                                                              tool_calls + state_hash → submission.jsonl
```

**[1] ASR (Automatic Speech Recognition, nhận dạng giọng nói):** chuyển audio thành chữ. Phần này phải nghe được giọng ba miền và từ tiếng Anh lẫn vào. Ví dụ có thể dùng PhoWhisper hoặc Whisper bản nhỏ.

**[2] LLM agent:** một mô hình ngôn ngữ nhỏ (dưới 5B tham số, ví dụ Qwen loại 1,5B đến 4B). Agent hoạt động theo vòng lặp:

- Đọc câu nói, ngữ cảnh và danh sách tool.
- Quyết định gọi tool nào, với tham số gì.
- Xem kết quả tool trả về, rồi quyết định bước tiếp theo.
- Lặp lại cho đến khi xong, hoặc khi thấy cần hỏi lại hay không nên làm gì.

**Các ràng buộc bắt buộc của BTC:**

- Toàn hệ thống dưới 6B tham số.
- Chạy hoàn toàn offline, không được gọi API như ChatGPT.
- Mỗi câu xử lý trong tối đa 30 giây, trên 1 GPU A30.
- Nộp bài dưới dạng Docker.

## 5. Điều khó nhất của bài

**Public test có 11 app mà train không hề có**, ví dụ chứng khoán, vé tàu, đỗ xe. Vì vậy không thể chỉ học thuộc "câu này thì gọi tool kia". Chương trình phải biết **đọc hướng dẫn sử dụng** (catalog) của một app lạ rồi tự dùng cho đúng. Đây là lý do chọn LLM chứ không làm mô hình phân loại đơn giản.

## 6. Thứ tự làm đề xuất

1. **Bản đơn giản nhất chạy được:** dùng luôn transcript có sẵn trong train (tạm bỏ qua ASR), cho LLM gọi tool, rồi chấm bằng `eval/score.py`. Bước này để có điểm khởi đầu.
2. **Thêm ASR:** thay transcript có sẵn bằng chữ do ASR nghe ra, xem điểm tụt bao nhiêu.
3. **Cải thiện dần:**
   - Viết prompt tốt hơn và nén danh sách tool cho vừa 16K token.
   - Dạy agent các quy tắc: khi nào hỏi lại, khi nào trả `[]`, không gọi lại thông tin đã có trong lịch sử.
   - Fine-tune LLM trên train.
4. **Chạy trên public_test**, nộp lên web xem thứ hạng.
5. **Đóng gói Docker** trước hạn 16/11.

---

# Phần II. Kết quả khám phá dữ liệu

## 7. Tóm tắt nhanh

- Dữ liệu nhỏ: khoảng 1.500 mẫu train và 634 mẫu public test, mỗi mẫu là một câu nói dài khoảng 7 giây.
- **Public test có 11 app mà train hoàn toàn không có.** Agent không thể học thuộc từng tool; agent phải đọc JSON schema của tool rồi tự suy ra cách gọi.
- Đáp án của train sạch: chạy lại đáp án qua `zaic_tools` cho ra đúng trạng thái và đúng kết quả như nhãn.
- Có hai quy ước ngầm quan trọng:
  1. Thông tin nào đã có trong lịch sử hội thoại thì **không** gọi lại tool để hỏi nữa.
  2. Tìm danh bạ ra nhiều người trùng tên thì **hỏi lại**, không tự chọn.

## 8. Quy mô dữ liệu

| | train | public_test |
|---|---|---|
| Số mẫu | 1.545 | 634 |
| Có transcript (`audio_text`) và đáp án (`label`) | Có | Không |
| Số app / tool trong catalog | 46 / 147 | 36 / 122 |
| Tổng thời lượng audio của câu cuối | 3,1 giờ | 1,36 giờ |
| Số file wav (tính cả các lượt cũ trong history) | 5.036 | 1.933 |
| Dung lượng | 794 MB | 321 MB |

**Audio:** toàn bộ là WAV 16 kHz, mono, 16-bit. Câu cuối dài trung bình 7,2 giây (ngắn nhất 1,7 giây, dài nhất 22 giây), khoảng 22 từ.

**Lịch sử hội thoại (history):** khoảng 2/3 số mẫu có history, trung bình 4,5 lượt, dài nhất 24 lượt. Mỗi lượt cũ của người dùng có file audio riêng (`<id>_u1.wav`, `<id>_u2.wav`, …). Train có kèm transcript cho các lượt này, public test thì không, nên agent phải tự nghe cả các lượt cũ.

**Ngữ cảnh:** `current_time` nằm trong khoảng 20/09 – 20/12/2026. Người dùng ở Hà Nội nhiều nhất, sau đó là TP.HCM, Đà Nẵng, Cần Thơ, Huế. Lời nói mang đậm giọng địa phương ("rứa", "hen", "hông") và hay nói số bằng chữ ("một trăm rưỡi" nghĩa là 150000, "8 giờ kém 15 tối mai" phải đổi thành giờ ISO).

## 9. App và tool

### 9.1. Train và public test dùng bộ app khác nhau

| | Danh sách |
|---|---|
| Chỉ có ở public test (11 app, 34 tool) | ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator |
| Chỉ có ở train (21 app) | appliances, bike_rental, calculator, door_lock, exchange_rate, fitness, home_automation, home_energy, home_security, household_chores, learning, library, lottery, loyalty, medication, news, package_tracking, podcast_radio, robot_vacuum, subscriptions, video_meeting |
| Có ở cả hai (khoảng 25 app) | contacts, calendar, reminders, chat_ott, smart_home, weather, banking, ewallet, sms, email, … |

Có thể đoán private test cũng sẽ có những app chưa từng thấy. Vì vậy cách tiếp cận phải dựa vào việc đọc schema, không dựa vào việc học thuộc tool.

### 9.2. Mỗi mẫu cài bao nhiêu app

Mỗi mẫu cài trung bình khoảng 9 app, tương đương khoảng 31 tool (ít nhất 8, nhiều nhất 62). Toàn bộ schema này phải đưa vào prompt, trong khi LLM chỉ được dùng tối đa 16K token context, nên cần nén schema cho gọn.

### 9.3. Đặc điểm của schema (train)

- Tool gồm 62 tool `read` (chỉ đọc), 84 tool `write` (thay đổi dữ liệu) và 1 tool `meta` (`sys.ask_missing_info`).
- 68 tham số có `x-ref`: giá trị phải là ID lấy từ một lần đọc trước đó, ví dụ `device_id` lấy từ `smart_home.get_status`, `contact_id` lấy từ `contacts.search`.
- 40 tham số có `x-match: fuzzy`: văn bản tự do như nội dung tin nhắn, tiêu đề, địa chỉ. Các tham số này được chấm gần đúng.
- 6 tool có `x-one-of`, tức chỉ cần một trong vài nhóm tham số, ví dụ `contact_id` hoặc `phone_number`.

## 10. Đáp án của train

### 10.1. Phân loại

| Loại | Số mẫu | Tỉ lệ |
|---|---|---|
| Thực hiện hành động | 1.217 | 79% |
| Hỏi lại thông tin còn thiếu, sau khi đã gọi một số tool | 157 | 10% |
| Hỏi lại thông tin còn thiếu ngay từ đầu | 60 | 4% |
| Ngoài phạm vi, không làm gì (`[]`) | 111 | 7% |

### 10.2. Số call mỗi mẫu

Trung bình 2,3 call, nhiều nhất 6 call.

| Số call | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Số mẫu | 111 | 369 | 428 | 359 | 161 | 107 | 10 |

### 10.3. Mẫu hình hay gặp

- **Đọc trước, ghi sau.** Agent phải đọc để lấy ID rồi mới thao tác, ví dụ:
  - `contacts.search("Huy")` rồi `ewallet.transfer(contact_id="c_07", ...)`;
  - `smart_home.get_status` rồi `smart_home.light_set(device_id=...)`.
- Tool được gọi nhiều nhất là `contacts.search` (234 lần), tiếp theo là `sys.ask_missing_info` (217 lần), `smart_home.light_set`, `reminders.create`, `bill_payment.list_bills`.
- Gần như mọi tool trong catalog train đều xuất hiện trong đáp án (146 trên 147).
- `sys.ask_missing_info` chủ yếu dùng cho `phone.call`, `sms.send`, `ewallet.transfer`, tức những lúc không xác định được người nhận.

### 10.4. Khi nào đáp án là `[]` (không làm gì)

- **Không có app phù hợp:** đặt khách sạn, vé tàu, tra phạt nguội, gọi xe khi máy không cài app gọi xe.
- **Người dùng tự huỷ:** "Tắt đèn phòng khách, à thôi khỏi, không cần đâu."
- **Giá trị vô lý:** "Đặt độ sáng đèn bàn thành âm mười phần trăm."
- **Không phải yêu cầu:** "Hôm nay mày làm tốt lắm đó, khen một tiếng nha!"
- **Đã có câu trả lời trong history:** "Lúc nãy giá vàng SJC mua vào bao nhiêu nhỉ?"
- **Câu điều kiện mà điều kiện không thỏa, và thông tin đã có trong history** (xem mục 11.2).

## 11. Kiểm chứng bằng `zaic_tools`

### 11.1. Đáp án sạch

Chạy lại toàn bộ đáp án train qua `zaic_tools`:

- 0 trên 3.541 call bị từ chối.
- Trạng thái sau khi chạy khớp 100% với `final_state` trong nhãn.
- Kết quả trả về của từng call khớp 100% với `tool_results` trong nhãn.

Như vậy environment chạy tất định và có thể dùng làm bộ chấm local đáng tin cậy.

**378 trên 1.545 mẫu (24%)** có đáp án không làm thay đổi trạng thái: 111 mẫu `[]`, số còn lại chỉ đọc hoặc hỏi lại. Với các mẫu này, `state_hash` đúng chỉ cần agent không ghi bậy; điểm số được quyết định bởi việc gọi đúng các call (F1).

### 11.2. Câu điều kiện: đã biết rồi thì không hỏi lại

Nhiều câu dạng "Nếu mai mưa thì…" hay "Nếu vàng dưới X triệu thì…" có đáp án `[]`. Nguyên nhân là thông tin đã được hỏi ở lượt trước trong history, và điều kiện không thỏa:

| Mẫu | Yêu cầu | Thông tin đã có trong history | Kết quả |
|---|---|---|---|
| train_0125 | Nhắc đi mua vàng nếu SJC mua vào dưới 129 triệu | Mua vào 130,8 triệu | Không thỏa, không làm gì |
| train_0232 | Như trên, ngưỡng 123 triệu | Mua vào 124,9 triệu | Không thỏa, không làm gì |
| train_0082, 0137, 0266, 0378 | Nếu mai mưa thì dời lịch và nhắn SMS | "Ngày mai … có mây" | Không thỏa, không làm gì |

Ngược lại, khi history không có thông tin (ví dụ train_0001), đáp án là gọi `weather.get_forecast` trước, thấy có mưa, rồi mới `calendar.update_event` và `sms.send`.

**Bài học:** agent cần nhớ những gì đã nói trong history. Gọi lại tool đọc không làm sai `state_hash`, nhưng call thừa làm giảm precision.

### 11.3. Danh bạ trùng tên thì hỏi lại

Với câu "Gọi cho Lan nhé", `contacts.search("Lan")` trả về hai người tên Lan (vợ và đồng nghiệp). Đáp án là:

```json
[{"name": "contacts.search", "arguments": {"query": "Lan"}},
 {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}]
```

Agent không được tự đoán người nào, mà phải hỏi lại `contact_id`.

### 11.4. Call sai schema vẫn bị tính

- Call bị từ chối khi sai schema: thiếu tham số bắt buộc, sai kiểu dữ liệu, tool không thuộc app đã cài. Ví dụ `weather.get_forecast({})` bị từ chối vì thiếu `date`.
- Call bị từ chối không đổi trạng thái nhưng **vẫn được ghi vào bài nộp** và bị tính là call không khớp, làm giảm precision.
- ID không tồn tại hay số dư không đủ thì **không** phải lỗi: tool vẫn chạy và trả về `ok: true, done: false`.

**Bài học:** luôn gọi `env.validate()` trước `env.call()`, sửa tham số nếu sai, rồi mới gọi thật.

## 12. Gợi ý cho hướng làm

1. **Pipeline:** ASR tiếng Việt (nhận cả giọng ba miền và số nói bằng chữ) → LLM gọi tool theo vòng lặp đọc/ghi → `validate` trước khi `call`.
2. **Prompt:** đưa vào schema đã nén của các app đã cài, ngữ cảnh người dùng (giờ, địa chỉ nhà, nơi làm việc) và transcript của cả history.
3. **Quy tắc cần dạy cho agent:** khi nào trả `[]`, khi nào hỏi lại (trùng tên, thiếu thông tin), và không đọc lại thông tin đã có trong history.
4. **Đánh giá:** dùng bộ chấm local chạy trên train, tách điểm theo từng loại mẫu (hành động, hỏi lại, ngoài phạm vi) để biết agent yếu ở đâu. Nên chừa riêng các app mà public test không có để mô phỏng tình huống gặp app lạ.
