# lifemate/agent/

Vòng lặp xử lý một mẫu. Chỉ dùng `core`: không mở file, không biết GPU, đồng hồ được truyền vào, nên cả vòng lặp chạy được trong test với LLM giả và trình mô phỏng giả.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `chatml.py` | Định dạng ChatML (tool calling gốc của Qwen), mặc định từ D50: dựng tin nhắn giống hệt `build_messages` của repo lead (danh sách tool ở trường `tools` của tin `system`, lời gọi `tool_calls`, kết quả ở vai `tool`), đọc khối `<tool_call>` của mô hình, `ChatMLPrompt` rút gọn khi quá dài. |
| `chatml_agent.py` | Lớp `ChatMLAgent`: vòng lặp như `Agent` (ngân sách, guard, hạn giờ) cho định dạng ChatML; lời gọi bị guard chặn thành tin `tool` `{"ok": false, "error": ...}`. Dùng khi `[agent].format = "chatml"`. |
| `agent.py` | Lớp `Agent` (định dạng JSON cũ, `[agent].format = "json"`): chuyển âm thanh thành chữ, dựng prompt, rồi lặp hỏi LLM → đọc JSON → qua guard → `validate` rồi `call` trình mô phỏng → thêm kết quả vào hội thoại, tới khi LLM báo xong hoặc hết giờ. Trả về `SampleResult`. |
| `budget.py` | Giới hạn mỗi mẫu: số bước, số lần thử lại, độ dài ngữ cảnh, và hạn thật 29 giây (bước đang chạy bị ngắt khi tới hạn). |
| `prompt.py` | Dựng hội thoại: prompt hệ thống, ngữ cảnh kèm lịch 14 ngày, lịch sử, yêu cầu hiện tại; thang rút gọn bốn mức khi vượt 16K token; hội thoại chỉ được thêm vào cuối. |
| `tooldocs.py` | Rút gọn danh mục tool thành tài liệu ngắn cho prompt, chỉ gồm các app đã cài; có cache. |
| `protocol.py` | Định dạng đầu ra của LLM: dựng JSON schema cho từng mẫu (chỉ tool được phép) và bộ đọc JSON chịu lỗi. |
| `guards.py` | Chuỗi kiểm tra trước mỗi lệnh: tool có được phép, tham số đúng schema, có gọi trùng lệnh ghi, có làm tiếp sau khi đã hỏi lại. Thêm guard mới chỉ cần đăng ký và thêm tên vào cấu hình. |
| `observers.py` | Nhận sự kiện bắt đầu, từng bước, kết thúc của agent: `NullObserver` (không làm gì), `Tracer` (ghi trace), `CompositeObserver` (gộp nhiều cái). |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Chỉ import `core`. Không mở file, không gọi mạng, không đọc đồng hồ hệ thống: đồng hồ, LLM, ASR và trình mô phỏng đều được truyền vào qua constructor.
- Prompt chỉ được thêm vào cuối, không sửa phần đã gửi, để vLLM tái dùng được phần đầu (prefix cache). Prompt lúc chạy phải giống từng byte với dữ liệu huấn luyện: định dạng ChatML so với `build_messages` của lead (`tests/test_chatml.py`), định dạng JSON dùng chung `PromptBuilder` với `lifemate/train/sft_data.py`.
- Guard mới: viết một lớp, đăng ký tên, thêm tên vào cấu hình. Không sửa `agent.py`.
- Luôn `validate` rồi mới `call` trình mô phỏng.
