# lifemate/agent/

Vòng lặp xử lý một mẫu. Chỉ dùng `core`: không mở file, không biết GPU, đồng hồ được truyền vào, nên cả vòng lặp chạy được trong test với LLM giả và trình mô phỏng giả.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `agent.py` | Lớp `Agent`: chuyển âm thanh thành chữ, dựng prompt, rồi lặp hỏi LLM → đọc JSON → qua guard → `validate` rồi `call` trình mô phỏng → thêm kết quả vào hội thoại, tới khi LLM báo xong hoặc hết giờ. Trả về `SampleResult`. |
| `budget.py` | Giới hạn mỗi mẫu: số bước, số lần thử lại, độ dài ngữ cảnh, và hạn thật 29 giây (bước đang chạy bị ngắt khi tới hạn). |
| `prompt.py` | Dựng hội thoại: prompt hệ thống, ngữ cảnh kèm lịch 14 ngày, lịch sử, yêu cầu hiện tại; thang rút gọn bốn mức khi vượt 16K token; hội thoại chỉ được thêm vào cuối. |
| `tooldocs.py` | Rút gọn danh mục tool thành tài liệu ngắn cho prompt, chỉ gồm các app đã cài; có cache. |
| `protocol.py` | Định dạng đầu ra của LLM: dựng JSON schema cho từng mẫu (chỉ tool được phép) và bộ đọc JSON chịu lỗi. |
| `guards.py` | Chuỗi kiểm tra trước mỗi lệnh: tool có được phép, tham số đúng schema, có gọi trùng lệnh ghi, có làm tiếp sau khi đã hỏi lại. Thêm guard mới chỉ cần đăng ký và thêm tên vào cấu hình. |
| `observers.py` | Nhận sự kiện bắt đầu, từng bước, kết thúc của agent: `NullObserver` (không làm gì), `Tracer` (ghi trace), `CompositeObserver` (gộp nhiều cái). |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Chỉ import `core`. Không mở file, không gọi mạng, không đọc đồng hồ hệ thống: đồng hồ, LLM, ASR và trình mô phỏng đều được truyền vào qua constructor.
- Prompt chỉ được thêm vào cuối, không sửa phần đã gửi, để vLLM tái dùng được phần đầu (prefix cache). Prompt lúc train dựng bằng đúng `PromptBuilder` này, nên phải giống từng byte với lúc chạy.
- Guard mới: viết một lớp, đăng ký tên, thêm tên vào cấu hình. Không sửa `agent.py`.
- Luôn `validate` rồi mới `call` trình mô phỏng.
