# configs/prompts/

Prompt hệ thống của LLM. File nào được dùng do trường `agent.system_prompt` trong cấu hình chạy quyết định. Chỗ `{tool_docs}` ở cuối được thay bằng tài liệu tool của các app đã cài.

| File | Nội dung |
|---|---|
| `system_v0.md` | Bản đầu tiên, viết trước khi có số đo. |
| `system_v1.md` | Bản đang dùng: 14 quy tắc rút từ lỗi của lần chạy thật đầu tiên (không tự đặt ID, gọi tool đọc không lọc, không thêm tham số thừa, câu có điều kiện, khi nào hỏi lại, đổi giờ tiếng Việt). Đi kèm bảng lịch 14 ngày do `agent/prompt.py` thêm vào ngữ cảnh. |
| `system_v1_alternatives.md` | `system_v1.md` thêm quy tắc 15 về chỗ nghe chưa chắc dạng `{A \| B}`; chỉ dùng cho nguồn J4 của benchmark ASR lần 3 (`docs/reports/benchmark/asr_v3.md`, mục 9), chọn bằng `--set agent.system_prompt=...`. |

Dữ liệu fine-tune được dựng bằng đúng prompt lúc chạy, nên đổi prompt thì phải dựng lại dữ liệu và huấn luyện lại.
