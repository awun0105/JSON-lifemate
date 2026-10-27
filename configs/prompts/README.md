# configs/prompts/

Prompt hệ thống của LLM. File nào được dùng do trường `agent.system_prompt` trong cấu hình chạy quyết định. Chỗ `{tool_docs}` ở cuối được thay bằng tài liệu tool của các app đã cài.

| File | Nội dung |
|---|---|
| `system_v0.md` | Bản đầu tiên, viết trước khi có số đo. |
| `system_v1.md` | Bản đang dùng: 14 quy tắc rút từ lỗi của lần chạy thật đầu tiên (không tự đặt ID, gọi tool đọc không lọc, không thêm tham số thừa, câu có điều kiện, khi nào hỏi lại, đổi giờ tiếng Việt). Đi kèm bảng lịch 14 ngày do `agent/prompt.py` thêm vào ngữ cảnh. |

Dữ liệu fine-tune được dựng bằng đúng prompt lúc chạy, nên đổi prompt thì phải dựng lại dữ liệu và huấn luyện lại.
