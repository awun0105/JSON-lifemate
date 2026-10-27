# lifemate/run/

Lắp ghép và chạy. Gói duy nhất được import mọi gói khác.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `cli.py` | Dòng lệnh chính (`python -m lifemate.run.cli`): đọc cấu hình và profile, nạp mẫu, lắp hệ thống, chạy, in điểm. Tham số: `--config`, `--env`, `--split`, `--ids`, `--name`, `--out`, `--root`, `--resume`, `--limit`. |
| `factory.py` | Chỗ duy nhất lắp các thành phần từ cấu hình, qua các bảng đăng ký theo tên (`LLM_BACKENDS`, `TRANSCRIBERS`, guard, engine ASR). |
| `runner.py` | Chạy toàn bộ mẫu theo thứ tự id: warm-up, mỗi mẫu một trình mô phỏng, bắt lỗi ở ranh giới từng mẫu (chỗ duy nhất bắt mọi `Exception`), ghi kết quả, cuối cùng chấm điểm hoặc kiểm tra file nộp. |
| `timing.py` | Bọc LLM và ASR để đo thời gian mỗi lần gọi mà không sửa lớp gốc. |
| `report.py` | In bảng các lần chạy từ `index.csv` và bảng loss của một lần huấn luyện; vẽ biểu đồ nếu có matplotlib. |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- `factory.py` là chỗ duy nhất tạo đối tượng từ cấu hình. Các gói khác nhận đối tượng đã lắp sẵn.
- `runner.py` là chỗ duy nhất bắt `Exception` chung, ở ranh giới từng mẫu. Nơi khác chỉ bắt đúng loại lỗi cụ thể.
- Không có đường dẫn gắn cứng: đường dẫn đến từ profile trong `configs/env/`.
