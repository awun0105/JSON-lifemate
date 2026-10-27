# configs/

Đầu vào cố định của hệ thống. Đổi hành vi bằng cách sửa file ở đây, không sửa mã.

| File / thư mục | Chức năng |
|---|---|
| `v0.toml` | Cấu hình chạy chính: mô hình LLM và ASR (tên, phiên bản ghim), kiểu số, độ dài ngữ cảnh, hạn giờ, guard, prompt hệ thống. |
| `docker.toml` | Như `v0.toml` nhưng trỏ tới trọng số nằm sẵn trong ảnh Docker (`/app/models/...`) và bật chế độ tất định. |
| `fake.toml` | Thay LLM bằng mô hình giả phát lại đáp án, để kiểm tra cả đường ống trên máy không có GPU. |
| `env/` | Profile cho từng loại máy: dữ liệu ở đâu, lưu kết quả ở đâu, lấy HF token từ đâu. |
| `prompts/` | Prompt hệ thống của LLM. |
| `splits/` | Danh sách id của từng tập chia dữ liệu. Không có trong git; sinh bằng `make splits`. |
| `train/` | Cấu hình fine-tune LLM và ASR. |
