# lifemate/

Gói Python chính. Chia thành bảy gói con theo vai trò; phụ thuộc chỉ đi một chiều: `run` dùng mọi gói, `eval` và `train` dùng `core`, `io`, `run`; `backends` và `io` dùng `core`; `agent` chỉ dùng `core`; `core` không dùng gói nào.

| File / thư mục | Chức năng |
|---|---|
| `__init__.py` | Khai báo gói và phiên bản. |
| `__main__.py` | Cho phép chạy `python -m lifemate`, chuyển thẳng sang dòng lệnh trong `run/cli.py`. |
| `core/` | Kiểu dữ liệu, cấu hình, JSON chuẩn hoá, đếm token, thông tin GPU, đường dẫn gốc. |
| `agent/` | Vòng lặp agent cho một mẫu: prompt, định dạng đầu ra, guard, hạn giờ, ghi sự kiện. |
| `backends/` | Mô hình thật: LLM qua vLLM, ASR qua transformers và các engine khác. |
| `io/` | Đọc dữ liệu thi, mở trình mô phỏng, ghi thư mục kết quả, kiểm tra file nộp. |
| `run/` | Lắp ghép từ cấu hình, dòng lệnh, chạy toàn bộ mẫu, báo cáo. |
| `eval/` | Chấm điểm, chia dữ liệu, đo chất lượng ASR. |
| `train/` | Dựng dữ liệu và fine-tune LLM, ASR, kèm pipeline một lệnh. |
