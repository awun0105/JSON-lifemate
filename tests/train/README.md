# tests/train/

Test cho gói `lifemate/train`. Không huấn luyện thật; chỉ kiểm tra các phần thuần logic.

| File | Kiểm tra |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `test_sft_data.py` | Quy tắc dựng dữ liệu huấn luyện LLM (chỉ dùng tập train, bỏ mẫu dính app bị rút), che loss chỉ trên phần trả lời, các hàm hỗ trợ của pipeline. |
| `test_asr_data.py` | Số cặp âm thanh và chữ, không trộn tập kiểm tra vào tập học. |
| `test_sft_augment.py` | Tăng cường dữ liệu: lượt gọi sai không tính loss và dùng câu lỗi guard thật, nhân bản đúng tỉ lệ, app gây nhiễu không gồm app giữ lại và không thêm vào mẫu nhãn rỗng, chỉ áp dụng cho tập train. |
| `test_swift_sft.py` | Phần nối với ms-swift: tham số dòng lệnh theo cấu hình, xuất dữ liệu giữ cờ `loss`, so mã hóa báo đúng chỗ khác, đọc checkpoint và loss, chạy tiếp từ checkpoint mới nhất. |
| `test_common.py` | Cố định seed, quản lý checkpoint, ghi metrics, ghi `train_config.json`. |
