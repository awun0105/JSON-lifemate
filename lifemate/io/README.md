# lifemate/io/

Mọi thao tác với file và với trình mô phỏng.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `data.py` | Đọc `<root>/<split>/<split>.jsonl` thành danh sách `Sample`; đọc file danh sách id. |
| `zaic.py` | Nạp đúng gói trình mô phỏng đi kèm từng split, đọc danh mục tool, mở đúng một `ToolEnv` cho mỗi mẫu. Đây là chỗ duy nhất được chạm vào trình mô phỏng; không file nào khác được đọc trạng thái ứng dụng trực tiếp. |
| `store.py` | Thư mục kết quả của một lần chạy: ghi `trace.jsonl` sau mỗi mẫu (flush, fsync), dựng `submission.jsonl` và `summary.json` bằng ghi nguyên tử, chạy tiếp sau khi bị ngắt (`--resume`), thêm dòng vào `index.csv`. |
| `check_submission.py` | Kiểm tra file nộp của split không có nhãn: đủ id, đúng định dạng, chạy lại được qua trình mô phỏng; ghi `check.txt`. |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- `zaic.py` là file duy nhất được chạm vào thư mục `*_environment/` và `state.json`.
- Mỗi mẫu một `ToolEnv` mới, không dùng lại giữa các mẫu.
- File kết quả ghi nguyên tử (ghi file tạm rồi đổi tên) để không bị hỏng khi máy tắt giữa chừng.
