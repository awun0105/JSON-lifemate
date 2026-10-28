# lifemate/eval/

Đo lường. Chạy như script độc lập (`python -m lifemate.eval.<tên>`).

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `score.py` | Bộ chấm local theo công thức chính thức: chạy lại các lệnh đã nộp qua trình mô phỏng để so mã băm trạng thái, ghép lệnh để tính F1; in điểm theo loại đáp án, ma trận quyết định, loại lỗi và từng app; chế độ `strict` và `canon`. |
| `make_splits.py` | Chia tập train có nhãn, chuyển thể từ repo của team lead. Trước hết, mọi mẫu dùng 6 app giữ lại vào `val_unseen`. Phần còn lại được gom cụm câu gần trùng, rồi chia 5 fold phân tầng theo số call và ca hỏi lại: fold 0 là `val_seen`, fold 1 đến 4 là `train`. Thêm 30 mẫu `smoke`. Ghi vào `configs/splits/` (không có trong git) kèm kết quả kiểm tra rò rỉ. |
| `audit.py` | Rà từng lời gọi của một lần chạy so với nhãn, chuyển thể từ repo của team lead: thiếu hoặc thừa lời gọi, gọi lại tool đọc khi lịch sử đã có kết quả, gọi tool ở mẫu nhãn rỗng, tham số thiếu, thừa, sai kiểu, sai giá trị, chỉ khác chữ hoa. Ghép lời gọi theo tên tool, không theo vị trí. Dùng để tìm lỗi của model và nhãn đáng ngờ. |
| `eval_asr.py` | Đo riêng ASR: WER, WER không tính số (`wer_nonum`), CER, số câu rỗng, số câu bịa (trên chữ trước bộ lọc), tỷ lệ nghe sai tên và số có trong tham số tool, số câu viết đúng tên app và kênh (`TERMS`), thời gian nghe. |
| `asr_tune.py` | Chọn mô hình và cấu hình ASR công bằng: dò từng bước trên `asr_tune.txt` cho mỗi mô hình (whisper-large-v3, PhoWhisper, ChunkFormer CTC và RNNT, Parakeet) với cùng hàm mục tiêu, rồi đo cấu hình thắng trên `val_seen` và `val_unseen`. |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Mỗi file chạy được độc lập bằng `python -m lifemate.eval.<tên>`. Tham số đi qua dòng lệnh, có giá trị mặc định.
- Kết quả phải tất định: cùng dữ liệu, cùng seed thì ra đúng từng byte.
- Bộ chấm đi theo công thức chính thức. Đổi cách chấm phải sửa SPEC trong cùng commit.
