# configs/asr_merge/

Trọng số của engine ghép hai mô hình ASR (`lifemate/backends/asr_merged.py › MergedASR`): ChunkFormer RNNT làm nền, whisper-large-v3 với câu mồi theo mẫu bổ sung từ tiếng Anh, bộ phân loại H chọn ở từng vùng khác nhau, rồi đổi số đọc thành chữ số. File nào được dùng do trường `asr.merge_weights` trong cấu hình chạy quyết định (đường dẫn tính từ gốc repo). Đổi trọng số thì tạo file `_v<n+1>` mới, không sửa file cũ.

| File | Nội dung | Nguồn |
|---|---|---|
| `h_v1.json` | Bộ phân loại hồi quy logistic H (10 đặc trưng, chuẩn hoá bằng trung bình và độ lệch, L2 = 10) học trên cả 1 093 mẫu train (D54); bảng độ tin cậy từng từ của câu mồi `en_full_v1`; câu mồi biến thể E2; lớp bảo vệ g4 (4 từ, nén 2,4); ngưỡng 0,5; bật đổi số (D56) | `python -m lifemate.eval.asr_v4_report export-h` từ chữ và điểm của các phiên K1–K3 (`docs/reports/benchmark/asr_v3.md` mục 7.2, `asr_v4.md` mục 3) |

Mọi số trong file đều học trên `train`; không có số nào từ `val_seen` hay `val_unseen`. Thứ tự đặc trưng trong trường `features` phải trùng `lifemate/eval/asr_v3_analysis.py › FEATURES`; lệch thì engine từ chối nạp.
