# artifacts/models/

Sổ đăng ký các mô hình đã fine-tune. Trọng số không bao giờ vào git; chúng nằm trên Kaggle Models hoặc HF riêng tư, cột `location` ghi chỗ lưu.

| File | Nội dung |
|---|---|
| `registry.csv` | Mỗi mô hình một dòng: `name, base, data_version, checkpoint, val_seen, val_unseen, location`. Do `python -m lifemate.train.llm_pipeline` thêm vào ở bước `register`. |
| `<name>/train_config.json` | (không trong git) Cấu hình huấn luyện đầy đủ, phiên bản mã, sha256 dữ liệu, seed, GPU, phiên bản thư viện, checkpoint được chọn. |
| `<name>/metrics.jsonl` | (không trong git) Loss học và loss kiểm tra theo từng bước. |
| `<name>/eval.json` | (không trong git) Điểm agent của các checkpoint ứng viên trên `val_seen`, `val_unseen`. |

Tên mô hình có dạng `ft-<base>-sft<v>-r<nn>`, ví dụ `ft-qwen34binstruct2507-sft1-r01`.
