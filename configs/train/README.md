# configs/train/

Cấu hình fine-tune, đọc bởi các pipeline trong `lifemate/train/`.

| File | Nội dung |
|---|---|
| `llm_v1.toml` | Fine-tune LLM: mô hình gốc và phiên bản, cấu hình chạy dùng để dựng prompt, đường dẫn dữ liệu, tham số QLoRA (nén 4 bit, hạng LoRA, các lớp áp dụng), tham số huấn luyện (learning rate, lịch learning rate, warmup, số epoch, seed, tần suất lưu checkpoint), nguồn chữ của dữ liệu (`gold`, `asr`, `mix`). Khóa `tuner_backend` chọn lõi train của ms-swift (`peft` hoặc `unsloth`). Giá trị hiện tại theo team lead: LoRA r=32, alpha=64, lr 2e-4 cosine, warmup 5%, 3 epoch, batch hiệu dụng 16, max_len 10240, 70% chữ ASR, 3% mẫu phục hồi lỗi. Các khóa `oversample_hard`, `distractor_apps`, `dedup_max_k` bật các bước tăng cường tiếp theo trong lộ trình của lead, mặc định tắt. Chưa được đo trên pipeline của repo này. |
| `asr_v1.toml` | Fine-tune ASR: mô hình gốc, kiểu huấn luyện (toàn phần, giai đoạn 1 chỉ học decoder), dữ liệu, tham số huấn luyện. |
