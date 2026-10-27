# lifemate/train/

Dựng dữ liệu và fine-tune. Chỉ chạy trên máy có GPU (Kaggle, Colab, máy thuê). Mỗi pipeline lưu tiến độ sau từng bước nên chạy lại được từ chỗ dừng.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `sft_data.py` | Dựng dữ liệu huấn luyện LLM bằng đúng `PromptBuilder` lúc chạy: mỗi dòng là hội thoại "prompt → lệnh đúng", kết quả lệnh lấy bằng cách chạy lại qua trình mô phỏng; chỉ dùng id trong `configs/splits/train.txt`; chữ người dùng lấy từ chữ chuẩn (`gold`), chữ ASR (`asr`) hoặc trộn (`mix`, mặc định 30% chữ chuẩn và 70% chữ ASR). Ở chế độ trộn, mẫu ASR làm mất tên hoặc số của đáp án sẽ dùng lại chữ chuẩn. App giữ lại bị bỏ khỏi prompt, và lệnh từ chối chạy khi danh sách id có mẫu thuộc tập kiểm tra. Chỉ với tập train, có thể tăng cường dữ liệu theo lead (`Augment`): mẫu phục hồi lỗi, nhân bản ca khó, app gây nhiễu, lọc cụm câu trùng. |
| `asr_cache.py` | Chạy ASR một lần trên các tập rồi lưu chữ vào `data/asr/train_asr.json`, để dựng dữ liệu `asr` hoặc `mix` mà không cần chạy lại ASR. Cần GPU. |
| `sft.py` | Cấu hình fine-tune LLM, cách tính nhãn loss chỉ trên phần trả lời (dùng làm chuẩn để so với ms-swift), gộp adapter vào mô hình gốc. Trình train Unsloth cũ vẫn còn trong file, đã comment lại. |
| `swift_sft.py` | Train LLM bằng ms-swift (`swift sft`). Xuất dữ liệu sang định dạng ms-swift, so token và nhãn loss của ms-swift với chat template lúc chạy rồi mới train, dựng tham số dòng lệnh từ `configs/train/*.toml`, đọc checkpoint và loss trên `val_unseen` để chọn bản tốt nhất. |
| `llm_pipeline.py` | Một lệnh cho LLM: dựng dữ liệu → huấn luyện bằng ms-swift → chọn 2 checkpoint loss thấp nhất → gộp → chạy agent thật để lấy điểm → chọn bản tốt nhất → ghi sổ đăng ký. |
| `asr_data.py` | Dựng cặp (file âm thanh, chữ chuẩn) cho lượt cuối và mọi lượt người dùng trong lịch sử. |
| `asr.py` | Fine-tune ASR: toàn phần hoặc LoRA, giai đoạn 1 chỉ học decoder; chọn checkpoint theo tỷ lệ nghe sai tên và số. |
| `asr_pipeline.py` | Một lệnh cho ASR: dựng dữ liệu → huấn luyện → chọn checkpoint → gộp → đo lại. |
| `common.py` | Phần dùng chung: cố định seed, quản lý checkpoint, ghi `metrics.jsonl`, ghi `train_config.json`, đọc cấu hình TOML chặt. |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Dữ liệu fine-tune chỉ lấy id trong `configs/splits/train.txt`. Thiếu `configs/splits/meta.json` thì dừng với lỗi, không chạy tiếp.
- Mọi tham số huấn luyện nằm trong `configs/train/*.toml`. Mỗi lần chạy ghi lại cấu hình, phiên bản mã, sha256 dữ liệu và seed vào `train_config.json`.
- Không chạy trên máy phát triển. Test của gói này dùng runner và tokenizer giả.
