# ops/notebooks/

Notebook chạy tay trên Kaggle hoặc Colab. Notebook không chứa logic, chỉ gọi lệnh có sẵn trong `lifemate/`. Mã được đưa lên bằng gói nén từ `make pack`.

| File | Chức năng |
|---|---|
| `kaggle_run.ipynb` | Chạy agent trên Kaggle: cài thư viện (kể cả `faster-whisper`), sinh splits nếu chưa có (`make_splits --env`), chạy test, chạy tập `smoke` rồi `val_seen`, `val_unseen`, lưu kết quả thành phiên bản mới của dataset riêng tư. Chạy với chữ ASR thật (`--transcripts asr`) cần GPU ≥ 24 GB (L4, A30, A100); T4 15 GB của Kaggle chỉ đủ cho transcript gold hoặc việc chỉ chạy ASR. |
| `kaggle_train_llm.ipynb` | Fine-tune LLM trên Kaggle bằng `lifemate.train.llm_pipeline`, khôi phục lần chạy trước nếu bị ngắt. |
| `kaggle_train_asr.ipynb` | Fine-tune ASR trên Kaggle bằng `lifemate.train.asr_pipeline`. |
| `colab_run.ipynb` | Như `kaggle_run.ipynb`, dữ liệu và kết quả trên Google Drive. Nơi chạy baseline với chữ ASR thật (`--transcripts asr`, Colab Pro, GPU ≥ 24 GB); sinh splits nếu chưa có (`make_splits --env`). |
| `colab_train_llm.ipynb` | Như `kaggle_train_llm.ipynb` trên Colab. |
| `colab_train_asr.ipynb` | Như `kaggle_train_asr.ipynb` trên Colab. |
