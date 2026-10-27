# ops/notebooks/

Notebook chạy tay trên Kaggle hoặc Colab. Notebook không chứa logic, chỉ gọi lệnh có sẵn trong `lifemate/`. Mã được đưa lên bằng gói nén từ `make pack`.

| File | Chức năng |
|---|---|
| `kaggle_run.ipynb` | Chạy agent trên Kaggle: cài thư viện, chạy test, chạy tập `smoke` rồi `val_seen`, `val_unseen`, lưu kết quả thành phiên bản mới của dataset riêng tư. |
| `kaggle_train_llm.ipynb` | Fine-tune LLM trên Kaggle bằng `lifemate.train.llm_pipeline`, khôi phục lần chạy trước nếu bị ngắt. |
| `kaggle_train_asr.ipynb` | Fine-tune ASR trên Kaggle bằng `lifemate.train.asr_pipeline`. |
| `colab_run.ipynb` | Như `kaggle_run.ipynb`, dữ liệu và kết quả trên Google Drive. |
| `colab_train_llm.ipynb` | Như `kaggle_train_llm.ipynb` trên Colab. |
| `colab_train_asr.ipynb` | Như `kaggle_train_asr.ipynb` trên Colab. |
