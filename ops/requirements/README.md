# ops/requirements/

Danh sách thư viện cho môi trường không dùng `uv` (notebook Kaggle, Colab giữ sẵn torch và CUDA của nền tảng).

| File | Nội dung |
|---|---|
| `notebook.txt` | Thư viện chính kèm nhóm `tokenizer`, phiên bản ghim, sinh tự động từ `uv.lock` bằng `ops/scripts/export_notebook_requirements.py`. Không gồm torch và vLLM (vLLM cài riêng khi có GPU). Không sửa tay. |
| `train.txt` | Thư viện cho fine-tune (unsloth, trl, peft, datasets, bitsandbytes, soundfile, jiwer). Không ghim phiên bản vì chúng đi theo torch có sẵn trên máy. |
