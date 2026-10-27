# configs/env/

Profile máy. Chọn bằng `--env <tên>`. Mọi khác biệt giữa các nơi chạy nằm ở đây; mã không chứa đường dẫn gắn cứng.

Mỗi file có các trường: `data_root` (thư mục dữ liệu thi), `runs_dir` (nơi lưu kết quả), `hf_cache` (nơi tải mô hình), `hf_token_source` (đọc HF token từ đâu, chỉ lúc chạy), `persist` (cách giữ kết quả sau khi máy tắt).

| File | Dùng cho |
|---|---|
| `local.toml` | Máy phát triển: dữ liệu trong `data/`, kết quả trong `artifacts/runs/`. |
| `kaggle.toml` | Kaggle: dữ liệu từ dataset riêng tư trong `/kaggle/input/`, kết quả trong `/kaggle/working/`, token từ Kaggle Secrets. |
| `colab.toml` | Colab: dữ liệu và kết quả trên Google Drive, token từ Colab userdata. |
| `runpod.toml` | RunPod: mọi thứ trên ổ `/workspace` (ổ duy nhất còn lại sau khi tắt pod), token từ biến môi trường. |
| `rented.toml` | Máy GPU thuê khác: dữ liệu và kết quả trong thư mục nhà. |
| `docker.toml` | Bên trong ảnh Docker lúc chấm: dữ liệu qua symlink, kết quả trong `result/`, không cần token. |
