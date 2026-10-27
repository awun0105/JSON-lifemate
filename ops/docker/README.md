# ops/docker/

Ảnh Docker để nộp bài. Build từ thư mục gốc: `docker build -f ops/docker/Dockerfile .` hoặc `make docker-build`. Chi tiết ở `docs/docker.md`.

| File | Chức năng |
|---|---|
| `Dockerfile` | Hai tầng: tầng build cài thư viện bằng `uv` từ `uv.lock`, tầng chạy chỉ giữ Python, mã và trọng số mô hình (`/app/models/llm`, `/app/models/asr`). Chạy không có mạng. |
| `Dockerfile.dockerignore` | Loại `data/`, kết quả chạy, tài liệu, test và notebook khỏi ảnh. |
| `entrypoint.sh` | Lệnh khởi động: tìm file catalog trong `/data` để biết tên split, tạo symlink theo cấu trúc trình mô phỏng cần, rồi chạy dòng lệnh với `configs/docker.toml`; kết quả ghi vào `result/`. |
| `smoke.sh` | Build rồi chạy ảnh trên một thư mục dữ liệu với `--network none`, giống cách chấm. |
