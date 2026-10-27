# ops/scripts/

| File | Chức năng |
|---|---|
| `pack_code.sh` | Đóng gói mã (`lifemate/`, `configs/`, `ops/requirements/`, `tests/`) cùng file `VERSION` (commit hiện tại) thành `artifacts/dist/lifemate-code-<commit>.zip` để tải lên notebook. Gọi qua `make pack`. |
| `runpod_setup.sh` | Cài đặt một lệnh trên máy thuê RunPod: cài `uv`, clone hoặc cập nhật mã vào `/workspace`, cài thư viện có vLLM, tải hai mô hình đúng phiên bản ghim (Qwen3-4B và whisper-large-v3 bản CTranslate2), kiểm tra dữ liệu, sinh lại `configs/splits/`, chạy test và 30 mẫu thử. Tuỳ chọn: `--repo`, `--ref`, `--no-clone`, `--no-smoke`. |
| `export_notebook_requirements.py` | Sinh `ops/requirements/notebook.txt` từ `uv.lock`. |
