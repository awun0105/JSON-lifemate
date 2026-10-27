# ops/kaggle/

Kernel Kaggle chạy tự động bằng Kaggle CLI. Cả hai kernel đều riêng tư, dùng GPU T4, gắn dataset dữ liệu thi riêng tư, và clone mã từ GitHub theo một nhánh hoặc tag nên mỗi lần chạy đúng một phiên bản mã đã biết. Kết quả tải về bằng `kaggle kernels output <kernel> -p <thư mục>`.

| File | Chức năng |
|---|---|
| `smoke/lifemate_smoke.py` | Kernel chạy thử: tìm dữ liệu, clone mã, chỉnh profile Kaggle theo đường dẫn thật, cài thư viện, vLLM và faster-whisper, sinh lại `configs/splits/` từ dữ liệu, chạy test, đo ASR ở float32 và float16, chạy agent với chữ chuẩn và với chữ ASR, đếm số bước bị ngắt ở hạn giờ, in báo cáo. Các tuỳ chọn đặt qua biến môi trường (`LIFEMATE_REF`, `RUN_IDS`, `RUN_TAG`, `RUN_GOLD`, `RUN_ASR`, `RUN_ASR_EVAL`). |
| `smoke/kernel-metadata.json` | Khai báo kernel `lifemate-smoke`: GPU, mạng, dataset đi kèm. |
| `eval/lifemate_eval.py` | Kernel cho vòng chỉnh prompt: đặt sẵn tuỳ chọn (274 mẫu `val_seen`, chữ ASR), lấy `smoke/lifemate_smoke.py` từ đúng tag rồi chạy script đó. |
| `eval/kernel-metadata.json` | Khai báo kernel `lifemate-eval`. |

Đẩy lên chạy: `kaggle kernels push -p ops/kaggle/smoke --accelerator NvidiaTeslaT4`.
