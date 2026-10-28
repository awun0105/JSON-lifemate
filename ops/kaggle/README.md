# ops/kaggle/

Kernel Kaggle chạy tự động bằng Kaggle CLI. Mọi kernel đều riêng tư, dùng GPU T4, gắn dataset dữ liệu thi riêng tư, và clone mã từ GitHub theo một nhánh hoặc tag nên mỗi lần chạy đúng một phiên bản mã đã biết. Kết quả tải về bằng `kaggle kernels output <kernel> -p <thư mục>`.

| File | Chức năng |
|---|---|
| `smoke/lifemate_smoke.py` | Kernel chạy thử: tìm dữ liệu, clone mã, chỉnh profile Kaggle theo đường dẫn thật, cài thư viện, vLLM và faster-whisper, sinh lại `configs/splits/` từ dữ liệu, chạy test, đo chất lượng ASR, chọn cấu hình ASR, dựng cache ASR, chạy agent với chữ chuẩn (và với chữ ASR khi bật), đếm số bước bị ngắt ở hạn giờ, in báo cáo. Các tuỳ chọn đặt qua biến môi trường (`LIFEMATE_REF`, `RUN_IDS`, `RUN_TAG`, `RUN_GOLD`, `RUN_ASR`, `RUN_ASR_EVAL`, `ASR_EVAL_IDS`, `ASR_TUNE`, `RUN_ASR_CACHE`), mô tả ở docstring đầu file. |
| `smoke/kernel-metadata.json` | Khai báo kernel `lifemate-smoke`: GPU, mạng, dataset đi kèm. |
| `eval/lifemate_eval.py` | Kernel cho vòng chỉnh prompt: đặt sẵn tuỳ chọn (`val_seen` và `val_unseen`, chữ chuẩn), lấy `smoke/lifemate_smoke.py` từ đúng tag rồi chạy script đó. |
| `eval/kernel-metadata.json` | Khai báo kernel `lifemate-eval`. |
| `asr_tune/lifemate_asr_tune.py` | Kernel chỉ chạy ASR: chọn mô hình và cấu hình ASR bằng `lifemate.eval.asr_tune` cho whisper-large-v3, PhoWhisper, ChunkFormer (CTC và RNNT) và Parakeet (thư viện riêng của ChunkFormer và NeMo cài ngay trước khi đo), ghi kết quả vào `asr_tune/<mô hình>/`. |
| `asr_tune/kernel-metadata.json` | Khai báo kernel `lifemate-asr-tune`. |
| `asr_preflight/lifemate_asr_preflight.py` | Kernel CPU chạy thử trước `asr_tune`: cùng chuỗi cài đặt, chạy mọi lựa chọn của từng mô hình trên 3 mẫu (`asr_tune --preflight 3`), bắt lỗi cài đặt và tham số mà không tốn GPU. |
| `asr_preflight/kernel-metadata.json` | Khai báo kernel `lifemate-asr-preflight` (không bật GPU). |
| `asr_cache/lifemate_asr_cache.py` | Kernel chỉ chạy ASR: dựng cache chữ ASR cho dữ liệu fine-tune (`asr_cache/train_asr.json`, tải về `data/asr/`). |
| `asr_cache/kernel-metadata.json` | Khai báo kernel `lifemate-asr-cache`. |

Đẩy lên chạy: `kaggle kernels push -p ops/kaggle/<kernel> --accelerator NvidiaTeslaT4`.

**Giới hạn của T4.** T4 có 15 GB, không chứa nổi vLLM cùng whisper-large-v3 ở `asr.dtype = "auto"`: lần eval 28/10/2026 hết bộ nhớ ở 431/452 mẫu. Vì vậy trên Kaggle chỉ chạy agent với chữ chuẩn (`RUN_ASR` mặc định tắt), cùng các kernel chỉ chạy ASR (`asr_tune`, `asr_cache`). Baseline trên chữ ASR thật chạy bằng `make eval` trên GPU từ 24 GB (Colab L4 hoặc A100, máy thuê).
