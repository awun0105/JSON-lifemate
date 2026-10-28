# lifemate/backends/

Mô hình thật, đứng sau các `Protocol` trong `core/types.py`. Thư viện nặng (`vllm`, `torch`, `transformers`) chỉ được import bên trong đúng lớp cần đến.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `llm.py` | `VLLMBackend`: chạy LLM bằng vLLM, ép đầu ra theo JSON schema, warm-up trước khi chạy, và ngắt yêu cầu khi tới hạn của mẫu. `FakeLLM`: mô hình giả phát lại đáp án, dùng trong test và `configs/fake.toml`. |
| `asr.py` | Chuyển âm thanh thành chữ. `GoldTranscriber` trả chữ chuẩn có sẵn trong tập train; `ASRTranscriber` gom mọi file âm thanh của một mẫu thành batch; bốn engine: `faster_whisper` (whisper-large-v3 qua CTranslate2, engine đang dùng), `hf_seq2seq` (PhoWhisper qua transformers), `chunkformer` (ChunkFormer CTC), `nemo_ctc` (Parakeet CTC qua NeMo). `hf_seq2seq` nhận cùng tên khóa giải mã với faster-whisper (`hf_generate_kwargs` tự dịch). Kiểu số tự chọn theo GPU. Khi `sanitize = true`, chữ đi qua `asr_sanitizer`. Khi `prompt_guard_ngram` bật, câu hỏng do câu mồi được chép lại không câu mồi (`transcribe_batch_unprompted`, chỉ hai engine Whisper có). |
| `asr_guard.py` | Lớp bảo vệ câu mồi: phát hiện câu ASR hỏng do câu mồi (rỗng, bộ lọc ảo giác có cắt, chép lại cụm từ của câu mồi, tỉ lệ nén cao) mà không cần đáp án. |
| `asr_sanitizer.py` | Bộ lọc chữ ASR, chuyển thể từ repo của team lead: cắt câu ảo giác kiểu phụ đề YouTube ("hãy đăng ký kênh", "cảm ơn các bạn đã theo dõi"), gộp từ lặp vô hạn, bỏ lượt lịch sử chỉ toàn câu ảo giác. Chữ số chỉ bị gộp khi lặp từ 8 lần, để giữ số điện thoại có nhiều số 0. Kèm `entities_preserved` để kiểm tra chữ ASR còn giữ tên và số của đáp án không. |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Mỗi lớp mô hình cài đúng một `Protocol` trong `core/types.py` (`LLMBackend`, `Transcriber`). Engine mới được đăng ký vào bảng theo tên (`ASR_ENGINES`) và chọn bằng cấu hình, không sửa nơi gọi.
- Thư viện nặng (`vllm`, `torch`, `transformers`, `faster_whisper`) chỉ được import bên trong lớp cần đến, để test chạy được trên CPU không cài các thư viện này.
- Mọi tham số mô hình (tên, revision, dtype, tham số giải mã) đến từ cấu hình. Không ghi cứng trong mã.
