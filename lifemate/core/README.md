# lifemate/core/

Phần nền dùng chung. Không import gói nào khác trong `lifemate`, nên mọi gói khác đều dùng được.

| File | Chức năng |
|---|---|
| `__init__.py` | Đánh dấu gói. |
| `types.py` | Value object bất biến (`Sample`, `Turn`, `Action`, `Generation`, `Transcript`, `StepRecord`, `SampleResult`), các `Enum` (lý do dừng, mã lỗi, kết quả guard) và các `Protocol` cho phần thay thế được (`LLMBackend`, `Transcriber`, `TokenCounter`, `ToolEnvLike`, `AgentObserver`). |
| `config.py` | Đọc cấu hình chạy (`[llm]`, `[agent]`, `[asr]`) và profile máy (`configs/env/*.toml`) từ TOML; báo lỗi ngay khi có khoá lạ, thiếu khoá hoặc giá trị sai; đọc HF token lúc chạy. |
| `envinfo.py` | Phát hiện GPU, compute capability, phiên bản mã và thư viện; chọn kiểu số (`bfloat16` trên A30, `float16` trên T4). |
| `jsonutil.py` | `canonical_json` (khoá sắp xếp, không khoảng trắng thừa) để mọi JSON chạy lại ra đúng từng byte, và hàm cắt chuỗi. |
| `tokens.py` | Đếm token của hội thoại bằng chat template của mô hình (`HFTokenCounter`), hoặc đếm theo khoảng trắng khi không có transformers. |
| `textsim.py` | Gom các câu gần trùng thành cụm (difflib), dùng chung cho việc chia dữ liệu và lọc dữ liệu fine-tune. |
| `paths.py` | Định nghĩa một lần đường dẫn gốc của repo (`REPO_ROOT`). |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Không import gói nào khác trong `lifemate`.
- Value object là `@dataclass(frozen=True, slots=True)`. Lý do dừng, mã lỗi, kết quả guard là `Enum`.
- Mọi ngưỡng (giây, số bước, số lần thử lại, độ dài) nằm trong `Config`. Khoá cấu hình mới phải được khai báo ở đây, vì khoá lạ bị báo lỗi.
- HF token chỉ được đọc lúc chạy, không ghi vào log hay file.
