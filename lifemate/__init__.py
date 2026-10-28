"""Agent LifeMate: biến yêu cầu nói bằng tiếng Việt thành lời gọi tool trên bộ mô phỏng `zaic_tools` (`docs/SPEC.md`).

Nội dung chính:
- `__version__`: phiên bản của gói.

Lưu ý:
- Mã chia theo thư mục: `core/` (config, kiểu dữ liệu), `io/` (đọc dữ liệu, ghi run), `backends/` (ASR, LLM),
  `agent/` (prompt, guard, vòng lặp agent), `run/` (CLI, factory, runner), `eval/` (chấm điểm, chia tập),
  `train/` (dữ liệu và script fine-tune).
"""

__version__ = "0.1.0"
