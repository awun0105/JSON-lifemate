"""Ngân sách bước, lần thử lại, ngữ cảnh và hạn chót cho từng mẫu (SPEC 4.10).

Nội dung: lớp `Budget`, được vòng lặp agent hỏi trước mỗi bước để biết có được chạy tiếp hay phải dừng.

Cách xây dựng:
- Hạn chót là thời gian thật, không dự đoán. Một bước được bắt đầu khi còn thời gian, và backend LLM tự
  ngắt bước đang chạy khi tới hạn (`LLMBackend.generate(time_left_s=...)`). Các lời gọi tool đã gửi tới env
  vẫn nằm trong bài nộp.
- Đồng hồ được truyền vào (`Clock`) nên test có thể giả lập thời gian.
- Ba điều kiện dừng được kiểm tra theo thứ tự cố định: số bước, tràn ngữ cảnh, hết giờ.
  Số lần thử lại được đếm riêng qua `retry()`.
"""

from __future__ import annotations

from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import Clock, StopReason


class Budget:
    """Bắt đầu tính giờ ngay khi khởi tạo (trước ASR) và quyết định bước tiếp theo có được chạy không.

    Args:
        cfg: cấu hình agent (`max_steps`, `max_retries`, `deadline_s`).
        llm_cfg: cấu hình LLM (`max_tokens`, `max_model_len`) để kiểm tra tràn ngữ cảnh.
        clock: hàm trả về số giây đơn điệu; thời điểm bắt đầu lấy ngay trong hàm khởi tạo.
    """

    def __init__(self, cfg: AgentConfig, llm_cfg: LLMConfig, clock: Clock) -> None:
        self._cfg = cfg
        self._llm_cfg = llm_cfg
        self._clock = clock
        # Lấy mốc ngay khi tạo: thời gian ASR cũng bị tính vào hạn chót của mẫu.
        self._start = clock()
        self.steps = 0
        self.retries = 0

    @property
    def elapsed(self) -> float:
        """Số giây đã trôi qua kể từ khi ngân sách bắt đầu."""
        return self._clock() - self._start

    @property
    def remaining(self) -> float:
        """Số giây còn lại tới hạn chót (không bao giờ âm)."""
        return max(0.0, self._cfg.deadline_s - self.elapsed)

    def before_step(self, prompt_tokens: int) -> StopReason | None:
        """Trả về lý do dừng trước bước tiếp theo, hoặc None nếu được chạy tiếp.

        Args:
            prompt_tokens: số token prompt hiện tại của hội thoại.

        Returns:
            `MAX_STEPS`, `CONTEXT_OVERFLOW`, `TIMEOUT` hoặc None.
        """
        if self.steps >= self._cfg.max_steps:
            return StopReason.MAX_STEPS
        # Phải chừa đủ chỗ cho `max_tokens` token sinh ra, nếu không backend sẽ từ chối hoặc cắt cụt đầu ra.
        if prompt_tokens + self._llm_cfg.max_tokens > self._llm_cfg.max_model_len:
            return StopReason.CONTEXT_OVERFLOW
        if self.remaining <= 0:
            return StopReason.TIMEOUT
        return None

    def after_step(self, seconds: float) -> None:
        """Ghi nhận một bước đã xong (hoặc bị ngắt giữa chừng); `seconds` hiện chưa được dùng."""
        self.steps += 1

    def retry(self) -> bool:
        """Dùng một lần thử lại; trả về False khi đã dùng hết `cfg.max_retries` lần."""
        if self.retries >= self._cfg.max_retries:
            return False
        self.retries += 1
        return True
