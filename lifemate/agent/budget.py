"""Theo dõi ngân sách của từng mẫu (số bước, số lần thử lại, độ dài context, hạn giờ) cho vòng lặp agent (SPEC 4.10).

Nội dung chính:
- `Budget`: vòng lặp agent hỏi lớp này trước mỗi bước để biết được chạy tiếp hay phải dừng.

Cách hoạt động:
- Trước mỗi bước, `before_step` kiểm tra ba điều kiện dừng theo thứ tự cố định: số bước, tràn context, hết giờ.
- Sau mỗi bước, `after_step` tăng bộ đếm bước.
- Số lần thử lại được đếm riêng qua `retry()`.

Lưu ý:
- Hạn giờ tính theo thời gian thật, không dự đoán thời lượng bước tiếp theo. Bước nào bắt đầu khi còn giờ thì
  được chạy; tới hạn, backend LLM tự ngắt bước đang chạy (`LLMBackend.generate(time_left_s=...)`). Các lời gọi
  tool đã gửi tới env vẫn được tính vào bài nộp.
- Đồng hồ (`Clock`) được truyền từ ngoài vào, nên test có thể giả lập thời gian.
"""

from __future__ import annotations

from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import Clock, StopReason


class Budget:
    """Ngân sách của một mẫu: bắt đầu tính giờ ngay khi khởi tạo (trước ASR) và quyết định bước tiếp có được chạy không.

    Tham số:
        cfg: config agent (`max_steps`, `max_retries`, `deadline_s`).
        llm_cfg: config LLM (`max_tokens`, `max_model_len`), dùng để kiểm tra tràn context.
        clock: hàm trả về số giây tăng đơn điệu; mốc bắt đầu được lấy ngay trong hàm khởi tạo.

    Thuộc tính:
        steps: số bước đã chạy.
        retries: số lần thử lại đã dùng.
    """

    def __init__(self, cfg: AgentConfig, llm_cfg: LLMConfig, clock: Clock) -> None:
        self._cfg = cfg
        self._llm_cfg = llm_cfg
        self._clock = clock
        # Lấy mốc ngay khi tạo, vì thời gian ASR cũng tính vào hạn giờ của mẫu.
        self._start = clock()
        self.steps = 0
        self.retries = 0

    @property
    def elapsed(self) -> float:
        """Số giây đã trôi qua kể từ khi tạo ngân sách."""
        return self._clock() - self._start

    @property
    def remaining(self) -> float:
        """Số giây còn lại tới hạn giờ; không bao giờ âm."""
        return max(0.0, self._cfg.deadline_s - self.elapsed)

    def before_step(self, prompt_tokens: int) -> StopReason | None:
        """Kiểm tra xem bước tiếp theo có được chạy không.

        Tham số:
            prompt_tokens: số token prompt hiện tại của hội thoại.

        Trả về:
            `StopReason.MAX_STEPS`, `StopReason.CONTEXT_OVERFLOW`, `StopReason.TIMEOUT`, hoặc `None` nếu được chạy tiếp.
        """
        if self.steps >= self._cfg.max_steps:
            return StopReason.MAX_STEPS
        # Phải chừa đủ chỗ cho `max_tokens` token sinh ra; không đủ thì backend sẽ từ chối hoặc cắt cụt đầu ra.
        if prompt_tokens + self._llm_cfg.max_tokens > self._llm_cfg.max_model_len:
            return StopReason.CONTEXT_OVERFLOW
        if self.remaining <= 0:
            return StopReason.TIMEOUT
        return None

    def after_step(self, seconds: float) -> None:
        """Ghi nhận một bước đã xong hoặc bị ngắt giữa chừng; tham số `seconds` hiện chưa dùng tới."""
        self.steps += 1

    def retry(self) -> bool:
        """Dùng một lần thử lại.

        Trả về:
            `True` nếu còn lượt thử lại, `False` khi đã dùng hết `cfg.max_retries` lần.
        """
        if self.retries >= self._cfg.max_retries:
            return False
        self.retries += 1
        return True
