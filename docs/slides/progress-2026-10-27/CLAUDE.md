# LifeMate: báo cáo tiến độ — bộ nhớ của deck

Đọc file này trước khi sửa deck, và cập nhật nó: mỗi khi người dùng góp ý
hoặc sửa điều gì, ghi vào mục "Quyết định & góp ý" để không lặp lại lỗi cũ.

## Giới thiệu
- Dịp / mục tiêu: báo cáo tiến độ nội bộ dự án LifeMate (trợ lý giọng nói tiếng Việt → tool call), tình hình đến 27/10/2026
- Khán giả: người trong team / người hướng dẫn, biết khái niệm ASR, LLM nhưng chưa đọc mã
- Thời lượng: không áp dụng · deck để ĐỌC (gửi người khác tự đọc), 12 slide
- Ngôn ngữ & giọng văn: tiếng Việt, chi tiết, dễ hiểu; thuật ngữ giải thích ngay bên cạnh

## Thiết kế
- Mẫu: `blue-professional` — giữ nguyên font, bảng màu, lưới bố cục và chi tiết trang trí
  (deck-kit AGENTS.md §3; mô tả đầy đủ ở templates/blue-professional/design.md)
- Ý nghĩa màu: chỉ một màu nhấn cobalt của mẫu; xanh lá = đã xong, đỏ = còn thiếu (metric-change)
- Quy tắc: slide sơ đồ dùng layout-diagram (tự thiết kế theo hệ của mẫu): cột ghi chú trái, iframe sơ đồ Archify phải; sơ đồ nằm ở diagrams/*.html, sao chép từ .archify/ ở gốc repo

## Nội dung
- Nguồn / số liệu phải theo: docs/approach.md, docs/SPEC.md của repo; số đo audio (6969 file, 16 kHz, max 22,3 s) đo ngày 27/10
- Từ ngữ hoặc khẳng định cần tránh: không nêu tên cuộc thi/ban tổ chức cụ thể nếu deck sẽ public; chỉ nêu số đo đã có trong artifacts/runs/kaggle/ (Kaggle T4, 27/10); không suy ra kết quả trên A30

## Quyết định & góp ý (mới nhất ở trên)
- 2026-10-27 — người dùng: thêm thiết lập, kết quả và phân tích insight của lần chạy thử trên GPU → 4 slide mới (7–10, phần "Thử nghiệm"), cập nhật thẻ GPU ở slide 2 và bước 1 ở slide kế hoạch. Nguồn số: artifacts/runs/kaggle/report_gold_before_prompt_tuning.md và asr_eval_t4_v6/.
- 2026-10-27 — người dùng: "nội dung ghi quá vắn tắt khó hiểu, câu văn cần dài và rõ ràng chi tiết" → viết câu đầy đủ, giải thích thuật ngữ ngay tại chỗ; không dùng gạch đầu dòng cụt. Sơ đồ 4 (agent) phải xem được: iframe cần src đúng, thu nhỏ bằng biến --z.
- 2026-10-27 — người dùng chọn mẫu blue-professional (ấm, dễ đọc), yêu cầu tiếng Việt chi tiết dễ hiểu, 5–8 trang, ghép 3 sơ đồ Archify.
- 2026-10-27 — tạo deck.

## Cách làm việc với deck này
- `index.html` là deck. Mỗi slide có `data-title`, `data-sec`, `data-min`, ghi chú người nói trong `<aside class="notes">`.
- Người dùng có thể chỉ một phần tử bằng phím **K** (chép đường dẫn file, dòng, slide và selector) — sửa đúng phần tử đó.
- Sửa xong chạy check-deck.mjs; chia sẻ bằng export.mjs hoặc deploy.mjs (deck-kit/scripts).
