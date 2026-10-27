# LifeMate: hệ thống đã xây — bộ nhớ của deck

Đọc file này trước khi sửa deck, và cập nhật nó: mỗi khi người dùng góp ý
hoặc sửa điều gì, ghi vào mục "Quyết định & góp ý" để không lặp lại lỗi cũ.

## Giới thiệu
- Dịp / mục tiêu: trình bày toàn bộ hệ thống LifeMate đã xây (bài toán, kiến trúc, luồng dữ liệu, đo lường, fine-tune, vận hành) cho team
- Khán giả: người trình bày và thành viên team; biết ASR/LLM, chưa chắc đã đọc mã
- Thời lượng: không giới hạn · deck vừa để ĐỌC vừa để trình bày, 42 slide (7 phần)
- Ngôn ngữ & giọng văn: tiếng Việt theo skill vietnamese-writing (không đại từ nhân xưng), câu đầy đủ, chi tiết

## Thiết kế
- Mẫu: `signal` — giữ nguyên font, bảng màu, lưới bố cục và chi tiết trang trí
  (deck-kit AGENTS.md §3; mô tả đầy đủ ở templates/signal/design.md)
- Ý nghĩa màu: vàng (accent) = đường chính / thành phần trọng tâm; viền đứt = chưa làm hoặc không bật mặc định
- Quy tắc: sơ đồ là SVG inline vẽ trong slide (KHÔNG dùng skill vẽ sơ đồ, KHÔNG dùng lại sơ đồ cũ trong docs/diagrams hay deck progress); slide sơ đồ = layout slide--figure: sơ đồ trái + cột giải thích phải (hoặc sơ đồ rộng + 3 cột chú thích dưới)

## Nội dung
- Nguồn / số liệu phải theo: docs/SPEC.md, docs/approach.md, docs/overview.md và mã nguồn lifemate/ (tên module, hàm trên sơ đồ là tên thật)
- Từ ngữ hoặc khẳng định cần tránh: chỉ nêu số đo đã có ở artifacts/runs/kaggle/ (Kaggle T4, 27/10: vLLM và PhoWhisper đã chạy; train/merge và Docker CHƯA chạy); không suy ra kết quả trên A30
- Hạn giờ: hạn thật deadline_s = 29 (commit 436fe1c), KHÔNG còn bộ canh dự đoán / hard_limit_s = 28

## Quyết định & góp ý (mới nhất ở trên)
- 2026-10-27 — người dùng: thêm thiết lập, kết quả và insight của lần chạy thử trên GPU vào ĐÚNG deck này (không phải deck progress) → chương 07 · Thử nghiệm (slide 37–41); viết lại slide Budget và Runner theo hạn thật 29 s; cập nhật Ràng buộc, Bộ test (183), Trạng thái, Mốc, Tóm tắt, Nội dung (7 phần). Số trang trong chrome và chân trang được đánh lại tự động.
- 2026-10-27 — người dùng: "ghi script cho người thuyết trình ở mỗi trang, đọc slide không hiểu gì" → mỗi slide có <aside class="notes"> là lời thoại đọc được nguyên văn, giải thích bằng lời thường + ví dụ đời thường, giải thích thuật ngữ tại chỗ. Giữ chuẩn này cho slide mới.
- 2026-10-27 — người dùng chọn mẫu signal; yêu cầu sơ đồ hệ thống, luồng dữ liệu, pipeline fine-tune tự vẽ trong slide kèm lời giải thích.
- 2026-10-27 — tạo deck. Nguồn dựng: scratchpad build.py + diagrams.py (sinh SVG) — sửa trực tiếp index.html nếu không còn script.

## Cách làm việc với deck này
- `index.html` là deck. Mỗi slide có `data-title`, `data-sec`, `data-min`, ghi chú người nói trong `<aside class="notes">`.
- Người dùng có thể chỉ một phần tử bằng phím **K** (chép đường dẫn file, dòng, slide và selector) — sửa đúng phần tử đó.
- Sửa xong chạy check-deck.mjs; chia sẻ bằng export.mjs hoặc deploy.mjs (deck-kit/scripts).
