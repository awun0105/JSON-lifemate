# docs/

Tài liệu của dự án. `SPEC.md` là nguồn sự thật cho mã: đổi hành vi thì sửa SPEC trong cùng commit.

| File / thư mục | Nội dung |
|---|---|
| `SPEC.md` | Đặc tả trạng thái hiện tại: bản đồ repo, từng thành phần, định dạng dữ liệu, test nghiệm thu, câu hỏi còn mở; phụ lục về pipeline, đánh giá, hạ tầng đo, nộp bài, danh sách model, kế hoạch fine-tune và mốc thời gian. |
| `decisions.md` | Nhật ký quyết định: ngày, quyết định, lý do, quyết định bị thay. Mỗi lần đổi hướng thêm một dòng. |
| `experiments.md` | Nhật ký thí nghiệm: mỗi lần chạy một dòng, kèm điểm và đường dẫn kết quả thô. |
| `reports/benchmark/v1.md` | Báo cáo so sánh mô hình và cấu hình ASR (whisper-large-v3, PhoWhisper, ChunkFormer, Parakeet): vấn đề, cấu hình từng mô hình kèm lý do, phương pháp, kết quả. |
| `reports/benchmark/v2.md` | Báo cáo benchmark ASR lần 2: bốn mô hình tốt nhất của lần 1 đo trên các câu tiếng Việt chêm tiếng Anh, câu mồi, ghép RNNT + large-v3 (xong). |
| `reports/benchmark/v3.md` | Báo cáo benchmark ASR lần 3: câu mồi theo mẫu, độ tin cậy từng từ, bộ phân loại ghép, hai phương án cho LLM, thời gian và bộ nhớ (kế hoạch, chưa chạy). |
| `rationale.md` | Phân tích nguyên lý gốc về fine-tune LLM và ASR (27/10/2026). |
| `overview.md` | Đề bài: đầu vào, đầu ra, cách chấm, ràng buộc, cấu trúc dữ liệu. |
| `eda.md` | Phân tích dữ liệu: số mẫu, phân bố app, độ dài, đặc điểm âm thanh. |
| `CONTRIBUTING.md` | Quy ước viết mã, quy trình commit, cách chia gói. |
| `docker.md` | Build và chạy thử ảnh Docker. |
| `runpod.md` | Chạy trên máy thuê RunPod: chọn GPU, cài đặt một lệnh, tải dữ liệu. |
| `diagrams/` | Sơ đồ dạng ảnh (`png/`), mã nguồn HTML của hai sơ đồ trong README gốc (`src/`) và mã nguồn Mermaid cũ (`mmd/`). |
| `slides/` | Bộ slide trình bày kiến trúc và mã nguồn. Đang sửa, chưa có trong git. |
| tệp PDF | Thể lệ gốc của ban tổ chức. |
