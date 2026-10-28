# tests/

Bộ kiểm thử pytest. Chạy bằng `make test`, trên CPU, không cần GPU hay vLLM. Test cần dữ liệu thi đánh dấu `data`, test cần tokenizer đánh dấu `tokenizer`; thiếu thì tự bỏ qua kèm lý do. Docstring của mỗi test ghi rõ test đó bảo vệ điều gì.

| File | Kiểm tra |
|---|---|
| `__init__.py` | Biến `tests` thành gói thường để không bị gói `tests` cài sẵn trên máy khác che mất. |
| `conftest.py` | Fixture dùng chung: đường dẫn repo, đường dẫn dữ liệu (bỏ qua khi thiếu). |
| `fakes.py` | Đồ giả dùng trong test: `FakeClock` (đồng hồ chỉnh tay), `FakeEnv` (trình mô phỏng trong bộ nhớ), `WhitespaceTokenCounter`. |
| `test_agent.py` | Vòng lặp agent từ đầu đến cuối với LLM giả và trình mô phỏng giả, kể cả dừng đúng hạn. |
| `test_budget.py` | Hạn giờ thật, giới hạn số bước, độ dài ngữ cảnh, số lần thử lại. |
| `test_guards.py` | Từng guard và việc thêm guard mới không cần sửa agent. |
| `test_prompt.py` | Dựng prompt, lịch ngày, app bị rút không lọt vào, hội thoại chỉ thêm vào cuối, thang rút gọn. |
| `test_protocol.py` | Bộ đọc JSON đầu ra và schema theo từng mẫu. |
| `test_tooldocs.py` | Tài liệu tool đủ mọi trường của danh mục và ra cùng một chuỗi mỗi lần. |
| `test_tokens.py` | Đếm token tăng dần khớp với đếm lại toàn bộ. |
| `test_jsonutil.py` | `canonical_json` và hàm cắt chuỗi. |
| `test_config.py` | Đọc và kiểm tra cấu hình, profile của mọi loại máy. |
| `test_envinfo.py` | Chọn kiểu số theo GPU, lấy phiên bản mã khi không có git. |
| `test_factory.py` | Lắp ghép từ cấu hình và các bảng đăng ký. |
| `test_runner.py` | Runner cô lập lỗi từng mẫu; file nộp ghi lại sau mỗi mẫu, đủ dòng khi chết giữa chừng; bước sau file nộp lỗi không làm crash. |
| `test_store.py` | Chạy tiếp sau khi bị ngắt, dòng cuối hỏng, ghi nguyên tử, dòng giữ chỗ cho id chưa chạy, file tổng hợp. |
| `test_docker_entrypoint.py` | Entrypoint Docker ở chế độ in lệnh: không `--resume`, dọn trace cũ, đúng một catalog. |
| `test_cli.py` | Chạy dòng lệnh với LLM giả phải ra 100 điểm; không lộ HF token; không có đường dẫn gắn cứng trong mã. |
| `test_data.py` | Đọc dữ liệu thi. |
| `test_zaic.py` | Nạp đúng trình mô phỏng của từng split, một env mỗi mẫu. |
| `test_score.py` | Bộ chấm: so tham số kiểu `strict` và `canon`, điểm của đáp án đúng và của bài nộp rỗng. |
| `test_audit.py` | Công cụ rà lời gọi: ghép theo tên không theo thứ tự, phân loại đúng từng kiểu sai. |
| `test_splits.py` | Tập `smoke` có 30 mẫu, đủ ba loại đáp án, ổn định giữa các lần chạy; câu gần trùng nằm chung một fold; các tập đã sinh không rò rỉ vào nhau. |
| `test_asr_sanitizer.py` | Bộ lọc chữ ASR: cắt câu ảo giác, giữ yêu cầu thật có chữ "đăng ký", giữ số điện thoại nhiều số 0, kiểm tra giữ tên và số. |
| `test_sft_mix.py` | Trộn chữ chuẩn và chữ ASR: tỉ lệ, tất định theo id, dùng lại chữ chuẩn khi ASR làm mất tên hoặc số. |
| `test_check_submission.py` | Kiểm tra file nộp trên dữ liệu thật. |
| `test_asr.py` | `GoldTranscriber` và việc chọn kiểu số cho ASR. |
| `test_asr_transcriber.py` | `ASRTranscriber` với backend giả: giữ thứ tự, chia batch, chọn engine, engine `faster_whisper` và bộ lọc. |
| `test_eval_asr.py` | Các chỉ số đo ASR, gồm chỉ số câu chêm tiếng Anh, và việc chấm lại chữ ASR đã có (`rescore_asr`). |
| `test_en_mixed.py` | Tìm từ tiếng Anh chêm trong câu, xếp nhóm, và chọn mẫu cho các tập `*_en_mixed`. |
| `test_asr_merge.py` | Ghép chữ hai ASR: gióng hàng, chỉ lấy từ tiếng Anh mới, bỏ câu bị lọc ảo giác. |
| `test_asr_prompts.py` | File câu mồi: không trùng cụm 4 từ nào của 5 036 câu đáp án, cụm 3 từ dưới 0,5% (marker `data`). |
| `test_asr_guard.py` | Lớp bảo vệ câu mồi: bốn dấu hiệu câu hỏng, `ASRTranscriber` chỉ chép lại câu hỏng, báo lỗi khi engine không chép lại được. |
| `test_vllm_backend.py` | `VLLMBackend` với engine giả: tham số sinh, warm-up, ngắt khi tới hạn. |
| `test_fakes.py` | Bản thân các đồ giả hoạt động đúng. |
| `test_report.py` | Bảng báo cáo lần chạy và huấn luyện. |
| `test_pack.py` | Gói mã nén cho notebook, file requirements, notebook không chứa logic. |
| `test_docker_entrypoint.py` | Entrypoint Docker: nhận split, tạo symlink, đúng lệnh chạy; nội dung `.dockerignore`. |
| `train/` | Test cho gói `lifemate/train`. |

## Quy ước riêng

Quy ước chung cho mọi file (công cụ, kiểu viết, log, commit) nằm ở `docs/CONTRIBUTING.md`. Phần dưới chỉ ghi các luật riêng của gói này.

- Test chạy trên CPU, không mạng, không GPU. Thư viện nặng được thay bằng module giả qua `monkeypatch`.
- Test cần dữ liệu thi gắn marker `data`; test cần tokenizer gắn marker `tokenizer`. Thiếu thì bỏ qua kèm lý do, không báo lỗi.
- Mỗi test có docstring một dòng nói test đó bảo vệ điều gì. Lỗi đã từng gặp được giữ lại bằng một test.
