# Nhật ký thí nghiệm

Mỗi lần chạy (agent, ASR, train) thêm một dòng vào bảng dưới, kể cả lần chạy hỏng. Điểm lấy từ `lifemate/eval/score.py` với `--ids configs/splits/val_seen.txt` và `--ids configs/splits/val_unseen.txt`. `artifacts/runs/index.csv` ghi tự động mọi lần chạy; bảng này chỉ chép lại các lần chạy đáng nhớ, kèm nhận xét.

Quy ước:

- Điểm đo trên cách chia cũ (trước 28/10/2026) không so được với điểm theo cách chia hiện tại (`SPEC.md` phụ lục C.2).
- Dòng ghi "không hợp lệ" hoặc "chỉ tham khảo" không dùng làm baseline.
- Cột "Kết quả thô" trỏ tới thư mục đã tải về máy, trong `artifacts/runs/kaggle/` (không có trong git).
- Thay đổi hướng làm rút ra từ một lần chạy ghi ở `decisions.md`; `SPEC.md` chỉ mô tả trạng thái hiện tại.

| Ngày | Lần chạy / phiên bản | Thay đổi | ASR / LLM | val_seen | val_unseen | Public | Nhận xét | Kết quả thô |
|---|---|---|---|---|---|---|---|---|
| 27/10/2026 | v0 pipeline check | Agent v0 + hạ tầng đo xong (`c720d5e`) | gold / FakeLLM (phát lại gold) | 100,00 | 100,00 | | Chỉ kiểm vòng lặp, guard, ghi file và bộ chấm; chưa có model thật | |
| 27/10/2026 | smoke gold T4 v3, v6 | Prompt `system_v0`, hạn giờ dự đoán (bản cũ), 30 mẫu smoke | gold / Qwen3-4B base (vLLM 0.32, fp16) | 9,33 (smoke) | | | Lần lượt 8 và 12 mẫu dừng vì `timeout`: hạn giờ dự đoán bằng p95 bị bước đầu chậm (đọc cả prompt) kéo lên. Dẫn tới đổi sang hạn giờ thật (S12). Cách chia cũ | `smoke_gold_t4_v3`, `smoke_gold_t4_v6`, `report_gold_before_prompt_tuning.md` |
| 27/10/2026 | ASR eval T4 (PhoWhisper) | Đo riêng ASR PhoWhisper-medium trên Kaggle T4 | PhoWhisper-medium / – | WER 29,1% | | | Cách chia cũ; số viết bằng chữ, khác `audio_text` (dẫn tới quyết định không làm Normalizer) | `asr_eval_t4_v6` |
| 28/10/2026 | smoke T4 v7 | ASR large-v3 với tham số của team lead (có `initial_prompt`, chống lặp) | large-v3 / Qwen3-4B base | 40,17 (gold, smoke), 14,50 (ASR, smoke) | | | ASR trên val_seen: WER 44,0%, 190/916 câu rỗng, bỏ sót thực thể 52% | `smoke_t4_v7` |
| 28/10/2026 | ASR sweep v1 | 6 biến thể ASR trên val_seen (274 mẫu, 916 câu) | chỉ ASR | | | | Bỏ `initial_prompt` + `repetition_penalty` + `no_repeat_ngram_size`: WER 29,7%, rỗng 24, bỏ sót 31%, p95 7,6 s. PhoWhisper: WER 27,9%, rỗng 0, bỏ sót 38,5%, p95 2,9 s. Chọn large-v3 không câu mồi; baseline quyết định cuối | `asr_sweep_v1` |
| 28/10/2026 | base28 eval v2 (không hợp lệ) | Prompt v1, transcript ASR large-v3 float16 | large-v3 / Qwen3-4B base | 8,03 | 2,81 | | Hết bộ nhớ GPU T4 15 GB khi ASR chạy cạnh vLLM: 253/274 và 178/178 mẫu lỗi (431/452). Dẫn tới yêu cầu GPU ≥ 24 GB cho eval có ASR | `eval_v1` |
| 28/10/2026 | eval v3 (hỏng, Kaggle) | Thử ASR `int8_float16` qua biến `ASR_DTYPE` | large-v3 / Qwen3-4B base | | | | Lỗi script, chưa chạy mẫu nào: lệnh sửa `ASR_DTYPE` chạy trước bước chép config fp32 nên bị ghi đè | `eval_v3` |
| 28/10/2026 | eval v4 (hỏng, Kaggle) | Sửa thứ tự áp `ASR_DTYPE` | large-v3 / Qwen3-4B base | | | | Lỗi script, chưa chạy mẫu nào: kiểm tra config từ chối `asr.dtype = "int8_float16"` (lúc đó chỉ nhận auto/bfloat16/float16/float32) | `eval_v4` |
| 28/10/2026 | eval v5 (chỉ tham khảo) | ASR `int8_float16` trên T4, sau khi config nhận dtype int8 của CTranslate2 | large-v3 int8_float16 / Qwen3-4B base | | | | Chưa tải kết quả về. Không phải baseline: int8 khác cấu hình chạy trên A30 lúc chấm (`asr.dtype = "auto"`) | |
| 28/10/2026 | ASR tune v3 (Kaggle T4, chỉ ASR) | Dò từng bước 5 mô hình trên `asr_tune.txt`, đo cấu hình thắng trên val (D36) | 5 mô hình ASR | | | | val_seen / val_unseen. ChunkFormer RNNT: WER 20,0 / 19,8%, bỏ sót 32,1 / 48,1%, rỗng 0, p95 2,5 s. ChunkFormer CTC: WER 21,3 / 21,0%, bỏ sót 23,9 / 55,6%. PhoWhisper + câu mồi ngắn: WER 27,9%, bỏ sót 38,5 / 48,1%, p95 3,0 s. large-v3 (temp fallback): WER 30,6 / 30,2%, bỏ sót 31,2 / 57,4%, p95 8 s, vượt ngưỡng. Parakeet: WER 35–38%, rỗng 37 / 51. Chi tiết trong `docs/reports/asr_benchmark.md` | `artifacts/runs/kaggle/asr_tune_v3` |
