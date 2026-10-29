# configs/splits/

Danh sách id của từng tập, sinh bằng `make splits` (`lifemate/eval/make_splits.py`). Mỗi dòng một id mẫu của tập train có nhãn.

**Các file trong thư mục này không được đưa lên git** (xem `.gitignore`), vì chúng được suy ra từ dữ liệu cuộc thi. Sau khi clone, chạy lệnh sau để sinh lại:

```bash
make splits   # = python -m lifemate.eval.make_splits --smoke 30 --asr-tune 300 --en-mixed
```

Script cho kết quả giống hệt nhau ở mọi máy (seed 42). Kernel Kaggle và `ops/scripts/runpod_setup.sh` tự chạy lệnh này sau khi có dữ liệu. Nếu thiếu `meta.json`, bước dựng dữ liệu fine-tune báo lỗi thay vì chạy tiếp, để không làm lộ schema của các app giữ lại.

| File | Nội dung |
|---|---|
| `train.txt` | 1.093 mẫu dùng để huấn luyện (fold 1 đến 4). Dữ liệu fine-tune chỉ được lấy từ đây. |
| `val_seen.txt` | 274 mẫu kiểm tra (fold 0) với các app đã có trong tập học. |
| `val_unseen.txt` | 178 mẫu dùng 6 app bị giữ lại (medication, fitness, flights, bike_rental, library, subscriptions), mô phỏng app lạ ở bộ thi thật. |
| `folds/fold_0..4.txt` | 5 fold chia theo cụm câu gần trùng, phân tầng theo số call và ca hỏi lại. |
| `smoke.txt` | 30 mẫu cố định lấy từ `val_seen`, đủ ba loại đáp án, để chạy thử nhanh. |
| `asr_tune.txt` | 300 mẫu lấy từ `train`, phân tầng như `smoke`, làm tập dò cấu hình ASR (`lifemate/eval/asr_tune.py`). |
| `asr_tune_en_mixed.txt`, `val_seen_en_mixed.txt`, `val_unseen_en_mixed.txt` | Tập con của `asr_tune`, `val_seen`, `val_unseen` gồm các mẫu có lượt người dùng chêm từ tiếng Anh (`lifemate/eval/foreign_words.py`), cho benchmark ASR lần 2: 165, 152 và 95 mẫu. |
| `meta.json` | Tham số khi chia, danh sách app giữ lại, số mẫu theo loại đáp án, kết quả kiểm tra rò rỉ. |
