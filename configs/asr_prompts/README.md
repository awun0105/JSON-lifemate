# configs/asr_prompts/

Câu mồi (`initial_prompt`) cho ASR kiểu Whisper (whisper-large-v3, PhoWhisper). File nào được dùng do trường `asr.initial_prompt_file` trong cấu hình chạy quyết định (đường dẫn tính từ gốc repo). Nội dung file (bỏ khoảng trắng hai đầu) được đưa vào `generate_kwargs.initial_prompt` lúc đọc config. Đổi câu mồi thì tạo file `_v<n+1>` mới, không sửa file cũ, để kết quả cũ vẫn tái lập được.

Câu mồi đóng vai "đoạn chữ ngay trước audio": Whisper bắt chước cách viết (chính tả từ tiếng Anh, tên riêng), không làm theo lời dặn. Chỉ 223 token cuối được dùng.

| File | Nội dung | Token (large-v3 / PhoWhisper) |
|---|---|---|
| `short_v1.txt` | Câu mồi ngắn của benchmark lần 1 (thắng ở PhoWhisper), giữ để so sánh. Trùng nhiều mẫu câu thật nên lớp bảo vệ câu mồi hay báo nhầm. | 18 / 19 |
| `en_top10_v1.txt` | 10 từ tiếng Anh whisper-large-v3 chép sai nhiều nhất (đo trên 793 mẫu train không thuộc tập dò `asr_tune`). | 56 / 59 |
| `en_top20_v1.txt` | Thêm 3 câu chứa 10 từ kế tiếp, đặt trước `en_top10`. | 106 / 114 |
| `en_full_v1.txt` | Thêm 5 câu chứa khoảng 20 từ kế tiếp, đặt trước `en_top20`. | 183 / 193 |

Quy tắc viết (benchmark ASR lần 2, `docs/reports/benchmark/asr_v2.md` mục 4):

- Câu tự viết, giữ đúng từ khoá tiếng Anh nhưng đặt trong ngữ cảnh khác dữ liệu, không chép câu nào của dữ liệu thi. Lý do: lớp bảo vệ câu mồi (`asr.prompt_guard_ngram`) coi câu ASR có cụm từ trùng câu mồi là "chép lại câu mồi"; câu mồi giống lời nói thật thì câu nói đúng cũng bị báo nhầm. `tests/test_asr_prompts.py` kiểm tra trên 5 036 câu đáp án: cụm 4 từ không trùng câu nào, cụm 3 từ trùng dưới 0,5% số câu.
- Câu nói tự nhiên, không viết dạng liệt kê; không để chữ số đi liền tên (lần 1, "VTV3" trong câu mồi làm large-v3 đếm tiếp "VTV4 VTV5…").
- Từ sai nhiều nhất đặt cuối, vì phần cuối gần audio nhất và phần đầu bị cắt khi quá 223 token.
- Tên riêng (Sarah, Grace Nakamura, MinhMart…) là dữ liệu tổng hợp của cuộc thi, được phép đưa lên git.
