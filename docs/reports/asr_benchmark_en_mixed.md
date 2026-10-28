# Báo cáo benchmark ASR lần 2: câu tiếng Việt chêm tiếng Anh

Ngày: 29/10/2026. Trạng thái: **đang chạy**. Thử nghiệm A và D (offline) đã có kết quả. Thử nghiệm B và C cần GPU,
đang chạy trên Kaggle T4; kết quả sẽ điền vào mục 4.7 và 5.5.

Báo cáo trả lời ba câu hỏi:

1. Trong bốn mô hình ASR tốt nhất của lần 1 (`docs/reports/asr_benchmark.md`), mô hình nào chép đúng nhất những câu
   tiếng Việt có chêm từ tiếng Anh?
2. Câu mồi (`initial_prompt`) chứa từ tiếng Anh, đi kèm một lớp bảo vệ, có giúp whisper-large-v3 và PhoWhisper chép
   đúng từ tiếng Anh hơn mà không làm hỏng câu thường không?
3. Ghép hai mô hình (ChunkFormer RNNT chép chính, Whisper bổ sung từ tiếng Anh) có cho chữ tốt hơn từng mô hình riêng
   lẻ không?

Phạm vi: chỉ đo ASR (chữ chép từ âm thanh). Không chạy LLM, không chấm điểm thi.

**Kết quả thô.** Không có trên git.

| Thư mục | Nội dung |
|---|---|
| `artifacts/runs/kaggle/asr/10-28_tune_v3/` | Kết quả lần 1, dùng lại cho thử nghiệm A và D: `<mô hình>__<lần đo>.hyps.jsonl` (chữ từng câu), `.json` (chỉ số), `.toml` (config) |
| `artifacts/runs/local/asr/10-29_en_mixed_offline_v1/` | Lần chấm offline đầu tiên (chỉ số bản đầu, trước khi sửa cách đếm ở mục 8); chỉ để đối chiếu |
| `artifacts/runs/local/asr/10-29_en_mixed_offline_v2/` | Thử nghiệm A và D: `<mô hình>__<tập>.json` (chấm lại), `compare__<tập>.{md,json}` (bảng so sánh có khoảng tin cậy), `merge_*.hyps.jsonl` (chữ đã ghép), `merge_effect.json` |
| `artifacts/runs/kaggle/asr/10-29_tune_v5_en_mixed/` | Thử nghiệm B và C (chạy GPU), cùng cấu trúc file với lần 1 |

## Mục lục

- [0. Bối cảnh và thuật ngữ](#0-bối-cảnh-và-thuật-ngữ)
- [1. Vấn đề chung và mục tiêu](#1-vấn-đề-chung-và-mục-tiêu)
- [2. Dữ liệu, thước đo và công cụ dùng chung](#2-dữ-liệu-thước-đo-và-công-cụ-dùng-chung)
- [3. Thử nghiệm A: chấm offline bốn mô hình trên câu chêm tiếng Anh](#3-thử-nghiệm-a-chấm-offline-bốn-mô-hình-trên-câu-chêm-tiếng-anh)
- [4. Thử nghiệm B: câu mồi tiếng Anh và lớp bảo vệ câu mồi](#4-thử-nghiệm-b-câu-mồi-tiếng-anh-và-lớp-bảo-vệ-câu-mồi)
- [5. Thử nghiệm C: kiểm chứng chấm offline bằng một lần chạy thật](#5-thử-nghiệm-c-kiểm-chứng-chấm-offline-bằng-một-lần-chạy-thật)
- [6. Thử nghiệm D: ghép ChunkFormer RNNT với Whisper](#6-thử-nghiệm-d-ghép-chunkformer-rnnt-với-whisper)
- [7. Tổng hợp và kết luận](#7-tổng-hợp-và-kết-luận)
- [8. Rà code trước khi chạy: lỗi đã vá](#8-rà-code-trước-khi-chạy-lỗi-đã-vá)
- [Giới hạn đã biết](#giới-hạn-đã-biết)

## 0. Bối cảnh và thuật ngữ

Bài thi LifeMate nhận lời nói tiếng Việt của người dùng, chép thành chữ (ASR), rồi cho LLM gọi tool. Đề bài ghi rõ lời
nói "bằng tiếng Việt, có thể xen lẫn các từ tiếng Anh thông dụng" (`docs/overview.md`). Dữ liệu train của BTC xác nhận:
833 trên 1 545 mẫu có ít nhất một từ ngoại (Zalo, SJC, check, voucher, tên người nước ngoài…). Lần 1 chọn mô hình theo
mọi câu; lần 2 nhìn riêng các câu chêm tiếng Anh.

| Thuật ngữ | Nghĩa |
|---|---|
| Câu chêm tiếng Anh | Câu nói tiếng Việt có ít nhất một từ ngoại trong đáp án (mục 2.1) |
| Từ ngoại, 4 nhóm | Từ tìm được bằng quy tắc ở mục 2.1, chia nhóm: **từ thường** (check, online), **viết tắt** (SMS, SJC, VTV), **tên** (Sarah, MinhMart, Zalo), **từ mượn** đã Việt hoá (tivi, gara) |
| Chép đúng từ ngoại | Từ ngoại của đáp án có mặt trong chữ ASR, so sau khi bỏ dấu và chữ hoa (mục 2.3) |
| Câu mồi (`initial_prompt`) | Đoạn chữ đưa cho Whisper như "câu trước đó". Whisper bắt chước cách viết của câu mồi (chính tả, tên riêng), không làm theo lời dặn. Chỉ 223 token cuối được dùng |
| Lớp bảo vệ câu mồi | Bước kiểm tra sau khi chép có câu mồi: câu nào có dấu hiệu hỏng thì chép lại không câu mồi (mục 4.3.2) |
| Chấm offline | Chấm lại chữ ASR đã ghi ở lần chạy trước (`hyps.jsonl`) bằng chỉ số mới, không chạy lại mô hình |
| Tập dò, tập kiểm tra | Tập dò (`asr_tune`) dùng để chọn cấu hình; tập kiểm tra (`val_seen`, `val_unseen`) chỉ dùng để đo một lần sau khi đã chọn |
| Khoảng tin cậy 95% (CI) | Khoảng chứa giá trị thật với độ tin 95%, ước lượng bằng bootstrap (mục 2.4). Hai hệ thống chỉ được coi là khác nhau khi CI của chênh lệch không chứa 0 |
| Ghép mô hình | Kết hợp chữ của hai ASR cho cùng một câu (mục 6) |
| Oracle | Trần lý thuyết: với mỗi câu chọn bản tốt hơn trong hai bản, dựa vào đáp án. Không làm được lúc thi, chỉ để biết ghép tối đa được bao nhiêu |

## 1. Vấn đề chung và mục tiêu

**Vấn đề.** Lần 1 xếp hạng theo WER và tỉ lệ bỏ sót giá trị tham số trên mọi câu. Hai chỉ số này không cho biết mô hình
chép từ tiếng Anh ra sao: một câu dài 20 từ chỉ sai một chữ "check" vẫn có WER thấp. Trong khi đó, từ tiếng Anh hay là
từ quyết định tool nào được gọi: "Zalo" quyết định app nhắn tin, "SJC" quyết định tra giá vàng, "voucher", "playlist",
"meeting" là tên đối tượng. Lần 1 đã thấy large-v3 gần như không bao giờ viết đúng "Zalo" (1/19 trên val_seen).

**Mục tiêu.**

1. Đo bốn mô hình tốt nhất của lần 1 (ChunkFormer RNNT, ChunkFormer CTC, PhoWhisper, large-v3) trên riêng các câu chêm
   tiếng Anh, có khoảng tin cậy, theo từng nhóm từ (thử nghiệm A).
2. Thử cải thiện hai mô hình Whisper bằng câu mồi tiếng Anh và lớp bảo vệ câu mồi, đo trên cả tập của lần 1 (để biết câu
   thường có bị hại không) lẫn tập câu chêm tiếng Anh (thử nghiệm B).
3. Kiểm chứng rằng chấm offline cho đúng số như chạy thật (thử nghiệm C).
4. Thử ghép RNNT với một mô hình Whisper (thử nghiệm D).

**Bỏ Parakeet.** Parakeet đã bị loại ở lần 1, và chấm offline chỉ chép đúng 14% từ tiếng Anh, thấp nhất trong năm mô
hình.

## 2. Dữ liệu, thước đo và công cụ dùng chung

### 2.1. Cách xác định từ tiếng Anh

**Phương pháp chung.** Đáp án của BTC (`audio_text`) không đánh dấu từ nào là tiếng Anh, nên phải tự tìm. Có hai cách:
rà tay một danh sách từ, hoặc dùng quy tắc. Lần 2 dùng **quy tắc**, vì danh sách rà tay lấy từ dữ liệu thi thì không được
đưa lên git (`AGENTS.md`), mà kernel Kaggle và máy khác cần cùng một cách tìm.

**Code.** `lifemate/eval/en_mixed.py`, hàm `english_words(text, lowercase_seen)`. Với từng từ chỉ gồm chữ cái, dài từ 2
ký tự (chữ số là ranh giới từ, nên "RON95" cho từ "RON"):

1. Từ có dấu tiếng Việt (kể cả "đ") là tiếng Việt, bỏ qua.
2. Từ trong `LOANWORDS` (tivi, gara) thuộc nhóm **từ mượn**: viết theo chính tả Việt, mô hình tiếng Việt chép như âm
   tiết Việt, nên tách riêng để không làm đẹp số của từ tiếng Anh thật.
3. Từ viết hoa toàn bộ (SMS, SJC, VTV) thuộc nhóm **viết tắt**.
4. Từ không dấu mà **không ghép được thành một âm tiết tiếng Việt** (phụ âm đầu + 1 đến 3 nguyên âm + phụ âm cuối hợp
   lệ), hoặc có chữ f/j/w/z, hoặc nằm trong `EN_LIKE_VI` (chat, top, hot: tiếng Anh nhưng trông như âm tiết Việt), là từ
   ngoại. Viết thường thì thuộc nhóm **từ thường**; viết hoa chữ đầu giữa câu thì thuộc nhóm **tên**. Từ viết hoa đứng
   đầu câu ("Check…" hay "Sarah…") không phân biệt được bằng chữ hoa, nên xếp **từ thường** nếu từ đó từng xuất hiện viết
   thường ở đâu đó trong tập đang chấm (`lowercase_seen`), còn lại xếp **tên**.

Ví dụ: "Nhắn Zalo cho Lan, giá vàng SJC, xăng RON95" cho Zalo (tên), SJC và RON (viết tắt); "Lan" ghép được âm tiết
Việt nên bị bỏ qua, đúng ý muốn vì tên Việt không dấu không phải từ ngoại.

**Giới hạn.** Quy tắc bỏ sót từ tiếng Anh trông như âm tiết Việt mà không có trong `EN_LIKE_VI` (ví dụ "man", "ban").
Tên tiếng Việt viết liền không dấu ("MinhMart") được tính là tên ngoại; đây là chủ ý, vì đó là tên thương hiệu.

### 2.2. Các tập và lý do

**Phương pháp chung.** Không chia lại dữ liệu. Lần 2 lấy **tập con** của các tập đã có: những mẫu có ít nhất một lượt
người dùng (lượt cuối hoặc trong lịch sử) chêm từ ngoại.

**Code.** `lifemate/eval/make_splits.py --en-mixed` (hàm `en_mixed_ids`), chạy trong `make splits`. Ghi ba file vào
`configs/splits/` (không lên git):

| Tập | Lấy từ | Số mẫu | Số câu chêm tiếng Anh | Số từ ngoại | Dùng để |
|---|---|---|---|---|---|
| `asr_tune_en_mixed` | `asr_tune` (300 mẫu của train) | 165 | 276 | — | Báo cáo phụ trên tập dò |
| `val_seen_en_mixed` | `val_seen` (274) | 152 | 245 | 335 | Đo kết quả |
| `val_unseen_en_mixed` | `val_unseen` (178) | 95 | 162 | 252 | Đo kết quả, app chưa gặp |

Lý do:

- **Dùng lại tập của lần 1** thì mọi chữ ASR đã chép ở lần 1 dùng lại được, và kết quả so được với lần 1.
  `asr_tune.txt` sinh lại hôm nay khớp đúng từng id với lần 1 (seed cố định).
- **Vẫn tách tập dò và tập kiểm tra.** Nếu vừa chọn cấu hình vừa báo kết quả trên cùng một tập, cấu hình thắng khớp
  riêng với tập đó và con số đẹp hơn thực tế.
- **Không dò trên 793 mẫu train còn lại**, vì 793 mẫu này đã dùng để chọn từ cho câu mồi (mục 4.3.1). Dò trên đó thì câu
  mồi khớp sẵn với tập dò.
- **Giữ seen/unseen.** Hai tập chia theo app chứ không theo ngôn ngữ, nhưng app lạ kéo theo từ lạ (vitamin, homestay,
  tên hãng bay), nên `val_unseen` cho biết mô hình có chép đúng từ tiếng Anh chưa gặp ở tập dò không. Bộ thi riêng của
  BTC có app lạ.
- **Thử nghiệm B dò và đo trên tập đầy đủ**, rồi chấm lại tập con. Tập con nằm trong tập đầy đủ, nên một lần chạy GPU
  trả lời cả hai câu hỏi (câu chêm tiếng Anh có tốt hơn không, câu thường có bị hại không).

### 2.3. Thước đo

Mọi tỉ lệ tính trên cả tập: cộng tổng lỗi rồi chia tổng, không lấy trung bình theo câu. Code: `lifemate/eval/eval_asr.py`.

| Chỉ số | Cách tính | Vì sao cần |
|---|---|---|
| **`en_word_recall`** (chép đúng từ ngoại) | Số từ ngoại của đáp án có mặt trong chữ ASR (so sau khi bỏ dấu, chữ thường), chia tổng số từ ngoại. Từ xuất hiện k lần trong đáp án chỉ được tính đúng tối đa bằng số lần nó có trong chữ ASR (`en_mixed.hits`) | Chỉ số chính của lần 2 |
| `en_word_recall_lenient` | Như trên, nhưng chấp nhận cách viết tương đương LLM vẫn hiểu: "TV" cho "tivi", "okay"/"oke" cho "ok", "mail" cho "email", tên ghép bị tách ("Minh Mart" cho "MinhMart") | Tách lỗi nghe thật khỏi lỗi cách viết |
| `en_groups` | Chép đúng theo 4 nhóm từ | Biết mô hình yếu ở loại từ nào |
| **`en_mixed_wer`** | WER chỉ trên các câu chêm tiếng Anh | Chất lượng chung của câu chêm |
| **`vi_only_wer`** | WER trên các câu **không** có từ ngoại | Câu mồi tiếng Anh hoặc phép ghép có làm hại câu thuần Việt không |
| `prompt_insertions` | Số lần một từ ngoại của câu mồi xuất hiện trong chữ ASR mà đáp án câu đó không có | Câu mồi có làm mô hình viết ra từ không được nói không |
| `en_prompt` | Chép đúng, tách theo từ **có** và **không có** trong câu mồi | Câu mồi giúp thật hay chỉ khớp sẵn từ của nó |
| `n_prompt_retries` | Số câu lớp bảo vệ đã chép lại | Tần suất câu mồi làm hỏng câu, và chi phí thời gian |
| Giữ từ lần 1 | WER, `wer_nonum`, CER, bỏ sót giá trị tham số, câu rỗng, câu bịa, `terms` (Zalo, VTV…), p50/p95/max | So được với lần 1 |
| `counts` | Tử số và mẫu số của các tỉ lệ trên | Để cộng dồn khi bootstrap |

**Về tên "bỏ sót tên riêng" của lần 1.** Chỉ số này thực ra đo **giá trị tham số** trong tool call của đáp án mà người
dùng có nói ra (tên người, kênh, số…), không có bước nhận diện tên riêng. Báo cáo này gọi là "bỏ sót giá trị tham số".

### 2.4. Công cụ chung

| Module | Việc | Thuật toán |
|---|---|---|
| `lifemate/eval/rescore_asr.py` | Chấm lại `hyps.jsonl` trên một tập con | `ReplayTranscriber` dựng lại transcript từ chữ đã ghi rồi gọi đúng hàm `evaluate` của `eval_asr`, nên chỉ số giống hệt chạy thật. Thời gian bằng 0 (lấy từ lần chạy gốc) |
| `lifemate/eval/asr_compare.py` | Bảng so sánh nhiều hệ thống, có khoảng tin cậy | Chấm từng mẫu một lần để có tử số và mẫu số (`AsrReport.counts`). Bootstrap 1 000 vòng, seed 0: mỗi vòng lấy ngẫu nhiên có hoàn lại `N` mẫu, cộng tử và mẫu, tính tỉ lệ. CI là phân vị 2,5 và 97,5. **Ghép cặp**: mọi hệ thống dùng cùng các mẫu ở mỗi vòng, nên CI của chênh lệch hẹp và công bằng. Lấy mẫu theo **mẫu** chứ không theo câu, vì các câu cùng mẫu (cùng người nói) không độc lập |

**Kiểm tra công cụ.** Chấm lại toàn bộ val_seen và val_unseen của 4 mô hình bằng `rescore_asr` cho **đúng từng số**
(WER, `wer_nonum`, CER, bỏ sót, câu rỗng, câu bịa, số câu) như báo cáo gốc của lần 1, ở cả 8 cặp mô hình × tập.

## 3. Thử nghiệm A: chấm offline bốn mô hình trên câu chêm tiếng Anh

### 3.1. Vấn đề

Chưa biết mô hình nào chép từ tiếng Anh tốt nhất, và thứ hạng theo WER của lần 1 có giữ nguyên trên câu chêm tiếng Anh
không.

### 3.2. Mục tiêu

Xếp hạng bốn mô hình theo chỉ số tiếng Anh, có khoảng tin cậy, không tốn GPU.

### 3.3. Phương pháp

**Chung.** Giải mã ở lần 1 là tất định (trừ large-v3, xem dưới), nên chạy lại cùng cấu hình cho cùng chữ. Lần 1 đã chạy
toàn bộ val_seen và val_unseen, nên mọi mẫu của tập con đã có chữ. Chấm lại chữ đó bằng chỉ số mới cho đúng số mà một lần
chạy lại sẽ cho.

**Code.** `rescore_asr` cho từng mô hình × tập, `asr_compare` cho bảng có CI. Lệnh đầy đủ ở `docs/experiments.md`.

**Lưu ý về large-v3.** Cấu hình thắng của large-v3 ở lần 1 dùng nhiệt độ thử lại (`temperature = [0, 0,2, …, 1]`): khi
câu hỏng, mô hình lấy mẫu ngẫu nhiên. Cấu hình này không tất định và nhiều khả năng trái thể lệ (mục 4.3.4), nên số của
large-v3 ở đây chỉ để tham khảo. Thử nghiệm B đo lại large-v3 với cấu hình tất định.

### 3.4. Chỉ số

`en_word_recall` (chính), `en_word_recall_lenient`, `en_groups`, `en_mixed_wer`, `vi_only_wer`, WER, bỏ sót giá trị
tham số; CI 95% và CI của chênh lệch so với ChunkFormer RNNT.

### 3.5. Kết quả

Tập câu chêm tiếng Anh. Số trong ngoặc là CI 95%.

| Mô hình | Tập | Chép đúng từ ngoại | Bản nới lỏng | WER câu chêm | WER câu thuần Việt | p95 lần 1 (s) |
|---|---|---|---|---|---|---|
| ChunkFormer RNNT | seen | 49,9 [44,3; 55,3] | 51,9 | **24,8** [22,8; 26,7] | **16,9** | 2,5 |
| | unseen | 51,6 [44,8; 58,6] | 52,4 | **24,1** [21,6; 26,8] | **17,0** | 2,5 |
| ChunkFormer CTC | seen | 34,6 [29,8; 39,7] | 40,6 | 27,5 | 17,5 | 2,7 |
| | unseen | 39,7 [32,2; 47,4] | 40,5 | 26,9 | 17,0 | 2,3 |
| PhoWhisper (câu mồi ngắn) | seen | 36,1 [30,6; 41,9] | 39,4 | 32,5 | 24,7 | 3,0 |
| | unseen | 36,5 [29,7; 43,4] | 38,1 | 33,1 | 24,6 | 3,0 |
| large-v3 (tham khảo) | seen | **51,6** [46,4; 56,7] | **54,9** | 32,8 | 29,2 | 8,2 |
| | unseen | **56,0** [48,1; 63,4] | **57,5** | 33,9 | 29,1 | 7,9 |

Theo nhóm từ (số từ chép đúng / tổng):

| Mô hình | Tập | Từ thường | Viết tắt | Tên | Từ mượn |
|---|---|---|---|---|---|
| ChunkFormer RNNT | seen | 50/105 | **47/82** | 60/133 | **10/15** |
| | unseen | 56/97 | 18/23 | 55/130 | 1/2 |
| ChunkFormer CTC | seen | 47/105 | 29/82 | 40/133 | 0/15 |
| | unseen | 48/97 | 17/23 | 34/130 | 1/2 |
| PhoWhisper | seen | 34/105 | 24/82 | 58/133 | 5/15 |
| | unseen | 40/97 | 11/23 | 41/130 | 0/2 |
| large-v3 | seen | **61/105** | 43/82 | **67/133** | 2/15 |
| | unseen | 52/97 | **21/23** | **67/130** | 1/2 |

Chênh lệch so với RNNT (điểm %, CI 95% của chênh lệch), tập câu chêm tiếng Anh:

| Mô hình | Chép đúng từ ngoại, seen | unseen | WER câu chêm, seen | unseen |
|---|---|---|---|---|
| CTC | −15,2 [−20,6; −9,3] | −11,9 [−17,2; −6,2] | +2,7 [+1,2; +4,3] | +2,8 [+1,2; +4,4] |
| PhoWhisper | −13,7 [−19,9; −7,1] | −15,1 [−22,0; −8,7] | +7,6 [+5,7; +9,4] | +9,0 [+7,1; +11,0] |
| large-v3 | +1,8 [−4,5; +8,3] | +4,4 [−2,8; +10,7] | +7,9 [+5,4; +10,5] | +9,8 [+7,0; +12,6] |

### 3.6. Phân tích

- **ChunkFormer RNNT vẫn tốt nhất xét tổng thể.** WER câu chêm thấp nhất (khoảng 24%), thấp hơn mọi mô hình khác với CI
  không chứa 0. Chép đúng khoảng một nửa số từ ngoại. Mạnh ở viết tắt (47/82 trên seen) và từ mượn.
- **large-v3 chép từ tiếng Anh ngang hoặc hơn RNNT một chút** (+1,8 và +4,4 điểm), nhưng CI chứa 0 nên chưa đủ để nói
  hơn. Mạnh nhất ở **tên** (67/133 và 67/130) và từ thường; yếu ở từ mượn ("tivi" thường thành "TV": bản nới lỏng tăng
  3 điểm). Bù lại, phần tiếng Việt kém xa: WER câu thuần Việt khoảng 29%, gần gấp đôi RNNT.
- **PhoWhisper và ChunkFormer CTC chép từ tiếng Anh kém rõ rệt** (thấp hơn RNNT 12–15 điểm, CI không chứa 0).
  PhoWhisper được VinAI huấn luyện thêm trên tiếng Việt, nên hay "Việt hoá" từ tiếng Anh. CTC yếu nhất ở tên và viết tắt.
- **Thứ hạng của lần 1 giữ nguyên ở đầu bảng** (RNNT nhất), nhưng đảo ở giữa: theo từ tiếng Anh, large-v3 đứng thứ hai,
  còn CTC và PhoWhisper tụt xuống.
- **Hai mô hình mạnh ở hai phần khác nhau**: RNNT mạnh ở phần tiếng Việt, large-v3 mạnh ở tên và từ tiếng Anh. Đây là cơ
  sở của thử nghiệm D.

### 3.7. Kết luận

Trên câu chêm tiếng Anh, ChunkFormer RNNT vẫn là mô hình đơn tốt nhất. PhoWhisper không còn là ứng viên dự phòng tốt cho
câu chêm tiếng Anh. large-v3 có giá trị như nguồn bổ sung từ tiếng Anh, không phải mô hình chính.

## 4. Thử nghiệm B: câu mồi tiếng Anh và lớp bảo vệ câu mồi

### 4.1. Vấn đề

Hai mô hình Whisper là hai mô hình duy nhất nhận câu mồi. Lần 1 cho thấy câu mồi **có giúp viết đúng tên** (PhoWhisper
với câu mồi ngắn viết đúng "Zalo" 16/19), nhưng với large-v3 thì **mọi câu mồi đều làm kết quả kém đi**. Đọc chữ của lần
1 trên tập dò thấy ba cơ chế hỏng:

1. **Chép lại câu mồi.** Audio khó nghe thì large-v3 bỏ audio và in lại câu mồi: "Chuyển sang kênh VTV3." xuất hiện ở
   42 câu không hề nói câu đó; `hotwords` bị in khoảng 200 lần.
2. **Đếm tiếp danh sách.** Câu mồi có số đi liền tên làm mô hình sinh "VTV3 VTV4 VTV5 … VTV21"; "LifeMate" trong câu mồi
   của lead sinh "Mate 2.0.0.0…".
3. **Bịa thêm câu phụ đề.** "Xin chào các bạn.", "Hãy subscribe…".

Đồng thời large-v3 không câu mồi sai có hệ thống những từ tiếng Anh nhất định. Đo trên 793 mẫu train không thuộc tập dò
(chữ large-v3 có sẵn trong `data/asr/train_asr.json`): 458 lần xuất hiện từ ngoại, sai 206 lần; 10 từ sai nhiều nhất
chiếm 43% số lần sai, 30 từ đầu chiếm 65%:

| Từ | Đúng / tổng | Ví dụ chép sai (tập dò) |
|---|---|---|
| Zalo | 1/36 | "Nhắn que **Yellow**", "**Giá lô** nhé em" |
| SJC | 1/19 | "vàng **Ếch, trí trí**", "**SGC**" |
| SMS | 32/41 | |
| check | 5/11 | "**Chắc** danh mục cổ phiếu…" |
| voucher | 3/8 | "đổi hết ra **vào chờ** 100.000" |
| note, book, list, mail | | "thêm vô **nốt**", "**búc** dùm cô", "**lít** đi chợ", "cái **mèo** sao kê" |
| podcast, deadline | | "một tấm **postcard**", "hôm nay **Đất Lai**…" |

35% số lần sai còn lại là đuôi dài: 51 tên người và thương hiệu chỉ gặp một lần. Câu mồi không liệt kê được tập mở này,
chỉ giúp gián tiếp bằng cách đẩy mô hình giữ chính tả tiếng Anh.

### 4.2. Mục tiêu

Tìm câu mồi và mức bảo vệ làm tăng `en_word_recall` của large-v3 và PhoWhisper mà không làm tăng đáng kể
`vi_only_wer`, câu rỗng, câu bịa và `prompt_insertions`. Đồng thời có số của hai mô hình này với **cấu hình tất định**
trên cả tập của lần 1 và tập câu chêm tiếng Anh.

### 4.3. Phương pháp

#### 4.3.1. Thiết kế câu mồi

**Chung.** Viết câu mồi chứa những từ large-v3 đang sai, theo các nguyên tắc rút ra từ ba cơ chế hỏng ở mục 4.1 và từ
hướng dẫn prompt Whisper của OpenAI (Whisper bắt chước cách viết của câu mồi; câu dài bám cách viết ổn định hơn câu ngắn;
chỉ 224 token cuối được dùng):

| Nguyên tắc | Lý do |
|---|---|
| Câu nói tự nhiên, không viết dạng danh sách | `hotwords` và câu mồi của lead (dạng liệt kê) là hai lựa chọn tệ nhất ở lần 1 |
| Không có chữ số đi liền tên ("VTV" thay cho "VTV3", "xăng RON" thay cho "RON95") | Chữ số làm mô hình đếm tiếp |
| Không có tên sản phẩm lạ kiểu "LifeMate" | Sinh ra "Mate 2.0.0…" |
| Chỉ chứa từ đang sai, lấy theo số đo trên 793 mẫu train ngoài tập dò | Không dò trên chính tập đã dùng để chọn từ |
| Từ sai nhiều nhất (Zalo, SJC, SMS, check) đặt cuối | Phần cuối gần audio nhất; câu quá dài thì phần đầu bị cắt |
| Chính tả theo đáp án của BTC ("MinhMart", "SJC") | Viết đúng như đáp án mới được tính là chép đúng |
| **Câu tự viết, không chép câu nào của dữ liệu**, giữ từ khoá nhưng đặt vào ngữ cảnh khác | Lớp bảo vệ coi câu ASR có cụm từ trùng câu mồi là "chép lại câu mồi". Câu mồi giống lời nói thật thì câu nói đúng cũng bị báo nhầm. Bản nháp đầu (viết theo mẫu câu thật, có "nhắn SMS cho") báo nhầm 2,1–2,6% câu chuẩn ở cụm 3 từ |

**Ba câu mồi lồng nhau theo độ dài** (`configs/asr_prompts/`): câu dài hơn bằng câu ngắn hơn cộng thêm các câu đặt phía
trước, nên kết quả trả lời thẳng câu hỏi "dài hơn có tốt hơn không".

| File | Từ khoá | Token (large-v3 / PhoWhisper) | Báo nhầm trên 5 036 câu chuẩn (cụm 4 / 3 từ) |
|---|---|---|---|
| `short_v1.txt` (câu mồi ngắn của lần 1, để so sánh) | Zalo, VTV3 | 18 / 19 | không yêu cầu (trùng nhiều câu thật) |
| `en_top10_v1.txt` | check, file, book, robot, online, Sarah, voucher, SMS, Zalo, SJC | 56 / 59 | 0 / 0 |
| `en_top20_v1.txt` | thêm tivi, list, chill, camera, gara, mail, tour, workshop, Greenhaven Foods, Brightpath, MinhMart | 106 / 114 | 0 / 0 |
| `en_full_v1.txt` | thêm QR, app, email, Grace Nakamura, Peter Hale, laptop, demo, Nexora Labs, note, MinhPoint, podcast, playlist, meeting, deadline, RON, VTV, HTV | 183 / 193 | 0 / 2 câu (0,04%, đều do tên "Grace Nakamura") |

Nội dung `en_top10_v1.txt`: "Ok, robot đồ chơi hết pin. File ảnh Sarah giữ, tối book bàn online. Voucher nằm trong SMS,
check kỹ rồi gửi Zalo cả xóm, coi miếng SJC luôn."

**Code.**

- Câu mồi là **file có phiên bản**, giống system prompt của LLM: config `[asr]` trỏ tới file bằng khoá mới
  `initial_prompt_file`; `load_config` (`lifemate/core/config.py`) đọc file vào `generate_kwargs.initial_prompt`, báo lỗi
  khi file thiếu, rỗng, hoặc khai cả hai cách. Đổi câu mồi thì tạo file `_v2`, không sửa file cũ.
- `tests/test_asr_prompts.py` (marker `data`) giữ quy tắc "không trùng lời nói thật": với chữ chuẩn của toàn bộ 5 036
  câu, cụm 4 từ phải trùng 0 câu và cụm 3 từ dưới 0,5%. Câu mồi mới nào vi phạm thì test fail.

#### 4.3.2. Lớp bảo vệ câu mồi

**Chung.** Câu mồi có ích ở đa số câu nhưng làm hỏng một số câu. Thay vì bỏ câu mồi, phát hiện riêng các câu hỏng rồi
chép lại chúng **không câu mồi**. Việc phát hiện chỉ nhìn chữ vừa chép ra, không cần đáp án, nên dùng được lúc thi.

**Thuật toán** (`lifemate/backends/asr_guard.py`, hàm `needs_retry`). Một câu chép với câu mồi bị coi là hỏng khi có một
trong bốn dấu hiệu:

1. Rỗng sau khi lọc.
2. Bộ lọc ảo giác (`asr_sanitizer.sanitize`) có cắt gì đó: mô hình đang bịa câu phụ đề.
3. Có **n từ liên tiếp trùng câu mồi** (`echoes_prompt`; n = `prompt_guard_ngram`, 3 hoặc 4): mô hình chép lại câu mồi.
4. **Tỉ lệ nén zlib** lớn hơn `prompt_guard_max_compression` (2,4): chữ lặp vô hạn nén được rất nhỏ. Ngưỡng 2,4 là ngưỡng
   faster-whisper dùng để phát hiện câu lặp.

Ví dụ với câu mồi `short_v1`: "Chuyển sang kênh VTV3." trùng 3 từ "chuyển sang kênh" nên hỏng; "Nhắn Zalo giùm mẹ nha."
chỉ trùng 2 từ nên giữ. "Nhắn Zalo cho mẹ nha." trùng 3 từ: ở mức 3 từ bị báo nhầm, ở mức 4 từ thì không. Đây là lý do dò
cả hai mức.

**Code.**

- `ASRTranscriber._guard` (`lifemate/backends/asr.py`): sau khi chép mọi câu của một mẫu, kiểm tra từng câu, gom các câu
  hỏng, gọi `transcribe_batch_unprompted` theo batch, thay kết quả vào (cả chữ trước lọc), tăng bộ đếm
  `n_prompt_retries` và ghi log `asr_prompt_retry`.
- `transcribe_batch_unprompted` có ở hai engine Whisper: `FasterWhisperASR` bỏ `initial_prompt` và `hotwords`,
  `HFSeq2SeqASR` bỏ `prompt_ids`. Dùng cùng mô hình đã nạp, không tốn thêm bộ nhớ.
- Config: `prompt_guard_ngram` (0 là tắt, mặc định; phải từ 2 trở lên; cần có câu mồi), `prompt_guard_max_compression`.
  Bật lớp bảo vệ cho engine không chép lại được (ChunkFormer, NeMo) thì báo lỗi ngay lúc dựng.
- Vẫn tất định: không có ngẫu nhiên.

#### 4.3.3. Dò 13 tổ hợp cùng lúc

**Chung.** Câu mồi và lớp bảo vệ được dò **cùng lúc**: 1 lựa chọn không câu mồi + 4 câu mồi × 3 mức bảo vệ (tắt, 3 từ,
4 từ) = 13 tổ hợp trong một bước.

**Lý do không dò lần lượt.** Nếu chọn câu mồi trước khi bật lớp bảo vệ, câu mồi dài (giúp nhiều nhất về từ tiếng Anh
nhưng cũng hay bị chép lại nhất) dễ thua ngay ở bước đầu, và không bao giờ được thử cùng lớp bảo vệ, dù chính lớp bảo vệ
là thứ cứu nó. Số offline của lần 1 cho thấy đúng hiện tượng này (mục 4.6): câu mồi của lead không bảo vệ là tệ nhất,
có bảo vệ thì chép đúng tiếng Anh cao nhất.

**Code** (`lifemate/eval/asr_tune.py`):

- `PROMPT_FILES`, `GUARDS`, `prompt_options()` sinh 13 lựa chọn của bước `prompt`. Tên lựa chọn dạng `en_full+g4`.
- `--stages prompt`: chỉ dò bước câu mồi, giữ nguyên các bước khác.
- `--start-option engine=vad`: bật VAD từ đầu cho large-v3; engine không hỗ trợ (PhoWhisper, ChunkFormer) thì bỏ qua.
- `--objective en_mixed`: hàm mục tiêu (nhỏ hơn là tốt hơn) = (vượt p95 6 s hay không; bỏ sót giá trị tham số + tỉ lệ
  câu rỗng + (1 − `en_word_recall`) + `prompt_insertions` / số từ ngoại; `wer_nonum`). So theo thứ tự từng phần, hoà thì
  giữ lựa chọn đứng trước. Hàm mục tiêu tính trên **cả** tập dò, nên câu mồi làm hại câu thường sẽ bị phạt qua bỏ sót và
  câu rỗng.
- Đã bỏ khỏi `STAGES`: câu mồi của lead, `hotwords`, nhiệt độ thử lại (D40).

#### 4.3.4. Cấu hình xuất phát

| Mô hình | Cấu hình | Lý do |
|---|---|---|
| large-v3 | giải mã tham lam, VAD (cắt khoảng lặng ≥ 500 ms), không chống lặp, `condition_on_previous_text = false` | Thay nhiệt độ thử lại bằng VAD: nhiệt độ thử lại lấy mẫu ngẫu nhiên khi câu hỏng nên không tất định, trái yêu cầu "greedy hoặc `temperature = 0`" của thể lệ, và chỉ hơn VAD 1 giá trị tham số trên 123 ở tập dò lần 1 (40,7% so với 41,5%), trong khi VAD có ít câu bịa hơn một nửa (8 so với 18) |
| PhoWhisper | giải mã tham lam, không chống lặp | Như cấu hình thắng lần 1; engine transformers không có VAD |

#### 4.3.5. Tập và số lần đo

| Bước | Tập | Số lần đo mỗi mô hình |
|---|---|---|
| Dò 13 tổ hợp | `asr_tune` đầy đủ (300 mẫu, 1 024 câu) | 13 |
| Đo cấu hình thắng, đúng một lần | `val_seen`, `val_unseen` đầy đủ | 2 |
| Số của tập câu chêm tiếng Anh | chấm lại tập con bằng `rescore_asr` | 0 (không chạy GPU) |

Ước tính trên T4 (theo p95 của lần 1): large-v3 khoảng 20 phút mỗi lần đo trên tập dò, PhoWhisper khoảng 9 phút. Kernel
Kaggle `ops/kaggle/asr_tune` chạy `ASR_TUNE = large-v3,phowhisper,chunkformer-rnnt` với
`ASR_TUNE_ARGS = --stages prompt --start-option engine=vad --objective en_mixed`. Trước đó kernel CPU
`ops/kaggle/asr_preflight` chạy mọi tổ hợp trên 3 mẫu để bắt lỗi cài đặt và tham số.

### 4.4. Code đã chuẩn bị (tóm tắt)

| File | Vai trò |
|---|---|
| `configs/asr_prompts/*.txt`, `README.md` | Câu mồi có phiên bản và quy tắc viết |
| `lifemate/core/config.py` | `initial_prompt_file`, `prompt_guard_ngram`, `prompt_guard_max_compression`, kiểm tra giá trị |
| `lifemate/backends/asr_guard.py` | Bốn dấu hiệu câu hỏng |
| `lifemate/backends/asr.py` | `ASRTranscriber._guard`, `transcribe_batch_unprompted` |
| `lifemate/eval/asr_tune.py` | 13 tổ hợp, `--stages`, `--start-option`, `--objective en_mixed` |
| `lifemate/eval/eval_asr.py`, `en_mixed.py` | Chỉ số tiếng Anh, `vi_only_wer`, `prompt_insertions`, `n_prompt_retries` |
| `tests/test_asr_guard.py`, `test_asr_prompts.py`, `test_asr_tune.py`, `test_config.py` | Test |

### 4.5. Chỉ số

Toàn bộ bộ chỉ số ở mục 2.3, trên ba mức: tập dò đầy đủ (chọn cấu hình), val đầy đủ (so với lần 1, phát hiện câu thường
bị hại), tập con câu chêm tiếng Anh (câu hỏi chính). Thêm `n_prompt_retries` và p95 để biết chi phí thời gian của lớp
bảo vệ.

### 4.6. Bằng chứng offline trước khi chạy

Trên chữ của lần 1 (large-v3, tập dò), mô phỏng lớp bảo vệ: câu bị phát hiện hỏng thì thay bằng chữ của lần chạy không
câu mồi. Dấu hiệu dùng: rỗng, bộ lọc có cắt, trùng cụm 3 từ với câu mồi, tỉ lệ nén > 2,4.

| Cấu hình large-v3 (lần 1) | WER | Chép đúng từ ngoại (quy tắc nháp) | Câu phải chép lại |
|---|---|---|---|
| không câu mồi | **30,8%** | 46,9% | 0 |
| câu mồi ngắn | 34,7% | 51,6% | 0 |
| câu mồi ngắn + bảo vệ | 31,4% | 51,9% | 62/1 024 |
| câu mồi lead | 48,6% | 53,9% | 0 |
| câu mồi lead + bảo vệ | 32,7% | **55,7%** | 80/1 024 |

Câu mồi có bảo vệ tăng 5–9 điểm chép đúng từ ngoại, WER chỉ tăng 0,6–1,9 điểm, và phải chép lại 6–8% số câu. Đây là
mô phỏng trên câu mồi cũ, chưa phải câu mồi mới.

### 4.7. Kết quả

_(Chờ lần chạy GPU `artifacts/runs/kaggle/asr/10-29_tune_v5_en_mixed/`: bảng 13 tổ hợp trên tập dò, cấu hình thắng của
mỗi mô hình, số trên val đầy đủ và trên tập câu chêm tiếng Anh, so với thử nghiệm A.)_

## 5. Thử nghiệm C: kiểm chứng chấm offline bằng một lần chạy thật

### 5.1. Vấn đề

Thử nghiệm A và D dựa trên giả định: chạy lại cùng cấu hình cho cùng chữ. Giả định này đúng về lý thuyết (giải mã tham
lam), nhưng có thể lệch vì thư viện khác phiên bản, GPU khác, hoặc batch khác.

### 5.2. Mục tiêu

Xác nhận chữ của một mô hình chạy lại hôm nay khớp chữ đã ghi ở lần 1.

### 5.3. Phương pháp

Chạy lại ChunkFormer RNNT với đúng cấu hình lần 1 trên tập dò và hai tập val, trong cùng kernel với thử nghiệm B. Bước
`prompt` của `asr_tune` không có lựa chọn nào khác "không câu mồi" cho ChunkFormer, nên kernel chỉ chạy một cấu hình.
RNNT được chọn vì nhanh nhất và là mô hình chính.

### 5.4. Chỉ số

Tỉ lệ câu có chữ giống hệt lần 1; chênh lệch WER, bỏ sót và `en_word_recall` giữa hai lần.

### 5.5. Kết quả

_(Chờ lần chạy GPU.)_

## 6. Thử nghiệm D: ghép ChunkFormer RNNT với Whisper

### 6.1. Vấn đề

Thử nghiệm A cho thấy hai mô hình mạnh ở hai phần khác nhau. RNNT viết phần tiếng Việt tốt nhất (WER câu thuần Việt
khoảng 17%) và không bịa câu, nhưng hay nghe từ tiếng Anh thành từ Việt gần âm ("chắc" thay cho "check"). large-v3 chép
tên và từ tiếng Anh tốt hơn nhưng phần tiếng Việt kém (khoảng 29%).

### 6.2. Mục tiêu

Tạo chữ có phần tiếng Việt của RNNT và từ tiếng Anh của Whisper. Đo xem có tốt hơn RNNT đứng riêng không, với cặp
RNNT + large-v3 và cặp RNNT + PhoWhisper.

### 6.3. Phương pháp

**Chung.** RNNT là model chính. Chỉ lấy của model phụ những đoạn mang tới **từ tiếng Anh mới**, còn lại giữ nguyên RNNT.
Không dùng bỏ phiếu theo từ (ROVER), vì bỏ phiếu cần ít nhất ba hệ thống, và cũng không dùng độ tự tin, vì gói
`chunkformer` không trả độ tự tin.

**Thuật toán** (`lifemate/eval/asr_merge.py`):

1. Tách hai câu thành từ, giữ nguyên cách viết để ghép lại. Gióng hàng trên dạng đã gập (chữ thường, bỏ dấu, bỏ dấu câu)
   bằng khoảng cách Levenshtein có truy vết (`align`).
2. Gom các cặp không khớp nằm giữa hai từ khớp thành một **vùng khác nhau**.
3. Ở mỗi vùng: lấy chữ của model phụ khi (a) phía phụ có từ ngoại mà phía chính không có, và (b) vùng phía phụ dài không
   quá `2 × (số từ phía chính) + 2`, để chặn câu bịa dài. Còn lại giữ chữ RNNT.
4. Bỏ hẳn câu của model phụ khi rỗng, hoặc khi bộ lọc ảo giác đã cắt câu đó.

Điều kiện (a) không đòi phía RNNT "không có từ ngoại", vì chữ rác của RNNT ("s tp") cũng không ghép được âm tiết Việt
nên bị coi là từ ngoại. Quy tắc được viết **trước khi** nhìn số và không chỉnh theo val. Tập dò được dùng để kiểm tra lại.

Ví dụ: RNNT "chắc ví dùm còn trên ba triệu", large-v3 "Check ví dùm, còn trên 3 triệu." cho bản ghép "Check ví dùm còn
trên ba triệu": lấy "Check", giữ cách viết số của RNNT.

**Dữ liệu.** Chữ của lần 1. RNNT: cấu hình thắng. large-v3: trên val là cấu hình nhiệt độ thử lại (chỉ để tham khảo,
mục 3.3); trên tập dò là cấu hình **VAD tất định** (lần đo `engine_vad` của lần 1). PhoWhisper: câu mồi ngắn.

### 6.4. Chỉ số

Bộ chỉ số mục 2.3 có CI và chênh lệch so với RNNT, cộng thêm:

- **Số câu bị đổi**, **tốt hơn** và **tệ hơn** RNNT (theo số lỗi từ của câu).
- **Oracle**: WER khi mỗi câu chọn bản tốt hơn trong hai bản. Đây là trần của mọi phép ghép chọn nguyên câu.
- Chi phí: tổng tham số và tổng thời gian của hai mô hình.

### 6.5. Kết quả

Chênh lệch so với RNNT đứng riêng (điểm %, CI 95% của chênh lệch):

| Cặp | Tập | Chép đúng từ ngoại | WER | WER câu chêm | WER câu thuần Việt |
|---|---|---|---|---|---|
| RNNT + large-v3 | val_seen | **+8,7** [+2,5; +14,4] | −0,1 [−0,4; +0,3] | −0,8 [−1,9; +0,3] | +0,3 [+0,1; +0,5] |
| | val_unseen | **+9,9** [+4,4; +15,3] | −0,2 [−0,6; +0,2] | −1,1 [−2,2; +0,1] | +0,3 [+0,1; +0,5] |
| | tập dò (large-v3 VAD) | **+8,1** [+3,1; +12,7] | −0,1 [−0,5; +0,2] | −1,1 | +0,4 [+0,2; +0,6] |
| RNNT + PhoWhisper | val_seen | −0,6 [−5,2; +4,4] | +0,1 [−0,1; +0,4] | +0,4 | 0,0 |
| | val_unseen | −2,0 [−7,3; +3,2] | +0,2 [0,0; +0,5] | +0,6 | 0,0 |
| | tập dò | −2,3 [−6,3; +1,4] | +0,2 [0,0; +0,5] | +0,7 | 0,0 |

Giá trị tuyệt đối trên tập câu chêm tiếng Anh: RNNT + large-v3 chép đúng **58,5%** (seen) và **61,5%** (unseen) từ
ngoại, so với 49,9% và 51,6% của RNNT; bản nới lỏng 62,1% và 63,1%. WER câu chêm 24,0% và 23,0%.

Số câu bị đổi:

| Cặp | Tập | Câu | Bị đổi | Tốt hơn | Tệ hơn | WER RNNT | WER ghép | Oracle |
|---|---|---|---|---|---|---|---|---|
| RNNT + large-v3 | val_seen | 916 | 117 | 48 | 49 | 20,0 | 19,9 | 17,6 |
| | val_seen, câu chêm | 245 | 99 | 47 | 35 | 24,8 | 24,0 | 21,6 |
| | val_unseen | 527 | 76 | 41 | 21 | 19,8 | 19,6 | 18,0 |
| | val_unseen, câu chêm | 162 | 67 | 40 | 14 | 24,1 | 23,0 | 21,7 |
| RNNT + PhoWhisper | val_seen | 916 | 91 | 28 | 34 | 20,0 | 20,2 | 18,2 |
| | val_unseen | 527 | 59 | 16 | 29 | 19,8 | 20,0 | 18,3 |

"Tốt hơn" và "tệ hơn" đếm theo số lỗi từ của cả câu; một câu có thể tốt hơn ở từ tiếng Anh nhưng tệ hơn ở chỗ khác.

Ví dụ (val_unseen):

| | Chữ |
|---|---|
| Đáp án | …có chuyến **RV219** tám giờ bay vào Cần Thơ… |
| RNNT | …có chuyến **và vê hai một chín** tám giờ bay vào cần thơ… |
| Ghép | …có chuyến **RV219** 8 giờ bay vào cần thơ… |
| Đáp án | Cuộc họp với **Mr. Fujimoto** hôm đó mấy giờ hè? |
| RNNT | cuộc họp với **micro full mô tô** hôm đó mấy giờ nè |
| Ghép | cuộc họp với **Mr. Funmoto** hôm đó mấy giờ nè |
| Đáp án (ca hỏng) | Mai không mưa thì giữ cho anh cuốn Mùa đông ở Lisbon… |
| RNNT | mai không mưa thì giữ cho anh cuốn mùa đông ở disban… |
| large-v3 | **Mời các bạn thử chọn những quần bộ đông nhất của** Lisbon… |
| Ghép | **Mời các bạn thử chọn những quần bộ đông nhất của** Lisbon… |

### 6.6. Phân tích

- **RNNT + large-v3 tăng chép đúng từ tiếng Anh khoảng 8–10 điểm, có ý nghĩa thống kê trên cả ba tập** (CI không chứa
  0), trong khi WER tổng không đổi. Kết quả trên tập dò (large-v3 VAD, tất định) khớp với val (large-v3 nhiệt độ thử
  lại), nên lợi ích không phụ thuộc phần ngẫu nhiên của cấu hình cũ.
- **Cái giá là phần tiếng Việt kém đi một chút**: WER câu thuần Việt tăng 0,3–0,4 điểm, CI không chứa 0. Nguyên nhân:
  large-v3 bịa từ ngoại trong câu thuần Việt ("Toyota D-Diều" cho "tôi ra đê"), và quy tắc lấy cả vùng. Ca hỏng ở bảng
  ví dụ cho thấy giới hạn độ dài vùng chưa chặn hết câu bịa: vùng bịa dài 10 từ được lấy vì có "Lisbon" đúng.
- **RNNT + PhoWhisper không giúp**: chép đúng từ tiếng Anh không tăng (CI chứa 0, nghiêng về âm), WER nhích lên. Khớp với
  thử nghiệm A: PhoWhisper chép từ tiếng Anh kém hơn chính RNNT.
- **Oracle cho thấy còn nhiều chỗ**: WER câu chêm có thể xuống khoảng 21,6–21,7% nếu chọn đúng câu, so với 23–24% của bản
  ghép hiện tại.
- **Chi phí.** Tham số: RNNT 0,11B + large-v3 1,55B + LLM 4,02B = 5,68B, dưới trần 6B nhưng sát. Thời gian: cộng hai mô
  hình, khoảng 2,5 s + 6,8 s p95 trên T4 (VAD), ăn vào 30 s mỗi mẫu cùng với LLM. Chưa đo trên A30 và chưa đo khi chạy cạnh
  vLLM.

### 6.7. Kết luận (tạm)

Ghép RNNT + large-v3 là hướng có lợi rõ nhất cho từ tiếng Anh trong lần 2, nhưng chưa sẵn sàng để nộp:

1. Cần chạy lại với chữ large-v3 của cấu hình thắng ở thử nghiệm B (tất định, có thể có câu mồi) trên val.
2. Cần siết quy tắc để giảm hại ở câu thuần Việt (ví dụ giới hạn độ dài vùng chặt hơn, hoặc bỏ vùng khi model phụ bịa câu).
   Mọi chỉnh sửa phải chọn trên **tập dò**, không chọn trên val.
3. Cần đo thời gian và bộ nhớ của cả hai mô hình chạy cạnh vLLM trên GPU từ 24 GB.

Code ghép hiện chỉ có ở dạng chấm offline (`asr_merge`). Chưa có engine ASR ghép trong `lifemate/backends/asr.py`; việc
này để sau khi có kết quả thử nghiệm B.

## 7. Tổng hợp và kết luận

_(Phần offline. Cập nhật sau khi có kết quả GPU.)_

1. **Mô hình đơn tốt nhất trên câu chêm tiếng Anh vẫn là ChunkFormer RNNT**: WER câu chêm khoảng 24%, chép đúng khoảng
   50% từ ngoại.
2. **large-v3 chép từ tiếng Anh ngang hoặc hơn RNNT** nhưng phần tiếng Việt kém xa. **PhoWhisper và ChunkFormer CTC chép
   từ tiếng Anh kém rõ rệt.**
3. **Ghép RNNT + large-v3 tăng chép đúng từ tiếng Anh khoảng 9 điểm có ý nghĩa thống kê**, WER tổng không đổi, đổi lại
   phần tiếng Việt kém đi khoảng 0,3 điểm. Đây là ứng viên mạnh nhất cho câu chêm tiếng Anh, cần thêm thử nghiệm B và đo
   thời gian trước khi chọn.
4. Chưa đổi `configs/v0.toml`.

## 8. Rà code trước khi chạy: lỗi đã vá

Trước khi chạy GPU, code được rà lại theo câu hỏi "đo như vậy đã đúng chưa". Các lỗi tìm thấy và cách sửa:

| Lỗi | Hậu quả | Sửa |
|---|---|---|
| Dò câu mồi rồi mới dò lớp bảo vệ (lần lượt) | Câu mồi chỉ tốt khi có bảo vệ bị loại oan | Gộp thành một bước 13 tổ hợp (mục 4.3.3) |
| Câu mồi nháp dựa theo mẫu câu thật ("nhắn SMS cho") | Lớp bảo vệ báo nhầm 2,1–2,6% câu đúng | Viết lại câu mồi, thêm test trên 5 036 câu chuẩn |
| Câu mồi gán cứng trong code | Không có phiên bản, khó so và khó đổi | File `configs/asr_prompts/*_v1.txt` và khoá `asr.initial_prompt_file` |
| "Chép đúng" đếm có/không | Đáp án có "check" hai lần mà ASR chỉ viết một lần vẫn tính đúng cả hai | Đếm theo số lần xuất hiện (`en_mixed.hits`) |
| Vị trí "đầu câu" tính trên chuỗi gốc trong khi tìm từ trên chuỗi đã chuẩn hoá NFC | Lệch vị trí với chuỗi dạng NFD, xếp nhóm sai | Tính cả hai trên cùng chuỗi đã chuẩn hoá |
| Không có WER câu thuần Việt | Không biết câu mồi hay phép ghép có hại câu thường | Thêm `vi_only_wer` |
| "TV" cho "tivi", "Minh Mart" cho "MinhMart" bị tính sai | Phạt lỗi cách viết mà LLM vẫn hiểu | Thêm bản nới lỏng, báo cáo cả hai |
| Bootstrap chấm lại cả tập mỗi vòng | Quá chậm (hàng chục phút) | Chấm từng mẫu một lần lấy tử và mẫu (`AsrReport.counts`), bootstrap chỉ cộng; 1 000 vòng trong vài giây |
| Nhiệt độ thử lại trong cấu hình large-v3 | Không tất định, trái thể lệ | Bỏ khỏi `asr_tune`, dùng VAD |

**Về cache vLLM.** Không tìm thấy chỉ dẫn "phải xoá cache vLLM" trong repo, trong docs hay trong lịch sử các phiên
trước. Có một điểm liên quan: `VLLMBackend` (`lifemate/backends/llm.py`) luôn bật `enable_prefix_caching = True`, kể cả
khi bật `batch_invariant`. Hướng dẫn batch invariance của vLLM bản cho Ascend ghi rằng prefix caching không được hỗ trợ
cùng batch invariance; tài liệu chính của vLLM không nói gì về điểm này. Việc này thuộc phần LLM, ngoài phạm vi benchmark
ASR; cần đo trên A30 (hai lần chạy giống từng byte, SPEC mục 6) trước khi kết luận.

## Giới hạn đã biết

- Thời gian đo trên T4 chạy ASR riêng, chưa phải A30 và chưa chạy cạnh vLLM.
- Mỗi nhóm từ chỉ có vài chục lần xuất hiện (từ mượn chỉ 15 và 2), nên chênh vài từ đã là vài điểm phần trăm; dùng CI ở
  mục 3.5 và 6.5 thay vì so số trần.
- Quy tắc tìm từ tiếng Anh bỏ sót từ trông như âm tiết Việt; nhóm của từ viết hoa đầu câu phụ thuộc tập đang chấm.
- Chép đúng từ ngoại so theo từ có mặt trong câu, không kiểm tra vị trí, nên có thể khớp nhầm.
- Số của large-v3 ở thử nghiệm A và ở val của thử nghiệm D đến từ cấu hình nhiệt độ thử lại (không tất định).
- Câu mồi chọn từ theo chữ large-v3 beam 5 trong `data/asr/train_asr.json`, không phải cấu hình tham lam + VAD đang dò.
