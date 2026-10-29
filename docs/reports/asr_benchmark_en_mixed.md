# Báo cáo benchmark ASR lần 2: câu tiếng Việt chêm tiếng Anh

Ngày: 29/10/2026. Trạng thái: **xong**. Cả bốn thử nghiệm đã có kết quả: A và D chấm offline; B và C chạy GPU trên
Kaggle T4 ở hai phiên v5a (PhoWhisper, ChunkFormer RNNT) và v5b (whisper-large-v3). Tổng hợp ở mục 7.

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
| `artifacts/runs/kaggle/asr/10-29_tune_v5a_en_mixed/` | Thử nghiệm B và C, phiên v5a (PhoWhisper, ChunkFormer RNNT), cùng cấu trúc file với lần 1; `compare/` là bảng so sánh có khoảng tin cậy |
| `artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/` | Thử nghiệm B và C, phiên v5b (whisper-large-v3), cùng cấu trúc và có `compare/` |
| `artifacts/runs/local/asr/10-29_en_mixed_merge_v3/` | Thử nghiệm D với large-v3 cấu hình thắng (mục 6.5.2): chữ đã ghép, `cmp_*.json`, `merge_effect.json` |

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
| ChunkFormer RNNT | seen | 49,9 [44,3; 55,3] | 52,5 | **24,8** [22,8; 26,7] | **16,9** | 2,5 |
| | unseen | 51,6 [44,8; 58,6] | 52,4 | **24,1** [21,6; 26,8] | **17,0** | 2,5 |
| ChunkFormer CTC | seen | 34,6 [29,8; 39,7] | 41,5 | 27,5 | 17,5 | 2,7 |
| | unseen | 39,7 [32,2; 47,4] | 40,5 | 26,9 | 17,0 | 2,3 |
| PhoWhisper (câu mồi ngắn) | seen | 36,1 [30,6; 41,9] | 39,7 | 32,5 | 24,7 | 3,0 |
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

Lần chạy GPU: `artifacts/runs/kaggle/asr/10-29_tune_v5a_en_mixed/` (PhoWhisper, phiên v5a, T4, 2 giờ 15 phút, không lỗi).
large-v3: `artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/` (phiên v5b), mục 4.7.6–4.7.9.

#### 4.7.1. PhoWhisper: 13 tổ hợp trên tập dò

Tập dò đầy đủ (300 mẫu, 1 024 câu, 394 từ ngoại). Số `prompt_insertions` đã chấm lại sau khi sửa cách đếm (mục 8).

| Lựa chọn | Chép đúng từ ngoại | Bản nới lỏng | Chèn từ câu mồi | Chép lại | WER | WER câu chêm | WER câu thuần Việt | p95 (s) |
|---|---|---|---|---|---|---|---|---|
| không câu mồi | 31,0 | 34,5 | 0 | 0 | 28,11 | 34,34 | **24,96** | 3,71 |
| short | 33,0 | 37,1 | 0 | 0 | 28,14 | 33,88 | 25,23 | 2,94 |
| short + g3 | 32,5 | 36,5 | 0 | 3 | 28,19 | 34,03 | 25,23 | 2,94 |
| short + g4 | 33,0 | 37,1 | 0 | 0 | 28,14 | 33,88 | 25,23 | 3,04 |
| en_top10 (± g3, g4) | 35,5 | 39,1 | 0 | 0 | **27,82** | 33,28 | 25,06 | 3,0 |
| **en_top20** (± g3, g4) | **37,6** | **40,4** | 3 | 0 | 27,95 | **33,09** | 25,35 | 3,0 |
| en_full (± g3, g4) | 34,5 | 38,3 | 4 | 0 | 28,35 | 33,82 | 25,59 | 3,1 |

Ba mức bảo vệ của cùng một câu mồi dài cho chữ giống hệt nhau, nên gộp một dòng. Cấu hình thắng theo hàm mục tiêu
`en_mixed`: **en_top20, không bảo vệ** (hoà với g3, g4 thì giữ lựa chọn đứng trước). Chấm lại với cách đếm chèn đã sửa
vẫn chọn en_top20.

Chênh lệch so với không câu mồi trên tập con câu chêm của tập dò (165 mẫu), CI 95%:

| Câu mồi | Chép đúng từ ngoại | WER | WER câu thuần Việt |
|---|---|---|---|
| en_top10 | +4,6 [+1,8; +7,1] | −0,5 [−1,0; 0,0] | 0,0 [−0,5; +0,5] |
| en_top20 | **+6,6** [+3,7; +9,7] | −0,5 [−1,0; 0,0] | +0,2 [−0,4; +0,8] |
| en_full | +3,5 [+0,2; +6,4] | +0,2 [−0,4; +0,7] | **+0,8** [+0,2; +1,4] |

#### 4.7.2. PhoWhisper: cấu hình thắng trên val

So với PhoWhisper của lần 1 (câu mồi ngắn, cùng cách giải mã), tập con câu chêm tiếng Anh, CI 95% của chênh lệch:

| Tập | Chép đúng từ ngoại: lần 1 → en_top20 | Chênh lệch | WER: chênh lệch | WER câu thuần Việt: chênh lệch | Chèn từ câu mồi |
|---|---|---|---|---|---|
| val_seen (245 câu chêm) | 36,1 → 37,0 | +0,9 [−1,9; +3,6] | +0,2 [−0,3; +0,9] | +0,1 [−0,7; +1,0] | 4 |
| val_unseen (162 câu chêm) | 36,5 → 36,5 | 0,0 [−2,9; +2,8] | 0,0 [−0,7; +0,7] | 0,0 [−0,7; +0,8] | 0 |

Trên val đầy đủ: WER 28,1% (seen) và 28,0% (unseen), lần 1 là 27,9% và 27,9%. So với ChunkFormer RNNT, PhoWhisper
en_top20 vẫn kém rõ ở mọi chỉ số: chép đúng từ ngoại thấp hơn 12,9 (seen) và 15,1 (unseen) điểm, CI không chứa 0.

Tách theo từ **có** trong en_top20 và từ **không có** (số từ chép đúng / tổng; mốc là câu mồi ngắn):

| Tập | Từ có trong câu mồi: mốc → en_top20 | Từ không có: mốc → en_top20 |
|---|---|---|
| tập dò | 56 → **76** / 161 | 74 → 72 / 233 |
| val_seen | 55 → 58 / 133 | 66 → 66 / 202 |
| val_unseen | 20 → 21 / 45 | 72 → 71 / 207 |

Theo từng từ (mốc → en_top20 / tổng): trên tập dò, tăng chủ yếu ở tivi (2 → 8 / 14), robot (8 → 12 / 13), gara (5 → 7 / 7),
SMS (5 → 7 / 18). Trên val_seen, chính các từ đó không đổi: tivi 5 → 5 / 14, robot 9 → 9 / 14. **SJC (0 / 26 và 0 / 22)
và check (0 / 8 và 0 / 5) không bao giờ được chép đúng**, có câu mồi hay không.

#### 4.7.3. Lớp bảo vệ câu mồi trên PhoWhisper

- Với ba câu mồi mới, **lớp bảo vệ không kích hoạt lần nào** (0 câu chép lại ở cả g3 và g4): PhoWhisper không chép lại
  câu mồi, không liệt kê, không bịa câu (`n_hallucinated = 0` ở mọi lần đo).
- Với câu mồi ngắn của lần 1 ("Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."), g3 chép lại 3 câu, **cả 3 đều là báo nhầm**:
  người dùng nói đúng cụm "nhắn Zalo cho". Chép lại không câu mồi còn làm hỏng câu: 2/3 câu "Zalo" thành "giao lưu".
  Đúng rủi ro đã nêu ở mục 4.3.1 (câu mồi giống lời nói thật thì lớp bảo vệ báo nhầm); g4 không báo nhầm.
- `prompt_insertions` thật của en_top20 trên tập dò là 3 câu trên 1 024: "Lofi" thành "file", "lít" thành "list",
  "VTV1" thành "vê tivi một". Bản đếm cũ ra 6 vì tính "ti vi" và "TV" của đáp án là khác "tivi" (mục 8).

#### 4.7.4. Phân tích PhoWhisper

- **Câu mồi có giúp trên tập dò nhưng không giúp trên val.** Trên tập dò, en_top20 tăng 6,6 điểm (CI không chứa 0). Trên
  val, chênh lệch so với lần 1 là +0,9 và 0,0 điểm, CI chứa 0 và rộng chưa tới ±3,6 điểm. Đây là lý do bắt buộc phải đo
  cấu hình thắng một lần trên val: số của tập dò đã bị chọn lọc (13 lựa chọn, chọn cái tốt nhất), nên luôn lạc quan.
- **Lợi ích trên tập dò dồn vào vài từ của vài người nói**: tivi, robot, gara chiếm phần lớn 20 từ tăng thêm. Trên val,
  cùng các từ đó không đổi. Mẫu tập dò ít (161 lần xuất hiện của từ có trong câu mồi), nên vài người nói cùng một cách
  phát âm đủ tạo ra chênh lệch.
- **Câu mồi không dạy được từ mà mô hình không nghe ra**: SJC và check luôn sai. PhoWhisper được huấn luyện thêm trên
  tiếng Việt, nên với các âm này nó chọn chữ Việt gần âm, câu mồi không kéo lại được.
- **Câu mồi dài hơn không tốt hơn**: en_full kém en_top20 ở mọi chỉ số và là câu mồi duy nhất làm hại câu thuần Việt có ý
  nghĩa thống kê (+0,8 điểm, CI [+0,2; +1,4]).
- **Câu thuần Việt không bị hại** ở en_top10 và en_top20 (CI chứa 0), và không có câu bịa, nên câu mồi an toàn với
  PhoWhisper, chỉ là không có ích trên val.

#### 4.7.5. Kết luận PhoWhisper

Câu mồi không cứu được PhoWhisper trên câu chêm tiếng Anh: sau khi dò, PhoWhisper vẫn kém ChunkFormer RNNT khoảng 13–15
điểm chép đúng từ ngoại và khoảng 8 điểm WER. Giữ kết luận của thử nghiệm A: PhoWhisper không phải ứng viên cho câu chêm
tiếng Anh, kể cả làm mô hình phụ khi ghép (thử nghiệm D). Không cần chạy lại phép ghép RNNT + PhoWhisper với cấu hình
mới, vì chữ của PhoWhisper trên val gần như không đổi.

#### 4.7.6. large-v3: 13 tổ hợp trên tập dò

Lần chạy GPU: `artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/` (phiên v5b, T4, 3 giờ 7 phút, không lỗi). Mọi cấu hình
đều bật VAD và giải mã tham lam, nên tất định. p95 của mọi lựa chọn vượt ngưỡng 6 s trên T4 nên phần đầu hàm mục tiêu
bằng nhau, các lựa chọn so với nhau ở phần sau.

| Lựa chọn | Chép đúng từ ngoại | Bản nới lỏng | Chèn từ câu mồi | Chép lại | WER | WER câu chêm | WER câu thuần Việt | Câu bịa | Bỏ sót | p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| không câu mồi | 47,7 | 52,0 | 0 | 0 | 30,83 | 32,72 | 29,87 | 8 | 51 | 8,41 |
| short | 51,8 | 56,1 | 46 | 0 | 35,12 | 35,86 | 34,74 | 21 | 51 | 6,76 |
| short + g3 | 51,8 | 56,3 | 5 | 67 | 30,93 | 32,61 | 30,07 | 5 | 49 | 7,08 |
| short + g4 | 52,5 | 57,1 | 6 | 63 | 30,98 | 32,49 | 30,22 | 5 | 49 | 6,99 |
| en_top10 | 62,4 | 67,0 | 28 | 0 | 30,35 | 31,66 | 29,68 | 5 | 44 | 6,91 |
| en_top10 + g3 / g4 | 62,7 | 67,3 | 18 | 7 | 30,15 | 31,22 | 29,61 | 3 | 44 | 6,9 |
| en_top20 | 66,0 | 70,3 | 34 | 0 | 29,93 | 31,18 | 29,30 | 2 | 46 | 7,06 |
| en_top20 + g3 / g4 | 66,0 | 70,3 | 29 | 4 | 29,88 | 31,18 | **29,23** | 2 | 46 | 7,08 |
| en_full | **70,6** | **74,9** | 26 | 0 | 29,79 | **30,03** | 29,67 | 2 | 42 | 7,30 |
| en_full + g3 | 70,6 | 74,9 | 26 | 3 | 29,70 | 30,10 | 29,50 | 1 | 42 | 7,32 |
| **en_full + g4** | **70,6** | **74,9** | 26 | 2 | **29,68** | **30,03** | 29,50 | **1** | **42** | 7,31 |

Cấu hình thắng: **en_full + g4** (câu mồi dài nhất, lớp bảo vệ cụm 4 từ). Chấm lại với cách đếm chèn đã sửa vẫn chọn
en_full + g4. `prompt_insertions` có mốc: chép không câu mồi, large-v3 tự viết các từ của en_full mà không được nói 3 lần
trên 1 024 câu; với en_full + g4 là 26 lần. Các lần chèn thật hay gặp: "lịch" thành "list" (5), "Mai" thành "Mail"
(5), "ok" chen vào đầu câu (3), "robot" thay cho từ Việt gần âm (3).

#### 4.7.7. large-v3: cấu hình thắng trên val

Tập con câu chêm tiếng Anh. Mốc "lần 1" là cấu hình nhiệt độ thử lại (không tất định, chỉ tham khảo). CI 95%.

| Tập | Hệ thống | Chép đúng từ ngoại | WER câu chêm | WER câu thuần Việt | Câu bịa | Chèn từ câu mồi |
|---|---|---|---|---|---|---|
| val_seen | ChunkFormer RNNT | 49,9 [44,3; 55,3] | **24,8** | **16,9** | 0 | 1 |
| | large-v3 lần 1 | 51,6 [46,4; 56,7] | 32,8 | 29,2 | 8 | 0 |
| | large-v3 en_full + g4 | **72,2** [67,2; 77,3] | 28,7 | 28,8 | 1 | 16 |
| val_unseen | ChunkFormer RNNT | 51,6 [44,8; 58,6] | **24,1** | **17,0** | 0 | 3 |
| | large-v3 lần 1 | 56,0 [48,1; 63,4] | 33,9 | 29,1 | 3 | 0 |
| | large-v3 en_full + g4 | **61,1** [52,7; 69,3] | 33,5 | 30,0 | 0 | 4 |

Chênh lệch của large-v3 en_full + g4 so với RNNT: chép đúng từ ngoại **+22,3** [+16,5; +28,6] (seen) và **+9,5**
[+2,9; +15,1] (unseen); WER câu chêm +3,9 [+1,9; +5,9] và +9,4 [+6,7; +12,3]; WER câu thuần Việt khoảng +12 điểm ở cả
hai tập. Val đầy đủ: WER 29,0% (seen) và 30,2% (unseen); p95 8,8 s và 8,6 s, max 10,8 s trên T4; lớp bảo vệ chép lại 1
và 0 câu.

Tách theo từ có và không có trong en_full (số từ chép đúng / tổng):

| Tập | Hệ thống | Từ có trong câu mồi | Từ không có |
|---|---|---|---|
| val_seen | RNNT | 111 / 184 | 56 / 151 |
| | large-v3 lần 1 | 89 / 184 | 84 / 151 |
| | large-v3 en_full + g4 | **155** / 184 | 87 / 151 |
| val_unseen | RNNT | 43 / 62 | 87 / 190 |
| | large-v3 lần 1 | 35 / 62 | 106 / 190 |
| | large-v3 en_full + g4 | **51** / 62 | 103 / 190 |

Theo nhóm từ, val_seen (large-v3 lần 1 → en_full + g4): từ thường 61 → 82 / 105, viết tắt 43 → 70 / 82, tên 67 → 84 /
133.

#### 4.7.8. Phân tích large-v3

- **Câu mồi có tác dụng thật với large-v3, khác hẳn PhoWhisper.** Từ có trong câu mồi được chép đúng từ 48% lên 84%
  (seen) và từ 56% lên 82% (unseen), trên val chưa dùng để dò. Từ không có trong câu mồi không đổi (84 → 87, 106 → 103).
  Vậy câu mồi không làm large-v3 "giỏi tiếng Anh hơn" mà chỉ dạy nó **cách viết** của đúng những từ được đưa vào: đúng
  cơ chế đã nêu ở mục 4.3.1.
- **Lợi ích phụ thuộc độ phủ của câu mồi.** Ở val_seen, 55% số từ ngoại có trong câu mồi, nên tăng 20,6 điểm. Ở
  val_unseen (app chưa gặp), chỉ 25% có trong câu mồi, nên chỉ tăng 5,1 điểm. Bộ thi riêng có app lạ, nên số thật nằm
  giữa hai mức, gần unseen hơn nếu app lạ chiếm nhiều.
- **Câu mồi dài hơn tốt hơn với large-v3** (en_full > en_top20 > en_top10 > short), ngược với PhoWhisper. Câu mồi dài
  còn **giảm** câu bịa và câu rỗng (8 → 1) và giảm bỏ sót giá trị tham số (51 → 42 trên 123): chữ đi trước dài và ổn định
  giúp decoder không lạc.
- **Lớp bảo vệ cần cho câu mồi ngắn, gần như không cần cho câu mồi dài.** Câu mồi ngắn không bảo vệ làm WER tăng 4,3
  điểm, 21 câu bịa; bảo vệ chép lại 63–67 câu và kéo WER về mức không câu mồi. Với en_full, bảo vệ chỉ chép lại 2–3 câu.
  g4 không thua g3 ở lựa chọn nào, nên dùng g4.
- **Cái giá là chèn từ câu mồi**: 26 lần trên 1 024 câu ở tập dò (mốc không câu mồi: 3), 16 lần ở val_seen. "Mai" thành
  "Mail" và "lịch" thành "list" có thể làm LLM gọi sai tool, nên đây là rủi ro chính khi đưa vào hệ thống.
- **large-v3 đứng riêng vẫn không thay được RNNT**: phần tiếng Việt kém khoảng 12 điểm WER, và p95 8,6–8,8 s trên T4.

#### 4.7.9. Kết luận thử nghiệm B

- large-v3: dùng **en_full_v1 + lớp bảo vệ 4 từ, VAD, giải mã tham lam** làm cấu hình của large-v3 cho lần 2. Đây là
  nguồn chữ tiếng Anh tốt nhất hiện có, dùng làm mô hình phụ trong thử nghiệm D.
- PhoWhisper: câu mồi không giúp trên val (mục 4.7.5).
- Câu mồi giúp theo độ phủ từ; muốn tăng trên app lạ thì phải đưa thêm từ vào câu mồi, nhưng câu mồi đã gần trần 223
  token (183 token), và từ thêm vào cũng là nguồn chèn mới. Thêm từ phải chọn trên dữ liệu ngoài tập dò và val.

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

**ChunkFormer RNNT.** Chạy lại trong phiên v5a (T4, thư viện cài mới), so từng câu với `hyps.jsonl` của lần 1
(`10-28_tune_v3`):

| Tập | Câu | Giống hệt lần 1 |
|---|---|---|
| tập dò | 1 024 | 1 024 (100%) |
| val_seen | 916 | 916 (100%) |
| val_unseen | 527 | 527 (100%) |

Mọi chỉ số vì vậy trùng khớp: WER tập dò 20,29%, bỏ sót 46/123, chép đúng từ ngoại 49,0%.

**large-v3 VAD.** Lựa chọn "không câu mồi" của phiên v5b có config giống hệt lần đo `engine_vad` của lần 1, và cho chữ giống
hệt ở **1 024/1 024 câu** của tập dò.

### 5.6. Kết luận

Giả định của thử nghiệm A và D đúng với cả ChunkFormer RNNT và large-v3 VAD: chạy lại trên một máy T4 khác, thư viện cài
mới, cho đúng từng ký tự. Số chấm offline dùng được như số chạy thật, và large-v3 VAD đạt yêu cầu tất định của thể lệ.
PhoWhisper lần 1 và lần 2 không cùng cấu hình nên không so từng ký tự được.

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

#### 6.5.1. Với chữ Whisper của lần 1

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

#### 6.5.2. Với large-v3 cấu hình thắng của thử nghiệm B (en_full + g4)

Chạy lại `asr_merge` trên chữ thật của phiên v5b, cùng quy tắc, không chỉnh gì. Kết quả ở
`artifacts/runs/local/asr/10-29_en_mixed_merge_v3/`. Chênh lệch so với RNNT (điểm %, CI 95% của chênh lệch):

| Tập | Chép đúng từ ngoại | WER | WER câu chêm | WER câu thuần Việt | Bỏ sót giá trị tham số |
|---|---|---|---|---|---|
| val_seen, câu chêm | **+22,7** [+17,0; +28,9] | −0,4 [−1,0; +0,2] | **−2,0** [−3,1; −1,0] | +1,1 [+0,6; +1,7] | +1,5 [−6,8; +8,7] |
| val_unseen, câu chêm | **+12,3** [+6,4; +17,5] | −0,4 [−1,0; +0,2] | **−1,3** [−2,2; −0,3] | +0,6 [+0,1; +1,2] | −9,4 [−20,6; 0,0] |
| val_seen đầy đủ | +22,7 [+16,8; +28,2] | +0,1 [−0,4; +0,6] | | +1,1 [+0,6; +1,5] | +0,9 [−4,2; +5,7] |
| val_unseen đầy đủ | +12,3 [+7,0; +17,6] | 0,0 [−0,4; +0,4] | | +0,6 [+0,3; +1,0] | −5,6 [−13,0; 0,0] |
| tập dò, câu chêm | +23,4 [+17,3; +29,0] | −0,2 [−0,9; +0,5] | −1,8 [−2,9; −0,8] | +1,3 [+0,6; +2,0] | −4,1 [−9,5; 0,0] |

Giá trị tuyệt đối, câu chêm tiếng Anh: chép đúng từ ngoại **72,5%** (seen) và **63,9%** (unseen), so với RNNT 49,9% và
51,6%, và bản ghép với large-v3 lần 1 58,5% và 61,5%. WER câu chêm 22,8% ở cả hai tập (RNNT 24,8% và 24,1%).

Số câu bị đổi, tách câu chêm và câu thuần Việt:

| Tập | Câu | Bị đổi | Tốt hơn | Tệ hơn | WER RNNT | WER ghép | Oracle |
|---|---|---|---|---|---|---|---|
| val_seen, câu chêm | 245 | 130 | 78 | 33 | 24,8 | 22,8 | 20,7 |
| val_seen, câu thuần Việt | 671 | 44 | **1** | **37** | 17,7 | 18,8 | 15,7 |
| val_unseen, câu chêm | 162 | 70 | 37 | 14 | 24,1 | 22,8 | 21,7 |
| val_unseen, câu thuần Việt | 365 | 20 | **1** | **17** | 17,5 | 18,2 | 16,2 |
| tập dò, câu chêm | 276 | 138 | 74 | 36 | 25,8 | 24,0 | 21,7 |
| tập dò, câu thuần Việt | 748 | 58 | **1** | **48** | 17,5 | 18,8 | 16,0 |

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
- **Với large-v3 có câu mồi (6.5.2), lợi ích tăng gần gấp ba trên seen và tăng thêm trên unseen**: +22,7 và +12,3 điểm
  chép đúng từ ngoại, và WER câu chêm giảm có ý nghĩa thống kê (CI không chứa 0) ở cả hai tập, điều bản ghép cũ chưa đạt.
  Bản ghép giữ gần trọn phần tiếng Anh của large-v3 (72,5% so với 72,2% của large-v3 đứng riêng trên seen, 63,9% so với
  61,1% trên unseen) mà WER chỉ bằng RNNT.
- **Nhưng câu thuần Việt bị hại rõ hơn**: +1,1 (seen) và +0,6 (unseen) điểm WER, CI không chứa 0. Bảng số câu bị đổi
  chỉ ra nguyên nhân: ở câu thuần Việt, mọi câu bị đổi gần như đều tệ hơn (37/44, 17/20, 48/58; mỗi tập chỉ 1 câu tốt
  hơn). Đây chính là các lần chèn từ câu mồi của large-v3 (mục 4.7.6: "Mai" thành "Mail", "lịch" thành "list"): từ chèn
  là "từ tiếng Anh mới" nên quy tắc ghép lấy vào. Câu mồi làm tăng loại lỗi này, nên hại câu thuần Việt tăng từ 0,3 lên
  0,6–1,1 điểm.
- **Thứ hạng giữ trên tập dò**: tập dò cho cùng hướng và cùng độ lớn (+23,4 chép đúng, +1,3 câu thuần Việt), nên mọi
  chỉnh sửa quy tắc chọn trên tập dò sẽ chuyển được sang val.

### 6.7. Kết luận

Ghép RNNT + large-v3 (en_full + g4) là cấu hình ASR tốt nhất cho câu chêm tiếng Anh của lần 2: chép đúng từ ngoại
72,5% / 63,9% (seen / unseen) so với 49,9% / 51,6% của RNNT, WER câu chêm thấp hơn RNNT, WER tổng không đổi. Chưa sẵn
sàng để nộp vì ba việc:

1. **Giảm hại ở câu thuần Việt** (+0,6–1,1 điểm WER, do từ câu mồi chèn vào). Hướng xử lý: không lấy vùng khi từ tiếng
   Anh mới là từ của câu mồi và phía RNNT là âm tiết Việt có dấu hợp lệ, hoặc khi vùng chỉ có một từ ngoại đứng đầu câu
   ("Ok, …"). Quy tắc phải chọn trên **tập dò** (câu thuần Việt của tập dò có 48/58 câu bị hại, đủ để đo), sau đó đo
   một lần trên val. Đây là việc kế tiếp, chưa làm.
2. **Đo thời gian và bộ nhớ** khi chạy cả hai mô hình cạnh vLLM trên GPU từ 24 GB. Trên T4, large-v3 có p95 8,6–8,8 s
   mỗi câu.
3. **Viết engine ASR ghép** trong `lifemate/backends/asr.py`. Hiện phép ghép chỉ có ở dạng chấm offline (`asr_merge`).

Tổng tham số: RNNT 0,11B + large-v3 1,55B + LLM 4,02B = 5,68B, dưới trần 6B.

## 7. Tổng hợp và kết luận

Đã đủ bốn thử nghiệm A, B, C, D.

1. **Mô hình đơn tốt nhất trên câu chêm tiếng Anh vẫn là ChunkFormer RNNT** (WER câu chêm khoảng 24%, chép đúng khoảng
   50% từ ngoại). Không mô hình Whisper nào thay được RNNT vì phần tiếng Việt kém khoảng 8–12 điểm WER.
2. **Câu mồi tiếng Anh có tác dụng với large-v3, không có tác dụng với PhoWhisper.** large-v3 với en_full_v1 + lớp bảo vệ
   4 từ chép đúng 72,2% (seen) và 61,1% (unseen) từ ngoại, so với 51,6% và 56,0% không có câu mồi phù hợp. Lợi ích dồn
   vào các từ có trong câu mồi, nên trên app lạ thấp hơn nhiều. PhoWhisper tăng 6,6 điểm trên tập dò nhưng 0,0–0,9 điểm
   trên val (CI chứa 0).
3. **Lớp bảo vệ câu mồi chỉ cần thiết với câu mồi ngắn giống lời nói thật.** Với câu mồi mới, lớp bảo vệ gần như không
   kích hoạt (0–3 câu). Mức cụm 3 từ báo nhầm câu đúng, nên dùng cụm 4 từ.
4. **Chấm offline tin được**: ChunkFormer RNNT (2 467 câu) và large-v3 VAD (1 024 câu) chạy lại cho chữ giống hệt lần 1.
5. **Ghép RNNT + large-v3 (en_full + g4) là hướng tốt nhất**: chép đúng từ ngoại +22,7 điểm (seen) và +12,3 điểm (unseen)
   so với RNNT, WER câu chêm giảm 1,3–2,0 điểm, WER tổng không đổi, CI không chứa 0. Cái giá là WER câu thuần Việt tăng
   0,6–1,1 điểm do từ câu mồi chèn vào; cần siết quy tắc ghép trên tập dò trước khi dùng.
6. **Việc kế tiếp**, theo thứ tự: siết quy tắc ghép (6.7, mục 1); đo thời gian và bộ nhớ cạnh vLLM; viết engine ghép.
   Chưa đổi `configs/v0.toml`.

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
| (sau v5a) `prompt_insertions` coi "ti vi", "TV" của đáp án là khác "tivi" của câu mồi | Đếm 6 lần chèn cho en_top20 thay vì 3; phạt oan câu mồi có từ mượn | `en_mixed.spelled_in`: tính cả cách viết tương đương và từ bị tách 2–3 phần. Chấm lại 13 tổ hợp: cấu hình thắng không đổi |
| (sau v5a) `EQUIVALENTS` chỉ tra một chiều | Đáp án "TV", ASR "tivi" bị tính sai ở bản nới lỏng | Tra hai chiều (`en_mixed.spellings`); bản nới lỏng ở mục 3.5 tăng 0,3–0,9 điểm ở ba ô, các số khác không đổi |

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
- Số của large-v3 ở thử nghiệm A và ở mục 6.5.1 đến từ cấu hình nhiệt độ thử lại (không tất định); mục 4.7.6–4.7.7 và 6.5.2
  dùng cấu hình tất định.
- Cấu hình thắng được chọn trên tập dò 300 mẫu (394 từ ngoại); số trên tập dò lạc quan hơn thực tế, dùng số val.
- Câu mồi chọn từ theo chữ large-v3 beam 5 trong `data/asr/train_asr.json`, không phải cấu hình tham lam + VAD đang dò.
