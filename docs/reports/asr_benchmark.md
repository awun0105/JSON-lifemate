# Báo cáo so sánh mô hình và cấu hình ASR

Ngày: 28/10/2026. Trạng thái: **xong** (kernel Kaggle `quanlam/lifemate-asr-tune`, lần chạy v3).

Báo cáo trả lời câu hỏi: trong năm mô hình nhận dạng giọng nói tiếng Việt, mô hình nào, với cấu hình nào, chép lời
người dùng tốt nhất cho bài thi LifeMate. Phần 0 giải thích bối cảnh và thuật ngữ cho người đọc chưa theo dõi dự án.

**Kết quả thô.** Nằm trong `artifacts/runs/kaggle/asr/10-28_tune_v3/` (không có trên git), mỗi mô hình một thư mục
con: `large-v3/`, `phowhisper/`, `chunkformer/`, `chunkformer-rnnt/`, `parakeet/`.

| File (trong `<mô hình>/`) | Chứa gì | Dùng ở mục |
|---|---|---|
| `summary.json` | Toàn bộ kết quả của mô hình: `path` (lựa chọn giữ lại ở từng bước), `best_generate_kwargs` (cấu hình thắng), `runs` (chỉ số mọi lần đo trên tập chỉnh), `report` (chỉ số cấu hình thắng trên val_seen và val_unseen) | 6.1, 6.2, 6.3 |
| `<mô hình>__<bước>_<lựa chọn>.toml` | Config đầy đủ của một lần đo trên tập chỉnh, ví dụ `large-v3__prompt_lead.toml` | 4 |
| `<mô hình>__<bước>_<lựa chọn>.json` | Chỉ số của lần đo đó: WER, WER bỏ số, CER, bỏ sót tên riêng (kèm danh sách tên bị sót từng mẫu), câu rỗng, câu bịa, số lần đúng từng tên app, thời gian p50/p95/max | 6.1 |
| `<mô hình>__<bước>_<lựa chọn>.hyps.jsonl` | Chữ chép từng câu, mỗi dòng một câu: `id` (mẫu), `turn` (số lượt hoặc `final`), `ref` (đáp án của BTC), `hyp` (chữ sau bộ lọc), `raw` (chữ trước bộ lọc) | 6.4 |
| `<mô hình>__report_val_seen.{toml,json,hyps.jsonl}`, `<mô hình>__report_val_unseen.{…}` | Ba loại file trên cho cấu hình thắng, đo trên hai tập kiểm tra | 6.3, 6.4 |

Ngoài ra, `lifemate-asr-tune.log` ở thư mục gốc là log đầy đủ của kernel (cài đặt, từng lần đo, cảnh báo). Các ví dụ ở
mục 6.4 được trích nguyên văn từ các file `hyps.jsonl`. Cách xếp thư mục của mọi lần chạy xem
`artifacts/runs/README.md`.

## Mục lục

- [0. Bối cảnh và thuật ngữ](#0-bối-cảnh-và-thuật-ngữ)
- [1. Vấn đề](#1-vấn-đề)
- [2. Dữ liệu](#2-dữ-liệu)
  - [2.1. Dữ liệu gốc có cấu trúc thế nào](#21-dữ-liệu-gốc-có-cấu-trúc-thế-nào)
  - [2.2. Chia tập và lý do](#22-chia-tập-và-lý-do)
  - [2.3. Thước đo](#23-thước-đo)
- [3. Bộ cấu hình chung](#3-bộ-cấu-hình-chung)
- [4. Mô hình và cấu hình riêng](#4-mô-hình-và-cấu-hình-riêng)
- [5. Phương pháp thử nghiệm](#5-phương-pháp-thử-nghiệm)
- [6. Kết quả](#6-kết-quả)
  - [6.1. Đường dò trên tập chỉnh (300 mẫu, 1 024 câu, 123 tên riêng)](#61-đường-dò-trên-tập-chỉnh-300-mẫu-1-024-câu-123-tên-riêng)
  - [6.2. Cấu hình thắng của mỗi mô hình](#62-cấu-hình-thắng-của-mỗi-mô-hình)
  - [6.3. Đo trên tập kiểm tra](#63-đo-trên-tập-kiểm-tra)
  - [6.4. Ví dụ từ lần chạy](#64-ví-dụ-từ-lần-chạy)
- [7. Phân tích và đánh giá](#7-phân-tích-và-đánh-giá)
- [8. Kết luận](#8-kết-luận)
- [Giới hạn đã biết](#giới-hạn-đã-biết)

## 0. Bối cảnh và thuật ngữ

**Bài thi.** LifeMate (Zalo AI Challenge 2026) là trợ lý ảo điều khiển bằng giọng nói. Mỗi mẫu thi gồm một file ghi
âm yêu cầu của người dùng ("Đặt báo thức 5 rưỡi sáng mai"), có thể kèm vài lượt hội thoại trước đó, cùng danh sách
app đã cài. Hệ thống phải gọi đúng tool với đúng tham số, ví dụ `alarm.create(time="…05:30…")`. Điểm chấm dựa trên
lời gọi tool, không chấm chữ chép lời.

**Pipeline** (chuỗi xử lý) gồm hai bước:
1. **ASR** (Automatic Speech Recognition, nhận dạng giọng nói): biến file ghi âm thành chữ.
2. **LLM** (mô hình ngôn ngữ lớn, ở đây Qwen3-4B): đọc chữ, quyết định gọi tool nào với tham số gì.

Chữ ASR sai thì LLM không thể sửa: nghe "Zalo" thành "cho lô" thì LLM không biết phải mở Zalo. Báo cáo này chỉ đo
bước 1.

**Thuật ngữ dùng trong báo cáo:**

| Thuật ngữ | Nghĩa |
|---|---|
| Whisper, large-v3 | Nhóm mô hình ASR đa ngôn ngữ của OpenAI. `large-v3` là bản lớn nhất, đang được dùng trong config hiện tại `configs/v0.toml` |
| PhoWhisper | Whisper được VinAI huấn luyện thêm trên tiếng Việt |
| ChunkFormer, Parakeet | Các mô hình ASR kiểu khác (không phải Whisper), xem mục 4 |
| CTC, RNNT | Hai cách mô hình sinh chữ từ âm thanh. CTC đoán từng khung âm độc lập; RNNT đoán có nhìn lại chữ đã sinh. Cả hai không "viết tiếp câu" như Whisper nên gần như không bịa câu |
| Engine | Thư viện chạy mô hình: `faster_whisper` (CTranslate2), `hf_seq2seq` (transformers), `chunkformer`, `nemo_ctc` (NeMo của NVIDIA) |
| Giải mã tham lam (greedy), beam 5 | Cách chọn chữ khi sinh: tham lam lấy chữ có xác suất cao nhất ở mỗi bước; beam 5 giữ 5 phương án song song rồi chọn câu tốt nhất (chậm hơn) |
| Câu mồi (`initial_prompt`) | Đoạn chữ đưa cho Whisper trước khi nghe, coi như "câu trước đó", để gợi cách viết tên riêng (Zalo, VTV3) |
| `hotwords` | Danh sách từ khóa gợi ý cho faster-whisper. Thực chất được chèn vào cùng chỗ với câu mồi |
| Chống lặp | Hai tham số `repetition_penalty` và `no_repeat_ngram_size`, phạt hoặc cấm lặp lại cụm từ |
| VAD | Voice Activity Detection: cắt bỏ đoạn im lặng trước khi nhận dạng |
| Nhiệt độ thử lại (temperature fallback) | Nếu câu sinh ra có dấu hiệu hỏng (lặp, xác suất thấp), giải mã lại với độ ngẫu nhiên tăng dần 0,2; 0,4; … |
| Ảo giác, câu bịa | Whisper được huấn luyện trên nhiều phụ đề video YouTube, nên đôi khi "nghe" ra câu kiểu "Hãy đăng ký kênh…" dù audio không có |
| Bộ lọc (sanitizer) | Code của dự án (`lifemate/backends/asr_sanitizer.py`) cắt các cụm phụ đề bịa đã biết khỏi chữ ASR |
| WER | Word Error Rate: số từ phải sửa (thêm, xóa, thay) để chữ ASR thành đáp án, chia cho số từ đáp án. 20% nghĩa là cứ 10 từ sai khoảng 2 |
| p95 | 95% số mẫu chép xong trong thời gian này (giây). Đo trên GPU T4 |
| Sweep v1, sweep v2 | Hai lần thử ASR trước lần này, ghi trong `docs/experiments.md` |
| Cấu hình của lead | Cấu hình ban đầu do trưởng nhóm đề xuất: large-v3, beam 5, VAD, câu mồi dài, bật chống lặp |
| BTC | Ban tổ chức cuộc thi |

## 1. Vấn đề

Các lần đo trước để lại ba vấn đề:

1. **Cấu hình của lead cho nhiều câu rỗng.** Trên tập val_seen: WER 44,0%, 190/916 câu ra chuỗi rỗng, bỏ sót 52% tên
   riêng (sweep v1). Đếm lại trên chữ trước bộ lọc: 160/185 câu rỗng là câu Whisper bịa kiểu phụ đề YouTube rồi bị bộ
   lọc xóa, dù audio có độ dài và độ to bình thường. Ví dụ ở mục 6.4.
2. **Bỏ câu mồi thì sai tên riêng.** Không câu mồi: WER 29,7%, câu rỗng còn 24, nhưng "Zalo" chỉ viết đúng 2/19 câu
   (có câu mồi: 13/19). Tên app và tên kênh quyết định tool nào được gọi.
3. **So sánh trước đó không công bằng.** Sweep v1 chạy PhoWhisper với giải mã tham lam, không VAD, tắt bộ lọc, còn
   large-v3 chạy beam 5, có VAD, bật bộ lọc. Chênh lệch không tách được là do mô hình hay do cấu hình.

Lần đo này: mỗi mô hình được chỉnh với cùng công sức, trên cùng dữ liệu, theo cùng thước đo.

## 2. Dữ liệu

### 2.1. Dữ liệu gốc có cấu trúc thế nào

Tập train của BTC có 1 545 mẫu. Mỗi mẫu là một dòng trong `train.jsonl` kèm thư mục audio:

```json
{"id": "train_0007",
 "audio": "audio/train_0007/train_0007.wav",
 "audio_text": "Đặt báo thức 5 giờ nhé, à không, 5 rưỡi sáng mai đó",
 "history": [],
 "label": {"tool_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], …}}
```

- `audio`: file ghi âm **yêu cầu cuối**, phần được chấm điểm.
- `history`: các lượt hội thoại trước. Mỗi lượt người dùng có file audio riêng (`<id>_u1.wav`, `<id>_u2.wav`…) và chữ
  đáp án `audio_text`.
- `audio_text`: chữ đáp án do BTC ghi lại, dùng làm "đáp án" khi tính WER.
- `label.tool_calls`: lời gọi tool đúng. Dùng để biết tên riêng nào phải nghe đúng (mục 2.3).

Một mẫu có thể chứa nhiều câu nói (lượt cuối và các lượt trong lịch sử). Vì vậy 300 mẫu tương ứng 1 024 câu.

### 2.2. Chia tập và lý do

| Tập | Số mẫu | Số câu | Số tên riêng | Lấy từ đâu | Dùng để |
|---|---|---|---|---|---|
| `asr_tune.txt` | 300 | 1 024 | 123 | Rút từ tập train (1 093 mẫu) | Dò cấu hình (chọn lựa chọn ở từng bước) |
| `val_seen.txt` | 274 | 916 | 109 | Fold 0 | Đo cấu hình thắng, app quen |
| `val_unseen.txt` | 178 | 527 | 54 | Mẫu dùng 6 app bị giữ lại | Đo cấu hình thắng, app lạ |

Cách tạo (`lifemate/eval/make_splits.py`, seed cố định 42, chạy ở máy nào cũng ra giống nhau):

1. **Tách app lạ trước.** Mọi mẫu dùng một trong 6 app (medication, fitness, flights, bike_rental, library,
   subscriptions) vào `val_unseen`. Lý do: bộ thi riêng của BTC có app chưa từng thấy, nên cần một tập mô phỏng điều đó.
2. **Gom cụm câu gần trùng.** Các mẫu có `audio_text` giống nhau từ 85% trở lên được gom thành một cụm và luôn nằm cùng
   một tập. Lý do: nếu hai câu gần giống nhau nằm ở hai tập, kết quả kiểm tra sẽ đẹp hơn thực tế.
3. **Chia 5 fold phân tầng.** Phần còn lại chia thành 5 phần có cùng tỉ lệ số lời gọi tool và tỉ lệ câu phải hỏi lại.
   Fold 0 là `val_seen`, fold 1–4 là `train`.
4. **Rút `asr_tune.txt` từ train** (`--asr-tune 300`): lấy 300 mẫu, phân tầng theo loại đáp án (thực hiện, hỏi lại,
   ngoài phạm vi) và có/không lịch sử.

Vì sao dò cấu hình trên một tập riêng lấy từ train:
- **Không "chấm lại trên chính đề đã luyện".** Nếu vừa chọn cấu hình vừa báo kết quả trên val_seen, cấu hình được chọn
  sẽ khớp riêng với val_seen và con số báo cáo sẽ đẹp hơn thực tế. Ở đây val_seen và val_unseen chỉ được đo đúng một
  lần sau khi đã chọn xong.
- **Không rò rỉ.** Các mô hình ASR không học trên tập train của dự án, nên lấy mẫu train để dò không làm lộ đáp án.
  Nếu sau này fine-tune ASR trên train thì phải loại 300 mẫu này ra.
- **Các tập chính không đổi.** `train`, `val_seen`, `val_unseen` giữ nguyên từng byte, các kết quả cũ vẫn so được.

### 2.3. Thước đo

Mọi chỉ số tính trên cả tập: cộng tổng lỗi rồi chia tổng, không lấy trung bình theo câu. Trước khi so, chữ được chuyển
về chữ thường và bỏ dấu câu.

| Chỉ số | Cách tính | Vì sao cần |
|---|---|---|
| **Bỏ sót tên riêng** | Lấy các giá trị tham số trong lời gọi đáp án (tên người, số, tên kênh…) mà người dùng có nói ra trong câu cuối. Tên nào không xuất hiện trong chữ ASR (so không dấu, khớp nguyên từ) thì tính là bỏ sót | Đây là thứ ảnh hưởng thẳng tới điểm: sai tên thì sai tham số |
| **Câu rỗng** | Câu có đáp án nhưng ASR không ra chữ nào | Mất trắng cả lượt |
| WER | Như mục 0 | So với các số trước |
| **WER bỏ số** (`wer_nonum`) | WER sau khi bỏ chữ số và từ đọc số ở cả hai phía | Đáp án viết "5 giờ", ChunkFormer viết "năm giờ": WER thường phạt cách viết chứ không phải lỗi nghe (ví dụ 5 ở mục 6.4) |
| Câu bịa | Chữ trước bộ lọc khớp mẫu phụ đề bịa | Đo mức ảo giác |
| Tên riêng cụ thể | Số câu viết đúng Zalo, VTV, HTV, voucher… trên số câu đáp án có tên đó | Tên app quyết định gọi tool của app nào |
| p95 | Như mục 0 | Hạn mỗi mẫu 29 s, phần lớn dành cho LLM |

**Hàm mục tiêu** (để xếp hạng lựa chọn, nhỏ hơn là tốt hơn), giống nhau cho mọi mô hình:

1. Lựa chọn có p95 > 6 s xếp sau mọi lựa chọn đạt ngưỡng. Ngưỡng mềm vì large-v3 vượt 6 s ở mọi lựa chọn; loại hẳn thì
   không còn gì để so.
2. Bỏ sót tên riêng + tỉ lệ câu rỗng.
3. Nếu hòa, WER bỏ số.

## 3. Bộ cấu hình chung

Phần giữ giống hệt cho mọi mô hình:

| Mục | Giá trị | Lý do |
|---|---|---|
| Câu được đo | Mọi lượt người dùng trong lịch sử và lượt cuối | LLM đọc chữ của mọi lượt |
| Phần cứng | Một phiên Kaggle T4 cho cả năm mô hình, `dtype = "auto"` (float16 trên T4) | Thời gian so được với nhau |
| Giải mã | Tất định: `temperature = 0` (trừ lựa chọn nhiệt độ thử lại) | Chạy lại cho cùng kết quả |
| Bộ lọc ảo giác | Bật | Chạy sau ASR, không phụ thuộc engine, nên áp giống hệt cho mọi mô hình |
| `condition_on_previous_text` | `false` | Mỗi câu nghe độc lập, câu bịa không lây sang câu sau |
| Code đo | `lifemate/eval/eval_asr.py` | Cùng cách chuẩn hóa chữ và tính chỉ số |

## 4. Mô hình và cấu hình riêng

| Mô hình | Engine | Tham số | Giấy phép, ngày | Revision |
|---|---|---|---|---|
| `Systran/faster-whisper-large-v3` | `faster_whisper` (CTranslate2) | 1,55B | MIT | `edaa852e…` |
| `vinai/PhoWhisper-medium` | `hf_seq2seq` (transformers) | 0,77B | BSD-3 | `55a7e3eb…` |
| `khanhld/chunkformer-ctc-large-vie` | `chunkformer` | 0,11B | CC-BY-NC-4.0 (phi thương mại), 02/2025 | `311fc035…` |
| `khanhld/chunkformer-rnnt-large-vie` | `chunkformer` | 0,11B | CC-BY-4.0, 10/2025 | `9863592e…` |
| `nvidia/parakeet-ctc-0.6b-Vietnamese` | `nemo_ctc` (NeMo 2.6.2) | 0,6B | NVIDIA Open Model License, 01/2026 | `b0493142…` |

Revision là mã phiên bản cố định trên Hugging Face, để chạy lại luôn tải đúng bản đã đo. Cả năm mô hình phát hành
trước 26/10/2026; cộng với LLM 4,02B đều dưới giới hạn 6B của cuộc thi. ChunkFormer CTC có giấy phép phi thương mại,
cần BTC xác nhận trước khi dùng để nộp bài.

**Các bước dò.** Dò lần lượt từng bước; mỗi bước giữ lựa chọn thắng của các bước trước. Cách này tốn ít lần chạy hơn
thử mọi tổ hợp (large-v3 chỉ cần 8 lần đo thay vì 2 × 4 × 2 × 3 = 48). Lựa chọn engine không hỗ trợ thì bỏ qua.

| Bước | Lựa chọn | Lý do thử |
|---|---|---|
| 1. Giải mã | tham lam, beam 5 | Beam xem nhiều phương án hơn nhưng chậm hơn và dễ chọn câu "thuộc lòng" có xác suất cao |
| 2. Câu mồi | không; câu của lead (dài, liệt kê tên riêng); câu ngắn giống lời nói ("Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."); `hotwords` | Câu mồi giúp viết đúng tên riêng nhưng có thể gây bịa; sweep v2 cho thấy câu ngắn viết đúng Zalo nhiều hơn |
| 3. Chống lặp | tắt; `repetition_penalty = 1.2` + `no_repeat_ngram_size = 3` | Có trong cấu hình của lead; có thể làm sai âm tiết lặp hợp lệ trong tiếng Việt ("từ từ", "năm năm") |
| 4. Riêng engine | không; VAD (cắt khoảng lặng ≥ 500 ms); nhiệt độ thử lại `[0; 0,2; …; 1]` | Chỉ engine có tính năng đó mới thử |

Câu mồi của lead: `"LifeMate: trợ lý ảo tiếng Việt điều khiển smart home, tivi, đặt xe, ngân hàng, HTV7, VTV1, VTV3,
deadline, voucher, Zalo."`

**Lựa chọn mỗi mô hình được thử, và lý do:**

| Mô hình | Bước 1 | Bước 2 | Bước 3 | Bước 4 | Lý do |
|---|---|---|---|---|---|
| large-v3 | tham lam, beam 5 | không, lead, ngắn, `hotwords` | tắt, bật | không, VAD, nhiệt độ thử lại | faster-whisper có đủ mọi tham số |
| PhoWhisper | tham lam, beam 5 | không, lead, ngắn | tắt, bật | không | transformers có beam, câu mồi (`prompt_ids`), chống lặp; không có VAD, `hotwords`. Nhiệt độ thử lại của transformers cần thêm ngưỡng riêng nên không thử, để giống cách tính của faster-whisper |
| ChunkFormer CTC | tham lam | không | tắt | không | Gói `chunkformer` chỉ giải mã tham lam; mô hình CTC không có câu mồi hay chống lặp |
| ChunkFormer RNNT | tham lam | không | tắt | không | Như trên (cùng gói) |
| Parakeet CTC | tham lam | không | tắt | không | Beam + mô hình ngôn ngữ KenLM đi kèm repo không thử được: gói flashlight-text không build được trên Python 3.13 của Kaggle |

**Chi tiết theo engine, và lý do:**

- **PhoWhisper không chuyển sang CTranslate2** (định dạng của faster-whisper), vì chuyển định dạng có thể làm kết quả
  lệch so với bản gốc. Thay vào đó, engine transformers nhận cùng tên tham số với faster-whisper và tự dịch
  (`hf_generate_kwargs` trong `lifemate/backends/asr.py`): `beam_size` thành `num_beams`, `initial_prompt` thành
  `prompt_ids`, `condition_on_previous_text` thành `condition_on_prev_tokens`. `best_of` bị bỏ vì chỉ có tác dụng khi
  giải mã ngẫu nhiên.
- **ChunkFormer:** gói tự tải mô hình nhưng bỏ qua revision, nên engine tải đúng revision trước khi nạp. Tham số chia
  đoạn giữ mặc định (`chunk_size = 64`, ngữ cảnh trái/phải 128), không dò, để số lần chạy ngang với các mô hình kia.
- **Parakeet:** không dùng hàm `transcribe` có sẵn vì bộ nạp dữ liệu của NeMo lỗi với torch 2.13 trên Kaggle. Engine
  tự đọc WAV, gọi thẳng mô hình rồi bộ giải mã CTC. NeMo hạ phiên bản transformers, nên Parakeet đo sau cùng.

## 5. Phương pháp thử nghiệm

1. **Chạy thử trên CPU** (kernel `quanlam/lifemate-asr-preflight`): cài đúng thư viện như kernel GPU, chạy mọi lựa
   chọn của từng mô hình trên 3 mẫu, để bắt lỗi cài đặt và nạp mô hình trước khi tốn giờ GPU. Lần 1 Parakeet lỗi mọi
   lựa chọn và flashlight-text không build được; sau khi sửa engine và bỏ lựa chọn KenLM, lần 2 mọi mô hình chạy được.
2. **Dò cấu hình trên GPU** (kernel `quanlam/lifemate-asr-tune`): mỗi mô hình chạy `python -m lifemate.eval.asr_tune`
   theo các bước ở mục 4 trên `asr_tune.txt`. Lựa chọn "giữ nguyên" dùng lại kết quả đã đo.
3. **Đo cấu hình thắng** của mỗi mô hình đúng một lần trên val_seen và val_unseen.

Tái tạo: `make splits` (sinh `asr_tune.txt`), rồi `kaggle kernels push -p ops/kaggle/asr_tune --accelerator
NvidiaTeslaT4`, hoặc trên máy có GPU: `python -m lifemate.eval.asr_tune --env <profile> --out <thư mục>`.

## 6. Kết quả

Một phiên T4, khoảng 4 giờ. Mọi mô hình chạy hết, không lần đo nào lỗi. Log có một cảnh báo hết bộ nhớ GPU ở lựa chọn
beam 5 của PhoWhisper, nhưng lần đo đó vẫn hoàn tất.

### 6.1. Đường dò trên tập chỉnh (300 mẫu, 1 024 câu, 123 tên riêng)

Lựa chọn được giữ in đậm.

| Mô hình | Bước | Lựa chọn | Bỏ sót | Rỗng | WER | WER bỏ số | Bịa | p95 (s) |
|---|---|---|---|---|---|---|---|---|
| large-v3 | giải mã | **tham lam** | 41,5% | 10 | 30,8% | 28,9% | 17 | 8,32 |
| | | beam 5 | 42,3% | 31 | 30,8% | 28,7% | 39 | 7,65 |
| | câu mồi | **không** | 41,5% | 10 | 30,8% | 28,9% | 17 | 8,32 |
| | | lead | 49,6% | 22 | 48,6% | 42,6% | 22 | 7,19 |
| | | ngắn | 43,9% | 15 | 34,7% | 32,4% | 14 | 6,76 |
| | | `hotwords` | 59,3% | 4 | 48,7% | 44,6% | 5 | 6,75 |
| | chống lặp | **tắt** | 41,5% | 10 | 30,8% | 28,9% | 17 | 8,32 |
| | | bật | 42,3% | 9 | 31,6% | 29,7% | 13 | 6,69 |
| | engine | không | 41,5% | 10 | 30,8% | 28,9% | 17 | 8,32 |
| | | VAD | 41,5% | 8 | 30,8% | 28,8% | 8 | 6,81 |
| | | **nhiệt độ thử lại** | 40,7% | 11 | 30,8% | 28,8% | 18 | 6,72 |
| PhoWhisper | giải mã | **tham lam** | 48,0% | 0 | 28,1% | 28,8% | 0 | 3,18 |
| | | beam 5 | 48,0% | 0 | 27,6% | 28,3% | 0 | 9,21 |
| | câu mồi | không | 48,0% | 0 | 28,1% | 28,8% | 0 | 3,18 |
| | | lead | 47,2% | 0 | 28,0% | 28,6% | 0 | 3,23 |
| | | **ngắn** | 46,3% | 0 | 28,1% | 28,8% | 0 | 3,09 |
| | chống lặp | **tắt** | 46,3% | 0 | 28,1% | 28,8% | 0 | 3,09 |
| | | bật | 46,3% | 0 | 29,6% | 30,3% | 0 | 3,25 |
| ChunkFormer CTC | (một cấu hình) | tham lam | 39,0% | 0 | 21,0% | 20,5% | 0 | 2,94 |
| ChunkFormer RNNT | (một cấu hình) | tham lam | 37,4% | 0 | 20,3% | 19,5% | 0 | 2,71 |
| Parakeet | (một cấu hình) | tham lam | 63,4% | 70 | 36,5% | 36,8% | 0 | 0,86 |

Cách đọc: với large-v3, bước 1 chọn tham lam vì beam 5 cho nhiều câu rỗng hơn (31 so với 10). Ở bước 2, không câu mồi
nào tốt hơn "không câu mồi". Bước 4 chọn nhiệt độ thử lại vì bỏ sót thấp nhất; mọi lựa chọn của large-v3 đều vượt 6 s
nên chỉ so với nhau.

### 6.2. Cấu hình thắng của mỗi mô hình

| Mô hình | Cấu hình thắng |
|---|---|
| large-v3 | tham lam, không câu mồi, không chống lặp, nhiệt độ thử lại `[0.0, 0.2, …, 1.0]` |
| PhoWhisper | tham lam, câu mồi ngắn `"Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."`, không chống lặp |
| ChunkFormer CTC, ChunkFormer RNNT, Parakeet | tham lam, tham số mặc định |

### 6.3. Đo trên tập kiểm tra

| Mô hình | Tập | Bỏ sót | Rỗng | WER | WER bỏ số | Bịa | p95 (s) | Zalo | VTV |
|---|---|---|---|---|---|---|---|---|---|
| large-v3 | val_seen | 31,2% | 4 | 30,6% | 28,9% | 12 | 8,19 | 1/19 | 9/13 |
| | val_unseen | 57,4% | 3 | 30,2% | 27,9% | 5 | 7,93 | 0/6 | 1/1 |
| PhoWhisper | val_seen | 38,5% | 0 | 27,9% | 28,6% | 0 | 3,04 | 16/19 | 5/13 |
| | val_unseen | 48,1% | 0 | 27,9% | 28,9% | 0 | 3,01 | 1/6 | 1/1 |
| ChunkFormer CTC | val_seen | **23,9%** | 0 | 21,3% | 21,0% | 0 | 2,68 | 5/19 | 10/13 |
| | val_unseen | 55,6% | 0 | 21,0% | 20,9% | 0 | 2,30 | 0/6 | 1/1 |
| ChunkFormer RNNT | val_seen | 32,1% | 0 | **20,0%** | **19,7%** | 0 | 2,53 | **17/19** | 10/13 |
| | val_unseen | **48,1%** | 0 | **19,8%** | **19,8%** | 0 | 2,49 | **5/6** | 1/1 |
| Parakeet | val_seen | 54,1% | 37 | 35,3% | 35,7% | 0 | 0,87 | 5/19 | 1/13 |
| | val_unseen | 79,6% | 51 | 38,2% | 38,7% | 0 | 0,74 | 0/6 | 1/1 |

### 6.4. Ví dụ từ lần chạy

Trích nguyên văn từ `*__report_val_seen.hyps.jsonl` (cấu hình thắng của mỗi mô hình) và
`large-v3__prompt_lead.hyps.jsonl`. "Đáp án" là `audio_text` của BTC; "trước lọc" là chữ Whisper sinh ra trước khi bộ
lọc cắt.

**Ví dụ 1. Whisper bịa câu phụ đề YouTube.**

| | Chữ |
|---|---|
| Đáp án | Số điện thoại của Sếp Dũng là số nào ấy em? |
| large-v3, trước lọc | Hãy xem video này và hãy đăng ký kênh để ủng hộ kênh của mình nhé. |
| large-v3, sau lọc | Hãy xem video này |
| ChunkFormer RNNT | số điện thoại của sếp dũng là số nào ấy em |

large-v3 không nghe ra câu nào gần đáp án mà sinh hẳn một câu khác. Bộ lọc chỉ cắt được phần "đăng ký kênh" đã biết,
phần còn lại vẫn sai. ChunkFormer chép đúng hoàn toàn. Đây là lý do large-v3 có 12 câu bịa trên val_seen còn các mô
hình CTC/RNNT có 0.

**Ví dụ 2. Câu bịa nối vào sau câu đúng.**

| | Chữ |
|---|---|
| Đáp án | Tài khoản của anh còn được bao nhiêu? |
| large-v3, trước lọc | Tài khoản của anh còn được bao nhiêu? Hẹn gặp lại các bạn trong những video tiếp theo. |
| large-v3, sau lọc | Tài khoản của anh còn được bao nhiêu? |

Trường hợp này bộ lọc cứu được, vì phần bịa nằm riêng sau câu đúng.

**Ví dụ 3. Câu mồi của lead làm mất trắng câu nói** (tập chỉnh, large-v3 với câu mồi của lead).

| | Chữ |
|---|---|
| Đáp án | Vàng SJC hôm nay giá cả sao rồi? |
| large-v3, trước lọc | Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn |
| large-v3, sau lọc | (rỗng) |

Câu mồi dài, dạng liệt kê, trông giống phần mô tả video, nên Whisper tiếp tục theo kiểu phụ đề YouTube. Bộ lọc xóa
hết, lượt đó mất trắng. Câu mồi của lead làm câu rỗng tăng từ 10 lên 22 trên tập chỉnh.

**Ví dụ 4. Tên app "Zalo".**

| | Chữ |
|---|---|
| Đáp án | Mở giúp em đoạn chat Zalo với chị Lan, em muốn đọc lại tin nhắn chị ấy gửi. |
| large-v3 | Mơ giúp em đàn chất cho lô vậy chị lán Em muộn đọc lại tí nhận chị ấy gửi |
| PhoWhisper | mơ giúp em đoàn toát cho lô vợ chị lan é mộ đọc lại tín dụng chị ấy gửi. |
| ChunkFormer CTC | m giúp em đàn choắt cho lô vợ chị lan em muốn đọc lại từ nhắn chị ấy gửi |
| ChunkFormer RNNT | m giúp em đàn choắt zalo với chị lan em muốn đọc lại từ nhậm chị ấy gửi |

Câu này khó với mọi mô hình (giọng nói nhanh, "đoạn chat" bị nghe sai). Chỉ ChunkFormer RNNT viết ra "zalo"; các mô
hình khác nghe thành "cho lô", nên LLM không biết phải dùng app Zalo.

**Ví dụ 5. Cách viết số, lý do cần WER bỏ số.**

| | Chữ |
|---|---|
| Đáp án | Đặt báo thức 5 giờ nhé, à không, 5 rưỡi sáng mai đó |
| large-v3 | Đặt báo thức 5 giờ nhé. À không, 5 rưỡi sáng mai đó. |
| ChunkFormer RNNT | tập báo thức năm giờ nhé à không năm rưỡi sáng mai đó |

ChunkFormer nghe đúng số nhưng viết bằng chữ ("năm giờ"); WER thường tính "5" khác "năm" là lỗi. WER bỏ số loại cả
hai cách viết số nên chỉ còn lỗi nghe thật ("tập" thay cho "đặt").

**Ví dụ 6. Parakeet ra ký tự lạ.** Mọi câu "rỗng" của Parakeet (37 trên val_seen, 51 trên val_unseen) đều có chữ
ASR là đúng một ký tự `⁇`, kể cả câu ngắn và rõ như "Gần đây có quán pizza nào không?". `⁇` là ký hiệu "token không
biết" của bộ tách từ. Xem mục 7.

## 7. Phân tích và đánh giá

- **ChunkFormer RNNT tốt nhất xét tổng thể.** WER thấp nhất trên cả ba tập, khoảng 20%. Trên val_unseen, bỏ sót ít
  nhất (48,1%, bằng PhoWhisper) và viết đúng "Zalo" nhiều nhất (17/19 và 5/6). Không có câu rỗng hay câu bịa, p95
  2,5 s, chưa tới một nửa ngưỡng. Giấy phép CC-BY-4.0.
- **ChunkFormer CTC bỏ sót ít nhất trên val_seen (23,9%), nhưng kém trên val_unseen (55,6%).** Gần như luôn sai tên
  app (Zalo 5/19 và 0/6, ví dụ 4). Khoảng cách giữa hai tập cho thấy lợi thế trên val_seen đến từ vài loại tên riêng
  hay gặp. Giấy phép phi thương mại.
- **large-v3 (cấu hình hiện tại) vượt ngưỡng thời gian ở mọi lựa chọn**, p95 khoảng 8 s. WER khoảng 30%, cao hơn
  ChunkFormer 10 điểm, vẫn còn câu bịa dù đã bật bộ lọc (ví dụ 1), gần như không bao giờ viết đúng "Zalo" (1/19). Câu
  mồi và `hotwords` đều làm kết quả kém hơn (ví dụ 3), khớp với sweep v2.
- **PhoWhisper không bịa và nhanh hơn large-v3** (p95 khoảng 3 s). Câu mồi ngắn giúp viết đúng "Zalo" trên val_seen
  (16/19), nhưng bỏ sót vẫn cao (38–48%) vì sai nhiều ở tên riêng khác. Beam 5 gần như không cải thiện mà chậm gấp ba.
- **Parakeet nhanh nhất nhưng kết quả lần này chưa đáng tin.** p95 dưới 1 s, nhưng 37–70 câu ra `⁇` (ví dụ 6) và bỏ
  sót 54–80%. Kiểu lỗi "cả câu chỉ còn token không biết", rải rác ở mọi kiểu câu, giống lỗi tính toán trong engine tự
  viết (nhiều khả năng float16 tràn số) hơn là mô hình nghe kém. Chữ cũng có nhiều ký tự hỏng ("điệnữ", "giúpánh").
  Muốn kết luận về Parakeet phải chạy lại ở float32.
- **Độ tin cậy.** val_unseen có ít câu chứa từng tên riêng (Zalo 6 câu, VTV 1 câu), nên chênh vài câu ở đó chưa đủ để
  kết luận. Trên cả ba tập, ChunkFormer RNNT có WER ổn định (19,5–20,3%) và luôn ở nhóm đầu về bỏ sót; kết luận dựa
  chủ yếu vào sự ổn định này.

## 8. Kết luận

1. Chọn **ChunkFormer RNNT large** (`khanhld/chunkformer-rnnt-large-vie`, giải mã tham lam, tham số mặc định) làm ứng
   viên ASR chính. So với large-v3: WER giảm từ khoảng 30% xuống 20%, không còn câu bịa hay câu rỗng, p95 giảm từ 8 s
   xuống 2,5 s, giấy phép cho phép dùng.
2. Giữ **PhoWhisper medium + câu mồi ngắn** làm ứng viên dự phòng: không bịa, p95 3 s, viết đúng tên app trên val_seen.
3. Chưa đổi `configs/v0.toml`. Chỉ số ASR chỉ dùng để rút gọn danh sách; chọn cuối cùng phải dựa trên điểm thi của cả
   hệ thống (ASR + LLM, chấm bằng `score.py`) trên GPU từ 24 GB, so ChunkFormer RNNT với large-v3 hiện tại, có thể thêm
   PhoWhisper.
4. Parakeet: chạy lại ở float32 trước khi loại. ChunkFormer CTC chỉ xét lại nếu BTC cho dùng giấy phép phi thương mại
   và điểm thi cho thấy lợi thế trên val_seen là thật.

## Giới hạn đã biết

- Thời gian đo trên T4, chưa phải GPU A30 lúc chấm.
- Tập chỉnh có ít câu chứa từng tên riêng, nên chênh vài câu đã là khác biệt lớn.
- Bỏ sót tên riêng so không dấu và khớp nguyên từ: "zalo" và "Zalo" được tính là đúng, nhưng "dalo" bị tính là sai.
- Nếu sau này fine-tune ASR trên train, phải loại `asr_tune.txt` khỏi dữ liệu train ASR.
