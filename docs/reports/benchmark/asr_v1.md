# Báo cáo benchmark ASR lần 1: so sánh năm mô hình và cấu hình ASR

Ngày: 28/10/2026, cập nhật 29/10/2026 và 30/10/2026. Trạng thái: **xong** (kernel Kaggle `quanlam/lifemate-asr-tune`: lần chạy v3
cho bốn mô hình, lần chạy v4 chạy lại riêng Parakeet sau khi sửa lỗi engine, xem mục 6.5).

Báo cáo trả lời câu hỏi: trong năm mô hình nhận dạng giọng nói tiếng Việt, mô hình nào, với cấu hình nào, chép lời
người dùng tốt nhất cho bài thi LifeMate. Phần 0 giải thích bối cảnh và thuật ngữ cho người đọc chưa theo dõi dự án.

## Tóm tắt cho cả team

Phần này đọc riêng được, không cần đọc các mục sau. Thuật ngữ được giải thích ngay tại chỗ; định nghĩa đầy đủ ở mục 0.

### Pipeline của hệ thống

Mỗi mẫu thi gồm file âm thanh của câu cuối (yêu cầu cần thực hiện), các lượt hội thoại trước đó, danh sách app đã cài,
catalog tool của các app đó và ngữ cảnh người dùng (giờ, vị trí, hồ sơ). Lượt người dùng trong lịch sử **chỉ có âm
thanh**, nên cũng phải chép. Hệ thống chạy hai bước, trong giới hạn 30 giây mỗi mẫu và tổng tham số không quá 6B:

```
âm thanh (câu cuối + các câu người dùng trong lịch sử, tới 13 file)
  → ASR: mô hình chép thành chữ → bộ lọc ảo giác cắt câu phụ đề bịa
  → LLM (Qwen3-4B): đọc chữ của cả hội thoại, chọn tool và tham số
  → lời gọi tool, ví dụ alarm.create(time="…05:30…")
```

Điểm thi chấm lời gọi tool, không chấm chữ. Nhưng chữ ASR sai thì LLM không sửa được: nghe "Zalo" thành "cho lô" thì LLM
không biết phải mở Zalo. Báo cáo này chỉ đo bước ASR.

### Vì sao có lần đo này

Cấu hình ASR lúc đó (whisper-large-v3, beam 5, VAD, câu mồi dài liệt kê tên riêng, bật chống lặp) có ba vấn đề:

| Vấn đề | Số đo trước lần 1 (`val_seen`) |
|---|---|
| Nhiều câu rỗng: Whisper bịa câu phụ đề YouTube ("Hãy đăng ký kênh…") rồi bộ lọc xoá mất | WER 44,0%; 190/916 câu rỗng; 160/185 câu rỗng là câu bịa |
| Bỏ câu mồi thì giảm câu rỗng nhưng sai tên app | Không câu mồi: WER 29,7%, nhưng "Zalo" chỉ đúng 2/19 câu (có câu mồi: 13/19) |
| Các lần so trước không công bằng | PhoWhisper chạy tham lam, không VAD, tắt bộ lọc; large-v3 chạy beam 5, có VAD, bật bộ lọc |

### Đã làm gì

- **Năm mô hình:** whisper-large-v3 (1,55B), PhoWhisper-medium (0,77B, Whisper học thêm tiếng Việt), ChunkFormer CTC và
  ChunkFormer RNNT (0,11B, mô hình tiếng Việt kiểu khác Whisper), Parakeet CTC (0,6B).
- **So công bằng:** mọi mô hình chạy trên cùng GPU T4, cùng bộ lọc, cùng cách chấm, giải mã tất định. Phần khác nhau
  (cách giải mã, câu mồi, chống lặp, VAD) được dò riêng cho từng mô hình, cùng một cách: vặn lần lượt từng "núm", giữ
  lựa chọn tốt nhất rồi sang núm tiếp theo.
- **Dữ liệu:** dò trên tập dò `asr_tune` (300 mẫu train, 1 024 câu); đo cấu hình thắng của mỗi mô hình **đúng một lần**
  trên hai tập kiểm tra `val_seen` (274 mẫu, app đã biết) và `val_unseen` (178 mẫu, 6 app chưa gặp). Chấm **mọi câu
  người dùng**, cả câu cuối lẫn lịch sử, vì LLM đọc tất cả.
- **Thước đo chính:** tỉ lệ bỏ sót tên riêng (giá trị tham số mà người dùng nói ra, như tên người, tên kênh, số tiền),
  số câu rỗng, WER (tỉ lệ từ sai), số câu bịa, p95 (95% số mẫu chép xong trong số giây này).

### Kết quả chính

Cấu hình thắng của mỗi mô hình, đo một lần trên hai tập kiểm tra (`val_seen` / `val_unseen`):

| Mô hình | Cấu hình thắng | WER | Bỏ sót tên riêng | Câu rỗng | Câu bịa | p95 (s) | "Zalo" viết đúng |
|---|---|---|---|---|---|---|---|
| whisper-large-v3 | tham lam, không câu mồi, nhiệt độ thử lại | 30,6% / 30,2% | 31,2% / 57,4% | 4 / 3 | 12 / 5 | 8,2 / 7,9 | 1/19 / 0/6 |
| PhoWhisper-medium | tham lam, câu mồi ngắn | 27,9% / 27,9% | 38,5% / 48,1% | 0 / 0 | 0 / 0 | 3,0 / 3,0 | 16/19 / 1/6 |
| ChunkFormer CTC | tham lam, mặc định | 21,3% / 21,0% | **23,9%** / 55,6% | 0 / 0 | 0 / 0 | 2,7 / 2,3 | 5/19 / 0/6 |
| **ChunkFormer RNNT** | tham lam, mặc định | **20,0% / 19,8%** | 32,1% / **48,1%** | 0 / 0 | 0 / 0 | 2,5 / 2,5 | **17/19 / 5/6** |
| Parakeet CTC | tham lam, mặc định (float32) | 31,6% / 31,1% | 50,5% / 77,8% | 0 / 0 | 0 / 0 | 2,0 / 1,9 | 5/19 / 0/6 |

Đọc bảng:

1. **ChunkFormer RNNT tốt nhất xét tổng thể:** WER thấp nhất, khoảng 20% (large-v3 khoảng 30%); không câu rỗng, không
   câu bịa; p95 2,5 giây; viết đúng "Zalo" nhiều nhất; giấy phép CC-BY-4.0 cho phép dùng.
2. **Với large-v3, mọi câu mồi đều làm kết quả kém đi.** Câu mồi dài dạng liệt kê tăng WER từ 30,8% lên 48,6% và câu
   rỗng từ 10 lên 22 trên tập dò, vì Whisper coi câu mồi như mô tả video rồi viết tiếp kiểu phụ đề. `hotwords` (danh
   sách từ khoá) còn kém hơn. Chống lặp và beam 5 cũng không giúp mô hình nào.
3. **ChunkFormer CTC** bỏ sót ít nhất trên `val_seen` nhưng kém trên `val_unseen` và gần như luôn sai tên app; giấy phép
   phi thương mại. **Parakeet** kém nhất về tên riêng, kể cả sau khi sửa lỗi float16 (mục 6.5).
4. **Ví dụ:** câu "Số điện thoại của Sếp Dũng là số nào ấy em?" được large-v3 chép thành "Hãy xem video này và hãy đăng
   ký kênh để ủng hộ kênh của mình nhé."; ChunkFormer RNNT chép đúng hoàn toàn (mục 6.4).

### Kết luận

- Chọn **ChunkFormer RNNT large** (`khanhld/chunkformer-rnnt-large-vie`, giải mã tham lam, tham số mặc định) làm ASR
  chính. Giữ PhoWhisper kèm câu mồi ngắn làm dự phòng. Bỏ Parakeet.
- Chưa đổi `configs/v0.toml`: chỉ số ASR chỉ dùng để rút gọn danh sách, lựa chọn cuối phải dựa trên điểm thi của cả hệ
  thống.

### Sau lần 1

Lần 1 để lại bốn điểm: chưa đo riêng từ tiếng Anh chêm trong câu (tên app, tên dịch vụ); mọi câu mồi đều làm large-v3
kém đi; cấu hình thắng của large-v3 (nhiệt độ thử lại) không tất định; mới chọn một mô hình, chưa thử kết hợp.

- **Lần 2** (`docs/reports/benchmark/asr_v2.md`) đo riêng các câu chêm tiếng Anh, viết câu mồi mới dạng câu nói tự
  nhiên kèm lớp bảo vệ (large-v3 chép đúng từ tiếng Anh từ 52% lên 72%), và ghép chữ RNNT với chữ large-v3.
- **Lần 3** (`docs/reports/benchmark/asr_v3.md`) dựng câu mồi riêng cho từng câu từ tên app đã cài và lời trợ lý liền
  trước, thử các cách quyết định khi ghép, và đo điểm thi bằng cách chạy agent trên chữ ASR.

### Đọc tiếp ở đâu

| Muốn biết | Đọc |
|---|---|
| Thuật ngữ | Mục 0 |
| Dữ liệu và cách chia tập | Mục 2 |
| Bộ cấu hình chung và cách dò từng mô hình | Mục 3, 4 |
| Kết quả đầy đủ, ví dụ, lỗi Parakeet | Mục 6 |
| Phân tích và kết luận | Mục 7, 8 |

**Kết quả thô.** Nằm trong `artifacts/runs/kaggle/asr/10-28_tune_v3/` (không có trên git), mỗi mô hình một thư mục
con: `large-v3/`, `phowhisper/`, `chunkformer/`, `chunkformer-rnnt/`, `parakeet/`. Kết quả Parakeet trong thư mục này
là lần chạy v3, **không hợp lệ** (mục 6.5). Kết quả Parakeet dùng trong báo cáo nằm ở
`artifacts/runs/kaggle/asr/10-29_tune_v4_parakeet/parakeet/`, cùng cấu trúc file. Lần chẩn đoán lỗi nằm ở
`artifacts/runs/kaggle/asr/10-29_parakeet_diag_v1/`.

| File (trong `<mô hình>/`) | Chứa gì | Dùng ở mục |
|---|---|---|
| `summary.json` | Toàn bộ kết quả của mô hình: `path` (lựa chọn giữ lại ở từng bước), `best_generate_kwargs` (cấu hình thắng), `runs` (chỉ số mọi lần đo trên tập dò), `report` (chỉ số cấu hình thắng trên val_seen và val_unseen) | 6.1, 6.2, 6.3 |
| `<mô hình>__<bước>_<lựa chọn>.toml` | Config đầy đủ của một lần đo trên tập dò, ví dụ `large-v3__prompt_lead.toml` | 4 |
| `<mô hình>__<bước>_<lựa chọn>.json` | Chỉ số của lần đo đó: WER, WER bỏ số, CER, bỏ sót tên riêng (kèm danh sách tên bị sót từng mẫu), câu rỗng, câu bịa, số lần đúng từng tên app, thời gian p50/p95/max | 6.1 |
| `<mô hình>__<bước>_<lựa chọn>.hyps.jsonl` | Chữ chép từng câu, mỗi dòng một câu: `id` (mẫu), `turn` (số lượt hoặc `final`), `ref` (đáp án của BTC), `hyp` (chữ sau bộ lọc), `raw` (chữ trước bộ lọc) | 6.4 |
| `<mô hình>__report_val_seen.{toml,json,hyps.jsonl}`, `<mô hình>__report_val_unseen.{…}` | Ba loại file trên cho cấu hình thắng, đo trên hai tập kiểm tra | 6.3, 6.4 |

Ngoài ra, `lifemate-asr-tune.log` ở thư mục gốc là log đầy đủ của kernel (cài đặt, từng lần đo, cảnh báo). Các ví dụ ở
mục 6.4 được trích nguyên văn từ các file `hyps.jsonl`. Cách xếp thư mục của mọi lần chạy xem
`artifacts/runs/README.md`.


## Mục lục

- [Tóm tắt cho cả team](#tóm-tắt-cho-cả-team)
- [0. Bối cảnh và thuật ngữ](#0-bối-cảnh-và-thuật-ngữ)
- [1. Vấn đề](#1-vấn-đề)
- [2. Dữ liệu dùng cho benchmark](#2-dữ-liệu-dùng-cho-benchmark)
  - [2.1. Dữ liệu gốc của BTC (chung cho cả dự án)](#21-dữ-liệu-gốc-của-btc-chung-cho-cả-dự-án)
  - [2.2. Ba tập của benchmark và lý do](#22-ba-tập-của-benchmark-và-lý-do)
  - [2.3. Thước đo](#23-thước-đo)
- [3. Bộ cấu hình chung](#3-bộ-cấu-hình-chung)
- [4. Mô hình và cấu hình riêng](#4-mô-hình-và-cấu-hình-riêng)
  - [4.1. Năm mô hình](#41-năm-mô-hình)
  - [4.2. Cách dò cấu hình cho từng mô hình](#42-cách-dò-cấu-hình-cho-từng-mô-hình)
  - [4.3. Điều chỉnh kỹ thuật theo engine](#43-điều-chỉnh-kỹ-thuật-theo-engine)
- [5. Phương pháp thử nghiệm](#5-phương-pháp-thử-nghiệm)
- [6. Kết quả](#6-kết-quả)
  - [6.1. Đường dò trên tập dò (300 mẫu, 1 024 câu, 123 tên riêng)](#61-đường-dò-trên-tập-chỉnh-300-mẫu-1-024-câu-123-tên-riêng)
  - [6.2. Cấu hình thắng của mỗi mô hình](#62-cấu-hình-thắng-của-mỗi-mô-hình)
  - [6.3. Đo trên tập kiểm tra](#63-đo-trên-tập-kiểm-tra)
  - [6.4. Ví dụ từ lần chạy](#64-ví-dụ-từ-lần-chạy)
  - [6.5. Sửa lỗi Parakeet và chạy lại](#65-sửa-lỗi-parakeet-và-chạy-lại)
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
| Tập dò | Tập `asr_tune` (300 mẫu train) dùng để dò và chọn cấu hình; khác với hai tập kiểm tra chỉ dùng để đo một lần |
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

## 2. Dữ liệu dùng cho benchmark

**Tóm tắt.** Benchmark chạy trên **752 mẫu (2 467 câu nói)**, chia thành ba tập. Tập `asr_tune` gồm 300 mẫu, dùng để
chọn cấu hình. Hai tập `val_seen` (274 mẫu) và `val_unseen` (178 mẫu) dùng để đo kết quả cuối. Tất cả đều lấy từ tập
train có nhãn của BTC (1 545 mẫu). 793 mẫu còn lại không dùng trong benchmark này. Không dùng tập public test vì tập
đó không có đáp án.

Mục 2.1 mô tả dữ liệu gốc của BTC, chung cho cả dự án. Mục 2.2 mô tả cách lấy ra ba tập trên.

### 2.1. Dữ liệu gốc của BTC (chung cho cả dự án)

Tập train của BTC có 1 545 mẫu, nằm trong `data/train+public_test/train/` (không có trên git). Mỗi mẫu là một dòng
trong `train.jsonl`, kèm thư mục audio:

```json
{"id": "train_0007",
 "audio": "audio/train_0007/train_0007.wav",
 "audio_text": "Đặt báo thức 5 giờ nhé, à không, 5 rưỡi sáng mai đó",
 "history": [],
 "label": {"tool_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], …}}
```

- `audio`: file ghi âm **yêu cầu cuối**, tức yêu cầu hệ thống phải thực hiện. Điểm thi tính trên lời gọi tool đáp lại
  yêu cầu này (so với `label.tool_calls`), không chấm chữ ASR.
- `history`: các lượt hội thoại trước. Mỗi lượt người dùng có file audio riêng (`<id>_u1.wav`, `<id>_u2.wav`…) và chữ
  đáp án `audio_text` riêng.
- `audio_text`: chữ đáp án do BTC ghi lại. Benchmark dùng trường này làm "đáp án" khi tính WER.
- `label.tool_calls`: lời gọi tool đúng. Benchmark dùng trường này để biết tên riêng nào phải nghe đúng (mục 2.3).

Benchmark chấm **mọi câu nói của người dùng** trong một mẫu: lượt cuối và từng lượt trong lịch sử. Lý do là LLM đọc chữ
của mọi lượt. Vì vậy số câu lớn hơn số mẫu: 300 mẫu của `asr_tune` có 1 024 câu.

### 2.2. Ba tập của benchmark và lý do

| Tập | Thuộc | Số mẫu | Số câu | Số tên riêng | Có lịch sử | Thực hiện / hỏi lại / ngoài phạm vi | Dùng để |
|---|---|---|---|---|---|---|---|
| `asr_tune.txt` | **Tạo riêng** cho benchmark ASR, rút từ `train` | 300 | 1 024 | 123 | 194 | 231 / 46 / 23 | Dò cấu hình: chọn lựa chọn thắng ở từng bước |
| `val_seen.txt` | Tập kiểm tra **chung** của dự án | 274 | 916 | 109 | 183 | 213 / 39 / 22 | Đo cấu hình thắng, với app đã có trong train |
| `val_unseen.txt` | Tập kiểm tra **chung** của dự án | 178 | 527 | 54 | 129 | 162 / 11 / 5 | Đo cấu hình thắng, với 6 app không có trong train |

Giải thích các cột:
- **Số tên riêng:** số giá trị tham số trong đáp án được nói ra ở câu cuối (tên người, giờ, số tiền, tên kênh…), dùng
  để tính tỉ lệ bỏ sót (mục 2.3).
- **Có lịch sử:** số mẫu có hội thoại trước lượt cuối.
- **Loại đáp án:** "thực hiện" là gọi tool để làm việc; "hỏi lại" là gọi `sys.ask_missing_info` khi thiếu thông tin;
  "ngoài phạm vi" là không gọi tool nào.

`val_seen` và `val_unseen` là hai tập chung mà mọi phần của dự án (LLM, fine-tune, ASR) đều đo trên đó. Benchmark chỉ
dùng lại, không sửa. `asr_tune` là tập duy nhất tạo thêm, và không trùng mẫu nào với hai tập val.

**Cách chia** (`lifemate/eval/make_splits.py`, seed cố định 42, chạy ở máy nào cũng ra giống nhau). Bước 1–3 tạo các
tập chung của dự án, bước 4 tạo tập riêng cho benchmark:

1. **Tách app lạ trước.** Mọi mẫu dùng một trong 6 app (medication, fitness, flights, bike_rental, library,
   subscriptions), cộng 5 mẫu "ngoài phạm vi" liên quan tới các app đó được gắn tay (đáp án không gọi tool nào nên không
   tự nhận ra app), vào `val_unseen`. Lý do: bộ thi riêng của BTC có app chưa từng thấy, nên cần một tập mô phỏng điều
   đó.
2. **Gom cụm câu gần trùng.** Các mẫu có `audio_text` giống nhau từ 85% trở lên được gom thành một cụm và luôn nằm cùng
   một tập. Lý do: nếu hai câu gần giống nhau nằm ở hai tập, kết quả kiểm tra sẽ đẹp hơn thực tế.
3. **Chia 5 fold phân tầng.** 1 367 mẫu còn lại chia thành 5 phần có cùng tỉ lệ số lời gọi tool và tỉ lệ câu phải hỏi
   lại. Fold 0 là `val_seen` (274 mẫu), fold 1–4 là `train` (1 093 mẫu).
4. **Rút `asr_tune.txt` từ `train`** (`make_splits --asr-tune 300`): lấy 300 trong 1 093 mẫu train, phân tầng theo loại
   đáp án và có/không lịch sử, để tỉ lệ các loại câu giống tập train.

**Vì sao dò cấu hình trên một tập riêng lấy từ train:**
- **Không "chấm lại trên chính đề đã luyện".** Nếu vừa chọn cấu hình vừa báo kết quả trên val_seen, cấu hình được chọn
  sẽ khớp riêng với val_seen và con số báo cáo sẽ đẹp hơn thực tế. Ở đây val_seen và val_unseen chỉ được đo đúng một
  lần, sau khi đã chọn xong.
- **Không rò rỉ.** Các mô hình ASR không học trên tập train của dự án, nên lấy mẫu train để dò không làm lộ đáp án.
  Nếu sau này fine-tune ASR trên train thì phải loại 300 mẫu này ra.
- **Các tập chung không đổi.** `train`, `val_seen`, `val_unseen` giữ nguyên từng byte, các kết quả cũ vẫn so được.
- **Vì sao 300 mẫu:** đủ nhiều tên riêng (123) để so các lựa chọn, mà vẫn chạy xong 5 mô hình trong một phiên GPU T4.

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
| p95 | Như mục 0 | Hệ thống tự ngắt mỗi mẫu ở 29 giây (`deadline_s` trong `configs/v0.toml`, tính từ đầu mẫu, gồm cả ASR), chừa 1 giây dưới giới hạn 30 giây của thể lệ. Phần lớn thời gian đó dành cho LLM |

**Hàm mục tiêu** (để xếp hạng lựa chọn, nhỏ hơn là tốt hơn), giống nhau cho mọi mô hình:

1. Lựa chọn có p95 > 6 s xếp sau mọi lựa chọn đạt ngưỡng. Ngưỡng mềm vì large-v3 vượt 6 s ở mọi lựa chọn; loại hẳn thì
   không còn gì để so.
2. Bỏ sót tên riêng + tỉ lệ câu rỗng.
3. Nếu hòa, WER bỏ số.

## 3. Bộ cấu hình chung

**Mục đích.** Muốn biết mô hình nào tốt hơn thì mọi thứ khác ngoài mô hình phải giống nhau. Ví dụ, nếu mô hình A chạy
trên GPU mạnh còn mô hình B chạy trên GPU yếu, thì B chậm hơn có thể chỉ vì máy. Nếu A bật bộ lọc còn B tắt, thì A ít
câu bịa hơn có thể chỉ vì bộ lọc. Lần so sánh trước (sweep v1, mục 1) mắc đúng lỗi này. Bảng dưới liệt kê mọi thứ
được giữ cố định, gọi là "bộ cấu hình chung".

Phần giữ giống hệt cho mọi mô hình:

| Mục | Giá trị | Lý do |
|---|---|---|
| Câu được đo | Mọi lượt người dùng trong lịch sử và lượt cuối | LLM đọc chữ của mọi lượt |
| Phần cứng | Một phiên Kaggle T4 cho mỗi lần chạy, `dtype = "auto"` (float16 trên T4; riêng engine NeMo là float32) | Thời gian so được với nhau |
| Giải mã | Tất định: `temperature = 0` (trừ lựa chọn nhiệt độ thử lại) | Chạy lại cho cùng kết quả |
| Bộ lọc ảo giác | Bật | Chạy sau ASR, không phụ thuộc engine, nên áp giống hệt cho mọi mô hình |
| `condition_on_previous_text` | `false` | Mỗi câu nghe độc lập, câu bịa không lây sang câu sau |
| Code đo | `lifemate/eval/eval_asr.py` | Cùng cách chuẩn hóa chữ và tính chỉ số |

Giải thích thêm từng dòng:

- **Câu được đo.** Không chỉ đo câu cuối mà đo cả các câu trong lịch sử, vì LLM đọc lại toàn bộ hội thoại để hiểu yêu
  cầu. Một câu trong lịch sử bị chép sai ("chị Lan" thành "chị lán") có thể làm LLM gọi sai người ở câu cuối.
- **Phần cứng và `dtype`.** Cả năm mô hình chạy nối tiếp trong cùng một phiên Kaggle với cùng một GPU T4, nên thời gian
  p95 của các mô hình so được với nhau. `dtype` là độ chính xác của phép tính: `"auto"` để engine tự chọn, trên T4 là
  float16 (số thực 16 bit, nhanh hơn và tốn ít bộ nhớ hơn float32). Đây cũng là giá trị dùng khi nộp bài. Ngoại lệ:
  engine NeMo (Parakeet) chọn float32 cho `"auto"`, vì float16 làm đầu ra của mô hình này thành NaN (mục 6.5).
  Parakeet được chạy lại trong một phiên T4 khác, nên thời gian của Parakeet so với các mô hình khác chỉ gần đúng. Mỗi lần đưa
  vào mô hình 8 câu cùng lúc (`batch_size = 8`), âm thanh được đọc ở 16 000 mẫu mỗi giây (`sample_rate = 16000`).
- **Giải mã tất định.** `temperature` là độ ngẫu nhiên khi chọn chữ. Đặt `0` thì mô hình luôn chọn chữ có xác suất cao
  nhất, nên chạy lại bao nhiêu lần cũng ra cùng một chữ. Nếu có ngẫu nhiên, hai lần chạy cùng cấu hình có thể chênh
  nhau, và không biết chênh lệch giữa hai cấu hình là thật hay do may rủi. Ngoại lệ duy nhất là lựa chọn "nhiệt độ thử
  lại" ở bước 4 (mục 4): ngẫu nhiên chỉ được dùng khi lần giải mã đầu tiên bị hỏng.
- **Bộ lọc ảo giác.** Bộ lọc chạy trên chữ đầu ra, sau khi mô hình chép xong, nên không cần biết mô hình nào sinh ra
  chữ đó. Vì vậy có thể bật giống hệt cho cả năm mô hình. Bộ lọc cũng được bật khi nộp bài, nên đo với bộ lọc bật mới
  phản ánh đúng thứ LLM sẽ nhận.
- **`condition_on_previous_text = false`.** Khi bật, Whisper dùng chữ của đoạn trước làm câu mồi cho đoạn sau. Nếu đoạn
  trước bị bịa ("Hãy đăng ký kênh…"), đoạn sau dễ bịa tiếp. Tắt đi thì mỗi câu được nghe riêng, lỗi không lan sang câu
  khác. Tham số này chỉ có ở Whisper; các mô hình CTC/RNNT vốn đã nghe từng câu riêng nên bỏ qua.
- **Code đo.** Cả năm mô hình được chấm bằng cùng một script, cùng cách chuẩn hóa chữ (chữ thường, bỏ dấu câu) và cùng
  công thức tính chỉ số. Nhờ vậy, chênh lệch chỉ số đến từ chữ mô hình chép ra, không đến từ cách chấm.

Các tham số **không** nằm trong bảng trên (cách giải mã, câu mồi, chống lặp, VAD) thì mỗi mô hình được dò riêng, ở mục 4.
Lý do: mỗi mô hình có điểm mạnh và tính năng khác nhau; ép tất cả dùng một cấu hình giống nhau sẽ thiệt cho mô hình nào
không hợp với cấu hình đó. Vì vậy mỗi mô hình được dò với cùng cách làm và cùng số bước, rồi mới đem cấu hình tốt nhất
của từng mô hình ra so.

## 4. Mô hình và cấu hình riêng

Mục này gồm ba phần: năm mô hình được so là gì (4.1), cách dò cấu hình riêng cho từng mô hình (4.2), và những điều
chỉnh kỹ thuật cần làm để chạy được từng mô hình (4.3).

### 4.1. Năm mô hình

| Mô hình | Engine | Tham số | Giấy phép, ngày | Revision |
|---|---|---|---|---|
| `Systran/faster-whisper-large-v3` | `faster_whisper` (CTranslate2) | 1,55B | MIT | `edaa852e…` |
| `vinai/PhoWhisper-medium` | `hf_seq2seq` (transformers) | 0,77B | BSD-3 | `55a7e3eb…` |
| `khanhld/chunkformer-ctc-large-vie` | `chunkformer` | 0,11B | CC-BY-NC-4.0 (phi thương mại), 02/2025 | `311fc035…` |
| `khanhld/chunkformer-rnnt-large-vie` | `chunkformer` | 0,11B | CC-BY-4.0, 10/2025 | `9863592e…` |
| `nvidia/parakeet-ctc-0.6b-Vietnamese` | `nemo_ctc` (NeMo 2.6.2) | 0,6B | NVIDIA Open Model License, 01/2026 | `b0493142…` |

Giải thích các cột:
- **Engine:** thư viện dùng để chạy mô hình. Cùng một mô hình có thể chạy bằng nhiều thư viện; bảng ghi thư viện đã dùng.
- **Tham số:** kích thước mô hình, B là tỉ (billion). Mô hình lớn hơn thường nghe tốt hơn nhưng chậm hơn và tốn bộ nhớ
  hơn. Ở đây hai ChunkFormer nhỏ hơn large-v3 khoảng 14 lần.
- **Giấy phép:** điều kiện được phép dùng mô hình. MIT, BSD-3, CC-BY-4.0 cho dùng tự do, chỉ cần ghi nguồn.
  CC-BY-NC-4.0 cấm dùng thương mại.

Vì sao chọn năm mô hình này:
- **large-v3:** mô hình đang dùng trong config hiện tại, làm mốc so sánh.
- **PhoWhisper-medium:** Whisper được huấn luyện thêm trên tiếng Việt; kiểm tra xem huấn luyện thêm có giúp nghe tên
  riêng tiếng Việt tốt hơn không.
- **ChunkFormer CTC, ChunkFormer RNNT, Parakeet:** mô hình không phải Whisper, được huấn luyện riêng cho tiếng Việt.
  Kiểu mô hình CTC/RNNT không "viết tiếp câu" như Whisper, nên được kỳ vọng không bịa câu phụ đề (vấn đề 1 ở mục 1).

Revision là mã phiên bản cố định trên Hugging Face, để chạy lại luôn tải đúng bản đã đo. Cả năm mô hình phát hành
trước 26/10/2026; cộng với LLM 4,02B đều dưới giới hạn 6B của cuộc thi. ChunkFormer CTC có giấy phép phi thương mại,
cần BTC xác nhận trước khi dùng để nộp bài.

### 4.2. Cách dò cấu hình cho từng mô hình

**Ý tưởng.** Mỗi mô hình có nhiều "núm vặn" (tham số) ảnh hưởng tới chữ chép ra. Benchmark vặn lần lượt từng núm trên
tập `asr_tune`, giữ vị trí tốt nhất của mỗi núm, rồi mới sang núm tiếp theo. Giống như chỉnh một chiếc radio: chỉnh
tần số cho rõ tiếng trước, rồi mới chỉnh âm lượng, rồi chỉnh âm trầm. "Tốt nhất" được quyết định bằng hàm mục tiêu ở
mục 2.3: ưu tiên đạt ngưỡng thời gian, rồi ít bỏ sót tên riêng và ít câu rỗng nhất.

**Các bước dò.** Dò lần lượt từng bước; mỗi bước giữ lựa chọn thắng của các bước trước. Cách này tốn ít lần chạy hơn
thử mọi tổ hợp (large-v3 chỉ cần 8 lần đo thay vì 2 × 4 × 2 × 3 = 48). Lựa chọn engine không hỗ trợ thì bỏ qua.

Với mỗi bước, lựa chọn đầu tiên là "giữ nguyên" (không bật gì thêm; ở bước 1 là giải mã tham lam). Nếu không lựa chọn nào tốt hơn thì giữ
nguyên, nghĩa là chỉ bật thêm một tính năng khi tính năng đó thật sự cải thiện kết quả.

| Bước            | Lựa chọn                                                                                                                       | Lý do thử                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| 1. Giải mã      | tham lam, beam 5                                                                                                               | Beam xem nhiều phương án hơn nhưng chậm hơn và dễ chọn câu "thuộc lòng" có xác suất cao                    |
| 2. Câu mồi      | không; câu của lead (dài, liệt kê tên riêng); câu ngắn giống lời nói ("Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."); `hotwords` | Câu mồi giúp viết đúng tên riêng nhưng có thể gây bịa; sweep v2 cho thấy câu ngắn viết đúng Zalo nhiều hơn |
| 3. Chống lặp    | tắt; `repetition_penalty = 1.2` + `no_repeat_ngram_size = 3`                                                                   | Có trong cấu hình của lead; có thể làm sai âm tiết lặp hợp lệ trong tiếng Việt ("từ từ", "năm năm")        |
| 4. Riêng engine | không; VAD (cắt khoảng lặng ≥ 500 ms); nhiệt độ thử lại `[0; 0,2; …; 1]`                                                       | Chỉ engine có tính năng đó mới thử                                                                         |

Giải thích thêm từng bước:

1. **Giải mã.** Tham lam chọn ngay chữ có xác suất cao nhất ở mỗi bước, nhanh nhưng có thể đi sai hướng từ sớm. Beam 5
   giữ 5 câu đang dở dang cùng lúc và cuối cùng chọn câu có tổng xác suất cao nhất, thường chính xác hơn nhưng chậm
   hơn vài lần. Với Whisper, beam còn có mặt trái: câu phụ đề quen thuộc ("Hãy đăng ký kênh…") có xác suất tổng rất
   cao, nên beam dễ chọn đúng câu đó.
2. **Câu mồi.** Whisper đọc câu mồi như thể đó là câu ngay trước đoạn ghi âm, nên câu mồi có chữ "Zalo" thì Whisper dễ
   viết "Zalo" hơn. Ba kiểu được thử: câu mồi dài dạng liệt kê của lead, câu mồi ngắn giống lời nói thật, và
   `hotwords` (danh sách từ khóa). Rủi ro: câu mồi lạ có thể đẩy Whisper sang văn phong khác, như phụ đề video.
3. **Chống lặp.** Khi bị "kẹt", mô hình có thể lặp một cụm từ nhiều lần. `repetition_penalty = 1.2` giảm xác suất của
   chữ đã xuất hiện; `no_repeat_ngram_size = 3` cấm lặp lại bất kỳ cụm 3 từ nào. Rủi ro: tiếng Việt có lặp hợp lệ, như
   "từ từ" hay số điện thoại "chín chín chín", và chống lặp có thể làm sai những chỗ đó.
4. **Riêng engine.** VAD cắt đoạn im lặng trước khi nhận dạng, vì Whisper hay bịa chữ trên đoạn im lặng. Nhiệt độ thử
   lại giải mã lại câu khi thấy dấu hiệu hỏng (lặp nhiều, xác suất quá thấp), mỗi lần tăng độ ngẫu nhiên thêm 0,2.

Câu mồi của lead: `"LifeMate: trợ lý ảo tiếng Việt điều khiển smart home, tivi, đặt xe, ngân hàng, HTV7, VTV1, VTV3,
deadline, voucher, Zalo."`

Ví dụ đường dò của large-v3: bước 1 so tham lam với beam 5, giữ tham lam. Bước 2 thử 4 kiểu câu mồi (đều kèm tham
lam), không kiểu nào tốt hơn nên giữ "không câu mồi". Bước 3 thử bật chống lặp, không tốt hơn nên giữ tắt. Bước 4 thử
VAD và nhiệt độ thử lại, nhiệt độ thử lại tốt nhất nên giữ. Kết quả đầy đủ ở mục 6.1.

Không phải mô hình nào cũng có đủ các núm vặn trên. Một lựa chọn chỉ được thử khi engine của mô hình hỗ trợ; nếu
không, bước đó chỉ còn lựa chọn "giữ nguyên". Vì vậy số lần đo của mỗi mô hình khác nhau, nhưng cách dò là một.

**Lựa chọn mỗi mô hình được thử, và lý do:**

| Mô hình | Bước 1 | Bước 2 | Bước 3 | Bước 4 | Lý do |
|---|---|---|---|---|---|
| large-v3 | tham lam, beam 5 | không, lead, ngắn, `hotwords` | tắt, bật | không, VAD, nhiệt độ thử lại | faster-whisper có đủ mọi tham số |
| PhoWhisper | tham lam, beam 5 | không, lead, ngắn | tắt, bật | không | transformers có beam, câu mồi (`prompt_ids`), chống lặp; không có VAD, `hotwords`. Nhiệt độ thử lại của transformers cần thêm ngưỡng riêng nên không thử, để giống cách tính của faster-whisper |
| ChunkFormer CTC | tham lam | không | tắt | không | Gói `chunkformer` chỉ giải mã tham lam; mô hình CTC không có câu mồi hay chống lặp |
| ChunkFormer RNNT | tham lam | không | tắt | không | Như trên (cùng gói) |
| Parakeet CTC | tham lam | không | tắt | không | Beam + mô hình ngôn ngữ KenLM đi kèm repo không thử được: gói flashlight-text không build được trên Python 3.13 của Kaggle |

Hệ quả: ba mô hình CTC/RNNT chỉ có một cấu hình, nên kết quả của chúng là kết quả "dùng ngay, không chỉnh". Hai
mô hình Whisper được chỉnh nhiều hơn mà vẫn đứng sau (mục 6), nên kết luận không bị thiên về phía mô hình được chỉnh
kỹ hơn.

### 4.3. Điều chỉnh kỹ thuật theo engine

Các điều chỉnh dưới đây không đổi cách mô hình nghe, chỉ để chạy được mô hình trong dự án và giữ đúng bộ cấu hình chung.

- **PhoWhisper không chuyển sang CTranslate2** (định dạng của faster-whisper), vì chuyển định dạng có thể làm kết quả
  lệch so với bản gốc. Thay vào đó, engine transformers nhận cùng tên tham số với faster-whisper và tự dịch
  (`hf_generate_kwargs` trong `lifemate/backends/asr.py`): `beam_size` thành `num_beams`, `initial_prompt` thành
  `prompt_ids`, `condition_on_previous_text` thành `condition_on_prev_tokens`. `best_of` bị bỏ vì chỉ có tác dụng khi
  giải mã ngẫu nhiên.
- **ChunkFormer:** gói tự tải mô hình nhưng bỏ qua revision, nên engine tải đúng revision trước khi nạp. Tham số chia
  đoạn giữ mặc định (`chunk_size = 64`, ngữ cảnh trái/phải 128), không dò, để số lần chạy ngang với các mô hình kia.
- **Parakeet:** không dùng hàm `transcribe` có sẵn vì bộ nạp dữ liệu của NeMo lỗi với torch 2.13 trên Kaggle. Engine
  tự đọc WAV, gọi thẳng mô hình rồi bộ giải mã CTC. NeMo hạ phiên bản transformers, nên Parakeet đo sau cùng. Với
  `dtype = "auto"`, engine chạy float32 (sửa ngày 29/10, mục 6.5); nếu chọn float16 tường minh, câu nào ra NaN được
  tự chạy lại ở float32.

## 5. Phương pháp thử nghiệm

**Tổng quan.** Benchmark chạy qua ba giai đoạn, nối tiếp nhau:

```text
Chạy thử trên CPU  ──►  Dò cấu hình trên GPU  ──►  Đo cấu hình thắng
(3 mẫu, bắt lỗi)        (asr_tune, 300 mẫu)        (val_seen + val_unseen)
```

Mọi giai đoạn chạy trên Kaggle bằng kernel riêng tư. Kernel là một phiên máy ảo: tự tải mã nguồn của dự án từ GitHub,
gắn dữ liệu thi (dataset riêng tư trên Kaggle), cài thư viện, chạy lệnh rồi lưu kết quả để tải về.

1. **Chạy thử trên CPU** (kernel `quanlam/lifemate-asr-preflight`): cài đúng thư viện như kernel GPU, chạy mọi lựa
   chọn của từng mô hình trên 3 mẫu, để bắt lỗi cài đặt và nạp mô hình trước khi tốn giờ GPU. Lần 1 Parakeet lỗi mọi
   lựa chọn và flashlight-text không build được; sau khi sửa engine và bỏ lựa chọn KenLM, lần 2 mọi mô hình chạy được.
2. **Dò cấu hình trên GPU** (kernel `quanlam/lifemate-asr-tune`): mỗi mô hình chạy `python -m lifemate.eval.asr_tune`
   theo các bước ở mục 4 trên `asr_tune.txt`. Lựa chọn "giữ nguyên" dùng lại kết quả đã đo.
3. **Đo cấu hình thắng** của mỗi mô hình đúng một lần trên val_seen và val_unseen.

Giải thích thêm từng giai đoạn:

1. **Vì sao chạy thử trên CPU trước.** Kaggle chỉ cho mỗi tài khoản một số giờ GPU mỗi tuần, và một lần dò đầy đủ mất
   khoảng 4 giờ. Nếu một mô hình lỗi lúc cài đặt hay lúc nạp, lỗi đó chỉ lộ ra sau vài giờ chạy các mô hình trước, và
   phải chạy lại từ đầu. Chạy thử trên CPU (không tốn giờ GPU) với 3 mẫu cho mỗi lựa chọn bắt được các lỗi này trong vài
   phút. Kết quả chạy thử chỉ để biết "chạy được hay không", không dùng để so sánh.
2. **Dò cấu hình.** Script `lifemate/eval/asr_tune.py` làm theo cách ở mục 4.2. Với mỗi lựa chọn, script ghi một file
   config, gọi `lifemate/eval/eval_asr.py` để chép toàn bộ 300 mẫu và tính chỉ số, rồi dùng hàm mục tiêu để chọn. Một
   cấu hình đã đo thì không đo lại: lựa chọn "giữ nguyên" ở bước sau chính là cấu hình thắng ở bước trước. Tổng cộng có
   16 lần đo trên tập dò: large-v3 8 lần, PhoWhisper 5 lần, ba mô hình còn lại mỗi mô hình 1 lần.
3. **Đo cấu hình thắng.** Cấu hình thắng của mỗi mô hình được chạy đúng một lần trên val_seen và một lần trên
   val_unseen, tổng 10 lần đo. Không có lựa chọn nào được đổi sau khi xem kết quả trên hai tập này.

**Thứ tự chạy.** Năm mô hình chạy nối tiếp trong cùng một phiên: large-v3, PhoWhisper, ChunkFormer CTC, ChunkFormer
RNNT, Parakeet. Thư viện riêng của ChunkFormer và Parakeet chỉ được cài ngay trước khi đo mô hình đó, vì các thư viện
này thay đổi phiên bản của thư viện dùng chung (Parakeet hạ phiên bản transformers). Cài sớm có thể làm hỏng các mô hình
đo trước.

Tái tạo: `make splits` (sinh `asr_tune.txt`), rồi `kaggle kernels push -p ops/kaggle/asr_tune --accelerator
NvidiaTeslaT4`, hoặc trên máy có GPU: `python -m lifemate.eval.asr_tune --env <profile> --out <thư mục>`.

## 6. Kết quả

Một phiên T4, khoảng 4 giờ. Mọi mô hình chạy hết, không lần đo nào lỗi. Log có một cảnh báo hết bộ nhớ GPU ở lựa chọn
beam 5 của PhoWhisper, nhưng lần đo đó vẫn hoàn tất.

### 6.1. Đường dò trên tập dò (300 mẫu, 1 024 câu, 123 tên riêng)

Bảng này là "nhật ký" của giai đoạn dò: mọi lần đo trên tập dò, theo đúng thứ tự các bước ở mục 4.2. Mỗi dòng là
một lần đo với cấu hình gồm lựa chọn ở dòng đó cộng các lựa chọn đã giữ ở các bước trước. Các cột:
- **Bỏ sót:** tỉ lệ tên riêng bị chép sai, trên 123 tên riêng của tập dò. Càng thấp càng tốt.
- **Rỗng:** số câu (trên 1 024) mà mô hình không ra chữ nào sau bộ lọc.
- **WER, WER bỏ số:** tỉ lệ từ sai, như mục 2.3.
- **Bịa:** số câu có cụm phụ đề bịa bị bộ lọc phát hiện.
- **p95 (s):** 95% số mẫu chép xong trong số giây này. Ngưỡng là 6 s.

Một số dòng trùng số liệu với nhau, ví dụ "tham lam" ở bước 1 và "không" ở bước 2 của large-v3. Đó là cùng một cấu
hình được dùng lại, không đo lại (mục 5).

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
| Parakeet (v4) | (một cấu hình) | tham lam | 59,3% | 0 | 31,4% | 31,7% | 0 | 2,04 |

Cách đọc: với large-v3, bước 1 chọn tham lam vì beam 5 cho nhiều câu rỗng hơn (31 so với 10). Ở bước 2, không câu mồi
nào tốt hơn "không câu mồi". Bước 4 chọn nhiệt độ thử lại vì bỏ sót thấp nhất; mọi lựa chọn của large-v3 đều vượt 6 s
nên chỉ so với nhau.

Giải thích chi tiết hơn:

- **Cách chọn ở mỗi bước** là so tổng "bỏ sót + tỉ lệ câu rỗng" (mục 2.3). Ví dụ bước 4 của large-v3: "không" là
  41,5% + 10/1 024 ≈ 42,5%; VAD là 41,5% + 8/1 024 ≈ 42,3%; nhiệt độ thử lại là 40,7% + 11/1 024 ≈ 41,8%. Nhiệt độ
  thử lại thấp nhất nên được giữ.
- **large-v3 vượt ngưỡng ở mọi lần đo** (p95 từ 6,7 đến 8,3 s), nên các lựa chọn của large-v3 chỉ được so với nhau theo
  độ chính xác. Thời gian của cùng một mô hình dao động khoảng 1,5 s giữa các lần đo, nên chênh lệch p95 nhỏ giữa hai
  lựa chọn của large-v3 chưa chắc do lựa chọn đó gây ra.
- **Câu mồi với large-v3:** cả ba kiểu đều kém hơn không câu mồi. Câu mồi của lead tăng WER từ 30,8% lên 48,6% và câu
  rỗng từ 10 lên 22. `hotwords` ít câu rỗng nhất (4) nhưng bỏ sót tới 59,3%, vì Whisper chép sai cả câu chứ không chỉ
  tên riêng.
- **Beam 5:** không giúp mô hình nào. Với large-v3, beam tăng câu bịa từ 17 lên 39. Với PhoWhisper, beam giảm WER khoảng
  0,5 điểm mà p95 tăng từ 3,2 lên 9,2 s.
- **Câu mồi với PhoWhisper:** câu mồi ngắn giảm bỏ sót từ 48,0% xuống 46,3%, tức khoảng 2 tên riêng trên 123. Mức cải
  thiện này nhỏ, nhưng là lựa chọn tốt nhất theo hàm mục tiêu nên được giữ.
- **Chống lặp:** không giúp mô hình nào, còn làm WER của PhoWhisper tăng 1,5 điểm.
- **Ba mô hình CTC/RNNT** chỉ có một dòng vì không có tham số nào để dò (mục 4.2). Hai ChunkFormer đã tốt hơn mọi cấu
  hình của hai mô hình Whisper ngay trên tập dò: bỏ sót 37–39% so với 41–59%, WER 20–21% so với 28–49%.

### 6.2. Cấu hình thắng của mỗi mô hình

Cấu hình cuối cùng mà mỗi mô hình mang sang giai đoạn đo (mục 5, giai đoạn 3). Config đầy đủ nằm trong
`<mô hình>__report_val_seen.toml` (xem đầu báo cáo).

| Mô hình | Cấu hình thắng |
|---|---|
| large-v3 | tham lam, không câu mồi, không chống lặp, nhiệt độ thử lại `[0.0, 0.2, …, 1.0]` |
| PhoWhisper | tham lam, câu mồi ngắn `"Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."`, không chống lặp |
| ChunkFormer CTC, ChunkFormer RNNT, Parakeet | tham lam, tham số mặc định |

### 6.3. Đo trên tập kiểm tra

Đây là kết quả chính để so sánh các mô hình: cấu hình thắng của mỗi mô hình, đo một lần trên hai tập kiểm tra mà giai
đoạn dò chưa từng thấy. val_seen có 916 câu và 109 tên riêng; val_unseen có 527 câu và 54 tên riêng. Hai cột cuối đếm
số câu viết đúng "Zalo" và "VTV" trên số câu đáp án có tên đó. Giá trị tốt nhất mỗi cột được in đậm.

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
| Parakeet (v4) | val_seen | 50,5% | 0 | 31,6% | 32,0% | 0 | 2,00 | 5/19 | 1/13 |
| | val_unseen | 77,8% | 0 | 31,1% | 31,8% | 0 | 1,92 | 0/6 | 1/1 |

Cách đọc bảng:

- **Hai tập cho kết quả khác nhau ở bỏ sót, không khác ở WER.** WER của mỗi mô hình gần như bằng nhau trên hai tập
  (chênh dưới 1 điểm), nghĩa là độ khó âm thanh của hai tập tương đương. Nhưng bỏ sót trên val_unseen cao hơn hẳn (ví
  dụ ChunkFormer CTC: 23,9% lên 55,6%). Lý do: val_unseen dùng 6 app không có trong train, nên tên riêng ở đó là tên
  thuốc, chuyến bay, sách… lạ hơn và khó nghe đúng hơn.
- **So với tập dò,** bỏ sót trên val_seen thấp hơn ở hầu hết mô hình (large-v3: 40,7% xuống 31,2%). Tập dò có tỉ
  lệ tên riêng khác nên con số giữa hai tập không so trực tiếp được. Thứ hạng giữa các mô hình mới là thứ cần so.
- **Số câu chứa tên app rất ít,** nhất là trên val_unseen (Zalo 6 câu, VTV 1 câu). Một câu đúng hay sai đã đổi tỉ lệ
  đáng kể, nên hai cột này chỉ để tham khảo.
- **Parakeet** là kết quả lần chạy lại v4, sau khi sửa lỗi float16 (mục 6.5). Lần v3 có 37 và 51 câu ra ký tự `⁇`;
  lần v4 không còn câu nào.

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

**Ví dụ 3. Câu mồi của lead làm mất trắng câu nói** (tập dò, large-v3 với câu mồi của lead).

| | Chữ |
|---|---|
| Đáp án | Vàng SJC hôm nay giá cả sao rồi? |
| large-v3, trước lọc | Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn |
| large-v3, sau lọc | (rỗng) |

Câu mồi dài, dạng liệt kê, trông giống phần mô tả video, nên Whisper tiếp tục theo kiểu phụ đề YouTube. Bộ lọc xóa
hết, lượt đó mất trắng. Câu mồi của lead làm câu rỗng tăng từ 10 lên 22 trên tập dò.

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

**Ví dụ 6. Parakeet ra ký tự lạ ở lần chạy v3, và sau khi sửa.**

| | Chữ |
|---|---|
| Đáp án | Gần đây có quán pizza nào không? |
| Parakeet v3 (float16) | ⁇ |
| Parakeet v4 (float32) | Gần đây có quán biza nào không? |

Ở lần v3, mọi câu "rỗng" của Parakeet (37 trên val_seen, 51 trên val_unseen) đều chỉ có đúng một ký tự `⁇`, kể cả câu
ngắn và rõ như câu trên. `⁇` là ký hiệu "token không biết" của bộ tách từ. Nguyên nhân và cách sửa ở mục 6.5.

Sau khi sửa, Parakeet vẫn hay dính hoặc mất chữ, ví dụ "Tài khoản của anh còn được bao nhiêu?" thành "Tài khoản của
ăn gọn l được bao nhiêu?". Lỗi này có ở cả float16 lẫn float32, nên là cách mô hình tự chép, không phải lỗi engine.

### 6.5. Sửa lỗi Parakeet và chạy lại

**Triệu chứng (lần chạy v3).** 70/1 024 câu của tập dò, 37/916 câu của val_seen và 51/527 câu của val_unseen chỉ ra
ký tự `⁇`. Các câu này rải rác ở mọi độ dài và mọi kiểu câu.

**Chẩn đoán** (kernel `quanlam/lifemate-asr-diag`, 29/10). Lấy 42 mẫu có câu hỏng (216 câu) ở lần v3, chạy lại với
bốn cách:

| Cách chạy | Câu ra `⁇` | Đầu ra có NaN | WER | p95 (s) |
|---|---|---|---|---|
| float16, batch 8 (như v3) | 70 | có (49 batch) | 56,1% | 4,54 |
| float32, batch 8 | 0 | không | 28,9% | 0,39 |
| float16, batch 1 | 69 | có (69 câu) | 55,8% | 4,00 |
| float32, batch 1 | 0 | không | 28,9% | 0,64 |

Kết luận: lỗi do **float16 tràn số**, không do đệm batch. Ở float16, phép tính trong mô hình vượt giới hạn của kiểu số
16 bit, đầu ra thành NaN ("không phải số"), và bước chọn chữ rơi vào token số 0 là token không biết. Chạy float32 thì
hết lỗi, và còn nhanh hơn float16 trên T4.

**Cách sửa** (`lifemate/backends/asr.py`, lớp `NemoCTCASR`; áp cho mọi mô hình chạy bằng engine NeMo, không riêng
Parakeet):
- `dtype = "auto"` chạy float32.
- Nếu chọn float16 hoặc bfloat16 tường minh, câu nào có đầu ra NaN hoặc inf được tự chạy lại ở float32, kèm cảnh báo
  `asr_nonfinite_output` trong log.
- `eval_asr` có thêm chỉ số `n_unk` (số câu chỉ gồm `⁇`), để lỗi kiểu này không lẫn vào "câu rỗng" nữa.

**Chạy lại (lần v4).** Cùng quy trình như mục 5: cùng tập dò, cùng các bước dò, rồi đo một lần trên val_seen và
val_unseen. Không còn câu `⁇` nào (`n_unk = 0`) và log không có cảnh báo NaN. Bốn mô hình còn lại không chạy lại.

| Parakeet | Tập | Bỏ sót | Rỗng | WER | p95 (s) |
|---|---|---|---|---|---|
| v3 (float16, không hợp lệ) | val_seen | 54,1% | 37 | 35,3% | 0,87 |
| v4 (float32) | val_seen | 50,5% | 0 | 31,6% | 2,00 |
| v3 (float16, không hợp lệ) | val_unseen | 79,6% | 51 | 38,2% | 0,74 |
| v4 (float32) | val_unseen | 77,8% | 0 | 31,1% | 1,92 |

p95 của v4 cao hơn v3, vì ở v3 các câu hỏng kết thúc sớm (chỉ ra một token), và vì v4 chạy trong một phiên T4 khác.

## 7. Phân tích và đánh giá

- **ChunkFormer RNNT tốt nhất xét tổng thể.** WER thấp nhất trên cả ba tập, khoảng 20%. Trên val_unseen, bỏ sót ít
  nhất (48,1%, bằng PhoWhisper) và viết đúng "Zalo" nhiều nhất (17/19 và 5/6). Không có câu rỗng hay câu bịa kiểu
  phụ đề, p95 2,5 s, chưa tới một nửa ngưỡng. Giấy phép CC-BY-4.0. Có một lỗi lặp trên val_seen: một câu đọc số điện
  thoại ra chữ "chín" lặp khoảng 250 lần; bộ lọc gộp phần lặp lại nên chữ cuối chỉ còn sai vài từ.
- **ChunkFormer CTC bỏ sót ít nhất trên val_seen (23,9%), nhưng kém trên val_unseen (55,6%).** Gần như luôn sai tên
  app (Zalo 5/19 và 0/6, ví dụ 4). Khoảng cách giữa hai tập cho thấy lợi thế trên val_seen đến từ vài loại tên riêng
  hay gặp. Giấy phép phi thương mại.
- **large-v3 (cấu hình hiện tại) vượt ngưỡng thời gian ở mọi lựa chọn**, p95 khoảng 8 s. WER khoảng 30%, cao hơn
  ChunkFormer 10 điểm, vẫn còn câu bịa dù đã bật bộ lọc (ví dụ 1), gần như không bao giờ viết đúng "Zalo" (1/19). Câu
  mồi và `hotwords` đều làm kết quả kém hơn (ví dụ 3), khớp với sweep v2.
- **PhoWhisper không bịa và nhanh hơn large-v3** (p95 khoảng 3 s). Câu mồi ngắn giúp viết đúng "Zalo" trên val_seen
  (16/19), nhưng bỏ sót vẫn cao (38–48%) vì sai nhiều ở tên riêng khác. Beam 5 gần như không cải thiện mà chậm gấp ba.
- **Parakeet, sau khi sửa lỗi float16, vẫn kém hai ChunkFormer.** Không còn câu rỗng, WER giảm từ 35–38% xuống khoảng
  31%, ngang large-v3. Nhưng bỏ sót tên riêng cao nhất trong năm mô hình (50,5% và 77,8%), tên app gần như luôn sai
  (Zalo 5/19 và 0/6), và chữ hay dính hoặc mất (ví dụ 6). Mô hình nhanh (p95 khoảng 2 s) và không bịa câu phụ đề,
  nhưng không có lợi thế nào bù được độ chính xác.
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
   PhoWhisper. Lần 2 và lần 3 tiếp tục từ kết luận này (phần "Sau lần 1" ở đầu báo cáo).
4. Bỏ Parakeet: đã chạy lại sau khi sửa lỗi float16 (mục 6.5), vẫn kém hai ChunkFormer ở mọi chỉ số chính xác.
   ChunkFormer CTC chỉ xét lại nếu BTC cho dùng giấy phép phi thương mại
   và điểm thi cho thấy lợi thế trên val_seen là thật.

## Giới hạn đã biết

- Thời gian đo trên T4, chưa phải GPU A30 lúc chấm.
- Tập dò có ít câu chứa từng tên riêng, nên chênh vài câu đã là khác biệt lớn.
- Bỏ sót tên riêng so không dấu và khớp nguyên từ: "zalo" và "Zalo" được tính là đúng, nhưng "dalo" bị tính là sai.
- Nếu sau này fine-tune ASR trên train, phải loại `asr_tune.txt` khỏi dữ liệu train ASR.
