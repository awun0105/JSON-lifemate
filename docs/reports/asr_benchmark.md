# Báo cáo so sánh mô hình và cấu hình ASR

Ngày: 28/10/2026. Trạng thái: **đang chạy**. Đã có phần vấn đề, cấu hình và phương pháp; phần kết quả, phân tích và
kết luận điền sau khi kernel `quanlam/lifemate-asr-tune` chạy xong.

Báo cáo chỉ ghi số liệu tổng hợp. Không trích câu nói, id mẫu hay chữ chép lại từ dữ liệu thi (dữ liệu riêng tư).
Kết quả thô nằm trong `artifacts/runs/kaggle/` (không có trên git).

## 1. Vấn đề

ASR (nhận dạng giọng nói) là bước đầu của pipeline: chữ sai thì LLM gọi sai tool hoặc sai tham số, và mẫu đó mất
điểm. Các lần đo trước để lại ba vấn đề:

1. **Cấu hình của team lead cho nhiều câu rỗng.** whisper-large-v3 với câu mồi (`initial_prompt`) và hai tham số
   chống lặp cho WER 44,0%, 190/916 câu rỗng, bỏ sót 52% thực thể trên val_seen (ASR sweep v1). Đếm lại trên chữ
   trước bộ lọc: 160/185 câu rỗng là câu Whisper bịa kiểu phụ đề YouTube ("Hãy subscribe cho kênh…") rồi bị bộ lọc
   xóa, trên audio có độ dài và độ to bình thường.
2. **Bỏ câu mồi thì sai tên riêng.** Không câu mồi: WER 29,7%, câu rỗng còn 24, nhưng "Zalo" chỉ viết đúng 2/19 câu
   (có câu mồi: 13/19). Tên app và tên kênh quyết định tool nào được gọi.
3. **So sánh trước đó không công bằng.** Sweep v1 so PhoWhisper chạy giải mã tham lam, không VAD, tắt bộ lọc, với
   large-v3 chạy beam 5, có VAD, bật bộ lọc. Chênh lệch không tách được là do mô hình hay do cấu hình.

Câu hỏi của lần đo này: **mô hình nào, với cấu hình nào, cho chữ ASR tốt nhất cho bài toán**, khi mỗi mô hình được
chỉnh với cùng công sức và đo trên cùng dữ liệu.

## 2. Bộ cấu hình chung

Phần giữ giống hệt cho mọi mô hình:

| Mục | Giá trị | Lý do |
|---|---|---|
| Dữ liệu chỉnh | `configs/splits/asr_tune.txt`, 300 mẫu lấy từ train, phân tầng theo loại câu trả lời và có/không lịch sử | ASR không học trên train nên không rò; nhiều câu có tên riêng hơn val_seen; các split chính không đổi |
| Dữ liệu báo cáo | `val_seen` (274 mẫu) và `val_unseen` (178 mẫu), đo đúng một lần cho cấu hình thắng | Tránh "chọn trên đề rồi chấm lại trên chính đề đó" |
| Câu được đo | Mọi lượt người dùng trong lịch sử và lượt cuối | Agent dùng chữ của mọi lượt |
| Phần cứng | Một phiên Kaggle T4, `dtype = "auto"` (float16 trên T4) | Thời gian của các mô hình so được với nhau |
| Giải mã | Tất định: `temperature = 0` (trừ lựa chọn "nhiệt độ thử lại") | Chạy lại cho cùng kết quả |
| Bộ lọc ảo giác | Bật (`sanitize = true`) | Bộ lọc chạy sau ASR, không phụ thuộc engine, nên áp được giống hệt |
| `condition_on_previous_text` | `false` | Mỗi câu nghe độc lập, câu bịa không lây sang câu sau |
| Code đo | `lifemate/eval/eval_asr.py`, cùng cách chuẩn hóa chữ | |

**Chỉ số** (mọi chỉ số tính trên cả kho, cộng lỗi rồi chia tổng):

| Chỉ số | Ý nghĩa |
|---|---|
| Bỏ sót thực thể | Giá trị tham số của lời gọi tool đáp án (tên, số, tên kênh) được nói ra nhưng không có trong chữ ASR lượt cuối |
| Tỉ lệ câu rỗng | Câu có đáp án nhưng ASR trả rỗng (mất trắng lượt đó) |
| `wer_nonum` | WER sau khi bỏ số ở cả hai phía: đáp án viết "tám giờ", Whisper viết "8 giờ"; WER thường phạt cách viết chứ không phải lỗi nghe |
| WER, CER | Để so với các số trước |
| Câu bịa | Chữ trước bộ lọc khớp mẫu phụ đề bịa |
| Tên riêng | Số câu viết đúng Zalo, VTV, HTV, voucher… trên số câu đáp án có tên đó |
| p50, p95 | Thời gian ASR mỗi mẫu (giây, trên T4) |

**Hàm mục tiêu** (nhỏ hơn là tốt hơn), giống nhau cho mọi mô hình:

1. Cấu hình có p95 > 6 s xếp sau mọi cấu hình đạt ngưỡng (6 s chừa phần lớn hạn 29 s cho LLM). Ngưỡng mềm vì
   large-v3 đo được p95 khoảng 8 s trên T4 ở mọi biến thể: loại hẳn sẽ không còn gì để so.
2. Bỏ sót thực thể + tỉ lệ câu rỗng: hai thứ ảnh hưởng thẳng tới điểm thi.
3. Hoà thì `wer_nonum`.

## 3. Mô hình và cấu hình riêng

| Mô hình | Engine | Tham số | Giấy phép, ngày | Revision |
|---|---|---|---|---|
| `Systran/faster-whisper-large-v3` | `faster_whisper` (CTranslate2) | 1,55B | MIT | `edaa852e…` |
| `vinai/PhoWhisper-medium` | `hf_seq2seq` (transformers) | 0,77B | BSD-3 | `55a7e3eb…` |
| `khanhld/chunkformer-ctc-large-vie` | `chunkformer` | 0,11B | CC-BY-NC-4.0 (phi thương mại), 02/2025 | `311fc035…` |
| `khanhld/chunkformer-rnnt-large-vie` | `chunkformer` | 0,11B | CC-BY-4.0, 10/2025 | `9863592e…` |
| `nvidia/parakeet-ctc-0.6b-Vietnamese` | `nemo_ctc` (NeMo 2.6.2) | 0,6B | NVIDIA Open Model License, 01/2026 | `b0493142…` |

Cả năm phát hành trước 26/10/2026; tổng với LLM 4,02B đều dưới 6B. ChunkFormer CTC có giấy phép phi thương mại,
cần BTC xác nhận trước khi dùng để nộp bài.

**Các bước dò** (mỗi bước giữ lựa chọn thắng của các bước trước; lựa chọn engine không có thì bỏ qua):

| Bước | Lựa chọn | Lý do thử |
|---|---|---|
| 1. Giải mã | tham lam (beam 1), beam 5 | Beam xem nhiều phương án hơn nhưng chậm hơn và dễ chọn câu thuộc lòng có xác suất cao |
| 2. Câu mồi | không; câu của lead; câu ngắn giống lời nói ("Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."); `hotwords` | Câu mồi giúp viết đúng tên riêng nhưng có thể gây bịa; sweep v2 cho thấy câu ngắn viết đúng Zalo nhiều hơn |
| 3. Chống lặp | tắt; `repetition_penalty = 1.2` + `no_repeat_ngram_size = 3` | Cấu hình của lead; có thể làm sai âm tiết lặp hợp lệ trong tiếng Việt |
| 4. Riêng engine | không; VAD (cắt khoảng lặng ≥ 500 ms); nhiệt độ thử lại `[0; 0,2; …; 1]`; beam + KenLM | Chỉ engine có tính năng đó mới thử |

**Lựa chọn mỗi mô hình được thử, và lý do:**

| Mô hình | Bước 1 | Bước 2 | Bước 3 | Bước 4 | Lý do |
|---|---|---|---|---|---|
| large-v3 | tham lam, beam 5 | không, lead, ngắn, `hotwords` | tắt, bật | không, VAD, nhiệt độ thử lại | faster-whisper có đủ mọi tham số |
| PhoWhisper | tham lam, beam 5 | không, lead, ngắn | tắt, bật | không | transformers có beam, câu mồi (`prompt_ids`), chống lặp; không có VAD, `hotwords`. Nhiệt độ thử lại của transformers cần thêm ngưỡng riêng nên không thử, để giống cách tính của faster-whisper |
| ChunkFormer CTC | tham lam | không | tắt | không | Gói `chunkformer` chỉ giải mã tham lam; mô hình CTC không có câu mồi hay chống lặp |
| ChunkFormer RNNT | tham lam | không | tắt | không | Như trên (cùng gói) |
| Parakeet CTC | tham lam | không | tắt | không, beam + KenLM | CTC tham lam, hoặc beam 64 với mô hình ngôn ngữ KenLM 4-gram và lexicon đi kèm repo (`alpha = 0,3`, `beta = 0,5` theo model card) |

**Chi tiết cấu hình theo engine, và lý do:**

- **PhoWhisper không chuyển sang CTranslate2.** Chuyển định dạng có thể làm kết quả lệch so với bản gốc. Thay vào đó,
  engine transformers nhận cùng tên khóa với faster-whisper và tự dịch (`hf_generate_kwargs` trong
  `lifemate/backends/asr.py`): `beam_size` thành `num_beams`, `initial_prompt` thành `prompt_ids`,
  `condition_on_previous_text` thành `condition_on_prev_tokens`. `best_of` bị bỏ vì faster-whisper chỉ dùng `best_of` khi
  lấy mẫu (`temperature > 0`), nên với giải mã tất định không có tác dụng.
- **ChunkFormer:** gói tự tải mô hình nhưng bỏ qua revision, nên engine tải snapshot đúng revision trước khi nạp.
  Tham số chunk giữ mặc định của gói (`chunk_size = 64`, ngữ cảnh trái/phải 128): không dò để giữ số lần chạy ngang
  nhau giữa các mô hình.
- **Parakeet:** lựa chọn beam + KenLM dùng đúng các giá trị ví dụ trong model card, không dò thêm `alpha`, `beta`.
- Các khóa chung kiểu Whisper (`language`, `task`, `temperature = 0`, `beam_size = 1`) bị engine CTC bỏ qua vì không
  áp dụng; khóa khác (câu mồi, VAD…) làm engine CTC báo lỗi thay vì lặng lẽ bỏ qua.

## 4. Phương pháp thử nghiệm

1. **Chạy thử trên CPU** (kernel `quanlam/lifemate-asr-preflight`): cài đúng các thư viện như kernel GPU, chạy mọi
   lựa chọn của từng mô hình trên 3 mẫu. Mục đích: bắt lỗi cài đặt, nạp mô hình và tham số trước khi tốn GPU.
2. **Dò cấu hình trên GPU** (kernel `quanlam/lifemate-asr-tune`, một phiên T4 cho cả năm mô hình): với mỗi mô hình,
   chạy `python -m lifemate.eval.asr_tune` theo các bước ở mục 3 trên `asr_tune.txt`. Cấu hình đã đo thì dùng lại
   kết quả.
3. **Đo cấu hình thắng** của mỗi mô hình đúng một lần trên val_seen và val_unseen.
4. ChunkFormer và Parakeet đo sau cùng, thư viện riêng của chúng cài ngay trước khi đo, vì có thể đổi phiên bản torch.

Tái tạo: `make splits` (sinh `asr_tune.txt`), rồi `kaggle kernels push -p ops/kaggle/asr_tune --accelerator
NvidiaTeslaT4`, hoặc trên máy có GPU: `python -m lifemate.eval.asr_tune --env <profile> --out <thư mục>`.

## 5. Kết quả

Chưa có.

## 6. Phân tích và đánh giá

Chưa có.

## 7. Kết luận

Chưa có.

## Giới hạn đã biết

- Thời gian đo trên T4, chưa phải A30 lúc thi.
- Chỉ số ASR chỉ để rút gọn danh sách; chọn cuối cùng phải bằng điểm thi thật (chạy cả hệ thống, chấm bằng
  `score.py`), cần GPU từ 24 GB.
- Tập chỉnh có ít câu chứa từng tên riêng, nên chênh vài câu đã là khác biệt lớn.
- Nếu sau này fine-tune ASR trên train, phải loại `asr_tune.txt` khỏi dữ liệu train ASR.
