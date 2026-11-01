# Báo cáo benchmark ASR lần 4: giá trị tham số trong chữ ASR (đổi số, sửa tên, dữ liệu fine-tune từ chữ ASR)

Ngày lập kế hoạch: 31/10/2026. Trạng thái: **L1 đã đo và nhận (mục 3.4); L2 đã đo và loại (mục 4.4); L3a xong (mục
5.2); code `MergedASR` xong (mục 6); L0, L3b và mọi việc cần GPU khác chưa chạy; L0, J1 xong (mục 2)**. Phần không cần GPU (L1, L2, L3a) làm trước trên máy dev; phần cần GPU ghi rõ và chờ phiên GPU.

Lần 3 ([`asr_v3.md`](asr_v3.md), mục "Nghiệm thu lần 3") kết thúc với ba số đếm trên `val_seen`:

1. Bộ quyết định ghép hoàn hảo chỉ đạt 80,3% recall (độ phủ) từ tiếng Anh (H đang 76,7%); 66 trên 335 từ không có ở bên nào.
2. Tỉ lệ giá trị tham số của lời gọi đáp án vắng trong chữ ASR bằng nhau ở ba nguồn (RNNT 32,1%, large-v3 29,4%, ghép
   30,3%); phần lớn giá trị mất là chữ và tên Việt.
3. 59% giá trị tham số có chữ số; chữ RNNT không có chữ số ở câu nào (bản ghép 6/95 câu).

Và một kết quả: 25,6 điểm độ phủ từ tiếng Anh đổi được +0,4 điểm thi (J3 so với J1, CI chứa 0).

Báo cáo lần 4 trả lời ba câu hỏi:

1. ASR hoàn hảo thì điểm thi của agent hiện tại là bao nhiêu, tức mọi cải tiến ASR còn đáng tối đa bao nhiêu điểm? (L0)
2. Hai bước xử lý chữ không cần mô hình, đổi số đọc thành chữ số (L1) và sửa tên riêng theo danh sách có sẵn của mẫu
   (L2), giảm được bao nhiêu giá trị tham số bị thiếu, và có làm hỏng câu thường không?
3. LLM học trên chữ ASR thật thay vì chữ chuẩn có chịu lỗi ASR tốt hơn không? (L3)

## Tóm tắt cho cả team

Điểm thi chấm **lời gọi tool đúng tên và đúng giá trị tham số**. Ba lần trước đo chữ bằng WER và recall (độ phủ) từ tiếng Anh;
lần 4 đo bằng **giá trị tham số có trong chữ** (`entity_miss_rate`, SPEC 6.1.1) và điểm thi. Lý do: hai thước đo cũ đã
được tối ưu tới gần trần mà điểm thi gần như không đổi.

Ba hướng, theo thứ tự rẻ và chắc:

| Hướng | Làm gì | Cần GPU | Đo bằng |
|---|---|---|---|
| L1 đổi số | Luật xác định đổi "năm rưỡi sáng", "ba mươi phần trăm", "không chín không một…" thành `5 rưỡi sáng`, `30%`, `0901…` trên chữ RNNT | Không | `entity_miss_rate`, WER, số câu có chữ số, so với lời chuẩn trên chữ đã lưu |
| L2 sửa tên | So từng cụm chữ ASR với danh sách tên của mẫu (app đã cài, enum trong catalog tool, kết quả tool) bằng khoảng cách chuỗi không dấu; thay khi đủ gần và không có ứng viên thứ hai. **Đã đo, loại**: 69% tên bị mất không có trong mẫu; 15 lần thay trên 3 593 câu, 6 sai (mục 4) | Không | `entity_miss_rate`, độ phủ từ tiếng Anh, số lần sửa nhầm, WER câu thuần Việt |
| L3 dữ liệu SFT từ chữ ASR | Dựng dòng SFT với câu người dùng là chữ ASR thật của train (đã có từ phiên K2), lời gọi đáp án giữ nguyên | Dựng: không; fine-tune: có (lead) | Điểm thi của mô hình fine-tune trên `val_seen` |

Trước khi chạy agent với bất kỳ hướng nào, cần **L0**: điểm thi với chữ chuẩn trên `val_seen` (J0 của lần 3). Nếu
J0 − J1 nhỏ hơn 3 điểm thì ASR không còn là nút thắt và L1, L2 chỉ giữ ở mức rẻ; nếu lớn thì L1, L2, L3 có giá trị.

Không có từ điển tay trong L2: mọi tên ứng viên sinh từ mẫu và từ đặc tả tool của BTC, là phần chung giữa train và bộ
thi riêng, nên không học từ lỗi của train. Kiểm chứng tổng quát hoá bằng `val_unseen` một lần khi nộp (D53).

## 1. Dữ liệu, thước đo, cách chọn

Theo SPEC 6.1.1 và D53:

| Tập | Dùng cho |
|---|---|
| `train` (1 093 mẫu, chữ RNNT và large-v3 E2 đã có từ phiên K2) | Dò ngưỡng của L1, L2 bằng kiểm định chéo 5 phần theo mẫu (`fold_of`); dựng dữ liệu L3 |
| `val_seen` (274) | Đo một lần mỗi ứng viên đã chốt; L0 |
| `val_unseen` (178) | Khoá; đo một lần cho cấu hình sắp nộp |

Thước đo chính: `entity_miss_rate` (giá trị tham số của lời gọi đáp án, xuất hiện nguyên văn trong lời chuẩn, vắng trong
chữ ASR). Thước đo chặn: WER câu thuần Việt không tệ hơn mốc có ý nghĩa; số lần sửa nhầm của L2 (cụm bị thay mà lời chuẩn
không có tên đó). Thước đo cuối: điểm thi (`strict`) khi chạy agent. Thống kê: bootstrap ghép cặp theo mẫu, 1 000 vòng,
seed 0, như ba lần trước.

Mốc so sánh: bản ghép H của lần 3 (chữ đã lưu). L1 và L2 áp lên chữ của mốc.

## 2. Thử nghiệm L0: trần của ASR (cần GPU)

Chạy agent với chữ chuẩn của BTC (`--transcripts gold`) trên `val_seen`, định dạng ChatML (D50), cùng cấu hình với lần
chạy lại J. Khoảng 274 mẫu, 1,5 giờ trên T4. Số J0 − J1 là phần điểm ASR còn mất. Đồng thời chạy lại J1 và J với nguồn
ghép H ở ChatML để có mốc cùng định dạng.

**Kết quả (31/10/2026, Colab T4, vLLM 0.32.0, định dạng ChatML với luật ép đầu ra, Qwen3-4B chưa fine-tune; `val_seen`
274 mẫu; chênh lệch kèm CI 95% bootstrap ghép cặp theo mẫu, 1 000 vòng, seed 0):**

| Nguồn chữ | Điểm (`strict`) | Thực thi / hỏi lại / ngoài phạm vi | Mẫu hết giờ | Điểm trên mẫu không hết giờ | Thời gian p50 / p95 |
|---|---|---|---|---|---|
| L0 chữ chuẩn | **25,33** | 25,32 / 8,89 / 54,55 | 50 (18%) | 29,32 | 18,2 / 29,1 s |
| J1 chữ RNNT | **15,33** | 13,90 / 6,15 / 45,45 | 48 (18%) | 18,41 | 18,1 / 29,1 s |
| **L0 − J1** | **+10,00 [+6,25; +14,06]** | | 37 mẫu hết giờ ở cả hai | | |

L0 cao hơn J1 ở 51 mẫu, thấp hơn ở 11 mẫu. Kết quả thô: `artifacts/runs/colab/agent/10-31_l0_j1/`.

**Phân tích.**

- **Chữ RNNT đứng riêng làm mất 10 điểm so với chữ chuẩn, có ý nghĩa thống kê.** Đây là phần điểm tối đa mà cải tiến ASR
  có thể lấy lại so với RNNT. Pipeline đã chốt (ghép H + đổi số, J5) chưa được chạy, nên chưa biết pipeline đó lấy lại
  được bao nhiêu trong 10 điểm này, và chưa kết luận được cải tiến ASR thêm có đáng hay không.
- **Chữ chuẩn vẫn chỉ được 25 điểm**: phần lớn điểm mất nằm ở LLM chưa fine-tune và ở thời gian.
- **Hết giờ là vấn đề lớn của định dạng ChatML trên T4**: 18% số mẫu (5% ở định dạng JSON của lần 3). Mẫu hết giờ có
  bước gọi LLM đầu tiên trung vị 22,5 giây, so với 10,8 giây ở mẫu khác; các bước sau khoảng 2 giây nhờ cache phần đầu.
  Nguyên nhân bước đầu chậm chưa tách được (đọc prompt khoảng 5 000 token, biên dịch luật ép đầu ra của từng mẫu). Trên
  mẫu không hết giờ, L0 được 29,3 điểm.

**Kết luận.** Việc kế tiếp bắt buộc là J5 (pipeline đầy đủ: ghép H + đổi số) trên cùng 274 mẫu, cùng định dạng: J5 gần
L0 thì ASR hiện tại đã đủ; J5 gần J1 thì còn khoảng 10 điểm cho ASR. Song song, tìm nguyên nhân bước đầu chậm (thử `constrain_tools = false`, `compact_tools = true`
trên 30 mẫu) vì đây là phần điểm mất không phụ thuộc ASR.

## 3. Thử nghiệm L1: đổi số đọc thành chữ số

### 3.1. Vấn đề

RNNT viết mọi số bằng chữ; lời chuẩn và giá trị tham số viết chữ số. Ví dụ `train_0007`: lời chuẩn "Đặt báo thức 5 giờ
nhé, à không, 5 rưỡi sáng mai", RNNT "tập báo thức năm giờ nhé à không năm rưỡi sáng mai đó". LLM phải tự đổi; mỗi lần
đổi sai là một tham số sai.

### 3.2. Giải pháp

Bộ luật `lifemate/backends/asr_numbers.py` (tên dự kiến) chạy sau bộ lọc câu bịa, trước khi ghép: tìm dãy từ đọc số
("không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười", "mươi", "trăm",
"nghìn", "ngàn", "triệu", "tỷ", "rưỡi", "linh", "lẻ") và viết lại:

- Dãy chữ số đọc rời từ 4 từ trở lên: ghép thành dãy số (số điện thoại, mã).
- Số có hàng: "ba mươi" → 30, "một trăm hai mươi lăm" → 125, "một triệu hai" → 1 200 000 (kiểu viết chọn sau khi đếm
  cách viết của lời chuẩn trên train).
- Giờ: "bảy giờ kém mười lăm" giữ nguyên chữ "giờ" và "kém", chỉ đổi số: "7 giờ kém 15"; không tự tính 6:45, việc đó
  để LLM vì cần ngữ cảnh sáng, chiều.
- "năm" chỉ đổi khi đứng trong dãy số hoặc trước đơn vị (giờ, phút, phần trăm, nghìn, triệu, độ, kênh, tuổi, ngày,
  tháng); "năm nay", "năm sau" giữ nguyên.

### 3.3. Cách đo

Bộ đổi không có tham số dò: luật và kiểu viết chọn theo cách lời chuẩn của train viết số (đếm trên 3 593 câu người
dùng của train: 459 token chữ số, hầu hết là số nguyên trần "5 giờ", "500 nghìn", "127 triệu", "30 phần trăm"; chữ
"nghìn", "triệu", "rưỡi", "phần trăm" giữ bằng chữ). Đo trên train (1 093 mẫu, tập phát triển) rồi `val_seen` một lần,
bằng `python -m lifemate.eval.asr_v4_report l1` ([`lifemate/eval/asr_v4_report.py › stage_l1`](../../../lifemate/eval/asr_v4_report.py)):
bốn hệ thống RNNT, RNNT + L1, ghép H (mốc; học lại H trên cả train như D54), ghép H + L1; `compare` bootstrap ghép cặp
theo mẫu.

Tiêu chí đặt trước: `entity_miss_rate` giảm có ý nghĩa so với mốc và WER câu thuần Việt không tăng có ý nghĩa.

**Điều phát hiện khi đo, ảnh hưởng tới cách đọc WER.** Lời chuẩn của BTC viết số **không nhất quán**: cùng kiểu câu,
"tám giờ" bằng chữ xuất hiện 84 lần, "5 giờ" bằng chữ số 82 lần. Vì vậy WER so với lời chuẩn gốc phạt cả khi đổi đúng
("8 giờ" so với "tám giờ" là một lỗi), và không tách được lỗi đổi số khỏi khác biệt cách viết. Báo cáo vì thế thêm hai
số: WER bỏ số (`wer_nonum`, bỏ mọi chữ số và từ đọc số ở hai phía; phải **không đổi** nếu bộ đổi không đụng gì khác) và
WER so với **đáp án đã đổi số cùng luật**. Tiêu chí chặn về WER câu thuần Việt được đọc trên hai số này, không trên WER
gốc; đây là thay đổi so với tiêu chí viết lúc lập kế hoạch và được ghi lại ở đây thay vì sửa âm thầm.

### 3.4. Kết quả

**Train (1 093 mẫu, 455 giá trị tham số; chênh lệch so với ghép H kèm CI 95%):**

| Hệ thống | Giá trị tham số thiếu | WER gốc | WER bỏ số | WER so đáp án đã đổi số | Câu có chữ số: đáp án / ASR | Chữ số đúng / tổng (đáp án đã đổi) | Chữ số lạ | Câu bị đổi |
|---|---|---|---|---|---|---|---|---|
| RNNT | 36,3% [+1,1; +4,6] | 20,86% | 20,23% | 23,42% | 357 / 0 | 0 / 1 285 | 0 | 0 |
| RNNT + L1 | 27,7% [−9,6; −1,7] | 22,07% | 20,23% | 19,93% | 357 / 256 | 991 / 1 285 | 85 | 880 |
| Ghép H (mốc) | 33,4% | 19,99% | 19,33% | 22,51% | 357 / 28 | 34 / 1 285 | 15 | 0 |
| **Ghép H + L1** | **24,8%** [−12,0; −5,1] | 21,23% [+0,9; +1,6] | 19,33% [0; 0] | **19,08%** | 357 / 269 | **1 005 / 1 285** | 98 | 861 |

**`val_seen` (274 mẫu, 109 giá trị tham số), đo một lần:**

| Hệ thống | Giá trị tham số thiếu | WER gốc | WER bỏ số | WER so đáp án đã đổi số | Câu có chữ số: đáp án / ASR | Chữ số đúng / tổng | Chữ số lạ | Câu bị đổi |
|---|---|---|---|---|---|---|---|---|
| Ghép H (mốc) | 30,3% (33/109) | 18,97% | 18,63% | 21,19% | 95 / 5 | 6 / 299 | 1 | 0 |
| **Ghép H + L1** | **26,6%** (29/109) [−9,4; +1,1] | 20,28% [+0,8; +1,8] | 18,63% [0; 0] | **18,40%** | 95 / 63 | **228 / 299** | 20 | 200 |

*"Chữ số lạ": chữ số trong chữ ASR không có trong đáp án đã đổi số; gồm cả đổi sai lẫn nghe sai. Rà 88 trường hợp trên
chữ RNNT của train: phần lớn là nghe sai ("131 triệu" nghe thành "một trăm hai mươi mốt triệu", số điện thoại thiếu
chữ số), không phải lỗi của bộ đổi; ba lỗi đổi tìm được khi rà ("không" phủ định thành 0, nhiệt độ đọc tắt "hai sáu
độ", "chín h") đã sửa trước khi đo lần cuối.*

### 3.5. Phân tích và kết luận

- **Trên train, L1 đạt tiêu chí**: giá trị tham số thiếu giảm 8,6 điểm (CI không chứa 0); WER bỏ số không đổi, nghĩa là
  bộ đổi không đụng gì ngoài số; WER so đáp án đã đổi số giảm 3,4 điểm; 1 005 trên 1 285 chữ số của đáp án nay có trong
  chữ (trước đó 34).
- **Trên `val_seen` cùng chiều nhưng CI chứa 0** (−3,7 điểm, CI [−9,4; +1,1]): tập này chỉ có 109 giá trị tham số, quá
  ít để tách 4 giá trị; số đo chữ số (6 lên 228 trên 299) và WER so đáp án đã đổi (21,2% xuống 18,4%) cho thấy hiệu
  quả giống train.
- **WER gốc tăng 1,2 điểm là do cách viết của lời chuẩn**, không phải do chữ sai: WER bỏ số không đổi. Khi lời chuẩn
  viết "tám giờ", chữ "8 giờ" bị tính là một lỗi dù LLM đọc "8 giờ" dễ hơn.
- Thước đo `entity_miss_rate` **đếm thiếu** phần số: chỉ tính giá trị tham số xuất hiện nguyên văn trong lời chuẩn, nên
  giá trị `30` của câu lời chuẩn viết "ba mươi phần trăm" không được tính. Mức giảm thật với LLM có thể lớn hơn số đo.

**Kết luận.** L1 được đưa vào pipeline (áp lên chữ RNNT trước khi ghép, trong `MergedASR`) với điều kiện: điểm thi của
agent với chữ H + L1 không thấp hơn chữ H (đo ở L0 cùng phiên GPU). Code: [`lifemate/backends/asr_numbers.py`](../../../lifemate/backends/asr_numbers.py)
(`digitize`), test `tests/test_asr_numbers.py`. Kết quả thô: `artifacts/runs/colab/asr/10-30_v3/analysis/l1_train.json`,
`l1_val_seen.json`.

### 3.6. Giới hạn

- Không đổi giờ thành dạng `HH:MM`, không gộp "một triệu hai" thành `1200000`; LLM vẫn phải làm bước cuối.
- Luật "một", "hai" đơn không đổi trước danh từ đếm ("hai chỗ", "một lít") vì lời chuẩn viết bằng chữ; nếu LLM nhầm ở
  đó, cần đo lại.
- Chưa đo điểm thi; chưa đo `val_unseen`.

## 4. Thử nghiệm L2: sửa tên riêng theo danh sách của mẫu

### 4.1. Vấn đề

66 từ tiếng Anh và nhiều tên Việt bị chép lệch ở cả hai mô hình, nhưng gần đúng về âm: "Olivia Broad" cho Olivia Brown,
"Harbour View" cho Harbor View, "scg" cho SJC, "len" cho Lan, "sếp chuẩn" cho sếp Dũng. Tên đúng có sẵn trong mẫu
(danh bạ, app đã cài, enum của tool) hoặc trong đặc tả tool.

### 4.2. Giải pháp

Với mỗi mẫu, danh sách ứng viên gồm: tên app đã cài (tiếng Việt và tên gốc), giá trị `enum` trong mô tả tool của mẫu,
tên trong kết quả tool đã nhận ở các lượt trước (danh bạ, thiết bị), và các giá trị cố định suy từ catalog (loại vàng,
loại xăng, kênh). Không có từ điển tay.

Với từng cụm 1–3 từ của chữ ASR: chuẩn hoá (bỏ dấu, viết thường, bỏ khoảng trắng) rồi tính khoảng cách sửa tới từng ứng
viên. Thay khi khoảng cách ≤ ngưỡng theo độ dài (dò trên train; dự kiến 0 với từ dưới 4 ký tự, 1 với 4–6, 2 với dài hơn)
và ứng viên gần nhì cách xa hơn ít nhất 1. Chỉ xét cụm có dấu hiệu không phải tiếng Việt thường: không có dấu, không
phải âm tiết Việt, hoặc là từ mà RNNT và large-v3 bất đồng. Mọi thay thế ghi vào `trace`.

### 4.3. Cách đo

Như L1, thêm: độ phủ từ tiếng Anh, số lần sửa nhầm, và **trần của L2**: trong các từ tiếng Anh của đáp án không có ở
cả RNNT lẫn large-v3, bao nhiêu từ có mặt trong danh sách ứng viên của mẫu (L2 chỉ sửa được những từ này). Tiêu chí
đặt trước: `entity_miss_rate` giảm có ý nghĩa, số sửa nhầm dưới 1 trên 10 lần sửa, WER câu thuần Việt không tăng có ý
nghĩa. Lệnh: `python -m lifemate.eval.asr_v4_report l2` ([`lifemate/eval/asr_v4_report.py › stage_l2`](../../../lifemate/eval/asr_v4_report.py));
mốc là ghép H + L1.

### 4.4. Kết quả (train, 1 093 mẫu, 3 593 câu)

| | H + L1 (mốc) | H + L1 + L2 |
|---|---|---|
| Giá trị tham số thiếu | 24,8% | 24,8% (không đổi) |
| Độ phủ từ tiếng Anh | 75,7% | 76,0% [0; +0,7] |
| WER, WER bỏ số, WER thuần Việt | 21,22% / 19,33% / 20,68% | như mốc (chênh dưới 0,02 điểm) |
| Số lần thay | | **15** trên 3 593 câu: 9 đúng, 6 sai |
| Trần: từ tiếng Anh mất ở cả hai mô hình có trong danh sách ứng viên | | **91 / 294 (31%)** |

Chín lần sửa đúng đều là tên gần đúng về âm ("Priya Shahma" thành Priya Sharma, "liam potter" thành Liam Porter,
"Luminar studio" thành Lumina Studio, "euro" thành EUR). Sáu lần sai cùng một kiểu: cụm ba từ "… nhắn sms" ("đi nhắn
sms", "viên nhận sms") bị thay bằng tên app "Tin nhắn SMS" ở khoảng cách 2. Danh sách ứng viên mỗi lượt có trung vị 161
tên, tối đa 262. Trên `val_seen` (chạy thử trước khi đo chính thức): 3 lần thay, 1 đúng; không đo chính thức vì L2 không
được chốt. Kết quả thô: `artifacts/runs/colab/asr/10-30_v3/analysis/l2_train.json`.

### 4.5. Phân tích và kết luận

- **Giả định "tên đúng có sẵn trong mẫu" sai với phần lớn trường hợp.** 69% từ tiếng Anh mà cả hai mô hình đều không
  chép ra **không có** trong app đã cài, catalog tool, ngữ cảnh hay kết quả tool của mẫu: tên khách sạn, tên thuốc,
  kênh truyền hình, thương hiệu vàng, món ăn chỉ xuất hiện trong lời nói. Không danh sách nào sinh từ mẫu khôi phục
  được chúng; từ điển tay thì học từ train (mục 4.2 đã loại).
- Với 31% còn lại, so chuỗi theo khoảng cách sửa chỉ tìm được 9 chỗ trên 3 593 câu, và cứ 3 lần sửa thì 2 lần đúng,
  không đạt tiêu chí "dưới 1 sai trên 10". Siết ngưỡng về khoảng cách 1 còn 8 đúng, 1 sai, vẫn quá ít để đo được.
- Điểm có ích rút ra: nhiều từ mất là từ ngắn ba chữ cái ("SJC", "RON", "VTV") mà hai mô hình chép thành cụm gần âm
  ("scg", "non", "vê tê vi"); khoảng cách ký tự không bắt được kiểu lệch theo âm này.

**Kết luận.** L2 **không đưa vào pipeline**. Code giữ lại ([`lifemate/backends/asr_entities.py`](../../../lifemate/backends/asr_entities.py),
test `tests/test_asr_entities.py`) để dùng lại nếu sau này ngữ cảnh mẫu có danh bạ hay danh sách thiết bị ngay từ lượt
đầu; khi đó trần sẽ khác. Phần việc còn lại cho tên riêng chuyển sang L3 (LLM học chịu lỗi ASR) và, nếu cần, sang bước
ASR (câu mồi theo mẫu đã làm ở lần 3 là cách duy nhất đưa tên vào mô hình mà không cần từ điển).

## 5. Thử nghiệm L3: dữ liệu fine-tune từ chữ ASR thật

### 5.1. Vấn đề

Dữ liệu SFT hiện dựng từ `audio_text` (chữ chuẩn); lúc thi LLM nhận chữ ASR. Mô hình học trên chữ sạch nhưng chạy trên
chữ bẩn (lệch giữa huấn luyện và chạy).

### 5.2. Giải pháp

Bộ dựng SFT ([`lifemate/train/sft_data.py`](../../../lifemate/train/sft_data.py)) đã có sẵn lựa chọn
`transcript_source = "asr" | "mix"` đọc chữ ASR từ một cache (SPEC 6.2.1); cái thiếu là cache của **pipeline đã chốt**:
cache cũ `data/asr/train_asr.json` (28/10) là large-v3 beam 5 đứng riêng, không còn là ASR của hệ thống.

L3a (không cần GPU, đã làm 31/10, D58): [`lifemate/train/hyps_cache.py`](../../../lifemate/train/hyps_cache.py) dựng
cache từ chữ đã lưu của phiên K2 và K3 qua đúng chuỗi xử lý đã chốt (ghép H học trên cả train, rồi `digitize`), đúng
định dạng của `asr_cache.py`, cho `train` (1 093 mẫu) và `val_seen` (274); `val_unseen` để dành (D53). Kết quả:
`data/asr/train_asr_v3h.json` (không có trên git, thư mục `data/` bị bỏ qua). Dữ liệu SFT dựng từ cache này với
`configs/train/llm_v2.toml` (ChatML, D50): `data/sft/v2_asr.jsonl` 1 047 dòng (46 mẫu bỏ vì lời gọi đáp án dùng app giữ
lại, không liên quan ASR; 3% dòng phục hồi lỗi), `data/sft/v2_val_seen_asr.jsonl` 274 dòng để đo loss. Kiểm tra bằng
mắt: câu người dùng của `train_0007` trong dữ liệu là "tập báo thức 5 giờ nhé à không 5 rưỡi sáng mai đó", đúng chữ RNNT
sau khi đổi số.

L3b (cần GPU, phối hợp lead): fine-tune với dữ liệu này và đo điểm thi trên `val_seen` so với mô hình học trên chữ
chuẩn; đo cả hai nguồn chữ lúc chạy (chữ chuẩn và chữ H + L1) để tách phần LLM tự sửa được.

Trạng thái: L3a xong; L3b chờ GPU.

## 6. Việc kỹ thuật kèm theo (từ "Sau lần 3")

| Việc | Cần GPU | Trạng thái |
|---|---|---|
| Engine `MergedASR` (RNNT + large-v3 E2 + điểm âm thanh + H + L1; trọng số H ở `configs/asr_merge/h_v1.json`) | Viết: không; chạy thật: có | **Code xong 31/10 (D59)**: [`lifemate/backends/asr_merged.py`](../../../lifemate/backends/asr_merged.py); `configs/v0.toml`, `docker.toml` đã trỏ sang; test phát lại K2, K3 trùng từng byte với bản ghép offline trên 40 mẫu `val_seen`. Còn chạy thật trên GPU (đối chiếu chữ, thời gian, bộ nhớ) |
| Chạy agent ChatML thật trên `val_seen` (kiểm chứng D50, D51 trên vLLM) | Có | Chưa làm |
| Đo cả hệ thống trên GPU 24 GB: thời gian từng mẫu, số mẫu quá 30 giây, bộ nhớ | Có | Chưa làm |
| Chốt cấu hình nộp, đo `val_unseen` một lần, nộp public test một lần (D53) | Có | Chưa làm |

## 6b. Lần chạy cuối của lần 4: J5, L0 mới, J6

### 6b.1. Thay đổi phía agent từ sau L0 (đều có test, chưa đo trên GPU)

Phân tích lỗi của L0 trên `val_seen` cho thấy điểm mất chủ yếu ở LLM, nên trước lần chạy cuối có các thay đổi sau:

| Thay đổi | Quyết định | Sửa lỗi nào của L0 |
|---|---|---|
| Đọc trước tin `system` trong lúc ASR chạy (`prefill_prefix`) | D60 | Lần gọi đầu đọc khoảng 6 000 token, mất khoảng 18 s trên T4; 50/274 mẫu hết giờ |
| Guard G5 `grounded_ids`: chặn mã chưa xuất hiện trong hội thoại | D61 | Mã đoán (`evt_12345`, `contact_123`): 39 lời gọi ở 34 mẫu |
| Guard G6 `duplicate_read`; chặn của G5, G6 không tốn lượt thử lại | D62 | Lớp an toàn (1 lời gọi lặp ở L0) |
| 6 quy tắc thêm vào lời dặn hệ thống (`configs/prompts/chatml_rules_v1.txt`, 295 token) | D62–D64 | Điền thêm tham số không bắt buộc; tin nhắn dài dòng; mã đoán; không hỏi lại khi mơ hồ (35/39 mẫu hỏi lại bị làm luôn); gọi tool khi lẽ ra không làm gì (8/22); tính giờ tương đối sai |

### 6b.2. Ba lần chạy

| Lần | Chạy gì | Đầu ra | Chấm |
|---|---|---|---|
| **J5** | Engine `MergedASR` (RNNT, large-v3 câu mồi E2, điểm âm thanh, H, đổi số) trên âm thanh thật của 274 mẫu `val_seen`, bằng `python -m lifemate.train.asr_cache` | Chữ từng câu (`.hyps.jsonl`), thời gian từng mẫu và từng phần (`.timing.jsonl`), bộ nhớ GPU đỉnh | Sau khi chạy: WER, chép đúng từ tiếng Anh, giá trị tham số thiếu so với lời chuẩn; đối chiếu chữ với bản ghép offline (phải trùng, trừ sai khác số học của GPU); thời gian p50, p95, tối đa |
| **L0v2** | Lời chuẩn qua agent với cấu hình mới (6b.1) | Run agent | Bộ chấm của BTC; so với L0 (cấu hình cũ, cùng 274 mẫu) |
| **J6** | Chữ bản 2 đã lưu (`v2_val_seen.hyps.jsonl`) qua cùng agent; engine `MergedASR` phát lại trùng từng byte với bản offline (`tests/test_asr_merge_v2.py`), nên không cần chờ J5 | Run agent | Bộ chấm của BTC; so với L0v2 (ASR còn làm mất bao nhiêu) và với J1 (pipeline ghép lấy lại được bao nhiêu so với RNNT) |

J5 chạy riêng ASR trên T4 (hai mô hình ASR khoảng 3,6 GB), không cùng vLLM, vì T4 không chứa nổi cả hai; thời gian ASR
vì vậy là thời gian khi ASR có GPU riêng. Thời gian của cả hệ thống (ASR cùng vLLM, đọc trước) cần GPU từ 24 GB, đo sau.

**Thứ tự chạy**, gộp với benchmark hai mô hình mới (mục 6d, 6e, 6f):

| Thứ tự | Lần chạy | Máy | Vì sao ở vị trí này |
|---|---|---|---|
| 1 | Đo VieGem và PhoWhisper-CS: kiểm tra, chép `asr_tune` và `val_seen` (khoảng 2 đến 3 giờ), rồi chấm bảng mục 6f trên CPU | GPU, chỉ ASR | Kết quả quyết định pipeline ASR nào được chạy thật ở bước 5; không phụ thuộc lần chạy agent nào |
| 2 | L0v2 | GPU, vLLM | Quyết định giữ hay tắt cấu hình agent mới (mục 6b.3); không phụ thuộc ASR; mọi lần chạy agent sau dùng cấu hình đã chốt ở bước này |
| 3 | J6 trên chữ bản 2 đã lưu | GPU, vLLM | Điểm thi của bản 2 là mốc ở mọi kết cục, kể cả khi mô hình mới thắng (bước xác nhận của mục 6f.4) |
| 4 | Agent trên chữ của mô hình mới, so với J6 (chỉ khi bước 1 chọn thay đổi) | GPU, vLLM | Xác nhận bằng điểm thi theo mục 6f.4 |
| 5 | J5: chạy thật pipeline được chọn trên âm thanh `val_seen` (bản 2, hoặc pipeline mới nếu bước 4 nhận) | GPU, chỉ ASR | Chỉ có giá trị cho pipeline được chọn cuối cùng: kiểm tra chữ chạy thật trùng chữ offline và đo thời gian thật |

Gom theo loại máy: bước 1 chỉ cần ASR, các bước 2 đến 4 cùng dựng vLLM một lần.

### 6b.3. Chỉ số và cách so

Tổng hợp bằng `python -m lifemate.eval.agent_report` (D63): điểm `strict` và `canon`; chênh lệch so với run mốc kèm CI
95% bootstrap ghép cặp theo mẫu (1 000 vòng, seed 0); số mẫu tăng, giảm điểm; điểm theo loại câu (thực thi, hỏi lại,
ngoài phạm vi); độ chính xác quyết định; loại lỗi của bộ chấm (sai trạng thái cuối, thiếu tool, sai tham số, gọi thừa);
số mẫu hết giờ và điểm trên mẫu không hết giờ; thời gian mỗi mẫu và lần gọi đầu; số bước; số lần mỗi guard chặn.

Tiêu chí đặt trước:

- **Cấu hình agent mới được giữ** khi L0v2 cao hơn L0 với CI không chứa 0, và điểm nhóm thực thi không giảm có ý nghĩa.
  Không đạt thì xem bảng theo loại câu và theo guard để biết thay đổi nào gây hại, tắt riêng thay đổi đó.
- **Pipeline ASR** được đánh giá bằng J6 − J1 (lấy lại bao nhiêu so với RNNT) và L0v2 − J6 (còn lại bao nhiêu cho ASR).
  J1 chạy bằng cấu hình agent cũ, nên J6 − J1 lẫn cả tác dụng của cấu hình mới; con số sạch về ASR là L0v2 − J6.

### 6b.4. Giới hạn đã biết trước

- Các quy tắc và G5 được thiết kế từ lỗi của L0 trên `val_seen`, nên L0v2 − L0 trên `val_seen` có thể lạc quan. Con số
  này dùng để quyết định giữ hay tắt cấu hình mới; xác nhận sạch cần một tập chưa nhìn (phần của `train`, hoặc
  `val_unseen` khi chốt bài nộp theo D53).
- Một lần chạy đổi nhiều thay đổi cùng lúc; nếu L0v2 tốt hơn thì không tách được phần của từng thay đổi.
- Trên T4 đọc trước không giấu được thời gian vì L0v2, J6 phát lại chữ (không có ASR chạy song song); 295 token quy tắc
  thêm khoảng 1 s cho lần gọi đầu.
- Whisper đôi khi chép lại lời trợ lý liền trước nằm trong câu mồi E2 (ví dụ `val_seen`: RNNT "chiều rồi tỷ giá đô hôm
  nay bao nhiêu em", Whisper "Chị đặt vé vào Sài Gòn thứ Bảy tuần sau rồi nhé."); lớp bảo vệ g4 chỉ so phần cố định nên
  không bắt. Đo trên train (khoảng 3 600 câu): chỉ 3 câu thật sự chép từ câu mồi; so với lời trợ lý bắt 57 câu (54 nhầm,
  người dùng thật sự lặp lời trợ lý), dùng RNNT làm nhân chứng (bỏ khi RNNT không có cụm đó) bắt 24 câu (21 nhầm, vì
  RNNT nghe lệch một từ trong cụm). Cái giá lớn hơn cái lợi, nên giữ g4 như lần 3.

Trạng thái: chưa chạy.

## 6c. Bộ ghép bản 2: quyết định cả vùng chữ Việt

**Vấn đề.** Pipeline chỉ đưa cho H các vùng mà phía large-v3 có từ tiếng Anh mới; vùng chỉ có chữ Việt luôn giữ RNNT.
Ví dụ `train_1717` lượt 2: lời chuẩn "Bỏ cái tắt máy lạnh phòng ngủ lúc năm rưỡi sáng đi, trời mát rồi.", RNNT "bỏ cái
tóc máy lạnh phòng ngủ năm bưởi sáng đi trời mát gọi", large-v3 đúng từng chữ, chữ ghép giống RNNT. Đếm trên train
(1 093 mẫu): 6 915 vùng chữ Việt khác nhau; RNNT gần lời chuẩn hơn ở 3 664 (53%), large-v3 ở 1 897 (27%), ngang nhau 1 354.
Chọn đúng mọi vùng chữ Việt thì WER (so lời chuẩn đã đổi số) giảm từ 19,93% xuống 15,83% (trần).

**Thiết kế đã chốt (01/11/2026):**

1. Đổi số cho cả hai bản ngay sau bước lọc, trước khi gióng, để "năm" và "5" không thành vùng khác nhau giả.
2. Gióng có giữ dấu thanh cho phần chữ Việt: giữ ranh giới vùng của phép gióng không dấu, tách thêm các từ chỉ khác dấu
   thanh ("tắt"/"tất") thành vùng chữ Việt.
3. Bộ phân loại riêng H-Việt (hồi quy logistic như H), tín hiệu chính là chênh điểm âm thanh, cộng độ dài hai phía và độ
   tin cậy chung của large-v3; nhãn tính so với lời chuẩn đã đổi số. H giữ nguyên cho vùng có từ tiếng Anh.
4. Điều kiện chặn đặt trước: WER câu thuần Việt giảm có ý nghĩa (kiểm định chéo trên train), rồi đo một lần trên
   `val_seen`.

**Đã cân nhắc và bỏ:** chấm điểm mọi tổ hợp lựa chọn của các vùng trong câu thay vì từng vùng riêng. Đo trên train (đã
đổi số trước khi gióng): 942/3 074 câu có vùng (30,6%) có ít nhất hai vùng cách nhau một từ, nhưng trần của hai cách bằng
nhau tuyệt đối (2 748 lỗi từ giảm được; không câu nào mà tổ hợp tốt hơn chọn từng vùng), vì hai vùng luôn cách nhau ít
nhất một từ khớp nên lỗi của các vùng cộng độc lập.

**Rà lại tín hiệu của H (H-Anh).** Kiểm định chéo 5 phần trên train (580 vùng có nhãn, L2 = 10, điểm âm thanh đã
lưu): như hiện tại 81,4%; điểm âm thanh trung bình mỗi token (bỏ thiên vị độ dài) 81,0%; thêm chênh lệch số token
82,2%; cả hai 80,9%; bỏ `no_speech_prob` 82,1%; bỏ hai tín hiệu câu mồi (`in_prompt`, `min_reliability`) 79,1%. Các
cách sửa thiên vị độ dài thay đổi trong ±1 điểm, nhỏ hơn sai số chuẩn (khoảng 1,6 điểm), nên giữ H như `h_v1.json`. Bỏ
tín hiệu câu mồi mất 2,3 điểm: H có phụ thuộc vào câu mồi (dựng từ từ tiếng Anh hay gặp của train), cần đo trên
`val_unseen` khi chốt bài nộp. Hai giới hạn còn lại: `avg_logprob`, `no_speech_prob` tính cho cả câu nên không phân
biệt các vùng trong cùng câu; người chấm điểm âm thanh (large-v3 không câu mồi) cùng họ với phía B.

**Tín hiệu đề xuất cho H-Việt:** chênh điểm âm thanh (như H); chênh lệch số token (bù thiên vị độ dài); số từ hai phía;
độ giống nhau về chữ giữa hai phía (khoảng cách sửa theo ký tự đã bỏ dấu, chia cho độ dài); kiểu vùng (thay, thêm,
bớt); vùng ở cuối câu; từ phía B có trong ngữ cảnh mẫu; `avg_logprob`; lượt cuối. Không dùng tín hiệu câu mồi và
`no_speech_prob`. Báo độ chính xác riêng cho vùng RNNT đúng và vùng large-v3 đúng (lớp lệch 53% / 27%), chọn ngưỡng
bằng kiểm định chéo; mọi tín hiệu đếm trên train đều đếm ngoài phần.

**Kết quả (01/11/2026).** Phiên K5 (Colab T4, `asr_v3 score --v2`, `artifacts/runs/colab/asr/11-01_k5/`) chấm điểm âm thanh
mọi vùng bản 2 của `asr_tune`, `asr_learn`, `val_seen`; máy bị thu hồi và tài khoản hết hạn mức GPU khi còn 10/793 mẫu
`asr_learn`, nên 10 mẫu đó bị bỏ khỏi bước học (1 083/1 093 mẫu train; danh sách trong `merge_v2_cv.json`). `val_seen`:
1 964 vùng (1 791 chữ Việt, 173 tiếng Anh), bộ nhớ GPU đỉnh 3,4 GB.

Kiểm định chéo 5 phần theo mẫu trên train (`merge_v2_report cv`; mốc là bản 2 chỉ H-Anh; CI 95% bootstrap ghép cặp):

| Hệ thống | WER câu thuần Việt | WER mọi câu | Chép đúng từ tiếng Anh | Giá trị tham số thiếu |
|---|---|---|---|---|
| Bản 2, chỉ H-Anh (mốc) | 20,74% | 21,36% | 73,95% | 24,94% |
| **Bản 2, H-Anh + H-Việt** | **19,74%** [−1,23; −0,79] | **20,43%** [−1,11; −0,76] | 74,02% [−0,43; +0,52] | **21,41%** [−5,62; −1,72] |
| Giữ RNNT ở mọi vùng | 20,65% | 22,11% | 52,33% | 27,81% |

Cả ba tiêu chí đặt trước đạt. Ở vùng chữ Việt có nhãn, H-Việt giữ đúng 89,5% vùng RNNT đúng (3 589 vùng) và lấy đúng
41,6% vùng large-v3 đúng (1 900 vùng). Trọng số học trên cả 1 083 mẫu: `configs/asr_merge/h_v2.json` (L2 = 10 cho cả
hai, ngưỡng H-Việt 0,5).

Đo một lần trên `val_seen` (`merge_v2_report measure`; mốc là bản 1 đang dùng, `h_v1` + đổi số sau khi ghép):

| Hệ thống | WER câu thuần Việt | WER mọi câu | Chép đúng từ tiếng Anh | Giá trị tham số thiếu |
|---|---|---|---|---|
| Bản 1 (mốc) | 19,19% | 20,28% | 76,72% | 26,61% |
| Bản 2, chỉ H-Anh | 19,19% | 20,30% | 76,12% [−2,11; +0,93] | 26,61% |
| **Bản 2, H-Anh + H-Việt** | **18,56%** [−0,99; −0,24] | **19,62%** [−1,00; −0,31] | 76,42% [−1,81; +1,21] | **23,85%** [−8,25; +2,59] |

Giá trị tham số thiếu cùng chiều với train nhưng CI chứa 0 (chỉ 109 giá trị). `v0.toml`, `docker.toml` chuyển sang
`h_v2.json`. `MergedASR` bản 2 phát lại chữ K2 và điểm K5 trùng từng byte với bản ghép offline trên 40 mẫu `val_seen`
(`tests/test_asr_merge_v2.py`). Dữ liệu SFT dựng lại từ chữ bản 2: `data/asr/train_asr_v4m2.json`,
`data/sft/v2_asr.jsonl` (1 047 dòng).

**Bộ dữ liệu học của H-Anh và H-Việt** (`merge_v2_report dataset`, chỉ trong `data/`, không commit vì chứa lời chuẩn của
BTC): `data/derived/merge_v2/train_regions.jsonl` và `val_seen_regions.jsonl`, mỗi vùng một dòng gồm mã mẫu, lượt, phần
kiểm định chéo, loại vùng (`en`, `vi`), chữ RNNT và large-v3 của vùng, hai câu ứng viên, lời chuẩn đã chuẩn hoá, điểm âm
thanh K5, vector tín hiệu có tên, nhãn (1 lấy large-v3, 0 giữ RNNT, `null` hoà). Train (1 083 mẫu): 641 vùng `en` (294 /
245 / 102 hoà), 6 827 vùng `vi` (1 900 / 3 589 / 1 338 hoà); `val_seen`: 173 vùng `en`, 1 791 vùng `vi`. Đây đúng là
các dòng mà `cv` và `export` dùng để học; `val_seen` chỉ dùng để đo. Kèm `regions.meta.json` (số dòng theo loại và nhãn,
10 mẫu chưa chấm bị bỏ).

**Phân tích lỗi trên `val_seen` và kiểm tra lại trên train.** Theo loại vùng (vùng có nhãn của `val_seen`): vùng có từ
tiếng Anh mới 87,6% đúng; vùng chữ Việt khác nhiều 74,9% (bỏ sót 180 vùng large-v3 đúng, ví dụ "chín lâm nay"/"95
này"); khác ít chữ cái 68,6%; RNNT thiếu từ 61,9% (15 lần lấy từ large-v3 tự thêm); chỉ khác dấu thanh 56,4% (luôn giữ
RNNT sẽ đúng 68%). Hai cách sửa rút ra từ đó (luôn giữ RNNT ở vùng chỉ khác dấu thanh; luôn giữ RNNT khi phía RNNT có từ
tiếng Anh mà large-v3 làm mất, kiểu "team"/"tim") được kiểm tra bằng quyết định ngoài phần trên train, không đo lại trên
`val_seen`: vùng chỉ khác dấu thanh (476 vùng) H-Việt 62,4%, luôn giữ RNNT 62,0%; vùng RNNT có từ tiếng Anh bị mất (93
vùng) H-Việt 71,0%, luôn giữ RNNT 61,3%; các vùng chữ Việt còn lại (4 920) H-Việt 74,0%, luôn giữ RNNT 65,8%. Cách sửa
thứ nhất không có lợi (khác biệt trên `val_seen` là dao động của 101 vùng), cách thứ hai sẽ làm hại; giữ `h_v2.json`.
Vùng chỉ khác dấu thanh vẫn gần như không được cải thiện vì điểm âm thanh của large-v3 không tách được dấu thanh; hướng
tiếp theo là thêm điểm âm thanh của chính RNNT (`chunkformer` có sẵn `Transducer._cal_transducer_score`), làm sau J6.

**Giới hạn.** Thời gian chấm điểm âm thanh tăng vì số vùng tăng khoảng 10 lần: trên T4, p50 1,0 s, p95 4,1 s mỗi mẫu
(so với dưới 1 s khi chỉ chấm vùng tiếng Anh); `score_texts` chấm từng câu ứng viên một, chấm theo lô chưa làm vì cần đo
trên GPU. J5 đo thời gian thật của `MergedASR` bản 2. `val_unseen` chưa đo (D53).

## 6d. VieGem-ASR dùng nguyên bản

**Mô hình.** `5CD-AI/VieGem-ASR` (commit `95eae70b`): CTC streaming 0,36B trên encoder âm thanh của
`google/embeddinggemma-2` (commit `f9972ac2`, Apache-2.0); đầu ra chữ thường, không dấu câu, số viết bằng chữ (như
RNNT). Tác giả báo WER tiếng Việt trung bình 6,34 trên 8 bộ, ViMedCSS (y tế, chêm tiếng Anh) 14,3, viết đúng 66% thuật
ngữ tiếng Anh. Giấy phép CC-BY-NC-SA-4.0 (phi thương mại) và điều khoản Gemma: dùng để nộp bài cần BTC xác nhận (SPEC
Q2). Ngân sách: chỉ thay được RNNT (hệ thống 5,93B) hoặc thay cả pipeline ghép (khoảng 4,4B khi bỏ whisper-large-v3);
thêm làm mô hình thứ ba là 6,04B, vượt giới hạn.

**Đã đọc mã nguồn trước khi dùng** (`asr/model.py`, `text.py`, `features.py`, `lm_decode.py`, app demo): không có `eval`,
`exec` hay gọi mạng ngoài tải mô hình. Ba chỗ dễ sai của code mẫu được xử lý trong engine `viegem`
(`lifemate/backends/asr_viegem.py`): `load_state_dict(strict=False)` bỏ qua im lặng khoá thiếu (engine báo lỗi khi có khoá
thiếu ngoài `encoder.` hoặc khoá thừa); `autocast(bfloat16)` không hợp với T4 (bf16 trên sm80 trở lên, float32 trên GPU
cũ hơn); `torch.load` đọc pickle (nạp bằng `weights_only=True`); code của repo được nạp ở đúng commit, mô hình gốc ở đúng
revision.

**Cấu hình theo tác giả, không dò:** không streaming; cắt đoạn ≤ 20 s tại chỗ im lặng nhất (như app); giải mã tham
lam, hoặc KenLM của tác giả với alpha 0,35, beta 1,5, chùm 32, `token_min_logp` −5, `beam_prune_logp` −10. Thêm gợi ý
từ theo ngữ cảnh (`bias_words` của chính bộ giải mã KenLM): từ tiếng Anh có trong ngữ cảnh mẫu, cùng nguồn với từ điển
ngữ cảnh của H-Anh; trọng số dò trên lưới đặt trước {0, 2, 4, 8}.

**Cách chạy** (`lifemate/eval/viegem_bench.py`): `check` (GPU) trên 10 file mẫu kèm lời chuẩn của Space tác giả; `dump`
(GPU, chạy mô hình một lần cho mỗi tập) lưu chữ tham lam và 32 token log xác suất cao nhất mỗi khung; `decode` (CPU,
KenLM là mô hình n-gram) giải mã lại với từng trọng số gợi ý mà không chạy lại mô hình; `score` so với RNNT và với
pipeline ghép bản 2 bằng bootstrap ghép cặp theo mẫu.

**Tiêu chí đặt trước:**

1. `check` đạt khi: mọi khoá thiếu thuộc `encoder.`, không có khoá thừa; WER tham lam trên 10 file mẫu (chuẩn hoá theo
   `text.normalize` của tác giả) dưới 25%; chữ float32 trùng chữ kiểu số thấp ở ít nhất 9/10 file. Không đạt thì dừng,
   tìm lỗi cấu hình, không chạy dữ liệu thi.
2. Chọn cấu hình trên `asr_tune` (thứ tự từ đơn giản: tham lam, KenLM không gợi ý, gợi ý 2, 4, 8): chỉ xét cấu hình có
   thời gian giải mã thêm p95 ≤ 1 s mỗi câu; lấy cấu hình có WER đã đổi số (`wer_digits`) thấp nhất; nếu CI của chênh lệch so với cấu hình đơn
   giản hơn chứa 0 thì lấy cấu hình đơn giản hơn.
3. Đo một lần trên `val_seen` với cấu hình đã chọn:
   - **Thay RNNT** khi WER câu thuần Việt thấp hơn RNNT có ý nghĩa, F1 tiếng Anh, F1 số và
     tham số thiếu không tệ hơn theo ngưỡng của mục 6f.3. Khi đó phải làm lại K5 và học lại H với VieGem làm phía A.
   - **Thay cả pipeline ghép** (bỏ whisper-large-v3, H-Anh, H-Việt): theo tiêu chí chung của mục 6f.

**Kết quả kiểm tra** (Colab T4, 01/11/2026, `viegem_bench check`): đạt. Mô hình nạp đủ khoá (mọi khoá thiếu thuộc
`encoder.`, không khoá thừa), 359 337 557 tham số. WER tham lam float32 trên 10 file mẫu của Space là 0,56%, dưới ngưỡng
25%. Float16 không chạy được trên T4: attention của encoder Gemma che mặt nạ bằng một giá trị vượt giới hạn float16 (lỗi
`value cannot be converted to type c10::Half without overflow`). Engine vốn dùng float32 trên GPU dưới sm80, nên điều kiện
"float32 trùng kiểu số thấp" không áp dụng trên T4; bước kiểm tra ghi lỗi này và đi tiếp. Giải mã KenLM chạy trong một
môi trường Python 3.12 riêng, vì KenLM không biên dịch được trên Python 3.13 của Colab.

## 6e. PhoWhisper-large-CS dùng nguyên bản

**Mô hình.** Bộ LoRA `rinhoooo/phowhisper-large-vien-cs-asr` (commit `a98f55e0`, CC-BY-4.0, r 32, alpha 64, 57,7M tham
số trên q, k, v, out, fc1, fc2) trên `vinai/PhoWhisper-large` (commit `b9136a44`, BSD-3, kiến trúc Whisper large-v2:
80 kênh mel, 32 + 32 tầng, từ vựng 51 865), fine-tune cho câu tiếng Việt chêm tiếng Anh. Gộp LoRA vào mô hình gốc thì vẫn
1,55B tham số: chỉ thay được whisper-large-v3 ở vai trò mô hình phụ B (ngân sách không đổi). Tác giả báo WER 4,56% trên
tập giữ lại của bộ dữ liệu chêm tiếng Anh do tác giả tự thu (chưa công bố), và kém PhoWhisper gốc trên tiếng Việt thuần (VIVOS 7,20%
so với 4,61%). Model card không nói dạng chữ lúc fine-tune (dấu câu, cách viết số, thẻ ngôn ngữ; chỉ ghi "language tags
stripped" khi chấm) và không nói cấu hình giải mã; `generation_config` gốc không ép ngôn ngữ.

**Cách dùng trong pipeline.** `lifemate/eval/phowhisper_cs_prepare.py` (chạy trên máy GPU): tải đúng commit, gộp LoRA
(`merge_and_unload`), lưu float16, chuyển sang CTranslate2 float16 kèm `tokenizer.json` và `preprocessor_config.json` (80
kênh mel, faster-whisper đọc từ file này). Như vậy mô hình chạy bằng engine `faster_whisper` sẵn có, dùng chung câu mồi,
lớp bảo vệ g4, bộ lọc câu bịa và chấm điểm âm thanh. Bước `--check-wavs` so chữ của bản gộp chạy trong transformers với
bản CTranslate2 và in chữ thô để thấy thẻ ngôn ngữ nếu có. Bộ chạy benchmark nhận `--whisper-model` (`asr_v3 transcribe`,
`score`) và biến thể câu mồi mới `N` (không câu mồi), vì bộ LoRA nhiều khả năng được học không có câu mồi.

**Cấu hình cố định, như large-v3 đang dùng:** ép `language = vi`, `task = transcribe`, tham lam (beam 1), nhiệt độ 0,
VAD (khoảng lặng 500 ms), `condition_on_previous_text = false`, bộ lọc câu bịa, lớp bảo vệ g4 khi có câu mồi. Chỉ so hai
biến thể câu mồi đặt trước: `N` và `E2`.

**Tiêu chí đặt trước:**

1. Kiểm tra chuyển đổi: chữ bản gộp (transformers) và bản CTranslate2 trùng nhau ở ít nhất 9/10 file; chữ thô không có
   thẻ ngôn ngữ, hoặc có thì phải viết bước lọc trước khi benchmark.
2. Trên `asr_tune` (chữ large-v3 E2 của phiên K1 làm mốc): chọn giữa `N` và `E2` theo F1 từ tiếng Anh (mục 6f), trừ khi
   WER câu thuần Việt tệ hơn có ý nghĩa; hoà thì chọn `N` (đơn giản hơn).
3. Thay large-v3 ở vai trò B khi, trên `val_seen` đo một lần, PhoWhisper-CS đứng riêng có F1 từ tiếng Anh cao hơn có
   ý nghĩa, hoặc không tệ hơn (ngưỡng mục 6f.3, cả WER thuần Việt và F1 số) mà số câu bị bộ lọc câu bịa cắt ít hơn. Khi đó làm lại K5 với PhoWhisper-CS (vừa chép phía B
   vừa chấm điểm âm thanh), học lại H-Anh và H-Việt, kiểm định chéo với tiêu chí của D67, rồi so pipeline ghép mới với
   bản 2.

**Kết quả kiểm tra** (Colab T4, 01/11/2026, 10 câu đầu của `asr_tune`): phép chuyển đổi đúng; tiêu chí 1 được làm rõ.

| So sánh | Số file chữ trùng |
|---|---|
| transformers (bản gộp) với faster-whisper (CTranslate2) | 6/10 |
| transformers với CTranslate2, cùng đặc trưng mel do transformers tính | 10/10 |

Tiêu chí 1 đặt trước so chữ của transformers với faster-whisper, tức là gộp hai nguồn khác nhau: trọng số sau chuyển đổi
và bộ tính đặc trưng mel (faster-whisper tự tính bằng cách riêng). Phép so thứ hai giữ đặc trưng cố định và cho 10/10,
nên trọng số sau gộp LoRA và chuyển sang CTranslate2 là đúng; 4 file lệch ở phép so đầu chỉ khác 1 đến 2 từ ("Điều giả"
và "Diều tả", "trong email" và "tramin"), do bộ tính đặc trưng. whisper-large-v3 trong pipeline cũng chạy qua đúng bộ tính
này, nên benchmark bằng faster-whisper vẫn so cùng điều kiện với large-v3. Chữ thô không có thẻ ngôn ngữ. Mô hình viết
có dấu câu, chữ hoa và **viết số bằng chữ** ("năm mươi nghìn", "sáu rưỡi"); dạng chung của số (mục 6f.1) xử lý được kiểu
này. Hai sửa lỗi môi trường trước khi chạy: gỡ `torchao` 0.10 có sẵn của Colab (peft từ chối chạy với bản này) và chép
`preprocessor_config.json`, `tokenizer.json` từ mô hình gốc (transformers 5 lưu cấu hình processor thành
`processor_config.json`).

## 6f. So chung bốn mô hình và pipeline ghép bản 2

**Câu hỏi.** Ngoài việc thay từng vai A hoặc B (mục 6d, 6e), cần biết một mô hình đứng riêng có thay được cả pipeline
ghép không. Nếu được, pipeline chỉ còn một mô hình nghe kèm bước hậu xử lý, không cần chấm điểm âm thanh và H-Anh, H-Việt.
Vì vậy năm hệ thống được chấm trên cùng tập câu, cùng thước đo: RNNT, whisper-large-v3 E2, VieGem, PhoWhisper-CS (mỗi mô
hình đứng riêng) và pipeline ghép bản 2.

**Hậu xử lý của một mô hình đứng riêng** giống phần bản 2 làm với từng phía trước khi ghép: `prepare` (chữ thường, bỏ dấu
câu hai đầu từ, đổi số đọc thành chữ số; `asr_compare --prepare`). Mô hình Whisper giữ thêm bộ lọc câu bịa và lớp bảo vệ
g4 (chữ đã lưu của large-v3 E2 đã qua hai bước này). VieGem là CTC, không bịa câu phụ đề và không dùng câu mồi, nên không
cần hai bước đó.

### 6f.1. Thước đo

| Thước đo | Đo gì | Vì sao cần |
|---|---|---|
| WER câu thuần Việt, đã đổi số (`vi_only_wer_digits`) | Tỉ lệ từ sai ở câu không có từ tiếng Anh | Chất lượng tiếng Việt |
| WER lượt cuối, đã đổi số (`final_wer_digits`) | Như trên, chỉ lượt cuối, mọi câu | Lượt cuối quyết định lời gọi tool; WER gộp có 642 lượt trước so với 274 lượt cuối |
| Recall (độ phủ), precision (độ chính xác), F1 từ tiếng Anh (`en_word_*`) | Recall: phần từ tiếng Anh của đáp án được chép đúng. Precision: phần từ tiếng Anh ASR viết ra là đúng, ở mọi câu kể cả câu thuần Việt. F1: trung bình điều hoà của hai số | Recall không phạt từ tiếng Anh viết thừa hoặc viết sai ("phòng" thành "phone") |
| Recall, precision, F1 của số (`number_*`) | So các từ có chữ số của đáp án và của chữ ASR trong cùng câu (đếm lặp) | Số quyết định ngày, giờ, số tiền, số điện thoại trong tham số |
| Tham số thiếu, mọi lượt (`entity_miss_rate_all`) | Giá trị tham số được nói ở bất kỳ lượt user nào mà chữ ASR không có | Bám sát điểm thi nhất, nhưng chỉ có 123 giá trị trên `val_seen` nên rất nhiễu |
| Thời gian mỗi mẫu p50, p95 (`asr_compare --timing`) | Tổng thời gian ASR của mọi lượt; bản 2 gồm RNNT, Whisper và chấm điểm âm thanh | 18% mẫu của agent đang hết giờ |

**Dạng chung của số.** Đáp án và chữ ASR viết số theo nhiều kiểu: "50.000", "50 nghìn", "năm mươi nghìn"; "9h30",
"chín giờ ba mươi"; "095-0536-0367"; bộ đổi số lại giữ nguyên chữ "hai" đứng riêng ("mua hai cái"), trong khi Whisper viết
"2". Mọi thước đo "đã đổi số" và thước đo số vì vậy đưa cả hai phía về cùng một dạng (`eval_asr.digitized_text`): viết mọi
chữ số thành chữ (`read_number`, kiểm tra đọc ngược đúng bằng `parse_number`), rồi đổi lại bằng đúng một bộ luật
(`digitize`). Chênh lệch còn lại là do nghe. Trước khi đưa về dạng chung, precision số của large-v3 chỉ là 45%, vì
"50.000" bị tách thành hai số và "2" không khớp "hai"; sau đó là 86%, và các chỗ lệch còn lại là nghe sai thật (25 thành
35, 15 thành 30).

Thước đo cũ (`wer`, `vi_only_wer`, `en_word_recall`, `entity_miss_rate` chỉ lượt cuối) vẫn được ghi để so với các lần
trước, nhưng không dùng để quyết định ở mục 6d, 6e, 6f.

### 6f.2. Mốc đã có

`val_seen`, chấm lại chữ đã lưu, không chạy mô hình. Phần trăm, chênh lệch so với bản 2 kèm CI 95% (khoảng tin cậy:
khoảng dao động của chênh lệch qua 1 000 lần lấy mẫu lại; chứa 0 nghĩa là chưa phân thắng thua):

| Hệ thống | WER thuần Việt | WER lượt cuối | F1 tiếng Anh | F1 số | Tham số thiếu, mọi lượt | Thời gian p50 / p95 (s) |
|---|---|---|---|---|---|---|
| Bản 2 (mốc) | 16,49 | 17,45 | 79,18 | 87,88 | 24,39 | 4,70 / 15,04 |
| RNNT đứng riêng | 17,06 [+0,06; +0,99] | 19,05 [+0,83; +2,33] | 55,56 [−29,58; −18,01] | 86,21 [−3,77; +0,35] | 28,46 [−1,69; +9,38] | 1,51 / 3,57 |
| large-v3 E2 đứng riêng | 24,27 [+6,45; +9,36] | 22,85 [+3,96; +7,00] | 72,24 [−10,28; −3,94] | 83,49 [−8,07; −1,07] | 28,46 [−1,59; +9,57] | 2,09 / 7,60 |

Chi tiết tiếng Anh: recall 76,42 / 49,85 / 74,93 và precision 82,14 / 62,74 / 69,75 (bản 2 / RNNT / large-v3). large-v3
chép được gần đủ từ tiếng Anh như bản 2 (CI của recall chứa 0) nhưng viết sai nhiều hơn rõ rệt; nếu chỉ nhìn recall thì
mô hình này bị đánh giá sai. Thời gian của bản 2 cộng ba bước chạy nối tiếp; một mô hình đứng riêng bỏ được bước chấm
điểm âm thanh, vốn chiếm phần lớn p95.

### 6f.3. Ngưỡng chấp nhận

Với 274 mẫu, CI rộng tới mức "CI chứa 0" không đủ để gọi là "không thua": một hệ thống tệ hơn thật vẫn lọt qua. Vì vậy
mọi điều kiện "không tệ hơn" ở mục 6d, 6e, 6f dùng ngưỡng đặt trước, áp lên cận phía tệ của CI của chênh lệch:

| Thước đo | "Không tệ hơn" khi | Bề rộng CI của chênh lệch trên mốc |
|---|---|---|
| WER thuần Việt, đã đổi số | cận trên ≤ +1,0 điểm | khoảng 1 đến 3 điểm |
| F1 tiếng Anh | cận dưới ≥ −3,0 điểm | khoảng 6 đến 12 điểm |
| F1 số | cận dưới ≥ −3,0 điểm | khoảng 4 đến 7 điểm |
| Tham số thiếu, mọi lượt | điểm đo không cao hơn mốc quá +2,0 điểm (không dùng CI: chỉ 123 giá trị, CI rộng khoảng 11 điểm) | khoảng 11 điểm |
| Thời gian p95 | không cao hơn mốc | (không bootstrap) |

"Tốt hơn có ý nghĩa" vẫn là CI của chênh lệch nằm hẳn về phía tốt. Theo bảng mốc, RNNT đứng riêng đạt ngưỡng WER (cận
trên +0,99) nhưng trượt F1 tiếng Anh, nên không thay được bản 2: ngưỡng cho kết luận đúng với điều đã biết.

### 6f.4. Tiêu chí đặt trước

Đo một lần trên `val_seen`, mốc là bản 2:

1. **Một mô hình thay cả pipeline** khi mô hình đó đứng riêng không tệ hơn bản 2 ở cả bốn thước đo của mục 6f.3 (WER
   thuần Việt, F1 tiếng Anh, F1 số, tham số thiếu) và không chậm hơn. Nếu cả VieGem và PhoWhisper-CS đều đạt, chọn mô
   hình có WER thuần Việt thấp hơn; CI chứa 0 thì chọn VieGem (ít tham số hơn, nhanh hơn).
2. **Không mô hình nào đạt bước 1:** giữ pipeline ghép, xét thay vai theo mục 6d (VieGem thay RNNT) và 6e (PhoWhisper-CS
   thay large-v3), với ngưỡng của mục 6f.3.
3. **Cả hai cùng thay được vai đã định:** dựng đúng một pipeline VieGem (A) + PhoWhisper-CS (B), chấm điểm âm thanh lại,
   học lại H-Anh và H-Việt một lần, kiểm định chéo theo tiêu chí D67, rồi so với bản 2. Không thử các cặp lẻ khác.
4. **Xác nhận bằng điểm thi:** thước đo ASR chỉ là đại diện (lần 3: recall tiếng Anh tăng 25,6 điểm chỉ đổi được +0,4
   điểm thi, CI chứa 0). Thay đổi được chọn ở bước 1, 2 hoặc 3 chỉ được nhận khi chạy agent trên `val_seen` với chữ của
   hệ thống mới và của bản 2 (cùng cấu hình agent, ghép cặp theo mẫu) không cho kết quả tệ hơn có ý nghĩa (CI của chênh
   lệch điểm thi không nằm hẳn dưới 0).
5. Bảng đủ năm hệ thống, kèm so từng cặp mô hình đứng riêng, được ghi để tham khảo; quyết định chỉ theo các bước trên.

**Lệnh chấm** (CPU, sau khi có chữ của hai mô hình mới):

```bash
python -m lifemate.eval.asr_compare --env local --ids configs/splits/val_seen.txt \
  --system ban2=<k5>/analysis/v2_val_seen.hyps.jsonl --system rnnt=<k2_val_seen>/rnnt.hyps.jsonl \
  --system v3_E2=<k2_val_seen>/v3_E2.hyps.jsonl --system viegem=<viegem>/val_seen/<cấu hình>.hyps.jsonl \
  --system phowhisper_cs=<phowhisper_cs>/<biến thể>.hyps.jsonl \
  --prepare rnnt --prepare v3_E2 --prepare viegem --prepare phowhisper_cs \
  --timing ban2=<k2_val_seen>/timing.jsonl:rnnt_s --timing ban2=<k2_val_seen>/timing.jsonl:whisper_s.E2 \
  --timing ban2=<k5>/scores_val_seen.timing.jsonl:score_s --timing rnnt=<k2_val_seen>/timing.jsonl:rnnt_s \
  --timing v3_E2=<k2_val_seen>/timing.jsonl:whisper_s.E2 --timing viegem=<viegem>/val_seen/timing.jsonl:seconds \
  --timing phowhisper_cs=<phowhisper_cs>/timing.jsonl:whisper_s.<biến thể> --out <thư mục>/standalone_val_seen.json
```

Thời gian giải mã KenLM của VieGem (nếu cấu hình chọn dùng KenLM) chạy trên CPU sau bước `dump`, nên được cộng thêm từ
kết quả `decode`.

## 7. Kế hoạch

| Bước | Việc | Cần GPU | Trạng thái |
|---|---|---|---|
| 1 | L1: bộ đổi số, test, đo trên chữ đã lưu (train, `val_seen` một lần) | Không | Xong 31/10: đạt trên train, cùng chiều trên `val_seen` (mục 3.4) |
| 2 | L2: danh sách ứng viên từ mẫu và catalog, bộ sửa tên, test, đo như L1 | Không | Xong 31/10: loại (15 lần thay trên 3 593 câu, 6 sai; trần 31%) |
| 3 | L3a: dựng dữ liệu SFT từ chữ ASR của train | Không | Xong 31/10 (D58) |
| 4 | `MergedASR` với H và L1 (phần code và test không cần GPU) | Không | Xong 31/10 (D59); L2 không nối vào (D57) |
| 5 | L0 và chạy lại J ở ChatML (J0, J1, ghép H, ghép H + L1 + L2) trên `val_seen` | Có | Chờ GPU |
| 6 | L3b: fine-tune và đo | Có | Chờ lead |
| 7 | Đo cả hệ thống, chốt, `val_unseen` một lần, nộp | Có | Chờ GPU |

*Chỉ số: `entity_miss_rate`: tỉ lệ giá trị tham số của lời gọi đáp án (chuỗi và số xuất hiện nguyên văn trong lời chuẩn) không có trong chữ ASR sau chuẩn hoá; WER: tỉ lệ từ sai trên tổng số từ của lời chuẩn; CI: khoảng tin cậy 95% bootstrap theo mẫu, 1 000 vòng; "có ý nghĩa": CI của chênh lệch không chứa 0.*
