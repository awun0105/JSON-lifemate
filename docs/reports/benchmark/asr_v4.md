# Báo cáo benchmark ASR lần 4: giá trị tham số trong chữ ASR (đổi số, sửa tên, dữ liệu fine-tune từ chữ ASR)

Ngày lập kế hoạch: 31/10/2026. Trạng thái: **L1 đã đo và nhận (mục 3.4); L2 đã đo và loại (mục 4.4); L3a xong (mục
5.2); code `MergedASR` xong (mục 6); L0, L3b và mọi việc cần GPU khác chưa chạy; L0, J1 xong (mục 2)**. Phần không cần GPU (L1, L2, L3a) làm trước trên máy dev; phần cần GPU ghi rõ và chờ phiên GPU.

Lần 3 ([`asr_v3.md`](asr_v3.md), mục "Nghiệm thu lần 3") kết thúc với ba số đếm trên `val_seen`:

1. Bộ quyết định ghép hoàn hảo chỉ đạt 80,3% độ phủ từ tiếng Anh (H đang 76,7%); 66 trên 335 từ không có ở bên nào.
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

Điểm thi chấm **lời gọi tool đúng tên và đúng giá trị tham số**. Ba lần trước đo chữ bằng WER và độ phủ từ tiếng Anh;
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
| **J6** | Chữ của J5 qua cùng agent | Run agent | Bộ chấm của BTC; so với L0v2 (ASR còn làm mất bao nhiêu) và với J1 (pipeline ghép lấy lại được bao nhiêu so với RNNT) |

J5 chạy riêng ASR trên T4 (hai mô hình ASR khoảng 3,6 GB), không cùng vLLM, vì T4 không chứa nổi cả hai; thời gian ASR
vì vậy là thời gian khi ASR có GPU riêng. Thời gian của cả hệ thống (ASR cùng vLLM, đọc trước) cần GPU từ 24 GB, đo sau.

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
