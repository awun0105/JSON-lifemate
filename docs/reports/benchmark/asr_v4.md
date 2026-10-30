# Báo cáo benchmark ASR lần 4: giá trị tham số trong chữ ASR (đổi số, sửa tên, dữ liệu fine-tune từ chữ ASR)

Ngày lập kế hoạch: 31/10/2026. Trạng thái: **L1 đã đo và nhận (mục 3.4); L2 đã đo và loại (mục 4.4); L3a đang làm;
L0, L3b và các việc cần GPU chưa chạy**. Phần không cần GPU (L1, L2, L3a) làm trước trên máy dev; phần cần GPU ghi rõ và chờ phiên GPU.

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

Trạng thái: chưa chạy.

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

L3a (không cần GPU): thêm vào bộ dựng SFT (`lifemate/train/sft_data.py`) lựa chọn lấy câu người dùng từ chữ ASR đã lưu
(RNNT, ghép H, sau L1 và L2 nếu đã chốt) thay cho chữ chuẩn, giữ nguyên lời gọi đáp án; mỗi mẫu có thể sinh cả hai bản
(chuẩn và ASR). Chỉ dùng chữ của `train` (chữ K2 của 1 093 mẫu đã có), không dùng val. L3b (cần GPU, phối hợp lead):
fine-tune với dữ liệu này và đo điểm thi trên `val_seen` so với mô hình học trên chữ chuẩn.

Trạng thái: chưa làm.

## 6. Việc kỹ thuật kèm theo (từ "Sau lần 3")

| Việc | Cần GPU | Trạng thái |
|---|---|---|
| Engine `MergedASR` (RNNT + large-v3 E2 + điểm âm thanh + H; trọng số H vào `configs/` có phiên bản; L1, L2 nối vào khi đạt tiêu chí) | Viết: không; đối chiếu với chữ offline: có | Chưa làm |
| Chạy agent ChatML thật trên `val_seen` (kiểm chứng D50, D51 trên vLLM) | Có | Chưa làm |
| Đo cả hệ thống trên GPU 24 GB: thời gian từng mẫu, số mẫu quá 30 giây, bộ nhớ | Có | Chưa làm |
| Chốt cấu hình nộp, đo `val_unseen` một lần, nộp public test một lần (D53) | Có | Chưa làm |

## 7. Kế hoạch

| Bước | Việc | Cần GPU | Trạng thái |
|---|---|---|---|
| 1 | L1: bộ đổi số, test, đo trên chữ đã lưu (train, `val_seen` một lần) | Không | Xong 31/10: đạt trên train, cùng chiều trên `val_seen` (mục 3.4) |
| 2 | L2: danh sách ứng viên từ mẫu và catalog, bộ sửa tên, test, đo như L1 | Không | Xong 31/10: loại (15 lần thay trên 3 593 câu, 6 sai; trần 31%) |
| 3 | L3a: dựng dữ liệu SFT từ chữ ASR của train | Không | Chưa làm |
| 4 | `MergedASR` với H, L1, L2 (phần code và test không cần GPU) | Không | Chưa làm |
| 5 | L0 và chạy lại J ở ChatML (J0, J1, ghép H, ghép H + L1 + L2) trên `val_seen` | Có | Chờ GPU |
| 6 | L3b: fine-tune và đo | Có | Chờ lead |
| 7 | Đo cả hệ thống, chốt, `val_unseen` một lần, nộp | Có | Chờ GPU |

*Chỉ số: `entity_miss_rate`: tỉ lệ giá trị tham số của lời gọi đáp án (chuỗi và số xuất hiện nguyên văn trong lời chuẩn) không có trong chữ ASR sau chuẩn hoá; WER: tỉ lệ từ sai trên tổng số từ của lời chuẩn; CI: khoảng tin cậy 95% bootstrap theo mẫu, 1 000 vòng; "có ý nghĩa": CI của chênh lệch không chứa 0.*
