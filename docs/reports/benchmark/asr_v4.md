# Báo cáo benchmark ASR lần 4: giá trị tham số trong chữ ASR (đổi số, sửa tên, dữ liệu fine-tune từ chữ ASR)

Ngày lập kế hoạch: 31/10/2026. Trạng thái: **kế hoạch; chưa chạy thử nghiệm nào**. Phần không cần GPU (L1, L2, L3a) làm
trước trên máy dev; phần cần GPU (L0, L3b, L4) ghi rõ và chờ phiên GPU.

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
| L2 sửa tên | So từng cụm chữ ASR với danh sách tên của mẫu (app đã cài, enum trong catalog tool, kết quả tool) bằng khoảng cách chuỗi không dấu; thay khi đủ gần và không có ứng viên thứ hai | Không | `entity_miss_rate`, độ phủ từ tiếng Anh, số lần sửa nhầm, WER câu thuần Việt |
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

Trên chữ RNNT và bản ghép H của train (kiểm định chéo để chọn kiểu viết và danh sách đơn vị) rồi `val_seen` một lần:
`entity_miss_rate`, WER mọi câu, WER câu thuần Việt, số câu có chữ số sau đổi so với lời chuẩn, số chỗ đổi sai (chữ số
sau đổi không có trong lời chuẩn).

Tiêu chí đặt trước: `entity_miss_rate` giảm có ý nghĩa so với mốc và WER câu thuần Việt không tăng có ý nghĩa. Đạt thì
L1 vào pipeline (cũng vào `MergedASR`).

Trạng thái: chưa làm.

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

Như L1, thêm: độ phủ từ tiếng Anh, số lần sửa nhầm, và tách riêng nhóm tên có trong mẫu / chỉ có trong catalog, để biết
phần nào đóng góp. Tiêu chí đặt trước: `entity_miss_rate` giảm có ý nghĩa, số sửa nhầm dưới 1 trên 10 lần sửa, WER câu
thuần Việt không tăng có ý nghĩa.

Trạng thái: chưa làm.

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
| 1 | L1: bộ đổi số, test, đo trên chữ đã lưu (train kiểm định chéo, `val_seen` một lần) | Không | Chưa làm |
| 2 | L2: danh sách ứng viên từ mẫu và catalog, bộ sửa tên, test, đo như L1 | Không | Chưa làm |
| 3 | L3a: dựng dữ liệu SFT từ chữ ASR của train | Không | Chưa làm |
| 4 | `MergedASR` với H, L1, L2 (phần code và test không cần GPU) | Không | Chưa làm |
| 5 | L0 và chạy lại J ở ChatML (J0, J1, ghép H, ghép H + L1 + L2) trên `val_seen` | Có | Chờ GPU |
| 6 | L3b: fine-tune và đo | Có | Chờ lead |
| 7 | Đo cả hệ thống, chốt, `val_unseen` một lần, nộp | Có | Chờ GPU |

*Chỉ số: `entity_miss_rate`: tỉ lệ giá trị tham số của lời gọi đáp án (chuỗi và số xuất hiện nguyên văn trong lời chuẩn) không có trong chữ ASR sau chuẩn hoá; WER: tỉ lệ từ sai trên tổng số từ của lời chuẩn; CI: khoảng tin cậy 95% bootstrap theo mẫu, 1 000 vòng; "có ý nghĩa": CI của chênh lệch không chứa 0.*
