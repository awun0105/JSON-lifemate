# Báo cáo benchmark ASR lần 2: câu tiếng Việt chêm tiếng Anh, thử nghiệm các câu mồi (initial_prompt)

Ngày: 29/10/2026. Trạng thái: **xong**. Cả bốn thử nghiệm đã có kết quả: A và D chấm offline; B và C chạy GPU trên
Kaggle T4 ở hai phiên v5a (PhoWhisper, ChunkFormer RNNT) và v5b (whisper-large-v3). Tóm tắt cho cả team ở ngay dưới.

Báo cáo trả lời ba câu hỏi:

1. Trong bốn mô hình ASR tốt nhất của lần 1 ([`docs/reports/benchmark/asr_v1.md`](asr_v1.md)), mô hình nào chép đúng nhất những câu
 tiếng Việt có chêm từ tiếng Anh?
2. Câu mồi (`initial_prompt`) chứa từ tiếng Anh, đi kèm một lớp bảo vệ, có giúp whisper-large-v3 và PhoWhisper chép
 đúng từ tiếng Anh hơn mà không làm hỏng câu thường không?
3. Ghép hai mô hình (ChunkFormer RNNT chép chính, Whisper bổ sung từ tiếng Anh) có cho chữ tốt hơn từng mô hình riêng
 lẻ không?

Phạm vi: chỉ đo ASR (chữ chép từ âm thanh). Không chạy LLM, không chấm điểm thi.

## Tóm tắt cho cả team

Phần này đọc riêng được, không cần đọc các mục sau. Thuật ngữ được giải thích ngay tại chỗ; định nghĩa đầy đủ ở mục 0.3.

### Pipeline của hệ thống

Mỗi mẫu thi gồm file âm thanh của lượt cuối, các lượt hội thoại trước đó, danh sách app đã cài, catalog tool của các app
đó và ngữ cảnh người dùng (giờ, vị trí, hồ sơ). Lượt người dùng trong lịch sử **chỉ có âm thanh**, nên cũng phải chép.
Hệ thống chạy hai bước, trong giới hạn 30 giây mỗi mẫu và tổng tham số không quá 6B:

```
âm thanh (lượt cuối + lượt người dùng trong lịch sử, tới 13 file)
  → ASR: mô hình chép thành chữ → bộ lọc ảo giác (asr_sanitizer) cắt câu phụ đề bịa và phần lặp
  → PromptBuilder: ghép chữ ASR với lịch sử, danh sách app, catalog tool, ngữ cảnh
  → LLM (Qwen3-4B, chạy bằng vLLM): chọn tool và tham số
  → lời gọi tool, ví dụ alarm.create(time="…05:30…")
```

Điểm thi chấm lời gọi tool, không chấm chữ. Nhưng chữ ASR sai từ khoá thì LLM không sửa được: nghe "Zalo" thành "giá lô"
thì LLM không biết phải mở Zalo.

### Lần 1 đã làm gì và để lại gì

Lần 1 ([`docs/reports/benchmark/asr_v1.md`](asr_v1.md)) so **năm mô hình ASR**: whisper-large-v3, PhoWhisper, ChunkFormer CTC, ChunkFormer RNNT và Parakeet.
Mỗi mô hình được dò cấu hình với cùng công sức trên tập dò `asr_tune`, rồi đo một lần trên hai tập kiểm tra. Thước đo là
chất lượng chung trên mọi câu: WER, tỉ lệ bỏ sót tên riêng, số câu rỗng, số câu bịa và thời gian.

| Mô hình (cấu hình thắng của lần 1) | WER (seen / unseen) | Bỏ sót tên riêng (seen / unseen) | Câu bịa (seen / unseen) | p95 trên T4 |
|---|---|---|---|---|
| ChunkFormer RNNT | **20,0% / 19,8%** | 32,1% / **48,1%** | 0 / 0 | 2,5 s |
| ChunkFormer CTC | 21,3% / 21,0% | **23,9%** / 55,6% | 0 / 0 | 2,3–2,7 s |
| PhoWhisper + câu mồi ngắn | 27,9% / 27,9% | 38,5% / 48,1% | 0 / 0 | 3,0 s |
| whisper-large-v3, không câu mồi | 30,6% / 30,2% | 31,2% / 57,4% | 12 / 5 | 7,9–8,2 s |
| Parakeet | 31,6% / 31,1% | 50,5% / 77,8% | 0 / 0 | 2,0 s |

Kết luận của lần 1: chọn **ChunkFormer RNNT** làm ASR chính, giữ PhoWhisper + câu mồi ngắn làm dự phòng, bỏ Parakeet.

Lần 1 để lại bốn điểm chưa rõ. Lần 2 sinh ra từ bốn điểm này:

| Điều lần 1 để lại | Đề xuất cải tiến | Thử nghiệm lần 2 |
|---|---|---|
| Chỉ đo trên mọi câu, chưa đo riêng **từ tiếng Anh**. Trong khi đó hơn một nửa số mẫu train có từ tiếng Anh, và những từ này thường quyết định tool nào được gọi | Tách tập con các câu có chêm tiếng Anh, thêm thước đo "chép đúng từ tiếng Anh", chấm lại chữ của lần 1 mà không chạy GPU | A |
| Với large-v3, **mọi câu mồi đều làm kết quả kém đi**, theo ba cơ chế: in lại câu mồi ("Chuyển sang kênh VTV3." xuất hiện ở 42 câu không ai nói), đếm tiếp danh sách ("VTV3 VTV4 … VTV21"), bịa câu phụ đề. Nhưng không có câu mồi thì large-v3 sai có hệ thống một số từ tiếng Anh, và "Zalo" chỉ đúng 1/19 | Viết câu mồi mới từ những từ large-v3 hay sai, tránh giống lời nói thật. Thêm **lớp bảo vệ**: câu nào in lại câu mồi, lặp vô hạn hoặc bịa thì chép lại một lần không câu mồi | B |
| Cấu hình large-v3 của lần 1 dùng nhiệt độ thử lại, tức là lấy mẫu ngẫu nhiên khi câu hỏng: **không tất định**, nhiều khả năng trái thể lệ | Đổi sang giải mã tham lam + VAD, chạy lại để kiểm tra chữ có giống hệt nhau không. Kiểm tra luôn việc chấm lại trên chữ cũ có dùng được như số chạy thật không | C |
| Mỗi mô hình mạnh một phần: RNNT viết tiếng Việt tốt nhất, Whisper biết tiếng Anh hơn. Lần 1 chỉ chọn **một** mô hình | **Ghép hai mô hình**: RNNT chép chính, Whisper chỉ bổ sung đoạn có từ tiếng Anh | D |

### Vì sao có benchmark này

Người dùng hay chêm từ tiếng Anh vào câu tiếng Việt: hơn một nửa số mẫu train của BTC có ít nhất một từ như "Zalo",
"SJC", "check", "voucher" hay tên người nước ngoài. Những từ này thường quyết định tool nào được gọi ("nhắn Zalo", "giá
vàng SJC"). ASR chép "Zalo" thành "giá lô" thì LLM gọi sai tool, dù phần tiếng Việt đúng hết.

Lần 1 đã chọn mô hình ASR theo chất lượng chung trên mọi câu. Lần 2 đo riêng việc chép **từ tiếng Anh**, và thử hai cách
cải thiện:

- **Câu mồi** (`initial_prompt`): một đoạn chữ đưa cho mô hình Whisper trước khi nghe, đóng vai "câu nói trước đó".
  Whisper bắt chước cách viết trong câu mồi, nên câu mồi chứa sẵn "Zalo", "SJC"… có thể giúp mô hình viết đúng các từ đó.
  Câu mồi đi kèm một **lớp bảo vệ**: câu nào có dấu hiệu bị câu mồi làm hỏng (in lại nguyên câu mồi thay vì nghe, lặp
  vô hạn, bịa câu) thì được chép lại một lần không có câu mồi.
- **Ghép hai mô hình**: lấy chữ của ChunkFormer RNNT (giỏi tiếng Việt), chỉ thay những đoạn mà whisper-large-v3 (giỏi
  tiếng Anh hơn) nghe ra một từ tiếng Anh mà RNNT không có.

### Dữ liệu cần chuẩn bị và cách chuẩn bị

**Nguồn dữ liệu.** Chỉ có một nguồn có đáp án: phần `train` trong gói dữ liệu của BTC (1 545 mẫu, 5 036 câu người dùng,
mỗi câu có file âm thanh và chữ chuẩn). Phần `public_test` không có đáp án nên không chấm được. Gói dữ liệu nằm ở
[`data/train+public_test/`](../../../data/train+public_test/) trên máy, không có trên git.

**Nguyên tắc chuẩn bị.** Dữ liệu dùng để **chọn** (chọn từ cho câu mồi, chọn cấu hình) phải khác dữ liệu dùng để **đo** kết
quả cuối. Giống luyện đề và thi thật: nếu đề thi là đề đã luyện thì điểm cao hơn thực lực. Vì vậy 1 545 mẫu được chia
thành các phần không chồng nhau:

Mọi bước chia do một script làm: [`lifemate/eval/make_splits.py`](../../../lifemate/eval/make_splits.py), chạy bằng `make splits` (tương đương
`python -m lifemate.eval.make_splits --smoke 30 --asr-tune 300 --en-mixed`). Script không sao chép dữ liệu: script chỉ
ghi các file `configs/splits/*.txt`, mỗi dòng một mã mẫu. Lúc chạy thử nghiệm, [`lifemate/io/data.py`](../../../lifemate/io/data.py) › `read_ids` (file
`.txt` → danh sách mã) và `load_split` (thư mục dữ liệu, tên split, danh sách mã → các mẫu) lấy đúng các mẫu đó từ
[`data/train+public_test/train/train.jsonl`](../../../data/train+public_test/train/train.jsonl) của BTC. Các bước chạy tuần tự, bước sau dùng kết quả bước trước. Sơ đồ dưới ghi cả lần benchmark
dùng từng tập:

```
train.jsonl (1 545 mẫu, dữ liệu gốc của BTC)
 │
 ├─1. Tách app lạ ──────────────► val_unseen.txt (178)
 │                                  · lần 1: đo cấu hình thắng của 5 mô hình, một lần
 │                                  · lần 2: đo cách câu mồi thắng và cách ghép, một lần
 │
 └─ còn 1 367 mẫu
     │
     2. Gom câu gần trùng thành nhóm (không ghi file riêng; số liệu ghi trong meta.json)
     │
     3. Xếp nguyên nhóm vào 5 phần ──► fold_0..4.txt
     │                               ├─ phần 0 ──► val_seen.txt (274)
     │                               │              · lần 1, 2: như val_unseen
     │                               └─ phần 1–4 ► train.txt (1 093)
     │                                              · không dùng trực tiếp; được chia tiếp ở bước 5
     │
     4. Rút từ val_seen ─────────────► smoke.txt (30)
     │                                  · chạy thử nhanh để bắt lỗi, không dùng để chấm
     │
     5. Rút từ train ───────────────► asr_tune.txt (300) = tập dò
     │                               │  · lần 1: dò cấu hình từng mô hình
     │                               │  · lần 2: thử 13 cách dùng câu mồi, chọn cách ghép
     │                               └─ 793 mẫu train còn lại (không ghi file riêng)
     │                                  · lần 2: đếm từ tiếng Anh large-v3 hay sai để viết
     │                                          câu mồi en_full_v1 (dùng chữ large-v3 có sẵn
     │                                          trong data/asr/train_asr.json)
     │
     6. Lọc mẫu có câu chêm tiếng Anh, làm riêng trên từng tập
     │  (đọc chữ đáp án của BTC, ở câu cuối và các câu người dùng trong lịch sử):
     │
     │    asr_tune.txt   (300) ──lọc──► asr_tune_en_mixed.txt   (165)
     │    val_seen.txt   (274) ──lọc──► val_seen_en_mixed.txt   (152)
     │    val_unseen.txt (178) ──lọc──► val_unseen_en_mixed.txt  (95)
     │
     │                                  · lần 2: chấm riêng tỉ lệ chép đúng từ tiếng Anh
     │                                    trên các mẫu này; mô hình vẫn chép toàn bộ tập gốc
     │
     7. Ghi số liệu tổng hợp ───────► meta.json
                                        · kiểm tra không có mẫu trùng giữa các tập
```

Giải thích từng bước:

1. **Tách app lạ.** Mẫu nào có lời gọi tool đúng dùng một trong 6 app (medication, fitness, flights, bike_rental,
   library, subscriptions), cộng vài mẫu "ngoài phạm vi" liên quan tới các app đó được gắn sẵn trong code
   (`UNSEEN_OUT_OF_SCOPE`), vào `val_unseen`. Lý do: bộ thi riêng của BTC có app chưa gặp. Tách trước khi chia nên không
   mẫu nào của `val_unseen` lọt vào `train`.
2. **Gom câu gần trùng** (`cluster_near_duplicates`): các mẫu có chữ đáp án giống nhau từ 85% trở lên thành một nhóm
   (64 nhóm, 174 mẫu; các mẫu khác đứng một mình). Đây là bước trung gian cho bước 3 nên không ghi file danh sách.
3. **Chia 5 phần** (`stratified_group_folds`): xếp nguyên cả nhóm vào một phần, sao cho các phần có cùng tỉ lệ số lời
   gọi tool và tỉ lệ câu hỏi lại. Nhờ vậy hai câu gần giống nhau không nằm ở hai tập khác nhau.
4. **`smoke`** (`--smoke 30`): 30 mẫu của `val_seen`, phân tầng theo loại câu trả lời và có/không lịch sử.
5. **`asr_tune`** (`--asr-tune 300`): 300 mẫu của `train`, cùng cách phân tầng. 793 mẫu `train` còn lại chưa có file
   danh sách riêng ở lần 2; lần 2 chỉ đọc chữ large-v3 có sẵn của các mẫu này để chọn từ cho câu mồi.
6. **`*_en_mixed`** (`--en-mixed`, hàm `en_mixed_ids`, thêm ở lần 2): giữ mẫu có ít nhất một câu người dùng chêm từ
   tiếng Anh, theo [`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py) › `has_foreign_word` (câu → có từ ngoại hay không; luật ở mục 2.1).
   Việc lọc do script làm; luật đã được rà tay (mục 2.1). Mỗi file `_en_mixed` nằm trọn trong tập gốc của nó.
7. **[`configs/splits/meta.json`](../../../configs/splits/meta.json)**: số mẫu từng tập, tỉ lệ nhãn từng phần, số nhóm gần trùng, và số mẫu trùng giữa `train` với hai tập
   kiểm tra (đều bằng 0).

| Phần dữ liệu | Số mẫu (số câu) | Dùng để | Vì sao tách riêng |
|---|---|---|---|
| 793 mẫu train còn lại | 793 | Đếm từ tiếng Anh large-v3 hay sai, rồi đưa các từ đó vào câu mồi | Nếu lấy từ trên tập dò hay tập kiểm tra thì câu mồi "biết trước đề" |
| Tập dò `asr_tune` | 300 (1 024 câu) | Thử 13 cách dùng câu mồi, chọn cách tốt nhất | Chọn trên tập này thì số của tập này luôn đẹp hơn thực tế, nên cần tập khác để đo |
| Tập kiểm tra `val_seen` | 274 (916 câu) | Đo kết quả cuối, một lần, trên app đã biết | Chưa dùng để chọn gì |
| Tập kiểm tra `val_unseen` | 178 (527 câu) | Đo kết quả cuối, một lần, trên app chưa gặp | Bộ thi thật có app lạ, kéo theo từ tiếng Anh lạ |

Trong mỗi tập, benchmark chấm riêng **các câu có chêm tiếng Anh** (165, 152 và 95 mẫu ở tập dò, `val_seen`, `val_unseen`).
Một quy tắc tự động nhận ra từ tiếng Anh trong câu (mục 2.1), nên không cần đánh dấu tay.

**Cách chuẩn bị.**

1. Đặt gói dữ liệu của BTC vào [`data/train+public_test/`](../../../data/train+public_test/).
2. Chạy `make splits`. Lệnh này chia dữ liệu như sơ đồ trên và ghi danh sách mã mẫu của từng phần vào [`configs/splits/`](../../../configs/splits/).
   Lệnh dùng số ngẫu nhiên cố định, nên máy nào chạy cũng ra đúng cùng danh sách. Trong lúc chia, các câu gần giống nhau
   được giữ chung một phần, để tập kiểm tra không chứa câu "đã gặp" ở train. Danh sách không đưa lên git (dữ liệu thi),
   nên mỗi máy mới clone repo phải chạy lại lệnh này.
3. Chữ large-v3 của 793 mẫu train đã có sẵn trong [`data/asr/train_asr.json`](../../../data/asr/train_asr.json), không cần chạy lại mô hình.
4. Chữ ASR của lần 1 (thư mục [`artifacts/runs/kaggle/asr/10-28_tune_v3/`](../../../artifacts/runs/kaggle/asr/10-28_tune_v3/)) được dùng lại để chấm so sánh bốn mô hình
   và thử phép ghép mà không cần GPU. Chỉ các lần thử câu mồi và lần chạy lại để kiểm chứng (mục 5) phải chạy GPU trên Kaggle.

Lần 2 dùng lại đúng cách chia của lần 1, nên kết quả hai lần so được với nhau. Chi tiết và lý do từng bước ở mục 2.2.

### Các câu mồi đã thử

Có bốn câu mồi, mỗi câu là một file chữ trong [`configs/asr_prompts/`](../../../configs/asr_prompts/). Câu mồi ngắn là câu của lần 1, giữ lại để so sánh.
Ba câu mồi mới được viết cho lần 2 và **lồng nhau**: câu dài hơn bằng câu ngắn hơn cộng thêm vài câu phía trước. Nhờ vậy,
kết quả trả lời thẳng câu hỏi "đưa nhiều từ hơn vào câu mồi có tốt hơn không".

Từ đưa vào câu mồi là những từ tiếng Anh mà large-v3 hay chép sai, đếm trên phần train không dùng để dò hay kiểm tra (mục
2.2.4). Câu mồi là câu tự viết, không chép câu nào của dữ liệu thi. Lý do: lớp bảo vệ coi câu ASR có cụm từ trùng câu mồi
là "in lại câu mồi", nên câu mồi giống lời nói thật thì câu nói đúng cũng bị coi là hỏng. Từ hay sai nhất (Zalo, SJC, SMS,
check) đặt cuối, vì phần cuối gần đoạn âm thanh nhất.

| Câu mồi | Số từ khoá | Độ dài (token của large-v3) | Nội dung |
|---|---|---|---|
| `short_v1` (lần 1) | 2 | 18 | Nhắn Zalo cho Lan. Chuyển sang kênh VTV3. |
| `en_top10_v1` | 10 | 56 | Ok, robot đồ chơi hết pin. File ảnh Sarah giữ, tối book bàn online. Voucher nằm trong SMS, check kỹ rồi gửi Zalo cả xóm, coi miếng SJC luôn. |
| `en_top20_v1` | 21 | 106 | Tivi phát cả list chill. Camera gara thấy shipper, mail báo tour biển và workshop gốm của Greenhaven Foods. Brightpath tặng điểm, MinhMart giảm sâu. *(rồi toàn bộ `en_top10_v1`)* |
| `en_full_v1` | 38 | 183 | QR trên app mờ quá. Email của Grace Nakamura tới, Peter Hale mượn laptop chạy demo cho Nexora Labs, nhớ viết note. Điểm MinhPoint cộng chậm, podcast vô playlist, meeting dời vì deadline. Xăng RON, bóng đá VTV với HTV. *(rồi toàn bộ `en_top20_v1`)* |

Whisper chỉ dùng 223 token cuối của câu mồi, nên `en_full_v1` (183 token) đã gần chạm trần.

**Lớp bảo vệ có ba mức để thử.** Đôi khi Whisper không nghe mà in lại một đoạn của câu mồi. Lớp bảo vệ nhận ra trường hợp
này bằng cách so chữ ASR với câu mồi:

- **Không bảo vệ:** giữ nguyên chữ ASR chép có câu mồi.
- **g3:** chữ ASR có 3 từ liền nhau giống hệt 3 từ liền nhau trong câu mồi thì coi là "in lại câu mồi", và chép lại câu đó
  một lần không có câu mồi.
- **g4:** giống g3 nhưng phải trùng 4 từ liền nhau. g4 khó bị kích hoạt hơn, nên ít bắt nhầm câu nói đúng hơn.

Ví dụ với câu mồi `short_v1` ("Nhắn Zalo cho Lan…"): người dùng nói thật "nhắn Zalo cho mẹ" và ASR chép đúng. Chữ ASR có 3
từ "nhắn Zalo cho" trùng câu mồi, nên g3 bắt nhầm câu này và chép lại. g4 không bắt, vì chỉ trùng 3 từ.

**Có 13 cách dùng câu mồi để so.** 4 câu mồi × 3 mức bảo vệ = 12 cách, cộng thêm cách không dùng câu mồi là 13. Mỗi mô hình
Whisper (large-v3 và PhoWhisper) chạy cả 13 cách.

**Chọn trên tập dò, đo trên tập kiểm tra.** Cả 13 cách chạy trên tập dò (1 024 câu, xem phần dữ liệu ở trên), rồi chọn cách
tốt nhất. Chỉ cách tốt nhất được chạy thêm một lần trên hai tập kiểm tra để lấy số cuối. Lý do: khi chọn cái tốt nhất trong
13 cái trên cùng một tập, cái thắng thường có phần may, nên số trên tập dò đẹp hơn thực tế. Ví dụ, câu mồi tốt nhất của
PhoWhisper tăng 6,6 điểm trên tập dò nhưng gần như không tăng trên tập kiểm tra.

Kết quả trên tập dò, dòng tốt nhất của mỗi câu mồi:

| Câu mồi | large-v3: chép đúng từ tiếng Anh | large-v3: lỗi từ câu thuần Việt | large-v3: số lần chèn từ không được nói | PhoWhisper: chép đúng từ tiếng Anh |
|---|---|---|---|---|
| Không câu mồi | 47,7% | 29,9% | 0 | 31,0% |
| `short_v1` không bảo vệ | 51,8% | 34,7% (21 câu bịa) | 46 | 33,0% |
| `short_v1` + g4 | 52,5% | 30,2% | 6 | 33,0% |
| `en_top10_v1` + g4 | 62,7% | 29,6% | 18 | 35,5% |
| `en_top20_v1` + g4 | 66,0% | 29,2% | 29 | **37,6%** |
| **`en_full_v1` + g4** | **70,6%** | 29,5% | 26 | 34,5% |

"Chèn từ không được nói" đếm số câu mà ASR viết ra một từ của câu mồi trong khi người dùng không nói từ đó (ví dụ "Mai"
thành "Mail").

**Câu mồi tốt nhất: `en_full_v1` + lớp bảo vệ g4, dùng cho whisper-large-v3.** Kết luận:

- **Với large-v3, câu mồi càng dài càng tốt.** Mỗi lần thêm từ, tỉ lệ chép đúng từ tiếng Anh tăng: 48% → 53% → 63% → 66% →
  71% trên tập dò. Câu mồi dài còn làm large-v3 ít bịa câu hơn. Trên hai tập kiểm tra, `en_full_v1` + g4 vẫn giữ được mức
  tăng: 52% → 72% (seen) và 56% → 61% (unseen).
- **Câu mồi chỉ giúp các từ có trong câu mồi.** Trên seen, từ có trong câu mồi được chép đúng từ 48% lên 84%. Từ không có
  trong câu mồi gần như không đổi. Vì vậy mức tăng trên unseen nhỏ hơn: app lạ dùng nhiều từ không có trong câu mồi.
- **Câu mồi ngắn của lần 1 là lựa chọn tệ.** Không có lớp bảo vệ, `short_v1` làm câu thuần Việt sai thêm gần 5 điểm và
  sinh 21 câu bịa. Lý do là câu mồi này quá giống lời nói thật ("nhắn Zalo cho…"). Lớp bảo vệ kéo lỗi về mức không câu
  mồi, nhưng gần như không còn lợi ích.
- **Cần dùng lớp bảo vệ g4, không cần g3.** g4 không thua g3 ở tổ hợp nào. g3 bắt cả những câu nói đúng mà có 3 từ trùng
  câu mồi.
- **Với PhoWhisper, không câu mồi nào đáng dùng.** `en_top20_v1` tốt nhất trên tập dò (+6,6 điểm). Trên hai tập kiểm tra,
  mức tăng gần bằng 0 (+0,9 và 0,0 điểm, nằm trong sai số). Câu mồi dài hơn còn làm PhoWhisper kém đi. PhoWhisper đã được
  huấn luyện thêm trên tiếng Việt, nên khi nghe "SJC" hay "check", mô hình vẫn viết thành chữ Việt gần âm, câu mồi không
  kéo lại được.
- **Rủi ro còn lại là chèn từ.** Với `en_full_v1` + g4, large-v3 chèn từ của câu mồi vào 26 trên 1 024 câu ở tập dò.
  Không có câu mồi thì chỉ có 3 câu như vậy. Cách sửa là thêm bộ lọc cho phép ghép (mục "Việc tiếp theo").

Chi tiết cách viết câu mồi ở mục 4.3.1. Bảng đủ 13 tổ hợp ở mục 4.7.1 (PhoWhisper) và 4.7.6 (large-v3).

### Kết quả chính

Bốn mô hình ASR được so là bốn mô hình tốt nhất của lần 1: **ChunkFormer RNNT** và **ChunkFormer CTC** (hai mô hình tiếng
Việt cùng một dòng mô hình, khác cách sinh chữ), **PhoWhisper** (Whisper được VinAI huấn luyện thêm trên tiếng Việt) và
**whisper-large-v3** (Whisper nhiều ngôn ngữ của OpenAI). Chỉ hai mô hình Whisper nhận được câu mồi.

Đo trên các câu có chêm tiếng Anh của hai tập kiểm tra: **seen** (app đã có trong dữ liệu học) và **unseen** (app chưa gặp,
giống phần app lạ của bộ thi thật). "Chép đúng từ tiếng Anh" là tỉ lệ từ tiếng Anh của đáp án mà ASR viết đúng (cao là
tốt). "Tỉ lệ lỗi từ" (WER) là số từ sai chia cho số từ của câu (thấp là tốt).

| Hệ thống | Chép đúng từ tiếng Anh (seen / unseen) | Tỉ lệ lỗi từ trên câu chêm tiếng Anh (seen / unseen) |
|---|---|---|
| ChunkFormer RNNT (mô hình ASR đang tốt nhất từ lần 1) | 49,9% / 51,6% | 24,8% / 24,1% |
| ChunkFormer CTC | 34,6% / 39,7% | 27,5% / 26,9% |
| PhoWhisper | 36,1% / 36,5% | 32,5% / 33,1% |
| whisper-large-v3, không câu mồi | 51,6% / 56,0% | 32,8% / 33,9% |
| whisper-large-v3 + câu mồi tốt nhất (`en_full_v1` + g4) | 72,2% / 61,1% | 28,7% / 33,5% |
| Ghép RNNT + whisper-large-v3 có câu mồi, không lọc | 72,5% / 63,9% | 22,8% / 22,8% |
| **Ghép RNNT + whisper-large-v3 có câu mồi, có bộ lọc ngữ cảnh** | **72,2% / 63,1%** | **22,1% / 22,8%** |

Đọc bảng:

1. **RNNT vẫn là mô hình đơn tốt nhất.** RNNT chép đúng khoảng một nửa số từ tiếng Anh, và phần tiếng Việt tốt hơn hẳn
   các mô hình khác. Không mô hình Whisper nào thay được RNNT: phần tiếng Việt của Whisper sai nhiều hơn khoảng 8–12
   điểm phần trăm.
2. **Câu mồi giúp whisper-large-v3 rõ rệt, không giúp PhoWhisper.** Câu mồi tốt nhất là `en_full_v1` + lớp bảo vệ g4
   (mục "Các câu mồi đã thử" ở trên). Với large-v3, tỉ lệ chép đúng từ tiếng Anh tăng từ
   khoảng 52% lên 72% trên seen. Phần tăng gần như chỉ đến từ các từ **có trong câu mồi**. Vì vậy trên unseen, nơi chỉ
   một phần tư số từ tiếng Anh có trong câu mồi, mức tăng nhỏ hơn nhiều (56% lên 61%). Với PhoWhisper, câu mồi gần như
   không đổi kết quả trên tập kiểm tra.
3. **Ghép RNNT + large-v3 là hướng tốt nhất hiện có.** So với RNNT đứng riêng, bản ghép có bộ lọc ngữ cảnh chép đúng thêm 22,3 điểm
   (seen) và 11,5 điểm (unseen) từ tiếng Anh, sai ít từ hơn trên câu chêm tiếng Anh (22,1% so với 24,8% trên seen), và
   tỉ lệ lỗi từ tính trên mọi câu cũng thấp hơn RNNT (19,4% so với 20,0% trên seen). Các chênh lệch này đã được kiểm tra
   thống kê (khoảng tin cậy 95% không chứa 0, mục 2.4), không phải do may rủi.
4. **Cái giá còn lại: câu thuần Việt tệ đi rất ít.** Với bản ghép không lọc, tỉ lệ lỗi từ trên câu không có tiếng Anh tăng 1,1
   điểm (seen, 17,7% lên 18,8%) và 0,6 điểm (unseen), vì phép ghép lấy cả những từ tiếng Anh mà large-v3 viết ra dù
   người dùng không nói. Bộ lọc ngữ cảnh giảm phần lỗi tăng thêm này còn 0,4 điểm (seen, 18,1%) và 0,3 điểm (unseen), mà gần như
   không mất phần tiếng Anh (mục "Quy tắc ghép hai mô hình và bộ lọc ngữ cảnh" ở dưới).
5. **Kết quả tin được.** Chạy lại RNNT (2 467 câu) và large-v3 (1 024 câu) trên một máy khác cho ra chữ giống hệt từng ký
   tự (mục 5). Vì vậy số chấm lại trên chữ cũ dùng được như số chạy thật, và cấu hình large-v3 đạt yêu cầu "kết quả lặp
   lại được" của thể lệ. Quy tắc tìm từ tiếng Anh cũng đã được rà tay trên 2 467 câu: đúng 99,7% khi nhận và tìm được
   94,3% số từ tiếng Anh. Chấm lại theo nhãn rà tay, mọi tỉ lệ chép đúng lệch tối đa 1,1 điểm và thứ hạng không đổi.

**Chi phí.** Ghép hai mô hình cộng thêm whisper-large-v3: tổng tham số RNNT 0,11B + large-v3 1,55B + LLM 4,02B = 5,68B,
dưới trần 6B của thể lệ. Riêng large-v3 chép xong 95% số **mẫu** trong khoảng 8,7 giây trên GPU T4. Một mẫu gồm lượt
cuối và mọi lượt người dùng trong lịch sử, tới 13 file âm thanh, và thể lệ cho 30 giây mỗi mẫu cho cả ASR lẫn LLM. Chưa
đo trên A30 và chưa đo khi chạy cùng LLM.

### Quy tắc ghép hai mô hình và bộ lọc ngữ cảnh

**Vấn đề.** RNNT chép tiếng Việt tốt nhất nhưng chỉ đúng khoảng một nửa số từ tiếng Anh. large-v3 có câu mồi chép tiếng
Anh tốt hơn nhiều nhưng phần tiếng Việt sai nhiều hơn. Cần một bản chữ lấy được điểm mạnh của cả hai.

**Quy tắc hiện tại** ([`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `merge_transcripts`, chi tiết ở mục 6.3):

1. Lấy chữ của RNNT làm bản chính.
2. Đặt hai bản chữ cạnh nhau, từ ứng với từ, và tìm những đoạn hai bản viết khác nhau.
3. Ở mỗi đoạn khác nhau, thay bằng chữ của large-v3 khi đủ hai điều kiện: đoạn của large-v3 có một từ tiếng Anh mà
   đoạn của RNNT không có, và đoạn đó không dài quá gấp đôi đoạn của RNNT cộng 2 từ (để chặn câu bịa dài). Còn lại giữ
   chữ RNNT.

| Người dùng nói | RNNT | large-v3 | Bản ghép | Đúng? |
|---|---|---|---|---|
| nhắn Zalo cho mẹ | nhắn **giá lô** cho mẹ | nhắn **Zalo** cho mẹ | nhắn Zalo cho mẹ | Đúng: đây là lợi ích chính |
| mai đi khám | **mai** đi khám | **Mail** đi khám | Mail đi khám | Sai: large-v3 bị câu mồi kéo, quy tắc lấy theo |

**Vì sao cần lọc.** Những ca như ca thứ hai làm câu thuần Việt sai thêm: tỉ lệ lỗi từ tăng 1,1 điểm trên seen so với RNNT. Ở câu thuần
Việt, gần như mọi câu bị phép ghép đổi đều tệ hơn (37/44 câu trên seen). Từ bị lấy nhầm có hai loại: từ của câu mồi
("Mail" cho "mai", "list" cho "lịch") và từ large-v3 tự viết theo kiểu tiếng Anh dù không có trong câu mồi ("Toyota
D-Diều" cho "tôi ra đê"). Loại thứ hai nhiều hơn hẳn: trên tập dò là 124 trong 148 từ bị lấy nhầm. Từ chen vào như vậy có
thể làm LLM gọi sai tool.

**Khó khăn chính.** Nếu chỉ nhìn chữ, hai ca trong bảng giống hệt nhau: RNNT ra chữ Việt gần âm, large-v3 ra một từ của
câu mồi. Một điều kiện như "không lấy khi RNNT có chữ Việt hợp lệ" sẽ chặn luôn ca "Zalo" và mất phần lợi chính (+22,7
điểm chép đúng từ tiếng Anh trên seen). Vì vậy bộ lọc phải dựa vào **tín hiệu ngoài chữ**:

| Cách lọc | Ý tưởng | Cần gì | Rủi ro |
|---|---|---|---|
| Từ điển ngữ cảnh của mẫu (**đã làm**) | Chỉ nhận từ tiếng Anh khi từ đó có trong ngữ cảnh của mẫu: catalog của app đã cài, ngữ cảnh người dùng, lượt trợ lý trong lịch sử | Chạy offline được | Mẫu không có lịch sử có ít ngữ cảnh |
| Độ tin cậy riêng của từng từ | Đếm trên phần train chưa dùng xem mỗi từ của câu mồi bị chèn nhầm thường không ("Mail", "list" hay bị; "SJC" ít khi). Từ hay bị chèn nhầm thì cần thêm bằng chứng | Chữ large-v3 có câu mồi trên phần train đó (một phiên GPU) | Ít mẫu thì ước lượng không chắc |
| So với lần chép không câu mồi | Từ chỉ xuất hiện khi có câu mồi, còn không câu mồi thì large-v3 nghe ra chữ Việt, thì nhiều khả năng là chèn nhầm | Chạy large-v3 hai lần | Gấp đôi thời gian chạy large-v3 |
| Bộ phân loại học từ dữ liệu | Thay luật viết tay bằng một mô hình nhỏ (hồi quy logistic) quyết định từng đoạn, dùng các tín hiệu trên làm đầu vào | Nhãn tự động từ đáp án train | Phải tách dữ liệu học khỏi tập dò và tập kiểm tra |

**Cách chọn.** Mọi cách lọc được thử và chọn trên tập dò. Sau đó cách thắng được đo đúng một lần trên hai tập kiểm tra.
Không chỉnh quy tắc theo các ca hỏng đã thấy trên tập kiểm tra (mục 6.5), vì như vậy tập kiểm tra không còn là đề chưa
gặp.

**Kết quả: đã thử cách đầu tiên, từ điển ngữ cảnh** (chi tiết ở mục 6.8). Mỗi mẫu có một từ điển gồm các từ có sẵn dạng
chữ trong đầu vào lúc thi: thông tin của các app đã cài trong catalog, ngữ cảnh người dùng, và lượt trợ lý trong lịch sử.
Phép ghép chỉ nhận một từ tiếng Anh **không có trong câu mồi** khi từ đó có trong từ điển này. Từ có trong câu mồi vẫn
nhận như cũ.

| Hệ thống | Chép đúng từ tiếng Anh (seen / unseen) | Lỗi từ câu thuần Việt (seen / unseen) | Lỗi từ mọi câu (seen / unseen) |
|---|---|---|---|
| RNNT đứng riêng | 49,9% / 51,6% | 17,7% / 17,5% | 20,0% / 19,8% |
| Ghép không lọc | 72,5% / 63,9% | 18,8% / 18,2% | 20,1% / 19,8% |
| **Ghép có bộ lọc ngữ cảnh** | **72,2% / 63,1%** | **18,1% / 17,8%** | **19,4% / 19,5%** |

- So với ghép không lọc, bộ lọc ngữ cảnh giảm lỗi ở câu thuần Việt 0,7 điểm (seen) và 0,4 điểm (unseen), có ý nghĩa thống
  kê. Phần chép đúng từ tiếng Anh giảm 0,3 và 0,8 điểm, nhưng mức này nằm trong sai số (khoảng tin cậy chứa 0).
- So với RNNT đứng riêng, câu thuần Việt vẫn sai nhiều hơn 0,4 và 0,3 điểm. Trên tập dò, bản có bộ lọc ngữ cảnh còn lấy nhầm
  30 từ, trong đó 24 từ là từ của câu mồi, vì quy tắc vẫn nhận mọi từ của câu mồi.
- Vì sao không lọc luôn từ của câu mồi bằng ngữ cảnh: đã thử, và cách đó mất nhiều tiếng Anh hơn hẳn (mục 6.8). Phần lớn
  từ của câu mồi mà large-v3 viết ra là đúng (66 trong 90 lần ở tập dò), nhưng nhiều từ đúng trong số đó không có trong
  ngữ cảnh của mẫu.

**Việc chưa làm: lọc nốt từ của câu mồi.** Quy tắc hiện tại nhận mọi từ của câu mồi mà large-v3 viết ra. Nhưng các từ
này không đáng tin như nhau: có từ hay bị large-v3 viết ra dù người dùng không nói ("Mail" cho "mai", "list" cho "lịch",
mục 4.7.6), có từ hầu như chỉ xuất hiện khi người dùng nói thật. Cách làm:

1. Với từng từ của câu mồi, đếm số lần large-v3 viết ra từ đó, và trong các lần đó có bao nhiêu lần người dùng thật sự
   nói từ ấy (so với đáp án).
2. Từ hay đúng thì nhận như hiện nay. Từ hay sai thì chỉ nhận khi có trong ngữ cảnh của mẫu.

Việc đếm không được làm trên tập dò hay tập kiểm tra, vì như vậy quy tắc được chọn theo chính đề dùng để chấm. Phải đếm
trên 793 mẫu train chưa dùng vào việc gì. Các mẫu này chưa có chữ của large-v3 chạy với câu mồi `en_full_v1`, nên cần
chạy large-v3 trên Kaggle một lần (khoảng 1 giờ GPU T4). Sau đó mọi bước còn lại làm offline.

### Việc tiếp theo

Theo thứ tự:

1. **Lọc từ khi ghép: đã làm bước đầu** (từ điển ngữ cảnh, mục 6.8). Câu thuần Việt vẫn sai nhiều hơn RNNT 0,3–0,4
   điểm, chủ yếu vì từ của câu mồi bị viết nhầm. Muốn giảm tiếp thì cần chữ large-v3 có câu mồi trên phần train chưa dùng
   (một phiên GPU), để đo độ tin cậy riêng của từng từ.
2. **Đo thời gian và bộ nhớ** khi chạy cả hai mô hình ASR cùng LLM trên GPU thi.
3. **Viết phần ghép vào hệ thống thật** ([`lifemate/backends/asr.py`](../../../lifemate/backends/asr.py)). Hiện phép ghép mới chỉ chạy được khi chấm offline.

Cấu hình hệ thống đang dùng ([`configs/v0.toml`](../../../configs/v0.toml)) chưa đổi.

Benchmark lần 3 ([`docs/reports/benchmark/asr_v3.md`](asr_v3.md)) tiếp tục hai việc đầu, cùng với câu mồi theo mẫu (thử nghiệm E),
chấm điểm hai phương án bằng âm thanh (G) và bộ phân loại ghép (H): lọc từ của câu mồi theo độ tin cậy là thử nghiệm F;
thời gian và bộ nhớ được đo kèm mọi phiên GPU (mục 2.4 của báo cáo lần 3). Việc thứ ba (engine ghép `MergedASR`, bước
chuẩn bị P8) chưa làm ở lần 3, để sau khi có điểm thi của thử nghiệm J.

### Đọc tiếp ở đâu

| Muốn biết | Đọc |
|---|---|
| Thuật ngữ, dữ liệu được chia và dùng thế nào | Mục 0, 2 |
| So sánh bốn mô hình | Mục 3 (thử nghiệm A) |
| Câu mồi được viết và chọn thế nào | Mục 4 (thử nghiệm B) |
| Vì sao tin được số chấm offline | Mục 5 (thử nghiệm C) |
| Quy tắc ghép hai mô hình, được gì và mất gì | Mục 6 (thử nghiệm D) |
| File, hàm, cấu hình nhắc tới trong báo cáo | Phụ lục A |

**Kết quả thô.** Không có trên git.


| Thư mục                                               | Nội dung                                                                                                                                                                    |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`artifacts/runs/kaggle/asr/10-28_tune_v3/`](../../../artifacts/runs/kaggle/asr/10-28_tune_v3/)            | Kết quả lần 1, dùng lại cho thử nghiệm A và D: `<mô hình>__<lần đo>.hyps.jsonl` (chữ từng câu), `.json` (chỉ số), `.toml` (config)                                          |
| [`artifacts/runs/local/asr/10-29_en_mixed_offline_v1/`](../../../artifacts/runs/local/asr/10-29_en_mixed_offline_v1/) | Lần chấm offline đầu tiên (chỉ số bản đầu, trước khi sửa cách đếm "chèn từ câu mồi" và cách viết tương đương); chỉ để đối chiếu                                                                                |
| [`artifacts/runs/local/asr/10-29_en_mixed_offline_v2/`](../../../artifacts/runs/local/asr/10-29_en_mixed_offline_v2/) | Thử nghiệm A và D: `<mô hình>__<tập>.json` (chấm lại), `compare__<tập>.{md,json}` (bảng so sánh có khoảng tin cậy), `merge_*.hyps.jsonl` (chữ đã ghép), `merge_effect.json` |
| [`artifacts/runs/kaggle/asr/10-29_tune_v5a_en_mixed/`](../../../artifacts/runs/kaggle/asr/10-29_tune_v5a_en_mixed/)  | Thử nghiệm B và C, phiên v5a (PhoWhisper, ChunkFormer RNNT), cùng cấu trúc file với lần 1; `compare/` là bảng so sánh có khoảng tin cậy                                     |
| [`artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/`](../../../artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/)  | Thử nghiệm B và C, phiên v5b (whisper-large-v3), cùng cấu trúc và có `compare/`                                                                                             |
| [`artifacts/runs/local/asr/10-29_en_mixed_merge_v3/`](../../../artifacts/runs/local/asr/10-29_en_mixed_merge_v3/)   | Thử nghiệm D với large-v3 cấu hình thắng (mục 6.5.2): chữ đã ghép, `cmp_*.json`, `merge_effect.json`                                                                        |
| [`artifacts/runs/local/asr/10-29_foreign_words_audit_v1/`](../../../artifacts/runs/local/asr/10-29_foreign_words_audit_v1/) | Kiểm tra quy tắc tìm từ ngoại bằng rà tay (mục 2.1): `<tập>.json` gồm độ chính xác của quy tắc và tỉ lệ chép đúng theo hai cách gán nhãn |
| [`artifacts/runs/local/asr/10-29_en_mixed_merge_v4/`](../../../artifacts/runs/local/asr/10-29_en_mixed_merge_v4/)   | Bộ lọc ngữ cảnh cho phép ghép (mục 6.8): `merge_<cách lọc>__<tập>.hyps.jsonl`, `cmp_*.json` (so với RNNT), `cmp_vs_all_*.json` (so với ghép không lọc)                                   |


## Mục lục

- [Tóm tắt cho cả team](#tóm-tắt-cho-cả-team)
- [0. Bối cảnh và thuật ngữ](#0-bối-cảnh-và-thuật-ngữ)
- [1. Vấn đề chung và mục tiêu](#1-vấn-đề-chung-và-mục-tiêu)
- [2. Dữ liệu, thước đo và công cụ dùng chung](#2-dữ-liệu-thước-đo-và-công-cụ-dùng-chung)
- [3. Thử nghiệm A: chấm offline bốn mô hình trên câu chêm tiếng Anh](#3-thử-nghiệm-a-chấm-offline-bốn-mô-hình-trên-câu-chêm-tiếng-anh)
- [4. Thử nghiệm B: câu mồi tiếng Anh và lớp bảo vệ câu mồi](#4-thử-nghiệm-b-câu-mồi-tiếng-anh-và-lớp-bảo-vệ-câu-mồi)
- [5. Thử nghiệm C: kiểm chứng chấm offline bằng một lần chạy thật](#5-thử-nghiệm-c-kiểm-chứng-chấm-offline-bằng-một-lần-chạy-thật)
- [6. Thử nghiệm D: ghép ChunkFormer RNNT với Whisper](#6-thử-nghiệm-d-ghép-chunkformer-rnnt-với-whisper)
- [Giới hạn đã biết](#giới-hạn-đã-biết)
- [Phụ lục A. Bảng tra code](#phụ-lục-a-bảng-tra-code)

## 0. Bối cảnh và thuật ngữ

### 0.1. Bài toán

Bài thi LifeMate nhận lời nói tiếng Việt của người dùng và làm hai bước:

1. **ASR** (nhận dạng giọng nói) chép âm thanh thành chữ.
2. **LLM** đọc chữ đó và gọi đúng tool, ví dụ "gửi tin Zalo", "tra giá vàng SJC".

Đề bài ghi rõ lời nói "bằng tiếng Việt, có thể xen lẫn các từ tiếng Anh thông dụng" ([`docs/overview.md`](../../overview.md)). Dữ liệu train
của BTC cho thấy điều này rất phổ biến: 833 trên 1 545 mẫu có ít nhất một từ ngoại (Zalo, SJC, check, voucher, tên người
nước ngoài…). ASR chép sai một từ tiếng Anh thì LLM dễ gọi sai tool, dù phần tiếng Việt đúng hết.

Lần 1 ([`docs/reports/benchmark/asr_v1.md`](asr_v1.md)) chọn mô hình ASR theo mọi câu. Lần 2 nhìn riêng các câu chêm tiếng Anh. Lần 2
chỉ đo ASR, không chạy LLM và không chấm điểm thi.

### 0.2. Dữ liệu được tổ chức thế nào

- **Mẫu** (sample): một lượt hỏi của bài thi, có mã `id`. Mỗi mẫu có một **lượt cuối** (câu người dùng vừa nói) và có
  thể có **lịch sử** hội thoại trước đó.
- **Câu** (utterance): một file âm thanh của một lượt người dùng. Một mẫu có thể có nhiều câu (lượt cuối và các lượt
  người dùng trong lịch sử).
- **Đáp án** (`audio_text`): chữ chuẩn của câu do BTC viết. Đây là thứ để so với chữ ASR.
- **Chữ ASR** (`hyp`): chữ mô hình ASR chép ra. **Chữ trước lọc** (`raw`) là chữ ASR trước khi bộ lọc ảo giác cắt bỏ
  các câu bịa kiểu phụ đề ("Hãy subscribe…").
- **`hyps.jsonl`**: file kết quả của một lần chạy ASR, mỗi dòng là một câu dạng JSON
  `{"id": mã mẫu, "turn": "final" hoặc số thứ tự lượt trong lịch sử, "ref": đáp án, "hyp": chữ ASR, "raw": chữ trước lọc}`.
  Có file này thì chấm lại được bằng chỉ số mới mà không cần chạy lại mô hình.

### 0.3. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| WER | Tỉ lệ lỗi từ: số từ phải thêm, xoá hoặc thay để biến chữ ASR thành đáp án, chia cho số từ của đáp án. Càng thấp càng tốt |
| Từ ngoại, 4 nhóm | Từ không phải tiếng Việt, tìm bằng quy tắc ở mục 2.1. Chia 4 nhóm: **từ thường** (check, online), **viết tắt** (SMS, SJC, VTV), **tên** (Sarah, MinhMart, Zalo), **từ mượn** đã Việt hoá (tivi, gara) |
| Câu chêm tiếng Anh | Câu mà đáp án có ít nhất một từ ngoại. Câu không có từ ngoại gọi là **câu thuần Việt** |
| Chuẩn hoá chữ | Đổi chữ về dạng để so khớp: chữ thường, bỏ dấu tiếng Việt, "đ" thành "d". Ví dụ "Zalo" và "zalo" chuẩn hoá thành cùng "zalo" |
| Chép đúng từ ngoại | Từ ngoại của đáp án có mặt trong chữ ASR của cùng câu, so sau khi chuẩn hoá chữ (mục 2.3) |
| Giải mã tham lam | Ở mỗi bước, mô hình chọn đúng một từ có xác suất cao nhất. Không có ngẫu nhiên, nên chạy lại cho cùng chữ (**tất định**). Thể lệ yêu cầu tất định |
| Câu mồi (`initial_prompt`) | Đoạn chữ đưa cho Whisper như "câu nói trước đó". Whisper bắt chước cách viết của câu mồi (chính tả, tên riêng), không làm theo lời dặn. Chỉ 223 token cuối được dùng |
| Lớp bảo vệ câu mồi | Bước kiểm tra sau khi chép có câu mồi: câu nào có dấu hiệu hỏng thì chép lại không câu mồi (mục 4.3.2) |
| g3, g4 | Lớp bảo vệ câu mồi coi câu là "chép lại câu mồi" khi có 3 (g3) hoặc 4 (g4) từ liên tiếp trùng câu mồi. Tên cấu hình như `en_full+g4` nghĩa là câu mồi `en_full_v1` cộng lớp bảo vệ g4 |
| `hotwords` | Tham số của faster-whisper nhận một danh sách từ khoá để ưu tiên. Lần 1 thử và loại vì kém nhất |
| Câu mồi của lead | Câu mồi dạng liệt kê do trưởng nhóm viết, thử ở lần 1 |
| Beam 5 | Giải mã giữ 5 phương án tốt nhất ở mỗi bước rồi chọn câu tốt nhất; chậm hơn giải mã tham lam |
| D40, D41… | Mã một quyết định trong [`docs/decisions.md`](../../decisions.md) (ngày, nội dung, lý do) |
| VAD | Bộ phát hiện tiếng nói: cắt bỏ đoạn im lặng trước khi đưa vào Whisper, giảm câu bịa |
| p95 | Thời gian chép lời **một mẫu** (lượt cuối và mọi lượt người dùng trong lịch sử, tới 13 file) mà 95% số mẫu nhanh hơn. Đo trên GPU T4 của Kaggle ([`lifemate/eval/eval_asr.py`](../../../lifemate/eval/eval_asr.py) › `evaluate`) |
| Chấm offline | Chấm lại chữ ASR đã ghi ở lần chạy trước (`hyps.jsonl`) bằng chỉ số mới, không chạy lại mô hình |
| Tập dò, tập kiểm tra | Tập dò (`asr_tune`) dùng để chọn cấu hình. Tập kiểm tra (`val_seen`, `val_unseen`) chỉ dùng để đo một lần sau khi đã chọn, để con số không bị "chọn theo đề" |
| seen, unseen | `val_seen` gồm mẫu thuộc các app đã có trong train; `val_unseen` gồm mẫu thuộc app chưa gặp. Bộ thi riêng của BTC có app chưa gặp |
| Khoảng tin cậy 95% (CI) | Khoảng mà con số thật nằm trong đó với độ tin 95%, ước lượng bằng bootstrap (mục 2.4). Hai hệ thống chỉ được coi là khác nhau khi CI của chênh lệch không chứa 0 |
| Ghép mô hình | Kết hợp chữ của hai ASR cho cùng một câu (mục 6) |
| Bộ lọc ngữ cảnh | Bước lọc thêm vào phép ghép (cách lọc `context_other`): từ tiếng Anh không có trong câu mồi chỉ được lấy từ large-v3 khi có trong từ điển ngữ cảnh của mẫu. "Ghép không lọc" là phép ghép không có bước này (cách lọc `all`) (mục 6.8) |
| Từ điển ngữ cảnh | Tập các từ đã chuẩn hoá có sẵn dạng chữ trong đầu vào của mẫu lúc thi: catalog của app đã cài, ngữ cảnh người dùng, lượt trợ lý trong lịch sử (mục 6.8) |
| Oracle | Trần lý thuyết: với mỗi câu chọn bản tốt hơn trong hai bản, dựa vào đáp án. Không làm được lúc thi, chỉ để biết ghép tối đa được bao nhiêu |

### 0.4. Cách đọc các chỗ nhắc tới code

Báo cáo viết cho người không đọc code. Khi cần nhắc tới code, báo cáo ghi theo dạng "file › hàm: việc hàm làm", ví dụ
[`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py) › `find_foreign_words`: tìm các từ ngoại trong một câu. Mọi đường dẫn tính từ thư mục
gốc của repo. Bảng tra đầy đủ (file, hàm, đầu vào, đầu ra, cấu trúc dữ liệu) ở [Phụ lục A](#phụ-lục-a-bảng-tra-code).

## 1. Vấn đề chung và mục tiêu

**Vấn đề.** Lần 1 xếp hạng theo WER và tỉ lệ bỏ sót giá trị tham số trên mọi câu. Hai chỉ số này không cho biết mô hình
chép từ tiếng Anh ra sao: một câu dài 20 từ chỉ sai một chữ "check" vẫn có WER thấp (5%). Trong khi đó, từ tiếng Anh
thường là từ quyết định tool nào được gọi: "Zalo" quyết định app nhắn tin, "SJC" quyết định tra giá vàng, "voucher",
"playlist", "meeting" là tên đối tượng. Lần 1 đã thấy large-v3 gần như không bao giờ viết đúng "Zalo" (1/19 câu trên
val_seen).

**Mục tiêu.**

1. Đo bốn mô hình tốt nhất của lần 1 (ChunkFormer RNNT, ChunkFormer CTC, PhoWhisper, whisper-large-v3) trên riêng các
   câu chêm tiếng Anh, có khoảng tin cậy, theo từng nhóm từ (thử nghiệm A).
2. Thử cải thiện hai mô hình Whisper bằng câu mồi chứa từ tiếng Anh và lớp bảo vệ câu mồi. Đo trên cả tập của lần 1 (để
   biết câu không chêm tiếng Anh có bị sai thêm không) lẫn tập câu chêm tiếng Anh (thử nghiệm B).
3. Kiểm chứng rằng chấm offline cho đúng số như chạy thật (thử nghiệm C).
4. Thử ghép RNNT với một mô hình Whisper (thử nghiệm D).

**Bỏ Parakeet.** Parakeet đã bị loại ở lần 1, và chấm offline chỉ chép đúng 14% từ tiếng Anh, thấp nhất trong năm mô
hình.

**Các công cụ chuẩn bị trước.** Mọi thử nghiệm dùng chung các công cụ ở mục 2:

| Công cụ | Mục | Dùng ở thử nghiệm |
|---|---|---|
| Quy tắc tìm từ ngoại | 2.1 | A, B, C, D: để biết câu nào là câu chêm, từ nào cần chép đúng |
| Cách chia dữ liệu và tập con câu chêm tiếng Anh | 2.2 | A, B, C, D |
| Bộ chỉ số tiếng Anh | 2.3 | A, B, C, D |
| Chấm lại offline và bảng so sánh có khoảng tin cậy | 2.4 | A, B, D |
| Câu mồi có phiên bản, lớp bảo vệ câu mồi, dò 13 tổ hợp | 4.3 | B |
| Quy tắc ghép hai ASR | 6.3 | D |

## 2. Dữ liệu, thước đo và công cụ dùng chung

### 2.1. Tìm từ ngoại trong câu

**Vấn đề.** Muốn đo "mô hình chép từ tiếng Anh có đúng không" thì trước hết phải biết trong mỗi câu, từ nào là từ tiếng
Anh. Đáp án của BTC (`audio_text`) không đánh dấu điều này. Việc nhận ra từ ngoại cần ở bốn chỗ, trên hai loại chữ:

| Chỗ dùng | Chạy trên | Để làm gì |
|---|---|---|
| Chọn tập con câu chêm tiếng Anh (mục 2.2) | Đáp án | Giữ lại các mẫu có từ ngoại |
| Chỉ số "chép đúng từ ngoại" (mục 2.3) | Đáp án | Lấy danh sách từ cần chép đúng, rồi tìm từng từ trong chữ ASR |
| Đếm từ câu mồi bị chèn vào (mục 2.3) | Câu mồi | Lấy danh sách từ ngoại của câu mồi, rồi xem ASR có viết ra từ đó khi người dùng không nói không |
| Quy tắc ghép hai ASR (mục 6.3) | **Chữ ASR** | Nhận ra đoạn nào của Whisper mang từ tiếng Anh mà RNNT không có. Lúc thi đây là câu chưa từng gặp |

**Hai cách làm và lý do chọn quy tắc.**

- **Rà tay**: người đọc từng câu đáp án, đánh dấu từ ngoại, ghi lại danh sách. Chính xác nhất, nhưng chỉ phủ được đáp án
  đã có. Không dùng được cho chỗ thứ tư (chữ ASR của câu mới) và chỗ thứ ba (câu mồi viết thêm sau này).
- **Quy tắc**: một hàm nhìn mặt chữ của từ và quyết định có phải từ ngoại không. Chạy được trên mọi chữ, kể cả chữ ASR lúc
  thi, nhưng có thể sai.

Lần 2 dùng quy tắc vì cần cùng một cách tìm ở cả bốn chỗ: nếu đáp án dùng danh sách rà tay còn phép ghép dùng quy tắc,
thì chỉ số đo một định nghĩa "từ tiếng Anh" còn phép ghép tối ưu một định nghĩa khác.

**Ý tưởng của quy tắc.** Tiếng Việt viết theo âm tiết, mỗi âm tiết có cấu trúc chặt: phụ âm đầu, vần, phụ âm cuối, và
thường có dấu thanh. Từ tiếng Anh hay vi phạm cấu trúc này ("check" không có vần hợp lệ, "Zalo" có chữ "z" không có
trong tiếng Việt). Vì vậy, một từ không dấu mà không ghép được thành âm tiết Việt thì gần như chắc là từ ngoại.

**Thuật toán.** Code: [`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py) › `find_foreign_words(text, lowercase_vocabulary)`: nhận một câu,
trả về danh sách từ ngoại theo thứ tự xuất hiện; mỗi từ kèm cách viết gốc, dạng đã chuẩn hoá và nhóm. Câu được tách thành từ ở
mọi chỗ không phải chữ cái (chữ số cũng là ranh giới, nên "RON95" cho từ "RON"). Từ chỉ có một chữ cái bị bỏ qua. Với
mỗi từ còn lại, xét theo thứ tự:

1. **Có dấu tiếng Việt** (kể cả "đ") thì là tiếng Việt, bỏ qua.
2. **Nằm trong danh sách từ mượn** (hằng số `VIETNAMIZED_LOANWORDS`, một tập cố định gồm "tivi", "gara") thì thuộc nhóm
   **từ mượn**. Các từ này viết theo chính tả Việt và mô hình tiếng Việt chép như âm tiết Việt, nên tách riêng để không
   làm đẹp số của từ tiếng Anh thật.
3. **Viết hoa toàn bộ** (SMS, SJC, VTV) thì thuộc nhóm **viết tắt**.
4. **Là từ ngoại** khi có một trong ba dấu hiệu:
   - không ghép được thành một âm tiết Việt không dấu. Mẫu âm tiết (hằng số `_VIETNAMESE_SYLLABLE`, một biểu thức chính
     quy) là: phụ âm đầu tuỳ chọn (b, ch, gh, ngh, tr…), rồi 1 đến 3 nguyên âm, rồi phụ âm cuối tuỳ chọn (c, ch, m, n,
     ng, nh, p, t). "check" không khớp vì vần "e" + "ck" không có; "lan" khớp;
   - có chữ f, j, w hoặc z (không có trong bảng chữ cái tiếng Việt);
   - nằm trong danh sách ngoại lệ `ENGLISH_SPELLED_LIKE_VIETNAMESE` (một tập cố định gồm "chat", "top", "hot"): từ tiếng
     Anh tình cờ ghép được thành âm tiết Việt không dấu nên dấu hiệu đầu không bắt được, phải liệt kê tay.

   Từ ngoại viết thường thuộc nhóm **từ thường**. Từ ngoại viết hoa chữ đầu, đứng giữa câu, thuộc nhóm **tên**. Từ viết
   hoa đứng đầu câu ("Check…" hay "Sarah…") không phân biệt được bằng chữ hoa. Khi đó hàm tra tham số
   `lowercase_vocabulary`: tập các từ (đã chuẩn hoá) từng xuất hiện viết thường ở bất kỳ đáp án nào của tập đang chấm. Có trong
   tập thì xếp **từ thường**, không có thì xếp **tên**.

Ví dụ: "Nhắn Zalo cho Lan, giá vàng SJC, xăng RON95" cho ba từ ngoại: Zalo (tên), SJC và RON (viết tắt). "Lan" ghép được
âm tiết Việt nên bị bỏ qua. Đây là ý muốn, vì tên người Việt viết không dấu không phải từ ngoại.

**Giới hạn.**

- Bỏ sót từ tiếng Anh trông như âm tiết Việt mà không có trong `ENGLISH_SPELLED_LIKE_VIETNAMESE`, ví dụ "man", "ban".
- Tên tiếng Việt viết liền không dấu ("MinhMart") được tính là tên ngoại. Đây là chủ ý, vì đó là tên thương hiệu.
- Nhóm của từ viết hoa đầu câu phụ thuộc tập đang chấm. Tổng số từ ngoại và số từ chép đúng thì không phụ thuộc.
- Độ chính xác đo bằng rà tay ở phần dưới: đúng 99,7% khi nhận, tìm được 94,3% số từ tiếng Anh.

#### Kiểm tra quy tắc bằng rà tay

**Vấn đề.** Mọi con số "chép đúng từ tiếng Anh" trong báo cáo dựa vào quy tắc ở trên. Nếu quy tắc nhận nhầm nhiều từ
Việt là từ tiếng Anh, hoặc bỏ sót nhiều từ tiếng Anh, thì các con số và thứ hạng giữa các hệ thống có thể sai.

**Cách kiểm tra.** Rà tay trên mọi câu đáp án mà các chỉ số dùng: tập dò, `val_seen` và `val_unseen`, tổng 2 467 câu.
Quy tắc chỉ có thể sai theo hai cách, và mỗi cách có một danh sách hữu hạn để rà:

- **Nhận nhầm:** rà từng từ quy tắc đã nhận là từ ngoại (291 từ khác nhau, 981 lần xuất hiện).
- **Bỏ sót:** quy tắc bỏ qua mọi từ có dấu tiếng Việt, và chỉ bỏ sót được từ không dấu ghép được thành âm tiết Việt.
  Vì vậy chỉ cần rà các từ không dấu mà quy tắc bỏ qua (185 từ khác nhau, 6 418 lần xuất hiện), xem từng từ trong câu
  khi từ đó có thể là cả hai (ví dụ "in" trong "check-in" và trong "bản in").

Kết quả rà tay ghi theo vị trí (mã mẫu, lượt, thứ tự từ trong câu, nhóm) vào [`data/labels/foreign_words_hand_v1.jsonl`](../../../data/labels/foreign_words_hand_v1.jsonl),
ngoài git. Code: [`lifemate/eval/foreign_words_audit.py`](../../../lifemate/eval/foreign_words_audit.py) › `audit_rule` (so quy tắc với nhãn rà tay) và
`english_recall` (chấm lại tỉ lệ chép đúng theo nhãn rà tay).

**Kết quả.**

| Tập | Từ ngoại theo quy tắc | Theo rà tay | Quy tắc đúng khi nhận (độ chính xác) | Quy tắc tìm được (độ phủ) |
|---|---|---|---|---|
| Tập dò | 394 | 407 | 99,5% | 96,3% |
| val_seen | 335 | 353 | 100% | 94,9% |
| val_unseen | 252 | 277 | 99,6% | 90,6% |
| **Cả ba** | **981** | **1 037** | **99,7%** | **94,3%** |

- **Nhận nhầm 3 lần**, đều là tiếng cười: "hihi" (2), "haha" (1).
- **Bỏ sót 59 lần**, đều là từ tiếng Anh viết giống một âm tiết Việt không dấu: "team" (15), "gym" (8), "in" trong
  "check-in" (6), "The" và "Tree" trong tên quán ("The Lemon Tree Café"), tên người ("Tom", "Lee", "Mia", "Ben", "Liam",
  "Ryan"), "Hong Kong", "Chiang", "up" ("top up", "follow-up"), "out" ("check-out"), "Cup" ("AFF Cup"), "Pay", "Ron"
  (trong "Ron 95"). val_unseen bỏ sót nhiều nhất (26 lần), vì app lạ có nhiều từ loại này ("gym", "check-in").
- **Xếp nhóm sai 11 lần** trên 978 từ quy tắc nhận đúng: 8 lần "Ok" đầu câu ở val_seen bị xếp vào nhóm tên (val_seen
  không có "ok" viết thường nào để quy tắc biết đây là từ thường), và "canxi", "magie", "siro" bị xếp vào từ thường thay
  vì từ mượn đã Việt hoá.

**Chấm lại tỉ lệ chép đúng từ tiếng Anh theo nhãn rà tay** (mọi câu của tập):

| Hệ thống | Tập dò: quy tắc / rà tay | val_seen: quy tắc / rà tay | val_unseen: quy tắc / rà tay |
|---|---|---|---|
| ChunkFormer RNNT | 49,0 / 48,9 | 49,9 / 49,6 | 51,6 / 51,6 |
| large-v3 + `en_full_v1` + g4 | 70,6 / 70,8 | 72,2 / 71,1 | 61,1 / 61,4 |
| Ghép không lọc | 72,3 / 72,0 | 72,5 / 71,4 | 63,9 / 64,3 |
| Ghép có bộ lọc ngữ cảnh (`context_other`) | 69,0 / 68,6 | 72,2 / 71,1 | 63,1 / 63,2 |

Số theo quy tắc trong bảng này trùng với số ở các mục sau. Hai cách gán nhãn lệch nhau tối đa 1,1 điểm, và thứ hạng các
hệ thống không đổi ở tập nào. Mức tăng của bản ghép có bộ lọc ngữ cảnh so với RNNT theo nhãn rà tay là +21,5 (seen) và +11,6
(unseen) điểm, so với +22,3 và +11,5 theo quy tắc.

**Kết luận.** Quy tắc đủ chính xác cho các phép so sánh trong báo cáo. Các con số giữ nguyên theo quy tắc, vì phép ghép
cũng dùng chính quy tắc này trên chữ ASR, nên chỉ số và phép ghép dùng cùng một định nghĩa "từ tiếng Anh".

**Giới hạn.**

- Rà tay do trợ lý AI (Claude) làm trong phiên làm việc, chưa có thành viên nào của team rà lại.
- Từ một chữ cái bị bỏ qua cả khi rà ("vitamin C"), giống quy tắc.
- "Mai" trong "Chiang Mai" được giữ là từ Việt, vì cùng mặt chữ với "mai" (ngày mai).

### 2.2. Dữ liệu: từ gói của BTC tới các tập của lần 2

#### 2.2.1. Dữ liệu gốc của BTC

Gói `train+public_test` (bản v3.2.0, nằm trong [`data/train+public_test/`](../../../data/train+public_test/), không có trên git) có hai phần:

| Phần | Số mẫu | Có đáp án? | Dùng được cho benchmark? |
|---|---|---|---|
| `train` | 1 545 mẫu, 5 036 câu người dùng | Có: chữ chuẩn `audio_text` của mọi lượt, và lời gọi tool đúng (`label.tool_calls`) | Có, đây là nguồn duy nhất |
| `public_test` | khoảng 600 mẫu | Không | Không: không có đáp án để tính WER. Thể lệ cũng cấm tự gán nhãn tập này |

Bộ thi thật (private test) có cùng cấu trúc với `public_test` và có thêm app chưa từng xuất hiện trong train. Mỗi mẫu
train là một dòng trong [`data/train+public_test/train/train.jsonl`](../../../data/train+public_test/train/train.jsonl) kèm các file âm thanh: một file cho lượt cuối, một file cho mỗi lượt người dùng
trong lịch sử. Benchmark chấm **mọi câu người dùng** (lượt cuối và lịch sử), vì LLM đọc chữ của mọi lượt.

#### 2.2.2. Cách chia chung của dự án

**Vấn đề.** Chỉ có 1 545 mẫu có đáp án, và cả dự án (LLM, fine-tune, ASR) đều cần tập để học, tập để chọn cấu hình và tập
để đo kết quả. Nếu đo trên chính dữ liệu đã dùng để học hoặc để chọn, con số sẽ đẹp hơn thực tế. Bộ thi thật lại có app
lạ, nên cần một tập mô phỏng app lạ.

**Giải pháp.** [`lifemate/eval/make_splits.py`](../../../lifemate/eval/make_splits.py) › `main` (lệnh `make splits`) chia 1 545 mẫu một lần cho cả dự án. Mọi bước
dùng seed cố định 42, nên máy nào chạy cũng ra đúng cùng danh sách. Kết quả là các file danh sách mã mẫu trong
[`configs/splits/`](../../../configs/splits/) (mỗi dòng một mã) và [`configs/splits/meta.json`](../../../configs/splits/meta.json) (tham số chia và kết quả kiểm tra rò rỉ).

1. **Tách app lạ trước.** Mọi mẫu có lời gọi tool đúng dùng một trong 6 app (medication, fitness, flights, bike_rental,
   library, subscriptions) vào `val_unseen` (178 mẫu). Lý do: mô phỏng app chưa gặp của bộ thi thật. Tách trước khi chia
   phần còn lại thì `val_unseen` không thể lọt vào train.
2. **Gom cụm câu gần trùng.** Các mẫu có `audio_text` giống nhau từ 85% trở lên được gom thành một cụm (64 cụm, 174
   mẫu). Một cụm luôn nằm trọn trong một tập. Lý do: hai câu gần giống nhau ở hai tập thì tập kiểm tra thành "đề đã gặp".
3. **Chia 5 phần (fold) phân tầng.** 1 367 mẫu còn lại chia thành 5 phần cùng tỉ lệ số lời gọi tool (0, 1, 2–3, từ 4) và
   tỉ lệ câu phải hỏi lại. Phần 0 là `val_seen` (274 mẫu), phần 1–4 là `train` (1 093 mẫu). Lý do: `val_seen` có cùng
   kiểu câu với train, nên đo được chất lượng trên app đã biết mà không lệch về loại câu.
4. **Kiểm tra rò rỉ.** [`configs/splits/meta.json`](../../../configs/splits/meta.json) ghi số mã trùng giữa `train` và hai tập val: bằng 0.

| Tập chung | Số mẫu | Số câu | Vai trò trong dự án |
|---|---|---|---|
| `train` | 1 093 | 3 593 | Học (fine-tune LLM), và nguồn để rút tập dò |
| `val_seen` | 274 | 916 | Đo kết quả trên app đã biết |
| `val_unseen` | 178 | 527 | Đo kết quả trên app lạ |

#### 2.2.3. Cách dùng dữ liệu ở lần 1

**Vấn đề.** Benchmark lần 1 phải chọn cấu hình cho từng mô hình ASR (nhiều lựa chọn giải mã, câu mồi, VAD…), rồi mới so
các mô hình. Chọn cấu hình trên `val_seen` thì cấu hình thắng khớp riêng với `val_seen`.

**Giải pháp.** Tạo thêm một **tập dò** riêng cho ASR, rút từ `train`:

- [`configs/splits/asr_tune.txt`](../../../configs/splits/asr_tune.txt) (`make_splits --asr-tune 300`): 300 trong 1 093 mẫu train, phân tầng theo loại đáp án và có/không lịch
  sử, để tỉ lệ các loại câu giống train. 300 mẫu có 1 024 câu và 123 giá trị tham số, đủ để so lựa chọn, mà vẫn dò xong
  5 mô hình trong một phiên GPU T4.
- Lấy từ train được vì các mô hình ASR không học trên train của dự án, nên dò trên đó không làm lộ đáp án. Nếu sau này
  fine-tune ASR trên train thì phải loại 300 mẫu này ra.
- Mỗi mô hình dò cấu hình trên `asr_tune`, rồi đo cấu hình thắng **đúng một lần** trên `val_seen` và `val_unseen`.
- 793 mẫu train còn lại không dùng trong lần 1.

#### 2.2.4. Cách chia và dùng dữ liệu ở lần 2

**Vấn đề.** Lần 2 cần thêm ba thứ: (1) tập chỉ gồm câu chêm tiếng Anh, (2) dữ liệu để biết large-v3 hay sai từ tiếng Anh
nào, để viết câu mồi, (3) tập dò và tập kiểm tra không lẫn với dữ liệu ở (2). Đồng thời kết quả phải so được với lần 1.

**Giải pháp: không chia lại, chỉ lấy tập con và dùng phần train còn trống.**

1. **Tập con câu chêm tiếng Anh.** [`lifemate/eval/make_splits.py`](../../../lifemate/eval/make_splits.py) › `en_mixed_ids(items)`: nhận danh sách mẫu, trả về
   danh sách mã mẫu (đã sắp xếp) có ít nhất một lượt người dùng (lượt cuối hoặc lịch sử) chứa từ ngoại theo quy tắc mục
   2.1. Lệnh `make splits` gọi hàm này cho ba tập và ghi ba file `*_en_mixed.txt`:

   | Tập con | Lấy từ | Số mẫu | Số câu chêm tiếng Anh | Số từ ngoại |
   |---|---|---|---|---|
   | `asr_tune_en_mixed` | `asr_tune` (300 mẫu) | 165 | 276 | 394 |
   | `val_seen_en_mixed` | `val_seen` (274 mẫu) | 152 | 245 | 335 |
   | `val_unseen_en_mixed` | `val_unseen` (178 mẫu) | 95 | 162 | 252 |

2. **Phần train còn trống dùng để viết câu mồi.** 793 mẫu train không thuộc `asr_tune` (1 093 − 300) được dùng để đếm
   large-v3 hay sai từ tiếng Anh nào (mục 4.1, 4.3.1). Chữ large-v3 của các mẫu này lấy từ [`data/asr/train_asr.json`](../../../data/asr/train_asr.json),
   một bộ đệm chữ ASR có sẵn cho dữ liệu fine-tune (large-v3 beam 5 chạy trên cả 1 545 mẫu); chỉ phần của 793 mẫu được
   đọc. Ba tập dò và kiểm tra không góp mẫu nào vào việc chọn từ cho câu mồi.

Cách từng thử nghiệm dùng dữ liệu:

| Việc | Dữ liệu | Để làm gì | Lý do chọn |
|---|---|---|---|
| Chọn từ cho câu mồi (mục 4.3.1) | 793 mẫu train ngoài `asr_tune`, chữ large-v3 trong [`data/asr/train_asr.json`](../../../data/asr/train_asr.json) | Đếm từ tiếng Anh large-v3 hay sai | Nếu chọn từ trên tập dò thì câu mồi khớp sẵn với tập dò, số dò lạc quan; trên val thì lộ đề |
| Kiểm tra câu mồi không trùng lời nói thật ([`tests/test_asr_prompts.py`](../../../tests/test_asr_prompts.py)) | Đáp án của 3 593 câu [`configs/splits/train.txt`](../../../configs/splits/train.txt) | Câu mồi không có cụm 4 từ nào trùng lời nói thật | Viết lại câu mồi theo kết quả test là một bước chọn, nên không đọc val (xem dưới) |
| Thử nghiệm A: chấm offline 4 mô hình | Chữ ASR của lần 1 trên `val_seen`, `val_unseen`, chấm lại trên tập con `*_en_mixed` | Xếp hạng trên câu chêm tiếng Anh | Không chọn gì, chỉ đo, nên đo thẳng trên tập kiểm tra |
| Thử nghiệm B: dò 13 tổ hợp | `asr_tune` **đầy đủ** (300 mẫu, 1 024 câu) | Chọn câu mồi và mức bảo vệ | Hàm mục tiêu cần cả câu thường để phạt câu mồi làm câu thuần Việt sai thêm; tập con chêm tiếng Anh được chấm lại sau để báo cáo phụ |
| Thử nghiệm B: đo cấu hình thắng | `val_seen`, `val_unseen` **đầy đủ**, đúng một lần; rồi chấm lại tập con `*_en_mixed` | Số cuối của thử nghiệm B | Một lần chạy GPU trả lời cả hai câu hỏi: câu chêm có tốt hơn không, câu thường có bị sai thêm không |
| Thử nghiệm C: kiểm chứng | `asr_tune`, `val_seen`, `val_unseen` (RNNT); `asr_tune` (large-v3 VAD) | So chữ chạy lại với chữ lần 1 | Cùng tập với lần 1 thì so được từng câu |
| Thử nghiệm D: ghép | Chữ RNNT và Whisper trên `asr_tune`, `val_seen`, `val_unseen`; chấm cả tập đầy đủ và tập con | Đo phép ghép sửa đúng được bao nhiêu và làm sai thêm bao nhiêu | Quy tắc ghép viết trước, không chỉnh theo val; muốn thêm bộ lọc thì chỉnh trên `asr_tune`, đo lại một lần trên val |

**Lý do của các lựa chọn.**

- **Dùng lại đúng các tập của lần 1.** Chữ ASR đã chép ở lần 1 dùng lại được (thử nghiệm A, D không tốn GPU), và kết quả
  so được với lần 1. [`configs/splits/asr_tune.txt`](../../../configs/splits/asr_tune.txt) sinh lại khớp từng mã với lần 1 nhờ seed cố định.
- **Tập con chứ không chia lại.** Chia lại thì mất khả năng so với lần 1 và với các phần khác của dự án đang đo trên
  `val_seen`, `val_unseen`.
- **Vẫn tách tập dò và tập kiểm tra.** Cấu hình thắng được chọn trong 13 lựa chọn, nên số trên tập dò luôn lạc quan. Thử
  nghiệm B cho thấy rõ điều này: PhoWhisper tăng 6,6 điểm trên tập dò nhưng 0,0–0,9 điểm trên val.
- **Ba phần dữ liệu không chồng nhau:** 300 mẫu dò + 793 mẫu chọn từ câu mồi = 1 093 mẫu train; hai tập val tách riêng
  từ đầu (kiểm tra rò rỉ bằng 0).
- **Giữ seen/unseen.** App lạ kéo theo từ lạ (vitamin, homestay, tên hãng bay), nên `val_unseen` cho biết lợi ích có giữ
  được với từ không có trong câu mồi không. Thử nghiệm B cho thấy câu mồi giúp ít hơn hẳn trên `val_unseen`.

**Kiểm tra câu mồi chỉ đọc `train`.** Bản đầu của test ([`tests/test_asr_prompts.py`](../../../tests/test_asr_prompts.py)) đọc đáp án của cả 5 036 câu,
gồm cả hai tập val. Viết lại câu mồi theo kết quả test là một bước chọn, nên test đã được sửa để chỉ đọc 3 593 câu của
[`configs/splits/train.txt`](../../../configs/splits/train.txt). Đếm lại cho thấy kết quả hiện tại không đổi: ba câu mồi mới không trùng câu nào của val, kể cả ở cụm 3 từ,
và vẫn qua test khi chỉ đọc train.

| Câu mồi | Câu trùng cụm 3 từ / 4 từ: train (3 593 câu) | val_seen (916) | val_unseen (527) |
|---|---|---|---|
| `en_top10_v1` | 0 / 0 | 0 / 0 | 0 / 0 |
| `en_top20_v1` | 0 / 0 | 0 / 0 | 0 / 0 |
| `en_full_v1` | 2 / 0 | 0 / 0 | 0 / 0 |

**Giới hạn.** Tập dò chỉ có 300 mẫu (394 từ ngoại), nên chênh vài từ đã là vài điểm phần trăm. Chọn từ câu mồi dùng chữ
large-v3 beam 5, không phải cấu hình tham lam + VAD đang dò, nên danh sách "từ hay sai" có thể lệch một ít.

### 2.3. Thước đo

**Vấn đề.** WER của lần 1 không tách được lỗi ở từ tiếng Anh khỏi lỗi ở phần tiếng Việt, và không đo được việc câu mồi
làm mô hình viết ra từ không được nói.

**Giải pháp.** Thêm các chỉ số dưới đây vào hàm chấm của lần 1. Code: [`lifemate/eval/eval_asr.py`](../../../lifemate/eval/eval_asr.py) › `evaluate(samples,
transcriber, prompt)`: chép (hoặc đọc lại) chữ ASR của mọi câu, so với đáp án, trả về một bản báo cáo `AsrReport` chứa mọi
chỉ số. Mọi tỉ lệ tính trên cả tập: cộng tổng lỗi của mọi câu rồi chia tổng số từ, không lấy trung bình theo câu, để câu
ngắn không nặng ký bằng câu dài.

| Chỉ số (tên trong file kết quả) | Cách tính | Vì sao cần |
|---|---|---|
| **Chép đúng từ ngoại** (`en_word_recall`) | Số từ ngoại của đáp án có mặt trong chữ ASR cùng câu (so sau khi chuẩn hoá chữ), chia tổng số từ ngoại. Từ xuất hiện 2 lần trong đáp án mà chữ ASR chỉ có 1 lần thì chỉ tính đúng 1. Code: [`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py) › `match_words_in_transcript` | Chỉ số chính của lần 2 |
| Chép đúng, bản nới lỏng (`en_word_recall_lenient`) | Như trên, nhưng chấp nhận cách viết khác mà LLM vẫn hiểu: "TV" cho "tivi", "okay"/"oke" cho "ok", "mail" cho "email", và tên ghép bị tách ("Minh Mart" cho "MinhMart"). Các cặp cách viết lưu ở hằng số `EQUIVALENT_SPELLINGS` (từ điển: từ → danh sách cách viết khác), tra theo cả hai chiều | Tách lỗi nghe thật khỏi lỗi cách viết |
| Theo nhóm (`en_groups`) | Chép đúng theo 4 nhóm từ, dạng nhóm → [số từ, số chép đúng] | Biết mô hình yếu ở loại từ nào |
| **WER câu chêm** (`en_mixed_wer`) | WER chỉ trên các câu chêm tiếng Anh | Chất lượng chung của câu chêm |
| **WER câu thuần Việt** (`vi_only_wer`) | WER trên các câu **không** có từ ngoại | Câu mồi tiếng Anh hoặc phép ghép có làm câu thuần Việt sai thêm không |
| Chèn từ câu mồi (`prompt_insertions`) | Số lần một từ ngoại của câu mồi xuất hiện trong chữ ASR mà đáp án câu đó không có (kể cả dưới cách viết khác, như "ti vi" hay "TV" cho "tivi"; hàm [`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py) › `contains_word`). Mỗi từ đếm tối đa một lần mỗi câu | Câu mồi có làm mô hình viết ra từ không được nói không. Ví dụ "Mai" thành "Mail" |
| Có/không có trong câu mồi (`en_prompt`) | Chép đúng, tách theo từ **có** và **không có** trong câu mồi | Câu mồi giúp thật hay chỉ khớp sẵn các từ có trong câu mồi |
| Số câu chép lại (`n_prompt_retries`) | Số câu lớp bảo vệ đã chép lại không câu mồi | Tần suất câu mồi làm hỏng câu, và chi phí thời gian |
| Giữ từ lần 1 | WER, WER bỏ số (`wer_nonum`), CER, bỏ sót giá trị tham số, câu rỗng, câu bịa, `terms` (Zalo, VTV…), p50/p95/max | So được với lần 1 |
| Tử số, mẫu số (`counts`) | Từ điển tên → số nguyên: số lỗi và số từ của từng loại WER, số từ ngoại, số chép đúng, số giá trị tham số bị bỏ sót và tổng | Để cộng dồn qua nhiều tập con khi tính khoảng tin cậy (mục 2.4) |

**Về tên "bỏ sót tên riêng" của lần 1.** Chỉ số này thực ra đo **giá trị tham số** trong tool call của đáp án mà người
dùng có nói ra (tên người, kênh, số…), không có bước nhận diện tên riêng. Báo cáo này gọi là "bỏ sót giá trị tham số".

**Giới hạn.** "Chép đúng" chỉ kiểm tra từ có mặt trong câu, không kiểm tra đúng vị trí, nên có thể khớp nhầm khi chữ ASR
viết từ đó ở chỗ khác.

### 2.4. Chấm lại offline và bảng so sánh có khoảng tin cậy

**Vấn đề.** Chạy lại bốn mô hình trên GPU chỉ để đo chỉ số mới thì tốn nhiều giờ. Ngoài ra, các tập con nhỏ (vài trăm
từ ngoại), nên chênh vài từ đã là vài điểm phần trăm. Cần biết chênh lệch nào là thật, chênh lệch nào do may rủi.

**Giải pháp 1: chấm lại offline.** Code: [`lifemate/eval/rescore_asr.py`](../../../lifemate/eval/rescore_asr.py). Lớp `ReplayTranscriber` đóng vai một mô hình ASR,
nhưng thay vì nghe âm thanh, lớp này trả lại chữ đã ghi trong `hyps.jsonl`. Đưa lớp này vào đúng hàm `evaluate` của mục 2.3
thì ra đúng các chỉ số mà một lần chạy thật cho, trừ thời gian (bằng 0, lấy từ lần chạy gốc). Cách này chỉ đúng khi chạy
lại mô hình cho cùng chữ; thử nghiệm C kiểm chứng điều đó.

**Giải pháp 2: bảng so sánh có khoảng tin cậy.** Code: [`lifemate/eval/asr_compare.py`](../../../lifemate/eval/asr_compare.py) › `compare(samples, systems)`: nhận
danh sách mẫu và nhiều hệ thống (mỗi hệ thống là một file `hyps.jsonl`), trả về chỉ số, CI và CI của chênh lệch so với
hệ thống đầu tiên. Thuật toán **bootstrap có ghép cặp**:

1. Chấm từng mẫu riêng một lần để có tử số và mẫu số của mọi chỉ số (`counts`).
2. Lặp 1 000 vòng. Mỗi vòng bốc ngẫu nhiên có hoàn lại N mẫu từ N mẫu của tập (một mẫu có thể bị bốc nhiều lần, mẫu khác
   không được bốc), cộng tử và mẫu của các mẫu được bốc, tính ra chỉ số.
3. CI 95% là khoảng từ giá trị đứng thứ 2,5% tới giá trị đứng thứ 97,5% trong 1 000 giá trị đó.

Lý do của từng lựa chọn:

- **Ghép cặp**: ở mỗi vòng, mọi hệ thống dùng cùng các mẫu được bốc. Chênh lệch giữa hai hệ thống vì vậy chỉ phản ánh khác
  biệt của hệ thống, không lẫn khác biệt do bốc mẫu khó hay dễ, nên CI của chênh lệch hẹp hơn và công bằng.
- **Bốc theo mẫu chứ không theo câu**: các câu cùng mẫu cùng người nói, cùng hội thoại, nên không độc lập với nhau. Bốc
  theo câu sẽ cho CI hẹp hơn thực tế.
- **Chấm từng mẫu một lần rồi chỉ cộng**: kết quả giống hệt chấm lại cả tập mỗi vòng, nhưng 1 000 vòng chạy trong vài
  giây thay vì hàng chục phút.
- Seed cố định (0), nên chạy lại cho cùng CI.

**Kiểm tra công cụ.** Chấm lại toàn bộ val_seen và val_unseen của 4 mô hình bằng `rescore_asr` cho **đúng từng số** (WER,
`wer_nonum`, CER, bỏ sót, câu rỗng, câu bịa, số câu) như báo cáo gốc của lần 1, ở cả 8 cặp mô hình × tập.

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

**Công cụ dùng** (chuẩn bị ở mục 2): tập con câu chêm (mục 2.2), bộ chỉ số tiếng Anh (mục 2.3), chấm lại offline
`rescore_asr` cho từng mô hình × tập, và bảng so sánh có CI `asr_compare` (mục 2.4). Lệnh đầy đủ ở [`docs/experiments.md`](../../experiments.md).

**Lưu ý về large-v3.** Cấu hình thắng của large-v3 ở lần 1 dùng nhiệt độ thử lại (`temperature = [0, 0,2, …, 1]`): khi
câu hỏng, mô hình lấy mẫu ngẫu nhiên. Cấu hình này không tất định và nhiều khả năng trái thể lệ (mục 4.3.4), nên số của
large-v3 ở đây chỉ để tham khảo. Thử nghiệm B đo lại large-v3 với cấu hình tất định.

### 3.4. Chỉ số

`en_word_recall` (chính), `en_word_recall_lenient`, `en_groups`, `en_mixed_wer`, `vi_only_wer`, WER, bỏ sót giá trị
tham số; CI 95% và CI của chênh lệch so với ChunkFormer RNNT.

### 3.5. Kết quả

Tập câu chêm tiếng Anh. Số trong ngoặc là CI 95%.


| Mô hình                   | Tập    | Chép đúng từ ngoại      | Bản nới lỏng | WER câu chêm            | WER câu thuần Việt | p95 lần 1 (s) |
| ------------------------- | ------ | ----------------------- | ------------ | ----------------------- | ------------------ | ------------- |
| ChunkFormer RNNT          | seen   | 49,9 \[44,3; 55,3\]     | 52,5         | **24,8** \[22,8; 26,7\] | **16,9**           | 2,5           |
|                           | unseen | 51,6 \[44,8; 58,6\]     | 52,4         | **24,1** \[21,6; 26,8\] | **17,0**           | 2,5           |
| ChunkFormer CTC           | seen   | 34,6 \[29,8; 39,7\]     | 41,5         | 27,5                    | 17,5               | 2,7           |
|                           | unseen | 39,7 \[32,2; 47,4\]     | 40,5         | 26,9                    | 17,0               | 2,3           |
| PhoWhisper (câu mồi ngắn) | seen   | 36,1 \[30,6; 41,9\]     | 39,7         | 32,5                    | 24,7               | 3,0           |
|                           | unseen | 36,5 \[29,7; 43,4\]     | 38,1         | 33,1                    | 24,6               | 3,0           |
| large-v3 (tham khảo)      | seen   | **51,6** \[46,4; 56,7\] | **54,9**     | 32,8                    | 29,2               | 8,2           |
|                           | unseen | **56,0** \[48,1; 63,4\] | **57,5**     | 33,9                    | 29,1               | 7,9           |


Theo nhóm từ (số từ chép đúng / tổng):


| Mô hình          | Tập    | Từ thường  | Viết tắt  | Tên        | Từ mượn   |
| ---------------- | ------ | ---------- | --------- | ---------- | --------- |
| ChunkFormer RNNT | seen   | 50/105     | **47/82** | 60/133     | **10/15** |
|                  | unseen | 56/97      | 18/23     | 55/130     | 1/2       |
| ChunkFormer CTC  | seen   | 47/105     | 29/82     | 40/133     | 0/15      |
|                  | unseen | 48/97      | 17/23     | 34/130     | 1/2       |
| PhoWhisper       | seen   | 34/105     | 24/82     | 58/133     | 5/15      |
|                  | unseen | 40/97      | 11/23     | 41/130     | 0/2       |
| large-v3         | seen   | **61/105** | 43/82     | **67/133** | 2/15      |
|                  | unseen | 52/97      | **21/23** | **67/130** | 1/2       |


Chênh lệch so với RNNT (điểm %, CI 95% của chênh lệch), tập câu chêm tiếng Anh:


| Mô hình    | Chép đúng từ ngoại, seen | unseen                | WER câu chêm, seen   | unseen               |
| ---------- | ------------------------ | --------------------- | -------------------- | -------------------- |
| CTC        | −15,2 \[−20,6; −9,3\]    | −11,9 \[−17,2; −6,2\] | +2,7 \[+1,2; +4,3\]  | +2,8 \[+1,2; +4,4\]  |
| PhoWhisper | −13,7 \[−19,9; −7,1\]    | −15,1 \[−22,0; −8,7\] | +7,6 \[+5,7; +9,4\]  | +9,0 \[+7,1; +11,0\] |
| large-v3   | +1,8 \[−4,5; +8,3\]      | +4,4 \[−2,8; +10,7\]  | +7,9 \[+5,4; +10,5\] | +9,8 \[+7,0; +12,6\] |


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
(chữ large-v3 có sẵn trong [`data/asr/train_asr.json`](../../../data/asr/train_asr.json)): 458 lần xuất hiện từ ngoại, sai 206 lần; 10 từ sai nhiều nhất
chiếm 43% số lần sai, 30 từ đầu chiếm 65%:


| Từ                     | Đúng / tổng | Ví dụ chép sai (tập dò)                                                     |
| ---------------------- | ----------- | --------------------------------------------------------------------------- |
| Zalo                   | 1/36        | "Nhắn que **Yellow**", "**Giá lô** nhé em"                                  |
| SJC                    | 1/19        | "vàng **Ếch, trí trí**", "**SGC**"                                          |
| SMS                    | 32/41       |                                                                             |
| check                  | 5/11        | "**Chắc** danh mục cổ phiếu…"                                               |
| voucher                | 3/8         | "đổi hết ra **vào chờ** 100.000"                                            |
| note, book, list, mail |             | "thêm vô **nốt**", "**búc** dùm cô", "**lít** đi chợ", "cái **mèo** sao kê" |
| podcast, deadline      |             | "một tấm **postcard**", "hôm nay **Đất Lai**…"                              |


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


| Nguyên tắc                                                                               | Lý do                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Câu nói tự nhiên, không viết dạng danh sách                                              | `hotwords` và câu mồi của lead (dạng liệt kê) là hai lựa chọn tệ nhất ở lần 1                                                                                                                                                       |
| Không có chữ số đi liền tên ("VTV" thay cho "VTV3", "xăng RON" thay cho "RON95")         | Chữ số làm mô hình đếm tiếp                                                                                                                                                                                                         |
| Không có tên sản phẩm lạ kiểu "LifeMate"                                                 | Sinh ra "Mate 2.0.0…"                                                                                                                                                                                                               |
| Chỉ chứa từ đang sai, lấy theo số đo trên 793 mẫu train ngoài tập dò                     | Không dò trên chính tập đã dùng để chọn từ                                                                                                                                                                                          |
| Từ sai nhiều nhất (Zalo, SJC, SMS, check) đặt cuối                                       | Phần cuối gần audio nhất; câu quá dài thì phần đầu bị cắt                                                                                                                                                                           |
| Chính tả theo đáp án của BTC ("MinhMart", "SJC")                                         | Viết đúng như đáp án mới được tính là chép đúng                                                                                                                                                                                     |
| **Câu tự viết, không chép câu nào của dữ liệu**, giữ từ khoá nhưng đặt vào ngữ cảnh khác | Lớp bảo vệ coi câu ASR có cụm từ trùng câu mồi là "chép lại câu mồi". Câu mồi giống lời nói thật thì câu nói đúng cũng bị báo nhầm. Bản nháp đầu (viết theo mẫu câu thật, có "nhắn SMS cho") báo nhầm 2,1–2,6% câu chuẩn ở cụm 3 từ |


**Ba câu mồi lồng nhau theo độ dài** ([`configs/asr_prompts/`](../../../configs/asr_prompts/)): câu dài hơn bằng câu ngắn hơn cộng thêm các câu đặt phía
trước, nên kết quả trả lời thẳng câu hỏi "dài hơn có tốt hơn không".


| File                                                | Từ khoá                                                                                                                                          | Token (large-v3 / PhoWhisper) | Báo nhầm trên 5 036 câu chuẩn (cụm 4 / 3 từ)   |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- | ---------------------------------------------- |
| [`configs/asr_prompts/short_v1.txt`](../../../configs/asr_prompts/short_v1.txt) (câu mồi ngắn của lần 1, để so sánh) | Zalo, VTV3                                                                                                                                       | 18 / 19                       | không yêu cầu (trùng nhiều câu thật)           |
| [`configs/asr_prompts/en_top10_v1.txt`](../../../configs/asr_prompts/en_top10_v1.txt)                                   | check, file, book, robot, online, Sarah, voucher, SMS, Zalo, SJC                                                                                 | 56 / 59                       | 0 / 0                                          |
| [`configs/asr_prompts/en_top20_v1.txt`](../../../configs/asr_prompts/en_top20_v1.txt)                                   | thêm tivi, list, chill, camera, gara, mail, tour, workshop, Greenhaven Foods, Brightpath, MinhMart                                               | 106 / 114                     | 0 / 0                                          |
| [`configs/asr_prompts/en_full_v1.txt`](../../../configs/asr_prompts/en_full_v1.txt)                                    | thêm QR, app, email, Grace Nakamura, Peter Hale, laptop, demo, Nexora Labs, note, MinhPoint, podcast, playlist, meeting, deadline, RON, VTV, HTV | 183 / 193                     | 0 / 2 câu (0,04%, đều do tên "Grace Nakamura") |


Nội dung [`configs/asr_prompts/en_top10_v1.txt`](../../../configs/asr_prompts/en_top10_v1.txt): "Ok, robot đồ chơi hết pin. File ảnh Sarah giữ, tối book bàn online. Voucher nằm trong SMS,
check kỹ rồi gửi Zalo cả xóm, coi miếng SJC luôn."

**Code.**

- Câu mồi là **file chữ có phiên bản** trong [`configs/asr_prompts/`](../../../configs/asr_prompts/), giống system prompt của LLM. File config của hệ
thống (TOML, bảng `[asr]` chứa mọi thiết lập ASR) trỏ tới file câu mồi bằng khoá `initial_prompt_file` (một đường dẫn).
[`lifemate/core/config.py`](../../../lifemate/core/config.py) › `load_config`: đọc file config; gặp khoá này thì đọc nội dung file câu mồi và đặt vào tham số
giải mã `initial_prompt` mà Whisper nhận. Hàm báo lỗi khi file thiếu, rỗng, hoặc config khai câu mồi cả hai cách (vừa
chữ trực tiếp vừa đường dẫn). Đổi câu mồi thì tạo file `_v2`, không sửa file cũ, để mọi kết quả cũ vẫn truy được câu
mồi đã dùng.
- [`tests/test_asr_prompts.py`](../../../tests/test_asr_prompts.py): test tự động giữ quy tắc "không trùng lời nói thật". Test đọc chữ chuẩn của 3 593 câu
thuộc [`configs/splits/train.txt`](../../../configs/splits/train.txt) (không đọc hai tập val, mục 2.2.4) và yêu cầu: không câu nào có cụm 4 từ liên tiếp trùng câu mồi, và dưới 0,5% số câu có cụm 3 từ trùng. Câu mồi
mới nào vi phạm thì test báo lỗi. Test chỉ chạy trên máy có gói dữ liệu (đánh dấu `data`).

#### 4.3.2. Lớp bảo vệ câu mồi

**Chung.** Câu mồi có ích ở đa số câu nhưng làm hỏng một số câu. Thay vì bỏ câu mồi, phát hiện riêng các câu hỏng rồi
chép lại chúng **không câu mồi**. Việc phát hiện chỉ nhìn chữ vừa chép ra, không cần đáp án, nên dùng được lúc thi.

**Thuật toán** ([`lifemate/backends/asr_guard.py`](../../../lifemate/backends/asr_guard.py), hàm `needs_retry`). Một câu chép với câu mồi bị coi là hỏng khi có một
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

- [`lifemate/backends/asr_guard.py`](../../../lifemate/backends/asr_guard.py) › `needs_retry(raw, cleaned, prompt, ngram, max_compression)`: nhận chữ trước lọc,
chữ sau lọc, câu mồi, độ dài cụm từ n và ngưỡng tỉ lệ nén; trả về đúng/sai (có cần chép lại không). Dấu hiệu 3 do hàm
`echoes_prompt` cùng file kiểm tra: lập tập mọi cụm n từ liên tiếp của câu mồi, rồi xem chữ ASR có cụm nào thuộc tập đó
không. Dấu hiệu 4 do hàm `compression_ratio` tính: độ dài chữ chia độ dài sau khi nén bằng zlib.
- [`lifemate/backends/asr.py`](../../../lifemate/backends/asr.py) › `ASRTranscriber._guard`: bước chạy sau khi chép xong mọi câu của một mẫu. Bước này gọi
`needs_retry` cho từng câu, gom các câu hỏng, chép lại chúng một lượt không câu mồi, thay kết quả vào (cả chữ trước lọc),
tăng bộ đếm `n_prompt_retries` (số nguyên) và ghi một dòng log `asr_prompt_retry`.
- Việc chép lại không câu mồi do hàm `transcribe_batch_unprompted` của từng engine làm. Chỉ hai engine Whisper có hàm
này: `FasterWhisperASR` (chạy large-v3) bỏ `initial_prompt` và `hotwords`, `HFSeq2SeqASR` (chạy PhoWhisper) bỏ
`prompt_ids`. Hàm dùng lại mô hình đã nạp, nên không tốn thêm bộ nhớ GPU.
- Hai khoá config mới trong bảng `[asr]`: `prompt_guard_ngram` (số nguyên n; 0 là tắt, mặc định; nếu bật phải từ 2 trở
lên và phải có câu mồi) và `prompt_guard_max_compression` (số thực, mặc định 2,4). Bật lớp bảo vệ cho engine không chép
lại được (ChunkFormer, NeMo) thì báo lỗi ngay khi khởi tạo, không đợi tới lúc chạy.
- Vẫn tất định: không có bước ngẫu nhiên nào.

#### 4.3.3. Dò 13 tổ hợp cùng lúc

**Chung.** Câu mồi và lớp bảo vệ được dò **cùng lúc**: 1 lựa chọn không câu mồi + 4 câu mồi × 3 mức bảo vệ (tắt, 3 từ,
4 từ) = 13 tổ hợp trong một bước.

**Lý do không dò lần lượt.** Nếu chọn câu mồi trước khi bật lớp bảo vệ, câu mồi dài (giúp nhiều nhất về từ tiếng Anh
nhưng cũng hay bị chép lại nhất) dễ thua ngay ở bước đầu, và không bao giờ được thử cùng lớp bảo vệ, dù chính lớp bảo vệ
là thứ cứu câu mồi dài. Số offline của lần 1 cho thấy đúng hiện tượng này (mục 4.6): câu mồi của lead không bảo vệ là tệ nhất,
có bảo vệ thì chép đúng tiếng Anh cao nhất.

**Cách dò.** [`lifemate/eval/asr_tune.py`](../../../lifemate/eval/asr_tune.py) là công cụ dò cấu hình ASR từ lần 1. Công cụ đi qua lần lượt các **bước dò**
(giải mã, câu mồi, chống lặp, VAD). Ở mỗi bước, công cụ thử mọi lựa chọn của bước đó trên một lần chạy thật trên tập dò,
tính **hàm mục tiêu** cho từng lựa chọn, giữ lựa chọn có giá trị nhỏ nhất rồi sang bước sau. Lần 2 thêm vào công cụ:

- Hằng số `PROMPT_FILES` (từ điển: tên câu mồi → đường dẫn file câu mồi) và `GUARDS` (từ điển: hậu tố tên → n của lớp bảo
vệ; `""` là tắt, `"+g3"` là n = 3, `"+g4"` là n = 4). Hàm `prompt_options()` ghép hai từ điển này thành 13 lựa chọn của
bước câu mồi, đặt tên dạng `en_full+g4`. Mỗi lựa chọn là một từ điển các khoá config được ghi đè.
- Tuỳ chọn dòng lệnh `--stages prompt`: chỉ dò bước câu mồi, giữ nguyên các bước khác ở cấu hình xuất phát (mục 4.3.4).
- `--start-option engine=vad`: bật VAD từ đầu cho large-v3. Engine không có VAD (PhoWhisper, ChunkFormer) thì bỏ qua.
- `--objective en_mixed`: chọn hàm mục tiêu của lần 2 (hàm `objective` cùng file). Hàm trả về bộ ba số, so theo thứ tự
từng số (số đầu bằng nhau thì so số thứ hai…), nhỏ hơn là tốt hơn:
  1. p95 có vượt 6 s không (0 hoặc 1);
  2. tổng tỉ lệ lỗi = tỉ lệ bỏ sót giá trị tham số + tỉ lệ câu rỗng + tỉ lệ từ ngoại chép sai (1 − chép đúng) + số lần
     chèn từ câu mồi chia cho số từ ngoại;
  3. WER bỏ số (`wer_nonum`), để phân thắng thua khi hoà.

  Hoà cả ba thì giữ lựa chọn đứng trước. Hàm mục tiêu tính trên **cả** tập dò, nên câu mồi làm câu thường sai thêm sẽ bị phạt
  qua bỏ sót và câu rỗng.
- Đã bỏ khỏi danh sách bước dò (`STAGES`): câu mồi của lead, `hotwords`, nhiệt độ thử lại (quyết định D40 trong
[`docs/decisions.md`](../../decisions.md)).

#### 4.3.4. Cấu hình xuất phát


| Mô hình    | Cấu hình                                                                                                | Lý do                                                                                                                                                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| large-v3   | giải mã tham lam, VAD (cắt khoảng lặng ≥ 500 ms), không chống lặp, `condition_on_previous_text = false` | Thay nhiệt độ thử lại bằng VAD: nhiệt độ thử lại lấy mẫu ngẫu nhiên khi câu hỏng nên không tất định, trái yêu cầu "greedy hoặc `temperature = 0`" của thể lệ, và chỉ hơn VAD 1 giá trị tham số trên 123 ở tập dò lần 1 (40,7% so với 41,5%), trong khi VAD có ít câu bịa hơn một nửa (8 so với 18) |
| PhoWhisper | giải mã tham lam, không chống lặp                                                                       | Như cấu hình thắng lần 1; engine transformers không có VAD                                                                                                                                                                                                                                         |


#### 4.3.5. Tập và số lần đo


| Bước                            | Tập                                    | Số lần đo mỗi mô hình |
| ------------------------------- | -------------------------------------- | --------------------- |
| Dò 13 tổ hợp                    | `asr_tune` đầy đủ (300 mẫu, 1 024 câu) | 13                    |
| Đo cấu hình thắng, đúng một lần | `val_seen`, `val_unseen` đầy đủ        | 2                     |
| Số của tập câu chêm tiếng Anh   | chấm lại tập con bằng `rescore_asr`    | 0 (không chạy GPU)    |


Ước tính trên T4 (theo p95 của lần 1): large-v3 khoảng 20 phút mỗi lần đo trên tập dò, PhoWhisper khoảng 9 phút.

**Chạy trên Kaggle.** [`ops/kaggle/asr_tune/lifemate_asr_tune.py`](../../../ops/kaggle/asr_tune/lifemate_asr_tune.py) là script chạy trên máy GPU T4 của Kaggle: tải code từ
GitHub, cài thư viện, rồi gọi `asr_tune`. Hai biến ở đầu script chọn việc cần làm: `ASR_TUNE` (danh sách mô hình) và
`ASR_TUNE_ARGS` (tuỳ chọn dòng lệnh, ở đây là `--stages prompt --start-option engine=vad --objective en_mixed`). Lần 2
chạy hai phiên vì giới hạn thời gian của Kaggle: v5a (`ASR_TUNE = phowhisper,chunkformer-rnnt`) và v5b
(`ASR_TUNE = large-v3`). Trước đó script [`ops/kaggle/asr_preflight`](../../../ops/kaggle/asr_preflight) chạy mọi tổ hợp trên 3 mẫu bằng CPU, để bắt lỗi cài
đặt và tham số mà không tốn giờ GPU.

### 4.4. Code đã chuẩn bị (tóm tắt)


| File                                                                                   | Vai trò                                                                                       |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `configs/asr_prompts/*.txt`, [`configs/asr_prompts/README.md`](../../../configs/asr_prompts/README.md)                                               | Câu mồi có phiên bản và quy tắc viết                                                          |
| [`lifemate/core/config.py`](../../../lifemate/core/config.py)                                                              | `initial_prompt_file`, `prompt_guard_ngram`, `prompt_guard_max_compression`, kiểm tra giá trị |
| [`lifemate/backends/asr_guard.py`](../../../lifemate/backends/asr_guard.py)                                                       | Bốn dấu hiệu câu hỏng                                                                         |
| [`lifemate/backends/asr.py`](../../../lifemate/backends/asr.py)                                                             | `ASRTranscriber._guard`, `transcribe_batch_unprompted`                                        |
| [`lifemate/eval/asr_tune.py`](../../../lifemate/eval/asr_tune.py)                                                            | 13 tổ hợp, `--stages`, `--start-option`, `--objective en_mixed`                               |
| [`lifemate/eval/eval_asr.py`](../../../lifemate/eval/eval_asr.py), [`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py)                                        | Chỉ số tiếng Anh, `vi_only_wer`, `prompt_insertions`, `n_prompt_retries`                      |
| [`tests/test_asr_guard.py`](../../../tests/test_asr_guard.py), [`tests/test_asr_prompts.py`](../../../tests/test_asr_prompts.py), [`tests/test_asr_tune.py`](../../../tests/test_asr_tune.py), [`tests/test_config.py`](../../../tests/test_config.py) | Test                                                                                          |


### 4.5. Chỉ số

Toàn bộ bộ chỉ số ở mục 2.3, trên ba mức: tập dò đầy đủ (chọn cấu hình), val đầy đủ (so với lần 1, phát hiện câu thường
bị sai thêm), tập con câu chêm tiếng Anh (câu hỏi chính). Thêm `n_prompt_retries` và p95 để biết chi phí thời gian của lớp
bảo vệ.

### 4.6. Bằng chứng offline trước khi chạy

Trên chữ của lần 1 (large-v3, tập dò), mô phỏng lớp bảo vệ: câu bị phát hiện hỏng thì thay bằng chữ của lần chạy không
câu mồi. Dấu hiệu dùng: rỗng, bộ lọc có cắt, trùng cụm 3 từ với câu mồi, tỉ lệ nén &gt; 2,4.


| Cấu hình large-v3 (lần 1) | WER       | Chép đúng từ ngoại (quy tắc nháp) | Câu phải chép lại |
| ------------------------- | --------- | --------------------------------- | ----------------- |
| không câu mồi             | **30,8%** | 46,9%                             | 0                 |
| câu mồi ngắn              | 34,7%     | 51,6%                             | 0                 |
| câu mồi ngắn + bảo vệ     | 31,4%     | 51,9%                             | 62/1 024          |
| câu mồi lead              | 48,6%     | 53,9%                             | 0                 |
| câu mồi lead + bảo vệ     | 32,7%     | **55,7%**                         | 80/1 024          |


Câu mồi có bảo vệ tăng 5–9 điểm chép đúng từ ngoại, WER chỉ tăng 0,6–1,9 điểm, và phải chép lại 6–8% số câu. Đây là
mô phỏng trên câu mồi cũ, chưa phải câu mồi mới.

### 4.7. Kết quả

Lần chạy GPU: [`artifacts/runs/kaggle/asr/10-29_tune_v5a_en_mixed/`](../../../artifacts/runs/kaggle/asr/10-29_tune_v5a_en_mixed/) (PhoWhisper, phiên v5a, T4, 2 giờ 15 phút, không lỗi).
large-v3: [`artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/`](../../../artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/) (phiên v5b), mục 4.7.6–4.7.9.

#### 4.7.1. PhoWhisper: 13 tổ hợp trên tập dò

Tập dò đầy đủ (300 mẫu, 1 024 câu, 394 từ ngoại). Số `prompt_insertions` (chèn từ câu mồi) đã chấm lại sau khi sửa cách đếm: bản đầu tính nhầm "ti vi" hay "TV" của đáp án là khác "tivi" của câu mồi.


| Lựa chọn                 | Chép đúng từ ngoại | Bản nới lỏng | Chèn từ câu mồi | Chép lại | WER       | WER câu chêm | WER câu thuần Việt | p95 (s) |
| ------------------------ | ------------------ | ------------ | --------------- | -------- | --------- | ------------ | ------------------ | ------- |
| không câu mồi            | 31,0               | 34,5         | 0               | 0        | 28,11     | 34,34        | **24,96**          | 3,71    |
| short                    | 33,0               | 37,1         | 0               | 0        | 28,14     | 33,88        | 25,23              | 2,94    |
| short + g3               | 32,5               | 36,5         | 0               | 3        | 28,19     | 34,03        | 25,23              | 2,94    |
| short + g4               | 33,0               | 37,1         | 0               | 0        | 28,14     | 33,88        | 25,23              | 3,04    |
| en\_top10 (± g3, g4)     | 35,5               | 39,1         | 0               | 0        | **27,82** | 33,28        | 25,06              | 3,0     |
| **en\_top20** (± g3, g4) | **37,6**           | **40,4**     | 3               | 0        | 27,95     | **33,09**    | 25,35              | 3,0     |
| en\_full (± g3, g4)      | 34,5               | 38,3         | 4               | 0        | 28,35     | 33,82        | 25,59              | 3,1     |


Ba mức bảo vệ của cùng một câu mồi dài cho chữ giống hệt nhau, nên gộp một dòng. Cấu hình thắng theo hàm mục tiêu
`en_mixed`: **en_top20, không bảo vệ** (hoà với g3, g4 thì giữ lựa chọn đứng trước). Chấm lại với cách đếm chèn đã sửa
vẫn chọn en_top20.

Chênh lệch so với không câu mồi trên tập con câu chêm của tập dò (165 mẫu), CI 95%:


| Câu mồi   | Chép đúng từ ngoại      | WER                 | WER câu thuần Việt      |
| --------- | ----------------------- | ------------------- | ----------------------- |
| en\_top10 | +4,6 \[+1,8; +7,1\]     | −0,5 \[−1,0; 0,0\]  | 0,0 \[−0,5; +0,5\]      |
| en\_top20 | **+6,6** \[+3,7; +9,7\] | −0,5 \[−1,0; 0,0\]  | +0,2 \[−0,4; +0,8\]     |
| en\_full  | +3,5 \[+0,2; +6,4\]     | +0,2 \[−0,4; +0,7\] | **+0,8** \[+0,2; +1,4\] |


#### 4.7.2. PhoWhisper: cấu hình thắng trên val

So với PhoWhisper của lần 1 (câu mồi ngắn, cùng cách giải mã), tập con câu chêm tiếng Anh, CI 95% của chênh lệch:


| Tập                        | Chép đúng từ ngoại: lần 1 → en\_top20 | Chênh lệch          | WER: chênh lệch     | WER câu thuần Việt: chênh lệch | Chèn từ câu mồi |
| -------------------------- | ------------------------------------- | ------------------- | ------------------- | ------------------------------ | --------------- |
| val\_seen (245 câu chêm)   | 36,1 → 37,0                           | +0,9 \[−1,9; +3,6\] | +0,2 \[−0,3; +0,9\] | +0,1 \[−0,7; +1,0\]            | 4               |
| val\_unseen (162 câu chêm) | 36,5 → 36,5                           | 0,0 \[−2,9; +2,8\]  | 0,0 \[−0,7; +0,7\]  | 0,0 \[−0,7; +0,8\]             | 0               |


Trên val đầy đủ: WER 28,1% (seen) và 28,0% (unseen), lần 1 là 27,9% và 27,9%. So với ChunkFormer RNNT, PhoWhisper
en_top20 vẫn kém rõ ở mọi chỉ số: chép đúng từ ngoại thấp hơn 12,9 (seen) và 15,1 (unseen) điểm, CI không chứa 0.

Tách theo từ **có** trong en_top20 và từ **không có** (số từ chép đúng / tổng; mốc là câu mồi ngắn):


| Tập         | Từ có trong câu mồi: mốc → en\_top20 | Từ không có: mốc → en\_top20 |
| ----------- | ------------------------------------ | ---------------------------- |
| tập dò      | 56 → **76** / 161                    | 74 → 72 / 233                |
| val\_seen   | 55 → 58 / 133                        | 66 → 66 / 202                |
| val\_unseen | 20 → 21 / 45                         | 72 → 71 / 207                |


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
"VTV1" thành "vê tivi một". Bản đếm cũ ra 6 vì tính "ti vi" và "TV" của đáp án là khác "tivi".

#### 4.7.4. Phân tích PhoWhisper

- **Câu mồi có giúp trên tập dò nhưng không giúp trên val.** Trên tập dò, en_top20 tăng 6,6 điểm (CI không chứa 0). Trên
val, chênh lệch so với lần 1 là +0,9 và 0,0 điểm, CI chứa 0 và rộng chưa tới ±3,6 điểm. Đây là lý do bắt buộc phải đo
cấu hình thắng một lần trên val: số của tập dò đã bị chọn lọc (13 lựa chọn, chọn cái tốt nhất), nên luôn lạc quan.
- **Lợi ích trên tập dò dồn vào vài từ của vài người nói**: tivi, robot, gara chiếm phần lớn 20 từ tăng thêm. Trên val,
cùng các từ đó không đổi. Mẫu tập dò ít (161 lần xuất hiện của từ có trong câu mồi), nên vài người nói cùng một cách
phát âm đủ tạo ra chênh lệch.
- **Câu mồi không dạy được từ mà mô hình không nghe ra**: SJC và check luôn sai. PhoWhisper được huấn luyện thêm trên
tiếng Việt, nên với các âm này PhoWhisper chọn chữ Việt gần âm, câu mồi không kéo lại được.
- **Câu mồi dài hơn không tốt hơn**: en_full kém en_top20 ở mọi chỉ số và là câu mồi duy nhất làm câu thuần Việt sai thêm có ý
nghĩa thống kê (+0,8 điểm, CI [+0,2; +1,4]).
- **Câu thuần Việt không bị sai thêm** ở en_top10 và en_top20 (CI chứa 0), và không có câu bịa, nên câu mồi an toàn với
PhoWhisper, chỉ là không có ích trên val.

#### 4.7.5. Kết luận PhoWhisper

Câu mồi không cứu được PhoWhisper trên câu chêm tiếng Anh: sau khi dò, PhoWhisper vẫn kém ChunkFormer RNNT khoảng 13–15
điểm chép đúng từ ngoại và khoảng 8 điểm WER. Giữ kết luận của thử nghiệm A: PhoWhisper không phải ứng viên cho câu chêm
tiếng Anh, kể cả làm mô hình phụ khi ghép (thử nghiệm D). Không cần chạy lại phép ghép RNNT + PhoWhisper với cấu hình
mới, vì chữ của PhoWhisper trên val gần như không đổi.

#### 4.7.6. large-v3: 13 tổ hợp trên tập dò

Lần chạy GPU: [`artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/`](../../../artifacts/runs/kaggle/asr/10-29_tune_v5b_en_mixed/) (phiên v5b, T4, 3 giờ 7 phút, không lỗi). Mọi cấu hình
đều bật VAD và giải mã tham lam, nên tất định. p95 của mọi lựa chọn vượt ngưỡng 6 s trên T4 nên phần đầu hàm mục tiêu
bằng nhau, các lựa chọn so với nhau ở phần sau.


| Lựa chọn            | Chép đúng từ ngoại | Bản nới lỏng | Chèn từ câu mồi | Chép lại | WER       | WER câu chêm | WER câu thuần Việt | Câu bịa | Bỏ sót | p95 (s) |
| ------------------- | ------------------ | ------------ | --------------- | -------- | --------- | ------------ | ------------------ | ------- | ------ | ------- |
| không câu mồi       | 47,7               | 52,0         | 0               | 0        | 30,83     | 32,72        | 29,87              | 8       | 51     | 8,41    |
| short               | 51,8               | 56,1         | 46              | 0        | 35,12     | 35,86        | 34,74              | 21      | 51     | 6,76    |
| short + g3          | 51,8               | 56,3         | 5               | 67       | 30,93     | 32,61        | 30,07              | 5       | 49     | 7,08    |
| short + g4          | 52,5               | 57,1         | 6               | 63       | 30,98     | 32,49        | 30,22              | 5       | 49     | 6,99    |
| en\_top10           | 62,4               | 67,0         | 28              | 0        | 30,35     | 31,66        | 29,68              | 5       | 44     | 6,91    |
| en\_top10 + g3 / g4 | 62,7               | 67,3         | 18              | 7        | 30,15     | 31,22        | 29,61              | 3       | 44     | 6,9     |
| en\_top20           | 66,0               | 70,3         | 34              | 0        | 29,93     | 31,18        | 29,30              | 2       | 46     | 7,06    |
| en\_top20 + g3 / g4 | 66,0               | 70,3         | 29              | 4        | 29,88     | 31,18        | **29,23**          | 2       | 46     | 7,08    |
| en\_full            | **70,6**           | **74,9**     | 26              | 0        | 29,79     | **30,03**    | 29,67              | 2       | 42     | 7,30    |
| en\_full + g3       | 70,6               | 74,9         | 26              | 3        | 29,70     | 30,10        | 29,50              | 1       | 42     | 7,32    |
| **en\_full + g4**   | **70,6**           | **74,9**     | 26              | 2        | **29,68** | **30,03**    | 29,50              | **1**   | **42** | 7,31    |


Cấu hình thắng: **en_full + g4** (câu mồi dài nhất, lớp bảo vệ cụm 4 từ). Chấm lại với cách đếm chèn đã sửa vẫn chọn
en_full + g4. `prompt_insertions` có mốc: chép không câu mồi, large-v3 tự viết các từ của en_full mà không được nói 3 lần
trên 1 024 câu; với en_full + g4 là 26 lần. Các lần chèn thật hay gặp: "lịch" thành "list" (5), "Mai" thành "Mail"
(5), "ok" chen vào đầu câu (3), "robot" thay cho từ Việt gần âm (3).

#### 4.7.7. large-v3: cấu hình thắng trên val

Tập con câu chêm tiếng Anh. Mốc "lần 1" là cấu hình nhiệt độ thử lại (không tất định, chỉ tham khảo). CI 95%.


| Tập         | Hệ thống               | Chép đúng từ ngoại      | WER câu chêm | WER câu thuần Việt | Câu bịa | Chèn từ câu mồi |
| ----------- | ---------------------- | ----------------------- | ------------ | ------------------ | ------- | --------------- |
| val\_seen   | ChunkFormer RNNT       | 49,9 \[44,3; 55,3\]     | **24,8**     | **16,9**           | 0       | 1               |
|             | large-v3 lần 1         | 51,6 \[46,4; 56,7\]     | 32,8         | 29,2               | 8       | 0               |
|             | large-v3 en\_full + g4 | **72,2** \[67,2; 77,3\] | 28,7         | 28,8               | 1       | 16              |
| val\_unseen | ChunkFormer RNNT       | 51,6 \[44,8; 58,6\]     | **24,1**     | **17,0**           | 0       | 3               |
|             | large-v3 lần 1         | 56,0 \[48,1; 63,4\]     | 33,9         | 29,1               | 3       | 0               |
|             | large-v3 en\_full + g4 | **61,1** \[52,7; 69,3\] | 33,5         | 30,0               | 0       | 4               |


Chênh lệch của large-v3 en_full + g4 so với RNNT: chép đúng từ ngoại **+22,3** [+16,5; +28,6] (seen) và **+9,5**
[+2,9; +15,1] (unseen); WER câu chêm +3,9 [+1,9; +5,9] và +9,4 [+6,7; +12,3]; WER câu thuần Việt khoảng +12 điểm ở cả
hai tập. Val đầy đủ: WER 29,0% (seen) và 30,2% (unseen); p95 8,8 s và 8,6 s, max 10,8 s trên T4; lớp bảo vệ chép lại 1
và 0 câu.

Tách theo từ có và không có trong en_full (số từ chép đúng / tổng):


| Tập         | Hệ thống               | Từ có trong câu mồi | Từ không có |
| ----------- | ---------------------- | ------------------- | ----------- |
| val\_seen   | RNNT                   | 111 / 184           | 56 / 151    |
|             | large-v3 lần 1         | 89 / 184            | 84 / 151    |
|             | large-v3 en\_full + g4 | **155** / 184       | 87 / 151    |
| val\_unseen | RNNT                   | 43 / 62             | 87 / 190    |
|             | large-v3 lần 1         | 35 / 62             | 106 / 190   |
|             | large-v3 en\_full + g4 | **51** / 62         | 103 / 190   |


Theo nhóm từ, val_seen (large-v3 lần 1 → en_full + g4): từ thường 61 → 82 / 105, viết tắt 43 → 70 / 82, tên 67 → 84 /
133.

#### 4.7.8. Phân tích large-v3

- **Câu mồi có tác dụng thật với large-v3, khác hẳn PhoWhisper.** Từ có trong câu mồi được chép đúng từ 48% lên 84%
(seen) và từ 56% lên 82% (unseen), trên val chưa dùng để dò. Từ không có trong câu mồi không đổi (84 → 87, 106 → 103).
Vậy câu mồi không làm large-v3 "giỏi tiếng Anh hơn" mà chỉ dạy mô hình **cách viết** của đúng những từ được đưa vào: đúng
cơ chế đã nêu ở mục 4.3.1.
- **Lợi ích phụ thuộc độ phủ của câu mồi.** Ở val_seen, 55% số từ ngoại có trong câu mồi, nên tăng 20,6 điểm. Ở
val_unseen (app chưa gặp), chỉ 25% có trong câu mồi, nên chỉ tăng 5,1 điểm. Bộ thi riêng có app lạ, nên số thật nằm
giữa hai mức, gần unseen hơn nếu app lạ chiếm nhiều.
- **Câu mồi dài hơn tốt hơn với large-v3** (en_full &gt; en_top20 &gt; en_top10 &gt; short), ngược với PhoWhisper. Câu mồi dài
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

Chạy lại ChunkFormer RNNT với đúng cấu hình lần 1 trên tập dò và hai tập val, trong cùng phiên Kaggle với thử nghiệm B.
ChunkFormer không nhận câu mồi, nên bước dò câu mồi chỉ có một lựa chọn ("không câu mồi") và công cụ dò chỉ chạy đúng cấu
hình của lần 1. RNNT được chọn vì nhanh nhất và là mô hình chính. Với large-v3, lựa chọn "không câu mồi" của phiên v5b
trùng cấu hình một lần đo của lần 1, nên kiểm chứng được luôn. So từng câu: chữ của hai lần có giống từng ký tự không.

### 5.4. Chỉ số

Tỉ lệ câu có chữ giống hệt lần 1; chênh lệch WER, bỏ sót và `en_word_recall` giữa hai lần.

### 5.5. Kết quả

**ChunkFormer RNNT.** Chạy lại trong phiên v5a (T4, thư viện cài mới), so từng câu với `hyps.jsonl` của lần 1
(`10-28_tune_v3`):


| Tập         | Câu   | Giống hệt lần 1 |
| ----------- | ----- | --------------- |
| tập dò      | 1 024 | 1 024 (100%)    |
| val\_seen   | 916   | 916 (100%)      |
| val\_unseen | 527   | 527 (100%)      |


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

**Thuật toán.** Code: [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `merge_transcripts(primary, secondary)`: nhận chữ của model chính
và model phụ cho cùng một câu, trả về chữ đã ghép. Hàm `merge_hyp_rows` cùng file áp dụng việc này cho từng câu của hai
file `hyps.jsonl`. Các bước:

1. **Gióng hàng.** Tách hai câu thành từ. So hai dãy từ ở dạng đã chuẩn hoá (bỏ thêm dấu câu dính ở hai đầu từ) để tìm cách
 đặt hai câu cạnh nhau sao cho số từ phải thêm, xoá hoặc thay là ít nhất (khoảng cách Levenshtein; hàm `align_words`).
 Kết quả là danh sách cặp vị trí: mỗi cặp là hai từ đứng cùng chỗ, hoặc một từ chỉ có ở một bên.
2. **Gom vùng khác nhau.** Các cặp không khớp nằm giữa hai từ khớp được gom thành một **vùng khác nhau**.
3. **Quyết định từng vùng.** Lấy chữ của model phụ cho cả vùng khi đủ hai điều kiện: (a) phía phụ có từ ngoại (theo quy
 tắc mục 2.1) mà phía chính không có, và (b) vùng phía phụ dài không quá `2 × (số từ phía chính) + 2`, để chặn câu bịa
 dài. Còn lại giữ chữ RNNT. Từ khớp luôn giữ cách viết của RNNT.
4. **Bỏ câu hỏng.** Bỏ hẳn câu của model phụ khi rỗng, hoặc khi bộ lọc ảo giác đã cắt câu đó (chữ trước lọc khác chữ sau
 lọc).

Minh hoạ bước 1–3:

| RNNT | chắc | ví | dùm | còn | trên | ba | triệu |
|---|---|---|---|---|---|---|---|
| large-v3 | Check | ví | dùm, | còn | trên | 3 | triệu. |
| Khớp? | khác | khớp | khớp | khớp | khớp | khác | khớp |
| Vùng | vùng 1: có từ ngoại mới "Check" → lấy large-v3 | | | | | vùng 2: không có từ ngoại → giữ RNNT | |

Điều kiện (a) không đòi phía RNNT "không có từ ngoại", vì chữ rác của RNNT ("s tp") cũng không ghép được âm tiết Việt
nên bị coi là từ ngoại. Điều kiện (b) là một ngưỡng đặt tay trước khi đo, không suy ra từ dữ liệu: cho phép đoạn
của large-v3 dài hơn đoạn RNNT một chút (một từ tiếng Anh có thể ứng với vài âm tiết Việt ở phía RNNT và ngược lại),
`+ 2` để đoạn RNNT rỗng hoặc chỉ 1 từ vẫn nhận được 1–2 từ chèn thêm. Trên thực tế điều kiện này gần như không bao giờ
quyết định (mục 6.8, "Giới hạn độ dài đoạn trên thực tế"). Quy tắc được viết **trước khi** nhìn số và không chỉnh theo val. Tập dò được dùng để kiểm tra lại.

Kết quả: "Check ví dùm còn trên ba triệu". Phép ghép lấy "Check" của large-v3 và giữ cách viết số của RNNT.

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


| Cặp               | Tập                   | Chép đúng từ ngoại       | WER                 | WER câu chêm        | WER câu thuần Việt  |
| ----------------- | --------------------- | ------------------------ | ------------------- | ------------------- | ------------------- |
| RNNT + large-v3   | val\_seen             | **+8,7** \[+2,5; +14,4\] | −0,1 \[−0,4; +0,3\] | −0,8 \[−1,9; +0,3\] | +0,3 \[+0,1; +0,5\] |
|                   | val\_unseen           | **+9,9** \[+4,4; +15,3\] | −0,2 \[−0,6; +0,2\] | −1,1 \[−2,2; +0,1\] | +0,3 \[+0,1; +0,5\] |
|                   | tập dò (large-v3 VAD) | **+8,1** \[+3,1; +12,7\] | −0,1 \[−0,5; +0,2\] | −1,1                | +0,4 \[+0,2; +0,6\] |
| RNNT + PhoWhisper | val\_seen             | −0,6 \[−5,2; +4,4\]      | +0,1 \[−0,1; +0,4\] | +0,4                | 0,0                 |
|                   | val\_unseen           | −2,0 \[−7,3; +3,2\]      | +0,2 \[0,0; +0,5\]  | +0,6                | 0,0                 |
|                   | tập dò                | −2,3 \[−6,3; +1,4\]      | +0,2 \[0,0; +0,5\]  | +0,7                | 0,0                 |


Giá trị tuyệt đối trên tập câu chêm tiếng Anh: RNNT + large-v3 chép đúng **58,5%** (seen) và **61,5%** (unseen) từ
ngoại, so với 49,9% và 51,6% của RNNT; bản nới lỏng 62,1% và 63,1%. WER câu chêm 24,0% và 23,0%.

Số câu bị đổi:


| Cặp               | Tập                   | Câu | Bị đổi | Tốt hơn | Tệ hơn | WER RNNT | WER ghép | Oracle |
| ----------------- | --------------------- | --- | ------ | ------- | ------ | -------- | -------- | ------ |
| RNNT + large-v3   | val\_seen             | 916 | 117    | 48      | 49     | 20,0     | 19,9     | 17,6   |
|                   | val\_seen, câu chêm   | 245 | 99     | 47      | 35     | 24,8     | 24,0     | 21,6   |
|                   | val\_unseen           | 527 | 76     | 41      | 21     | 19,8     | 19,6     | 18,0   |
|                   | val\_unseen, câu chêm | 162 | 67     | 40      | 14     | 24,1     | 23,0     | 21,7   |
| RNNT + PhoWhisper | val\_seen             | 916 | 91     | 28      | 34     | 20,0     | 20,2     | 18,2   |
|                   | val\_unseen           | 527 | 59     | 16      | 29     | 19,8     | 20,0     | 18,3   |


"Tốt hơn" và "tệ hơn" đếm theo số lỗi từ của cả câu; một câu có thể tốt hơn ở từ tiếng Anh nhưng tệ hơn ở chỗ khác.

Ví dụ (val_unseen):


|                  | Chữ                                                          |
| ---------------- | ------------------------------------------------------------ |
| Đáp án           | …có chuyến **RV219** tám giờ bay vào Cần Thơ…                |
| RNNT             | …có chuyến **và vê hai một chín** tám giờ bay vào cần thơ…   |
| Ghép             | …có chuyến **RV219** 8 giờ bay vào cần thơ…                  |
| Đáp án           | Cuộc họp với **Mr. Fujimoto** hôm đó mấy giờ hè?             |
| RNNT             | cuộc họp với **micro full mô tô** hôm đó mấy giờ nè          |
| Ghép             | cuộc họp với **Mr. Funmoto** hôm đó mấy giờ nè               |
| Đáp án (ca hỏng) | Mai không mưa thì giữ cho anh cuốn Mùa đông ở Lisbon…        |
| RNNT             | mai không mưa thì giữ cho anh cuốn mùa đông ở disban…        |
| large-v3         | **Mời các bạn thử chọn những quần bộ đông nhất của** Lisbon… |
| Ghép             | **Mời các bạn thử chọn những quần bộ đông nhất của** Lisbon… |


#### 6.5.2. Với large-v3 cấu hình thắng của thử nghiệm B (en_full + g4)

Chạy lại `asr_merge` trên chữ thật của phiên v5b, cùng quy tắc, không chỉnh gì. Kết quả ở
[`artifacts/runs/local/asr/10-29_en_mixed_merge_v3/`](../../../artifacts/runs/local/asr/10-29_en_mixed_merge_v3/). Chênh lệch so với RNNT (điểm %, CI 95% của chênh lệch):


| Tập                   | Chép đúng từ ngoại         | WER                 | WER câu chêm            | WER câu thuần Việt  | Bỏ sót giá trị tham số |
| --------------------- | -------------------------- | ------------------- | ----------------------- | ------------------- | ---------------------- |
| val\_seen, câu chêm   | **+22,7** \[+17,0; +28,9\] | −0,4 \[−1,0; +0,2\] | **−2,0** \[−3,1; −1,0\] | +1,1 \[+0,6; +1,7\] | +1,5 \[−6,8; +8,7\]    |
| val\_unseen, câu chêm | **+12,3** \[+6,4; +17,5\]  | −0,4 \[−1,0; +0,2\] | **−1,3** \[−2,2; −0,3\] | +0,6 \[+0,1; +1,2\] | −9,4 \[−20,6; 0,0\]    |
| val\_seen đầy đủ      | +22,7 \[+16,8; +28,2\]     | +0,1 \[−0,4; +0,6\] |                         | +1,1 \[+0,6; +1,5\] | +0,9 \[−4,2; +5,7\]    |
| val\_unseen đầy đủ    | +12,3 \[+7,0; +17,6\]      | 0,0 \[−0,4; +0,4\]  |                         | +0,6 \[+0,3; +1,0\] | −5,6 \[−13,0; 0,0\]    |
| tập dò, câu chêm      | +23,4 \[+17,3; +29,0\]     | −0,2 \[−0,9; +0,5\] | −1,8 \[−2,9; −0,8\]     | +1,3 \[+0,6; +2,0\] | −4,1 \[−9,5; 0,0\]     |


Giá trị tuyệt đối, câu chêm tiếng Anh: chép đúng từ ngoại **72,5%** (seen) và **63,9%** (unseen), so với RNNT 49,9% và
51,6%, và bản ghép với large-v3 lần 1 58,5% và 61,5%. WER câu chêm 22,8% ở cả hai tập (RNNT 24,8% và 24,1%).

Số câu bị đổi, tách câu chêm và câu thuần Việt:


| Tập                         | Câu | Bị đổi | Tốt hơn | Tệ hơn | WER RNNT | WER ghép | Oracle |
| --------------------------- | --- | ------ | ------- | ------ | -------- | -------- | ------ |
| val\_seen, câu chêm         | 245 | 130    | 78      | 33     | 24,8     | 22,8     | 20,7   |
| val\_seen, câu thuần Việt   | 671 | 44     | **1**   | **37** | 17,7     | 18,8     | 15,7   |
| val\_unseen, câu chêm       | 162 | 70     | 37      | 14     | 24,1     | 22,8     | 21,7   |
| val\_unseen, câu thuần Việt | 365 | 20     | **1**   | **17** | 17,5     | 18,2     | 16,2   |
| tập dò, câu chêm            | 276 | 138    | 74      | 36     | 25,8     | 24,0     | 21,7   |
| tập dò, câu thuần Việt      | 748 | 58     | **1**   | **48** | 17,5     | 18,8     | 16,0   |


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
- **Nhưng câu thuần Việt bị sai thêm rõ hơn**: +1,1 (seen) và +0,6 (unseen) điểm WER, CI không chứa 0. Ở câu thuần Việt, mọi
câu bị đổi gần như đều tệ hơn (37/44, 17/20, 48/58; mỗi tập chỉ 1 câu tốt hơn). Nguyên nhân: large-v3 viết ra từ tiếng
Anh mà người dùng không nói, và các từ đó là "từ tiếng Anh mới" nên quy tắc ghép lấy vào. Trên tập dò, 148 từ bị lấy
nhầm như vậy; 124 từ không có trong câu mồi ("Toyota D-Diều" cho "tôi ra đê"), 24 từ là từ của câu mồi ("Mail" cho
"mai", "list" cho "lịch"). Mục 6.8 xử lý phần này.
- **Thứ hạng giữ trên tập dò**: tập dò cho cùng hướng và cùng độ lớn (+23,4 chép đúng, +1,3 câu thuần Việt), nên mọi
chỉnh sửa quy tắc chọn trên tập dò sẽ chuyển được sang val.

### 6.7. Kết luận

Ghép RNNT + large-v3 (en_full + g4) là cấu hình ASR tốt nhất cho câu chêm tiếng Anh của lần 2: chép đúng từ ngoại
72,5% / 63,9% (seen / unseen) so với 49,9% / 51,6% của RNNT, WER câu chêm thấp hơn RNNT, WER tổng không đổi. Chưa sẵn
sàng để nộp vì ba việc:

1. **Giảm lỗi tăng thêm ở câu thuần Việt** (+0,6–1,1 điểm WER so với RNNT). Đã giảm còn +0,3–0,4 điểm bằng từ điển ngữ cảnh (mục 6.8).
2. **Đo thời gian và bộ nhớ** khi chạy cả hai mô hình cạnh vLLM trên GPU từ 24 GB. Trên T4, large-v3 có p95 8,6–8,8 s
 mỗi mẫu.
3. **Viết engine ASR ghép** trong [`lifemate/backends/asr.py`](../../../lifemate/backends/asr.py). Hiện phép ghép chỉ có ở dạng chấm offline (`asr_merge`).

Tổng tham số: RNNT 0,11B + large-v3 1,55B + LLM 4,02B = 5,68B, dưới trần 6B.

### 6.8. Bộ lọc ngữ cảnh cho phép ghép

**Vấn đề.** Quy tắc ở mục 6.3 lấy mọi đoạn của large-v3 có "từ tiếng Anh mới". Những từ large-v3 viết ra dù người dùng
không nói ("Mai" thành "Mail", "tôi ra đê" thành "Toyota D-Diều") cũng là từ tiếng Anh mới, nên bị lấy theo. Kết quả là
câu thuần Việt sai thêm 1,1 điểm (seen) và 0,6 điểm (unseen) so với RNNT.

**Vì sao không chặn theo mặt chữ.** Một điều kiện như "không lấy vùng khi phía RNNT là âm tiết Việt có dấu hợp lệ" chặn
cả ca có lợi: khi người dùng nói "Zalo", RNNT chép "giá lô" (âm tiết Việt có dấu hợp lệ). Phần lợi chính của phép ghép
rơi đúng vào loại ca này, nên không dùng điều kiện đó.

**Ý tưởng.** Chỉ nhìn chữ của hai bản thì không phân biệt được "sửa đúng" với "chèn nhầm". Cần một tín hiệu cho biết từ
tiếng Anh có thật sự được nói hay không. Tín hiệu rẻ nhất là **ngữ cảnh của mẫu**: người dùng nói "Zalo" thường vì máy có
cài app nhắn tin, hoặc trợ lý vừa nhắc tới; còn "Mail" chen vào câu nhắc lịch thì thường không liên quan gì tới mẫu.

**Giải pháp.** Code: [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py).

- `sample_context_words(sample, catalog_apps)`: nhận một mẫu và catalog (từ điển mã app → mục catalog của app đó), trả về
  **từ điển ngữ cảnh** của mẫu: một tập từ đã chuẩn hoá (`frozenset[str]`). Tập này gồm:
  - mã app và toàn bộ chữ trong mục catalog (tên, mô tả, tên tool, mô tả và giá trị liệt kê của tham số) của mọi app
    trong `installed_apps`;
  - các trường khác của ngữ cảnh người dùng (giờ, vị trí, hồ sơ);
  - lượt trợ lý trong lịch sử: câu trả lời, lời gọi tool và kết quả tool.

  Không dùng `state.json` (thể lệ cấm), và không dùng lượt người dùng trong lịch sử (lúc thi lượt này chỉ có âm thanh).
- `english_filter(policy, context_words, prompt_words)`: trả về hàm nhận một từ (đã chuẩn hoá) và cho biết có nhận từ đó không.
  `prompt_words` là tập từ ngoại của câu mồi. `CONTEXT_POLICIES` là bộ bốn cách lọc:

  | Cách lọc | Từ của câu mồi | Từ ngoài câu mồi |
  |---|---|---|
  | `all` (không lọc) | nhận | nhận |
  | `context` | chỉ nhận khi có trong ngữ cảnh | chỉ nhận khi có trong ngữ cảnh |
  | `context_prompt` | chỉ nhận khi có trong ngữ cảnh | nhận |
  | `context_other` | nhận | chỉ nhận khi có trong ngữ cảnh |

- `merge_transcripts(primary, secondary, accept_english)`: như mục 6.3, nhưng ở bước 3 chỉ đếm những từ tiếng Anh mới mà
  bộ lọc nhận. `merge_hyp_rows` nhận thêm từ điển mã mẫu → bộ lọc. Lệnh `python -m lifemate.eval.asr_merge` có thêm
  `--policy` và `--prompt-file`.

**Dữ liệu dựng từ điển, ví dụ thật** (mẫu `train_0130`, rút gọn). Trường `context` của mẫu:

```
current_time: 2026-11-17T18:00:00+07:00      timezone: Asia/Ho_Chi_Minh
location:     {lat: 21.0285, lng: 105.8542}
user_profile: {name: "Anh Khoa", home_address: "25 Láng Hạ, Đống Đa, Hà Nội", city: "Hà Nội", ...}
installed_apps: {weather, calendar, contacts, sms, ride_hailing, tv, ...}
```

`installed_apps` chỉ là mã app. Chữ đầy đủ của app nằm trong catalog ([`data/train+public_test/train/train_tools_catalog.json`](../../../data/train+public_test/train/train_tools_catalog.json)), ví dụ mục `sms`:
`name_vi: "Tin nhắn SMS"`, `description: "Nhắn tin SMS"`, tool `sms.send` ("Gửi SMS cho liên hệ hoặc số điện
thoại…"), `sms.read_recent` ("Tin nhắn gần đây"). Một lượt trợ lý trong lịch sử của cùng mẫu:

```
assistant: "Đã đặt báo thức 7 giờ sáng mai."
  tool_calls:   alarm.create {time: "2026-11-18T07:00:00+07:00"}
  tool_results: {ok: true, id: "alm_03", ...}
```

Lượt trợ lý có sẵn dạng chữ trong dữ liệu, **không mô hình ASR nào chép**. Chỉ lượt người dùng (trong lịch sử và câu
cuối) là âm thanh; RNNT và large-v3 cùng chép mọi lượt người dùng, mỗi mô hình ra một bản chữ.

**Cách dựng từ điển.** `sample_context_words` gom mọi chuỗi chữ của ba nguồn trên (hàm `_strings`: nhận một giá trị JSON
lồng nhau, trả về mọi chuỗi bên trong, **gồm cả khoá lẫn giá trị** của dict), nối lại, tách thành từ (ranh giới là mọi
ký tự không phải chữ cái, nên `user_profile` thành `user`, `profile`), chuẩn hoá từng từ (chữ thường, bỏ dấu, "đ"
thành "d") và bỏ trùng. Ở `train_0130`, từ điển có `sms`, `tin`, `nhan`, `alarm`, `weather`, `ha`, `noi`… Trên tập dò,
mỗi mẫu có trung bình khoảng 330 từ. Từ điển chỉ dựng từ chữ, không cần âm thanh.

**Từ điển chỉ được tra cho từ tiếng Anh.** Bộ lọc chỉ hỏi từ điển về những từ mà quy tắc tìm từ ngoại (mục 2.1) đã nhận
là từ ngoại, tức từ không ghép được thành âm tiết Việt ("zalo", "web", "youtube"). Từ điển cũng chứa từ tiếng Việt đã
chuẩn hoá ("đồ" thành "do"), nhưng "do" là âm tiết Việt hợp lệ nên không bao giờ là từ ngoại và không bao giờ được tra.
Phần tiếng Việt của từ điển vì vậy không ảnh hưởng kết quả.

**Từng bước trên một câu.** Phép ghép xét từng câu nói và từng đoạn khác nhau trong câu, không phân loại cả câu.

1. Lấy hai bản chữ của cùng một câu: RNNT và large-v3.
2. Đặt hai bản cạnh nhau, ghép từ ứng với từ (mục 6.3). Chỗ giống nhau giữ nguyên.
3. Ở mỗi đoạn khác nhau, tìm từ tiếng Anh mới: từ ngoại có ở đoạn large-v3 mà đoạn RNNT không có.
4. Bộ lọc (`context_other`) giữ một từ tiếng Anh mới khi từ đó có trong câu mồi hoặc có trong từ điển ngữ cảnh của mẫu.
5. Lấy đoạn của large-v3 khi (a) sau khi lọc còn ít nhất một từ và (b) đoạn của large-v3 không dài quá
   `2 × (số từ đoạn RNNT) + 2`. Còn lại giữ đoạn RNNT.

| Người dùng nói | RNNT | large-v3 | Đoạn khác nhau (RNNT → large-v3) | Bước 3–5 | Kết quả |
|---|---|---|---|---|---|
| nhắn Zalo cho mẹ (ví dụ ở phần tóm tắt) | nhắn giá lô cho mẹ | nhắn Zalo cho mẹ | [giá lô] → [Zalo] | "zalo" có trong câu mồi, giữ; 1 ≤ 2 × 2 + 2 | nhắn Zalo cho mẹ |
| Bật đèn bếp lên giúp tôi. (`train_0103`) | bật đèn bếp lên giúp tôi | Bạn điền web lên giúp tôi | [bật đèn bếp] → [Bạn điền web] | "web" không có trong câu mồi và từ điển, bị loại; còn 0 từ, hỏng (a) | bật đèn bếp lên giúp tôi |
| Ừ, bật VTV3 lên cho chị xem. (`train_1664`, lượt 6) | ừ chị xem | Oh, bóc phim tức giảm bản lên trái tim. | [ừ chị xem] → cả câu | Không lọc thì còn "oh"; 9 > 2 × 3 + 2, hỏng (b) | ừ chị xem |

**Bản ghép không lọc và bản có bộ lọc ngữ cảnh trên câu thật** (tập dò):

| Người dùng nói | Ghép không lọc | Ghép có bộ lọc ngữ cảnh |
|---|---|---|
| Bật đèn bếp lên giúp tôi. | Bạn điền **web** lên giúp tôi | bật đèn bếp lên giúp tôi |
| Ví của tôi còn bao nhiêu tiền vậy? | **Vip** ổn thông qua mạng **YouTube**. | ví của tôi còn bao nhiêu tiền vậy |
| Ngoài trời bây giờ nóng không ạ? | **Why do they don't come with?** | ngoài trời bây giờ nóng không mẹ |
| Đặt báo thức 6 giờ sáng mai cho anh, rồi chuyển ví điện tử cho Lan ít tiền nha. | … cho **Lime** ít tiền nha | … cho lan ít tiền nha |

**Vì sao large-v3 viết ra chữ tiếng Anh mà người dùng không nói.** large-v3 là mô hình nhiều thứ tiếng, phần lớn dữ
liệu huấn luyện là tiếng Anh. Khi một đoạn tiếng Việt nghe không rõ, mô hình có thể đoán thành chuỗi tiếng Anh gần âm
("tôi ra đê" thành "Toyota D-Diều", "ví của tôi" thành "Vip ổn"). Câu mồi `en_full_v1` là đoạn tiếng Việt chêm dày
tiếng Anh, nên đẩy mô hình về phía viết tiếng Anh: với câu mồi này large-v3 chèn từ của câu mồi vào 26 trên 1 024 câu ở
tập dò, không câu mồi chỉ 3 câu (phần tóm tắt). Quy tắc tìm từ ngoại coi "Toyota", "Vip", "YouTube" là từ tiếng Anh mới, nên
bản ghép không lọc lấy theo. Những từ tự bịa này hầu như không có trong ngữ cảnh của mẫu (không app nào tên "Toyota",
trợ lý cũng không nhắc), nên bộ lọc ngữ cảnh chặn được phần lớn.

**Giới hạn độ dài đoạn trên thực tế.** Đếm mọi đoạn khác nhau có ít nhất một từ tiếng Anh mới được giữ sau bước 4:

| Tập | Không lọc: số đoạn / bị (b) chặn | Có bộ lọc ngữ cảnh: số đoạn / bị (b) chặn |
|---|---|---|
| tập dò | 229 / 0 | 116 / 0 |
| val_seen | 191 / 1 | 110 / 0 |
| val_unseen | 98 / 0 | 40 / 0 |

Điều kiện (b) chỉ chặn 1 trên 518 đoạn, và không chặn đoạn nào khi đã có bộ lọc ngữ cảnh. Điều kiện này cũng không chặn
được ca "Lisbon" ở mục 6.5, vì phía RNNT tại đó cũng dài. Bộ lọc ngữ cảnh là phần làm việc thật; độ dài đoạn được đưa
vào làm một đặc trưng của bộ phân loại ở lần 3 ([`docs/reports/benchmark/asr_v3.md`](asr_v3.md), thử nghiệm H).

**Bằng chứng trước khi chọn** (tập dò, đếm từng từ tiếng Anh mới mà bản ghép không lọc lấy từ large-v3; "đúng" là từ có trong
đáp án):

| Nhóm từ | Đúng: có / không có trong ngữ cảnh | Sai: có / không có trong ngữ cảnh |
|---|---|---|
| Từ của câu mồi | 54 / 12 | 8 / 16 |
| Từ ngoài câu mồi | 27 / 23 | 6 / 118 |

Từ ngoài câu mồi chiếm 124 trên 148 từ sai. Ngữ cảnh loại được 118 từ sai trong nhóm này, đổi lại mất 23 từ đúng.

Ngữ cảnh ở đây gồm cả lượt trợ lý. Khi chỉ dùng catalog của app đã cài, số từ đúng có trong ngữ cảnh ít hơn hẳn: 34 thay
cho 54 ở từ của câu mồi, 8 thay cho 27 ở từ ngoài câu mồi.

**Kết quả trên tập dò** (1 024 câu; WER câu chêm và chép đúng từ tiếng Anh tính trên các câu chêm của tập dò):

| Hệ thống | WER mọi câu | WER câu chêm | WER câu thuần Việt | Chép đúng từ tiếng Anh |
|---|---|---|---|---|
| RNNT | 20,3 | 25,8 | **17,5** | 49,0 |
| `all` (không lọc) | 20,6 | **24,0** | 18,8 | **72,3** |
| `context` | **19,9** | 24,5 | 17,6 | 66,2 |
| `context_prompt` | 20,6 | 24,4 | 18,7 | 70,1 |
| **`context_other`** | **19,9** | 24,2 | 17,8 | 69,0 |

Chênh lệch so với bản ghép không lọc (CI 95%): `context_other` giảm WER câu thuần Việt [−1,5; −0,6] và giảm chép đúng từ tiếng
Anh [−6,4; −0,5]; `context` giảm WER câu thuần Việt nhiều hơn một chút [−1,7; −0,8] nhưng mất nhiều tiếng Anh hơn [−9,9;
−2,3]; `context_prompt` gần như không đổi WER câu thuần Việt [−0,3; 0,0].

**Chọn `context_other`.** Lý do: giảm gần hết phần lỗi tăng thêm ở câu thuần Việt (từ +1,3 xuống +0,3 điểm so với RNNT) mà mất ít
tiếng Anh nhất trong hai cách làm được việc đó. Tiêu chí đặt trước khi đo là "chép đúng từ tiếng Anh không giảm có ý
nghĩa thống kê". Trên tập dò, `context_other` không đạt tiêu chí này (−3,3 điểm, CI không chứa 0). Vẫn chọn, vì đây là
đánh đổi theo đúng hướng: thiếu một từ tiếng Anh thì phép ghép chỉ quay về chữ RNNT như trước, còn lấy nhầm một từ thì
có thể làm LLM gọi sai tool.

Trước khi chọn, đã thử thêm một ngoại lệ: nhận từ ngoài câu mồi khi đoạn của RNNT tại đó cũng là chữ không ra tiếng Việt.
Ngoại lệ này cứu được 11 từ đúng nhưng nhận lại 41 từ sai, nên bỏ.

**Kết quả trên tập kiểm tra** (đo một lần, sau khi đã chọn):

| Tập | Hệ thống | WER mọi câu | WER câu chêm | WER câu thuần Việt | Chép đúng từ tiếng Anh |
|---|---|---|---|---|---|
| val_seen | RNNT | 20,0 | 24,8 | **17,7** | 49,9 |
| | `all` | 20,1 | 22,8 | 18,8 | **72,5** |
| | **`context_other`** | **19,4** | **22,1** | 18,1 | 72,2 |
| val_unseen | RNNT | 19,8 | 24,1 | **17,5** | 51,6 |
| | `all` | 19,8 | **22,8** | 18,2 | **63,9** |
| | **`context_other`** | **19,5** | **22,8** | 17,8 | 63,1 |

Chênh lệch, CI 95%:

| Tập | So với | WER mọi câu | WER câu chêm | WER câu thuần Việt | Chép đúng từ tiếng Anh |
|---|---|---|---|---|---|
| val_seen | `all` | [−1,0; −0,4] | [−1,3; −0,1] | [−1,1; −0,4] | [−3,1; +2,6] |
| val_seen | RNNT | [−0,9; −0,2] | [−3,6; −1,8] | [+0,2; +0,6] | [+17,2; +27,2] |
| val_unseen | `all` | [−0,5; 0,0] | [−0,8; +0,7] | [−0,7; −0,1] | [−4,4; +2,7] |
| val_unseen | RNNT | [−0,5; 0,0] | [−1,9; −0,8] | [+0,1; +0,5] | [+7,9; +15,7] |

**Phân tích.**

- Trên tập kiểm tra, `context_other` mất ít tiếng Anh hơn trên tập dò: −0,3 và −0,8 điểm, CI chứa 0. Trên tập dò, mức mất
  là −3,3 điểm.
- Phần lỗi tăng thêm ở câu thuần Việt giảm có ý nghĩa thống kê ở cả hai tập, nhưng chưa hết: vẫn hơn RNNT 0,4 (seen) và 0,3
  (unseen) điểm. Phần còn lại đến từ từ của câu mồi, vì `context_other` nhận mọi từ của câu mồi.
- WER tính trên mọi câu lần đầu thấp hơn RNNT có ý nghĩa thống kê ở seen. Bản ghép nay tốt hơn RNNT ở mọi chỉ số, trừ
  câu thuần Việt.
- Ngữ cảnh mạnh nhất khi mẫu có lịch sử. Mẫu không có lịch sử chỉ còn catalog, nên với các mẫu đó phép ghép gần như chỉ
  nhận từ của câu mồi.

**Kết luận.** Dùng `context_other` làm quy tắc ghép của lần 2. Kết quả: [`artifacts/runs/local/asr/10-29_en_mixed_merge_v4/`](../../../artifacts/runs/local/asr/10-29_en_mixed_merge_v4/)
(`merge_<cách lọc>__<tập>.hyps.jsonl`, `cmp_*.json`). Bản `all` của lần này giống hệt từng byte bản ghép ở mục 6.5.2.

**Việc còn lại.** Lọc nốt từ của câu mồi theo độ tin cậy của từng từ, đếm trên 793 mẫu train chưa dùng. Cách làm và lý
do cần một phiên GPU: mục "Quy tắc ghép hai mô hình và bộ lọc ngữ cảnh" ở phần tóm tắt.

**Giới hạn.**

- Tập dò chỉ có khoảng 150 từ sai bị lấy vào, nên số đếm theo nhóm còn ít.
- Số "đúng" và "sai" dựa vào quy tắc tìm từ ngoại (mục 2.1; rà tay cho độ chính xác 99,7%, độ phủ 94,3%).
- Mục tiêu cuối là LLM gọi đúng tool. Một lỗi chèn nhầm có thể không đổi lời gọi tool nào, nên cách lọc thắng cần được
  kiểm tra lại bằng điểm của agent khi agent chạy được trên chữ ASR.

## Giới hạn đã biết

- Thời gian đo trên T4 chạy ASR riêng, chưa phải A30 và chưa chạy cạnh vLLM.
- Mỗi nhóm từ chỉ có vài chục lần xuất hiện (từ mượn chỉ 15 và 2), nên chênh vài từ đã là vài điểm phần trăm; dùng CI ở
mục 3.5 và 6.5 thay vì so số trần.
- Quy tắc tìm từ tiếng Anh bỏ sót từ trông như âm tiết Việt: tìm được 94,3% số từ tiếng Anh theo rà tay (90,6% trên
  val_unseen); nhóm của từ viết hoa đầu câu phụ thuộc tập đang chấm. Chấm lại theo nhãn rà tay lệch tối đa 1,1 điểm và
  không đổi thứ hạng (mục 2.1).
- Chép đúng từ ngoại so theo từ có mặt trong câu, không kiểm tra vị trí, nên có thể khớp nhầm.
- Từ điển ngữ cảnh lấy cả khoá JSON của catalog và của `context` (`type`, `description`, `properties`, `content`, `name`,
  `user`, `profile`…), nên các từ này có trong từ điển của mọi mẫu: large-v3 bịa ra đúng những từ này thì bộ lọc vẫn nhận.
  Trên tập dò, bỏ khoá chỉ bớt 1 từ lấy nhầm ("group") và không mất từ đúng nào. Lần 3 bỏ khoá ([`docs/reports/benchmark/asr_v3.md`](asr_v3.md), mục 2.1).
- Số của large-v3 ở thử nghiệm A và ở mục 6.5.1 đến từ cấu hình nhiệt độ thử lại (không tất định); mục 4.7.6–4.7.7 và 6.5.2
dùng cấu hình tất định.
- Cấu hình thắng được chọn trên tập dò 300 mẫu (394 từ ngoại); số trên tập dò lạc quan hơn thực tế, dùng số val.
- Câu mồi chọn từ theo chữ large-v3 beam 5 trong [`data/asr/train_asr.json`](../../../data/asr/train_asr.json), không phải cấu hình tham lam + VAD đang dò.


## Phụ lục A. Bảng tra code

Mọi đường dẫn tính từ thư mục gốc repo. "Từ điển" là kiểu `dict` của Python (khoá → giá trị), "tập" là `set`/`frozenset`
(các phần tử không trùng, không thứ tự), "danh sách" là `list` (có thứ tự, có thể trùng).

### A.1. Tìm từ ngoại: [`lifemate/eval/foreign_words.py`](../../../lifemate/eval/foreign_words.py)

| Tên | Loại | Đầu vào → đầu ra, hoặc nội dung | Việc |
|---|---|---|---|
| `ForeignWord` | lớp dữ liệu | ba trường: `text` (từ như viết trong câu, "Zalo"), `folded` (dạng đã chuẩn hoá, "zalo"), `group` (một trong `WORD_GROUPS`) | Một từ ngoại tìm được |
| `WORD_GROUPS` | hằng, bộ 4 chuỗi | `common`, `acronym`, `name`, `loanword` | Tên 4 nhóm: từ thường, viết tắt, tên, từ mượn |
| `VIETNAMIZED_LOANWORDS` | hằng, tập chuỗi | "tivi", "gara" | Từ mượn đã Việt hoá, xếp nhóm `loanword` |
| `ENGLISH_SPELLED_LIKE_VIETNAMESE` | hằng, tập chuỗi | "chat", "top", "hot" | Từ tiếng Anh ghép được thành âm tiết Việt không dấu; liệt kê tay vì quy tắc âm tiết không bắt được |
| `EQUIVALENT_SPELLINGS` | hằng, từ điển | từ → các cách viết khác: "tivi" → "tv"; "ok" → "okay", "oke"; "email" → "mail" | Cách viết LLM vẫn hiểu đúng, dùng cho bản nới lỏng |
| `_VIETNAMESE_SYLLABLE` | hằng, biểu thức chính quy | phụ âm đầu tuỳ chọn + 1–3 nguyên âm + phụ âm cuối tuỳ chọn | Kiểm tra một từ không dấu có phải âm tiết Việt không |
| `fold_text` | hàm | chuỗi → chuỗi | Chuẩn hoá chữ: chữ thường, bỏ dấu, "đ" thành "d" |
| `folded_words`, `folded_word_set` | hàm | chuỗi → danh sách / tập từ đã chuẩn hoá | Tách câu thành từ (ranh giới là mọi thứ không phải chữ cái) |
| `find_foreign_words` | hàm | câu, tập `lowercase_vocabulary` → danh sách `ForeignWord` | Quy tắc mục 2.1 |
| `has_foreign_word` | hàm | câu → đúng/sai | Câu có ít nhất một từ ngoại không |
| `match_words_in_transcript` | hàm | danh sách `ForeignWord` của đáp án, chữ ASR, `lenient` → danh sách đúng/sai | Từng từ ngoại của đáp án có được chép đúng không, đếm theo số lần xuất hiện |
| `equivalent_spellings` | hàm | từ đã chuẩn hoá → tập cách viết | Mọi cách viết tương đương của một từ, tra hai chiều |
| `contains_word` | hàm | từ đã chuẩn hoá, danh sách từ đã chuẩn hoá → đúng/sai | Dãy từ có chứa từ đó không, tính cả cách viết tương đương và từ bị tách 2–3 phần ("ti vi") |

| [`lifemate/eval/foreign_words_audit.py`](../../../lifemate/eval/foreign_words_audit.py) › `audit_rule` | hàm | danh sách mẫu, nhãn rà tay → từ điển kết quả | So quy tắc với nhãn rà tay: số lần đúng, nhận nhầm, bỏ sót, độ khớp nhóm (mục 2.1) |
| [`lifemate/eval/foreign_words_audit.py`](../../../lifemate/eval/foreign_words_audit.py) › `english_recall` | hàm | danh sách mẫu, dòng `hyps.jsonl`, nhãn rà tay → từ điển kết quả | Tỉ lệ chép đúng từ tiếng Anh theo quy tắc và theo nhãn rà tay |
| [`lifemate/eval/foreign_words_audit.py`](../../../lifemate/eval/foreign_words_audit.py) › `load_hand_labels` | hàm | đường dẫn file nhãn → từ điển (mã mẫu, lượt) → thứ tự từ → nhóm | Đọc nhãn rà tay ([`data/labels/foreign_words_hand_v1.jsonl`](../../../data/labels/foreign_words_hand_v1.jsonl), ngoài git) |

### A.2. Chấm điểm: [`lifemate/eval/`](../../../lifemate/eval/)

| File › tên | Đầu vào → đầu ra | Việc |
|---|---|---|
| [`lifemate/eval/eval_asr.py`](../../../lifemate/eval/eval_asr.py) › `evaluate` | danh sách mẫu, bộ chép chữ, câu mồi → `AsrReport` | Chép (hoặc đọc lại) mọi câu, tính mọi chỉ số mục 2.3 |
| [`lifemate/eval/eval_asr.py`](../../../lifemate/eval/eval_asr.py) › `AsrReport` | lớp dữ liệu | Báo cáo của một lần đo: mỗi chỉ số một trường, ghi ra file JSON; trường `counts` là từ điển tên → số nguyên chứa tử số và mẫu số |
| [`lifemate/eval/make_splits.py`](../../../lifemate/eval/make_splits.py) › `en_mixed_ids` | danh sách mẫu → danh sách mã mẫu | Chọn mẫu có câu chêm tiếng Anh (mục 2.2) |
| [`lifemate/eval/rescore_asr.py`](../../../lifemate/eval/rescore_asr.py) › `ReplayTranscriber` | danh sách dòng `hyps.jsonl` | Bộ chép chữ giả: trả lại chữ đã ghi thay vì nghe âm thanh (mục 2.4) |
| [`lifemate/eval/asr_compare.py`](../../../lifemate/eval/asr_compare.py) › `compare` | danh sách mẫu, từ điển tên hệ thống → (dòng `hyps.jsonl`, câu mồi) → từ điển kết quả | Chỉ số, CI 95% và CI của chênh lệch so với hệ thống đầu tiên, bằng bootstrap có ghép cặp |
| [`lifemate/eval/asr_compare.py`](../../../lifemate/eval/asr_compare.py) › `METRICS` | hằng, từ điển | tên chỉ số → (tên tử số, tên mẫu số) trong `counts`, ví dụ `en_word_recall` → (`en_hits`, `en_words`) |
| [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `align_words` | hai danh sách từ → danh sách cặp vị trí | Gióng hàng theo khoảng cách Levenshtein (mục 6.3) |
| [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `merge_transcripts` | chữ model chính, chữ model phụ → chữ đã ghép | Quy tắc ghép mục 6.3 |
| [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `merge_hyp_rows` | hai danh sách dòng `hyps.jsonl`, từ điển mã mẫu → bộ lọc → danh sách dòng đã ghép | Ghép từng câu theo (mã mẫu, lượt); bỏ câu phụ rỗng hoặc bị lọc ảo giác |
| [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `sample_context_words` | mẫu, catalog → tập từ đã chuẩn hoá | Từ điển ngữ cảnh của mẫu (mục 6.8) |
| [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `english_filter` | cách lọc, từ điển ngữ cảnh, từ của câu mồi → hàm nhận/từ chối một từ | Bộ lọc bước 3 của phép ghép (mục 6.8) |
| [`lifemate/eval/asr_merge.py`](../../../lifemate/eval/asr_merge.py) › `CONTEXT_POLICIES` | hằng, bộ tên | Bốn cách lọc: `all`, `context`, `context_prompt`, `context_other` |

### A.3. Câu mồi và lớp bảo vệ

| File › tên | Đầu vào → đầu ra, hoặc nội dung | Việc |
|---|---|---|
| `configs/asr_prompts/*_v1.txt` | file chữ, một dòng | Các câu mồi có phiên bản (mục 4.3.1); quy tắc viết ở [`configs/asr_prompts/README.md`](../../../configs/asr_prompts/README.md) |
| [`lifemate/core/config.py`](../../../lifemate/core/config.py) › `load_config` | đường dẫn file config TOML → đối tượng config | Đọc config; khoá `asr.initial_prompt_file` được thay bằng nội dung file câu mồi |
| Khoá config `asr.prompt_guard_ngram` | số nguyên, 0 là tắt | Độ dài cụm từ n của lớp bảo vệ |
| Khoá config `asr.prompt_guard_max_compression` | số thực, mặc định 2,4 | Ngưỡng tỉ lệ nén; vượt là câu lặp |
| [`lifemate/backends/asr_guard.py`](../../../lifemate/backends/asr_guard.py) › `needs_retry` | chữ trước lọc, chữ sau lọc, câu mồi, n, ngưỡng → đúng/sai | Bốn dấu hiệu câu hỏng (mục 4.3.2) |
| [`lifemate/backends/asr_guard.py`](../../../lifemate/backends/asr_guard.py) › `echoes_prompt` | chữ, câu mồi, n → đúng/sai | Chữ có cụm n từ liên tiếp trùng câu mồi không |
| [`lifemate/backends/asr_guard.py`](../../../lifemate/backends/asr_guard.py) › `compression_ratio` | chữ → số thực | Độ dài chữ chia độ dài sau khi nén zlib |
| [`lifemate/backends/asr.py`](../../../lifemate/backends/asr.py) › `ASRTranscriber._guard` | các câu vừa chép của một mẫu | Chép lại không câu mồi các câu `needs_retry` báo hỏng |
| [`lifemate/backends/asr.py`](../../../lifemate/backends/asr.py) › `transcribe_batch_unprompted` | danh sách file âm thanh → danh sách chữ | Chép không câu mồi, có ở `FasterWhisperASR` và `HFSeq2SeqASR` |

### A.4. Dò cấu hình: [`lifemate/eval/asr_tune.py`](../../../lifemate/eval/asr_tune.py)

| Tên | Loại, nội dung | Việc |
|---|---|---|
| `MODELS` | từ điển: tên mô hình → engine, tên mô hình trên Hugging Face, phiên bản | Các mô hình được so |
| `BASE_GENERATE_KWARGS` | từ điển tham số giải mã | Cấu hình xuất phát chung: tham lam, temperature 0, không câu mồi |
| `STAGES` | danh sách (tên bước, từ điển tên lựa chọn → khoá config ghi đè) | Các bước dò theo thứ tự |
| `PROMPT_FILES`, `GUARDS`, `prompt_options` | xem mục 4.3.3 | Sinh 13 lựa chọn của bước câu mồi |
| `objective` | báo cáo một lần đo → bộ ba số | Hàm mục tiêu (mục 4.3.3) |
| `tune` | mô hình, tập dò… → từ điển gồm cấu hình thắng, lựa chọn từng bước, mọi lần đo | Dò lần lượt từng bước |
| `report` | mô hình, cấu hình thắng, các tập kiểm tra → báo cáo từng tập | Đo cấu hình thắng đúng một lần trên mỗi tập kiểm tra |
