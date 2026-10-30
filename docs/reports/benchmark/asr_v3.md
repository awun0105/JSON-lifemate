# Báo cáo benchmark ASR lần 3: câu mồi theo mẫu, chọn cách quyết định ghép, và đo điểm thi

Ngày lập kế hoạch: 29/10/2026. Chạy: 30/10/2026 trên Colab T4. Trạng thái: **đã xong E, F, G, H, bảng chọn chung (mục 7)
và I1; thử nghiệm J (gồm I2) đang chạy dở** (phiên K4 dừng vì Colab tạm không cấp GPU). Các ô của J còn ghi "chưa đo".

Báo cáo trả lời bốn câu hỏi:

1. Câu mồi (`initial_prompt`) dựng riêng cho từng mẫu, từ ngữ cảnh có sẵn dạng chữ lúc thi (lượt trợ lý liền trước, app
đã cài), có giúp whisper-large-v3 chép đúng thêm những từ tiếng Anh **không có** trong câu mồi cố định của lần 2
không? (thử nghiệm E)
2. Ở mỗi đoạn hai bản viết khác nhau, cách quyết định nào tốt nhất: luật của lần 2, lọc theo độ tin cậy từng từ của
câu mồi (thử nghiệm F), cho mô hình chấm điểm hai phương án (thử nghiệm G), hay bộ phân loại học từ dữ liệu (thử nghiệm
H)? Mọi cách được so trong một bảng chung và chọn một lần (mục 7).
3. Nếu không chốt một bản chữ mà giữ cả hai phương án ở chỗ hai mô hình bất đồng, thì từ đúng nằm trong một trong hai
phương án ở bao nhiêu phần trăm số trường hợp, và tốn thêm bao nhiêu token cho LLM? (thử nghiệm I)
4. Chữ ASR tốt hơn làm điểm thi tăng bao nhiêu? Chạy agent trên chữ ASR đã lưu của từng cấu hình, chấm điểm như thi.
(thử nghiệm J)

Mọi lần chạy đều đo kèm thời gian và bộ nhớ GPU của phần ASR (mục 2.4), nên mỗi thử nghiệm báo được cái giá về thời gian
của lựa chọn đó. Số đo trên T4 của Colab; thời gian trên A30 của máy chấm đo ở lần chạy hệ thống với LLM sau lần 3.

Phạm vi: chọn cấu hình ASR bằng chất lượng chữ (E, F, G, H, I), rồi đo điểm thi của các cấu hình đó bằng cách chạy agent
trên chữ ASR đã lưu (J). Không đo thời gian của cả hệ thống; bước sau lần 3 mới chạy cả hệ thống (ASR và LLM cùng lúc)
trên GPU giống máy chấm để đo thời gian thật (mục "Sau lần 3").

Tên thử nghiệm nối tiếp lần 2: lần 2 dùng A, B, C, D (`docs/reports/benchmark/asr_v2.md`), lần 3 dùng E, F, G, H, I, J theo thứ tự
các bước trong pipeline. Các phiên GPU tên K1–K4; theo kế hoạch chạy trên Kaggle, thực tế chạy trên Colab T4 vì hạn mức
GPU tuần của Kaggle đã hết (Phụ lục B.2).

## Tóm tắt cho cả team

Phần này đọc riêng được. Thuật ngữ được giải thích ngay tại chỗ; định nghĩa đầy đủ ở mục 0.3.

### Pipeline của hệ thống

Mỗi mẫu thi gồm file âm thanh của lượt cuối, file âm thanh các lượt hội thoại trước đó, danh sách app đã cài, catalog tool của các app  
đó và ngữ cảnh người dùng. Lượt người dùng trong lịch sử chỉ có âm thanh, nên cũng phải chép. Giới hạn: 30 giây mỗi mẫu,  
tổng tham số không quá 6B.

```
âm thanh (lượt cuối + lượt người dùng trong lịch sử, tới 13 file)
  → ASR: ChunkFormer RNNT chép chính; whisper-large-v3 có câu mồi chép thêm
         → bộ lọc ảo giác → lớp bảo vệ câu mồi g4 → ghép hai bản chữ
  → PromptBuilder: ghép chữ ASR với lịch sử, danh sách app, catalog tool, ngữ cảnh
  → LLM (Qwen3-4B, chạy bằng vLLM): chọn tool và tham số
  → lời gọi tool
```

**Phần ASR và chỗ của từng thử nghiệm lần 3** (mỗi câu người dùng đi qua các bước sau):

```
1. âm thanh của câu
   ├─ ChunkFormer RNNT chép               → bản A (giỏi tiếng Việt)
   └─ whisper-large-v3 chép, có câu mồi   → bản B (giỏi tiếng Anh hơn)
         câu mồi: cố định en_full_v1, hay thêm lời trợ lý / tên app?      ← E
         → bộ lọc câu bịa ("hãy subscribe kênh…") → lớp bảo vệ g4
2. đặt A và B cạnh nhau, tìm các đoạn hai bản viết khác nhau
3. ở mỗi đoạn mà B có từ tiếng Anh mới: lấy chữ A hay chữ B?
         C0 luật của lần 2 | F độ tin cậy từng từ | G điểm âm thanh | H bộ phân loại
         → so trong bảng chọn chung, chọn một cách                       ← F, G, H, mục 7
4. (tuỳ chọn) đoạn không chắc: giữ cả hai phương án {A | B} cho LLM     ← I
5. chữ cuối → PromptBuilder → LLM → điểm thi                    ← J (chạy agent trên chữ đã lưu)
mọi bước đều được đo thời gian, mọi phiên đều đo bộ nhớ GPU (mục 2.4)
```

Phần ASR trên là cấu hình thắng của benchmark lần 2, chưa đưa vào `configs/v0.toml` (file này vẫn dùng large-v3 đứng
riêng). Điểm thi chấm lời gọi tool, không chấm chữ, nhưng chữ ASR sai từ khoá thì LLM không sửa được. Lần 3 chọn cấu hình ASR
bằng chất lượng chữ, rồi đo điểm thi của các cấu hình đó ở thử nghiệm J.

### Lần 1 và lần 2 đã làm gì

**Lần 1** (`asr_v1.md`) so năm mô hình ASR trên mọi câu, cùng công sức dò cấu hình. **ChunkFormer RNNT** thắng: WER khoảng
20% trên cả hai tập kiểm tra, so với khoảng 30% của whisper-large-v3; không có câu bịa; p95 2,5 s trên T4. Lần 1 để lại
bốn điểm: chưa đo riêng từ tiếng Anh; mọi câu mồi đều làm large-v3 kém đi (in lại câu mồi, đếm tiếp danh sách, bịa câu
phụ đề); cấu hình large-v3 không tất định; mới chọn một mô hình, chưa thử kết hợp.

**Lần 2** (`asr_v2.md`) giải quyết bốn điểm đó bằng bốn thử nghiệm:


| Thử nghiệm lần 2                 | Kết quả                                                                                                                                                                                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A. Chấm riêng câu chêm tiếng Anh | RNNT vẫn là mô hình đơn tốt nhất, nhưng chỉ chép đúng khoảng một nửa số từ tiếng Anh (49,9% / 51,6%). large-v3 không câu mồi đúng nhiều hơn một chút (51,6% / 56,0%) nhưng phần tiếng Việt sai nhiều hơn 8–12 điểm                                                        |
| B. Câu mồi mới + lớp bảo vệ      | `en_full_v1` + g4 nâng tỉ lệ chép đúng từ tiếng Anh của large-v3 từ 52% lên 72% (seen) và từ 56% lên 61% (unseen). Lớp bảo vệ chặn được lỗi in lại câu mồi, lặp vô hạn và bịa câu: lỗi câu thuần Việt về ngang mức không câu mồi. g4 thắng g3 vì g3 bắt nhầm câu nói đúng |
| C. Kiểm chứng tất định           | Chạy lại RNNT (2 467 câu) và large-v3 tham lam + VAD (1 024 câu) trên máy khác cho chữ giống hệt từng ký tự                                                                                                                                                               |
| D. Ghép hai mô hình              | Ghép RNNT + large-v3 có câu mồi, có bộ lọc ngữ cảnh: chép đúng từ tiếng Anh 72,2% / 63,1%, cao hơn RNNT đứng riêng 22,3 và 11,5 điểm; WER mọi câu 19,4% / 19,5%, thấp hơn RNNT                                                                         |


### Vì sao có lần 3

Lần 2 tìm được cấu hình ASR tốt nhất hiện có cho câu tiếng Việt chêm tiếng Anh: **ghép** chữ của ChunkFormer RNNT (giỏi
tiếng Việt) với chữ của whisper-large-v3 có câu mồi `en_full_v1` (giỏi tiếng Anh hơn), kèm bộ lọc ngữ cảnh. Lần 2 còn để lại bốn việc. Mỗi việc dẫn tới một cải tiến của lần 3:


| Việc còn lại từ lần 2                                                                             | Số đo của lần 2                                                                                                                        | Thử nghiệm lần 3 |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Câu mồi cố định chỉ giúp các từ **có trong** câu mồi. App lạ dùng nhiều từ không có trong câu mồi | Từ có trong câu mồi: chép đúng 48% → 84% (seen). Từ không có trong câu mồi gần như không đổi. Trên unseen, large-v3 chỉ tăng 56% → 61% | E                |
| Bản ghép vẫn làm câu thuần Việt sai nhiều hơn RNNT                                                | +0,4 điểm WER (seen), +0,3 (unseen). Trên tập dò, 24 trong 30 từ bị lấy nhầm là từ của câu mồi ("Mail" cho "mai")                      | F, G, H (so trong bảng chọn chung, mục 7) |
| Quy tắc ghép viết tay, nhìn chữ chứ không nghe âm thanh, chỉ dùng được một tín hiệu (từ có trong ngữ cảnh không) | Khi thêm bộ lọc, chép đúng từ tiếng Anh trên tập dò giảm 3,3 điểm; điều kiện độ dài chỉ chặn 1 trong 518 đoạn | G (điểm âm thanh), H (bộ phân loại) |
| Chưa đo thời gian khi hai mô hình, lớp bảo vệ và phép ghép chạy chung một tiến trình | Trên T4: RNNT p95 2,5 s, large-v3 p95 8,6–8,8 s mỗi mẫu, chạy riêng từng mô hình | Đo kèm mọi lần chạy (mục 2.4) |


Ngoài ra, lần 2 luôn chốt **một** bản chữ trước khi đưa cho LLM. Ở chỗ hai mô hình bất đồng mà quy tắc không đủ bằng
chứng, một trong hai phương án bị bỏ, dù LLM có thể chọn đúng nhờ danh sách app và kết quả tool. Thử nghiệm I đo xem giữ
cả hai phương án thì được bao nhiêu (mục 8). Cuối cùng, mọi lần đo trước chỉ chấm chữ; thử nghiệm J đo điểm thi của các
cấu hình ASR bằng cách chạy agent trên chữ đã lưu (mục 9).

### Ý tưởng chung: dùng thông tin mà từng mẫu có sẵn dạng chữ

Lúc thi, mỗi mẫu đi kèm một số thông tin dạng chữ: danh sách app đã cài và catalog tool của các app đó, ngữ cảnh người
dùng (giờ, vị trí, hồ sơ), và các lượt trợ lý trong lịch sử (câu trả lời, lời gọi tool, kết quả tool). Lượt người dùng
trong lịch sử **chỉ có âm thanh**, nên phải tự chép. Các thông tin dạng chữ cho biết người dùng **có khả năng** nói từ
nào. Ví dụ, trợ lý vừa hỏi "Gửi cho Sarah hay Sara?" thì câu người dùng trả lời ngay sau dễ chứa đúng tên đó.

Lần 2 mới dùng thông tin này ở **sau** khi chép, để lọc khi ghép. Lần 3 dùng thông tin đó ở ba chỗ:

1. **Trước khi chép** (E): đưa vào câu mồi của large-v3, để mô hình viết đúng các từ đó ngay từ đầu.
2. **Khi ghép** (F, H): đo độ tin cậy của từng từ, và dùng nhiều tín hiệu cùng lúc thay cho một luật. G thêm một tín
   hiệu không đến từ chữ: điểm âm thanh của hai phương án.
3. **Khi đưa cho LLM** (I): giữ cả hai phương án ở chỗ chưa chắc, để LLM chọn bằng ngữ cảnh mà LLM thấy (kết quả tool, danh sách app).

### Dữ liệu cần chuẩn bị và cách chuẩn bị

Lần 3 dùng **đúng cách chia của lần 1 và lần 2**, không chia lại. Khác biệt duy nhất: 793 mẫu train còn trống (lần 2
chỉ dùng để chọn từ cho câu mồi) nay là **tập học** cho thử nghiệm F, G và H. `public_test` không dùng ở lần 3.

**Cách chia.** Mọi bước chia do một script làm: `lifemate/eval/make_splits.py`, chạy bằng `make splits` (tương đương
`python -m lifemate.eval.make_splits --smoke 30 --asr-tune 300 --en-mixed`), seed cố định 42 nên chạy ở máy nào cũng
ra giống nhau. Script không sao chép dữ liệu: script chỉ ghi các file `configs/splits/*.txt`, mỗi dòng một mã mẫu. Lúc
chạy thử nghiệm, `lifemate/io/data.py › read_ids` (file `.txt` → danh sách mã) và `load_split` (thư mục dữ liệu, tên
split, danh sách mã → các mẫu) lấy đúng các mẫu đó từ `train/train.jsonl` của BTC. Các bước chạy tuần tự, bước sau dùng
kết quả bước trước. Sơ đồ dưới ghi cả lần benchmark và thử nghiệm dùng từng tập:

```
train.jsonl (1 545 mẫu, dữ liệu gốc của BTC)
 │
 ├─1. Tách app lạ ──────────────► val_unseen.txt (178)
 │                                  · lần 1: đo cấu hình thắng của 5 mô hình, một lần
 │                                  · lần 2: đo cách câu mồi thắng và cách ghép, một lần
 │                                  · lần 3: đo E2 + ghép C0, một lần; chạy agent (J0–J4)
 │
 └─ còn 1 367 mẫu
     │
     2. Gom câu gần trùng thành nhóm (không ghi file riêng; số liệu ghi trong meta.json)
     │
     3. Xếp nguyên nhóm vào 5 phần ──► fold_0..4.txt
     │                               ├─ phần 0 ──► val_seen.txt (274)
     │                               │              · lần 1, 2, 3: như val_unseen
     │                               └─ phần 1–4 ► train.txt (1 093)
     │                                              · không dùng trực tiếp; được chia tiếp ở bước 5
     │
     4. Rút từ val_seen ─────────────► smoke.txt (30)
     │                                  · chạy thử nhanh để bắt lỗi, không dùng để chấm
     │
     5. Rút từ train ───────────────► asr_tune.txt (300) = tập dò
     │                               │  · lần 1: dò cấu hình từng mô hình
     │                               │  · lần 2: thử 13 cách dùng câu mồi, chọn cách ghép
     │                               │  · lần 3: chọn câu mồi (E); bảng chọn chung C0/F/G/H
     │                               │          (mục 7); mức trần hai phương án (I1)
     │                               └─ train bỏ asr_tune ► asr_learn.txt (793) = tập học
     │                                  · lần 2: đếm từ tiếng Anh large-v3 hay sai để viết
     │                                          câu mồi en_full_v1 (lúc đó chưa ghi thành file)
     │                                  · lần 3: đếm độ tin cậy, chọn θ (F); điểm âm thanh (G);
     │                                          học bộ phân loại (H)
     │
     6. Lọc mẫu có câu chêm tiếng Anh, làm riêng trên từng tập
     │  (đọc chữ đáp án của BTC, ở câu cuối và các câu người dùng trong lịch sử):
     │
     │    asr_tune.txt   (300) ──lọc──► asr_tune_en_mixed.txt   (165)
     │    val_seen.txt   (274) ──lọc──► val_seen_en_mixed.txt   (152)
     │    val_unseen.txt (178) ──lọc──► val_unseen_en_mixed.txt  (95)
     │
     │                                  · lần 2, 3: chấm riêng tỉ lệ chép đúng từ tiếng Anh
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
   (64 nhóm, 174 mẫu; các mẫu khác không thuộc nhóm nào). Đây là bước trung gian cho bước 3 nên không ghi file danh sách.
3. **Chia 5 phần** (`stratified_group_folds`): xếp nguyên cả nhóm vào một phần, sao cho các phần có cùng tỉ lệ số lời
   gọi tool và tỉ lệ câu hỏi lại. Nhờ vậy hai câu gần giống nhau không nằm ở hai tập khác nhau.
4. **`smoke`** (`--smoke 30`): 30 mẫu của `val_seen`, phân tầng theo loại câu trả lời và có/không lịch sử.
5. **`asr_tune`** (`--asr-tune 300`): 300 mẫu của `train`, cùng cách phân tầng; phần `train` còn lại ghi ra
   `asr_learn.txt`.
6. **`*_en_mixed`** (`--en-mixed`, hàm `en_mixed_ids`): giữ mẫu có ít nhất một câu người dùng chêm từ tiếng Anh, theo
   `lifemate/eval/foreign_words.py › has_foreign_word` (câu → có từ ngoại hay không). Luật nhận từ ngoại: từ không dấu mà
   không ghép được thành âm tiết tiếng Việt, từ viết hoa toàn bộ, hoặc có chữ f, j, w, z; kèm hai danh sách viết sẵn cho
   trường hợp khó (từ mượn viết kiểu Việt như "tivi"; từ tiếng Anh trông giống âm tiết Việt như "chat"). Việc lọc do
   script làm; luật đã được rà tay ở lần 2. Mỗi file `_en_mixed` nằm trọn trong tập gốc tương ứng.
7. **`meta.json`**: số mẫu từng tập, tỉ lệ nhãn từng phần, số nhóm gần trùng, và số mẫu trùng giữa `train` với hai tập
   kiểm tra (đều bằng 0).

Nguyên tắc chung: tập học dùng để học tham số, tập dò dùng để chọn giữa các cách, tập kiểm tra chỉ dùng để đo cách đã
chọn, một lần.


| Phần dữ liệu                             | Số mẫu (số câu) | Dùng để                                                           | Vì sao tách riêng                                                                              |
| ---------------------------------------- | --------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Tập học (793 mẫu train ngoài `asr_tune`) | 793 (2 569)     | Đếm độ tin cậy từng từ và chọn θ (F); điểm âm thanh làm dữ liệu học (G); học bộ phân loại (H) | Học trên tập dò hay tập kiểm tra thì quy tắc "biết trước đề" |
| Tập dò `asr_tune`                        | 300 (1 024)     | Chọn biến thể câu mồi (E); chọn cách quyết định ghép (mục 7)      | Chọn trên tập nào thì số của tập đó đẹp hơn thực tế, nên cần tập khác để đo                    |
| Tập kiểm tra `val_seen`                  | 274 (916)       | Đo kết quả cuối một lần; chạy agent (J)                           | Chưa dùng để chọn gì ở lần 3                                                                   |
| Tập kiểm tra `val_unseen`                | 178 (527)       | Đo kết quả cuối một lần, trên app chưa gặp; chạy agent (J)        | Đây là chỗ câu mồi cố định của lần 2 giúp ít nhất                                              |


**Cách chuẩn bị.**

1. Đặt gói dữ liệu của BTC vào `data/train+public_test/` và chạy `make splits` như lần 2. Danh sách 793 mẫu tập học là
`train.txt` bỏ đi `asr_tune.txt`; lệnh ghi thêm file `configs/splits/asr_learn.txt` (việc chuẩn bị P1, Phụ lục B).
2. Chữ ASR của lần 2 dùng lại được: RNNT trên `asr_tune` và hai tập val, large-v3 có và không câu mồi trên `asr_tune`,
large-v3 `en_full_v1` + g4 trên hai tập val.
3. Phải chạy GPU để có thêm: large-v3 với các biến thể câu mồi mới trên `asr_tune`; RNNT và large-v3 (câu mồi thắng) trên
tập học; large-v3 câu mồi thắng trên hai tập val; điểm âm thanh cho G; agent trên chữ đã lưu cho J. Lịch các phiên GPU ở
Phụ lục B.

### Các thử nghiệm


| Thử nghiệm             | Câu hỏi                                                                                                          | Cách làm ngắn gọn                                                                                                                        | Dữ liệu                                                                                                      | Cần GPU?                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------- |
| E. Câu mồi theo mẫu    | Câu mồi dựng từ ngữ cảnh của mẫu có giúp từ tiếng Anh không có trong câu mồi cố định không? | Hai biến thể: thêm câu trả lời của trợ lý liền trước; thêm cả tên app đã cài | Chọn trên `asr_tune`, đo trên val | Có |
| F. Độ tin cậy từng từ  | Lọc từ hay bị viết nhầm của câu mồi có giảm lỗi ở câu thuần Việt không? | Đếm trên tập học: mỗi lần large-v3 viết ra từ đó, có bao nhiêu lần người dùng thật sự nói. Từ hay sai thì chỉ nhận khi có trong ngữ cảnh. Không cần mô hình lúc chạy, nên cũng là phương án dự phòng | Đếm và chọn ngưỡng trên tập học (kiểm định chéo), vào bảng chọn chung (mục 7) | Có (để có chữ trên tập học) |
| G. Chấm điểm hai phương án | Cho large-v3 chấm xác suất của hai phương án theo âm thanh có quyết định ghép tốt hơn luật không? | Dựng hai câu chỉ khác nhau ở đoạn cần quyết định; large-v3 không câu mồi chấm từng câu bằng ép chuỗi; chọn câu điểm cao hơn. Không có tham số | Vào bảng chọn chung (mục 7) | Có (phiên K3) |
| H. Bộ phân loại ghép   | Mô hình nhỏ học từ dữ liệu, dùng mọi tín hiệu, có quyết định ghép tốt hơn không? | Hồi quy logistic, mỗi đoạn khác nhau là một mẫu học, đầu vào là các tín hiệu của E, F, G và điểm tin cậy của large-v3 | Học trên tập học, vào bảng chọn chung (mục 7) | Có (dùng điểm của K3) |
| I. Giữ hai phương án   | Ở đoạn không chắc, đưa cả hai phương án `{A \| B}` cho LLM chọn có lợi không? | I1: dùng đáp án đếm trần, tức số đoạn cứu được nếu LLM luôn chọn đúng, và chi phí token. I2: nếu trần đủ cao, chạy LLM thật (một nguồn của J) | `asr_tune`, val | I1 không; I2 có (phiên K4) |
| J. Agent trên chữ đã lưu | Chữ ASR tốt hơn làm điểm thi tăng bao nhiêu? | Chạy agent (Qwen3-4B, vLLM, `ToolEnv`) với chữ đọc lại từ `hyps.jsonl`: chữ chuẩn, RNNT, ghép lần 2, ghép lần 3, ghép lần 3 có hai phương án | val | Có (phiên K4) |


**Thứ tự quyết định.** E chọn câu mồi trước. Sau đó F, G, H và luật của lần 2 (C0) cùng vào **một bảng chọn
chung** (mục 7): mọi tham số học trên tập học, tập dò chỉ dùng một lần để chọn ứng viên thắng, rồi đo ứng viên thắng một
lần trên val. I dùng kết quả của bước chọn chung. J chạy cuối, trên chữ của mọi cấu hình đã chốt.

### Cách chọn và tiêu chí đặt trước

Giống lần 2: mọi lựa chọn làm trên tập dò, cấu hình thắng đo **đúng một lần** trên hai tập kiểm tra. Tiêu chí của từng
thử nghiệm được ghi ở đây **trước khi đo**, để không chỉnh tiêu chí theo số:


| Thử nghiệm | Tiêu chí chọn (đo trên tập dò)                                                                                                                                                                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E          | Chọn biến thể có tỉ lệ chép đúng từ tiếng Anh **của bản ghép** cao nhất, với hai điều kiện: WER câu thuần Việt của bản ghép không tăng so với bản ghép dùng `en_full_v1` (CI 95% của chênh lệch không nằm hẳn trên 0), và p95 thời gian của large-v3 tăng không quá 1 giây |
| Quyết định ghép (C0, F, G, H) | Loại ứng viên tệ hơn C0 có ý nghĩa thống kê ở chép đúng từ tiếng Anh hoặc ở WER câu thuần Việt, hoặc tốn thêm quá 1 giây ở p95 mỗi mẫu trên T4. Trong số còn lại, chọn ứng viên có độ chính xác quyết định đoạn cao nhất; nếu không hơn có ý nghĩa thống kê một ứng viên đơn giản hơn thì chọn ứng viên đơn giản hơn (thứ tự C0, F, G, H). Chi tiết ở mục 7 |
| I          | I1 không chọn gì. Chạy I2 (nguồn J4 của J) khi trần trên `asr_tune` cứu thêm được từ 3 điểm phần trăm chép đúng từ tiếng Anh trở lên |
| J          | Không chọn gì trong lần 3. Bản ghép lần 3 vào `configs/v0.toml` nếu điểm thi không thấp hơn RNNT có ý nghĩa thống kê trên cả hai tập val; định dạng hai phương án vào dữ liệu fine-tune nếu J4 hơn J3 có ý nghĩa thống kê trên ít nhất một tập val và không thua trên tập còn lại |



### Kết quả chính

Cấu hình ASR thắng của lần 3: **RNNT + large-v3 với câu mồi E2** (câu mồi cố định, tên app đã cài, lời trợ lý liền trước)
**+ luật ghép `context_other` của lần 2** (C0 thắng bảng chọn chung). Số đo một lần trên hai tập kiểm tra:

| Hệ thống | Chép đúng từ tiếng Anh (seen / unseen) | WER câu chêm (seen / unseen) | WER câu thuần Việt (seen / unseen) | WER mọi câu (seen / unseen) |
|---|---|---|---|---|
| RNNT đứng riêng | 49,9% / 51,6% | 24,8% / 24,1% | 17,7% / 17,5% | 20,0% / 19,8% |
| Ghép, `en_full_v1`, `context_other` (lần 2) | 72,2% / 63,1% | 22,1% / 22,8% | 18,1% / 17,8% | 19,4% / 19,5% |
| **Ghép, câu mồi E2, `context_other` (lần 3)** | **75,5% / 68,7%** | **21,7% / 22,0%** | **17,9% / 17,6%** | **19,1% / 19,2%** |

So với lần 2: chép đúng từ tiếng Anh +3,3 điểm (`val_seen`, CI [0,0; +6,6]) và +5,6 điểm (`val_unseen`, CI [+2,2;
+9,8], có ý nghĩa thống kê); WER mọi câu giảm 0,3 điểm ở cả hai tập (có ý nghĩa thống kê). Bảng chọn chung (mục 7): bộ
phân loại H tốt nhất trên tập dò (độ chính xác quyết định đoạn 76,4% so với 70,7% của C0) nhưng chưa hơn C0 có ý nghĩa
thống kê, nên giữ C0.

| Thời gian và bộ nhớ trên Colab T4 (mỗi mẫu, seen / unseen) | p50 (s) | p95 (s) | Tối đa (s) | Bộ nhớ GPU đỉnh |
|---|---|---|---|---|
| RNNT đứng riêng | 1,51 / 1,43 | 3,57 / 3,64 | 5,15 / 5,60 | không đo riêng |
| large-v3 với câu mồi E2 (gồm lớp bảo vệ) | 2,09 / 1,98 | 7,60 / 7,35 | 9,21 / 9,29 | |
| **Cả phần ASR (RNNT + large-v3 + ghép)** | **3,51 / 3,41** | **11,27 / 11,06** | **13,79 / 14,54** | **3,6 GB** (hai mô hình) |
| Nếu thêm chấm điểm âm thanh (G, H) | +0,0 | +0,8 | +1,9 | 3,4 GB khi chấm |

Bước ghép tốn dưới 0,01 giây mỗi mẫu. Số đo trên T4 và không có vLLM chạy cùng (mục 2.4).

| Điểm thi khi chạy agent trên chữ đã lưu (thử nghiệm J) | val\_seen | val\_unseen |
| ------------------------------------------------------- | --------- | ----------- |
| J0 chữ chuẩn của BTC                                    | chưa đo   | chưa đo     |
| J1 RNNT đứng riêng                                      | chưa đo   | chưa đo     |
| J2 ghép lần 2                                           | chưa đo   | chưa đo     |
| J3 ghép lần 3                                           | chưa đo   | chưa đo     |
| J4 ghép lần 3, hai phương án (I2; I1 đạt nên chạy)      | chưa đo   | chưa đo     |


**Ví dụ thật** (tập dò; chi tiết và thêm ví dụ ở mục 3.7, 4.7, 5.7, 6.7, 8.7):

- **Câu mồi tốt nhất** gồm ba phần. Ví dụ ở mẫu `train_1650`, lượt 16: (1) câu mồi cố định "QR trên app mờ quá. Email
  của Grace Nakamura tới… Voucher nằm trong SMS, check kỹ rồi gửi Zalo cả xóm, coi miếng SJC luôn."; (2) "Ứng dụng đã
  cài: Nhắc việc, Ghi chú, Tỷ giá, Lịch, Họp trực tuyến, Tin nhắn SMS, Đặt đồ ăn, Danh bạ, Vé máy bay."; (3) lời trợ lý
  liền trước "Dạ mai có một cái nhắc: gửi slide cho Sếp David lúc tám giờ sáng anh." Người dùng nói "Thôi bỏ cái đó đi,
  anh gửi slide luôn rồi." Câu mồi cố định cho "…ăn gửi **Zalo** ngon"; câu mồi ba phần cho "…anh gửi **slide** luôn rồi".
- **Lời trợ lý giúp nghe đúng tên lạ:** trợ lý vừa liệt kê chuyến bay có "JetXanh"; người dùng nói "Lấy chuyến JetXanh
  chín giờ sáng". Câu mồi cố định cho "Ngày chuyến sẽ sẵn 9h sáng", câu mồi ba phần cho "…**JetXanh** chín giờ sáng".
- **Tên app giúp nghe đúng từ tiếng Anh:** app "Podcast & Radio" đã cài; "Mở podcast Nhà giả kim…" được viết "Mà ở
  **podcast** nhà giả kim…" thay cho "Mở bóc khác…".
- **Luật ghép vẫn còn lấy nhầm từ câu mồi:** "Mai anh có hẹn gì trong lịch không?" được large-v3 viết "…trong **list**
  không?"; luật C0 lấy "list" vì đó là từ của câu mồi. Chấm bằng âm thanh giữ đúng "lịch", nhưng ở câu khác lại lấy nhầm
  "Line" cho "Lan".
- **Hai phương án cho LLM:** "Nina Kowalski ở khách sạn…" thành "nina {ko washi | Kowalski} ở khách sạn…" (LLM chọn đúng thì
  sửa được); "Bật đèn bếp lên giúp tôi." thành "{bật đèn bếp | Bạn điền web} lên giúp tôi" (bản chốt đã đúng, LLM phải bỏ
  qua phương án sai).

### Kết luận

Phần ASR (E, F, G, H, bảng chọn chung, I1) đã có kết luận; phần điểm thi (J) chưa đo xong.

1. **Dùng câu mồi E2 cho large-v3** (`en_full_v1`, rồi tên tiếng Việt của các app đã cài, rồi lời trợ lý liền trước; lớp
   bảo vệ chỉ so phần cố định). Bản ghép chép đúng thêm 3,3 điểm (`val_seen`, CI chạm 0) và 5,6 điểm (`val_unseen`, có ý
   nghĩa thống kê) so với lần 2, WER mọi câu giảm 0,3 điểm, và chỉ tốn thêm khoảng 0,2 giây ở p95 (mục 3).
2. **Giữ luật ghép C0 (`context_other`) của lần 2.** F, G và H đều không hơn C0 có ý nghĩa thống kê trên tập dò. H tốt
   nhất (76,4% so với 70,7%, CI [0,0; +11,7]) và là hướng nên thử lại khi có thêm dữ liệu có nhãn (mục 7).
3. **Toàn bộ phần ASR mất p95 khoảng 11,3 giây mỗi mẫu trên T4**, tối đa 14,5 giây, bộ nhớ GPU đỉnh 3,6 GB. Còn khoảng
   15 giây cho LLM trong ngân sách 30 giây; thời gian thật trên A30 khi chạy cạnh vLLM chưa đo.
4. **Định dạng hai phương án có trần +4,8 điểm** trên tập dò (+2,4 / +4,0 trên val), vượt ngưỡng 3 điểm, nên được chạy
   thật với LLM trong J (nguồn J4). Số đoạn "rủi ro" gấp đôi số đoạn "cứu được", nên chưa biết LLM thật có được lợi không
   (mục 8).
5. **Chưa kết luận được bản ghép có làm điểm thi tăng không**: cần kết quả của J (mục 9).

### Kế hoạch thực hiện và tiến độ

Thứ tự bắt buộc theo phụ thuộc: E phải xong trước F, G, H, vì cả ba cần chữ large-v3 với câu mồi thắng trên tập học. G cần phiên K3 sau K2; H dùng điểm của G làm đặc trưng. Bước chọn chung (mục 7) chạy khi đã có đủ bốn ứng viên. I1 chạy offline sau bước chọn chung. J chạy sau I1.


| Bước    | Việc                                                                                                    | Ở đâu                                                            | Phụ thuộc | Trạng thái |
| ------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------- | ---------- |
| P1–P8   | Code chuẩn bị chung và cho E (Phụ lục B.1), test                                                        | Máy dev                                                          |           | Xong, trừ P8 và `ASRConfig.prompt_mode` (Phụ lục A.2) |
| K1      | Phiên GPU 1: ba biến thể câu mồi (E0, E1, E2) trên `asr_tune`                                           | Colab T4, hơn 1 giờ (3 máy nối tiếp)                             | P1–P8     | Xong       |
| P9–P15  | Code cho F, G, H, bảng chọn chung, I, J và bước kiểm tra cuối (Phụ lục B.1); viết trong lúc chờ K1, K2 | Máy dev | P1–P8 | Xong (code gộp vào `asr_v3_analysis.py`, `asr_v3_report.py`, `asr_v3_agent.py`) |
| E       | Chấm offline, chọn biến thể thắng                                                                       | Máy dev                                                          | K1        | Xong: E2   |
| K2      | Phiên GPU 2: tập học (RNNT, large-v3 biến thể thắng); val (large-v3 biến thể thắng)                     | Colab T4, khoảng 1 giờ 40 phút (2 máy)                           | E         | Xong       |
| K3      | Phiên GPU 3: chấm điểm âm thanh (large-v3 không câu mồi) trên tập học, `asr_tune`, val | Colab T4, khoảng 6 phút | K2, P10 | Xong |
| F, G, H | Đếm độ tin cậy và chọn θ, học H (chỉ trên tập học); bảng chọn chung trên `asr_tune`; đo ứng viên thắng một lần trên val (mục 7) | Máy dev | K2, K3, P9, P11, P12 | Xong: C0 thắng |
| I1      | Đếm trần của hai phương án                                                                      | Máy dev                                                          | F, G, H, P13 | Xong: trần +4,8, chạy I2 |
| K4      | Phiên GPU 4: agent trên chữ đã lưu, J0–J4 trên val | Colab T4, ước khoảng 5 giờ GPU (13,3 giây mỗi mẫu, 1 323 lượt chạy mẫu) | I1, P14 | Đang chạy dở: J1 `val_seen` 11/274 mẫu; dừng vì Colab tạm không cấp GPU |
| J       | Chấm điểm, so các nguồn, xếp loại lỗi | Máy dev | K4 | Chưa làm |
| Báo cáo | Điền kết quả, kết luận; cập nhật `docs/experiments.md`, `docs/decisions.md`, SPEC                       | Máy dev                                                          | Mọi bước  | Đã điền E–I; còn J |


Thời gian thực của từng phiên ở Phụ lục B.2.

### Sau lần 3

Độ chính xác của lời gọi tool với từng nguồn chữ đã đo ở thử nghiệm J. Bước sau lần 3 chạy **cả hệ thống** (ASR và LLM cùng
lúc, cùng một GPU) với cấu hình thắng, để đo **thời gian thật** từng mẫu (ASR, LLM, tool) và bộ nhớ khi hai phần chạy
cạnh nhau. Lần chạy này cần GPU từ 24 GB, vì T4 không chứa nổi vLLM cùng whisper-large-v3 (`ops/kaggle/README.md`).

### Đọc tiếp ở đâu


| Muốn biết                                       | Đọc                  |
| ----------------------------------------------- | -------------------- |
| Thuật ngữ, dữ liệu, thước đo, cách đo thời gian | Mục 0, 2             |
| Câu mồi theo mẫu được dựng thế nào              | Mục 3 (thử nghiệm E) |
| Độ tin cậy từng từ được đếm thế nào             | Mục 4 (thử nghiệm F) |
| Chấm điểm hai phương án bằng âm thanh           | Mục 5 (thử nghiệm G) |
| Bộ phân loại ghép học gì, từ dữ liệu nào        | Mục 6 (thử nghiệm H) |
| Chọn cách quyết định ghép (bảng chọn chung)     | Mục 7                |
| Định dạng hai phương án cho LLM                 | Mục 8 (thử nghiệm I) |
| Điểm thi của từng nguồn chữ                     | Mục 9 (thử nghiệm J) |
| Code đã viết và lịch các phiên GPU              | Phụ lục A, B         |


**Kết quả thô.** Không có trên git (thư mục `artifacts/` bị git bỏ qua), nằm ở `artifacts/runs/colab/asr/10-30_v3/`:


| Thư mục | Nội dung |
| --- | --- |
| `k1/` | Phiên K1 trên `asr_tune`: `rnnt.hyps.jsonl`, `v3_E0.hyps.jsonl`, `v3_E1.hyps.jsonl`, `v3_E2.hyps.jsonl` (chữ từng câu; bản large-v3 kèm câu mồi đã dùng, số token câu mồi, có chép lại không, điểm tin cậy), `timing.jsonl` (thời gian từng mẫu), `summary.json` (thời gian nạp, kiểm tra tất định, bộ nhớ đỉnh), `gpu_memory.csv` |
| `k2_asr_learn/`, `k2_val_seen/`, `k2_val_unseen/` | Phiên K2: `rnnt.hyps.jsonl`, `v3_E2.hyps.jsonl` và các file đo như K1, trên tập học và hai tập kiểm tra |
| `k3/` | Phiên K3: `scores_<tập>.jsonl` (mỗi đoạn cần quyết định một dòng: câu A, câu B, điểm âm thanh và số token của từng câu, cờ câu dài hơn 30 giây), `.timing.jsonl`, `.summary.json`, `.gpu_memory.csv` |
| `analysis/` | Chấm offline: `e.json`, `e_val.json` (thử nghiệm E), `select.json` (F, G, H, bảng chọn chung, đo trên val, I1), chữ của các nguồn J |
| `j_sources/` | Chữ đưa cho agent ở J: `j1_rnnt_*`, `j2_round2_*`, `j3_round3_*`, `j4_alternatives_*` (`.hyps.jsonl`) và danh sách mẫu có chữ khác J1 (`*_ids_*.txt`) |
| `k4/` | Phiên K4: mỗi nguồn và mỗi tập một thư mục run agent (`config.json`, `submission.jsonl`, `trace.jsonl`; `details.jsonl` có điểm từng mẫu khi chạy xong) |


## Mục lục

- [Tóm tắt cho cả team](#tóm-tắt-cho-cả-team)
- [0. Bối cảnh và thuật ngữ](#0-bối-cảnh-và-thuật-ngữ)
- [1. Vấn đề chung và mục tiêu](#1-vấn-đề-chung-và-mục-tiêu)
- [2. Dữ liệu, thước đo và công cụ dùng chung](#2-dữ-liệu-thước-đo-và-công-cụ-dùng-chung)
- [3. Thử nghiệm E: câu mồi theo mẫu](#3-thử-nghiệm-e-câu-mồi-theo-mẫu)
- [4. Thử nghiệm F: độ tin cậy từng từ của câu mồi](#4-thử-nghiệm-f-độ-tin-cậy-từng-từ-của-câu-mồi)
- [5. Thử nghiệm G: chấm điểm hai phương án bằng âm thanh](#5-thử-nghiệm-g-chấm-điểm-hai-phương-án-bằng-âm-thanh)
- [6. Thử nghiệm H: bộ phân loại cho quyết định ghép](#6-thử-nghiệm-h-bộ-phân-loại-cho-quyết-định-ghép)
- [7. Bảng chọn chung: chọn cách quyết định ghép](#7-bảng-chọn-chung-chọn-cách-quyết-định-ghép)
- [8. Thử nghiệm I: giữ hai phương án cho LLM](#8-thử-nghiệm-i-giữ-hai-phương-án-cho-llm)
- [9. Thử nghiệm J: chạy agent trên chữ ASR đã lưu](#9-thử-nghiệm-j-chạy-agent-trên-chữ-asr-đã-lưu)
- [Giới hạn đã biết](#giới-hạn-đã-biết)
- [Phụ lục A. Bảng tra code](#phụ-lục-a-bảng-tra-code)
- [Phụ lục B. Việc chuẩn bị và lịch các phiên GPU](#phụ-lục-b-việc-chuẩn-bị-và-lịch-các-phiên-gpu)

## 0. Bối cảnh và thuật ngữ

### 0.1. Bài toán

Bài thi LifeMate nhận lời nói tiếng Việt của người dùng, có thể chêm từ tiếng Anh, và làm hai bước: **ASR** chép âm
thanh thành chữ, rồi **LLM** đọc chữ đó và gọi tool. Điểm thi chấm lời gọi tool, không chấm chữ ASR. Thể lệ
(`docs/overview.md`) đặt ba giới hạn liên quan trực tiếp tới lần 3:

- **30 giây mỗi mẫu** cho toàn bộ hệ thống (ASR, LLM, tool). Mẫu quá giờ nhận 0 điểm. Khi bằng điểm, đội có thời gian
trung bình mỗi mẫu thấp hơn xếp trên.
- **Máy chấm**: một GPU NVIDIA A30 (24 GB), CPU Intel Xeon Gold 6442Y 16 nhân, 64 GB RAM.
- **Tất định**: cố định mọi seed và giải mã tất định cho cả ASR lẫn LLM. Kết quả chạy lại khác bản đã nộp thì BTC chạy 5
lần và lấy trung bình.

Ngoài ra, hệ thống không được đọc `state.json` hay bất kỳ file nào trong thư mục môi trường của BTC; mọi dữ liệu app chỉ
lấy qua tool. Vì vậy ASR chỉ thấy được những gì có trong đầu vào của mẫu (mục 0.2).

### 0.2. Dữ liệu của một mẫu lúc thi


| Phần                          | Có dạng gì lúc thi                                              | Ví dụ                     | Lần 3 dùng ở đâu                              |
| ----------------------------- | --------------------------------------------------------------- | ------------------------- | --------------------------------------------- |
| Lượt cuối của người dùng      | Chỉ âm thanh                                                    | Câu vừa nói               | Chép bằng ASR                                 |
| Lượt người dùng trong lịch sử | Chỉ âm thanh (lúc thi chỉ có trường `audio`)                    | Các câu đã nói trước      | Chép bằng ASR                                 |
| Lượt trợ lý trong lịch sử     | Chữ: câu trả lời (`text`), lời gọi tool, kết quả tool           | "Gửi cho Sarah hay Sara?" | Câu mồi theo mẫu (E), từ điển ngữ cảnh (F, H) |
| `installed_apps`              | Danh sách mã app                                                | `zalo`, `bike_rental`     | Câu mồi theo mẫu (E), từ điển ngữ cảnh        |
| Catalog tool                  | Chữ: tên tiếng Việt của app (`name_vi`), mô tả, tool, tham số   |                           | Câu mồi theo mẫu (E2), từ điển ngữ cảnh       |
| Các trường khác của ngữ cảnh  | Chữ: giờ, múi giờ, vị trí, hồ sơ người dùng                     |                           | Từ điển ngữ cảnh                              |


Số liệu, tính trên dữ liệu của BTC:


| Tập           | Số mẫu | Số câu người dùng | Câu mỗi mẫu: trung bình / p95 / tối đa | Câu có lượt trợ lý liền trước | Giây âm thanh mỗi mẫu: trung bình / p95 / tối đa |
| ------------- | ------ | ----------------- | -------------------------------------- | ----------------------------- | ------------------------------------------------ |
| `asr_tune`    | 300    | 1 024             | 3,4 / 11 / 13                          | 724 (71%)                     | 16,2 / 42,0 / 61,1                               |
| Tập học       | 793    | 2 569             |                                        | 1 776 (69%)                   |                                                  |
| `val_seen`    | 274    | 916               | 3,3 / 10 / 13                          | 642 (70%)                     | 16,2 / 37,1 / 64,6                               |
| `val_unseen`  | 178    | 527               | 3,0 / 10 / 13                          | 349 (66%)                     | 16,4 / 38,8 / 69,4                               |


"Câu có lượt trợ lý liền trước" là câu người dùng nói ngay sau một lượt trợ lý; đây là các câu mà câu mồi theo mẫu (E1)
có thêm thông tin. Câu trả lời của trợ lý dài trung vị 14–19 từ, p95 34–43 từ, tối đa 71 từ.

### 0.3. Thuật ngữ

Các thuật ngữ đã định nghĩa ở lần 2 (WER, từ ngoại và 4 nhóm, câu chêm, câu thuần Việt, chuẩn hoá chữ, chép đúng từ ngoại, câu
mồi, lớp bảo vệ g4, tập dò, tập kiểm tra, seen, unseen, CI, bootstrap, oracle) giữ nguyên nghĩa, xem mục 0.3 của
`docs/reports/benchmark/asr_v2.md`. Thuật ngữ mới của lần 3:


| Thuật ngữ                        | Nghĩa                                                                                                                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Câu mồi cố định                  | Câu mồi giống nhau cho mọi câu, như `en_full_v1` của lần 2                                                                                                                                           |
| Câu mồi theo mẫu                 | Câu mồi dựng riêng cho từng câu, từ thông tin dạng chữ của mẫu (mục 3)                                                                                                                               |
| Từ điển ngữ cảnh                 | Tập các từ (đã chuẩn hoá) có trong thông tin dạng chữ của mẫu: catalog của app đã cài, ngữ cảnh người dùng, lượt trợ lý trong lịch sử. Có từ lần 2 (mục 6.8 của báo cáo lần 2)                             |
| `context_other`                  | Bộ lọc ngữ cảnh của lần 2: từ tiếng Anh ngoài câu mồi chỉ được lấy từ large-v3 khi có trong từ điển ngữ cảnh; từ của câu mồi nhận hết                                                           |
| Độ tin cậy của một từ câu mồi    | Trong các lần large-v3 (có câu mồi) viết ra từ đó, tỉ lệ lần người dùng thật sự nói từ đó. Đếm trên tập học (mục 4)                                                                                  |
| Đoạn khác nhau                   | Một chỗ mà chữ RNNT và chữ large-v3 viết khác nhau, nằm giữa hai từ khớp, sau khi đặt hai bản cạnh nhau từ ứng với từ (mục 6.3 của báo cáo lần 2)                                                    |
| Bảng chọn chung                  | Bước so mọi cách quyết định ghép (C0, F, G, H) trên cùng các đoạn, bằng một quy tắc chọn, dùng tập dò một lần (mục 7) |
| Độ chính xác quyết định đoạn     | Tỉ lệ đoạn khác nhau mà một cách quyết định chọn đúng phía (RNNT hay large-v3) ít lỗi hơn so với đáp án (mục 7) |
| Kiểm định chéo 5 phần theo mẫu   | Chia tập học thành 5 phần, mỗi mẫu nằm trọn trong một phần; lần lượt học trên 4 phần và chấm trên phần còn lại, rồi lấy trung bình. Cho biết một lựa chọn tốt tới đâu trên mẫu chưa dùng để học, mà không cần đụng tới tập dò |
| Bộ phân loại                     | Một mô hình nhỏ nhận các tín hiệu của một đoạn khác nhau và trả về xác suất "nên lấy chữ của large-v3" (mục 6)                                                                                       |
| Hồi quy logistic                 | Bộ phân loại đơn giản nhất: cộng các tín hiệu theo trọng số rồi đổi thành xác suất từ 0 tới 1. Trọng số học từ dữ liệu. Ít tham số, nên học được từ vài trăm mẫu                                     |
| Đặc trưng                        | Một tín hiệu dạng số đưa vào bộ phân loại, ví dụ "từ có trong từ điển ngữ cảnh không" (1 hoặc 0)                                                                                                     |
| Nhãn tự động                     | Đáp án cho từng đoạn khác nhau ("lấy large-v3 thì đúng hơn hay sai hơn"), tính máy móc bằng cách so với đáp án của BTC; không cần người gán                                                          |
| Điểm tin cậy của large-v3        | `avg_logprob` (trung bình log xác suất của các token, càng gần 0 càng chắc) và `no_speech_prob` (xác suất đoạn đó không có lời nói) mà faster-whisper trả về cho mỗi đoạn âm thanh                   |
| Hai phương án                    | Cách viết chỗ bất đồng thành `{chữ RNNT | chữ large-v3}` để LLM tự chọn (mục 8)                                                                                                                      |
| Trần của hai phương án           | Tỉ lệ từ tiếng Anh của đáp án có mặt trong ít nhất một phương án. Là mức tối đa LLM có thể đạt nếu luôn chọn đúng                                                                                    |
| p50, p95, tối đa                 | Thời gian chép lời **một mẫu** (lượt cuối và mọi lượt người dùng trong lịch sử) mà 50%, 95%, 100% số mẫu nhanh hơn                                                                                   |
| RTF                              | Hệ số thời gian thực: số giây xử lý chia số giây âm thanh. RTF 0,2 nghĩa là 10 giây âm thanh xử lý mất 2 giây                                                                                        |
| Bộ nhớ GPU đỉnh                  | Lượng bộ nhớ GPU lớn nhất mà tiến trình ASR chiếm trong suốt lần chạy, đọc bằng `nvidia-smi` (mục 2.4)                                                                                               |
| Điểm âm thanh                    | Tổng log xác suất mà whisper-large-v3, không câu mồi, gán cho từng token của một câu cho sẵn khi nghe đoạn âm thanh; càng gần 0 thì câu càng khớp âm thanh (mục 5) |
| Ép chuỗi (teacher forcing)       | Đưa sẵn chuỗi token vào decoder và chỉ đọc xác suất của từng token, không để decoder tự sinh chữ                    |
| A30, T4                          | A30: GPU của máy chấm, 24 GB, thế hệ Ampere, hỗ trợ bfloat16. T4: GPU miễn phí của Kaggle và Colab, 15 GB (16 GB danh nghĩa), thế hệ Turing, không hỗ trợ bfloat16                                            |


### 0.4. Cách đọc các chỗ nhắc tới code

Giống lần 2: báo cáo ghi theo dạng "file › hàm: việc hàm làm". Mọi đường dẫn tính từ thư mục gốc của repo. Mọi tên
hàm trong báo cáo là tên thật trong code đã chạy; bảng đầy đủ ở Phụ lục A.

## 1. Vấn đề chung và mục tiêu

**Vấn đề.** Lần 2 cho thấy ba điều. Thứ nhất, câu mồi chỉ giúp large-v3 viết đúng những từ **có trong** câu mồi, nên
mức tăng trên app lạ nhỏ (56% → 61% trên `val_unseen`), trong khi bộ thi thật có app lạ. Thứ hai, bản ghép vẫn làm câu
thuần Việt sai thêm 0,3–0,4 điểm WER so với RNNT, chủ yếu vì large-v3 viết ra từ của câu mồi khi người dùng không nói.
Thứ ba, chưa biết phần ASR hai mô hình tốn bao nhiêu thời gian khi chạy chung một tiến trình.

**Suy từ mục tiêu thật.** Điểm thi chấm lời gọi tool. LLM cần bốn loại thông tin từ lời nói:


| Thông tin                 | Ví dụ                               | Cách chấm                                                        | ASR cần chính xác tới đâu                              |
| ------------------------- | ----------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------ |
| Ý định: chọn app và tool  | "nhắn Zalo", "đặt xe"               | Tên tool phải đúng                                               | Đủ để nhận ra ý định, không cần đúng chính tả          |
| Đối tượng có mã (`x-ref`) | liên hệ "Sarah", thiết bị "đèn bếp" | Mã phải đúng, mà mã lấy từ kết quả tool chứ không lấy từ lời nói | Đủ gần để LLM chọn đúng trong danh sách mà tool trả về |
| Giá trị chính xác         | 500 nghìn, 3 giờ chiều, 26 độ       | Phải khớp tuyệt đối                                              | Rất cao                                                |
| Văn bản tự do (`fuzzy`)   | Nội dung tin nhắn                   | Đạt khi token-F1 &gt; 0,6                                        | Sai vài từ vẫn qua                                     |


Từ bảng này: chính tả tiếng Anh quan trọng nhất ở loại đầu (tên app, tên dịch vụ) và một phần loại hai. Những từ này
thường **có trong ngữ cảnh của mẫu** (catalog của app đã cài, lượt trợ lý vừa nhắc). Danh bạ, thiết bị, playlist chỉ xuất
hiện sau khi LLM gọi tool đọc, nên ASR không bao giờ thấy được; chỉ LLM thấy. Vì vậy lần 3 đưa ngữ cảnh vào ASR ở chỗ
ASR thấy được (E, F, H), và chuẩn bị đường để LLM tự sửa ở chỗ chỉ LLM thấy được (I). G thêm bằng chứng từ chính âm
thanh, và J kiểm tra các lựa chọn bằng điểm thi.

**Mục tiêu.**

1. Đo xem câu mồi theo mẫu có tăng tỉ lệ chép đúng từ tiếng Anh, nhất là trên `val_unseen`, mà không làm câu thuần Việt
sai thêm không (E).
2. Đóng phần lỗi còn lại ở câu thuần Việt của bản ghép, bằng một trong ba cách quyết định ghép mới: độ tin cậy từng từ
(F), điểm âm thanh của large-v3 không câu mồi (G), bộ phân loại (H). Ba cách và luật của lần 2 được so trong bảng chọn
chung (mục 7).
3. Đo trần và chi phí của cách giữ hai phương án cho LLM (I), rồi đo thật nếu trần đủ cao.
4. Đo điểm thi của từng nguồn chữ bằng cách chạy agent trên chữ đã lưu (J).
5. Đo thời gian và bộ nhớ của phần ASR ở mọi lần chạy, để mỗi lựa chọn có kèm cái giá về thời gian.
6. Viết engine ghép vào hệ thống thật (`lifemate/backends/asr.py`), để thời gian đo được là thời gian của đúng thứ sẽ nộp,
và để bước sau chạy được cả hệ thống với LLM. **Chưa làm ở lần 3:** thời gian được đo bằng bộ chạy
`lifemate/eval/asr_v3.py`, chạy đúng các bước theo cùng thứ tự trong cùng một tiến trình; engine cho hệ thống thật để
sau khi J chọn xong cấu hình (Phụ lục A.2).

**Ngoài phạm vi lần 3, và lý do.**


| Ý tưởng                                                                              | Lý do chưa làm                                                                                                                                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cộng điểm cho từ trong danh sách ngay trong bộ giải mã của RNNT (contextual biasing) | Gói `chunkformer` chỉ có giải mã tham lam; phải tự viết beam search có cộng điểm cho RNNT. Với Whisper, lần 3 thử dạng tương đương có sẵn: câu mồi theo mẫu (E) |
| Thêm lớp cross-attention vào danh sách từ bên trong ASR (kiểu CLAS, TCPGen)          | Phải fine-tune mô hình ASR; dự án chỉ có khoảng 3 600 câu train và còn 18 ngày tới hạn nộp 16/11                                                                                       |
| Tiền huấn luyện biểu diễn tự giám sát (JEPA)                                         | Cần rất nhiều âm thanh không nhãn và nhiều GPU; các mô hình ASR đang dùng đã tiền huấn luyện sẵn                                                                                       |
| Fine-tune LLM trên định dạng hai phương án                                           | Thuộc phần fine-tune LLM (SFT). Lần 3 đo trần (I1) và đo với LLM chưa fine-tune (J4); nếu J4 đạt tiêu chí thì định dạng này được đưa vào dữ liệu SFT ở bước sau                                                           |


**Các công cụ chuẩn bị trước.**


| Công cụ                                                                                                                            | Mục      | Dùng ở thử nghiệm   |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------- |
| Công cụ của lần 2: quy tắc tìm từ ngoại, bộ chỉ số tiếng Anh, chấm lại offline, bảng so sánh có CI, quy tắc ghép, từ điển ngữ cảnh | 2.2      | E, F, H, I          |
| Bộ chạy các phiên GPU (`asr_v3.py`): hai mô hình trong một tiến trình, ghi thời gian từng bước, đọc bộ nhớ GPU                    | 2.4      | Mọi phiên GPU       |
| Câu mồi theo mẫu, lớp bảo vệ chỉ so phần cố định, điểm tin cậy của large-v3 trong `hyps.jsonl`                                     | 3.3      | E, H                |
| Bảng độ tin cậy từng từ và cách quyết định `context_reliable` (F)                                                                  | 4.3      | F, H                |
| Dựng câu ứng viên, chấm điểm âm thanh bằng `align` của CTranslate2                                                                 | 5.3      | G, H                |
| Đặc trưng, nhãn tự động, hồi quy logistic                                                                                          | 6.3      | H                   |
| Độ chính xác quyết định đoạn, bảng chọn chung                                                                                      | 7        | F, G, H             |
| Định dạng hai phương án, đếm trần và token                                                                                         | 8.3      | I                   |
| Chạy agent trên chữ đã lưu (`--transcripts replay`)                                                                                | 9.3      | J                   |


## 2. Dữ liệu, thước đo và công cụ dùng chung

### 2.1. Dữ liệu và cách dùng

**Vấn đề.** Lần 3 cần thêm dữ liệu để **học** so với lần 2: đếm độ tin cậy từng từ, chấm điểm âm thanh cho dữ liệu học
của bộ phân loại, học bộ phân loại. Dữ liệu học không được trùng tập dò hay tập kiểm tra.

**Giải pháp.**

1. **Tập học = 793 mẫu train ngoài `asr_tune`.** Lần 2 chỉ đọc chữ large-v3 của các mẫu này (bộ đệm
`data/asr/train_asr.json`, large-v3 beam 5 không câu mồi) để chọn từ cho câu mồi. Lần 3 dùng chính các mẫu này để
đếm độ tin cậy và học bộ phân loại. Lý do: đây là phần duy nhất có đáp án mà chưa dùng để dò hay đo. Bộ đệm cũ không
dùng được cho F và H vì chép bằng beam 5 không câu mồi, khác cấu hình sẽ dùng lúc thi (giải mã tham lam + VAD + câu
mồi); nên phiên K2 chép lại tập học bằng đúng cấu hình đó.
2. **Câu mồi viết lại theo tập học là một bước chọn.** Nếu E hoặc F dẫn tới viết lại câu mồi cố định, việc viết lại chỉ
được dựa vào tập học, như lần 2.
3. **Không dùng `public_test`** ở lần 3, kể cả để học, theo `AGENTS.md`.
4. **Từ điển ngữ cảnh chỉ lấy giá trị, bỏ khoá JSON.** Ở lần 2, `sample_context_words` lấy cả khoá của catalog và của
`context` (`type`, `description`, `properties`, `content`, `name`…), nên các từ này có trong từ điển của mọi mẫu và
large-v3 bịa ra đúng các từ này thì bộ lọc vẫn nhận. Khoá là tên trường do người viết schema đặt, không phải nội dung
người dùng có thể nhắc tới. Đếm thử offline trên tập dò: bỏ khoá bớt 1 từ lấy nhầm ("group") và không mất từ đúng
nào. Lần 3 dùng từ điển bỏ khoá cho mọi thử nghiệm (việc P3). Mốc so sánh `context_other` giữ số của lần 2, không chấm
lại, vì chênh lệch chỉ một từ trên tập dò.


| Việc                    | Dữ liệu                                             | Để làm gì                     | Lý do chọn                                                             |
| ----------------------- | --------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| E: dò biến thể câu mồi  | `asr_tune` đầy đủ (1 024 câu)                       | Chọn biến thể                 | Cần cả câu thường để thấy câu mồi có làm câu thuần Việt sai thêm không |
| E: đo biến thể thắng    | `val_seen`, `val_unseen` đầy đủ, một lần            | Số cuối                       | Như lần 2                                                              |
| F: đếm độ tin cậy       | Tập học, chữ large-v3 với câu mồi thắng của E       | Bảng độ tin cậy từng từ       | Phải cùng cấu hình câu mồi với lúc dùng                                |
| F: chọn ngưỡng          | Tập học, kiểm định chéo 5 phần theo mẫu             | Ngưỡng độ tin cậy             | Tập dò để dành cho bước chọn chung                                     |
| G: chấm điểm            | Tập học, `asr_tune`, val (phiên K3)                 | Điểm âm thanh từng câu ứng viên | Điểm trên tập học dùng làm đặc trưng của H                   |
| H: học bộ phân loại     | Tập học: chữ RNNT, chữ large-v3, điểm của G         | Trọng số của hồi quy logistic | Học trên tập không dùng để dò hay đo                                   |
| Chọn cách quyết định ghép | `asr_tune`, một lần; ứng viên thắng đo một lần trên val | Ứng viên thắng (mục 7) | Mọi tham số đã cố định trên tập học                     |
| I1: đo trần             | `asr_tune`, val                                     | Trần và chi phí token         | Không chọn gì                                                          |
| J: điểm thi             | `val_seen`, `val_unseen`, chữ đã lưu của các phiên K | Điểm thi từng nguồn chữ       | Không chọn gì trong lần 3                                              |


### 2.2. Thước đo

Giữ nguyên bộ chỉ số của lần 2 (mục 2.3 của báo cáo lần 2: `en_word_recall` và bản nới lỏng, `en_groups`,
`en_mixed_wer`, `vi_only_wer`, `prompt_insertions`, `en_prompt`, `n_prompt_retries`, WER, CER, bỏ sót giá trị tham số,
câu rỗng, câu bịa, p50/p95/max). Mọi chỉ số chất lượng được báo cho **bản ghép**, vì bản ghép là thứ sẽ đưa cho LLM; chỉ
số của large-v3 đứng riêng được báo kèm để giải thích.

Chỉ số mới:


| Chỉ số (tên trong file kết quả) | Cách tính | Vì sao cần |
| --- | --- | --- |
| Chép đúng theo có/không có trong ngữ cảnh (`en_context_v3`, `en_context_merged` trong `e.json`) | Như `en_word_recall`, tách theo từ của đáp án **có** và **không có** trong từ điển ngữ cảnh của mẫu; tính cho large-v3 đứng riêng và cho bản ghép | Câu mồi theo mẫu chỉ có thể giúp từ có trong ngữ cảnh. Tách ra để biết mức tăng đến từ đâu |
| Chép lại lượt trợ lý (`assistant_echo`) | Số câu mà chữ ASR có 4 từ liên tiếp trùng câu trả lời của trợ lý liền trước, trong khi đáp án không có 4 từ đó | Câu mồi theo mẫu chứa lời trợ lý. Đây là dấu hiệu large-v3 in lại lời trợ lý thay vì nghe |
| Số câu bị cắt câu mồi (`truncated`), số câu chép lại do lớp bảo vệ (`retries`) | Câu mồi dài hơn 223 token; câu mà lớp bảo vệ g4 bắt và chép lại không câu mồi | Câu mồi theo mẫu dài hơn câu mồi cố định |
| Từ bị lấy nhầm và từ đúng bị bỏ khi ghép (`word_errors` trong `select.json`: `wrong_prompt`, `wrong_other`, `missed_prompt`, `missed_other`) | Lấy nhầm: từ tiếng Anh mới mà phép ghép lấy từ large-v3 nhưng đáp án câu đó không có. Bỏ sót: từ tiếng Anh mới của large-v3 có trong đáp án mà phép ghép không lấy. Mỗi loại tách theo từ của câu mồi / ngoài câu mồi | Đo trực tiếp hai mặt của đánh đổi mà F, G, H muốn cải thiện. Lần 2 đếm tay bằng script; lần 3 đưa vào code |
| Độ chính xác quyết định đoạn (`decision_accuracy`) | Trên các đoạn có nhãn tự động (mục 6.3), tỉ lệ đoạn mà cách quyết định chọn đúng phía ít lỗi hơn so với đáp án; kèm CI và CI của chênh lệch so với C0 | Chỉ số chọn của bảng chọn chung (mục 7): đo thẳng các quyết định mà các ứng viên khác nhau |
| Trần của hai phương án (`I1` trong `select.json`: `chosen_recall`, `ceiling_recall`, `ceiling_diff_ci`, `counts`) | Mục 8 | I |
| Token thêm cho LLM | Mục 8; đếm bằng tokenizer của Qwen3-4B | I |
| Thời gian từng bước mỗi mẫu (`rnnt_s`, `whisper_s`, `merge_s` trong `timing.jsonl`) | Đồng hồ thực (`time.perf_counter`) quanh từng bước, cộng theo mẫu. `whisper_s` gồm cả lớp bảo vệ và lần chép lại nếu có | Biết bước nào tốn thời gian |
| Thời gian ASR mỗi mẫu: p50, p95, tối đa (`whisper_p50_p95_max`, `total_p50_p95_max` trong `e.json`) | Tổng thời gian các bước của mọi câu trong mẫu | Ngân sách 30 giây là theo mẫu |
| RTF | Tổng thời gian ASR chia tổng số giây âm thanh của tập | So được giữa các tập có độ dài khác nhau |
| Thời gian nạp mô hình (`load_s` trong `summary.json`) | Từ lúc nạp mô hình tới khi khởi động (`warmup`) xong | Không tính vào từng mẫu, nhưng cộng vào tổng thời gian chạy Docker |
| Bộ nhớ GPU đỉnh (`gpu_peak_mb` trong `summary.json`) | Mục 2.4 | Phần còn lại cho vLLM |


**Giới hạn.** `assistant_echo` dựa vào trùng 4 từ, nên bỏ sót trường hợp large-v3 in lại lời trợ lý nhưng đổi vài từ.

### 2.3. Cách chọn và đo

Như lần 2: chọn trên `asr_tune`, đo cấu hình thắng đúng một lần trên `val_seen` và `val_unseen`. So sánh dùng bootstrap có
ghép cặp theo mẫu, 1 000 vòng, seed 0 (`lifemate/eval/asr_compare.py` › `compare`). Hai hệ thống chỉ được coi là khác
nhau khi CI 95% của chênh lệch không chứa 0. Tiêu chí chọn của từng thử nghiệm đã ghi ở phần tóm tắt, trước khi đo.

Tập dò được dùng hai lần: chọn câu mồi (E), rồi chọn cách quyết định ghép (mục 7). Mọi tham số khác (θ của F, hệ số của
H) học trên tập học, không chọn trên tập dò. Vì tập dò vẫn được dùng hai lần, số trên tập dò của bước chọn thứ hai
lạc quan hơn thực tế; chỉ số trên val của cấu hình cuối mới là số để báo cáo. Hai bước chọn đều được đo một lần trên val
ngay sau khi chọn, để biết bước nào đóng góp bao nhiêu; không bước nào chỉnh lại theo số val.

### 2.4. Đo thời gian và bộ nhớ ở mọi lần chạy

**Vấn đề.** Lần 2 đo thời gian của từng mô hình riêng lẻ, bằng hàm chấm `evaluate`. Như vậy chưa biết thời gian khi hai
mô hình cùng nằm trong một tiến trình, thời gian của lớp bảo vệ, của phép ghép và của bước chấm điểm G, và bộ nhớ GPU
thật. Mỗi lựa chọn ở lần 3 (câu mồi dài hơn ở E, chấm điểm thêm ở G) cũng có cái giá về thời gian cần biết ngay khi chọn.

**Giải pháp.** Không có thử nghiệm riêng cho thời gian; mọi phiên GPU (K1, K2, K3) đo kèm.

- **Đo trong cùng một tiến trình.** Bộ chạy `lifemate/eval/asr_v3.py` nạp RNNT và large-v3 một lần rồi chạy lần lượt
RNNT, large-v3, lớp bảo vệ, phép ghép (phiên K1, K2) và chấm điểm G (phiên K3), đúng thứ tự các bước lúc nộp. Engine ghép
cho hệ thống thật chưa viết ở lần 3 (Phụ lục A.2).
- **Thời gian mỗi mẫu.** Bộ chạy ghi một dòng `timing.jsonl` cho mỗi mẫu: `{id, n_utterances, audio_s, rnnt_s,
whisper_s, merge_s, retries}`. Ở phiên K1, `whisper_s`, `merge_s` và `retries` có một giá trị cho mỗi biến thể câu mồi.
`whisper_s` gồm cả lớp bảo vệ và lần chép lại không câu mồi nếu có. Thời gian cả phần ASR của một mẫu là
`rnnt_s + whisper_s + merge_s`. Phiên K3 ghi `scores_<tập>.timing.jsonl` cho bước chấm điểm.
- **Khởi động trước khi đo.** Mỗi mô hình chạy một lần trên 1 giây im lặng (`warmup`) trước mẫu đầu tiên, và lần đó
không tính. Lý do: lần chạy đầu của GPU tốn thêm thời gian dựng kernel, không lặp lại ở các mẫu sau.
- **Bộ nhớ GPU.** large-v3 chạy trên CTranslate2, không phải PyTorch, nên `torch.cuda.max_memory_allocated` không thấy
bộ nhớ của large-v3. Vì vậy `lifemate/eval/gpu_memory.py` › `GpuMemoryMonitor` chạy nền
`nvidia-smi --query-gpu=memory.used --format=csv -lms 200` trong suốt phiên và ghi `gpu_memory.csv`; bộ nhớ đỉnh
(`gpu_peak_mb` trong `summary.json`) là giá trị lớn nhất trừ giá trị trước khi nạp mô hình.
- **Kiểm tra tất định.** Ở phiên K1, bộ chạy chép lại 50 mẫu đầu của `asr_tune` (145 câu) lần hai bằng RNNT và large-v3
E0, rồi so từng ký tự với lần một (`--rerun 50`; kết quả ở mục `determinism` của `summary.json`). Phiên K3 chấm lại 50
đoạn đầu và so điểm.
- **Báo ở đâu.** E báo p95 của large-v3 cho từng biến thể (mục 3.7). Bảng chọn chung báo thời gian chấm thêm của từng
ứng viên (mục 7). Phần tóm tắt báo thời gian mỗi mẫu và bộ nhớ đỉnh của cấu hình cuối trên val.

**Kết quả đo chung** (Colab T4, cấu hình E2 + C0):

| Phiên, tập | Số mẫu | Nạp hai mô hình (s) | Bộ nhớ GPU đỉnh | RTF cả phần ASR |
|---|---|---|---|---|
| K1, `asr_tune` (ba biến thể) | 300 | 48,5 | 3 607 MB | |
| K2, tập học | 793 | 50,6 | 3 607 MB | |
| K2, `val_seen` | 274 | 26,5 | 3 575 MB | 0,26 |
| K2, `val_unseen` | 178 | 29,2 | 3 543 MB | 0,25 |
| K3, chấm điểm (`asr_tune`) | 201 đoạn | 16,0 | 3 351 MB | |

Kiểm tra tất định: 0 câu khác ở cả RNNT lẫn large-v3 (145 câu), 0 đoạn có điểm khác (50 đoạn). Thời gian mỗi mẫu ở phần
tóm tắt ("Kết quả chính").

**Giới hạn.**

- Số đo trên T4 (Colab), không phải A30 của máy chấm, và không có vLLM chạy cùng. Thời gian và bộ nhớ trên A30 khi chạy
cạnh LLM đo ở lần chạy hệ thống sau lần 3.
- Thời gian đọc file âm thanh phụ thuộc ổ đĩa của máy chấm, không đo được trước.
- Mẫu dài nhất của val có 69,4 giây âm thanh, ngắn hơn mẫu dài nhất của `public_test` (82,8 giây); thời gian của mẫu chậm
nhất trên val có thể thấp hơn lúc chấm thật.

## 3. Thử nghiệm E: câu mồi theo mẫu

### 3.1. Vấn đề

Câu mồi cố định `en_full_v1` chứa 38 từ khoá, chọn từ những từ large-v3 hay chép sai trên tập học. Lần 2 cho thấy câu
mồi chỉ giúp đúng những từ đó: trên `val_seen`, từ có trong câu mồi được chép đúng từ 48% lên 84%, từ không có trong câu
mồi gần như không đổi. App lạ dùng nhiều từ không có trong câu mồi, nên trên `val_unseen` large-v3 chỉ tăng từ 56% lên
61%. Câu mồi đã gần chạm trần độ dài (183 trên 223 token), nên không thể thêm từ mãi.

### 3.2. Ý tưởng

Thay vì một danh sách từ cho mọi câu, đưa vào câu mồi **những từ mà câu đang chép có khả năng chứa**. Thông tin đó có sẵn
dạng chữ trong mẫu: lượt trợ lý liền trước (70% số câu có lượt này, mục 0.2) và các app đã cài. Câu mồi của Whisper vốn
được thiết kế là "đoạn nói ngay trước đó", nên câu trả lời của trợ lý liền trước đúng là loại chữ mà Whisper được huấn
luyện để dùng làm câu mồi.

Lần 1 cho thấy câu mồi dạng **liệt kê** (danh sách từ khoá) làm large-v3 chép lại danh sách ("VTV4 VTV5…"). Vì vậy biến
thể đưa thông tin vào dưới dạng **câu nói tự nhiên** (lời trợ lý), không dùng danh sách từ khoá. Dạng liệt kê qua
`hotwords` không thử lại, vì `hotwords` kém nhất ở lần 1.

### 3.3. Giải pháp và thuật toán

**Hai biến thể và một đối chứng.** Với mỗi câu người dùng `u` của mẫu (lượt trong lịch sử hoặc lượt cuối):


| Biến thể               | Câu mồi của câu `u`                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| E0 (đối chứng, lần 2)  | `en_full_v1`                                                                                                                |
| E1 "lời trợ lý"        | `en_full_v1`, rồi câu trả lời (`text`) của lượt trợ lý **liền trước** `u`                                                   |
| E2 "app và lời trợ lý" | `en_full_v1`, rồi một câu nêu tên tiếng Việt (`name_vi`) của các app đã cài, rồi câu trả lời của lượt trợ lý liền trước `u` |


Câu không có lượt trợ lý liền trước (lượt đầu tiên của hội thoại) dùng `en_full_v1` như E0, cộng phần tên app ở E2.

**Thứ tự trong câu mồi.** Phần riêng theo mẫu đặt **cuối**, vì Whisper chỉ dùng 223 token cuối của câu mồi và phần cuối
gần đoạn âm thanh nhất. Khi tổng dài quá 223 token, phần đầu của `en_full_v1` bị cắt; đó là các từ khoá ít sai nhất (lần
2 đặt từ hay sai nhất ở cuối). Lời trợ lý dài tối đa 71 từ, nên trường hợp xấu nhất mất khoảng một nửa `en_full_v1`. Số
câu bị cắt và số token bị cắt được ghi lại.

**Code.**

- `lifemate/backends/asr_prompt.py` › `build_sample_prompts(sample, catalog_apps, fixed_prompt, variant)`:
nhận một mẫu, catalog (từ điển mã app → mục catalog), câu mồi cố định và tên biến thể; trả về danh sách câu mồi, một
câu mồi cho mỗi câu người dùng theo đúng thứ tự file âm thanh của `ASRTranscriber.wav_paths` (lịch sử trước, lượt cuối
sau). Hàm chỉ đọc `sample.context`, `sample.history` (lượt trợ lý) và catalog; không đọc lượt người dùng và không đọc
file môi trường của BTC.
- `lifemate/backends/asr.py` › `FasterWhisperASR.transcribe_one(audio, prompt)`: chép một câu với câu mồi riêng, trả
thêm điểm tin cậy; `transcribe_batch(wav_paths, prompts=None)` nhận câu mồi riêng cho từng file. Bộ chạy
`lifemate/eval/asr_v3.py` gọi `build_sample_prompts` cho từng mẫu; khoá config `[asr].prompt_mode` cho hệ thống thật chưa
thêm (Phụ lục A.2).
- `hyps.jsonl` có thêm trường `prompt` (câu mồi đã dùng cho câu đó) và `scores` (`avg_logprob`,
`no_speech_prob` của từng đoạn, cho thử nghiệm H). File kết quả không lên git, nên chứa chữ của dữ liệu cũng được.

**Lớp bảo vệ khi câu mồi có lời trợ lý.** Lớp bảo vệ g4 của lần 2 coi câu có 4 từ liên tiếp trùng câu mồi là "in lại câu
mồi" và chép lại không câu mồi. Với câu mồi có lời trợ lý, người dùng hay **nhắc lại đúng lời trợ lý** ("Gửi cho Sarah
hay Sara?" → "gửi cho Sarah"), nên so với toàn bộ câu mồi sẽ bắt nhầm câu đúng. Vì vậy ở E1 và E2, lớp bảo vệ chỉ so với
**phần cố định** (`en_full_v1`), cộng kiểm tra tỉ lệ nén như cũ (bắt câu lặp). Việc in lại lời trợ lý được đo riêng bằng
`assistant_echo` (mục 2.2), không dùng để chép lại. Code: `lifemate/backends/asr_guard.py` › `needs_retry` giữ nguyên;
bộ chạy `lifemate/eval/asr_v3.py` truyền vào câu mồi phần cố định thay cho câu mồi đầy đủ.

**Phép ghép với câu mồi theo mẫu.** `context_other` nhận mọi từ của câu mồi. Với câu mồi theo mẫu, "từ của câu mồi" gồm
từ của phần cố định và từ của phần riêng; từ của phần riêng vốn đã có trong từ điển ngữ cảnh, nên quy tắc không đổi.
Vì vậy phép ghép chỉ cần tập từ của phần cố định (`en_full_v1`) làm "từ của câu mồi", giống lần 2
(`lifemate/eval/asr_v3_report.py` › `_fixed_prompt`).

### 3.4. Lý do

- **Lời trợ lý thay vì danh sách từ.** Whisper học câu mồi như đoạn nói trước đó, nên bắt chước cách viết của câu tự
nhiên tốt hơn danh sách; danh sách làm mô hình chép lại danh sách (lần 1).
- **Giữ `en_full_v1` ở đầu.** Bỏ hẳn phần cố định thì mất mức tăng đã đo được của lần 2 ở các câu không có lượt trợ lý
liền trước (30% số câu).
- **Tên app ở E2.** App lạ của bộ thi thật kéo theo tên app lạ; tên app có trong `installed_apps` của chính mẫu đó.

### 3.5. Dữ liệu và số lần đo

- Dò: ba biến thể trên `asr_tune` đầy đủ (1 024 câu), phiên K1. E0 chạy lại cùng phiên dù đã có chữ từ lần 2, để thời
gian ba biến thể đo trên cùng một máy.
- Đo: biến thể thắng trên `val_seen` và `val_unseen` đầy đủ, một lần, phiên K2.
- Mỗi biến thể được ghép với RNNT bằng `context_other` rồi chấm, vì tiêu chí chọn tính trên bản ghép.

### 3.6. Chỉ số và tiêu chí

Chỉ số chính: chép đúng từ tiếng Anh của bản ghép, tách theo `en_context` (từ có / không có trong ngữ cảnh). Chỉ số đi
kèm: WER câu thuần Việt của bản ghép, `prompt_insertions`, `assistant_echo`, `n_prompt_retries`, số câu bị cắt câu mồi,
p95 thời gian của large-v3.

Tiêu chí chọn (đặt trước): biến thể có chép đúng từ tiếng Anh của bản ghép cao nhất trên `asr_tune`, với điều kiện WER câu
thuần Việt của bản ghép không tăng so với E0 (CI của chênh lệch không nằm hẳn trên 0), và p95 thời gian của large-v3 tăng
không quá 1 giây so với E0. Không biến thể nào đạt thì giữ E0.

### 3.7. Kết quả

Phiên K1 chạy trên Colab T4 (mục "Máy chạy thực tế" ở Phụ lục B.2). 300 mẫu, 1 024 câu, mỗi câu chép bằng RNNT và bằng
large-v3 với ba biến thể câu mồi.

**Tập dò (`asr_tune`), large-v3 đứng riêng** (chênh lệch so với E0 kèm CI 95%):

| Biến thể | Chép đúng từ tiếng Anh | Từ có trong ngữ cảnh | Từ không có trong ngữ cảnh | WER câu thuần Việt | WER mọi câu | Chèn từ câu mồi | Chép lại lời trợ lý | Chép lại (lớp bảo vệ) | Câu bị cắt câu mồi | Độ dài câu mồi (token, trung vị) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 70,6% (278/394) | 188/252 | 107/142 | 29,5% | 29,7% | 26 | 0 | 2 | 0 | 183 |
| E1 | 72,6% [−0,6; +4,9] | 195/252 | 107/142 | 27,6% [−2,5; −1,3] | 27,7% [−2,6; −1,4] | 21 | 2 | 2 | 143 | 200 |
| E2 | 73,6% [−0,3; +6,2] | 192/252 | 108/142 | 26,0% [−4,4; −2,8] | 26,3% [−4,0; −2,7] | 20 | 2 | 3 | 990 | 270 |

**Tập dò, bản ghép (RNNT + biến thể, luật `context_other`)**, chênh lệch so với E0 kèm CI 95%:

| Biến thể | Chép đúng từ tiếng Anh | Từ có trong ngữ cảnh | Từ không có trong ngữ cảnh | WER câu thuần Việt | WER câu chêm | WER mọi câu | large-v3 p95 mỗi mẫu (s) |
|---|---|---|---|---|---|---|---|
| E0 | 69,0% | 194/252 | 95/142 | 17,8% | 24,2% | 19,9% | 7,79 |
| E1 | 70,8% [−0,6; +4,4] | 201/252 | 95/142 | 17,7% [−0,2; +0,1] | 23,8% [−0,7; 0,0] | 19,8% [−0,3; 0,0] | 7,82 |
| E2 | **72,1%** [0,0; +6,1] | 199/252 | 95/142 | 17,8% [−0,2; +0,2] | **23,5%** [−1,2; −0,1] | **19,7%** [−0,5; −0,0] | 7,95 |

**Chọn theo tiêu chí đặt trước:** cả ba biến thể đều không làm WER câu thuần Việt của bản ghép tăng (CI chứa 0), và p95
của large-v3 tăng nhiều nhất 0,16 giây (dưới ngưỡng 1 giây). Trong đó E2 có chép đúng từ tiếng Anh của bản ghép cao nhất,
nên **E2 thắng**. Mọi bước sau dùng chữ large-v3 với câu mồi E2.

**Kiểm tra tất định:** chép lại 50 mẫu đầu của tập dò (145 câu) với E0, chữ giống hệt lần một ở cả RNNT lẫn large-v3
(0 câu khác).

**Tập kiểm tra, đo một lần** (bản ghép với luật `context_other`; chênh lệch so với E0 kèm CI 95%):

| Tập | Hệ thống | Chép đúng từ tiếng Anh | Từ có / không có trong ngữ cảnh | WER câu thuần Việt | WER câu chêm | WER mọi câu |
|---|---|---|---|---|---|---|
| val\_seen | RNNT đứng riêng | 49,9% | | 17,7% | 24,8% | 20,0% |
| | Ghép, E0 (lần 2) | 72,2% | 185/238, 65/97 | 18,1% | 22,1% | 19,4% |
| | Ghép, **E2** | **75,5%** [0,0; +6,6] | 190/238, 66/97 | 17,9% [−0,4; 0,0] | 21,7% [−1,0; +0,1] | **19,1%** [−0,5; −0,0] |
| val\_unseen | RNNT đứng riêng | 51,6% | | 17,5% | 24,1% | 19,8% |
| | Ghép, E0 (lần 2) | 63,1% | 112/173, 49/79 | 17,8% | 22,8% | 19,5% |
| | Ghép, **E2** | **68,7%** [+2,2; +9,8] | 125/173, 49/79 | **17,6%** [−0,3; −0,0] | **22,0%** [−1,4; −0,3] | **19,2%** [−0,6; −0,2] |

large-v3 đứng riêng với E2 trên val: chép đúng từ tiếng Anh 74,9% / 67,9% (E0: 72,2% / 61,1%), WER câu thuần Việt 26,5% /
25,5% (E0: 29,2% / 28,5%).

**Ví dụ thật** (tập dò, phiên K1; chữ large-v3 đứng riêng, chưa ghép).

Câu mồi E2 đầy đủ của lượt 16 trong mẫu `train_1650` gồm ba phần nối liền nhau:

| Phần | Nội dung |
|---|---|
| 1. Câu mồi cố định `en_full_v1` (38 từ khoá, chỉ trích đầu và cuối) | "QR trên app mờ quá. Email của Grace Nakamura tới, Peter Hale mượn laptop chạy demo cho Nexora Labs, nhớ viết note. … Voucher nằm trong SMS, check kỹ rồi gửi Zalo cả xóm, coi miếng SJC luôn." |
| 2. Tên tiếng Việt của các app đã cài | "Ứng dụng đã cài: Nhắc việc, Ghi chú, Tỷ giá, Lịch, Họp trực tuyến, Tin nhắn SMS, Đặt đồ ăn, Danh bạ, Vé máy bay." |
| 3. Câu trả lời của trợ lý ở lượt liền trước | "Dạ mai có một cái nhắc: gửi slide cho Sếp David lúc tám giờ sáng anh." |

Câu người dùng nói sau đó là "Thôi bỏ cái đó đi, anh gửi slide luôn rồi." Với câu mồi cố định (E0), large-v3 viết "Phải
bỏ cái đó đi, ăn gửi **Zalo** ngon": từ "Zalo" của câu mồi kéo mô hình đi. Với E1 và E2, lời trợ lý có "slide", nên
large-v3 viết "Phải bỏ cái đó đi, anh gửi **slide** luôn rồi."

Các câu khác mà E2 viết đúng một từ tiếng Anh còn E0 viết sai:

| Mẫu | Người dùng nói (đáp án) | E0 (câu mồi cố định) | E2 | Từ đúng lấy từ đâu |
|---|---|---|---|---|
| `train_1641`, lượt 6 | Lấy chuyến JetXanh chín giờ sáng, một người thôi. | Ngày chuyến sẽ sẵn 9h sáng một người thôi. | Ngay chuyến, **JetXanh** chín giờ sáng một người thôi. | Lời trợ lý liền trước liệt kê các chuyến bay, có "JetXanh chín giờ" |
| `train_1740`, lượt cuối | Mở podcast Nhà giả kim bên Sách nói đêm khuya cho em… | Mở bóc khác, nhà giả kim… | Mà ở **podcast** nhà giả kim… | Tên app đã cài "Podcast & Radio" (chỉ E2 có; E1 vẫn viết "bóc khác") |
| `train_0111`, lượt 4 | Giá vàng SJC bữa nay thế nào? | Giá VASJC bữa nay thế nào? | Giá **vàng SJC** bữa nay thế nào? | Tên app đã cài "Giá vàng" (E1 vẫn viết "VASJC") |

Một trong hai câu bị tính là **chép lại lời trợ lý** (`train_0840`, lượt cuối): trợ lý hỏi "Danh bạ có hai chị Ngân:
một chị bên phòng pháp lý và một chị bên phòng tuyển dụng…", người dùng nói "Chị Ngân bên pháp lý ấy…", large-v3 viết
"Chị Ngân bên **phòng** pháp lý…". Mô hình chèn thêm một từ của lời trợ lý mà người dùng không nói; nghĩa không đổi.

### 3.8. Phân tích

- **Câu mồi theo mẫu giúp đúng chỗ dự đoán: các từ có trong ngữ cảnh của mẫu.** Trên `val_unseen`, số từ có trong ngữ cảnh
  được chép đúng tăng từ 112 lên 125 (trên 173), còn từ không có trong ngữ cảnh giữ nguyên 49/79. Đây là tập có app lạ,
  nơi câu mồi cố định của lần 2 giúp ít nhất (mục 3.1), và cũng là tập tăng nhiều nhất (+5,6 điểm, có ý nghĩa thống kê).
  Trên `val_seen` mức tăng +3,3 điểm nhưng CI chạm 0.
- **Lợi ích lớn hơn dự kiến ở phần tiếng Việt của large-v3.** WER câu thuần Việt của large-v3 đứng riêng giảm 2,7–3,5 điểm,
  có ý nghĩa thống kê ở cả ba tập. Lời trợ lý là câu tiếng Việt tự nhiên đúng chủ đề hội thoại, nên large-v3 viết tiếng
  Việt tốt hơn. Bản ghép giữ chữ RNNT ở phần tiếng Việt, nên phần lợi này chỉ thấy ở WER câu chêm và WER mọi câu của bản
  ghép (giảm 0,3–0,8 điểm).
- **Câu mồi E2 bị cắt ở gần mọi câu mà vẫn thắng.** Câu mồi E2 dài trung vị 270 token, vượt giới hạn 223 token, nên 990
  trên 1 024 câu bị cắt mất phần đầu của `en_full_v1`. Phần bị cắt là các từ khoá ít sai nhất (mục 3.3), nên mức chép đúng
  không giảm; phần còn lại cộng với tên app và lời trợ lý cho kết quả tốt hơn.
- **Rủi ro chép lại lời trợ lý rất nhỏ.** Chỉ 2 câu trên 1 024 có 4 từ liên tiếp của lời trợ lý mà đáp án không có, và số
  lần chèn từ của câu mồi giảm từ 26 xuống 20.
- **Chi phí thời gian không đáng kể.** p95 của large-v3 tăng 0,16 giây mỗi mẫu (7,79 lên 7,95 giây trên T4).

### 3.9. Kết luận

Dùng câu mồi E2 cho large-v3: `en_full_v1`, rồi tên tiếng Việt của các app đã cài, rồi lời trợ lý liền trước câu đó; lớp
bảo vệ chỉ so với phần cố định. So với lần 2, bản ghép chép đúng thêm 3,3 điểm (`val_seen`) và 5,6 điểm (`val_unseen`, có
ý nghĩa thống kê) mà câu thuần Việt không sai thêm, và tốn thêm khoảng 0,2 giây ở p95.

### 3.10. Giới hạn

- Câu trả lời của trợ lý trong dữ liệu train do BTC viết. Lúc thi, lượt trợ lý trong lịch sử cũng do BTC cung cấp, nên
hai phía cùng nguồn; chỉ lượt cuối là câu hệ thống phải tự xử lý.
- 30% số câu không có lượt trợ lý liền trước, nên E1 không đổi gì ở các câu đó.
- Mẫu không có lịch sử chỉ còn tên app (E2).

## 4. Thử nghiệm F: độ tin cậy từng từ của câu mồi

### 4.1. Vấn đề

Quy tắc `context_other` nhận mọi từ của câu mồi mà large-v3 viết ra. Trên tập dò của lần 2, quy tắc này còn lấy nhầm 30
từ, trong đó 24 từ là từ của câu mồi ("Mail" cho "mai", "list" cho "lịch"). Vì vậy câu thuần Việt của bản ghép vẫn sai
nhiều hơn RNNT 0,4 điểm (seen) và 0,3 điểm (unseen). Lọc mọi từ của câu mồi bằng ngữ cảnh thì mất nhiều tiếng Anh hơn hẳn,
vì phần lớn từ của câu mồi mà large-v3 viết ra là đúng (66 trong 90 lần ở tập dò) nhưng nhiều từ đúng không có trong ngữ
cảnh (mục 6.8 của báo cáo lần 2).

### 4.2. Ý tưởng

Các từ của câu mồi không đáng tin như nhau. Có từ trông giống một âm tiết Việt hay gặp ("Mail" giống "mai", "list" giống
"lịch"), nên large-v3 hay viết ra khi người dùng không nói. Có từ không giống âm Việt nào ("SJC", "Zalo"), nên hầu như chỉ
xuất hiện khi người dùng nói thật. Đo riêng từng từ thì có thể nhận hết từ đáng tin và chỉ đòi thêm bằng chứng với từ
hay sai.

**Ví dụ.** Cả "SJC" và "Mail" đều có trong câu mồi, nên luật của lần 2 lấy cả hai:

| Người dùng nói | RNNT | large-v3 | Bản ghép (C0) | Đúng? |
|---|---|---|---|---|
| giá vàng SJC | giá vàng ét giê xê | giá vàng SJC | giá vàng SJC | Đúng |
| mai đi khám | mai đi khám | Mail đi khám | Mail đi khám | Sai |

F đếm trên tập học xem mỗi khi large-v3 viết ra một từ của câu mồi (mà RNNT không có), người dùng có thật sự nói từ đó
không. Bảng dưới là số minh hoạ cách tính (số đo thật ở mục 4.7):

| Từ | Số lần viết ra | Số lần đúng | Độ tin cậy | Cách xử lý (với θ = 0,5) |
|---|---|---|---|---|
| SJC | 20 | 19 | (19 + 1) / (20 + 2) = 0,91 | Lấy như cũ |
| Zalo | 30 | 27 | (27 + 1) / (30 + 2) = 0,88 | Lấy như cũ |
| Mail | 15 | 3 | (3 + 1) / (15 + 2) = 0,24 | Chỉ lấy khi có trong từ điển ngữ cảnh của mẫu |
| list | 10 | 2 | (2 + 1) / (10 + 2) = 0,25 | Chỉ lấy khi có trong từ điển ngữ cảnh của mẫu |

Với câu "mai đi khám" ở mẫu không có app email và trợ lý không nhắc tới mail, "Mail" không có trong từ điển ngữ cảnh, nên
bản ghép giữ chữ RNNT "mai".

### 4.3. Giải pháp và thuật toán

**Đếm.** Trên tập học, với mỗi từ ngoại `w` của câu mồi (đã chuẩn hoá), đếm:

- `written(w)`: số câu mà chữ large-v3 (có câu mồi, biến thể thắng của E) có `w`, và chữ RNNT cùng câu **không** có `w`.
Điều kiện thứ hai giữ đúng tình huống phép ghép phải quyết định: chỉ khi RNNT không có từ đó thì phép ghép mới cân nhắc
lấy từ large-v3.
- `correct(w)`: trong số câu đó, số câu mà đáp án có `w` (so sau khi chuẩn hoá chữ, kể cả cách viết tương đương của lần 2).

Độ tin cậy là `(correct(w) + 1) / (written(w) + 2)`.

**Công thức này từ đâu ra.** Tỉ lệ thô `correct / written` không dùng được với từ ít gặp: từ gặp 1 lần và đúng 1 lần có
tỉ lệ 1 (tin tuyệt đối), từ gặp 1 lần và sai có tỉ lệ 0, và từ chưa gặp thì chia cho 0. Cộng 1 vào số lần đúng và 2 vào số
lần viết ra tương đương với việc giả sử trước khi đếm đã thấy mỗi từ **một lần đúng và một lần sai**. Đây là **quy tắc
kế tiếp của Laplace** (rule of succession, Laplace đưa ra năm 1814), còn gọi là làm mượt Laplace hay làm mượt cộng một.
Theo xác suất thống kê, công thức này là ước lượng trung bình của tỉ lệ đúng khi chưa biết gì về tỉ lệ đó (mọi tỉ lệ từ 0
tới 1 coi như có khả năng như nhau, tức phân phối Beta(1, 1)) rồi cập nhật bằng số lần đúng và sai đã đếm.

Tác dụng: từ đếm được nhiều lần thì độ tin cậy gần tỉ lệ thô (19/20 = 0,95, công thức cho 0,91); từ ít gặp bị kéo về 0,5
(1/1 thành 0,67, 0/1 thành 0,33); từ chưa gặp có độ tin cậy 0,5, tức "chưa biết". Hai con số 1 và 2 là lựa chọn chuẩn của
quy tắc này, không dò theo dữ liệu.

**Lọc.** Cách lọc mới `context_reliable` cho bước 3b của phép ghép:


| Loại từ tiếng Anh mới                          | `context_other` (lần 2)        | `context_reliable`             |
| ---------------------------------------------- | ------------------------------ | ------------------------------ |
| Từ của câu mồi, độ tin cậy từ ngưỡng θ trở lên | nhận                           | nhận                           |
| Từ của câu mồi, độ tin cậy dưới θ              | nhận                           | chỉ nhận khi có trong ngữ cảnh |
| Từ ngoài câu mồi                               | chỉ nhận khi có trong ngữ cảnh | chỉ nhận khi có trong ngữ cảnh |


**Chọn θ bằng kiểm định chéo trên tập học.** Chia tập học thành 5 phần theo mẫu. Với mỗi phần: đếm độ tin cậy trên 4
phần còn lại, rồi chấm bản ghép `context_reliable` trên phần đó với từng θ trong các giá trị 0,3; 0,4; 0,5; 0,6; 0,7;
0,8. Chọn θ có độ chính xác quyết định đoạn trung bình cao nhất qua 5 phần. Sau đó đếm lại độ tin cậy trên cả tập học với
θ đã chọn. Tập dò không dùng để chọn θ, vì tập dò để dành cho bước chọn chung (mục 7).

**Code.**

- `lifemate/eval/asr_v3_analysis.py` › `count_prompt_words(utterances, prompt_words)`: nhận các câu đã tách vùng
(`utterance_regions`) và tập từ của câu mồi; trả về từ điển `từ đã chuẩn hoá → (written, correct)`.
`reliability(counts)` trả về từ điển `từ đã chuẩn hoá → độ tin cậy`.
- `lifemate/eval/asr_v3_analysis.py` › `f_decider(context, prompt_words, reliability, theta)`: hàm quyết định cho từng
vùng theo `context_reliable`. Cách lọc này nằm trong code phân tích, chưa thêm vào `CONTEXT_POLICIES` của
`lifemate/eval/asr_merge.py`, vì F không thắng (mục 7).
- `lifemate/eval/asr_v3_report.py` › `stage_select`: kiểm định chéo chọn θ và ghi bảng độ tin cậy vào `select.json` (không
lên git, vì số đếm lấy từ dữ liệu thi).

### 4.4. Lý do

- **Đếm và chọn θ trên tập học, chọn θ bằng kiểm định chéo.** Nếu đếm và chấm θ trên cùng các câu, bảng độ tin cậy khớp
sẵn với chính các câu dùng để chọn; kiểm định chéo chấm mỗi phần bằng bảng đếm trên các phần khác.
- **Đếm với đúng cấu hình câu mồi sẽ dùng.** Độ tin cậy của một từ phụ thuộc câu mồi: cùng từ "Mail" có thể bị kéo ra
nhiều hơn khi câu mồi có nhiều câu về email. Vì vậy đếm trên chữ large-v3 với biến thể thắng của E.
- **Chỉ đếm khi RNNT không có từ đó.** Khi RNNT cũng có từ đó, phép ghép không lấy gì từ large-v3, nên các lần đó không
nói gì về quyết định cần học.

### 4.5. Dữ liệu và số lần đo

- Đếm và chọn θ: tập học (793 mẫu, 2 569 câu), chữ RNNT và chữ large-v3 E2, phiên K2.
- F với θ đã chọn vào bảng chọn chung trên `asr_tune` (mục 7); F không thắng nên không đo trên val.

### 4.6. Chỉ số và tiêu chí

Chọn θ: độ chính xác quyết định đoạn trung bình qua 5 phần của tập học (mục 4.3). Báo kèm cho từng θ: `merge_wrong_english`
tách theo từ của câu mồi, `merge_missed_english`. Tiêu chí so F với các ứng viên khác: bảng chọn chung (mục 7).

### 4.7. Kết quả

**Bảng độ tin cậy trên tập học** (chữ RNNT và chữ large-v3 E2 của 793 mẫu, phiên K2; các từ của câu mồi có số lần viết
ra từ 3 trở lên):

| Từ | Số lần large-v3 viết ra khi RNNT không có | Số lần đúng | Độ tin cậy |
|---|---|---|---|
| SJC | 18 | 18 | 0,95 |
| RON | 15 | 15 | 0,94 |
| SMS | 17 | 16 | 0,89 |
| voucher | 5 | 5 | 0,86 |
| workshop | 4 | 4 | 0,83 |
| tivi | 14 | 12 | 0,81 |
| meeting | 8 | 7 | 0,80 |
| VTV, MinhMart, Hale | 3 | 3 | 0,80 |
| check | 11 | 9 | 0,77 |
| robot | 10 | 8 | 0,75 |
| Sarah, HTV, chill | 7 / 4 / 4 | 5 / 3 / 3 | 0,67 |
| Foods | 6 | 4 | 0,63 |
| Zalo | 8 | 5 | 0,60 |
| demo, MinhPoint | 3 | 2 | 0,60 |
| file | 5 | 3 | 0,57 |
| Mail | 8 | 4 | 0,50 |
| email | 7 | 3 | 0,44 |
| list | 8 | 3 | 0,40 |
| tour | 3 | 1 | 0,40 |
| OK | 3 | 0 | 0,20 |

**Tập học, kiểm định chéo 5 phần, chọn θ** (độ chính xác quyết định đoạn, trung bình qua 5 phần; C0 không có tham số nên
tính trên cả tập học):

| Cách | C0 | θ = 0,3 | θ = 0,4 | θ = 0,5 | θ = 0,6 | θ = 0,7 | θ = 0,8 |
|---|---|---|---|---|---|---|---|
| Độ chính xác quyết định đoạn | 78,8% | 79,1% | **79,3%** | 78,8% | 77,8% | 77,8% | 76,1% |

θ thắng là **0,4**: từ của câu mồi có độ tin cậy dưới 0,4 chỉ được nhận khi có trong từ điển ngữ cảnh của mẫu. Trong
bảng trên chỉ có OK (0,20); ngoài bảng là các từ ít gặp mà lần nào viết ra cũng sai (ví dụ viết ra 1 lần, sai 1 lần, độ
tin cậy 0,33). List và tour đúng bằng 0,40 nên vẫn được nhận như C0. Kết quả của F trên tập dò và val nằm trong bảng chọn chung (mục 7).

**Ví dụ thật** (tập dò). "OK" có độ tin cậy 0,20 (dưới θ = 0,4), nên F chỉ nhận khi "ok" có trong từ điển ngữ cảnh của
mẫu:

| Mẫu | Người dùng nói | RNNT | large-v3 | C0 | F | Ai đúng |
|---|---|---|---|---|---|---|
| `train_0192`, lượt 0 | Ngày mai có cái hẹn nào không? | ngày mai có kế hẹn nó không | **Ok, may gọi** kể hết nào không? | lấy "Ok, may gọi" | giữ "ngày mai có" | F |
| `train_0417`, lượt 4 | Ok, được rồi, cảm ơn nhé. | thuốc cây được rồi cảm ơn nhé | **Ok,** được rồi, cảm ơn nhé. | lấy "Ok," | giữ "thuốc cây" | C0 |

Hai ví dụ cho thấy vì sao F chỉ hơn C0 rất ít: lọc "OK" sửa được câu đầu nhưng làm hỏng câu sau, nơi người dùng nói "ok"
thật.

### 4.8. Phân tích

- **Bảng độ tin cậy khớp với dự đoán ở mục 4.2.** Các từ không giống âm tiết Việt nào (SJC, RON, SMS, voucher) gần như
  luôn đúng khi large-v3 viết ra. Các từ trông như chữ Việt (Mail giống "mai", list giống "lịch", OK) đúng một nửa hoặc ít
  hơn.
- **Nhưng các từ bị lọc xuất hiện quá ít để đổi kết quả nhiều.** OK chỉ được viết ra 3 lần trên cả tập học; các từ dưới
  ngưỡng còn lại mỗi từ 1–2 lần. Trên tập dò, F giảm từ câu mồi bị lấy nhầm từ 20 xuống 17 và bỏ thêm 2 từ đúng; độ chính xác quyết định đoạn
  tăng 1,1 điểm (70,7% lên 71,8%), không có ý nghĩa thống kê.
- **θ cao hơn làm tệ đi** (θ từ 0,6 trở lên): nhiều từ đúng như Zalo (0,60) và Sarah (0,67) bắt đầu bị đòi ngữ cảnh, và
  các từ này thường không có trong ngữ cảnh.

### 4.9. Kết luận

F đúng hướng nhưng lợi ích nhỏ: các từ câu mồi hay bị viết nhầm hiếm gặp, nên lọc riêng chúng chỉ sửa được vài từ. F không hơn C0 có ý nghĩa thống kê trong bảng chọn chung (mục 7). Bảng độ tin cậy vẫn có ích làm đặc trưng
của H, và cho thấy nên viết lại câu mồi cố định bỏ các từ OK, list, tour ở lần sau.

### 4.10. Giới hạn

- Từ của câu mồi ít gặp trên tập học có độ tin cậy gần 0,5 do làm mượt, nên θ = 0,5 là ranh giới nhạy: θ dưới 0,5 nhận
mọi từ chưa gặp, θ trên 0,5 đòi ngữ cảnh cho mọi từ chưa gặp.
- Câu mồi E2 có phần riêng thay đổi theo mẫu; bảng độ tin cậy chỉ đếm từ của phần cố định, vì từ của phần riêng luôn có
trong ngữ cảnh.

## 5. Thử nghiệm G: chấm điểm hai phương án bằng âm thanh

### 5.1. Vấn đề

Mọi quyết định ghép ở lần 2 và ở F nhìn vào **chữ** của hai bản rồi áp một luật: từ có trong câu mồi không, có trong từ
điển ngữ cảnh không, đoạn có dài quá `2 × (số từ RNNT) + 2` không. Câu hỏi thật lại là câu hỏi về **âm thanh**: "người
dùng có nói 'Zalo' ở chỗ này không?". Chỉ nhìn chữ thì phải đoán bằng luật, và mỗi luật kèm một con số tự đặt. Ngưỡng độ
dài là ví dụ: số đặt tay trước khi đo, và trên thực tế chỉ chặn 1 trong 518 đoạn (mục 6.8 của báo cáo lần 2). H học
trọng số từ dữ liệu thay cho luật viết tay, nhưng nếu không có G thì các đặc trưng của H vẫn chủ yếu là tín hiệu về mặt
chữ.

### 5.2. Ý tưởng

Hỏi thẳng mô hình âm thanh. Ở mỗi đoạn khác nhau có hai phương án: A (chữ RNNT) và B (chữ large-v3). Dựng hai câu hoàn
chỉnh chỉ khác nhau ở đoạn đó, rồi cho whisper-large-v3 **không câu mồi** tính xác suất "câu này là lời của đoạn âm thanh
này" cho từng câu. Chọn câu có xác suất cao hơn. Cách này không có ngưỡng tự đặt, vì đây là so sánh hai xác suất do cùng
một mô hình tính trên cùng âm thanh.

### 5.3. Giải pháp và thuật toán

**large-v3 chép lời thế nào** (để hiểu điểm âm thanh lấy từ đâu):

1. Âm thanh 16 kHz được cắt thành cửa sổ 30 giây và đổi thành phổ log-mel (128 dải tần mỗi 10 ms).
2. **Bộ mã hoá** (encoder, 32 lớp transformer) đọc phổ của cả cửa sổ một lần và cho ra 1 500 vector, mỗi vector mô tả
   20 ms âm thanh.
3. **Bộ giải mã** (decoder) sinh chữ từng token một. Đầu vào của decoder là: câu mồi (nếu có, đặt sau token
   `<|startofprev|>`), rồi các token mở đầu `<|startoftranscript|><|vi|><|transcribe|>`, rồi các token đã sinh. Ở mỗi bước,
   decoder nhìn các vector của encoder và các token trước, trả về xác suất cho mọi token trong bộ từ vựng; giải mã tham lam
   lấy token có xác suất cao nhất. Dừng khi gặp token kết thúc `<|endoftext|>`.
4. Cấu hình thắng lần 2 có thêm VAD (bỏ khoảng lặng trước khi mã hoá), câu mồi `en_full_v1`, lớp bảo vệ g4 (câu ra chép lại
   cụm 4 từ của câu mồi thì chép lại không câu mồi) và bộ lọc câu bịa (`asr_sanitizer.py`).

Câu mồi chỉ là token đứng trước, nên câu mồi đổi xác suất ở **mọi** bước giải mã: "Mail" có trong câu mồi thì xác suất
viết "Mail" tăng, dù âm thanh là "mai".

**Điểm âm thanh tính thế nào.** Không sinh chữ mới. Đưa vào decoder đúng chuỗi token của câu ứng viên (cách gọi là "ép
chuỗi", teacher forcing), **không có câu mồi**, rồi đọc xác suất mà decoder gán cho từng token của chuỗi đó. Điểm của câu
là tổng log xác suất các token, kể cả token kết thúc. Ví dụ với câu "nhắn Zalo cho mẹ":

| Token | nhắn | Z | alo | cho | mẹ | kết thúc |
|---|---|---|---|---|---|---|
| Xác suất decoder gán | 0,9 | 0,6 | 0,95 | 0,9 | 0,8 | 0,9 |

Điểm = log 0,9 + log 0,6 + … + log 0,9. Câu "nhắn giá lô cho mẹ" được chấm cùng cách. Câu nào khớp âm thanh hơn thì
các token có xác suất cao hơn, điểm gần 0 hơn. Câu bịa dài ("Mời các bạn thử chọn…") gồm nhiều token không khớp âm thanh
nên điểm thấp, không cần ngưỡng độ dài. Số trong bảng là minh hoạ cách cộng, không phải số đo.

**Vì sao dùng log, và vì sao điểm càng gần 0 thì câu càng khớp.**

- Xác suất của cả câu bằng **tích** xác suất của từng token (xác suất token đầu, nhân xác suất token thứ hai khi đã có
  token đầu, và cứ thế). Log biến tích thành **tổng**: log(a × b) = log a + log b. Vì vậy điểm của câu, tức tổng log xác
  suất các token, chính là log của xác suất cả câu.
- Xác suất nằm trong khoảng 0 tới 1, nên log của xác suất luôn nhỏ hơn hoặc bằng 0: log 1 = 0, log 0,9 ≈ −0,11, log 0,5
  ≈ −0,69, log 0,1 ≈ −2,3, log 0,01 ≈ −4,6. Câu mà mọi token đều được mô hình chắc chắn (xác suất gần 1) có điểm gần 0;
  chỉ một token không khớp (xác suất 0,01) đã trừ 4,6.
- Ví dụ hai token: 0,9 × 0,9 = 0,81, điểm −0,11 + (−0,11) = −0,22; 0,9 × 0,1 = 0,09, điểm −0,11 + (−2,3) = −2,41.
- Lý do kỹ thuật: nhân vài chục số nhỏ hơn 1 cho ra số quá bé (ví dụ 0,1 mũ 40), máy tính không lưu chính xác được, còn
  tổng các log thì không gặp vấn đề này. Log giữ nguyên thứ tự: tích lớn hơn thì tổng log lớn hơn, nên so điểm cũng là so
  xác suất.
- Hiệu hai điểm cũng có nghĩa rõ: `điểm(B) − điểm(A) = log(P(B) / P(A))`. Hiệu bằng 2,3 nghĩa là mô hình cho câu B khả năng
  cao gấp khoảng 10 lần câu A. Vì vậy H dùng hiệu điểm làm đặc trưng.

CTranslate2 (lõi của faster-whisper) có sẵn hàm `align(encoder_output, start_sequence, text_tokens, num_frames)` trả về
`text_token_probs`, tức xác suất từng token của một chuỗi chữ cho sẵn. faster-whisper 1.2.1 dùng chính hàm này để tính
`word.probability` (`transcribe.py` › `find_alignment`, với `start_sequence` là các token mở đầu, không có câu mồi). Vậy
G không cần nạp thêm mô hình và không phải chép lại cả câu: chỉ cần chạy encoder một lần cho câu nói đó và một lượt
decoder ép chuỗi cho mỗi câu ứng viên.

**Dựng câu ứng viên.** Bắt đầu từ chữ RNNT của câu. Với mỗi đoạn khác nhau có từ tiếng Anh mới (đúng các
đoạn mà H xét), dựng câu A (đoạn đó là chữ RNNT) và câu B (đoạn đó là chữ large-v3); phần còn lại của hai câu giống hệt
nhau. Cả hai câu được đưa về cùng một kiểu viết trước khi chấm: chữ thường, bỏ dấu câu, như chữ RNNT. Lý do: large-v3
quen viết hoa và có dấu câu; nếu chỉ câu B giữ kiểu viết đó thì câu B được lợi chỉ vì kiểu viết. Một câu có nhiều đoạn
khác nhau thì mỗi đoạn có một cặp câu riêng; ở mọi đoạn khác, cả A và B giữ chữ RNNT, nên các đoạn được chấm độc lập.

**Quyết định.** Lấy chữ large-v3 khi `điểm_âm_thanh(B) > điểm_âm_thanh(A)`.

**Ví dụ từng bước.** Người dùng nói "nhắn Zalo cho mẹ"; RNNT ra "nhắn giá lô cho mẹ", large-v3 (có câu mồi) ra "Nhắn Zalo
cho mẹ."

1. Phép gióng tìm ra một đoạn khác nhau: [giá lô] ↔ [Zalo].
2. Dựng hai câu ứng viên, cùng kiểu viết (chữ thường, không dấu câu): A = "nhắn giá lô cho mẹ", B = "nhắn zalo cho mẹ".
3. Chạy encoder của large-v3 một lần trên âm thanh của câu.
4. Với câu A: đưa vào decoder các token mở đầu (không có câu mồi) rồi các token của A; decoder trả về xác suất của từng
   token A, ví dụ "nhắn" 0,9, "giá" 0,05, "lô" 0,3, "cho" 0,9, "mẹ" 0,8, kết thúc 0,9. Điểm A = tổng log ≈ −4,7.
5. Với câu B, làm như trên: "nhắn" 0,9, "z" 0,6, "alo" 0,95, "cho" 0,9, "mẹ" 0,8, kết thúc 0,9. Điểm B ≈ −1,1.
6. So: −1,1 > −4,7, tức câu B khớp âm thanh hơn, nên lấy "Zalo". Hiệu điểm 3,6 nghĩa là large-v3 cho B khả năng cao gấp
   khoảng 40 lần A.

Decoder không sinh ra chữ nào ở các bước 4 và 5, chỉ đọc xác suất của chuỗi đã đưa vào, nên không thể ra câu bịa hay câu
phụ đề. Số trong ví dụ là minh hoạ, không phải số đo.

**Nếu G chấm sai thì sao.** G có thể sai, ví dụ khi large-v3 quen một cách viết tiếng Việt hơn một tên riêng lạ dù người
dùng nói tên đó. Có bốn lớp kiểm soát:

- **Đo trước khi dùng.** G chỉ được dùng nếu thắng trong bảng chọn chung (mục 7): độ chính xác quyết định đoạn phải cao
  nhất và không làm chép đúng từ tiếng Anh hay WER câu thuần Việt tệ hơn C0. Số đoạn G chấm sai được đếm riêng theo hai
  loại (lấy nhầm, bỏ sót).
- **Phương án dự phòng.** Nếu G không đạt, dùng F hoặc C0.
- **H dùng G làm một đặc trưng.** Ở H, hiệu điểm của G được cân cùng ngữ cảnh và độ tin cậy từng từ; một lần G chấm sai
  có thể bị các tín hiệu khác bù lại.
- **Đoạn hai điểm sát nhau không chốt.** Khi hiệu điểm nhỏ (dưới 2,3, tức phương án hơn chỉ có khả năng cao gấp dưới 10
  lần), I giữ cả hai phương án cho LLM thay vì chốt (mục 8.3). Phân bố hiệu điểm của các đoạn G chấm đúng và chấm sai được
  báo ở mục 5.7, để xem các lần sai có dồn vào vùng hiệu điểm nhỏ không.

**Dùng trong H và I.** `Δâm_thanh = điểm_âm_thanh(B) − điểm_âm_thanh(A)` được thêm vào đặc trưng của H. Trong H, điều
kiện độ dài cố định của lần 2 không dùng nữa, vì số từ hai phía đã là đặc trưng. I lấy đoạn không chắc theo ứng viên thắng
của bảng chọn chung (mục 8.3).

**Câu dài hơn 30 giây.** Encoder chỉ nhận một cửa sổ 30 giây. Câu nói dài hơn 30 giây vẫn được chấm trên 30 giây đầu
nhưng được đánh dấu (cờ `long`), và quyết định của G cho câu đó lấy theo F. Thực tế không câu nào có đoạn cần quyết định
dài hơn 30 giây (0 trên 968 đoạn của cả bốn tập).

**Code.**

- `lifemate/eval/candidate_scoring.py` › `candidate_pairs(primary, secondary)`: nhận chữ RNNT và chữ large-v3 của một
  câu, trả về danh sách cặp (chỉ số đoạn, câu A, câu B) đã đưa về cùng kiểu viết (`scoring_text`).
- `lifemate/backends/asr.py` › `FasterWhisperASR.score_texts(audio, texts)`: nhận âm thanh một câu và các câu ứng viên,
  trả về (tổng log xác suất, số token) của từng câu (`model.encode` một lần, `model.model.align` không câu mồi cho từng
  câu).
- `lifemate/eval/asr_v3.py` › `score_candidates` (lệnh `score`): chạy phiên K3, ghi `scores_<tập>.jsonl`.
- `lifemate/eval/candidate_scoring.py` › `decide_g(score_a, score_b)`: `True` khi lấy chữ large-v3.
  `lifemate/eval/asr_v3_analysis.py` › `g_decider(fallback)` dùng hàm này cho từng đoạn, và dùng F cho đoạn thiếu điểm
  hoặc dài hơn 30 giây.

### 5.4. Lý do

- **Chấm không câu mồi.** Phần lớn từ lấy nhầm là do câu mồi kéo hoặc do large-v3 đoán thành tiếng Anh (mục 6.8 của báo
  cáo lần 2). Khi chấm không câu mồi, lực kéo của câu mồi không còn; điểm chỉ phản ánh âm thanh và thói quen ngôn ngữ của
  chính large-v3. Câu mồi vẫn dùng khi **chép**, vì câu mồi giúp large-v3 nghĩ ra phương án "Zalo" ngay từ đầu.
- **Chấm, không chép lại.** Chép lại không câu mồi (đã bỏ khỏi kế hoạch lần 3) tốn một lượt giải mã đầy đủ và chỉ cho
  biết large-v3 không câu mồi chọn gì, không cho biết phương án của RNNT được bao nhiêu điểm. Chấm ép chuỗi cho điểm của
  **cả hai** phương án, và chỉ chạy trên câu có đoạn cần quyết định.
- **Không thêm tham số.** G dùng large-v3 có sẵn; tổng vẫn 5,68B.
- **Không có con số tự đặt.** G chỉ so hai điểm.

### 5.5. Dữ liệu và số lần đo

- Chấm điểm âm thanh chạy ở phiên GPU K3 (Phụ lục B.2), sau K2, trên tập học (cho H), `asr_tune` và val.
- G vào bảng chọn chung trên `asr_tune` (mục 7); G không thắng nên không báo số trên val (điểm trên val đã chấm sẵn ở K3).

### 5.6. Chỉ số và tiêu chí

Chỉ số như F và H: độ chính xác quyết định đoạn, chép đúng từ tiếng Anh, WER câu thuần Việt, WER câu chêm, WER mọi câu,
`merge_wrong_english`, `merge_missed_english` (mục 2.2). Thêm thời gian chấm mỗi mẫu (encoder thêm và các lượt ép chuỗi)
và số câu bỏ qua vì dài hơn 30 giây.

Tiêu chí so G với các ứng viên khác: bảng chọn chung (mục 7).

### 5.7. Kết quả

Phiên K3 chấm 201 đoạn của tập dò, 486 đoạn của tập học, 178 đoạn của `val_seen` và 103 đoạn của `val_unseen`. Chấm lại 50
đoạn đầu của tập dò cho điểm giống hệt (0 đoạn khác). Không đoạn nào thuộc câu dài hơn 30 giây. Thời gian chấm thêm mỗi
mẫu trên T4: p95 0,73 giây (tập dò), 0,79 / 0,77 giây (hai tập val), tối đa 1,9 giây; bộ nhớ GPU đỉnh của bước chấm 3,4 GB.

Trên tập dò (chi tiết trong bảng chọn chung, mục 7):

| Cách | Độ chính xác quyết định đoạn | Từ câu mồi bị lấy nhầm | Từ ngoài câu mồi bị lấy nhầm | Từ câu mồi đúng bị bỏ | Từ ngoài câu mồi đúng bị bỏ |
|---|---|---|---|---|---|
| C0 (luật lần 2) | 70,7% | 20 | 17 | 0 | 23 |
| G (điểm âm thanh) | 69,0% | **4** | 60 | 11 | **13** |

**Hiệu điểm âm thanh của các đoạn G chấm đúng và chấm sai** (đoạn có nhãn; ngưỡng 2,3 là ngưỡng "không chắc" của G ở mục
8.3, tức phương án hơn có khả năng cao gấp dưới 10 lần):

| Tập | Đoạn có nhãn | Hiệu điểm dưới 2,3: đúng / sai | Hiệu điểm từ 2,3 trở lên: đúng / sai |
|---|---|---|---|
| `asr_tune` | 174 | 23 / 17 (sai 43%) | 97 / 37 (sai 28%) |
| Tập học | 406 | 48 / 31 (sai 39%) | 216 / 111 (sai 34%) |

**Ví dụ thật** (tập dò; điểm âm thanh đo ở phiên K3, câu A là phương án RNNT, câu B là phương án large-v3):

| Mẫu | Người dùng nói | Đoạn A ↔ B | Điểm A / B | C0 | G | Ai đúng |
|---|---|---|---|---|---|---|
| `train_0144`, lượt 0 | Mai anh có hẹn gì trong lịch không? | lịch ↔ list | −8,1 / −9,7 | lấy "list" (từ câu mồi) | giữ "lịch" | G |
| `train_0166`, lượt 0 | Tìm giúp em số của Lan đồng nghiệp với. | lan ↔ Line | −13,6 / −9,6 | giữ "lan" ("line" không có trong ngữ cảnh) | lấy "Line" | C0 |

Ở ví dụ đầu, chấm không câu mồi thì "list" không còn được câu mồi kéo, và "lịch" khớp âm thanh hơn. Ở ví dụ sau, large-v3
tự nghe "Lan" thành "Line" và vẫn chấm "Line" cao hơn, đúng loại lỗi "từ large-v3 tự bịa" ở mục 5.8.

### 5.8. Phân tích

- **G sửa đúng loại lỗi mà G nhắm tới, và gây ra loại lỗi khác.** Từ của câu mồi bị lấy nhầm giảm từ 20 xuống 4: khi chấm
  không câu mồi, large-v3 không còn bị "Mail" kéo, nên nghe ra "mai". G cũng cứu được 10 từ đúng ngoài câu mồi mà luật C0
  bỏ (23 xuống 13), vì luật đòi các từ này có trong ngữ cảnh còn G thì không.
- **Nhưng G lấy nhầm 60 từ ngoài câu mồi, so với 17 của C0.** Đây là các từ large-v3 tự bịa ("Toyota" cho "tôi ra đê"):
  large-v3 không câu mồi vẫn thấy chuỗi tiếng Anh gần âm đó khớp âm thanh hơn chữ Việt đúng của RNNT. Điểm âm thanh của
  large-v3 mang thói quen ngôn ngữ của chính large-v3 (rủi ro đã ghi ở mục 5.10), và với loại từ này thói quen đó sai.
- **Các lần G sai không dồn hết vào vùng hiệu điểm nhỏ.** Ở đoạn có hiệu điểm dưới 2,3, G sai nhiều hơn (43% trên tập dò),
  nhưng phần lớn lần sai (37 trên 54) có hiệu điểm lớn, tức large-v3 chấm sai mà vẫn "chắc". Vì vậy giữ hai phương án ở
  vùng hiệu điểm nhỏ chỉ cứu được một phần lỗi của G.
- **Kết quả: G đứng riêng không hơn C0.** Độ chính xác quyết định đoạn 69,0% so với 70,7%, CI của chênh lệch [−13,0; +9,9].
  Hai cách sai ở hai loại từ khác nhau, nên kết hợp chúng (H) có lý.

### 5.9. Kết luận

Điểm âm thanh là tín hiệu tốt cho từ của câu mồi và cho từ đúng mà ngữ cảnh không có, nhưng là tín hiệu xấu cho từ
large-v3 tự bịa. G không dùng riêng được; hiệu điểm âm thanh có giá trị làm một đặc trưng của H (trọng số lớn thứ ba, mục
6.7).

### 5.10. Giới hạn và rủi ro

- **Điểm âm thanh của large-v3 vẫn mang thói quen ngôn ngữ của large-v3.** Decoder của large-v3 cũng là một mô hình ngôn
  ngữ; với tên lạ, large-v3 có thể chấm cao một cách viết quen hơn cách viết đúng. G chỉ so hai phương án đã có, nên không
  cứu được trường hợp cả hai đều sai.
- **Kiểu viết.** Đưa cả hai câu về chữ thường không dấu câu là kiểu viết large-v3 ít gặp; điểm tuyệt đối thấp hơn bình
  thường nhưng hai câu chịu như nhau. Cách viết số khác nhau ("3" và "ba") không thuộc phạm vi vì đoạn không có từ tiếng
  Anh mới thì không chấm.
- **Tách token khác nhau.** "Zalo" và "giá lô" có số token khác nhau. Tổng log xác suất của cả câu (kể cả token kết thúc)
  là xác suất của cả chuỗi nên so được, nhưng mô hình có thể hơi nghiêng về câu ít token; hiện tượng này sẽ thấy ở
  `merge_missed_english` và `merge_wrong_english`.
- **Tất định.** CTranslate2 chấm ép chuỗi tất định trên cùng máy và cùng kiểu số; phiên K3 chấm lại 50 đoạn, điểm giống hệt.
- **Câu dài hơn 30 giây** chỉ được chấm trên 30 giây đầu và quyết định theo F (mục 5.3); trên dữ liệu đã chạy không có câu nào như vậy.

## 6. Thử nghiệm H: bộ phân loại cho quyết định ghép

### 6.1. Vấn đề

Quy tắc ghép hiện tại là luật viết tay: lấy đoạn của large-v3 khi có từ tiếng Anh mới, đoạn không quá dài, và từ đó qua
bộ lọc. Mỗi lần thêm một tín hiệu (ngữ cảnh, độ tin cậy) lại phải viết thêm một nhánh luật và chọn tay cách kết hợp. Có
những tín hiệu khác chưa dùng: điểm âm thanh của hai phương án (G), large-v3 chắc chắn tới đâu (`avg_logprob`),
đoạn của RNNT có phải âm tiết Việt có nghĩa không.

### 6.2. Ý tưởng

Coi mỗi đoạn khác nhau là một câu hỏi có/không: "lấy chữ của large-v3 thì đoạn này đúng hơn không?". Câu trả lời đúng tính
được máy móc trên tập học bằng cách so với đáp án. Vậy có thể **học** cách kết hợp các tín hiệu thay vì viết tay. Mô hình
học là hồi quy logistic: mỗi tín hiệu có một trọng số, cộng lại rồi đổi thành xác suất. Mô hình nhỏ như vậy học được từ
vài trăm đoạn, và trọng số đọc được (biết tín hiệu nào quan trọng).

### 6.3. Giải pháp và thuật toán

**Mẫu học.** Mỗi đoạn khác nhau mà large-v3 có ít nhất một từ tiếng Anh mới (đúng các đoạn mà bản ghép không lọc `all` cân nhắc
lấy). Các đoạn khác giữ chữ RNNT như cũ.

**Nhãn tự động.** Với mỗi đoạn, gióng chữ ghép với đáp án của câu và đếm số lỗi (thêm, xoá, thay) rơi vào đoạn đó trong
hai trường hợp: giữ RNNT và lấy large-v3. Nhãn là 1 khi lấy large-v3 ít lỗi hơn, 0 khi nhiều lỗi hơn. Đoạn hoà (cùng số
lỗi) được bỏ khỏi dữ liệu học, vì không có câu trả lời đúng.

**Đặc trưng** (mỗi đoạn một vector số):


| Đặc trưng                                          | Giá trị                        | Lấy từ                                |
| -------------------------------------------------- | ------------------------------ | ------------------------------------- |
| Có từ tiếng Anh mới trong từ điển ngữ cảnh         | 0/1                            | `sample_context_words` (lần 2)        |
| Có từ tiếng Anh mới thuộc câu mồi                  | 0/1                            | Từ của phần cố định `en_full_v1`      |
| Độ tin cậy thấp nhất của các từ câu mồi trong đoạn | 0–1; 1 nếu không có từ câu mồi | Bảng của F                            |
| `avg_logprob` của câu (trung bình theo số token các đoạn faster-whisper) | số âm | Trường `scores` |
| `no_speech_prob` của câu (lớn nhất trong các đoạn) | 0–1                            | Trường `scores`                       |
| Số từ phía RNNT, số từ phía large-v3               | số nguyên                      | Phép gióng                            |
| Tỉ lệ từ phía RNNT là âm tiết Việt có dấu          | 0–1                            | Quy tắc tìm từ ngoại (lần 2)          |
| Câu là lượt cuối hay lượt trong lịch sử            | 0/1                            | `hyps.jsonl`                          |
| `Δâm_thanh`: điểm âm thanh câu B trừ câu A         | số                             | Thử nghiệm G (phiên K3)               |


**Học.** Hồi quy logistic có phạt L2, chuẩn hoá đặc trưng về trung bình 0 và độ lệch 1 trên tập học. Hệ số phạt chọn bằng
kiểm định chéo 5 phần **theo mẫu** trên tập học (các đoạn của cùng một mẫu luôn nằm cùng một phần), trong các giá trị
0,01; 0,1; 1; 10. Cài bằng Python thuần (phương pháp Newton, 50 vòng), không thêm thư viện, chạy tất định. Phần kiểm
định chéo chia mẫu theo `fold_of` (CRC32 của mã mẫu chia 5), nên cách chia cố định.

**Quyết định.** Lấy chữ của large-v3 khi xác suất từ 0,5 trở lên (ngưỡng tự nhiên của hồi quy logistic, không dò). Điều kiện độ dài cố định của lần 2 (đoạn large-v3 không dài quá gấp đôi đoạn RNNT cộng 2 từ) không
dùng nữa: số từ hai phía đã là đặc trưng, và điều kiện đó chỉ chặn 1 trong 518 đoạn ở lần 2.

**Code.**

- `lifemate/eval/asr_v3_analysis.py` › `utterance_regions(rnnt_rows, v3_rows, scores)`: tách mỗi câu thành các đoạn
cần quyết định, kèm nhãn tự động (1, 0 hoặc `None` khi hoà) và điểm âm thanh của G.
- `lifemate/eval/asr_v3_analysis.py` › `region_features(u, region, k, context, prompt_words, reliability)`: vector đặc
trưng của một đoạn, theo thứ tự `FEATURES`.
- `lifemate/eval/asr_v3_analysis.py` › `fit_logistic(x, y, l2)`: trả về `LogisticModel` (trung bình, độ lệch, trọng
số). `predict(model, features)`: xác suất. `h_decider(model, context, prompt_words, reliability)`: hàm quyết định cho
từng đoạn.
- `lifemate/eval/asr_merge.py` › `split_regions`, `candidate_regions`, `assemble`: tách hai bản chữ thành đoạn khớp và
đoạn khác nhau, rồi ghép lại theo một hàm quyết định, để luật và bộ phân loại dùng chung một code gióng. Với luật
`rule_decision`, chữ ghép giống hệt lần 2 từng byte.
- `lifemate/eval/asr_v3_report.py` › `stage_select`: chọn hệ số phạt, học trên cả tập học, ghi trọng số vào `select.json`.

### 6.4. Lý do

- **Hồi quy logistic, không phải MLP hay attention.** Tập học thực tế chỉ có 486 đoạn, trong đó 406 đoạn có nhãn (mục
6.7). Mô hình nhiều tham số sẽ học thuộc tập học. Hồi quy logistic có 10 trọng số và một hệ số chặn, đọc được, và có thể
chuyển về luật nếu cần.
- **Bỏ đoạn hoà.** Đoạn hoà thường là hai cách viết cùng sai hoặc cùng đúng một từ; ép nhãn cho chúng chỉ thêm nhiễu.
- **Kiểm định chéo theo mẫu.** Các câu cùng mẫu cùng người nói và cùng ngữ cảnh; chia theo đoạn sẽ cho kết quả kiểm định
lạc quan.
- **Dùng điểm của G, không chép lại large-v3 không câu mồi.** Chép lại không câu mồi chỉ cho biết large-v3 không câu mồi
chọn gì và tốn một lượt giải mã đầy đủ; điểm âm thanh của G cho điểm của cả hai phương án với chi phí thấp hơn (mục 5.4).

### 6.5. Dữ liệu và số lần đo

- Học: tập học (phiên K2: RNNT, large-v3 E2; phiên K3: điểm của G; bảng độ tin cậy của F).
- H vào bảng chọn chung trên `asr_tune` (mục 7); H không thắng nên không báo số trên val.

### 6.6. Chỉ số và tiêu chí

Chỉ số chính: WER câu thuần Việt và chép đúng từ tiếng Anh của bản ghép. Chỉ số đi kèm: `merge_wrong_english`,
`merge_missed_english`, độ chính xác quyết định đoạn trong kiểm định chéo trên tập học, trọng số của
từng đặc trưng.

Tiêu chí so H với các ứng viên khác: bảng chọn chung (mục 7).

### 6.7. Kết quả

**Dữ liệu học:**

| | Số đoạn | Nhãn 1 (lấy large-v3 đúng hơn) | Nhãn 0 | Đoạn hoà (bỏ) |
|---|---|---|---|---|
| Tập học | 486 | 226 | 180 | 80 |
| `asr_tune` | 201 | 174 đoạn có nhãn | | 27 |

**Chọn hệ số phạt bằng kiểm định chéo 5 phần theo mẫu trên tập học** (độ chính xác quyết định đoạn): 81,0% (0,01), 81,0%
(0,1), 81,3% (1), **81,5% (10)**. Hệ số phạt thắng là 10.

**Trọng số** (học trên cả tập học, đặc trưng đã chuẩn hoá về trung bình 0, độ lệch 1; dương là nghiêng về lấy large-v3):

| Đặc trưng | Trọng số |
|---|---|
| Có từ tiếng Anh mới thuộc câu mồi | +0,82 |
| Có từ tiếng Anh mới trong từ điển ngữ cảnh | +0,80 |
| Hiệu điểm âm thanh (G) | +0,63 |
| `avg_logprob` của large-v3 | +0,47 |
| Câu là lượt cuối | +0,15 |
| Độ tin cậy thấp nhất của từ câu mồi trong đoạn (F) | +0,09 |
| `no_speech_prob` | +0,08 |
| Số từ phía large-v3 | −0,25 |
| Tỉ lệ từ phía RNNT là chữ Việt | −0,28 |
| Số từ phía RNNT | −0,31 |
| Hệ số chặn | +0,39 |

**Tập dò** (chênh lệch so với C0 kèm CI 95%; đủ bảng ở mục 7):

| Cách | Độ chính xác quyết định đoạn | Chép đúng từ tiếng Anh | WER câu thuần Việt | WER mọi câu | Từ lấy nhầm (câu mồi / ngoài) | Từ đúng bị bỏ (câu mồi / ngoài) |
|---|---|---|---|---|---|---|
| C0 | 70,7% | 72,1% | 17,8% | 19,7% | 20 / 17 | 0 / 23 |
| H | **76,4%** [0,0; +11,7] | 71,6% [−2,9; +2,0] | **17,6%** [−0,28; −0,03] | **19,5%** [−0,34; −0,02] | 12 / 17 | 5 / 21 |

**Ví dụ thật** (tập dò; xác suất "nên lấy large-v3" của bộ phân loại, lấy khi từ 0,5 trở lên):

| Mẫu | Người dùng nói | Đoạn RNNT ↔ large-v3 | C0 | H (xác suất) | Ai đúng |
|---|---|---|---|---|---|
| `train_0310`, lượt 0 | Ngoài trời bây giờ nóng không ạ? | ngoài trời ↔ Voucher | lấy "Voucher" (từ câu mồi) | giữ "ngoài trời" (0,21) | H |
| `train_1153`, lượt 2 | Mẹ mới nhắn Zalo gì vậy, đọc mình nghe. | cho tôi vì phải ↔ Zalo gì vậy? | lấy "Zalo gì vậy?" | giữ RNNT (0,30) | C0 |

Ở ví dụ đầu, "Voucher" thay cho hai từ Việt hợp lệ và điểm âm thanh nghiêng về RNNT, nên H giữ RNNT. Ở ví dụ sau, "Zalo"
là từ câu mồi đáng tin, nhưng đoạn dài và phía RNNT toàn là chữ Việt hợp lệ, nên H nghiêng sai về RNNT.

### 6.8. Phân tích

- **H học đúng các tín hiệu đã biết.** Hai trọng số lớn nhất là "từ thuộc câu mồi" và "từ có trong ngữ cảnh", tức chính
  hai điều kiện của luật C0. Tín hiệu mới đứng thứ ba là hiệu điểm âm thanh của G. Đoạn càng dài (cả hai phía) và phía
  RNNT càng là chữ Việt hợp lệ thì H càng nghiêng về giữ RNNT, đúng như ca "Toyota" cho "tôi ra đê".
- **Độ tin cậy từng từ của F gần như không được dùng** (trọng số +0,09): thông tin đó đã nằm trong "từ thuộc câu mồi" và
  hiệu điểm âm thanh.
- **H là cách tốt nhất trên tập dò, nhưng chưa đủ chắc để thắng theo quy tắc đặt trước.** Độ chính xác quyết định đoạn
  cao hơn C0 5,7 điểm và WER câu thuần Việt giảm có ý nghĩa thống kê (−0,14 điểm). Nhưng CI của chênh lệch độ chính xác
  là [0,0; +11,7]: cận dưới đúng bằng 0, nên theo quy tắc chọn (mục 7) H chưa hơn C0 có ý nghĩa thống kê. Tập dò chỉ có
  174 đoạn có nhãn, nên CI rộng.

### 6.9. Kết luận

Bộ phân loại kết hợp được điểm âm thanh với các luật cũ và tốt nhất trên tập dò, nhưng chưa thắng C0 một cách chắc chắn
với số đoạn hiện có. Theo quy tắc đặt trước, H không được chọn ở lần 3. Đây là ứng viên tốt nhất để thử lại khi có thêm
dữ liệu có nhãn (ví dụ gộp tập học và tập dò rồi đo bằng kiểm định chéo), vì độ chính xác trong kiểm định chéo trên tập học
cũng cao hơn C0 (81,5% so với 78,8%).

### 6.10. Giới hạn

- Nhãn tự động dựa vào đáp án của BTC và cách gióng từ; một đoạn có thể đúng về ý (LLM vẫn gọi đúng tool) mà vẫn có nhãn 0
vì sai chính tả.
- Mục tiêu học là ít lỗi chữ hơn, không phải gọi đúng tool hơn. Hai mục tiêu thường cùng chiều nhưng không trùng nhau;
thử nghiệm J kiểm tra bằng điểm thi.

## 7. Bảng chọn chung: chọn cách quyết định ghép

**Vấn đề.** F, G và H cùng trả lời một câu hỏi: ở mỗi đoạn hai bản viết khác nhau, lấy chữ RNNT hay chữ
large-v3. Nếu chọn nối tiếp (F trước, rồi G và H so với cách thắng của F), tập dò bị dùng để chọn nhiều lần liên tiếp, số
trên tập dò ngày càng đẹp hơn thực tế, và không có phép so cuối cùng giữa mọi cách. Vì vậy mọi cách được đưa vào **một
bảng chung** và chọn **một lần** bằng một quy tắc chung.

**Các ứng viên.**

| Ứng viên | Quyết định bằng | Tham số và cách đặt | Chi phí lúc chạy thêm so với C0 |
|---|---|---|---|
| C0 `context_other` (lần 2, mốc) | Luật: từ của câu mồi nhận hết, từ ngoài câu mồi phải có trong từ điển ngữ cảnh | Không | 0 |
| F `context_reliable` | Như C0, nhưng từ câu mồi có độ tin cậy dưới θ phải có trong từ điển ngữ cảnh | θ chọn bằng kiểm định chéo trên tập học (mục 4.3) | Gần 0 (tra bảng) |
| G | Điểm âm thanh của large-v3 không câu mồi | Không | Encoder và chấm ép chuỗi ở câu có đoạn cần quyết định |
| H | Hồi quy logistic trên mọi đặc trưng (mục 6.3) | Trọng số học trên tập học, hệ số phạt bằng kiểm định chéo trên tập học; ngưỡng 0,5 | Như G |

**Nguyên tắc dùng dữ liệu.** Mọi tham số của mọi ứng viên được học hoặc chọn **chỉ trên tập học**. Tập dò `asr_tune`
chỉ dùng một lần ở bước này, để chọn giữa các ứng viên đã cố định. Ngưỡng 0,5 của H là ngưỡng tự nhiên của hồi quy
logistic (lấy phía có xác suất cao hơn), không dò.

**Đầu vào chung.** Mọi ứng viên dùng cùng chữ RNNT, cùng chữ large-v3 (câu mồi thắng của E) và cùng các đoạn khác nhau.
Chỉ những đoạn mà phía large-v3 có từ tiếng Anh mới được quyết định; các đoạn khác giữ chữ RNNT ở mọi ứng viên. Nhờ vậy
các ứng viên chỉ khác nhau đúng ở các quyết định cần so.

**Chỉ số chọn: độ chính xác quyết định đoạn** (`merge_decision_accuracy`). Trên các đoạn có nhãn tự động (mục 6.3: nhãn
1 khi lấy large-v3 ít lỗi hơn so với đáp án, 0 khi nhiều lỗi hơn; đoạn hoà bị bỏ), đếm tỉ lệ đoạn mà ứng viên chọn đúng
phía ít lỗi hơn. Lý do chọn chỉ số này: các ứng viên chỉ khác nhau ở vài trăm đoạn, nên WER tính trên mọi câu bị pha loãng;
độ chính xác quyết định đoạn đo thẳng thứ khác nhau đó, và gộp cả hai loại sai (lấy nhầm và bỏ sót) vào một số.

**Điều kiện chặn.** So với C0 trên tập dò, ứng viên không được tệ hơn có ý nghĩa thống kê ở chép đúng từ tiếng Anh, và
không được tệ hơn có ý nghĩa thống kê ở WER câu thuần Việt của bản ghép. Thời gian chấm thêm không quá 1 giây ở p95 mỗi
mẫu trên T4.

**Quy tắc chọn (đặt trước).**

1. Loại các ứng viên vi phạm điều kiện chặn.
2. Trong các ứng viên còn lại, lấy ứng viên có độ chính xác quyết định đoạn cao nhất trên tập dò.
3. Nếu ứng viên đó không hơn có ý nghĩa thống kê một ứng viên đơn giản hơn, chọn ứng viên đơn giản hơn. Thứ tự từ đơn giản
   tới phức tạp: C0, F, G, H (ít tín hiệu hơn, ít tham số hơn, ít thời gian chạy hơn).
4. Nếu không ứng viên nào hơn C0 có ý nghĩa thống kê, giữ C0.

**Thống kê.** Bootstrap ghép cặp theo mẫu, 1 000 vòng, seed 0, như lần 2: mỗi vòng bốc lại mẫu có hoàn lại, tính lại mọi
chỉ số trên các câu và các đoạn của mẫu được bốc. Hai ứng viên khác nhau có ý nghĩa thống kê khi CI 95% của chênh lệch
không chứa 0.

**Đo trên tập kiểm tra.** Ứng viên thắng và C0 được đo **một lần** trên `val_seen` và `val_unseen`, với mọi chỉ số. Không
chọn lại theo số của val.

**Vì sao vẫn giữ F.** F không cần chấm điểm bằng mô hình nào lúc chạy, nên là phương án dự phòng nếu G tốn quá thời gian
khi chạy hệ thống trên A30, nếu chấm ép chuỗi gặp lỗi kỹ thuật, hoặc nếu G không qua điều kiện chặn. F chỉ cần số đếm trên tập
học đã có từ phiên K2, nên không tốn thêm GPU.

**Kết quả trên tập dò** (chữ RNNT và large-v3 E2 của phiên K1; 201 đoạn cần quyết định, 174 đoạn có nhãn, 27 đoạn hoà;
chênh lệch so với C0 kèm CI 95%):

| Ứng viên | Độ chính xác quyết định đoạn | Chép đúng từ tiếng Anh | WER câu thuần Việt | WER câu chêm | WER mọi câu | Từ lấy nhầm (câu mồi / ngoài câu mồi) | Từ đúng bị bỏ | Thời gian chấm thêm p95 (s) |
|---|---|---|---|---|---|---|---|---|
| C0 | 70,7% | 72,1% | 17,76% | 23,5% | 19,69% | 20 / 17 | 23 | 0 |
| F (θ = 0,4) | 71,8% [−1,1; +3,7] | 71,6% [−1,4; 0,0] | 17,71% [−0,12; 0,00] | 23,5% [−0,1; +0,1] | 19,67% [−0,08; +0,03] | 17 / 17 | 25 | 0 |
| G | 69,0% [−13,0; +9,9] | 71,1% [−4,7; +2,8] | 17,64% [−0,33; +0,07] | 23,4% [−0,9; +0,6] | 19,57% [−0,42; +0,18] | 4 / 60 | 24 | 0,73 |
| H | **76,4%** [0,0; +11,7] | 71,6% [−2,9; +2,0] | **17,62%** [−0,28; −0,03] | 23,3% [−0,7; +0,1] | **19,52%** [−0,34; −0,02] | 12 / 17 | 26 | 0,73 |
| RNNT đứng riêng (tham khảo) | | 49,0% | 17,48% | 25,8% | 20,29% | | | |

**Áp quy tắc chọn:**

1. Không ứng viên nào tệ hơn C0 có ý nghĩa thống kê ở chép đúng từ tiếng Anh hay WER câu thuần Việt, và thời gian chấm thêm
   của G, H (0,73 giây ở p95) dưới ngưỡng 1 giây. Cả bốn ứng viên qua điều kiện chặn.
2. Độ chính xác quyết định đoạn cao nhất là H (76,4%).
3. So H với ứng viên đơn giản nhất là C0: CI của chênh lệch [0,0; +11,7] có cận dưới bằng 0, tức H không hơn C0 có ý nghĩa
   thống kê. Theo quy tắc, chọn ứng viên đơn giản hơn.

**Ứng viên thắng: C0** (luật `context_other` của lần 2, trên chữ large-v3 với câu mồi E2).

**Kết quả trên tập kiểm tra** (đo một lần; vì C0 thắng, bảng chỉ có C0 và RNNT):

| Tập | Hệ thống | Độ chính xác quyết định đoạn | Chép đúng từ tiếng Anh | WER câu thuần Việt | WER câu chêm | WER mọi câu | Bỏ sót giá trị tham số |
|---|---|---|---|---|---|---|---|
| val\_seen | RNNT | | 49,9% | 17,7% | 24,8% | 20,0% | 32,1% |
| | C0 trên E2 | 80,3% (152 đoạn có nhãn) | 75,5% | 17,9% | 21,7% | 19,1% | 30,3% |
| val\_unseen | RNNT | | 51,6% | 17,5% | 24,1% | 19,8% | 48,1% |
| | C0 trên E2 | 77,5% (80 đoạn có nhãn) | 68,7% | 17,6% | 22,0% | 19,2% | 44,4% |

**Phân tích.**

- **Ba cách mới cùng không vượt được C0 một cách chắc chắn**, nhưng vì ba lý do khác nhau. F sửa đúng loại lỗi nhưng loại
  lỗi đó quá hiếm (mục 4.8). G sửa từ câu mồi nhưng lấy nhầm nhiều từ large-v3 tự bịa (mục 5.8). H kết hợp được cả hai và
  tốt nhất, nhưng 174 đoạn có nhãn chưa đủ để CI tách khỏi 0 (mục 6.8).
- **Phần còn lại để cải thiện ở bước ghép là nhỏ.** Trên tập dò, C0 còn lấy nhầm 37 từ và bỏ 23 từ đúng trên khoảng
  400 từ tiếng Anh; phần lớn mức chép đúng đã đến từ câu mồi E2 (mục 3.7).
- **Thứ hạng giữ trên val.** Độ chính xác quyết định đoạn của C0 trên val (77–80%) còn cao hơn trên tập dò (70,7%), nên
  không có dấu hiệu tập dò khó bất thường.

**Kết luận.** Giữ luật C0 cho bước ghép. Cấu hình ASR tốt nhất của lần 3 là: RNNT + large-v3 với câu mồi E2 + luật
`context_other` của lần 2 (từ điển ngữ cảnh bỏ khoá JSON). H là hướng nên thử lại khi có thêm dữ liệu có nhãn.

**Giới hạn.** Bốn ứng viên được so trên 300 mẫu, nên ứng viên thắng trên tập dò vẫn có thể thắng một phần nhờ may; số val
đo một lần là số để tin. Độ chính xác quyết định đoạn coi lấy nhầm và bỏ sót nặng như nhau, trong khi lấy nhầm một từ
tiếng Anh có thể làm LLM gọi sai tool; hai loại sai được báo riêng (`merge_wrong_english`, `merge_missed_english`) để
thấy ứng viên thắng nghiêng về phía nào. Nhãn tự động dựa vào đáp án và cách gióng từ, nên có thể sai ở đoạn mà hai phía
cùng gần đúng.

## 8. Thử nghiệm I: giữ hai phương án cho LLM

### 8.1. Vấn đề

Ở bước ghép, mọi cách quyết định (C0, F, G, H) đều phải **chốt một bản chữ** cho mỗi đoạn hai bản viết khác nhau. Ở đoạn
mà cách quyết định không chắc, chốt nhầm thì phương án đúng bị vứt đi. Trong khi đó LLM có thông tin mà ASR không thấy:
danh sách app đã cài, catalog tool, kết quả tool (danh bạ, thiết bị), và cả câu nói.

### 8.2. Ý tưởng

Ở đoạn không chắc, **không chốt**: viết cả hai phương án vào chữ đưa cho LLM, để LLM chọn.

| Cách đưa chữ cho LLM | Chữ LLM nhận | Điều có thể xảy ra |
|---|---|---|
| Chốt một bản, chốt nhầm | nhắn giá lô cho mẹ | LLM không biết "giá lô" là gì, có thể không gọi tool nhắn Zalo |
| Giữ hai phương án | nhắn {giá lô \| Zalo} cho mẹ | LLM thấy danh sách app đã cài có Zalo, không có app nào tên "giá lô", nên có thể chọn đúng |

I có hai bước:

1. **I1, đếm trần (rẻ, không chạy LLM).** Dùng đáp án của BTC để đếm: nếu LLM **luôn chọn đúng** thì cứu được bao nhiêu
   đoạn. Đây là mức tối đa cách này có thể đạt. Nếu trần thấp thì bỏ cách này, khỏi tốn GPU chạy LLM.
2. **I2, đo thật (chạy LLM).** Nếu trần đủ cao, chạy agent trên chữ có hai phương án và chấm điểm thi, để biết LLM thật sự
   chọn đúng bao nhiêu. I2 là một nguồn chữ của thử nghiệm J (mục 9).

### 8.3. Giải pháp và thuật toán

**Đoạn không chắc.** Đoạn khác nhau mà phía large-v3 có từ tiếng Anh mới (như H), và cách quyết định thắng ở bảng chọn
chung (mục 7) không chắc:

- C0 hoặc F (luật): từ đó bị bộ lọc loại, tức phép ghép giữ chữ RNNT nhưng large-v3 có đề xuất khác;
- G: hai điểm âm thanh chênh nhau dưới 2,3, tức phương án hơn chỉ có khả năng cao gấp dưới khoảng 10 lần phương án kia
  (mục 5.3);
- H: xác suất "nên lấy large-v3" nằm trong khoảng 0,2 tới 0,8.

Ở đoạn không chắc, chữ đưa cho LLM là `{<chữ đã chốt> | <phương án còn lại>}`, phương án đã chốt đứng trước. Các đoạn khác
giữ như bản đã chốt.

**I1: đếm bằng đáp án, không cần LLM.** Với mỗi đoạn không chắc, so cả hai phương án với đáp án của câu và xếp vào một trong
bốn loại. Ví dụ với bốn đoạn:

| Người dùng nói (đáp án) | Bản chốt | Phương án còn lại | Bản chốt đúng? | Phương án còn lại đúng? | Loại |
|---|---|---|---|---|---|
| Zalo | giá lô | Zalo | Sai | Đúng | **Cứu được**: LLM chọn đúng thì sửa được lỗi |
| mai | mai | Mail | Đúng | Sai | **Rủi ro**: bản chốt đã đúng, đưa thêm "Mail" thì LLM có thể chọn nhầm |
| SJC | SJC | ét giê xê | Đúng | Sai | **Rủi ro**, như trên |
| Sarah | Sara | Sera | Sai | Sai | **Không cứu được**: cả hai đều sai |

Từ bốn loại này đếm được, trên mỗi tập:

- **Trần**: tỉ lệ chép đúng từ tiếng Anh nếu ở mọi đoạn "cứu được" LLM chọn đúng; trừ đi tỉ lệ của bản chốt là phần tối đa
  có thể cứu thêm. Trong ví dụ, trần cứu thêm được 1 đoạn.
- **Rủi ro**: số đoạn "rủi ro" (trong ví dụ là 2) và tỉ lệ câu thuần Việt có ít nhất một đoạn không chắc.
- **Chi phí**: số token thêm vào mỗi mẫu, đếm bằng tokenizer của Qwen3-4B (chỉ nạp tokenizer, không chạy mô hình).

**Code.** `lifemate/eval/asr_v3_analysis.py` › `alternatives(utterances, decide, uncertain)`: nhận các câu đã tách
đoạn, hàm quyết định đã chốt và hàm chọn đoạn không chắc; trả về chữ dạng hai phương án của từng câu và số đoạn của từng
loại (không chắc, cứu được, rủi ro, hoà). `lifemate/eval/asr_v3_report.py` › `stage_select` tính trần, CI và ghi
`j4_alternatives_<tập>.hyps.jsonl` để thử nghiệm J đọc lại. Số token thêm được đếm một lần trên máy dev bằng file
tokenizer của `Qwen/Qwen3-4B-Instruct-2507` (chỉ nạp tokenizer, không chạy mô hình), so chữ J4 với chữ J3 từng mẫu; lệnh
đếm chưa đưa vào code.

### 8.4. Lý do

- **Đếm trần trước khi chạy LLM.** Trần dùng đáp án nên tính được ngay, không tốn GPU. Trần là giới hạn trên: LLM thật
  không thể cứu nhiều hơn trần. Nếu trần thấp thì bỏ I2.
- **Chỉ đưa hai phương án ở đoạn không chắc.** Đưa ở mọi đoạn khác nhau thì số token tăng, và LLM phải chọn cả ở những chỗ
  phép ghép đã chắc, tức tăng số đoạn "rủi ro".
- **Ngưỡng 0,2–0,8 và hiệu điểm 2,3.** Ngoài các khoảng này phép quyết định đủ chắc để chốt. Hai con số đặt trước, không
  dò. Xác suất 0,8 ứng với tỉ số khả năng 4, còn hiệu 2,3 ứng với tỉ số 10; khoảng của G rộng hơn để bù việc điểm của G
  chưa hiệu chỉnh thành xác suất.

### 8.5. Dữ liệu

`asr_tune`, `val_seen`, `val_unseen`, dùng chữ sau bước chọn chung. I1 không chọn gì nên đo thẳng trên val.

### 8.6. Chỉ số và tiêu chí

Tiêu chí (đặt trước): chạy I2 khi trần trên `asr_tune` cứu thêm được từ 3 điểm phần trăm chép đúng từ tiếng Anh trở lên
so với bản chốt. Dưới mức đó, lợi ích tối đa nhỏ hơn mức chênh thường gặp giữa hai lần chạy agent, nên không đáng chạy.

### 8.7. Kết quả

Cách thắng của bảng chọn chung là C0 (luật), nên đoạn không chắc là đoạn mà bộ lọc ngữ cảnh loại, tức phép ghép giữ chữ
RNNT dù large-v3 đề xuất một từ tiếng Anh mới (mục 8.3).

| Tập | Chép đúng, bản chốt | Trần với hai phương án | Trần trừ bản chốt (CI 95%) | Đoạn không chắc | Cứu được | Rủi ro | Hoà | Câu thuần Việt có đoạn không chắc | Token thêm mỗi mẫu (mọi mẫu): p50 / p95 / tối đa |
|---|---|---|---|---|---|---|---|---|---|
| `asr_tune` | 72,1% | 76,9% | **+4,8** [+2,3; +7,6] | 88 | 23 | 50 | 15 | 28 | không đếm |
| val\_seen | 75,5% | 77,9% | +2,4 [+0,3; +4,6] | 75 | 15 | 44 | 16 | 28 | 0 / 12 / 23 |
| val\_unseen | 68,7% | 72,6% | +4,0 [+1,7; +6,8] | 58 | 16 | 24 | 18 | 13 | 0 / 10 / 20 |

"Cứu được": bản chốt giữ RNNT nhưng chữ large-v3 ít lỗi hơn. "Rủi ro": bản chốt đã ít lỗi hơn, đưa thêm phương án large-v3
thì LLM có thể chọn nhầm. "Hoà": hai phương án cùng số lỗi. Trên val, 61 trên 274 mẫu (`val_seen`) và 45 trên 178 mẫu
(`val_unseen`) có ít nhất một đoạn hai phương án; các mẫu đó thêm trung vị 7 token, tối đa 23 token.

**Ví dụ thật** (tập dò; chữ đưa cho LLM, phương án đã chốt đứng trước):

| Loại | Mẫu | Người dùng nói | Chữ dạng hai phương án |
|---|---|---|---|
| Cứu được | `train_1497`, lượt 0 | Nina Kowalski ở khách sạn trên đường Bạch Đằng… | nina {ko washi \| Kowalski} ở khách sạn trên đường bạch đằng… |
| Cứu được | `train_1379`, lượt cuối | …gọi giùm anh một tô ramen gà ở Osaka Ramen, trả bằng ví nha. | …gọi chung ếch một {toramenga \| tô ramen gà} ở {osakaramen đã \| Osaka Ramen, trả} bằng ví ga |
| Rủi ro | `train_0103`, lượt 0 | Bật đèn bếp lên giúp tôi. | {bật đèn bếp \| Bạn điền web} lên giúp tôi |
| Rủi ro | `train_0166`, lượt 0 | Tìm giúp em số của Lan đồng nghiệp với. | tìm giúp em số của {lan \| Line} đồng nghiệp với |

Ở hai dòng "cứu được", tên người và tên quán không có trong ngữ cảnh nên luật giữ chữ RNNT sai; LLM thấy cả hai cách viết
và có thể chọn bản large-v3. Ở hai dòng "rủi ro", chữ RNNT đã đúng; LLM phải bỏ qua phương án sai ("Bạn điền web",
"Line").

**Áp tiêu chí đặt trước:** trần trên `asr_tune` cứu thêm 4,8 điểm phần trăm, từ 3 điểm trở lên, nên **chạy I2** (nguồn J4
của thử nghiệm J).

Kết quả của I2 nằm ở mục 9.

### 8.8. Phân tích

- **Trần có ý nghĩa thống kê trên cả ba tập**, từ +2,4 tới +4,8 điểm chép đúng từ tiếng Anh, nếu LLM luôn chọn đúng.
- **Nhưng số đoạn "rủi ro" gấp đôi số đoạn "cứu được"** (50 so với 23 trên tập dò; 44 so với 15 trên `val_seen`). Bộ lọc
  ngữ cảnh thường loại đúng: ở phần lớn đoạn bị loại, chữ RNNT mới là chữ đúng. LLM phải chọn đúng ở các đoạn rủi ro thì
  mới không mất điểm; vì vậy chỉ trần thôi chưa đủ để quyết định, cần I2.
- **Chi phí token rất nhỏ**: phần lớn mẫu không đổi, mẫu có đoạn không chắc chỉ thêm khoảng 7 token.

### 8.9. Kết luận

Định dạng hai phương án có trần đủ cao (+4,8 điểm trên tập dò) và rẻ (vài token), nên được thử thật với LLM ở thử nghiệm J
(nguồn J4). Rủi ro chính là các đoạn mà bản chốt đã đúng chiếm phần lớn đoạn không chắc.

### 8.10. Giới hạn

- Trần giả định LLM luôn chọn đúng; I2 đo mức thật.
- LLM chưa từng thấy định dạng `{a | b}`. Chưa fine-tune thì LLM có thể chép nguyên dấu ngoặc vào tham số tool; I2 thêm
  một câu giải thích định dạng vào prompt hệ thống (mục 9.3), nên kết quả I2 là cận dưới của mức đạt được sau fine-tune.
- Ngưỡng 3 điểm phần trăm là con số đặt tay, dựa trên mức dao động giữa các lần chạy agent, chưa đo chính xác.

## 9. Thử nghiệm J: chạy agent trên chữ ASR đã lưu

### 9.1. Vấn đề

Mọi thử nghiệm E–I chấm **chữ** (WER, chép đúng từ tiếng Anh), trong khi điểm thi chấm **lời gọi tool**. Chưa biết chữ
tốt hơn thì điểm thi tăng bao nhiêu. Lần đo bằng điểm thi duy nhất hiện có là smoke v7, với ASR cũ (large-v3 WER 44%):
40,17 điểm khi dùng chữ chuẩn, 14,50 điểm khi dùng chữ ASR. Chưa có lần đo nào với RNNT hay bản ghép. Thử nghiệm I1 cũng
chỉ biết trần, chưa biết LLM có chọn đúng ở định dạng hai phương án không.

### 9.2. Ý tưởng

Chữ ASR của mọi lần chạy đã được lưu trong `hyps.jsonl`. Chạy agent của hệ thống (Qwen3-4B qua vLLM, `ToolEnv` của BTC,
chấm điểm như thi) với chữ đọc lại từ các file đó thay vì chép âm thanh. Đổi file chữ là đổi nguồn chữ, mọi thứ khác giữ
nguyên, nên chênh lệch điểm chỉ do chữ.

Không cần chạy ASR cùng lúc, nên làm được trên T4: T4 không chứa nổi vLLM cùng large-v3, nhưng chứa được vLLM đứng riêng
(lần smoke v7 đã chạy agent trên T4).

### 9.3. Giải pháp và thuật toán

**Năm nguồn chữ**, mỗi nguồn một lần chạy agent trên `val_seen` và `val_unseen`:

| Nguồn | Chữ của mọi lượt người dùng (lượt cuối và lịch sử) | Để biết |
|---|---|---|
| J0 | Chữ chuẩn của BTC (`audio_text`) | Điểm khi ASR hoàn hảo, tức trần của mọi cải tiến ASR |
| J1 | RNNT đứng riêng | Mốc |
| J2 | Bản ghép của lần 2 (`en_full_v1`, C0) | Lần 2 giúp được bao nhiêu điểm thi |
| J3 | Bản ghép thắng của lần 3 (câu mồi thắng của E, cách thắng của mục 7) | Lần 3 giúp thêm bao nhiêu |
| J4 | J3 có hai phương án ở đoạn không chắc (I2) | LLM có chọn đúng không; chỉ chạy khi I1 đạt tiêu chí (mục 8.6) |

**Đọc lại chữ.** `lifemate/eval/rescore_asr.py` › `ReplayTranscriber` nhận các dòng `hyps.jsonl` và trả về
`Transcript` của từng mẫu như một ASR thật. `lifemate/run/cli.py` có thêm `--transcripts replay --hyps <file>` (dùng
`ReplayTranscriber`) và `--set khoá=giá_trị` (đổi một khoá config cho một lần chạy); J0 dùng `--transcripts gold` đã có.
Mọi nguồn chạy cùng `configs/v0.toml`: `Qwen/Qwen3-4B-Instruct-2507` qua vLLM 0.32.0, seed 0, prompt hệ thống
`configs/prompts/system_v1.md`.

**Chỉ chạy lại mẫu có chữ khác J1.** Agent giải mã tất định, nên mẫu có chữ giống hệt J1 ở mọi lượt cho cùng lời gọi tool.
Vì vậy J0 và J1 chạy đủ mọi mẫu; J2, J3, J4 chỉ chạy các mẫu có ít nhất một lượt khác chữ J1, các mẫu còn lại dùng điểm
của J1 (`lifemate/eval/asr_v3_agent.py` › `differing_ids`, `source_scores`). Số mẫu khác J1:

| Nguồn | `val_seen` (274 mẫu) | `val_unseen` (178 mẫu) |
|---|---|---|
| J2 ghép lần 2 | 85 | 34 |
| J3 ghép lần 3 | 77 | 35 |
| J4 hai phương án | 117 | 71 |

**Prompt cho J4.** `configs/prompts/system_v1_alternatives.md` là `system_v1.md` thêm quy tắc 15: "lời người dùng có thể
chứa đoạn dạng `{A | B}`, là hai cách nghe của cùng một đoạn âm thanh. Chọn cách hợp với danh sách app đã cài, kết quả
tool và ngữ cảnh; dùng đúng chữ của cách đã chọn, không chép dấu ngoặc `{`, `}` hay dấu `|` vào tham số." J4 chạy với
`--set agent.system_prompt=configs/prompts/system_v1_alternatives.md`; J0–J3 giữ prompt như cũ.

**Chấm điểm.** Điểm thi do `ToolEnv` của BTC chấm từng mẫu, như các lần chạy agent trước (`lifemate/run/runner.py`), ghi
vào `details.jsonl`. `lifemate/eval/asr_v3_agent.py` › `summarize` (lệnh `score`) tính điểm trung bình, CI 95% và CI
của chênh lệch so với J1 bằng bootstrap ghép cặp theo mẫu (1 000 vòng, seed 0), số mẫu điểm tối đa, số mẫu được và mất
điểm so với J1, đếm loại lỗi theo trường `errors` của `details.jsonl`, và với J4 đếm số mẫu có `{`, `}` hoặc `|` lọt vào
tham số tool.

**Giải mã tất định.** Như mọi lần chạy agent: nhiệt độ 0, cùng seed, cùng phiên bản mô hình, để chênh lệch giữa các nguồn
không do ngẫu nhiên.

### 9.4. Lý do

- **Đo bằng điểm thi.** Một lỗi chữ có thể không đổi lời gọi tool nào, và ngược lại một từ sai có thể làm sai cả lời gọi.
  Chỉ điểm thi trả lời được lựa chọn ASR nào thật sự đáng giá.
- **Đọc lại chữ đã lưu.** Không phải chạy lại ASR, không cần GPU lớn, và năm nguồn chạy trên đúng cùng các mẫu với cùng
  agent.
- **Có J0.** Khoảng cách J0 − J3 là phần điểm còn mất vì ASR sau lần 3; nếu khoảng này nhỏ thì công sức tiếp theo nên dồn
  sang LLM thay vì ASR.
- **J4 có điều kiện.** Chỉ chạy khi trần của I1 đủ cao, để không tốn GPU cho một định dạng không thể giúp nhiều.

### 9.5. Dữ liệu và số lần đo

`val_seen` (274 mẫu) và `val_unseen` (178 mẫu), mỗi nguồn một lần. Không chọn tham số nào theo kết quả của J, nên không
cần tập dò. Phiên GPU K4 (Phụ lục B.2).

### 9.6. Chỉ số và tiêu chí

Chỉ số: điểm thi trung bình mỗi mẫu; số mẫu được điểm tối đa; số mẫu được và mất điểm so với J1; số lỗi theo loại; với J4
thêm số mẫu có dấu ngoặc `{`, `}` hoặc `|` lọt vào tham số tool.

J không chọn gì trong pipeline ASR (lựa chọn đã chốt ở E và mục 7). Kết quả của J dùng cho hai quyết định của bước sau,
ghi trước khi đo:

- Bản ghép thắng của lần 3 (J3) được đưa vào `configs/v0.toml` thay cho large-v3 đứng riêng nếu điểm thi của J3 không thấp
  hơn J1 có ý nghĩa thống kê trên cả hai tập val.
- Định dạng hai phương án được đưa vào dữ liệu fine-tune LLM nếu J4 hơn J3 có ý nghĩa thống kê trên ít nhất một tập val và
  không thua có ý nghĩa thống kê trên tập còn lại.

### 9.7. Kết quả

Phiên K4 đang chạy dở: J1 trên `val_seen` mới xong 11/274 mẫu (trung bình 13,3 giây mỗi mẫu trên T4, từ 7 tới 23
giây) thì máy Colab bị thu hồi, và Colab tạm không cấp GPU mới. Bộ chạy có `--resume`, nên lần chạy sau tiếp tục từ mẫu
thứ 12.

| Nguồn | Điểm thi (seen / unseen) | Chênh lệch so với J1, CI 95% | Mẫu được / mất điểm so với J1 | Dấu ngoặc lọt vào tham số |
|---|---|---|---|---|
| J0 chữ chuẩn | chưa đo | chưa đo | chưa đo | |
| J1 RNNT | chưa đo | 0 (mốc) | 0 / 0 | |
| J2 ghép lần 2 | chưa đo | chưa đo | chưa đo | |
| J3 ghép lần 3 | chưa đo | chưa đo | chưa đo | |
| J4 hai phương án | chưa đo | chưa đo | chưa đo | chưa đo |

### 9.8. Phân tích

Chưa có: cần kết quả của phiên K4.

### 9.9. Kết luận

Chưa có: cần kết quả của phiên K4.

### 9.10. Giới hạn

- **Chỉ đo độ chính xác, không đo thời gian thật.** ASR không chạy cùng lúc, máy là T4 chứ không phải A30. Thời gian thật
  của cả hệ thống đo ở lần chạy hệ thống sau lần 3.
- **LLM chưa fine-tune.** Điểm và thứ hạng giữa các nguồn có thể đổi sau khi fine-tune LLM.
- **Dao động giữa các lần chạy.** Dù giải mã tất định, vLLM có thể cho chữ khác nhau rất nhỏ khi số câu trong lô khác nhau;
  bootstrap chỉ tính dao động do chọn mẫu, không tính dao động này.
- Kết quả trên val đã được dùng một lần ở bước chọn chung (mục 7), nhưng J không chọn gì theo val nên không làm số lạc
  quan thêm.

## Giới hạn đã biết

- Tập dò chỉ có 300 mẫu (394 từ ngoại), nên chênh vài từ đã là vài điểm phần trăm; tập dò dùng hai lần (E và bảng chọn
chung), nên số tập dò của bước chọn thứ hai lạc quan hơn (mục 2.3).
- Mọi chỉ số chất lượng dựa vào quy tắc tìm từ ngoại của lần 2 (rà tay: đúng 99,7% khi nhận, tìm được 94,3%; 90,6% trên
`val_unseen`). Từ tiếng Anh trông như âm tiết Việt ("team", "gym") không được tính là từ tiếng Anh.
- Mục tiêu cuối là gọi đúng tool. Các lựa chọn ASR làm theo chất lượng chữ; điểm thi chỉ đo cho các cấu hình đã chốt
(thử nghiệm J), với LLM chưa fine-tune.
- Thời gian đo trên T4, không có vLLM chạy cùng (mục 2.4).

## Phụ lục A. Bảng tra code

### A.1. Code có sẵn, dùng lại từ lần 2


| File › tên                                                          | Loại         | Đầu vào → đầu ra                                                             | Việc làm                                      |
| ------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------- | --------------------------------------------- |
| `lifemate/eval/foreign_words.py` › `find_foreign_words`             | hàm          | câu, tập từ viết thường → danh sách từ ngoại (cách viết gốc, dạng chuẩn hoá, nhóm) | Quy tắc tìm từ ngoại (báo cáo lần 2, mục 2.1) |
| `lifemate/eval/eval_asr.py` › `evaluate`                            | hàm          | mẫu, transcriber, câu mồi → `AsrReport`                                      | Chấm mọi chỉ số ASR                           |
| `lifemate/eval/rescore_asr.py` › `ReplayTranscriber`                | lớp          | `hyps.jsonl` → chữ từng câu                                                  | Chấm lại offline                              |
| `lifemate/eval/asr_compare.py` › `compare`                          | hàm          | mẫu, các hệ thống → chỉ số, CI, CI của chênh lệch                            | Bootstrap có ghép cặp theo mẫu                |
| `lifemate/eval/asr_merge.py` › `merge_transcripts`                  | hàm          | chữ RNNT, chữ large-v3, bộ lọc → chữ ghép                                    | Quy tắc ghép                                  |
| `lifemate/eval/asr_merge.py` › `sample_context_words`               | hàm          | mẫu, catalog → `frozenset[str]` từ đã chuẩn hoá                                    | Từ điển ngữ cảnh                              |
| `lifemate/eval/asr_merge.py` › `english_filter`, `CONTEXT_POLICIES` | hàm, hằng số | cách lọc, ngữ cảnh, từ câu mồi → hàm nhận/loại một từ                        | Bộ lọc bước 3b                                |
| `lifemate/backends/asr.py` › `ASRTranscriber`                       | lớp          | mẫu → `Transcript` (chữ từng lượt, `seconds`)                                | Chép mọi câu của mẫu, lớp bảo vệ, lọc ảo giác |
| `lifemate/backends/asr.py` › `FasterWhisperASR`                     | lớp          | danh sách file WAV → danh sách chữ                                           | Backend large-v3 (CTranslate2)                |
| `lifemate/backends/asr.py` › `ChunkFormerASR`                       | lớp          | danh sách file WAV → danh sách chữ                                           | Backend RNNT                                  |
| `lifemate/backends/asr_guard.py` › `needs_retry`                    | hàm          | chữ, chữ đã lọc, câu mồi, độ dài cụm, ngưỡng nén → có chép lại không         | Lớp bảo vệ câu mồi                            |
| `lifemate/run/timing.py` › `TimedTranscriber`                       | lớp          | bọc transcriber → cộng dồn số giây và số lần gọi                             | Đo thời gian trong lần chạy hệ thống          |
| `lifemate/io/zaic.py` › `load_catalog`                              | hàm          | thư mục dữ liệu, split → catalog tool                                        | Đọc catalog (không đọc file môi trường)       |


### A.2. Code đã viết hoặc sửa ở lần 3

| File › tên | Loại | Đầu vào → đầu ra | Việc làm | Thử nghiệm |
|---|---|---|---|---|
| `lifemate/eval/make_splits.py` | sửa | ghi thêm `configs/splits/asr_learn.txt` (793 mã) | Danh sách tập học | F, G, H |
| `lifemate/eval/asr_merge.py` › `sample_context_words` | sửa | mẫu, catalog → tập từ, chỉ lấy giá trị JSON | Từ điển ngữ cảnh bỏ khoá | mọi cách ghép |
| `lifemate/eval/asr_merge.py` › `Region`, `split_regions`, `candidate_regions`, `assemble`, `rule_decision` | mới | hai câu → từ khớp và vùng khác nhau; hàm quyết định → câu ghép | Dùng chung cho C0, F, G, H; ghép lại giống hệt lần 2 từng byte | mọi cách ghép |
| `lifemate/backends/asr_prompt.py` › `build_sample_prompts` | module mới | mẫu, catalog, câu mồi cố định, biến thể → câu mồi từng câu theo thứ tự file | Câu mồi theo mẫu | E |
| `lifemate/backends/asr.py` › `FasterWhisperASR.transcribe_one` | hàm mới | âm thanh, câu mồi → chữ, `avg_logprob`, `no_speech_prob` | Câu mồi riêng từng câu, điểm tin cậy | E, H |
| `lifemate/backends/asr.py` › `FasterWhisperASR.score_texts` | hàm mới | âm thanh, các câu → (tổng log xác suất, số token) | Chấm ép chuỗi không câu mồi (`encode` một lần, `align` từng câu) | G |
| `lifemate/eval/asr_v3.py` › `transcribe_variants`, `score_candidates` | module mới | file id, biến thể → `hyps.jsonl`, `timing.jsonl`, `summary.json`; chữ hai mô hình → điểm âm thanh | Bộ chạy các phiên K1–K3: RNNT, large-v3 với lớp bảo vệ so phần cố định, bộ lọc câu bịa, ghép, chấm điểm, đo thời gian từng bước, `--resume` | E, F, G, H |
| `lifemate/eval/gpu_memory.py` › `GpuMemoryMonitor` | module mới | chạy nền `nvidia-smi` → `gpu_memory.csv`, bộ nhớ đỉnh | Đo bộ nhớ | mọi phiên |
| `lifemate/eval/candidate_scoring.py` › `candidate_pairs`, `decide_g` | module mới | chữ RNNT, chữ large-v3 → câu ứng viên A, B; điểm → quyết định | Câu ứng viên của G | G |
| `lifemate/eval/asr_v3_analysis.py` | module mới | `hyps.jsonl`, điểm âm thanh → vùng có nhãn, bảng độ tin cậy, các cách quyết định C0, F, G, H, hồi quy logistic, độ chính xác quyết định đoạn kèm bootstrap, hai phương án | Phân tích offline | F, G, H, bảng chọn chung, I |
| `lifemate/eval/asr_v3_report.py` | module mới | thư mục kết quả K1–K3 → các bảng số (`e.json`, `select.json`) và chữ của các nguồn J | Tính số cho báo cáo | E–I |
| `lifemate/eval/asr_v3_agent.py` | module mới | thư mục run agent → điểm từng nguồn, CI, đếm loại lỗi, dấu ngoặc lọt vào tham số | Chấm thử nghiệm J; mẫu có chữ giống J1 dùng lại điểm J1 | J |
| `lifemate/run/cli.py` › `--transcripts replay --hyps`, `--set` | sửa | file `hyps.jsonl` → agent đọc chữ đã lưu | Chạy agent trên chữ đã lưu | J |
| `configs/prompts/system_v1_alternatives.md` | file mới | `system_v1.md` thêm quy tắc 15 | Prompt cho J4 | J |
| `ops/colab/asr_v3_setup.py`, `ops/colab/asr_v3_launch.py` | script mới | | Chuẩn bị máy Colab và chạy nền một phiên | mọi phiên |
| `ops/kaggle/asr_v3/` | kernel mới | | Bản Kaggle của K1–K3 (chưa dùng được vì hết hạn mức GPU tuần này) | |

**Chưa làm ở lần 3:** engine ghép cho hệ thống thật (`MergedASR` trong `lifemate/backends/asr.py`) và khoá config
`ASRConfig.prompt_mode`. Thời gian ở lần 3 được đo bằng bộ chạy `lifemate/eval/asr_v3.py`, chạy đúng các bước của engine
theo cùng thứ tự trong cùng một tiến trình; engine cho hệ thống thật là việc của bước sau lần 3.

## Phụ lục B. Việc chuẩn bị và lịch các phiên GPU

### B.1. Việc chuẩn bị (máy dev, không cần GPU)

Kế hoạch ban đầu tách code theo từng thử nghiệm (P9, P11, P13 mỗi việc một module). Khi viết, phần phân tích offline
của F, G, H, bảng chọn chung và I được gộp vào `lifemate/eval/asr_v3_analysis.py` và `lifemate/eval/asr_v3_report.py`;
cột "Đã làm ở" ghi file thật. Tên hàm ở Phụ lục A.2.

| Mã | Cho | Việc | Đã làm ở | Trạng thái |
| --- | --- | --- | --- | --- |
| P1 | Chung | `make_splits` ghi `asr_learn.txt` (793 mã = `train.txt` bỏ `asr_tune.txt`) | `lifemate/eval/make_splits.py` | Xong |
| P2 | Chung | Đọc mã nguồn faster-whisper 1.2.1: cách cắt câu mồi khi quá 223 token; xác nhận `align` chấm được chuỗi cho sẵn không kèm câu mồi | Mục 3.3 và 5.3 | Xong |
| P3 | Chung | `sample_context_words` bỏ khoá JSON, chỉ lấy giá trị | `lifemate/eval/asr_merge.py` | Xong |
| P4 | Chung | Chỉ số mới (mục 2.2) | `lifemate/eval/asr_v3_report.py` (`en_context_recall`, `assistant_echo`, `merge_word_errors`) | Xong |
| P5 | Chung | Đọc bộ nhớ GPU bằng `nvidia-smi` chạy nền | `lifemate/eval/gpu_memory.py` › `GpuMemoryMonitor` | Xong |
| P6 | E | Câu mồi theo mẫu | `lifemate/backends/asr_prompt.py` › `build_sample_prompts` | Xong |
| P7 | E | `FasterWhisperASR` chép với câu mồi riêng từng câu, trả điểm tin cậy; lớp bảo vệ so phần cố định; ghi `prompt` và `scores` vào `hyps.jsonl` | `lifemate/backends/asr.py`, `lifemate/eval/asr_v3.py` | Xong, trừ khoá config `ASRConfig.prompt_mode` |
| P8 | E | Engine `MergedASR` cho hệ thống thật (RNNT, large-v3, lớp bảo vệ, ghép) | Bộ chạy `lifemate/eval/asr_v3.py` thay cho engine khi đo | Chưa làm (để sau J) |
| P9 | F | Đếm, độ tin cậy, kiểm định chéo chọn θ, cách quyết định `context_reliable` | `asr_v3_analysis.py`, `asr_v3_report.py` › `stage_select` | Xong |
| P10 | G | Câu ứng viên, `FasterWhisperASR.score_texts`, lệnh chấm điểm cho K3 | `lifemate/eval/candidate_scoring.py`, `lifemate/backends/asr.py`, `lifemate/eval/asr_v3.py` › `score_candidates` | Xong |
| P11 | H | Đặc trưng, nhãn tự động, hồi quy logistic | `asr_v3_analysis.py` | Xong |
| P12 | Bảng chọn chung | Độ chính xác quyết định đoạn kèm bootstrap; chạy mọi ứng viên trên cùng các đoạn và áp quy tắc chọn | `asr_v3_analysis.py` › `decision_accuracy`, `bootstrap_accuracy`; `asr_v3_report.py` › `stage_select` | Xong |
| P13 | I | Định dạng hai phương án, đếm trần | `asr_v3_analysis.py` › `alternatives` | Xong (đếm token bằng lệnh riêng, mục 8.3) |
| P14 | J | `--transcripts replay`, `--set`, prompt cho J4, chấm các nguồn | `lifemate/run/cli.py`, `configs/prompts/system_v1_alternatives.md`, `lifemate/eval/asr_v3_agent.py` | Xong |
| P15 | Kết thúc | `make check`, `make test`, cập nhật SPEC và README các thư mục | | `make check`, `make test` qua; SPEC chờ kết quả J |

Kiểm tra: 284 test qua (2 bỏ qua), trong đó `tests/test_asr_v3.py`, `tests/test_asr_v3_analysis.py` và phần thêm của
`tests/test_asr_merge.py` là test của lần 3.

### B.2. Các phiên GPU

**Máy chạy thực tế.** Hạn mức GPU Kaggle của tuần đã hết (30 giờ) khi bắt đầu lần 3, nên mọi phiên chạy trên **Colab T4**
(15 GB) qua Colab CLI. Máy Colab miễn phí bị thu hồi sau khoảng 50–60 phút mỗi lần; bộ chạy có `--resume` (giữ các mẫu đã
xong, xoá dòng của mẫu đang dở), và kết quả được tải về máy dev mỗi 4–10 phút, nên mỗi lần bị thu hồi chỉ mất vài mẫu.
Máy Colab dùng CUDA 13; CTranslate2 cần thêm `nvidia-cublas-cu12` và cuDNN 9 (`ops/colab/asr_v3_setup.py`).

| Phiên | Máy | Chạy gì | Số câu hoặc mẫu | Thời gian thực |
|---|---|---|---|---|
| K1 | Colab T4, 3 máy nối tiếp | RNNT và large-v3 E0, E1, E2 trên `asr_tune`; chép lại 50 mẫu | 300 mẫu, 1 024 câu | Khoảng 27 phút cho máy cuối (gồm chép lại); tổng hơn 1 giờ |
| K2 | Colab T4, 2 máy | RNNT và large-v3 E2 trên `val_seen`, `val_unseen`, tập học | 1 245 mẫu, 4 012 câu | Khoảng 1 giờ 40 phút |
| K3 | Colab T4 | Điểm âm thanh của mọi vùng cần quyết định | 968 vùng | Khoảng 6 phút |
| K4 | Colab T4 | vLLM 0.32.0 với Qwen3-4B; agent trên val: J0, J1 đủ mẫu; J2, J3, J4 trên các mẫu có chữ khác J1 | 452 + 452 + 119 + 112 + 188 mẫu | Đang chạy dở: mới xong 11 mẫu J1 `val_seen`; ước khoảng 5 giờ GPU (13,3 giây mỗi mẫu) |

Kết quả thô: `artifacts/runs/colab/asr/10-30_v3/` (`k1/`, `k2_*/`, `k3/`, `k4/`, `analysis/`, `j_sources/`).
