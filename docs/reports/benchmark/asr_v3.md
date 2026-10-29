# Báo cáo benchmark ASR lần 3: dùng ngữ cảnh của từng mẫu, học quyết định ghép, và đo thời gian

Ngày lập kế hoạch: 29/10/2026. Trạng thái: **kế hoạch, chưa chạy**. Mọi bảng kết quả đang để trống ("chưa đo"); các
phần kết luận và phân tích sẽ được viết sau khi có số.

Báo cáo trả lời bốn câu hỏi:

1. Câu mồi (`initial_prompt`) dựng riêng cho từng mẫu, từ ngữ cảnh có sẵn dạng chữ lúc thi (lượt trợ lý liền trước, app
đã cài), có giúp whisper-large-v3 chép đúng thêm những từ tiếng Anh **không có** trong câu mồi cố định của lần 2
không? (thử nghiệm E)
2. Ở mỗi đoạn hai bản viết khác nhau, cách quyết định nào tốt nhất: luật của lần 2, lọc theo độ tin cậy từng từ của
câu mồi (thử nghiệm F), cho mô hình chấm điểm hai phương án (thử nghiệm J), hay bộ phân loại học từ dữ liệu (thử nghiệm
G)? Mọi cách được so trong một bảng chung và chọn một lần (mục 2.5).
3. Nếu không chốt một bản chữ mà giữ cả hai phương án ở chỗ hai mô hình bất đồng, thì từ đúng nằm trong một trong hai
phương án ở bao nhiêu phần trăm số trường hợp, và tốn thêm bao nhiêu token cho LLM? (thử nghiệm H)
4. Toàn bộ phần ASR (hai mô hình, lớp bảo vệ, phép ghép) chạy hết bao lâu thời gian và tốn bao nhiêu bộ nhớ GPU trên phần cứng  
giống máy chấm (A30), so với giới hạn 30 giây mỗi mẫu? (thử nghiệm I)

Phạm vi: chỉ đo ASR. Không chạy LLM, không chấm điểm thi. Bước sau lần 3 mới chạy cả hệ thống với LLM để đo thời gian
và độ chính xác của lời gọi tool (mục "Sau lần 3").

Tên thử nghiệm nối tiếp lần 2: lần 2 dùng A, B, C, D (`docs/reports/benchmark/asr_v2.md`), lần 3 dùng E, F, G, H, I, J.

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

Phần ASR trên là cấu hình thắng của benchmark lần 2, chưa đưa vào `configs/v0.toml` (file này vẫn dùng large-v3 đứng
riêng). Điểm thi chấm lời gọi tool, không chấm chữ, nhưng chữ ASR sai từ khoá thì LLM không sửa được. Lần 3 vẫn chỉ đo
bước ASR.

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
| Bản ghép vẫn làm câu thuần Việt sai nhiều hơn RNNT                                                | +0,4 điểm WER (seen), +0,3 (unseen). Trên tập dò, 24 trong 30 từ bị lấy nhầm là từ của câu mồi ("Mail" cho "mai")                      | F, G             |
| Quy tắc ghép viết tay, chỉ dùng được một tín hiệu (từ có trong ngữ cảnh không)                    | Khi thêm bộ lọc, chép đúng từ tiếng Anh trên tập dò giảm 3,3 điểm                                                                             | G                |
| Chưa đo thời gian và bộ nhớ trên GPU giống máy chấm                                               | Trên T4: RNNT p95 2,5 s, large-v3 p95 8,6–8,8 s mỗi mẫu, chạy riêng từng mô hình                                                       | I                |


Ngoài ra, lần 2 luôn chốt **một** bản chữ trước khi đưa cho LLM. Ở chỗ hai mô hình bất đồng mà quy tắc không đủ bằng
chứng, một trong hai phương án bị bỏ, dù LLM có thể chọn đúng nhờ danh sách app và kết quả tool. Thử nghiệm H đo xem giữ
cả hai phương án thì được bao nhiêu (mục 6).

### Ý tưởng chung: dùng thông tin mà từng mẫu có sẵn dạng chữ

Lúc thi, mỗi mẫu đi kèm một số thông tin dạng chữ: danh sách app đã cài và catalog tool của các app đó, ngữ cảnh người
dùng (giờ, vị trí, hồ sơ), và các lượt trợ lý trong lịch sử (câu trả lời, lời gọi tool, kết quả tool). Lượt người dùng
trong lịch sử **chỉ có âm thanh**, nên phải tự chép. Các thông tin dạng chữ cho biết người dùng **có khả năng** nói từ
nào. Ví dụ, trợ lý vừa hỏi "Gửi cho Sarah hay Sara?" thì câu người dùng trả lời ngay sau dễ chứa đúng tên đó.

Lần 2 mới dùng thông tin này ở **sau** khi chép, để lọc khi ghép. Lần 3 dùng thông tin đó ở ba chỗ:

1. **Trước khi chép** (E): đưa vào câu mồi của large-v3, để mô hình viết đúng các từ đó ngay từ đầu.
2. **Khi ghép** (F, G): đo độ tin cậy của từng từ, và dùng nhiều tín hiệu cùng lúc thay cho một luật.
3. **Khi đưa cho LLM** (H): giữ cả hai phương án ở chỗ chưa chắc, để LLM chọn bằng ngữ cảnh mà LLM thấy (kết quả tool, danh sách app).

### Dữ liệu cần chuẩn bị và cách chuẩn bị

Lần 3 dùng **đúng cách chia của lần 1 và lần 2**, không chia lại. Khác biệt duy nhất: 793 mẫu train còn trống (lần 2
chỉ dùng để chọn từ cho câu mồi) nay là **tập học** cho thử nghiệm F và G. Ngoài ra, `public_test` được chạy ASR để đo
**thời gian và bộ nhớ**, không đo chất lượng, vì tập này không có đáp án và giống bộ thi thật nhất về độ dài âm thanh.

```
1 545 mẫu train của BTC
├── val_unseen  178 mẫu  đo kết quả cuối, một lần, trên app chưa gặp
├── val_seen    274 mẫu  đo kết quả cuối, một lần, trên app đã biết
└── train     1 093 mẫu
    ├── asr_tune  300 mẫu  tập dò: chọn biến thể, chọn ngưỡng
    └── còn lại   793 mẫu  tập học: đếm độ tin cậy (F), học bộ phân loại (G), chọn θ của F
634 mẫu public_test        chỉ đo thời gian và bộ nhớ (I); không đọc chữ ASR, không chọn gì theo tập này
```


| Phần dữ liệu                             | Số mẫu (số câu) | Dùng để                                                           | Vì sao tách riêng                                                                              |
| ---------------------------------------- | --------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Tập học (793 mẫu train ngoài `asr_tune`) | 793 (2 569)     | Đếm độ tin cậy từng từ của câu mồi; học bộ phân loại ghép         | Học trên tập dò hay tập kiểm tra thì quy tắc "biết trước đề"                                   |
| Tập dò `asr_tune`                        | 300 (1 024)     | Chọn biến thể câu mồi, ngưỡng độ tin cậy, ngưỡng của bộ phân loại | Chọn trên tập nào thì số của tập đó đẹp hơn thực tế, nên cần tập khác để đo                    |
| Tập kiểm tra `val_seen`                  | 274 (916)       | Đo kết quả cuối một lần                                           | Chưa dùng để chọn gì ở lần 3                                                                   |
| Tập kiểm tra `val_unseen`                | 178 (527)       | Đo kết quả cuối một lần, trên app chưa gặp                        | Đây là chỗ câu mồi cố định của lần 2 giúp ít nhất                                              |
| `public_test`                            | 634 (1 933)     | Đo thời gian và bộ nhớ                                            | Có mẫu dài nhất (82,8 giây âm thanh mỗi mẫu); thể lệ cấm gán nhãn, nên chỉ chạy, không đọc chữ |


**Cách chuẩn bị.**

1. Đặt gói dữ liệu của BTC vào `data/train+public_test/` và chạy `make splits` như lần 2. Danh sách 793 mẫu tập học là
`train.txt` bỏ đi `asr_tune.txt`; lệnh sẽ ghi thêm file `configs/splits/asr_learn.txt` (việc chuẩn bị P1, Phụ lục B).
2. Chữ ASR của lần 2 dùng lại được: RNNT trên `asr_tune` và hai tập val, large-v3 có và không câu mồi trên `asr_tune`,
large-v3 `en_full_v1` + g4 trên hai tập val.
3. Phải chạy GPU để có thêm: large-v3 với các biến thể câu mồi mới trên `asr_tune`; RNNT và large-v3 (câu mồi thắng) trên
tập học; large-v3 câu mồi thắng trên hai tập val; điểm âm thanh cho J; toàn bộ phần ASR trên
`public_test` và val để đo thời gian.
Lịch các phiên GPU ở Phụ lục B.

### Các thử nghiệm


| Thử nghiệm             | Câu hỏi                                                                                                          | Cách làm ngắn gọn                                                                                                                        | Dữ liệu                                                                                                      | Cần GPU?                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------- |
| E. Câu mồi theo mẫu    | Câu mồi dựng từ ngữ cảnh của mẫu có giúp từ tiếng Anh không có trong câu mồi cố định không? | Hai biến thể: thêm câu trả lời của trợ lý liền trước; thêm cả tên app đã cài | Chọn trên `asr_tune`, đo trên val | Có |
| F. Độ tin cậy từng từ  | Lọc từ hay bị viết nhầm của câu mồi có giảm lỗi ở câu thuần Việt không? | Đếm trên tập học: mỗi lần large-v3 viết ra từ đó, có bao nhiêu lần người dùng thật sự nói. Từ hay sai thì chỉ nhận khi có trong ngữ cảnh. Không cần mô hình lúc chạy, nên cũng là phương án dự phòng | Đếm và chọn ngưỡng trên tập học (kiểm định chéo), vào bảng chọn chung (mục 2.5) | Có (để có chữ trên tập học) |
| J. Chấm điểm hai phương án | Cho large-v3 chấm xác suất của hai phương án theo âm thanh có quyết định ghép tốt hơn luật không? | Dựng hai câu chỉ khác nhau ở đoạn cần quyết định; large-v3 không câu mồi chấm từng câu bằng ép chuỗi; chọn câu điểm cao hơn. Không có tham số | Vào bảng chọn chung (mục 2.5) | Có (phiên K3) |
| G. Bộ phân loại ghép   | Mô hình nhỏ học từ dữ liệu, dùng mọi tín hiệu, có quyết định ghép tốt hơn không? | Hồi quy logistic, mỗi đoạn khác nhau là một mẫu học, đầu vào là các tín hiệu của E, F, J và điểm tin cậy của large-v3 | Học trên tập học, vào bảng chọn chung (mục 2.5) | Có (dùng điểm của K3) |
| H. Giữ hai phương án   | Giữ cả hai phương án ở chỗ chưa chắc thì trần chép đúng là bao nhiêu, tốn bao nhiêu token?                       | Viết chỗ bất đồng thành `{RNNT | large-v3}`, đo tỉ lệ từ đúng có mặt trong ít nhất một phương án                                         | `asr_tune`, val                                                                                              | Không                       |
| I. Thời gian và bộ nhớ | Toàn bộ phần ASR có vừa ngân sách thời gian và bộ nhớ trên A30 không, kể cả khi vLLM đã nạp Qwen3-4B trên cùng GPU? Cách chạy nào nhanh nhất mà không đổi chữ? | Đo từng mẫu: tuần tự, song song, chép theo lô, 8 bit; đo bộ nhớ đỉnh khi có vLLM; chạy lại để kiểm tra tất định | Thời gian trên `public_test` và val; T4 cho mọi thử nghiệm, A30 (hoặc GPU 24 GB gần nhất) cho bản chính thức | Có |


**Thứ tự quyết định.** E chọn câu mồi trước. Sau đó F, J, G và luật của lần 2 (C0) cùng vào **một bảng chọn
chung** (mục 2.5): mọi tham số học trên tập học, tập dò chỉ dùng một lần để chọn ứng viên thắng, rồi đo ứng viên thắng một
lần trên val. H dùng kết quả của bước chọn chung. I đo cấu hình cuối.

### Cách chọn và tiêu chí đặt trước

Giống lần 2: mọi lựa chọn làm trên tập dò, cấu hình thắng đo **đúng một lần** trên hai tập kiểm tra. Tiêu chí của từng
thử nghiệm được ghi ở đây **trước khi đo**, để không chỉnh tiêu chí theo số:


| Thử nghiệm | Tiêu chí chọn (đo trên tập dò, trừ I)                                                                                                                                                                                                                                      |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E          | Chọn biến thể có tỉ lệ chép đúng từ tiếng Anh **của bản ghép** cao nhất, với hai điều kiện: WER câu thuần Việt của bản ghép không tăng so với bản ghép dùng `en_full_v1` (CI 95% của chênh lệch không nằm hẳn trên 0), và p95 thời gian của large-v3 tăng không quá 1 giây |
| Quyết định ghép (C0, F, J, G) | Loại ứng viên tệ hơn C0 có ý nghĩa thống kê ở chép đúng từ tiếng Anh hoặc ở WER câu thuần Việt, hoặc tốn thêm quá 1 giây ở p95 mỗi mẫu trên T4. Trong số còn lại, chọn ứng viên có độ chính xác quyết định đoạn cao nhất; nếu không hơn có ý nghĩa thống kê một ứng viên đơn giản hơn thì chọn ứng viên đơn giản hơn (thứ tự C0, F, J, G). Chi tiết ở mục 2.5 |
| H          | Không chọn gì; chỉ đo trần và chi phí token, để quyết định có đưa định dạng này vào lần chạy hệ thống với LLM không                                                                                                                                                        |
| I          | Trên A30: thời gian ASR của mẫu chậm nhất trên `public_test` không quá 10 giây, và bộ nhớ GPU đỉnh của phần ASR không quá 6 GB. Trong các cách chạy đạt hai ngưỡng này và cho chữ giống hệt cách chạy tuần tự, chọn cách có thời gian trung bình thấp nhất                 |


Lý do của hai ngưỡng ở I nằm ở mục 7.2.

### Kết quả chính

*Chưa đo.* Bảng sẽ có dạng:


| Hệ thống                                           | Chép đúng từ tiếng Anh (seen / unseen) | WER câu chêm (seen / unseen) | WER câu thuần Việt (seen / unseen) | WER mọi câu (seen / unseen) |
| -------------------------------------------------- | -------------------------------------- | ---------------------------- | ---------------------------------- | --------------------------- |
| RNNT đứng riêng (lần 2)                            | 49,9% / 51,6%                          | 24,8% / 24,1%                | 17,7% / 17,5%                      | 20,0% / 19,8%               |
| Ghép, `en_full_v1`, `context_other` (lần 2)        | 72,2% / 63,1%                          | 22,1% / 22,8%                | 18,1% / 17,8%                      | 19,4% / 19,5%               |
| Ghép, câu mồi thắng của E, `context_other`         | chưa đo                                | chưa đo                      | chưa đo                            | chưa đo                     |
| Ghép, câu mồi thắng của E, ứng viên thắng của bảng chọn chung (mục 2.5) | chưa đo                                | chưa đo                      | chưa đo                            | chưa đo                     |





| Thời gian và bộ nhớ trên A30 (`public_test`) | p50 mỗi mẫu | p95 mỗi mẫu | Mẫu chậm nhất | Trung bình | Bộ nhớ GPU đỉnh |
| -------------------------------------------- | ----------- | ----------- | ------------- | ---------- | --------------- |
| RNNT đứng riêng                              | chưa đo     | chưa đo     | chưa đo       | chưa đo    | chưa đo         |
| Ghép, cách chạy thắng của I                  | chưa đo     | chưa đo     | chưa đo       | chưa đo    | chưa đo         |


### Kết luận

*Điền sau khi đo.*

### Kế hoạch thực hiện và tiến độ

Thứ tự bắt buộc theo phụ thuộc: E phải xong trước F, J, G, vì cả ba cần chữ large-v3 với câu mồi thắng trên tập học. J cần phiên K3 sau K2; G dùng điểm của J làm đặc trưng. Bước chọn chung (mục 2.5) chạy khi đã có đủ sáu ứng viên. H chạy offline sau bước chọn chung. I đo thời gian ở mọi phiên GPU, và có một phiên riêng trên A30 sau khi đã chốt cấu hình.


| Bước    | Việc                                                                                                    | Ở đâu                                                            | Phụ thuộc | Trạng thái |
| ------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------- | ---------- |
| P1–P12  | Viết code chuẩn bị (Phụ lục B.1), test, chạy thử trên CPU 3 mẫu                                         | Máy dev, kernel preflight Kaggle                                 |           | Chưa làm   |
| K1      | Phiên GPU 1: ba biến thể câu mồi (E0, E1, E2) trên `asr_tune`                                           | Kaggle T4, khoảng 1 giờ 10 phút                                  | P1–P12    | Chưa làm   |
| E       | Chấm offline, chọn biến thể thắng                                                                       | Máy dev                                                          | K1        | Chưa làm   |
| K2      | Phiên GPU 2: tập học (RNNT, large-v3 biến thể thắng); val (large-v3 biến thể thắng)                     | Kaggle T4, khoảng 2 giờ 30 phút                                  | E         | Chưa làm   |
| K3      | Phiên GPU 3: chấm điểm âm thanh (large-v3 không câu mồi) trên tập học, `asr_tune`, val | Kaggle T4, khoảng 1 giờ | K2 | Chưa làm |
| F, J, G | Đếm độ tin cậy và chọn θ, học G (chỉ trên tập học); bảng chọn chung trên `asr_tune`; đo ứng viên thắng một lần trên val (mục 2.5) | Máy dev | K2, K3 | Chưa làm |
| H       | Đo trần của hai phương án                                                                               | Máy dev                                                          | F, J, G   | Chưa làm   |
| R1      | Phiên GPU 4: đo thời gian và bộ nhớ của engine ghép trên `public_test` và val, có vLLM nạp Qwen3-4B cùng GPU                           | A30 thuê (RunPod), không có thì GPU 24 GB gần nhất; khoảng 2 giờ | F, J, G   | Chưa làm   |
| Báo cáo | Điền kết quả, kết luận; cập nhật `docs/experiments.md`, `docs/decisions.md`, SPEC                       | Máy dev                                                          | Mọi bước  | Chưa làm   |


Ước tính thời gian GPU dựa trên tốc độ của lần 2 trên T4: large-v3 khoảng 20 phút cho 1 024 câu của `asr_tune`, tức
khoảng 1,2 giây mỗi câu. Chi tiết từng phiên ở Phụ lục B.2.

### Sau lần 3

Dùng kết quả của lần 2 và lần 3 để chạy **cả hệ thống với LLM** trên val: đo thời gian từng mẫu (ASR, LLM, tool) và độ
chính xác của lời gọi tool, so ba nguồn chữ: chữ chuẩn của BTC, RNNT đứng riêng, và bản ghép thắng của lần 3. Nếu thử
nghiệm H cho trần đủ cao, thêm nguồn thứ tư: bản ghép giữ hai phương án. Lần chạy này cần GPU từ 24 GB vì T4 không chứa
nổi vLLM cùng whisper-large-v3 (`ops/kaggle/README.md`).

### Đọc tiếp ở đâu


| Muốn biết                                       | Đọc                  |
| ----------------------------------------------- | -------------------- |
| Thuật ngữ, dữ liệu, thước đo, cách đo thời gian | Mục 0, 2             |
| Câu mồi theo mẫu được dựng thế nào              | Mục 3 (thử nghiệm E) |
| Độ tin cậy từng từ được đếm thế nào             | Mục 4 (thử nghiệm F) |
| Bộ phân loại ghép học gì, từ dữ liệu nào        | Mục 5 (thử nghiệm G) |
| Định dạng hai phương án cho LLM                 | Mục 6 (thử nghiệm H) |
| Đo thời gian, bộ nhớ, và ngưỡng                 | Mục 7 (thử nghiệm I) |
| Chấm điểm hai phương án bằng mô hình            | Mục 8 (thử nghiệm J) |
| Chọn cách quyết định ghép (bảng chung)          | Mục 2.5              |
| Code cần viết và lịch các phiên GPU             | Phụ lục A, B         |


**Kết quả thô.** Không có trên git. Các thư mục dự kiến:


| Thư mục                                                 | Nội dung                                                                                                                          |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `artifacts/runs/kaggle/asr/<ngày>_v3_k1_sample_prompt/` | Phiên K1: `<biến thể>__asr_tune.hyps.jsonl` (chữ từng câu, kèm câu mồi đã dùng và điểm tin cậy), `.json` (chỉ số), `timing.jsonl` |
| `artifacts/runs/kaggle/asr/<ngày>_v3_k2_learn/`         | Phiên K2: chữ RNNT và large-v3 trên tập học và val                                                                                |
| `artifacts/runs/local/asr/<ngày>_v3_offline/`           | Chấm offline E, F, G, H: chữ đã ghép, bảng so sánh có khoảng tin cậy, bảng độ tin cậy từng từ, hệ số của bộ phân loại             |
| `artifacts/runs/kaggle/asr/<ngày>_v3_k3_scoring/`      | Phiên K3: điểm âm thanh của từng câu ứng viên (`scores.jsonl`), thời gian chấm |
| `artifacts/runs/runpod/asr/<ngày>_v3_r1_timing/`        | Phiên R1: `timing.jsonl`, `gpu_memory.csv`, so chữ giữa các cách chạy                                                             |


## Mục lục

- [Tóm tắt cho cả team](#tóm-tắt-cho-cả-team)
- [0. Bối cảnh và thuật ngữ](#0-bối-cảnh-và-thuật-ngữ)
- [1. Vấn đề chung và mục tiêu](#1-vấn-đề-chung-và-mục-tiêu)
- [2. Dữ liệu, thước đo và công cụ dùng chung](#2-dữ-liệu-thước-đo-và-công-cụ-dùng-chung)
- [2.5. Chọn cách quyết định ghép](#25-chọn-cách-quyết-định-ghép)
- [3. Thử nghiệm E: câu mồi theo mẫu](#3-thử-nghiệm-e-câu-mồi-theo-mẫu)
- [4. Thử nghiệm F: độ tin cậy từng từ của câu mồi](#4-thử-nghiệm-f-độ-tin-cậy-từng-từ-của-câu-mồi)
- [5. Thử nghiệm G: bộ phân loại cho quyết định ghép](#5-thử-nghiệm-g-bộ-phân-loại-cho-quyết-định-ghép)
- [6. Thử nghiệm H: giữ hai phương án cho LLM](#6-thử-nghiệm-h-giữ-hai-phương-án-cho-llm)
- [7. Thử nghiệm I: thời gian và bộ nhớ](#7-thử-nghiệm-i-thời-gian-và-bộ-nhớ)
- [8. Thử nghiệm J: chấm điểm hai phương án bằng mô hình](#8-thử-nghiệm-j-chấm-điểm-hai-phương-án-bằng-mô-hình)
- [Giới hạn đã biết](#giới-hạn-đã-biết)
- [Phụ lục A. Bảng tra code](#phụ-lục-a-bảng-tra-code)
- [Phụ lục B. Code cần viết và lịch các phiên GPU](#phụ-lục-b-code-cần-viết-và-lịch-các-phiên-gpu)

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
| Lượt người dùng trong lịch sử | Chỉ âm thanh (`public_test`: 1 299 lượt, chỉ có trường `audio`) | Các câu đã nói trước      | Chép bằng ASR                                 |
| Lượt trợ lý trong lịch sử     | Chữ: câu trả lời (`text`), lời gọi tool, kết quả tool           | "Gửi cho Sarah hay Sara?" | Câu mồi theo mẫu (E), từ điển ngữ cảnh (F, G) |
| `installed_apps`              | Danh sách mã app                                                | `zalo`, `bike_rental`     | Câu mồi theo mẫu (E), từ điển ngữ cảnh        |
| Catalog tool                  | Chữ: tên tiếng Việt của app (`name_vi`), mô tả, tool, tham số   |                           | Từ điển ngữ cảnh; danh sách từ khoá (E3)      |
| Các trường khác của ngữ cảnh  | Chữ: giờ, múi giờ, vị trí, hồ sơ người dùng                     |                           | Từ điển ngữ cảnh                              |


Số liệu, tính trên dữ liệu của BTC:


| Tập           | Số mẫu | Số câu người dùng | Câu mỗi mẫu: trung bình / p95 / tối đa | Câu có lượt trợ lý liền trước | Giây âm thanh mỗi mẫu: trung bình / p95 / tối đa |
| ------------- | ------ | ----------------- | -------------------------------------- | ----------------------------- | ------------------------------------------------ |
| `asr_tune`    | 300    | 1 024             | 3,4 / 11 / 13                          | 724 (71%)                     | 16,2 / 42,0 / 61,1                               |
| Tập học       | 793    | 2 569             |                                        | 1 776 (69%)                   |                                                  |
| `val_seen`    | 274    | 916               | 3,3 / 10 / 13                          | 642 (70%)                     | 16,2 / 37,1 / 64,6                               |
| `val_unseen`  | 178    | 527               | 3,0 / 10 / 13                          | 349 (66%)                     | 16,4 / 38,8 / 69,4                               |
| `public_test` | 634    | 1 933             | 3,0 / 10 / 13                          |                               | 15,7 / 38,3 / 82,8                               |


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
| `hotwords`                       | Tham số của faster-whisper nhận một chuỗi từ khoá để ưu tiên khi giải mã. Lần 1 thử với một danh sách cố định và loại vì kém nhất; lần 3 không dùng |
| Độ tin cậy của một từ câu mồi    | Trong các lần large-v3 (có câu mồi) viết ra từ đó, tỉ lệ lần người dùng thật sự nói từ đó. Đếm trên tập học (mục 4)                                                                                  |
| Đoạn khác nhau                   | Một chỗ mà chữ RNNT và chữ large-v3 viết khác nhau, nằm giữa hai từ khớp, sau khi đặt hai bản cạnh nhau từ ứng với từ (mục 6.3 của báo cáo lần 2)                                                    |
| Bảng chọn chung                  | Bước so mọi cách quyết định ghép (C0, F, J, G) trên cùng các đoạn, bằng một quy tắc chọn, dùng tập dò một lần (mục 2.5) |
| Độ chính xác quyết định đoạn     | Tỉ lệ đoạn khác nhau mà một cách quyết định chọn đúng phía (RNNT hay large-v3) ít lỗi hơn so với đáp án (mục 2.5) |
| Kiểm định chéo 5 phần theo mẫu   | Chia tập học thành 5 phần, mỗi mẫu nằm trọn trong một phần; lần lượt học trên 4 phần và chấm trên phần còn lại, rồi lấy trung bình. Cho biết một lựa chọn tốt tới đâu trên mẫu chưa dùng để học, mà không cần đụng tới tập dò |
| Bộ phân loại                     | Một mô hình nhỏ nhận các tín hiệu của một đoạn khác nhau và trả về xác suất "nên lấy chữ của large-v3" (mục 5)                                                                                       |
| Hồi quy logistic                 | Bộ phân loại đơn giản nhất: cộng các tín hiệu theo trọng số rồi đổi thành xác suất từ 0 tới 1. Trọng số học từ dữ liệu. Ít tham số, nên học được từ vài trăm mẫu                                     |
| Đặc trưng                        | Một tín hiệu dạng số đưa vào bộ phân loại, ví dụ "từ có trong từ điển ngữ cảnh không" (1 hoặc 0)                                                                                                     |
| Nhãn tự động                     | Đáp án cho từng đoạn khác nhau ("lấy large-v3 thì đúng hơn hay sai hơn"), tính máy móc bằng cách so với đáp án của BTC; không cần người gán                                                          |
| Điểm tin cậy của large-v3        | `avg_logprob` (trung bình log xác suất của các token, càng gần 0 càng chắc) và `no_speech_prob` (xác suất đoạn đó không có lời nói) mà faster-whisper trả về cho mỗi đoạn âm thanh                   |
| Hai phương án                    | Cách viết chỗ bất đồng thành `{chữ RNNT | chữ large-v3}` để LLM tự chọn (mục 6)                                                                                                                      |
| Trần của hai phương án           | Tỉ lệ từ tiếng Anh của đáp án có mặt trong ít nhất một phương án. Là mức tối đa LLM có thể đạt nếu luôn chọn đúng                                                                                    |
| p50, p95, tối đa                 | Thời gian chép lời **một mẫu** (lượt cuối và mọi lượt người dùng trong lịch sử) mà 50%, 95%, 100% số mẫu nhanh hơn                                                                                   |
| RTF                              | Hệ số thời gian thực: số giây xử lý chia số giây âm thanh. RTF 0,2 nghĩa là 10 giây âm thanh xử lý mất 2 giây                                                                                        |
| Bộ nhớ GPU đỉnh                  | Lượng bộ nhớ GPU lớn nhất mà tiến trình ASR chiếm trong suốt lần chạy, đọc bằng `nvidia-smi` (mục 2.4)                                                                                               |
| Chạy tuần tự, song song, theo lô | Tuần tự: chép hết bằng RNNT rồi mới chép bằng large-v3. Song song: hai mô hình chạy cùng lúc trên cùng GPU, mỗi mô hình một luồng. Theo lô: large-v3 nhận nhiều câu một lúc thay vì từng câu (mục 7) |
| Điểm âm thanh                    | Tổng log xác suất mà whisper-large-v3, không câu mồi, gán cho từng token của một câu cho sẵn khi nghe đoạn âm thanh; càng gần 0 thì câu càng khớp âm thanh (mục 8) |
| Ép chuỗi (teacher forcing)       | Đưa sẵn chuỗi token vào decoder và chỉ đọc xác suất của từng token, không để decoder tự sinh chữ                    |
| A30, T4                          | A30: GPU của máy chấm, 24 GB, thế hệ Ampere, hỗ trợ bfloat16. T4: GPU miễn phí của Kaggle, 15 GB (16 GB danh nghĩa), thế hệ Turing, không hỗ trợ bfloat16                                            |


### 0.4. Cách đọc các chỗ nhắc tới code

Giống lần 2: báo cáo ghi theo dạng "file › hàm: việc hàm làm". Mọi đường dẫn tính từ thư mục gốc của repo. Hàm đánh dấu
**(sẽ viết)** là code chưa có, sẽ viết ở bước chuẩn bị; tên có thể đổi khi viết, và Phụ lục A sẽ được cập nhật theo.

## 1. Vấn đề chung và mục tiêu

**Vấn đề.** Lần 2 cho thấy ba điều. Thứ nhất, câu mồi chỉ giúp large-v3 viết đúng những từ **có trong** câu mồi, nên
mức tăng trên app lạ nhỏ (56% → 61% trên `val_unseen`), trong khi bộ thi thật có app lạ. Thứ hai, bản ghép vẫn làm câu
thuần Việt sai thêm 0,3–0,4 điểm WER so với RNNT, chủ yếu vì large-v3 viết ra từ của câu mồi khi người dùng không nói.
Thứ ba, chưa biết phần ASR hai mô hình có vừa 30 giây mỗi mẫu trên A30 khi còn phải chừa thời gian cho LLM.

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
ASR thấy được (E, F, G), và chuẩn bị đường để LLM tự sửa ở chỗ chỉ LLM thấy được (H).

**Mục tiêu.**

1. Đo xem câu mồi theo mẫu có tăng tỉ lệ chép đúng từ tiếng Anh, nhất là trên `val_unseen`, mà không làm câu thuần Việt
sai thêm không (E).
2. Đóng phần lỗi còn lại ở câu thuần Việt của bản ghép bằng độ tin cậy từng từ (F) hoặc bộ phân loại (G).
3. Đo trần và chi phí của cách giữ hai phương án cho LLM (H).
4. Thay luật ghép bằng điểm âm thanh của large-v3 không câu mồi (J).
5. Đo thời gian và bộ nhớ của toàn bộ phần ASR trên GPU giống máy chấm, chọn cách chạy nhanh nhất mà vẫn cho cùng chữ
(I).
6. Viết engine ghép vào hệ thống thật (`lifemate/backends/asr.py`), để thử nghiệm I đo đúng thứ sẽ nộp và để bước sau
chạy được cả hệ thống với LLM.

**Ngoài phạm vi lần 3, và lý do.**


| Ý tưởng                                                                              | Lý do chưa làm                                                                                                                                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cộng điểm cho từ trong danh sách ngay trong bộ giải mã của RNNT (contextual biasing) | Gói `chunkformer` chỉ có giải mã tham lam; phải tự viết beam search có cộng điểm cho RNNT. Với Whisper, lần 3 thử dạng tương đương có sẵn: câu mồi theo mẫu (E) |
| Thêm lớp cross-attention vào danh sách từ bên trong ASR (kiểu CLAS, TCPGen)          | Phải fine-tune mô hình ASR; dự án chỉ có khoảng 3 600 câu train và còn 18 ngày tới hạn nộp 16/11                                                                                       |
| Tiền huấn luyện biểu diễn tự giám sát (JEPA)                                         | Cần rất nhiều âm thanh không nhãn và nhiều GPU; các mô hình ASR đang dùng đã tiền huấn luyện sẵn                                                                                       |
| Fine-tune LLM trên định dạng hai phương án                                           | Thuộc phần fine-tune LLM (SFT). Lần 3 chỉ đo trần (H); nếu trần đủ cao thì định dạng này được đưa vào dữ liệu SFT ở bước sau                                                           |


**Các công cụ chuẩn bị trước.**


| Công cụ                                                                                                                            | Mục      | Dùng ở thử nghiệm   |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------- |
| Công cụ của lần 2: quy tắc tìm từ ngoại, bộ chỉ số tiếng Anh, chấm lại offline, bảng so sánh có CI, quy tắc ghép, từ điển ngữ cảnh | 2.2      | E, F, G, H          |
| Câu mồi theo mẫu, lớp bảo vệ chỉ so phần cố định, điểm tin cậy của large-v3 trong `hyps.jsonl`                                     | 3.3      | E, G                |
| Bảng độ tin cậy từng từ và cách lọc `context_reliable`                                                                             | 4.3      | F, G                |
| Đặc trưng, nhãn tự động, hồi quy logistic                                                                                          | 5.3      | G                   |
| Định dạng hai phương án, đếm trần và token                                                                                         | 6.3      | H                   |
| Engine ghép trong hệ thống thật, ghi thời gian từng mô hình, đọc bộ nhớ GPU                                                        | 2.4, 7.3 | I, và mọi phiên GPU |
| Dựng câu ứng viên, chấm điểm âm thanh bằng `align` của CTranslate2                               | 8.3      | J, G                |


## 2. Dữ liệu, thước đo và công cụ dùng chung

### 2.1. Dữ liệu và cách dùng

**Vấn đề.** Lần 3 cần thêm hai loại dữ liệu so với lần 2: dữ liệu để **học** (đếm độ tin cậy từng từ, học bộ phân loại),
và dữ liệu giống bộ thi thật để **đo thời gian**. Dữ liệu học không được trùng tập dò hay tập kiểm tra; dữ liệu đo thời
gian không được dùng để chọn gì về chất lượng.

**Giải pháp.**

1. **Tập học = 793 mẫu train ngoài `asr_tune`.** Lần 2 chỉ đọc chữ large-v3 của các mẫu này (bộ đệm
`data/asr/train_asr.json`, large-v3 beam 5 không câu mồi) để chọn từ cho câu mồi. Lần 3 dùng chính các mẫu này để
đếm độ tin cậy và học bộ phân loại. Lý do: đây là phần duy nhất có đáp án mà chưa dùng để dò hay đo. Bộ đệm cũ không
dùng được cho F và G vì chép bằng beam 5 không câu mồi, khác cấu hình sẽ dùng lúc thi (giải mã tham lam + VAD + câu
mồi); nên phiên K2 chép lại tập học bằng đúng cấu hình đó.
2. **Câu mồi viết lại theo tập học là một bước chọn.** Nếu E hoặc F dẫn tới viết lại câu mồi cố định, việc viết lại chỉ
được dựa vào tập học, như lần 2.
3. **`public_test` chỉ để đo thời gian và bộ nhớ.** Chạy ASR trên `public_test` là chạy suy luận, không phải gán nhãn,
nên không trái thể lệ. Để không vô tình chọn theo tập này: không tính chỉ số chất lượng nào, không đọc chữ ASR, không
lưu chữ ASR của tập này ra file kết quả (chỉ lưu thời gian và bộ nhớ).
4. **Không dùng `public_test` cho bất kỳ dữ liệu học nào**, theo `AGENTS.md`.
5. **Từ điển ngữ cảnh chỉ lấy giá trị, bỏ khoá JSON.** Ở lần 2, `sample_context_words` lấy cả khoá của catalog và của
`context` (`type`, `description`, `properties`, `content`, `name`…), nên các từ này có trong từ điển của mọi mẫu và
large-v3 bịa ra đúng các từ này thì bộ lọc vẫn nhận. Khoá là tên trường do người viết schema đặt, không phải nội dung
người dùng có thể nhắc tới. Đếm thử offline trên tập dò: bỏ khoá bớt 1 từ lấy nhầm ("group") và không mất từ đúng
nào. Lần 3 dùng từ điển bỏ khoá cho mọi thử nghiệm (việc P10). Mốc so sánh `context_other` giữ số của lần 2, không chấm
lại, vì chênh lệch chỉ một từ trên tập dò.


| Việc                    | Dữ liệu                                             | Để làm gì                     | Lý do chọn                                                             |
| ----------------------- | --------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| E: dò biến thể câu mồi  | `asr_tune` đầy đủ (1 024 câu)                       | Chọn biến thể                 | Cần cả câu thường để thấy câu mồi có làm câu thuần Việt sai thêm không |
| E: đo biến thể thắng    | `val_seen`, `val_unseen` đầy đủ, một lần            | Số cuối                       | Như lần 2                                                              |
| F: đếm độ tin cậy       | Tập học, chữ large-v3 với câu mồi thắng của E       | Bảng độ tin cậy từng từ       | Phải cùng cấu hình câu mồi với lúc dùng                                |
| F: chọn ngưỡng          | Tập học, kiểm định chéo 5 phần theo mẫu             | Ngưỡng độ tin cậy             | Tập dò để dành cho bước chọn chung                                     |
| G: học bộ phân loại     | Tập học: chữ RNNT, chữ large-v3, điểm của J         | Trọng số của hồi quy logistic | Học trên tập không dùng để dò hay đo                                   |
| J: chấm điểm             | Tập học, `asr_tune`, val (phiên K3) | Điểm âm thanh từng câu ứng viên | Điểm trên tập học dùng làm đặc trưng của G |
| Chọn cách quyết định ghép | `asr_tune`, một lần; ứng viên thắng đo một lần trên val | Ứng viên thắng (mục 2.5) | Mọi tham số đã cố định trên tập học                     |
| H: đo trần              | `asr_tune`, val                                     | Trần và chi phí token         | Không chọn gì                                                          |
| I: thời gian trên T4    | Mọi phiên GPU                                       | So chi phí giữa các biến thể  | Đo kèm, không tốn thêm phiên                                           |
| I: thời gian chính thức | `public_test` và val, trên A30                      | Số thời gian và bộ nhớ cuối   | `public_test` giống bộ thi thật nhất về độ dài âm thanh                |


### 2.2. Thước đo

Giữ nguyên bộ chỉ số của lần 2 (mục 2.3 của báo cáo lần 2: `en_word_recall` và bản nới lỏng, `en_groups`,
`en_mixed_wer`, `vi_only_wer`, `prompt_insertions`, `en_prompt`, `n_prompt_retries`, WER, CER, bỏ sót giá trị tham số,
câu rỗng, câu bịa, p50/p95/max). Mọi chỉ số chất lượng được báo cho **bản ghép**, vì bản ghép là thứ sẽ đưa cho LLM; chỉ
số của large-v3 đứng riêng được báo kèm để giải thích.

Chỉ số mới:


| Chỉ số (tên dự kiến trong file kết quả)                                                      | Cách tính                                                                                                                     | Vì sao cần                                                                                 |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Chép đúng theo có/không có trong ngữ cảnh (`en_context`)                                     | Như `en_word_recall`, tách theo từ của đáp án **có** và **không có** trong từ điển ngữ cảnh của mẫu                           | Câu mồi theo mẫu chỉ có thể giúp từ có trong ngữ cảnh. Tách ra để biết mức tăng đến từ đâu |
| Chép đúng theo có/không có trong câu mồi cố định (`en_prompt`)                               | Như lần 2, nhưng "câu mồi" là phần cố định                                                                                    | So được với lần 2                                                                          |
| Chép lại lượt trợ lý (`assistant_echo`)                                                      | Số câu mà chữ ASR có 4 từ liên tiếp trùng câu trả lời của trợ lý liền trước, trong khi đáp án không có 4 từ đó                | Câu mồi theo mẫu chứa lời trợ lý. Đây là dấu hiệu large-v3 in lại lời trợ lý thay vì nghe  |
| Từ bị lấy nhầm khi ghép (`merge_wrong_english`)                                              | Số từ tiếng Anh mới mà phép ghép lấy từ large-v3 nhưng không có trong đáp án câu đó; tách theo từ của câu mồi / ngoài câu mồi | Đo trực tiếp thứ F và G muốn giảm. Lần 2 đếm tay bằng script; lần 3 đưa vào code           |
| Độ chính xác quyết định đoạn (`merge_decision_accuracy`) | Trên các đoạn có nhãn tự động (mục 5.3), tỉ lệ đoạn mà cách quyết định chọn đúng phía ít lỗi hơn so với đáp án | Chỉ số chọn của bảng chọn chung (mục 2.5): đo thẳng các quyết định mà các ứng viên khác nhau |
| Từ đúng bị bỏ khi ghép (`merge_missed_english`)                                              | Số từ tiếng Anh mới của large-v3 có trong đáp án mà phép ghép không lấy                                                       | Mặt kia của đánh đổi                                                                       |
| Trần của hai phương án (`alternatives_ceiling`)                                              | Mục 6                                                                                                                         | H                                                                                          |
| Token thêm cho LLM (`alternatives_tokens`)                                                   | Mục 6                                                                                                                         | H                                                                                          |
| Thời gian từng mô hình mỗi mẫu (`asr_s.rnnt`, `asr_s.whisper`, `asr_s.guard`, `asr_s.merge`) | Đồng hồ thực (`time.perf_counter`) quanh từng bước, cộng theo mẫu                                                             | Biết bước nào tốn thời gian                                                                |
| Thời gian ASR mỗi mẫu (`asr_s.total`): p50, p95, tối đa, trung bình                          | Từ lúc bắt đầu đọc file âm thanh đầu tiên tới khi có chữ ghép của mọi câu trong mẫu                                           | Ngân sách 30 giây là theo mẫu. Trung bình dùng khi bằng điểm                               |
| RTF                                                                                          | Tổng thời gian ASR chia tổng số giây âm thanh của tập                                                                         | So được giữa các tập có độ dài khác nhau                                                   |
| Thời gian nạp mô hình (`load_s`)                                                             | Từ lúc tạo engine tới khi khởi động (`warmup`) xong                                                                           | Không tính vào từng mẫu, nhưng cộng vào tổng thời gian chạy Docker                         |
| Bộ nhớ GPU đỉnh (`gpu_peak_mb`)                                                              | Mục 2.4                                                                                                                       | Phần còn lại cho vLLM                                                                      |


**Giới hạn.** `assistant_echo` dựa vào trùng 4 từ, nên bỏ sót trường hợp large-v3 in lại lời trợ lý nhưng đổi vài từ.

### 2.3. Cách chọn và đo

Như lần 2: chọn trên `asr_tune`, đo cấu hình thắng đúng một lần trên `val_seen` và `val_unseen`. So sánh dùng bootstrap có
ghép cặp theo mẫu, 1 000 vòng, seed 0 (`lifemate/eval/asr_compare.py` › `compare`). Hai hệ thống chỉ được coi là khác
nhau khi CI 95% của chênh lệch không chứa 0. Tiêu chí chọn của từng thử nghiệm đã ghi ở phần tóm tắt, trước khi đo.

Tập dò được dùng hai lần: chọn câu mồi (E), rồi chọn cách quyết định ghép (mục 2.5). Mọi tham số khác (θ của F, hệ số của
G) học trên tập học, không chọn trên tập dò. Vì tập dò vẫn được dùng hai lần, số trên tập dò của bước chọn thứ hai
lạc quan hơn thực tế; chỉ số trên val của cấu hình cuối mới là số để báo cáo. Hai bước chọn đều được đo một lần trên val
ngay sau khi chọn, để biết bước nào đóng góp bao nhiêu; không bước nào chỉnh lại theo số val.

### 2.4. Đo thời gian và bộ nhớ

**Vấn đề.** Lần 2 đo thời gian của từng mô hình riêng lẻ, trên T4, bằng hàm chấm `evaluate`. Như vậy chưa biết: thời
gian khi hai mô hình cùng nằm trong một tiến trình; thời gian của lớp bảo vệ và phép ghép; bộ nhớ GPU thật; và tốc độ
trên A30.

**Giải pháp.**

- **Đo trên engine thật.** Engine ghép (sẽ viết, Phụ lục B.1, P7) chạy RNNT, large-v3, lớp bảo vệ và phép ghép trong
cùng một tiến trình, đúng như lúc nộp. Mọi phiên GPU của lần 3 chạy qua engine này, nên thời gian đo được ở K1, K2 là
thời gian thật trên T4, và R1 là thời gian thật trên A30.
- **Thời gian mỗi mẫu.** `lifemate/backends/asr.py` › `ASRTranscriber.transcribe` đã ghi `Transcript.seconds` (đồng hồ
thực từ lúc bắt đầu tới khi có chữ của mọi câu). Engine ghép ghi thêm thời gian của từng bước vào một dict
(`tên bước → giây`) và một dòng `timing.jsonl` cho mỗi mẫu: `{id, n_utterances, audio_s, asr_s: {rnnt, whisper, guard, merge, total}}`.
- **Khởi động trước khi đo.** Mỗi mô hình chạy một lần trên 1 giây im lặng (`warmup`) trước mẫu đầu tiên, và lần đó
không tính. Lý do: lần chạy đầu của GPU tốn thêm thời gian dựng kernel, không lặp lại ở các mẫu sau.
- **Bộ nhớ GPU.** large-v3 chạy trên CTranslate2, không phải PyTorch, nên `torch.cuda.max_memory_allocated` không thấy
bộ nhớ của large-v3. Vì vậy đọc bằng `nvidia-smi --query-gpu=memory.used --format=csv -lms 200` chạy nền trong suốt
phiên, ghi `gpu_memory.csv`; bộ nhớ đỉnh là giá trị lớn nhất trừ giá trị trước khi nạp mô hình (**(sẽ viết)**
`lifemate/eval/gpu_memory.py` › `GpuMemoryMonitor`).
- **Kiểm tra tất định.** Mỗi cách chạy mới (song song, theo lô) được chạy lại lần hai trên 50 mẫu và so chữ từng ký tự
với lần một và với cách chạy tuần tự (`lifemate/eval/asr_compare.py`, so `hyps.jsonl`).

**Lý do chọn `public_test` để đo thời gian chính thức.** Thời gian phụ thuộc số câu và độ dài âm thanh của mẫu. Mẫu dài
nhất của `public_test` có 82,8 giây âm thanh, dài hơn mẫu dài nhất của mọi tập train (69,4 giây). Bộ thi thật có cùng
cấu trúc với `public_test`, nên phân phối thời gian trên `public_test` gần với lúc chấm nhất.

**Giới hạn.**

- Ở phiên R1, vLLM được nạp Qwen3-4B để giữ phần bộ nhớ như hệ thống thật; agent chưa chạy. Khi
agent sinh chữ cùng lúc với ASR, ASR có thể chậm hơn vì tranh GPU. Lần chạy hệ thống sau lần 3 sẽ đo phần này.
- Thời gian đọc file âm thanh phụ thuộc ổ đĩa của máy chấm, không đo được trước.
- Nếu không thuê được A30, số trên GPU 24 GB khác chỉ là ước lượng; tên GPU được ghi kèm mọi số.

### 2.5. Chọn cách quyết định ghép

**Vấn đề.** F, J và G cùng trả lời một câu hỏi: ở mỗi đoạn hai bản viết khác nhau, lấy chữ RNNT hay chữ
large-v3. Nếu chọn nối tiếp (F trước, rồi J và G so với cách thắng của F), tập dò bị dùng để chọn nhiều lần liên tiếp, số
trên tập dò ngày càng đẹp hơn thực tế, và không có phép so cuối cùng giữa mọi cách. Vì vậy mọi cách được đưa vào **một
bảng chung** và chọn **một lần** bằng một quy tắc chung.

**Các ứng viên.**

| Ứng viên | Quyết định bằng | Tham số và cách đặt | Chi phí lúc chạy thêm so với C0 |
|---|---|---|---|
| C0 `context_other` (lần 2, mốc) | Luật: từ của câu mồi nhận hết, từ ngoài câu mồi phải có trong từ điển ngữ cảnh | Không | 0 |
| F `context_reliable` | Như C0, nhưng từ câu mồi có độ tin cậy dưới θ phải có trong từ điển ngữ cảnh | θ chọn bằng kiểm định chéo trên tập học (mục 4.3) | Gần 0 (tra bảng) |
| J | Điểm âm thanh của large-v3 không câu mồi | Không | Encoder và chấm ép chuỗi ở câu có đoạn cần quyết định |
| G | Hồi quy logistic trên mọi đặc trưng (mục 5.3) | Trọng số học trên tập học, hệ số phạt bằng kiểm định chéo trên tập học; ngưỡng 0,5 | Như J |

**Nguyên tắc dùng dữ liệu.** Mọi tham số của mọi ứng viên được học hoặc chọn **chỉ trên tập học**. Tập dò `asr_tune`
chỉ dùng một lần ở bước này, để chọn giữa các ứng viên đã cố định. Ngưỡng 0,5 của G là ngưỡng tự nhiên của hồi quy
logistic (lấy phía có xác suất cao hơn), không dò.

**Đầu vào chung.** Mọi ứng viên dùng cùng chữ RNNT, cùng chữ large-v3 (câu mồi thắng của E) và cùng các đoạn khác nhau.
Chỉ những đoạn mà phía large-v3 có từ tiếng Anh mới được quyết định; các đoạn khác giữ chữ RNNT ở mọi ứng viên. Nhờ vậy
các ứng viên chỉ khác nhau đúng ở các quyết định cần so.

**Chỉ số chọn: độ chính xác quyết định đoạn** (`merge_decision_accuracy`). Trên các đoạn có nhãn tự động (mục 5.3: nhãn
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
   tới phức tạp: C0, F, J, G (ít tín hiệu hơn, ít tham số hơn, ít thời gian chạy hơn).
4. Nếu không ứng viên nào hơn C0 có ý nghĩa thống kê, giữ C0.

**Thống kê.** Bootstrap ghép cặp theo mẫu, 1 000 vòng, seed 0, như lần 2: mỗi vòng bốc lại mẫu có hoàn lại, tính lại mọi
chỉ số trên các câu và các đoạn của mẫu được bốc. Hai ứng viên khác nhau có ý nghĩa thống kê khi CI 95% của chênh lệch
không chứa 0.

**Đo trên tập kiểm tra.** Ứng viên thắng và C0 được đo **một lần** trên `val_seen` và `val_unseen`, với mọi chỉ số. Không
chọn lại theo số của val.

**Vì sao vẫn giữ F.** F không cần chấm điểm bằng mô hình nào lúc chạy, nên là phương án dự phòng nếu J tốn quá thời gian
trên A30 (phiên R1), nếu chấm ép chuỗi gặp lỗi kỹ thuật, hoặc nếu J không qua điều kiện chặn. F chỉ cần số đếm trên tập
học đã có từ phiên K2, nên không tốn thêm GPU.

**Kết quả trên tập dò** (chênh lệch so với C0 kèm CI 95%):

| Ứng viên | Độ chính xác quyết định đoạn | Chép đúng từ tiếng Anh | WER câu thuần Việt | WER câu chêm | WER mọi câu | Từ lấy nhầm (câu mồi / ngoài câu mồi) | Từ đúng bị bỏ | Thời gian chấm thêm p95 (s) |
|---|---|---|---|---|---|---|---|---|
| C0 | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | 0 |
| F | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| J | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| G | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

Số đoạn được quyết định, số đoạn có nhãn, số đoạn hoà: chưa đo.

**Kết quả trên tập kiểm tra** (đo một lần):

| Tập | Hệ thống | Độ chính xác quyết định đoạn | Chép đúng từ tiếng Anh | WER câu thuần Việt | WER câu chêm | WER mọi câu | Bỏ sót giá trị tham số |
|---|---|---|---|---|---|---|---|
| val\_seen | RNNT | | 49,9 | 17,7 | 24,8 | 20,0 | chưa đo |
| | C0 | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| | Ứng viên thắng | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| val\_unseen | RNNT | | 51,6 | 17,5 | 24,1 | 19,8 | chưa đo |
| | C0 | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| | Ứng viên thắng | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

Hàng C0 trên val dùng câu mồi thắng của E, nên có thể khác số của lần 2.

**Giới hạn.** Sáu ứng viên được so trên 300 mẫu, nên ứng viên thắng trên tập dò vẫn có thể thắng một phần nhờ may; số val
đo một lần là số để tin. Độ chính xác quyết định đoạn coi lấy nhầm và bỏ sót nặng như nhau, trong khi lấy nhầm một từ
tiếng Anh có thể làm LLM gọi sai tool; hai loại sai được báo riêng (`merge_wrong_english`, `merge_missed_english`) để
thấy ứng viên thắng nghiêng về phía nào. Nhãn tự động dựa vào đáp án và cách gióng từ, nên có thể sai ở đoạn mà hai phía
cùng gần đúng.

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

- **(sẽ viết)** `lifemate/backends/asr_prompt.py` › `build_sample_prompts(sample, catalog_apps, fixed_prompt, variant)`:
nhận một mẫu, catalog (từ điển mã app → mục catalog), câu mồi cố định và tên biến thể; trả về danh sách câu mồi, một
câu mồi cho mỗi câu người dùng theo đúng thứ tự file âm thanh của `ASRTranscriber.wav_paths` (lịch sử trước, lượt cuối
sau). Hàm chỉ đọc `sample.context`, `sample.history` (lượt trợ lý) và catalog; không đọc lượt người dùng và không đọc
file môi trường của BTC.
- **(sẽ sửa)** `lifemate/backends/asr.py` › `FasterWhisperASR.transcribe_batch(wav_paths, prompts=None)`:
nhận thêm câu mồi riêng cho từng file. `ASRTranscriber` gọi `build_sample_prompts` khi
`[asr].prompt_mode = "sample"` (khoá config mới; mặc định `"fixed"` giữ đúng hành vi cũ).
- **(sẽ sửa)** `hyps.jsonl` có thêm trường `prompt` (câu mồi đã dùng cho câu đó) và `scores` (`avg_logprob`,
`no_speech_prob` của từng đoạn, cho thử nghiệm G). File kết quả không lên git, nên chứa chữ của dữ liệu cũng được.

**Lớp bảo vệ khi câu mồi có lời trợ lý.** Lớp bảo vệ g4 của lần 2 coi câu có 4 từ liên tiếp trùng câu mồi là "in lại câu
mồi" và chép lại không câu mồi. Với câu mồi có lời trợ lý, người dùng hay **nhắc lại đúng lời trợ lý** ("Gửi cho Sarah
hay Sara?" → "gửi cho Sarah"), nên so với toàn bộ câu mồi sẽ bắt nhầm câu đúng. Vì vậy ở E1 và E2, lớp bảo vệ chỉ so với
**phần cố định** (`en_full_v1`), cộng kiểm tra tỉ lệ nén như cũ (bắt câu lặp). Việc in lại lời trợ lý được đo riêng bằng
`assistant_echo` (mục 2.2), không dùng để chép lại. Code: **(sẽ sửa)** `lifemate/backends/asr_guard.py` › `needs_retry`
nhận câu mồi phần cố định.

**Phép ghép với câu mồi theo mẫu.** `context_other` nhận mọi từ của câu mồi. Với câu mồi theo mẫu, "từ của câu mồi" gồm
từ của phần cố định và từ của phần riêng; từ của phần riêng vốn đã có trong từ điển ngữ cảnh, nên quy tắc không đổi.
**(sẽ sửa)** `lifemate/eval/asr_merge.py` đọc câu mồi của từng câu từ trường `prompt` của `hyps.jsonl` thay vì một file
câu mồi chung.

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

**Tập dò (`asr_tune`), large-v3 đứng riêng:**


| Biến thể | Chép đúng từ tiếng Anh | Từ có trong ngữ cảnh | Từ không có trong ngữ cảnh | WER câu thuần Việt | Chèn từ câu mồi | Chép lại lời trợ lý | Chép lại (lớp bảo vệ) | Câu bị cắt câu mồi | p95 (s) |
| -------- | ---------------------- | -------------------- | -------------------------- | ------------------ | --------------- | ------------------- | --------------------- | ------------------ | ------- |
| E0       | chưa đo                | chưa đo              | chưa đo                    | chưa đo            | chưa đo         | chưa đo             | chưa đo               | 0                  | chưa đo |
| E1       | chưa đo                | chưa đo              | chưa đo                    | chưa đo            | chưa đo         | chưa đo             | chưa đo               | chưa đo            | chưa đo |
| E2       | chưa đo                | chưa đo              | chưa đo                    | chưa đo            | chưa đo         | chưa đo             | chưa đo               | chưa đo            | chưa đo |


**Tập dò, bản ghép (RNNT + biến thể, `context_other`), chênh lệch so với E0 kèm CI 95%:**


| Biến thể | Chép đúng từ tiếng Anh | WER câu thuần Việt | WER câu chêm | WER mọi câu |
| -------- | ---------------------- | ------------------ | ------------ | ----------- |
| E0       | chưa đo                | chưa đo            | chưa đo      | chưa đo     |
| E1       | chưa đo                | chưa đo            | chưa đo      | chưa đo     |
| E2       | chưa đo                | chưa đo            | chưa đo      | chưa đo     |


**Tập kiểm tra, biến thể thắng (đo một lần):**


| Tập         | Hệ thống             | Chép đúng từ tiếng Anh | Từ có / không có trong ngữ cảnh | WER câu thuần Việt | WER câu chêm | WER mọi câu |
| ----------- | -------------------- | ---------------------- | ------------------------------- | ------------------ | ------------ | ----------- |
| val\_seen   | Ghép, E0 (lần 2)     | 72,2                   | chưa đo                         | 18,1               | 22,1         | 19,4        |
|             | Ghép, biến thể thắng | chưa đo                | chưa đo                         | chưa đo            | chưa đo      | chưa đo     |
| val\_unseen | Ghép, E0 (lần 2)     | 63,1                   | chưa đo                         | 17,8               | 22,8         | 19,5        |
|             | Ghép, biến thể thắng | chưa đo                | chưa đo                         | chưa đo            | chưa đo      | chưa đo     |


### 3.8. Phân tích

*Điền sau khi đo.*

### 3.9. Kết luận

*Điền sau khi đo.*

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
không. Bảng dưới là số minh hoạ, chưa phải số đo:

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
θ đã chọn. Tập dò không dùng để chọn θ, vì tập dò để dành cho bước chọn chung (mục 2.5).

**Code.**

- **(sẽ viết)** `lifemate/eval/prompt_word_reliability.py` › `count_prompt_words(samples, primary_rows, secondary_rows)`:
nhận các mẫu và hai danh sách dòng `hyps.jsonl` (RNNT, large-v3); trả về từ điển `từ đã chuẩn hoá → (written, correct)`.
`reliability(counts)` trả về từ điển `từ đã chuẩn hoá → độ tin cậy`. CLI ghi bảng ra JSON (không lên git, vì tên từ lấy từ câu
mồi tự viết nhưng số đếm lấy từ dữ liệu thi).
- **(sẽ sửa)** `lifemate/eval/asr_merge.py` › `english_filter(policy, context_words, prompt_words, reliability=None, threshold=None)`: thêm `context_reliable` vào `CONTEXT_POLICIES`.

### 4.4. Lý do

- **Đếm và chọn θ trên tập học, chọn θ bằng kiểm định chéo.** Nếu đếm và chấm θ trên cùng các câu, bảng độ tin cậy khớp
sẵn với chính các câu dùng để chọn; kiểm định chéo chấm mỗi phần bằng bảng đếm trên các phần khác.
- **Đếm với đúng cấu hình câu mồi sẽ dùng.** Độ tin cậy của một từ phụ thuộc câu mồi: cùng từ "Mail" có thể bị kéo ra
nhiều hơn khi câu mồi có nhiều câu về email. Vì vậy đếm trên chữ large-v3 với biến thể thắng của E.
- **Chỉ đếm khi RNNT không có từ đó.** Khi RNNT cũng có từ đó, phép ghép không lấy gì từ large-v3, nên các lần đó không
nói gì về quyết định cần học.

### 4.5. Dữ liệu và số lần đo

- Đếm và chọn θ: tập học (793 mẫu, 2 569 câu), chữ RNNT và chữ large-v3 biến thể thắng của E, phiên K2.
- F với θ đã chọn vào bảng chọn chung trên `asr_tune` (mục 2.5); chỉ đo trên val nếu thắng.

### 4.6. Chỉ số và tiêu chí

Chọn θ: độ chính xác quyết định đoạn trung bình qua 5 phần của tập học (mục 4.3). Báo kèm cho từng θ: `merge_wrong_english`
tách theo từ của câu mồi, `merge_missed_english`. Tiêu chí so F với các ứng viên khác: bảng chọn chung (mục 2.5).

### 4.7. Kết quả

**Bảng độ tin cậy trên tập học** (các từ của câu mồi có `written` từ 3 trở lên):


| Từ      | Số lần large-v3 viết ra khi RNNT không có | Số lần đúng | Độ tin cậy |
| ------- | ----------------------------------------- | ----------- | ---------- |
| chưa đo |                                           |             |            |


**Tập học, kiểm định chéo 5 phần, chọn θ** (trung bình qua 5 phần):


| θ               | Độ chính xác quyết định đoạn | Từ câu mồi bị lấy nhầm | Từ câu mồi đúng bị bỏ |
| --------------- | ---------------------------- | ---------------------- | --------------------- |
| `context_other` | chưa đo                      | chưa đo                | chưa đo               |
| 0,3             | chưa đo                      | chưa đo                | chưa đo               |
| 0,4             | chưa đo                      | chưa đo                | chưa đo               |
| 0,5             | chưa đo                      | chưa đo                | chưa đo               |
| 0,6             | chưa đo                      | chưa đo                | chưa đo               |
| 0,7             | chưa đo                      | chưa đo                | chưa đo               |
| 0,8             | chưa đo                      | chưa đo                | chưa đo               |


Kết quả của F trên tập dò và trên val nằm trong bảng chọn chung (mục 2.5).

### 4.8. Phân tích

*Điền sau khi đo.*

### 4.9. Kết luận

*Điền sau khi đo.*

### 4.10. Giới hạn

- Từ của câu mồi ít gặp trên tập học có độ tin cậy gần 0,5 do làm mượt, nên θ = 0,5 là ranh giới nhạy: θ dưới 0,5 nhận
mọi từ chưa gặp, θ trên 0,5 đòi ngữ cảnh cho mọi từ chưa gặp.
- Khi câu mồi theo mẫu thắng ở E, phần riêng của câu mồi thay đổi theo mẫu; bảng độ tin cậy chỉ đếm từ của phần cố định,
vì từ của phần riêng luôn có trong ngữ cảnh.

## 5. Thử nghiệm G: bộ phân loại cho quyết định ghép

### 5.1. Vấn đề

Quy tắc ghép hiện tại là luật viết tay: lấy đoạn của large-v3 khi có từ tiếng Anh mới, đoạn không quá dài, và từ đó qua
bộ lọc. Mỗi lần thêm một tín hiệu (ngữ cảnh, độ tin cậy) lại phải viết thêm một nhánh luật và chọn tay cách kết hợp. Có
những tín hiệu khác chưa dùng: điểm âm thanh của hai phương án (J), large-v3 chắc chắn tới đâu (`avg_logprob`),
đoạn của RNNT có phải âm tiết Việt có nghĩa không.

### 5.2. Ý tưởng

Coi mỗi đoạn khác nhau là một câu hỏi có/không: "lấy chữ của large-v3 thì đoạn này đúng hơn không?". Câu trả lời đúng tính
được máy móc trên tập học bằng cách so với đáp án. Vậy có thể **học** cách kết hợp các tín hiệu thay vì viết tay. Mô hình
học là hồi quy logistic: mỗi tín hiệu có một trọng số, cộng lại rồi đổi thành xác suất. Mô hình nhỏ như vậy học được từ
vài trăm đoạn, và trọng số đọc được (biết tín hiệu nào quan trọng).

### 5.3. Giải pháp và thuật toán

**Mẫu học.** Mỗi đoạn khác nhau mà large-v3 có ít nhất một từ tiếng Anh mới (đúng các đoạn mà bản ghép không lọc `all` cân nhắc
lấy). Các đoạn khác giữ chữ RNNT như cũ.

**Nhãn tự động.** Với mỗi đoạn, gióng chữ ghép với đáp án của câu và đếm số lỗi (thêm, xoá, thay) rơi vào đoạn đó trong
hai trường hợp: giữ RNNT và lấy large-v3. Nhãn là 1 khi lấy large-v3 ít lỗi hơn, 0 khi nhiều lỗi hơn. Đoạn hoà (cùng số
lỗi) được bỏ khỏi dữ liệu học, vì không có câu trả lời đúng.

**Đặc trưng** (mỗi đoạn một vector số):


| Đặc trưng                                          | Giá trị                        | Lấy từ                                |
| -------------------------------------------------- | ------------------------------ | ------------------------------------- |
| Có từ tiếng Anh mới trong từ điển ngữ cảnh         | 0/1                            | `sample_context_words` (lần 2)        |
| Có từ tiếng Anh mới thuộc câu mồi                  | 0/1                            | Trường `prompt` của `hyps.jsonl`      |
| Độ tin cậy thấp nhất của các từ câu mồi trong đoạn | 0–1; 1 nếu không có từ câu mồi | Bảng của F                            |
| `avg_logprob` của đoạn âm thanh chứa câu đó        | số âm                          | Trường `scores`                       |
| `no_speech_prob` của đoạn âm thanh                 | 0–1                            | Trường `scores`                       |
| Số từ phía RNNT, số từ phía large-v3               | số nguyên                      | Phép gióng                            |
| Tỉ lệ từ phía RNNT là âm tiết Việt có dấu          | 0–1                            | Quy tắc tìm từ ngoại (lần 2)          |
| Câu là lượt cuối hay lượt trong lịch sử            | 0/1                            | `hyps.jsonl`                          |
| `Δâm_thanh`: điểm âm thanh câu B trừ câu A         | số                             | Thử nghiệm J (phiên K3)               |


**Học.** Hồi quy logistic có phạt L2, chuẩn hoá đặc trưng về trung bình 0 và độ lệch 1 trên tập học. Hệ số phạt chọn bằng
kiểm định chéo 5 phần **theo mẫu** trên tập học (các đoạn của cùng một mẫu luôn nằm cùng một phần), trong các giá trị
0,01; 0,1; 1; 10. Cài bằng NumPy (phương pháp Newton), không thêm thư viện, chạy tất định.

**Quyết định.** Lấy chữ của large-v3 khi xác suất từ 0,5 trở lên (ngưỡng tự nhiên của hồi quy logistic, không dò). Điều kiện độ dài cố định của lần 2 (đoạn large-v3 không dài quá gấp đôi đoạn RNNT cộng 2 từ) không
dùng nữa: số từ hai phía đã là đặc trưng, và điều kiện đó chỉ chặn 1 trong 518 đoạn ở lần 2.

**Code.**

- **(sẽ viết)** `lifemate/eval/merge_classifier.py` › `region_features(sample, primary, secondary, scores, candidate_scores, context_words, prompt_words, reliability)`: trả về danh sách đoạn, mỗi đoạn kèm vector đặc trưng.
- `region_labels(regions, reference)`: trả về nhãn 0/1 hoặc `None` (đoạn hoà).
- `fit_logistic(features, labels, l2)`: trả về trọng số và tham số chuẩn hoá. `predict(model, features)`: xác suất.
- CLI `train` (ghi mô hình ra JSON, vài chục số) và `apply` (ghép với bộ phân loại, ghi `hyps.jsonl`).
- **(sẽ sửa)** `lifemate/eval/asr_merge.py` › `merge_transcripts` nhận một hàm quyết định cho từng đoạn, để luật và bộ phân
loại dùng chung một code gióng.

### 5.4. Lý do

- **Hồi quy logistic, không phải MLP hay attention.** Số đoạn học được ước tính chỉ vài trăm tới khoảng một nghìn: trên
tập dò của lần 2, bản ghép không lọc lấy khoảng 260 từ tiếng Anh mới từ large-v3, và tập học lớn gấp khoảng 2,5 lần tập dò. Mô hình nhiều tham số sẽ học thuộc tập học. Hồi quy logistic có
khoảng 10 trọng số, đọc được, và có thể chuyển về luật nếu cần.
- **Bỏ đoạn hoà.** Đoạn hoà thường là hai cách viết cùng sai hoặc cùng đúng một từ; ép nhãn cho chúng chỉ thêm nhiễu.
- **Kiểm định chéo theo mẫu.** Các câu cùng mẫu cùng người nói và cùng ngữ cảnh; chia theo đoạn sẽ cho kết quả kiểm định
lạc quan.
- **Dùng điểm của J, không chép lại large-v3 không câu mồi.** Chép lại không câu mồi chỉ cho biết large-v3 không câu mồi
chọn gì và tốn một lượt giải mã đầy đủ; điểm âm thanh của J cho điểm của cả hai phương án với chi phí thấp hơn (mục 8.4).

### 5.5. Dữ liệu và số lần đo

- Học: tập học (phiên K2: RNNT, large-v3 biến thể thắng của E; phiên K3: điểm của J; bảng độ tin cậy của F).
- G vào bảng chọn chung trên `asr_tune` (mục 2.5); chỉ đo trên val nếu thắng.

### 5.6. Chỉ số và tiêu chí

Chỉ số chính: WER câu thuần Việt và chép đúng từ tiếng Anh của bản ghép. Chỉ số đi kèm: `merge_wrong_english`,
`merge_missed_english`, độ chính xác quyết định đoạn trong kiểm định chéo trên tập học, trọng số của
từng đặc trưng.

Tiêu chí so G với các ứng viên khác: bảng chọn chung (mục 2.5).

### 5.7. Kết quả

**Dữ liệu học:**


|            | Số đoạn | Nhãn 1 (lấy large-v3 đúng hơn) | Nhãn 0  | Đoạn hoà (bỏ) |
| ---------- | ------- | ------------------------------ | ------- | ------------- |
| Tập học    | chưa đo | chưa đo                        | chưa đo | chưa đo       |
| `asr_tune` | chưa đo | chưa đo                        | chưa đo | chưa đo       |


**Trọng số** (sau chuẩn hoá; dương là nghiêng về lấy large-v3):


| Đặc trưng | Trọng số |
| --------- | -------- |
| chưa đo   |          |


Kết quả của G trên tập dò và trên val nằm trong bảng chọn chung (mục 2.5).

### 5.8. Phân tích

*Điền sau khi đo.*

### 5.9. Kết luận

*Điền sau khi đo.*

### 5.10. Giới hạn

- Nhãn tự động dựa vào đáp án của BTC và cách gióng từ; một đoạn có thể đúng về ý (LLM vẫn gọi đúng tool) mà vẫn có nhãn 0
vì sai chính tả.
- Mục tiêu học là ít lỗi chữ hơn, không phải gọi đúng tool hơn. Hai mục tiêu thường cùng chiều nhưng không trùng nhau; lần
chạy hệ thống với LLM sẽ kiểm tra.

## 6. Thử nghiệm H: giữ hai phương án cho LLM

### 6.1. Vấn đề

Mọi phép ghép ở trên đều phải chốt một bản chữ. Ở một đoạn khác nhau mà bằng chứng không đủ, phép ghép bỏ một phương án,
dù LLM có thể chọn đúng nhờ thông tin mà ASR không thấy: kết quả tool (danh bạ, thiết bị), và việc nhìn cả câu cùng danh
sách app. Ví dụ, LLM thấy không có app nào tên "giá lô" nhưng Zalo có trong danh sách app đã cài.

### 6.2. Ý tưởng

Ở các đoạn mà phép ghép **không chắc**, viết cả hai phương án vào chữ đưa cho LLM, ví dụ `nhắn {giá lô | Zalo} cho mẹ`.
Trước khi đưa định dạng này vào lần chạy hệ thống và vào dữ liệu fine-tune LLM, cần biết hai con số: có bao nhiêu từ đúng
chỉ cứu được bằng cách này (trần), và tốn bao nhiêu token.

### 6.3. Giải pháp và thuật toán

**Đoạn không chắc.** Đoạn khác nhau có từ tiếng Anh mới phía large-v3 (như G), mà:

- nếu ứng viên thắng của bảng chọn chung (mục 2.5) là luật (C0 hoặc F): từ đó bị bộ lọc loại (phép ghép giữ RNNT);
- nếu ứng viên thắng là J: hiệu điểm nhỏ hơn 2,3 về độ lớn, tức phương án hơn chỉ có khả năng cao gấp dưới khoảng 10 lần
  phương án kia (mục 8.3);
- nếu ứng viên thắng là G: xác suất nằm trong khoảng `[0,2; 0,8]`.

Ở đoạn không chắc, chữ đưa cho LLM là `{<chữ RNNT> | <chữ large-v3>}`; đoạn khác giữ như phép ghép đã chốt.

**Trần.** Tỉ lệ từ tiếng Anh của đáp án có mặt trong chữ đã chốt **hoặc** trong phương án thứ hai của một đoạn không chắc
cùng câu. Trần trừ đi tỉ lệ chép đúng của bản đã chốt là phần **tối đa** LLM có thể cứu thêm.

**Mặt trái.** Số từ tiếng Anh sai được đưa thêm vào dưới dạng phương án thứ hai (LLM có thể chọn nhầm), và tỉ lệ câu thuần
Việt có ít nhất một đoạn không chắc.

**Chi phí token.** Đếm số token thêm vào mỗi mẫu bằng tokenizer của Qwen3-4B (LLM của hệ thống), so với chữ đã chốt. Chỉ
nạp tokenizer, không chạy mô hình.

**Code.** **(sẽ viết)** `lifemate/eval/asr_alternatives.py` › `format_alternatives(primary, secondary, uncertain)`: trả về
chữ có dạng hai phương án. `alternatives_ceiling(samples, rows)`: trả về trần, số từ sai đưa thêm, số câu có đoạn không
chắc. `alternatives_tokens(rows, tokenizer)`: trả về p50, p95, tối đa của số token thêm mỗi mẫu.

### 6.4. Lý do

- **Chỉ đo trần, chưa đo LLM.** Lần 3 không chạy LLM (phạm vi). Nếu trần chỉ cao hơn bản đã chốt vài điểm thì không đáng
thêm định dạng mới vào dữ liệu fine-tune.
- **Chỉ đưa hai phương án ở đoạn không chắc.** Đưa ở mọi đoạn khác nhau thì số token tăng và LLM phải chọn cả ở những chỗ
phép ghép đã chắc.
- **Khoảng `[0,2; 0,8]` và hiệu điểm 2,3.** Ngoài các khoảng này phép quyết định đủ chắc để chốt. Hai con số đặt trước,
không dò, và tương đương nhau: xác suất 0,8 ứng với tỉ số khả năng 4, còn hiệu 2,3 ứng với tỉ số 10, nên khoảng của J rộng hơn để bù việc điểm của J chưa hiệu chỉnh thành xác suất. Khoảng này đặt trước, không dò; nếu thử nghiệm
H được đưa vào hệ thống, khoảng sẽ được chọn lại bằng điểm của agent.

### 6.5. Dữ liệu

`asr_tune`, `val_seen`, `val_unseen`, dùng chữ đã có sau bước chọn chung. Không chọn gì nên đo thẳng trên val.

### 6.6. Kết quả


| Tập         | Chép đúng, bản đã chốt | Trần với hai phương án | Từ sai đưa thêm | Câu thuần Việt có đoạn không chắc | Token thêm mỗi mẫu: p50 / p95 / tối đa |
| ----------- | ---------------------- | ---------------------- | --------------- | --------------------------------- | -------------------------------------- |
| `asr_tune`  | chưa đo                | chưa đo                | chưa đo         | chưa đo                           | chưa đo                                |
| val\_seen   | chưa đo                | chưa đo                | chưa đo         | chưa đo                           | chưa đo                                |
| val\_unseen | chưa đo                | chưa đo                | chưa đo         | chưa đo                           | chưa đo                                |


### 6.7. Phân tích

*Điền sau khi đo.*

### 6.8. Kết luận

*Điền sau khi đo.*

### 6.9. Giới hạn

- Trần giả định LLM luôn chọn đúng. LLM thật chọn sai một phần, nên lợi ích thật thấp hơn trần.
- LLM chưa từng thấy định dạng `{a | b}`; dùng mà không fine-tune có thể làm LLM chép nguyên dấu ngoặc vào tham số tool.

## 7. Thử nghiệm I: thời gian và bộ nhớ

### 7.1. Vấn đề

Thể lệ cho 30 giây mỗi mẫu cho cả ASR, LLM và tool; mẫu quá giờ nhận 0 điểm, và khi bằng điểm thì đội nhanh hơn xếp
trên. Phần ASR của bản ghép chạy hai mô hình trên mọi câu người dùng của mẫu (tối đa 13 câu, tới 82,8 giây âm thanh ở
`public_test`). Trên T4, lần 2 đo riêng từng mô hình: RNNT p95 2,5 giây, large-v3 p95 8,6–8,8 giây, tối đa 10,8 giây
mỗi mẫu. Chưa biết tổng khi chạy cùng nhau, chưa biết trên A30, và chưa biết bộ nhớ GPU còn lại cho vLLM.

### 7.2. Ngân sách và lý do của hai ngưỡng

**Thời gian: phần ASR của mẫu chậm nhất không quá 10 giây trên A30.**

- Ngân sách tính theo mẫu chậm nhất, không theo trung bình hay p95, vì một mẫu quá 30 giây là mất trọn điểm mẫu đó.
- 10 giây là một phần ba của 30 giây. Hai phần ba còn lại dành cho LLM và tool, vì agent thường cần nhiều lượt sinh
trong một mẫu (gọi tool, đọc kết quả, rồi gọi tiếp), và thời gian của LLM trên A30 chưa được đo.
- Ngưỡng này đặt trước và sẽ được xem lại ở lần chạy hệ thống với LLM, khi đo được thời gian thật của LLM.

**Bộ nhớ: phần ASR chiếm không quá 6 GB GPU.**

- A30 có 24 GB. Trọng số Qwen3-4B ở bfloat16 chiếm khoảng 8 GB (4,02 tỉ tham số × 2 byte). Phần còn lại dành cho bộ nhớ
đệm KV của vLLM (cần cho ngữ cảnh tới 16K token) và phần dư của CUDA.
- 6 GB cho ASR để lại khoảng 10 GB cho bộ nhớ đệm KV và phần dư. Trên T4 (15 GB) vLLM và large-v3 không chạy chung được
(`ops/kaggle/README.md`), nên con số trên A30 phải đo thật.

### 7.3. Các cách chạy được so


| Cách chạy                    | Mô tả                                                                                           | Có thể đổi chữ không                       | Code                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| I1 tuần tự                   | RNNT chép hết các câu của mẫu, rồi large-v3 chép từng câu, rồi lớp bảo vệ, rồi ghép             | Không (đây là chuẩn để so)                 | **(sẽ viết)** engine ghép `MergedASR` trong `lifemate/backends/asr.py` |
| I2 song song                 | RNNT và large-v3 chạy cùng lúc, mỗi mô hình một luồng, trên cùng GPU                            | Không nên đổi; kiểm tra bằng so chữ        | Như I1, thêm `ThreadPoolExecutor` hai luồng                            |
| I3 theo lô                   | large-v3 nhận mọi câu của mẫu một lúc (`BatchedInferencePipeline` của faster-whisper)           | Có thể: cách chia đoạn và thứ tự tính khác | Như I1, khoá config chọn pipeline theo lô                              |
| I4 `int8_float16`            | large-v3 lưu trọng số 8 bit (`compute_type`), tính bằng 16 bit                                  | Có: độ chính xác số khác                   | Đổi `[asr].dtype` của large-v3                                         |
| I5 cạnh vLLM                 | Cách chạy tốt nhất trong I1–I4, chạy khi vLLM đã nạp Qwen3-4B trên cùng GPU với cấu hình bộ nhớ của hệ thống thật | Không | Script R1 khởi động vLLM trước engine ghép |


Cách nào đổi chữ thì phải đo lại chất lượng (chép đúng từ tiếng Anh, WER câu thuần Việt của bản ghép) trên `asr_tune`, và
chỉ được dùng khi chất lượng không giảm có ý nghĩa thống kê.

### 7.4. Lý do

- **Song song (I2).** RNNT nhỏ (0,11 tỉ tham số) và nhanh; chạy cạnh large-v3 có thể giấu gần hết thời gian của RNNT nếu
GPU còn rảnh. Chữ không nên đổi vì mỗi mô hình vẫn tính như cũ, nhưng phải kiểm tra vì thứ tự tính trên GPU có thể khác.
- **Theo lô (I3).** large-v3 hiện chép từng câu một; một mẫu có tới 13 câu. Lô nhiều câu dùng GPU hiệu quả hơn, nhưng cách
chia đoạn bên trong pipeline theo lô khác pipeline thường, nên chữ có thể đổi.
- **8 bit (I4).** Giảm bộ nhớ và có thể nhanh hơn; đổi lại độ chính xác số giảm, phải đo lại chữ.
- **Nạp vLLM ở R1 (I5).** Ngưỡng 6 GB cho ASR là con số đặt trước; chỉ khi vLLM đã giữ phần bộ nhớ dành cho vLLM mới biết ASR
có thật sự vừa không. Agent vẫn chưa chạy; tranh GPU khi agent sinh chữ đo ở lần chạy hệ thống.

### 7.5. Dữ liệu và phiên chạy

- T4 (phiên K1, K2): mọi biến thể của E và mọi lần chạy trên tập học và val đều ghi `timing.jsonl`; chỉ dùng để so chi phí
giữa các biến thể.
- A30 hoặc GPU 24 GB gần nhất (phiên R1): cấu hình cuối sau bước chọn chung, các cách chạy I1–I5 trên `public_test` (634 mẫu) và
`val_seen` + `val_unseen` (452 mẫu); chạy lại 50 mẫu để kiểm tra tất định; ghi bộ nhớ đỉnh.

### 7.6. Chỉ số và tiêu chí

Chỉ số: thời gian ASR mỗi mẫu (p50, p95, tối đa, trung bình), thời gian từng bước, RTF, thời gian nạp mô hình, bộ nhớ GPU
đỉnh (riêng ASR, và tổng khi có vLLM ở I5), số câu khác chữ so với I1.

Tiêu chí (đặt trước): trên A30, chọn trong các cách chạy đạt cả hai ngưỡng ở mục 7.2 và cho chữ giống hệt I1 (hoặc, với
I3 và I4, chất lượng không giảm có ý nghĩa thống kê), cách có thời gian trung bình mỗi mẫu thấp nhất. Không cách nào đạt
ngưỡng thời gian thì ghi lại khoảng thiếu, và xét các phương án giảm việc: dùng F thay cho ứng viên cần chấm điểm (mục
2.5), bỏ large-v3 ở lượt lịch sử. I5 phải chạy được mà không lỗi hết bộ nhớ GPU.

### 7.7. Kết quả

**T4, theo biến thể câu mồi (phiên K1, `asr_tune`):**


| Biến thể | large-v3 p50 / p95 / tối đa (s) | Lớp bảo vệ: số câu chép lại, giây thêm | RNNT p95 (s) | Ghép p95 (s) | Tổng p95 / tối đa (s) |
| -------- | ------------------------------- | -------------------------------------- | ------------ | ------------ | --------------------- |
| E0       | chưa đo                         | chưa đo                                | chưa đo      | chưa đo      | chưa đo               |
| E1       | chưa đo                         | chưa đo                                | chưa đo      | chưa đo      | chưa đo               |
| E2       | chưa đo                         | chưa đo                                | chưa đo      | chưa đo      | chưa đo               |


**A30 (phiên R1), cấu hình cuối:**


| Cách chạy         | Tập           | p50 / p95 / tối đa / trung bình (s) | RTF     | Bộ nhớ GPU đỉnh (GB) | Câu khác chữ so với I1 | Nạp mô hình (s) |
| ----------------- | ------------- | ----------------------------------- | ------- | -------------------- | ---------------------- | --------------- |
| RNNT đứng riêng   | `public_test` | chưa đo                             | chưa đo | chưa đo              |                        | chưa đo         |
| I1 tuần tự        | `public_test` | chưa đo                             | chưa đo | chưa đo              | 0 (chuẩn)              | chưa đo         |
| I2 song song      | `public_test` | chưa đo                             | chưa đo | chưa đo              | chưa đo                | chưa đo         |
| I3 theo lô        | `public_test` | chưa đo                             | chưa đo | chưa đo              | chưa đo                | chưa đo         |
| I4 `int8_float16` | `public_test` | chưa đo                             | chưa đo | chưa đo              | chưa đo                | chưa đo         |
| I5 cạnh vLLM      | `public_test` | chưa đo                             | chưa đo | chưa đo (ASR / tổng) | chưa đo                | chưa đo         |
| Cách thắng        | val           | chưa đo                             | chưa đo | chưa đo              | chưa đo                | chưa đo         |


**Kiểm tra tất định (50 mẫu, chạy hai lần):**


| Cách chạy | Câu khác chữ giữa hai lần |
| --------- | ------------------------- |
| I1 – I5   | chưa đo                   |


### 7.8. Phân tích

*Điền sau khi đo.*

### 7.9. Kết luận

*Điền sau khi đo.*

### 7.10. Giới hạn

- vLLM chỉ được nạp để giữ bộ nhớ, agent chưa chạy (mục 2.4).
- `nvidia-smi` đọc bộ nhớ của cả GPU, không riêng tiến trình; phiên đo phải không có tiến trình GPU nào khác.
- Thời gian nạp mô hình không tính vào từng mẫu ở lần 3. Cách BTC tính giờ khi chạy Docker (từ lúc nạp mô hình hay từ mẫu
đầu tiên) cần đối chiếu lại với `docs/overview.md` ở lần chạy hệ thống.

## 8. Thử nghiệm J: chấm điểm hai phương án bằng âm thanh

### 8.1. Vấn đề

Mọi quyết định ghép ở lần 2 và ở F nhìn vào **chữ** của hai bản rồi áp một luật: từ có trong câu mồi không, có trong từ
điển ngữ cảnh không, đoạn có dài quá `2 × (số từ RNNT) + 2` không. Câu hỏi thật lại là câu hỏi về **âm thanh**: "người
dùng có nói 'Zalo' ở chỗ này không?". Chỉ nhìn chữ thì phải đoán bằng luật, và mỗi luật kèm một con số tự đặt. Ngưỡng độ
dài là ví dụ: số đặt tay trước khi đo, và trên thực tế chỉ chặn 1 trong 518 đoạn (mục 6.8 của báo cáo lần 2). G học
trọng số từ dữ liệu thay cho luật viết tay, nhưng nếu không có J thì các đặc trưng của G vẫn chủ yếu là tín hiệu về mặt
chữ.

### 8.2. Ý tưởng

Hỏi thẳng mô hình âm thanh. Ở mỗi đoạn khác nhau có hai phương án: A (chữ RNNT) và B (chữ large-v3). Dựng hai câu hoàn
chỉnh chỉ khác nhau ở đoạn đó, rồi cho whisper-large-v3 **không câu mồi** tính xác suất "câu này là lời của đoạn âm thanh
này" cho từng câu. Chọn câu có xác suất cao hơn. Cách này không có ngưỡng tự đặt, vì đây là so sánh hai xác suất do cùng
một mô hình tính trên cùng âm thanh.

### 8.3. Giải pháp và thuật toán

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
  cao gấp khoảng 10 lần câu A. Vì vậy G dùng hiệu điểm làm đặc trưng.

CTranslate2 (lõi của faster-whisper) có sẵn hàm `align(encoder_output, start_sequence, text_tokens, num_frames)` trả về
`text_token_probs`, tức xác suất từng token của một chuỗi chữ cho sẵn. faster-whisper 1.2.1 dùng chính hàm này để tính
`word.probability` (`transcribe.py` › `find_alignment`, với `start_sequence` là các token mở đầu, không có câu mồi). Vậy
J không cần nạp thêm mô hình và không phải chép lại cả câu: chỉ cần chạy encoder một lần cho câu nói đó và một lượt
decoder ép chuỗi cho mỗi câu ứng viên.

**Dựng câu ứng viên.** Bắt đầu từ chữ RNNT của câu. Với mỗi đoạn khác nhau có từ tiếng Anh mới (đúng các
đoạn mà G xét), dựng câu A (đoạn đó là chữ RNNT) và câu B (đoạn đó là chữ large-v3); phần còn lại của hai câu giống hệt
nhau. Cả hai câu được đưa về cùng một kiểu viết trước khi chấm: chữ thường, bỏ dấu câu, như chữ RNNT. Lý do: large-v3
quen viết hoa và có dấu câu; nếu chỉ câu B giữ kiểu viết đó thì câu B được lợi chỉ vì kiểu viết. Một câu có nhiều đoạn
khác nhau được chấm lần lượt từng đoạn, từ trái sang phải, đoạn sau dựng trên kết quả của đoạn trước.

**Quyết định.** Lấy chữ large-v3 khi `điểm_âm_thanh(B) > điểm_âm_thanh(A)`.

**Dùng trong G và H.** `Δâm_thanh = điểm_âm_thanh(B) − điểm_âm_thanh(A)` được thêm vào đặc trưng của G. Trong G, điều
kiện độ dài cố định của lần 2 không dùng nữa, vì số từ hai phía đã là đặc trưng. H lấy đoạn không chắc theo ứng viên thắng
của bảng chọn chung (mục 6.3).

**Câu dài hơn 30 giây.** Encoder chỉ nhận một cửa sổ 30 giây. Câu nói dài hơn 30 giây giữ quyết định của F, không chấm;
số câu như vậy được đếm và ghi vào kết quả.

**Code.**

- **(sẽ viết)** `lifemate/eval/candidate_scoring.py` › `candidate_pairs(primary, secondary)`: nhận chữ RNNT và chữ large-v3 của
  một câu, trả về danh sách (đoạn, câu A, câu B) đã đưa về cùng kiểu viết.
- **(sẽ viết)** `lifemate/backends/asr.py` › `FasterWhisperASR.score_texts(audio, texts)`: nhận âm thanh một câu và các câu
  ứng viên, trả về điểm âm thanh của từng câu (`model.encode` một lần, `model.align` không câu mồi cho từng câu).
- **(sẽ viết)** `lifemate/eval/candidate_scoring.py` › `decide_j(scores_a, scores_b)`: trả về quyết định cho từng đoạn.

### 8.4. Lý do

- **Chấm không câu mồi.** Phần lớn từ lấy nhầm là do câu mồi kéo hoặc do large-v3 đoán thành tiếng Anh (mục 6.8 của báo
  cáo lần 2). Khi chấm không câu mồi, lực kéo của câu mồi không còn; điểm chỉ phản ánh âm thanh và thói quen ngôn ngữ của
  chính large-v3. Câu mồi vẫn dùng khi **chép**, vì câu mồi giúp large-v3 nghĩ ra phương án "Zalo" ngay từ đầu.
- **Chấm, không chép lại.** Chép lại không câu mồi (đã bỏ khỏi kế hoạch lần 3) tốn một lượt giải mã đầy đủ và chỉ cho
  biết large-v3 không câu mồi chọn gì, không cho biết phương án của RNNT được bao nhiêu điểm. Chấm ép chuỗi cho điểm của
  **cả hai** phương án, và chỉ chạy trên câu có đoạn cần quyết định.
- **Không dùng LLM để chấm.** Đã cân nhắc thêm điểm ngữ cảnh do Qwen3-4B chấm (xác suất của câu khi đứng sau tên app đã
  cài và lời trợ lý), và bỏ. Lý do: LLM không nghe âm thanh nên nghiêng về câu "hợp lý" thay cho câu được nói; phần ASR
  phải chờ LLM chạy trước bước agent, tức ASR và LLM không còn tách rời; vLLM có thể cho log xác suất lệch nhỏ khi số câu
  trong lô khác nhau, đe doạ tính tất định; và xác suất của LLM đổi sau khi fine-tune, nên phải đo lại. Ngữ cảnh của mẫu vẫn
  được dùng qua từ điển ngữ cảnh (C0, F) và đặc trưng của G.
- **Không thêm tham số.** J dùng large-v3 có sẵn; tổng vẫn 5,68B.
- **Không có con số tự đặt.** J chỉ so hai điểm.

### 8.5. Dữ liệu và số lần đo

- Chấm điểm âm thanh chạy ở phiên GPU K3 (Phụ lục B.2), sau K2, trên tập học (cho G), `asr_tune` và val.
- J vào bảng chọn chung trên `asr_tune` (mục 2.5); chỉ đo trên val nếu thắng.

### 8.6. Chỉ số và tiêu chí

Chỉ số như F và G: độ chính xác quyết định đoạn, chép đúng từ tiếng Anh, WER câu thuần Việt, WER câu chêm, WER mọi câu,
`merge_wrong_english`, `merge_missed_english` (mục 2.2). Thêm thời gian chấm mỗi mẫu (encoder thêm và các lượt ép chuỗi)
và số câu bỏ qua vì dài hơn 30 giây.

Tiêu chí so J với các ứng viên khác: bảng chọn chung (mục 2.5).

### 8.7. Kết quả

Kết quả của J trên tập dò và trên val nằm trong bảng chọn chung (mục 2.5).

Phân bố `Δâm_thanh` trên tập dò (p5, p50, p95), tách theo nhãn tự động: chưa đo. Số câu dài hơn 30 giây bị bỏ qua: chưa
đo.

### 8.8. Phân tích

*Điền sau khi đo.*

### 8.9. Kết luận

*Điền sau khi đo.*

### 8.10. Giới hạn và rủi ro

- **Điểm âm thanh của large-v3 vẫn mang thói quen ngôn ngữ của large-v3.** Decoder của large-v3 cũng là một mô hình ngôn
  ngữ; với tên lạ, large-v3 có thể chấm cao một cách viết quen hơn cách viết đúng. J chỉ so hai phương án đã có, nên không
  cứu được trường hợp cả hai đều sai.
- **Kiểu viết.** Đưa cả hai câu về chữ thường không dấu câu là kiểu viết large-v3 ít gặp; điểm tuyệt đối thấp hơn bình
  thường nhưng hai câu chịu như nhau. Cách viết số khác nhau ("3" và "ba") không thuộc phạm vi vì đoạn không có từ tiếng
  Anh mới thì không chấm.
- **Tách token khác nhau.** "Zalo" và "giá lô" có số token khác nhau. Tổng log xác suất của cả câu (kể cả token kết thúc)
  là xác suất của cả chuỗi nên so được, nhưng mô hình có thể hơi nghiêng về câu ít token; hiện tượng này sẽ thấy ở
  `merge_missed_english` và `merge_wrong_english`.
- **Tất định.** CTranslate2 chấm ép chuỗi tất định trên cùng máy và cùng kiểu số. Thử nghiệm I kiểm tra bằng cách chạy
  lại 50 mẫu và so chữ từng ký tự.
- **Câu dài hơn 30 giây** không được chấm (mục 8.3).

## Giới hạn đã biết

- Tập dò chỉ có 300 mẫu (394 từ ngoại), nên chênh vài từ đã là vài điểm phần trăm; E, F, G chọn nối tiếp trên cùng tập
dò, nên số tập dò của bước sau lạc quan hơn (mục 2.3).
- Mọi chỉ số chất lượng dựa vào quy tắc tìm từ ngoại của lần 2 (rà tay: đúng 99,7% khi nhận, tìm được 94,3%; 90,6% trên
`val_unseen`). Từ tiếng Anh trông như âm tiết Việt ("team", "gym") không được tính là từ tiếng Anh.
- Mục tiêu cuối là gọi đúng tool. Lần 3 đo chữ; ảnh hưởng tới lời gọi tool đo ở lần chạy hệ thống với LLM.
- Thời gian đo khi vLLM đã nạp nhưng agent chưa sinh chữ.

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


### A.2. Code sẽ viết hoặc sửa ở lần 3


| File › tên                                                                                                         | Loại       | Đầu vào → đầu ra                                                             | Việc làm                             | Thử nghiệm              |
| ------------------------------------------------------------------------------------------------------------------ | ---------- | ---------------------------------------------------------------------------- | ------------------------------------ | ----------------------- |
| `lifemate/backends/asr_prompt.py` › `build_sample_prompts`                                                         | hàm (mới)  | mẫu, catalog, câu mồi cố định, biến thể → danh sách câu mồi theo thứ tự file | Dựng câu mồi theo mẫu                | E                       |
| `lifemate/backends/asr.py` › `FasterWhisperASR.transcribe_batch`                                                   | sửa        | thêm `prompts` theo từng file; trả thêm điểm tin cậy             | Câu mồi theo câu, điểm tin cậy       | E, G                    |
| `lifemate/core/config.py` › `ASRConfig.prompt_mode`                                                                | khoá mới   | `"fixed"` hoặc `"sample"`                                                    | Bật câu mồi theo mẫu                 | E                       |
| `lifemate/backends/asr_guard.py` › `needs_retry`                                                                   | sửa        | so với phần cố định của câu mồi                                              | Lớp bảo vệ khi câu mồi có lời trợ lý | E                       |
| `lifemate/backends/asr.py` › `MergedASR`                                                                           | lớp (mới)  | mẫu → chữ ghép của mọi câu, thời gian từng bước                              | Engine ghép trong hệ thống thật      | I, và lần chạy hệ thống |
| `lifemate/eval/eval_asr.py` › chỉ số `en_context`, `assistant_echo`, `merge_wrong_english`, `merge_missed_english` | sửa        |                                                                              | Chỉ số mới (mục 2.2)                 | E, F, G                 |
| `lifemate/eval/prompt_word_reliability.py` › `count_prompt_words`, `reliability`                                   | module mới | mẫu, chữ RNNT, chữ large-v3 → từ → (viết ra, đúng) → độ tin cậy              | Bảng độ tin cậy                      | F                       |
| `lifemate/eval/asr_merge.py` › `sample_context_words`                                                             | sửa        | chỉ lấy giá trị của dict, bỏ khoá                                            | Từ điển ngữ cảnh bỏ khoá JSON        | mọi thử nghiệm          |
| `lifemate/eval/asr_merge.py` › `context_reliable`                                                                  | sửa        | thêm bảng độ tin cậy và ngưỡng                                               | Cách lọc mới                         | F                       |
| `lifemate/eval/merge_classifier.py` › `region_features`, `region_labels`, `fit_logistic`, `predict`                | module mới | đoạn khác nhau → đặc trưng, nhãn → trọng số → xác suất                       | Bộ phân loại ghép                    | G                       |
| `lifemate/eval/asr_alternatives.py` › `format_alternatives`, `alternatives_ceiling`, `alternatives_tokens`         | module mới | chữ hai bản, đoạn không chắc → chữ hai phương án, trần, số token             | Thử nghiệm hai phương án             | H                       |
| `lifemate/eval/candidate_scoring.py` › `candidate_pairs`, `decide_j` | module mới | chữ RNNT, chữ large-v3 → câu ứng viên; điểm → quyết định | Chấm điểm hai phương án | J, G |
| `lifemate/backends/asr.py` › `FasterWhisperASR.score_texts` | hàm (mới) | âm thanh, các câu → điểm âm thanh từng câu | Chấm ép chuỗi không câu mồi | J |
| `lifemate/eval/gpu_memory.py` › `GpuMemoryMonitor`                                                                 | lớp mới    | chạy nền `nvidia-smi` → `gpu_memory.csv`, bộ nhớ đỉnh                        | Đo bộ nhớ                            | I                       |
| `lifemate/eval/make_splits.py`                                                                                     | sửa        | ghi thêm `asr_learn.txt`                                                     | Danh sách tập học                    | F, G                    |
| `ops/kaggle/asr_v3/`                                                                                               | kernel mới |                                                                              | Phiên K1, K2                         | E, F, G, I              |
| `ops/scripts/asr_timing_r1.sh`                                                                                     | script mới |                                                                              | Phiên R1 trên máy thuê               | I                       |


## Phụ lục B. Code cần viết và lịch các phiên GPU

### B.1. Việc chuẩn bị (máy dev, không cần GPU)


| Mã  | Việc                                                                                                                                 | Kiểm tra                                                                                                                                                       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | `make_splits` ghi `asr_learn.txt` (793 mã = `train.txt` bỏ `asr_tune.txt`)                                                           | Test: không trùng mã nào với `asr_tune`, `val_seen`, `val_unseen`                                                                                              |
| P2  | Đọc mã nguồn faster-whisper bản đang ghim: cách cắt câu mồi khi quá 223 token; xác nhận `align` chấm được chuỗi cho sẵn không kèm câu mồi | Ghi kết quả vào mục 3.3 và 8.3 |
| P3  | `asr_prompt.py`: `build_sample_prompts`; đếm số token bằng tokenizer của large-v3                           | Test: đúng thứ tự file, không đọc lượt người dùng, câu không có lượt trợ lý liền trước nhận phần cố định; test quét mã nguồn không đọc file môi trường vẫn qua |
| P4  | `FasterWhisperASR`, `ASRTranscriber`, `ASRConfig.prompt_mode`, `needs_retry` phần cố định; ghi `prompt` và `scores` vào `hyps.jsonl` | Test với backend giả: `prompt_mode = "fixed"` cho đúng chữ như cũ                                                                                              |
| P5  | Chỉ số mới trong `eval_asr`                                                                                                          | Test trên câu tự viết; chấm lại chữ lần 2 cho đúng số cũ ở các chỉ số cũ                                                                                       |
| P6  | `prompt_word_reliability.py`, `context_reliable`, `merge_classifier.py`, `asr_alternatives.py`                                       | Test trên dữ liệu tự viết; `merge_classifier` với trọng số tay tái tạo đúng quy tắc `context_other`                                                            |
| P7  | Engine `MergedASR` và khoá config chọn I1–I4; ghi `timing.jsonl`                                                                     | Test với hai backend giả: chữ ghép giống `asr_merge` offline từng byte                                                                                         |
| P8  | `GpuMemoryMonitor`, kernel `ops/kaggle/asr_v3/`, script R1 (khởi động vLLM trước engine ghép cho I5)                                                                           | Chạy kernel preflight trên CPU với 3 mẫu                                                                                                                       |
| P9  | `make check`, `make test`, cập nhật SPEC và README các thư mục                                                                       |                                                                                                                                                                |
| P12 | `merge_decision_accuracy`; bảng chọn chung (mục 2.5) chạy mọi ứng viên trên cùng các đoạn và áp quy tắc chọn | Test: với nhãn và quyết định tự viết, độ chính xác và thứ tự ưu tiên đúng như mục 2.5 |
| P10 | `sample_context_words` bỏ khoá JSON, chỉ lấy giá trị                                                                                 | Test: khoá như `type`, `properties` không có trong từ điển, giá trị vẫn có |
| P11 | `candidate_scoring.py`, `FasterWhisperASR.score_texts`; kernel K3 | Test: câu ứng viên chỉ khác nhau ở đúng một đoạn, cùng kiểu viết; `score_texts` với backend giả trả đúng thứ tự; kernel preflight trên Kaggle với 3 mẫu: điểm của chính câu large-v3 chép cao hơn câu bị đổi một từ (máy dev không chạy mô hình) |


### B.2. Các phiên GPU

Ước tính theo tốc độ của lần 2 trên T4: large-v3 khoảng 1,2 giây mỗi câu, RNNT nhanh hơn khoảng 3 lần.


| Phiên | Máy                                                | Chạy gì                                                                                                              | Số câu                   | Ước tính                                                       |
| ----- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------------------------- |
| K1    | Kaggle T4                                          | large-v3 E0, E1, E2 trên `asr_tune` (qua engine ghép, RNNT chạy kèm) | 3 × 1 024 | Khoảng 1 giờ 10 phút |
| K2    | Kaggle T4                                          | Tập học: RNNT, large-v3 biến thể thắng. Val: large-v3 biến thể thắng | 2 × 2 569 + 1 443 | Khoảng 2 giờ 30 phút |
| K3    | Kaggle T4                                          | Điểm âm thanh (encoder + `align` không câu mồi) cho câu có đoạn cần quyết định | Câu có đoạn cần quyết định của tập học, `asr_tune`, val (ước khoảng 20% số câu) | Khoảng 1 giờ |
| R1    | A30 thuê (RunPod), không có thì GPU 24 GB gần nhất | Cấu hình cuối, I1–I4 trên `public_test` và val; I5 khi vLLM đã nạp Qwen3-4B; chạy lại 50 mẫu; bộ nhớ đỉnh | Khoảng 5 × 1 933 + 1 443 | Khoảng 2 giờ 30 phút (A30 dự kiến nhanh hơn T4; số thật ghi sau phiên) |


Tổng GPU Kaggle khoảng 4 giờ 40 phút, trong hạn mức 30 giờ mỗi tuần. Phiên R1 tốn tiền thuê máy, cần xác nhận trước khi thuê.