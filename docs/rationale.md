# Phân tích nguyên lý gốc về fine-tune (27/10/2026)

Tài liệu giữ lập luận từ nguyên lý gốc cho các quyết định fine-tune. Trạng thái hiện tại của hệ thống xem `SPEC.md`, lịch sử quyết định xem `decisions.md`, số đo xem `experiments.md`.

Chuỗi lập luận dẫn tới kế hoạch fine-tune ở `SPEC.md` phụ lục E.5.2. Đọc mục này khi muốn biết **vì sao** chứ không chỉ **làm gì**.

> Mục này viết ngày 27/10/2026, trước khi chuyển theo các quyết định của team lead. Số liệu trong đây (1098 mẫu train, 8 app bị rút, PhoWhisper-medium, r=16) là của lúc đó. Lập luận vẫn đúng; số liệu hiện tại xem `SPEC.md` phụ lục C.2, E.5 và `decisions.md`.

## 1. Bản chất bài toán

Bỏ hết chi tiết kỹ thuật thì bài toán là: một người nói một câu tiếng Việt, hệ thống phải để lại trạng thái các
app đúng như người đó muốn. Điểm chấm có hai phần: hash trạng thái cuối (đúng hết hoặc 0) và F1 tập tool call
(đúng từng phần). Phần hash là "tất cả hoặc không": sai một ký tự trong tên người nhận, sai một số trong số tiền
là mất trọn phần hash của mẫu đó.

Chân lý nền, tức những điều không thể giảm thêm được nữa:

1. Điểm phụ thuộc vào **giá trị tham số chính xác từng ký tự** và **đúng tool, đúng thứ tự**. Không phụ thuộc
   vào câu trả lời hay cách diễn đạt.
2. Mọi thông tin để làm đúng nằm trong đúng ba nguồn: **âm thanh**, **ngữ cảnh mẫu** (danh bạ, thiết bị, app
   đã cài), **tài liệu tool**. Không có nguồn thứ tư.
3. Dữ liệu có nhãn chỉ có 1098 mẫu, khoảng 13 giờ audio. Model gốc đã học từ hàng nghìn tỉ token và hàng trăm
   nghìn giờ audio. Với lượng này không thể "dạy kiến thức", chỉ **dạy quy ước**.
4. Lúc thi sẽ gặp app lạ (8 app được tự rút ra khỏi train để mô phỏng). Thứ cần học là **cách đọc tài liệu rồi
   hành động**, không phải thuộc lòng app.
5. 30 giây mỗi mẫu trên A30, gồm cả nghe lẫn suy luận.

Ví dụ đời thường: thuê một kế toán giỏi (model gốc) về công ty. Không ai dạy lại kế toán từ đầu; chỉ dạy mẫu
biểu, quy ước đặt tên, quy trình duyệt. Và người đó cần xử lý được cả loại chứng từ mới chỉ bằng cách đọc
hướng dẫn kèm theo.

## 2. Giả định bị thách thức

| Giả định | Thách thức | Kết luận |
|---|---|---|
| "Fine-tune để model hiểu tiếng Việt hơn" | 13 giờ và 1098 câu không dạy được tiếng Việt. Qwen đã hiểu ở mức đủ. | Bỏ. Mục tiêu là quy ước đầu ra. |
| "Lỗi tên riêng là lỗi ASR, phải sửa ở ASR" | Chân lý 2: danh bạ nằm trong ngữ cảnh mẫu, LLM thấy, ASR không thấy. | Sửa: phần lớn lỗi tên riêng nên sửa ở LLM (mục 4). |
| "Model lớn hơn hoặc có thinking sẽ tốt hơn" | Chân lý 5 và 1: thinking tốn thời gian không đoán được; lỗi ở đây là lỗi quy ước, không phải lỗi suy luận sâu. | Giữ quyết định bỏ thinking. |
| "Càng nhiều epoch, loss càng thấp, càng tốt" | Chân lý 4: thuộc catalog train làm hỏng app lạ. | Chốt bằng điểm `val_unseen`, không bằng loss. |
| "WER thấp là ASR tốt" | Chân lý 1: chỉ từ nằm trong tham số tool mới ảnh hưởng điểm. | Đo tỉ lệ nghe sai thực thể. |

## 3. LLM: suy từ chân lý lên cách fine-tune

### 3.1. Bản chất

Một LLM decoder là hàm ước lượng xác suất token kế tiếp p(token | mọi token trước). Fine-tune nghĩa là dịch hàm
này một chút để trên đầu vào giống bài này, token có xác suất cao nhất là token mong muốn. Model không "hiểu thêm",
mà chỉ **lệch ưu tiên**.

### 3.2. Vì sao LoRA, vì sao QLoRA, vì sao r = 16

Từ chân lý 3 và 4: lượng thông tin cần đưa vào rất nhỏ (một bộ quy ước), thứ cần bảo toàn rất lớn (khả năng đọc
hiểu chung). Hai yêu cầu này dẫn thẳng tới **thay đổi có hạng thấp**. LoRA (Low-Rank Adaptation) đóng băng
trọng số W và học thêm ΔW = B·A với A ∈ ℝ^{r×d}, B ∈ ℝ^{d×r}, r nhỏ. Với d = 2560 và r = 16, LoRA chỉ cho phép
model thay đổi theo 16 hướng trong 2560 hướng có thể của mỗi ma trận. Đó là ràng buộc có chủ đích, giống việc cho
nhân viên mới chỉ được sửa mẫu biểu, không được sửa sổ cái.

Số tham số học được: 2·r·d cho mỗi ma trận d×d thay vì d². Trên 7 ma trận (q, k, v, o, gate, up, down) × 36 lớp
của Qwen3-4B, tổng vào khoảng 60–70M, tức 1,5% model. Dữ liệu tính loss chỉ khoảng 100–200K token (1098 mẫu ×
~3 lượt assistant × vài chục token), nên bài này đã ở chế độ "tham số nhiều hơn dữ liệu". Tăng r lên 64 chỉ thêm khả
năng ghi nhớ, không thêm thông tin, và tăng rủi ro overfit trên `val_unseen`. α/r giữ bằng 2 (α = 32) vì LoRA
nhân đầu ra ΔW·x với α/r; tăng r mà giữ α thì bước cập nhật hiệu dụng nhỏ đi.

Gắn LoRA vào cả attention và MLP vì tri thức dạng "từ → hành vi" ("chuyển khoản" → `bank.transfer`) phần lớn nằm
trong MLP, còn attention là "nhìn vào đâu" (đọc tool docs). Cần cả hai.

QLoRA (lượng tử hóa phần đóng băng xuống 4 bit theo phân phối chuẩn, NormalFloat4) là hệ quả của chân lý 5 và
bộ nhớ T4 16 GB: 4,02B tham số bf16 là 8 GB, cộng gradient và hai moment của AdamW cho full fine-tune là 48–64 GB.
Không có cách nào khác trên T4. NF4 chia khối 64 giá trị, chuẩn hóa theo max tuyệt đối, ánh xạ lên 16 mức đặt tại
các phân vị của N(0,1); vì trọng số mạng gần phân phối chuẩn nên 16 mức này tối ưu hơn lượng tử tuyến tính.
Gradient vẫn chảy qua khối đã giải lượng tử về bf16 lúc nhân, nên A và B học ở độ chính xác đầy đủ.

### 3.3. Thứ cần dạy

Liệt kê lỗi có thể xảy ra giữa "nghe được chữ" và "trạng thái đúng":

- Chọn sai tool, hoặc sai thứ tự khi câu có nhiều bước.
- Sai giá trị tham số: số viết bằng chữ ("hai trăm nghìn") phải thành số; tên người phải khớp **đúng chuỗi trong
  danh bạ**; giờ, ngày đúng định dạng của tool.
- Không hỏi lại khi thiếu thông tin, hoặc hỏi lại khi đã đủ.
- Gọi tool không tồn tại, sai schema.

Mỗi loại là một **quy ước ánh xạ** từ (chữ + ngữ cảnh + tài liệu) sang JSON. Đây chính xác là thứ 1098 mẫu dạy
được. Suy ra:

- Dữ liệu SFT phải cho model thấy đúng ba nguồn y như lúc thi (prompt giống từng byte, kết quả tool phát lại qua
  `zaic_tools`). Fine-tune ước lượng p(y | x) trên phân phối x của tập train; lúc chạy, x khác một dấu cách trong
  tool docs là model đang suy diễn trên phân phối khác.
- Loss chỉ tính trên phần model phải tự sinh (JSON của assistant). Nếu tính trên cả tool docs và kết quả tool
  (hơn 90% token của mẫu), mục tiêu biến thành "thuộc catalog", phá chân lý 4. Che bằng label = −100.

### 3.4. Chọn checkpoint

Loss là cross-entropy trung bình trên token; điểm thi là hàm 0/1 của hash cộng F1. Một token sai trong 50 token là
loss gần như không đổi nhưng hash về 0. Vậy loss chỉ để lọc thô. Điểm thật trên `val_unseen` gần nhất với thứ
cần tối ưu, nhưng nhìn vào điểm đó nhiều lần sẽ tự đánh lừa (so sánh nhiều lần làm điểm val lạc quan hơn thực tế), nên
chỉ chạy agent thật trên 2 checkpoint có loss `eval_unseen` thấp nhất.

### 3.5. Merge

W' = W + (α/r)·B·A là một phép cộng ma trận làm một lần; lúc chạy không còn phép nhân thêm, không tốn thời gian
trong 30 giây. W lúc train là bản nf4, W lúc merge là bản bf16 gốc, nên W' khác một chút so với thứ model nhìn
thấy khi train. Sai số nhỏ nhưng khác 0: luôn đánh giá bản đã merge, tức đúng thứ sẽ nộp.

## 4. ASR: suy từ chân lý lên cách fine-tune

### 4.1. Bản chất

Whisper là seq2seq hai nửa: encoder biến sóng âm thành biểu diễn "đã nghe được gì"; decoder là một mô hình ngôn
ngữ nhỏ biến biểu diễn đó thành chữ. Lỗi ASR vì thế có hai loại bản chất khác nhau:

- **Lỗi nghe** (encoder): nhiễu, giọng vùng miền, từ hiếm. Sửa cần nhiều giờ audio đa dạng. Dữ liệu chỉ có 13 giờ so với
  844 giờ encoder đã học; cố sửa là ước lượng hàng trăm triệu tham số từ mẫu nhỏ hơn 60 lần, phương sai tăng,
  model quên giọng và tiếng ồn không có trong 13 giờ đó (bias–variance).
- **Lỗi viết** (decoder): nghe đúng nhưng viết "15" thay vì "mười lăm", thêm dấu câu, viết hoa, từ tiếng Anh sai
  chính tả. Đây là quy ước, 3547 cặp dạy được.

Từ chân lý 1: chữ ASR phải ở **đúng dạng LLM đã được train**. Cập nhật 27/10/2026: PhoWhisper viết số bằng chữ còn
`audio_text` viết bằng chữ số, và BTC không chấm chữ ASR, nên LLM nộp bài học trên chính chữ ASR (v2). Fine-tune
ASR về `audio_text` thì đổi luôn cách viết, nên phải train lại LLM v2 bằng ASR mới. Vậy "giai đoạn 1 chỉ train decoder" là hệ
quả trực tiếp: lỗi cần sửa ở decoder, dữ liệu không đủ chạm encoder. Giai đoạn 2 mở encoder với lr nhỏ hơn 10
lần (discriminative learning rate) chỉ khi WER vẫn cao sau giai đoạn 1, để ΔW của encoder ở trong lân cận nghiệm
đã tốt.

Full fine-tune được vì 0,77B × 16 byte ≈ 12 GB, cộng activation của 30 giây mel với gradient checkpointing vừa
T4. Khi vừa bộ nhớ thì full cho ít ràng buộc hơn LoRA, và với ASR không chắc lỗi cần sửa nằm trong không gian hạng
thấp của decoder.

### 4.2. Thước đo

Chỉ từ sẽ thành tham số tool mới ảnh hưởng điểm. Sai "ừ", "nhé" không làm sai tool call; sai "Lan" thành "Lâm"
hay "năm trăm" thành "năm chăm" làm sai tham số và mất toàn bộ hash. Tỉ lệ nghe sai thực thể là ước lượng trực
tiếp xác suất mẫu bị hỏng vì ASR, gần hàm mất mát thật hơn WER. WER chỉ để theo dõi model có hỏng chung không.

## 5. Ba điều chỉnh rút ra từ phân tích (và vì sao)

### Điều chỉnh 1: fine-tune LLM phải có vòng 2 trên chữ ASR thật

**Từ đâu ra.** Chân lý 2 nói mọi thông tin nằm trong ba nguồn, và danh bạ, tên thiết bị, app đã cài thuộc nguồn
"ngữ cảnh mẫu". Thành phần nào nhìn thấy ngữ cảnh mẫu? Chỉ LLM. ASR chỉ nhận sóng âm. Vậy khi ASR nghe "Lan"
thành "Lang", ASR không có gì để đối chiếu, nhưng LLM có danh bạ gồm "Lan" và "Long" ngay trong prompt.

**Nghĩa là gì.** Nếu train LLM trên đúng chữ ASR thật (chạy ASR đã chọn trên audio của train, lấy chữ ASR nghe ra
làm transcript) với nhãn vẫn là tool call gold, model sẽ học quy ước "chọn tên trong danh bạ gần nhất với chữ
nghe được", "số tiền phải là số tròn hợp lý", v.v. Đây là sửa lỗi nghe bằng thông tin mà chỉ LLM có, và cách này không
tốn thêm thời gian lúc chạy. Vòng 1 (trên `audio_text` gold) vẫn cần trước để học quy ước sạch; vòng 2 là tinh
chỉnh để chịu được nhiễu.

**Ví dụ đời thường.** Thư ký nghe sếp nói qua điện thoại kém "chuyển cho Lang", nhìn danh bạ chỉ có "Lan" và
"Long", chọn "Lan". Người sửa lỗi là thư ký (có danh bạ), không phải cái điện thoại.

**Trong code.** `sft_data.py` đã có `transcript_source = "asr"`; vòng 2 là chạy lại pipeline với tùy chọn đó và
dữ liệu `data/sft/v2.jsonl`. Cần chọn ASR trước khi làm vòng 2.

### Điều chỉnh 2: quyết định fine-tune ASR chỉ đưa ra sau LLM vòng 2

**Từ đâu ra.** Điều chỉnh 1 nói phần lớn lỗi thực thể có thể sửa ở LLM rẻ hơn. Nếu đo "điểm mất do ASR" ngay sau
khi cắm ASR vào LLM vòng 1, con số đó gồm cả phần LLM vòng 2 sẽ tự sửa được. Dùng con số đó để quyết định fine-tune
ASR là bỏ 1–2 ngày GPU vào thứ có thể biến mất sau một vòng train LLM rẻ hơn.

**Thứ tự đúng.** ASR gốc → LLM v1 (gold) → LLM v2 (chữ ASR) → đo điểm mất do ASR và tỉ lệ nghe sai thực thể →
chỉ khi vẫn ≥ 3 điểm hoặc ≥ 10% mới fine-tune ASR. Hai ngưỡng này là ước lượng ban đầu của nhiễu giữa hai lần
chạy trên ~200 mẫu val; SPEC yêu cầu đo lại độ dao động rồi chốt.

**Ví dụ đời thường.** Trước khi thay điện thoại mới cho sếp, thử dạy thư ký cách đối chiếu danh bạ đã. Nếu sau
đó vẫn nghe sai nhiều thì mới thay điện thoại.

### Điều chỉnh 3: nếu fine-tune ASR, làm dữ liệu TTS nhắm vào ba nhóm thực thể trước khi nghĩ tới mở encoder

**Từ đâu ra.** Mục 9.4.2: lỗi gây mất điểm tập trung ở ba nhóm có thể liệt kê được: tên người trong danh bạ, tên
thiết bị và địa điểm, số tiền và giờ. Ba nhóm này sinh được bằng TTS (tổng hợp giọng từ chữ): lấy câu mẫu từ
`audio_text` của train, thay thực thể bằng giá trị khác, đọc bằng TTS, nhãn chính là câu đã thay. Dữ liệu như vậy
đánh thẳng vào loại lỗi gây mất điểm, hiệu quả hơn thêm giờ audio chung chung, và vẫn chỉ cần train decoder.

**Vì sao trước mở encoder.** Mở encoder là bước có rủi ro cao nhất (mục 4.1, lỗi nghe cần nhiều giờ audio đa
dạng). Dữ liệu TTS có mục tiêu là bước rủi ro thấp, rẻ, và kiểm chứng được ngay bằng tỉ lệ nghe sai thực thể trên
`val_unseen`. Theo nguyên tắc "thêm từng lớp và mỗi lớp phải tự chứng minh", lớp rẻ và đo được đi trước.

**Ràng buộc.** TTS chỉ được dùng lúc phát triển, cấm lúc thi. Không dùng câu của val hay public để sinh. Không
bao giờ dùng dữ liệu tổng hợp để đánh giá.

## 6. Điểm yếu còn lại của lập luận

Chân lý 2 giả định ngữ cảnh mẫu luôn chứa đủ ứng viên để LLM đối chiếu. Với địa chỉ tự do, tên không có trong
danh bạ, hay nội dung tin nhắn tự do thì không có gì để đối chiếu, và lúc đó chỉ ASR mới cứu được. Bảng lỗi sau
LLM v2 sẽ cho biết nhóm này chiếm bao nhiêu; nếu lớn thì điều chỉnh 2 phải đẩy fine-tune ASR lên sớm hơn.

## 7. Thứ tự làm, tóm tắt

1. Cắm ASR gốc (PhoWhisper-medium), đo WER, tỉ lệ nghe sai thực thể, thời gian.
2. LLM v1: QLoRA trên `audio_text` gold. Chọn checkpoint bằng điểm `val_unseen`.
3. LLM v2: dựng lại dữ liệu với chữ ASR thật, train tiếp từ v1 hoặc từ gốc (thử cả hai), chọn như trên.
4. Đo điểm mất do ASR sau v2. Nếu còn ≥ 3 điểm hoặc ≥ 10% thực thể: fine-tune ASR giai đoạn 1 (decoder) với
   dữ liệu thật + TTS có mục tiêu. Mở encoder chỉ khi WER vẫn cao.
5. Mỗi bước đều đo trên `val_seen` và `val_unseen`, ghi vào `experiments.md`.

## 8. Ba điều chỉnh, giải thích lại bằng một mẫu thật

Mẫu `train_0001`. Người dùng nói:

> "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng, nhắn **Sếp Dũng** giờ mới; không mưa thì thôi."

Nhãn gold gồm năm bước, trong đó có `contacts.search(query="Sếp Dũng")` rồi `sms.send(contact_id="c_08", ...)`.

### 8.1. Vì sao LLM cần vòng 2 train trên chữ ASR thật

**Chuyện gì xảy ra lúc thi.** ASR (nhận dạng giọng nói) nghe file audio và viết ra chữ. Giả sử ASR nghe nhầm
thành "nhắn **Sếp Dung** giờ mới". LLM chỉ nhận được dòng chữ này, không nghe audio. Nếu LLM được train chỉ trên
chữ gold "Sếp Dũng" (vòng 1), LLM chưa bao giờ thấy chữ sai, nên gọi `contacts.search(query="Sếp Dung")`. Tùy tool
có khớp mờ hay không, có thể không tìm ra `c_08`, tin nhắn không gửi, hash trạng thái sai, mất trọn phần điểm hash
của mẫu.

**Ai có thông tin để sửa?** ASR không biết danh bạ của người này có ai. LLM thì có thể biết: với nhiều app (thiết
bị nhà thông minh, app đã cài, lịch) danh sách thực thể nằm ngay trong ngữ cảnh mẫu mà LLM đọc trong prompt; với
danh bạ thì LLM nhìn thấy kết quả `contacts.search` trả về ở bước trước. LLM là thành phần duy nhất vừa thấy chữ
nghe được, vừa thấy danh sách ứng viên.

**Vòng 2 làm gì.** Chạy ASR đã chọn trên chính audio của tập train, lấy chữ ASR nghe ra (có lỗi thật), ghép với
nhãn gold không đổi. Train LLM trên cặp (chữ có lỗi → tool call đúng). Qua vài trăm mẫu, LLM học quy ước: chữ nghe
được chỉ là gợi ý, giá trị tham số phải là chuỗi có trong danh sách ứng viên gần nhất với chữ nghe được. Giống thư ký nghe sếp
nói qua điện thoại rè "nhắn cho Dung", mở danh bạ chỉ thấy "Dũng" và "Dung Nguyễn (kế toán)", nhìn ngữ cảnh "sếp"
thì chọn Dũng. Người sửa lỗi là thư ký vì thư ký cầm danh bạ, không phải cái điện thoại.

**Vì sao không tốn thêm thời gian lúc chạy.** Quy ước nằm trong trọng số đã merge; lúc chạy LLM sinh đúng số token
như trước, không thêm bước nào.

### 8.2. Vì sao chỉ quyết định fine-tune ASR sau LLM vòng 2

**"Điểm mất do ASR" là gì.** Chạy cùng LLM hai lần trên bộ val: một lần đưa chữ gold vào, một lần đưa chữ ASR
vào. Hiệu số điểm là phần mất vì ASR nghe sai.

**Bẫy khi đo sớm.** Giả sử đo với LLM v1 và thấy mất 8 điểm. Trong 8 điểm đó, một phần là lỗi kiểu "Sếp Dung"
mà LLM v2 sẽ tự sửa bằng danh bạ, một phần là lỗi kiểu nghe "chuyển năm trăm nghìn" thành "chuyển năm chăm
nghìn" hay nghe thiếu cả một mệnh đề, thứ không có danh sách nào để đối chiếu. Nếu nhìn con số 8 mà quyết định
fine-tune ASR ngay thì sẽ bỏ 1–2 ngày GPU để sửa cả phần LLM v2 lẽ ra sửa miễn phí. Đo lại sau v2, có thể chỉ còn
mất 2 điểm, dưới ngưỡng 3, và không cần fine-tune ASR nữa.

Ví dụ đời thường: nhân viên mới ghi sai tên khách nhiều. Trước khi thay cả hệ thống điện thoại của công ty, thử
phát cho nhân viên quyển danh bạ và dạy cách đối chiếu. Sau một tuần vẫn sai nhiều thì mới thay điện thoại.

**Thứ tự.** ASR gốc → LLM v1 (chữ gold) → LLM v2 (chữ ASR) → đo điểm mất và tỉ lệ nghe sai thực thể → nếu còn
≥ 3 điểm hoặc ≥ 10% mới fine-tune ASR.

### 8.3. Nếu fine-tune ASR: TTS nhắm vào ba nhóm thực thể trước, mở encoder sau

**Lỗi nào mới làm mất điểm.** Với `train_0001`, ASR nghe "dời chạy bộ" thành "rời chạy bộ" thì LLM vẫn hiểu,
không mất điểm. Nhưng nghe "muộn 1 tiếng" thành "muộn 2 tiếng" thì `start` của sự kiện sai, mất hash. Nên lỗi
đáng sửa gom về ba nhóm: tên người, tên thiết bị và địa điểm, số tiền và giờ. Đó là ba nhóm trở thành tham số tool.

**TTS có mục tiêu là gì.** TTS (text-to-speech, máy đọc chữ thành giọng). Lấy câu mẫu trong train, giữ nguyên
khung, chỉ thay thực thể: "nhắn Sếp Dũng" → "nhắn Hoàng", "nhắn Thu Hà"; "muộn 1 tiếng" → "muộn 3 tiếng";
"chuyển hai trăm nghìn" → "chuyển một triệu rưỡi". Mỗi câu đọc bằng TTS với vài giọng khác nhau, nhãn chính là
câu chữ đã thay. Được vài nghìn cặp audio-chữ chỉ xoay quanh đúng ba nhóm lỗi gây mất điểm, thay vì thêm giờ audio
chung chung mà phần lớn là từ không ảnh hưởng điểm. Luật cho phép TTS lúc phát triển, cấm lúc thi.

**Vì sao chỉ train decoder với dữ liệu này.** Whisper có hai nửa: encoder "nghe" (sóng âm → biểu diễn), decoder
"viết" (biểu diễn → chữ). Lỗi "năm chăm" thay vì "năm trăm", hay "Hoàng" viết thành "Hoàn", phần lớn là decoder
chọn chữ sai dù encoder đã nghe gần đúng. Dạy decoder bằng dữ liệu TTS là bước rẻ: train nhanh, ít tham số, đo
được ngay bằng tỉ lệ nghe sai thực thể trên `val_unseen`.

**Vì sao mở encoder là bước rủi ro nhất.** Encoder của PhoWhisper học từ 844 giờ audio thật đủ loại giọng, tiếng
ồn. Dữ liệu chỉ có 13 giờ thật cộng một ít TTS, mà TTS là giọng máy, sạch, đều. Mở encoder cho model học trên dữ liệu này
là bắt một người đã nghe tốt mọi giọng miền đi luyện nghe lại chỉ bằng giọng của một máy đọc; dễ "quen tai" với
giọng máy và nghe kém đi với giọng thật trong bài thi. Vì vậy mở encoder để sau cùng, chỉ khi dữ liệu TTS cho
decoder vẫn chưa đủ, và khi đó dùng tốc độ học nhỏ hơn 10 lần để model không đi xa khỏi cái đã học.

Tóm lại ba điều chỉnh đều xuất phát từ một ý: sửa lỗi ở chỗ có thông tin để sửa, làm bước rẻ và đo được trước,
bước đắt và rủi ro sau.
