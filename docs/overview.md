# LifeMate - Agentic AI That Turns Your Voice Into Action

**Zalo AI Challenge 2026**

## 1. Overview (Tổng quan)

Giọng nói đang trở thành phương thức chính để con người giao tiếp với máy móc. Thế hệ thiết bị tiếp theo, bao gồm smart speaker (loa thông minh), in-car assistant (trợ lý trên ô tô), smart TV (TV thông minh), earbuds (tai nghe), watch (đồng hồ) và home robot (robot gia đình), sẽ có màn hình nhỏ hoặc không có màn hình, nhưng vẫn kết nối với chính những ứng dụng và dịch vụ mà mọi người đang sử dụng.

Một phụ huynh bảo loa phòng khách làm mát ngôi nhà trước khi rời văn phòng; một tài xế yêu cầu ô tô thanh toán hóa đơn tiền điện; một nhân viên văn phòng bảo tai nghe đặt xe về nhà sau một cuộc họp muộn; một chủ cửa hàng nhờ loa trong bếp đặt bữa trưa cho nhân viên; một người ông/bà chưa từng học cách dùng ứng dụng nhờ TV đặt lời nhắc uống thuốc.

Trên điện thoại, mỗi tác vụ trên cần nhiều lần chạm qua nhiều ứng dụng. Trên screenless device (thiết bị không có màn hình), tác vụ chỉ thực hiện được nếu assistant (trợ lý) có khả năng tự hành động. Tuy nhiên, các voice assistant (trợ lý giọng nói) hiện nay chủ yếu trả lời câu hỏi thay vì hành động.

Điều người dùng mong muốn là chỉ cần nói bằng tiếng Việt thông thường, ví dụ: _“Tôi sắp về nhà, bật điều hòa phòng khách và đặt xe đón tôi”_, _“Chuyển cho mẹ 500 nghìn”_, _“Tắt hết đèn tầng một”_, _“Nhắc tôi họp lúc ba giờ chiều mai”_, và yêu cầu được thực hiện.

**LifeMate** yêu cầu Bạn xây dựng một AI agent (tác nhân AI) chuyển các spoken request (yêu cầu dạng giọng nói) tiếng Việt thành action (hành động). Với đầu vào là một audio request (yêu cầu dạng âm thanh), user context (ngữ cảnh người dùng) và các app (ứng dụng), service (dịch vụ) được kết nối với thiết bị, agent phải:

- Plan (lên kế hoạch) các bước, gọi đúng tool (công cụ) với đúng arguments (tham số) và đưa mọi app về đúng state (trạng thái) người dùng yêu cầu;
- Hỏi lại missing information (thông tin còn thiếu) khi yêu cầu chưa đầy đủ; và
- Không thực hiện action nào khi không có installed app (ứng dụng đã cài đặt) nào xử lý được yêu cầu.

Agent được đánh giá dựa trên những gì agent làm, không phải những gì agent nói.

Dữ liệu Cuộc thi gắn với đời sống số hằng ngày của người Việt, bao gồm smart home (nhà thông minh), banking (ngân hàng), ride-hailing (gọi xe), food delivery (giao đồ ăn) và email (thư điện tử), với các yêu cầu được nói bằng synthetic voice (giọng tổng hợp) và real voice (giọng thật) từ ba miền Bắc, Trung, Nam. Tuy nhiên, năng lực Bạn xây dựng mang tính tổng quát: voice-driven, multi-step tool use (sử dụng tool nhiều bước, điều khiển bằng giọng nói) bởi một AI agent có thể chạy trên mọi thiết bị, từ điện thoại, smart speaker đến ô tô.

## 2. Input (Đầu vào)

- **Spoken request (yêu cầu dạng giọng nói):** Audio file (tệp âm thanh) chứa yêu cầu mới nhất của người dùng bằng tiếng Việt, có thể xen lẫn các từ tiếng Anh thông dụng.
- **User context (ngữ cảnh người dùng):** Current time (thời gian hiện tại), timezone (múi giờ), location (vị trí) và user profile (hồ sơ người dùng), ví dụ: địa chỉ nhà và địa chỉ nơi làm việc.
- **Available apps (ứng dụng khả dụng):** Các app được kết nối với thiết bị (smart home, banking, ride-hailing, food delivery, email, …). Mỗi app cung cấp một tập tool được mô tả bằng JSON schema.
- **Conversation history (lịch sử hội thoại):** Các turn (lượt) trước đó của cùng cuộc hội thoại (nếu có).
- **Simulated environment (môi trường mô phỏng):** Môi trường lưu trữ app data (dữ liệu ứng dụng) của người dùng (devices, accounts, reminders, contacts, …). Agent chỉ được đọc và thay đổi dữ liệu này thông qua tool call (lệnh gọi công cụ).

## 3. Output (Đầu ra)

Agent tạo ra một ordered list (danh sách có thứ tự) các tool call, gồm tool name (tên công cụ) và arguments, để xử lý yêu cầu của người dùng. Tùy theo yêu cầu và thông tin hiện có, agent có thể:

- **Execute actions (thực hiện hành động):** Gọi lần lượt các tool cần thiết để hoàn thành task (tác vụ) và đạt được expected final app state (trạng thái ứng dụng cuối cùng như mong đợi).
- **Request missing information (yêu cầu thông tin còn thiếu):** Gọi `sys.ask_missing_info` để chỉ rõ intended tool (công cụ dự định gọi) và missing parameters (các tham số còn thiếu) khi cần thêm thông tin. Trường hợp này có thể xảy ra sau khi một số tool call khác đã được thực hiện.
- **Handle out-of-scope requests (xử lý yêu cầu ngoài phạm vi):** Trả về empty list (danh sách rỗng) các tool call (`[]`) nếu không có available app nào xử lý được yêu cầu.

**Final state hash (mã băm trạng thái cuối):** Cùng với các tool call, agent nộp `state_hash` do execution environment (môi trường thực thi) sinh ra thông qua `env.get_state_hash()`. Giá trị này được dùng để xác minh final app state (trạng thái ứng dụng cuối cùng).

## 4. Example (Ví dụ)

**Input (Đầu vào):**

- Audio (âm thanh): _“Tôi sắp về nhà, bật điều hòa phòng khách 26 độ và đặt xe đón tôi.”_
- Context (ngữ cảnh): người dùng đang ở nơi làm việc (227 Nguyễn Văn Cừ, Quận 5, TP.HCM), `user_profile.home` là 123 Trần Hưng Đạo, Quận 5, TP.HCM.
- Installed apps (ứng dụng đã cài đặt): `smart_home`, `ride`, `banking`, `email`.

**Output (Đầu ra):** Agent trước tiên đọc device list (danh sách thiết bị) để tìm điều hòa phòng khách, sau đó cài đặt điều hòa và đặt xe. Submission line (dòng nộp bài) cho mẫu này (mỗi dòng trong `submission.jsonl` là một JSON object) như sau:

```json
{"id": "s_000123", "tool_calls": [{"name": "smart_home.get_status", "arguments": {}},
  {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool", "temperature": 26}},
  {"name": "ride.book", "arguments": {"from": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM", "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM"}}],
 "state_hash": "9f2c4e7a1b8d3f60"}
```

## 5. Evaluation (Đánh giá)

Mỗi sample (mẫu) được chấm điểm dựa trên hai yếu tố: agent có đưa app data về đúng final state (trạng thái cuối) hay không, và các tool call của agent khớp với reference (gold) call (lệnh gọi tham chiếu) đến mức nào.

### 5.1. Metrics (Tiêu chí đánh giá)

- **State match (khớp trạng thái):** `state_hash` được nộp phải trùng với gold `state_hash`, nếu không sample nhận 0 điểm. Các free-text (fuzzy) field (trường văn bản tự do) không ảnh hưởng đến hash. `state_hash` phải được tính bởi môi trường `zaic_tools` (`env.get_state_hash()`) và có thể được Ban Tổ chức tái tạo bằng cách replay (chạy lại) các `tool_calls` đã nộp theo đúng thứ tự với phiên bản `zaic_tools` tương ứng. Nếu kết quả khác nhau, kết quả của Ban Tổ chức là kết quả cuối cùng.
- **Tool-call match (khớp lệnh gọi công cụ):** Một submitted call (lệnh gọi đã nộp) khớp với một gold call khi tool name giống nhau, mọi non-fuzzy argument (tham số không mờ) giống hệt nhau, và mọi fuzzy argument (tham số mờ) có token-F1 > 0.6 (sau khi lowercasing (chuyển về chữ thường) và chuẩn hóa Unicode NFC). Mỗi call chỉ được khớp với tối đa một (01) call; thứ tự không được xét đến.
- **Out-of-scope samples (mẫu ngoài phạm vi):** Nếu gold list (danh sách tham chiếu) rỗng, sample chỉ đạt 1 điểm khi submitted list (danh sách đã nộp) cũng rỗng.
- **Missing information (thông tin còn thiếu):** `sys.ask_missing_info` được chấm điểm như mọi tool khác.

### 5.2. Formula (Công thức)

- Nếu gold list rỗng: sample đạt 1 điểm khi submitted list cũng rỗng; ngược lại, sample nhận 0 điểm.
- Ngược lại (gold list không rỗng):

$$
\text{Score} =
\begin{cases}
0, & \text{if } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ or } M = 0 \\
\dfrac{2 \cdot P \cdot R}{P + R}, & \text{otherwise}
\end{cases}
$$

$$
P = \frac{M}{M + U}, \qquad R = \frac{M}{|G|}
$$

Final score (điểm chung cuộc):

$$
\text{Final score} = \frac{100}{N} \sum_{i=1}^{N} \text{Score}_i
$$

Trong đó:

- _P_ và _R_ là precision (độ chính xác) và recall (độ bao phủ);
- _M_ là số matched call (lệnh gọi khớp);
- _|G|_ là số gold call;
- _U_ là số unmatched submitted call (lệnh gọi đã nộp không khớp), bao gồm các call bị từ chối do schema error (lỗi schema);
- _N_ là số sample trong test set (tập kiểm thử).

## 6. Data (Dữ liệu)

Dữ liệu gồm ba (03) split (phần dữ liệu) có cùng cấu trúc:

- **train:** khoảng 1.600 sample có label (nhãn);
- **public_test:** khoảng 600 sample, dùng cho public leaderboard (bảng xếp hạng công khai);
- **private_test:** khoảng 600 sample, được công bố sau khi final Docker solution (giải pháp Docker cuối cùng) được freeze (đóng băng).

### 6.1. Folder structure (Cấu trúc thư mục)

```
<split>/                                  # train | public_test | private_test
├── audio/<id>/<id>.wav                   # spoken request of each sample
├── <split>.jsonl                         # one sample per line
├── <split>_tools_catalog.json            # schema of every app tool
├── <split>_environment/<id>/state.json   # initial app data of each sample
├── sample_submission.jsonl               # public_test only
└── zaic_tools/                           # execution environment library for this split
```

### 6.2. Files (Các tệp dữ liệu)

**(1) Audio (âm thanh):** Yêu cầu mới nhất của người dùng và mọi user turn (lượt của người dùng) trước đó trong history (lịch sử), bằng tiếng Việt xen lẫn một số từ tiếng Anh thông dụng (ví dụ: "book", "AC"). Định dạng WAV, 16 kHz, mono, 16-bit; khoảng 100% được ghi âm bởi real speaker (người nói thật) thuộc nhiều độ tuổi, giới tính và vùng miền khác nhau (Bắc, Trung, Nam).

**(2) `<split>.jsonl`:** Input của mỗi sample, bao gồm:

- `id`, đường dẫn `audio`;
- User context: `current_time`, `timezone`, current location (vị trí hiện tại), `user_profile`, `installed_apps`;
- Các conversation turn trước đó (`history`).

Các sample train có thêm:

- `audio_text`: transcript (bản chép lời) của mọi user turn;
- `label`: gồm gold `tool_calls`, các `tool_results` tương ứng và `final_state` (toàn bộ dữ liệu của mọi app mà các gold call đã **thay đổi**).

**(3) `<split>_tools_catalog.json`:** Danh sách app của split và schema của mọi tool, bao gồm:

- `name` (`app.tool`);
- `description` (mô tả);
- `kind` (loại): `read` (đọc), `write` (ghi), hoặc `meta` đối với `sys.ask_missing_info`;
- `parameters` dưới dạng JSON Schema;
- Tóm tắt ngắn về các returned field (trường trả về).

Parameters có thể mang các thuộc tính sau:

- `x-ref`: giá trị phải là một ID hợp lệ trong app data;
- `x-match: "fuzzy"`: free text (văn bản tự do), được chấm điểm bằng token-F1;
- `x-one-of`: bắt buộc có ít nhất một (01) trong các nhóm được liệt kê, ví dụ: `contact_id` hoặc `phone_number`.

Catalog bao phủ toàn bộ split; các tool mà một sample được phép gọi là các tool thuộc `context.installed_apps` của sample đó.

**(4) `state.json`:** Initial app data (dữ liệu ứng dụng ban đầu) của một sample (devices, accounts, contacts, reminders, …), với các action trong history đã được áp dụng, kèm một block `_meta` (`question_id`, `current_time`, `tools`). Tệp này được environment (môi trường) nạp; agent **không được** đọc trực tiếp.

### 6.3. Examples (Ví dụ)

**Sample record (bản ghi mẫu) trong `<split>.jsonl`:**

```
{"id": "train_xxxx", "audio": "audio/train_xxxx/train_xxxx.wav",
"context": {"current_time": "2026-11-16T17:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
"user_profile": {"home_address": "123 Trần Hưng Đạo, Quận 5, TP.HCM"},
"installed_apps": {"smart_home": {"tools": [...]}, "ride": {"tools": [...]}}},
"history": [],
"label": {"tool_calls": [...],"tool_results": [...], "final_state": {...}}}   # label: train only
```

**Sample tool (tool mẫu) trong `<split>_tools_catalog.json` (nằm trong `apps[].tools`):**

```
{"schema_version": "2.7.0",
"note": "kind của mỗi tool: read = ...; write = ...; meta = ...",
"apps": [
  {"id": "smart_home", "name_vi": "Nhà thông minh", "group": "home", "description": "...",
  "tools": [
    {
      "name": "smart_home.light_set",
      "kind": "write",
      "description": "Bật/tắt/chỉnh sáng đèn",
      "parameters": {
        "type": "object",
        "properties": {
          "device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"},
          "power": {"type": "string", "enum": ["on", "off"]},
          "brightness": {"type": "integer"},
          "color": {"type": "string"}
        },
        "required": ["device_id"]
      },
      "returns": "",
      "app": "smart_home"
    },
    ...
  ]},
  ...
]}
```

**Sample `state.json`:**

```
{
  "_meta": {"question_id": "train_0248", "current_time": "2026-11-19T18:30:00+07:00", "tools": [...]},
  "smart_home": {
    "devices": [
      {"id": "light_living", "name": "đèn phòng khách", "type": "light", "room": "phòng khách", "floor": 1,
      "state": {"power": "off", "brightness": 40}},
      ...
    ]
  },
  "contacts": {"contacts": [{"id": "c_01", "name": "Hương", "relation": "bạn", "phone": "09701957839"}, ...]},
  ...
}
```

### 6.4. Execution environment (Môi trường thực thi)

Ban Tổ chức cung cấp thư viện Python **`zaic_tools`**, mô phỏng các app trên máy cục bộ.

- Mỗi sample chạy trong một `ToolEnv` riêng, và `ToolEnv` chỉ được tạo một (01) lần.
- Environment kiểm tra mỗi call theo tool schema, thực thi call, cập nhật app data và trả về result (kết quả) để agent sử dụng ở bước tiếp theo.
- Một call thất bại sẽ trả về `"ok": false` kèm error message (thông báo lỗi), thường xảy ra khi tool không tồn tại hoặc một argument sai kiểu dữ liệu.
- Chỉ được gọi các tool thuộc các app được liệt kê trong `context.installed_apps` (cùng với `sys.ask_missing_info`).
- Mọi call, dù thành công hay thất bại, đều được ghi log (nhật ký). Call log này cùng với final state hash tạo thành submission (bài nộp).
- Mỗi split đi kèm phiên bản `zaic_tools` riêng.

```python
import sys
sys.path.insert(0, "public_test")              # zaic_tools of the split
from zaic_tools import ToolEnv

# Create one environment per sample; root is the folder that holds the split folders
env = ToolEnv("public_test", "public_0001", root=".")

# Check arguments against the tool schema (nothing is executed)
env.validate("smart_home.light_set", {"power": "off"})
# -> {"ok": false, "error": "thiếu tham số bắt buộc: device_id"}

# Execute a call
# -> {"ok": true, ...} or {"ok": false, "error": "..."}
env.call("smart_home.get_status", {})

# Call log and final state hash for submission
env.get_tool_calls(), env.get_state_hash()
```

**Quy định:** Agent/giải pháp **không được** đọc, sao chép hoặc parse (phân tích) bất kỳ tệp nào trong `<split>_environment/` (bao gồm `state.json`). Mọi app data chỉ được truy cập thông qua `zaic_tools` của Ban Tổ chức, với đúng một (01) `ToolEnv` cho mỗi sample, và mọi call được thực hiện trên `ToolEnv` đó phải xuất hiện trong submission.

### 6.5. Inference and submission template (Mẫu mã suy luận và nộp bài)

Template (mã mẫu) dưới đây chạy agent trên mọi sample của một split và ghi ra `submission.jsonl`: mỗi sample một dòng, gồm `id`, `tool_calls` và `state_hash`. Chạy template từ thư mục chứa `public_test/`. Tạo đúng một (01) `ToolEnv` cho mỗi sample bên trong `infer()`.

```python
import json
import sys
from pathlib import Path

ROOT = Path(".")                                  # folder that holds public_test/
SPLIT = "public_test"
sys.path.insert(0, str(ROOT / SPLIT))             # zaic_tools of the split
from zaic_tools import ToolEnv

def infer(item):
    env = ToolEnv(SPLIT, item["id"], root=ROOT)   # one environment per sample
    # Your pipeline: ASR (ROOT / SPLIT / item["audio"]) -> LLM -> env.call(tool, arguments) -> LLM -> ...
    # e.g. env.call("smart_home.get_status", {})
    #      env.call("smart_home.light_set", {"device_id": "light_kitchen", "power": "off"})
    return env.get_tool_calls(), env.get_state_hash()

with open(ROOT / SPLIT / f"{SPLIT}.jsonl", encoding="utf-8") as fin, \
     open("submission.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        tool_calls, state_hash = infer(item)
        fout.write(json.dumps({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
```

### 6.6. Submission file format (Định dạng tệp nộp bài)

Tệp nộp bài phải ở định dạng UTF-8 JSON Lines, với đúng một (01) dòng cho mỗi sample trong split; xem `sample_submission.jsonl` để biết định dạng.

Ví dụ tệp nộp bài (ba (03) dòng của `public_test/sample_submission.jsonl`):

```
{"id": "public_0039", "tool_calls": [{"name": "alarm.list", "arguments": {}}, {"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], "state_hash": "461fcd40…"}
{"id": "public_0001", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "music.add_to_playlist", "params": ["playlist_id", "track_id"]}}], "state_hash": "c83200c2…"}
{"id": "public_0002", "tool_calls": [], "state_hash": "0c9a2ec0…"}
```

Các lưu ý khác được mô tả trong `README.md` bên trong data package (gói dữ liệu).

### 6.7. Download (Tải dữ liệu)

- **Data Train + Public test:** [Download](https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/train+public_test.zip)

## 7. Competition rounds (Các vòng thi)

Cuộc thi diễn ra trên leaderboard (bảng xếp hạng) với hai (02) vòng:

- **Round 1 — Public Test (Vòng 1 — Kiểm thử công khai):** Các Đội thi phát triển giải pháp, chạy giải pháp trên `public_test` và nộp kết quả. Điểm số được hiển thị trên public leaderboard và chỉ mang tính tham khảo trong quá trình phát triển.
- **Final solution submission (Nộp giải pháp cuối cùng):** Giải pháp cuối cùng được nộp dưới dạng Docker image, theo hướng dẫn riêng của Ban Tổ chức. Để chấm điểm, Ban Tổ chức mount (gắn) thư mục private test (cùng cấu trúc với `public_test`) vào container tại **`/data`**, chạy inference script (mã suy luận) và thu kết quả tại **`result/submission.jsonl`**. Container phải chạy hoàn toàn offline (ngoại tuyến), không có network access (truy cập mạng).
- **Round 2 — Private Test (Vòng 2 — Kiểm thử riêng):** Sau khi các Đội thi nộp và freeze (đóng băng) giải pháp, Ban Tổ chức công bố `private_test` cùng environment tương ứng. Các Đội thi chạy giải pháp đã freeze với `zaic_tools` được cung cấp và nộp kết quả. Ban Tổ chức chạy lại Docker image đã nộp để xác minh kết quả, và final ranking (xếp hạng chung cuộc) được tính dựa trên điểm `private_test`.

## 8. Technical requirements (Yêu cầu kỹ thuật)

Mọi giải pháp phải đáp ứng đầy đủ các yêu cầu sau:

- **Model size (kích thước mô hình):** Toàn bộ hệ thống (ASR, LLM và mọi component (thành phần) khác) phải có tổng cộng dưới 6B parameters (tham số). LLM phải có dưới 5B parameters và context length (độ dài ngữ cảnh) tối đa 16K (16.384) tokens.
- **Data access (truy cập dữ liệu):** Agent không được đọc trực tiếp `state.json`; mọi thao tác đọc và ghi đều phải thông qua execution environment.
- **Allowed data and resources (dữ liệu và tài nguyên được phép sử dụng):**
  - Các Đội thi được sử dụng public data (dữ liệu công khai) và open-source model (mô hình mã nguồn mở) được phát hành trước ngày 26/10/2026, với điều kiện licence (giấy phép) cho phép.
  - Synthetic data (dữ liệu tổng hợp) có thể được tạo bằng các dịch vụ hoặc mô hình khác (LLM, TTS, …) trong quá trình phát triển, nhưng các dịch vụ hoặc mô hình này không được sử dụng khi inference (suy luận).
  - Không được phép gán nhãn `public_test` hoặc `private_test` (thủ công hoặc bằng dịch vụ, mô hình bên ngoài) để huấn luyện hoặc tinh chỉnh giải pháp.
- **Runtime and hardware (thời gian chạy và phần cứng):**
  - Mỗi sample phải được xử lý trong vòng 30 giây; các sample bị timeout (quá thời gian) hoặc thực thi thất bại nhận 0 điểm.
  - Giải pháp phải chạy hoàn toàn offline, không sử dụng external API (API bên ngoài) hoặc dịch vụ bên ngoài khi inference.
  - Scoring hardware (phần cứng chấm điểm): 01 GPU NVIDIA A30, CPU Intel Xeon Gold 6442Y 16 nhân và 64 GB RAM.
- **Reproducibility (khả năng tái lập):** Cố định mọi random seed (Python, NumPy, PyTorch, …) và sử dụng deterministic decoding (giải mã tất định) cho ASR và LLM, ví dụ greedy decoding hoặc `temperature = 0`. Nếu kết quả chạy lại khác với kết quả đã nộp, Ban Tổ chức sẽ chạy Docker image năm (05) lần và lấy điểm trung bình làm điểm chính thức.
- Giải pháp/agent **không được** đọc, sao chép hoặc parse bất kỳ tệp nào trong `<split>_environment/` (bao gồm `state.json`). Mọi app data chỉ được truy cập thông qua `zaic_tools` của Ban Tổ chức, với đúng một (01) `ToolEnv` cho mỗi sample, và mọi call được thực hiện trên `ToolEnv` đó phải xuất hiện trong submission.
- **Tie-breaking (xử lý trường hợp bằng điểm):** Nếu các Đội thi có cùng final score (điểm chung cuộc), Ban Tổ chức chấm lại `tool_calls` của các đội bằng công thức F1, bỏ qua bước kiểm tra `state_hash`. Nếu vẫn bằng điểm, Đội thi có average processing time per sample (thời gian xử lý trung bình mỗi sample) thấp hơn được xếp hạng cao hơn.

## 9. Submission schedule (Lịch trình nộp bài)

### 9.1. Public Test Round (Vòng kiểm thử công khai) (26/10 – 16/11)

- Các Đội thi tự xây dựng giải pháp, chạy giải pháp trên public test set (tập kiểm thử công khai) của Ban Tổ chức và nộp result file (tệp kết quả) (JSONL) qua mục Submission trên Zalo AI Challenge portal (cổng thông tin Zalo AI Challenge).
- Mỗi Đội thi được nộp tối đa năm (05) lần mỗi ngày; quota (hạn mức) sẽ được đặt lại vào cuối mỗi ngày.
- Điểm số và thứ hạng được hiển thị trên public leaderboard.
- Ngày 16/11, các Đội thi phải nộp complete solution (giải pháp hoàn chỉnh), bao gồm training code (mã huấn luyện), inference code (mã suy luận) và data (dữ liệu), dưới dạng Docker package (gói Docker) cho Ban Tổ chức qua form “Solution Submission” trên Zalo AI Challenge portal.
- Do Docker file thường có dung lượng khá lớn, các Đội thi nên cung cấp checksum (mã kiểm tra) và một public Google Drive link (liên kết Google Drive công khai) để Ban Tổ chức tải về.
- Mỗi Đội thi được nộp tối đa hai (02) lần. Ban Tổ chức chỉ đánh giá lần nộp cuối cùng. Hạn chót nộp: 23:59 (GMT+7).

### 9.2. Private Test Round (Vòng kiểm thử riêng) (20/11 – 23/11)

- Lúc 9:00 ngày 20/11, Ban Tổ chức công bố private test set (tập kiểm thử riêng).
- Các Đội thi chạy giải pháp đã nộp trên tập này và nộp result file (JSONL) qua mục Submission trên Zalo AI Challenge portal. Hạn chót nộp: 23:59 (GMT+7) ngày 23/11.
- Private leaderboard (bảng xếp hạng riêng) được hiển thị theo thứ tự thời gian (dựa trên thời điểm nộp bài).
- Ban Tổ chức sẽ đánh giá các giải pháp đã nộp để xác minh kết quả, kiểm tra source code (mã nguồn) và phát hiện mọi hành vi gian lận.
- Hai (02) Đội thi đứng đầu sẽ được công bố vào ngày 02/12.
