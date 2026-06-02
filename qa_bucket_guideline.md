# Hướng dẫn đánh giá QA tiếng Việt bằng Bucket Labels

**Phiên bản:** 1.0  
**Phạm vi sử dụng:** LLM-as-a-Judge và Human Verification  
**Đối tượng đánh giá:** Cặp câu hỏi - câu trả lời tiếng Việt đã qua rule-based validation  
**Nhãn chính:** `quality_band`, `difficulty_band`, `inferential_validity_band`

Tài liệu này chuẩn hóa cách đánh giá các mẫu QA tiếng Việt theo bucket labels. Mục tiêu là giảm tính cảm tính trong annotation, thống nhất tiêu chí giữa LLM-as-a-Judge và annotator con người, đồng thời tạo bằng chứng phương pháp luận cho báo cáo xây dựng dữ liệu.

> Nguyên tắc cốt lõi: rule-based validation chặn lỗi hình thức; bucket labels đánh giá chất lượng ngữ nghĩa và giá trị sử dụng của mẫu.

---

## 1. Giới thiệu bài toán

Vietnamese Domain WikiQA là bài toán xây dựng và đánh giá các cặp câu hỏi - câu trả lời tiếng Việt dựa trên ngữ cảnh văn bản. Mỗi mẫu gồm một đoạn `context`, một `question` và một `answer` dạng exact span. Nhiệm vụ đánh giá không chỉ kiểm tra mẫu có đúng hình thức hay không, mà còn xác định mẫu có đủ tự nhiên, hữu ích và đúng bản chất suy luận hay không.


---

## 2. Thành phần dữ liệu cần đánh giá

| Thành phần | Ý nghĩa | Vai trò trong đánh giá |
|---|---|---|
| `context` | Đoạn văn nguồn dùng để trả lời câu hỏi. | Nguồn duy nhất để kiểm chứng answer và evidence. |
| `question` | Câu hỏi tiếng Việt được sinh từ context. | Đánh giá độ tự nhiên, độ rõ nghĩa và mức paraphrase. |
| `answer` | Đáp án dạng exact span trong context. | Phải khớp ngữ nghĩa với question và có thể truy vết. |
| `reasoning_type` | Kiểu sinh ban đầu: `extraction` hoặc `multi-sentence`. | Quyết định bộ tiêu chí cần chấm. |


---

## 3. Nhãn đánh giá và phạm vi áp dụng

| Nhãn | Miền giá trị | Áp dụng cho | Mục đích |
|---|---|---|---|
| `quality_band` | `weak` / `usable` / `strong` | Extraction và multi-sentence | Đo chất lượng sử dụng thực tế của cặp QA. |
| `difficulty_band` | `easy` / `medium` / `hard` | Extraction và multi-sentence | Đo độ khó khi trả lời dựa trên context. |
| `inferential_validity_band` | `weak` / `usable` / `strong` | Chỉ multi-sentence | Đo xem câu hỏi có thật sự cần nhiều câu hay không. |

Khi phân vân giữa hai bucket, chọn bucket thấp hơn nếu bằng chứng chưa đủ rõ. Quy tắc này giúp nhãn bảo thủ hơn và hạn chế việc đánh giá quá dễ dãi.

---

## 4. Rule-based validation trước khi chấm bucket

Không dùng bucket labels để thay thế các kiểm tra hình thức. Các lỗi có thể xác định bằng quy tắc phải được loại trước, vì chúng không cần đến đánh giá chủ quan của LLM hoặc annotator.

| Nhóm lỗi | Mô tả | Hành động |
|---|---|---|
| Answer không hợp lệ | answer rỗng, không nằm trong context, quá dài hoặc mơ hồ. | Loại mẫu trước khi chấm bucket. |
| Question không hợp lệ | question quá ngắn, làm lộ answer, dùng cụm “theo đoạn văn”, hoặc hỏi nhiều đáp án. | Loại mẫu hoặc yêu cầu sinh lại. |
| Evidence không hợp lệ | evidence_span không liên tục, không chứa answer hoặc chỉ gồm một câu với multi-sentence. | Loại khỏi nhóm multi-sentence. |
| Answer type mismatch | Câu hỏi về khoảng thời gian nhưng answer là ngày tuyệt đối, hoặc số thiếu đơn vị. | Loại hoặc yêu cầu sửa. |

---

## 5. Tiêu chí 1: `quality_band`

`quality_band` đo chất lượng sử dụng thực tế của mẫu QA sau khi đã qua kiểm tra hình thức. Bucket này không hỏi “mẫu có đúng JSON không”, mà hỏi “mẫu này có đủ rõ, tự nhiên và hữu ích để đưa vào dataset không”.

| Bucket | Định nghĩa | Dấu hiệu nhận biết |
|---|---|---|
| `weak` | Mẫu đúng hình thức nhưng còn gượng, paraphrase kém hoặc chưa thuyết phục. | Question máy móc, bám sát câu gốc, answer chưa đủ đẹp khi ghép với question. |
| `usable` | Mẫu rõ ràng, đúng và dùng được cho huấn luyện hoặc đánh giá. | Question hiểu được ngay, answer khớp nghĩa, không có lỗi ngữ nghĩa đáng kể. |
| `strong` | Mẫu tự nhiên, gọn, khớp chặt và có thể dùng làm ví dụ tiêu biểu. | Question paraphrase tốt, độc lập, answer chính xác và thuyết phục. |

Quy tắc phân xử: nếu mẫu đúng nhưng không tự nhiên, không đẩy lên `strong`. Nếu question tự nhiên, không lộ answer và cặp QA gọn đẹp, ưu tiên `strong`.

---

## 6. Tiêu chí 2: `difficulty_band`

`difficulty_band` đo độ khó thực tế khi trả lời câu hỏi dựa trên context. Không tự động gán multi-sentence là hard, vì có những mẫu multi-sentence chỉ cần nối ý rất trực tiếp.

| Bucket | Định nghĩa | Dấu hiệu nhận biết |
|---|---|---|
| `easy` | Answer lộ hoặc chỉ cần quét một câu là thấy. | Tên riêng, ngày tháng, số liệu nổi bật; question gần câu gốc; không có distractor. |
| `medium` | Cần đọc kỹ context hoặc đối chiếu hai ý không sâu. | Có paraphrase rõ; answer không quá lộ; có ít yếu tố gây nhiễu. |
| `hard` | Cần tổng hợp rõ hoặc loại trừ thông tin cạnh tranh. | Nhiều mốc thời gian, nhiều thực thể, distractor, hoặc nhiều thông tin đúng một phần. |

---

## 7. Tiêu chí 3: `inferential_validity_band`

`inferential_validity_band` chỉ áp dụng cho mẫu multi-sentence. Tiêu chí này đo xem câu hỏi có thật sự cần kết hợp nhiều câu hay chỉ được gắn nhãn multi-sentence một cách hình thức.

| Bucket | Định nghĩa | Sentence removal test |
|---|---|---|
| `weak` | Thực chất một câu là đủ, hoặc câu phụ không thật sự cần thiết. | Bỏ câu phụ vẫn trả lời chắc chắn như cũ. |
| `usable` | Cần nối ít nhất hai ý, nhưng mối nối còn khá trực tiếp. | Bỏ một câu làm giảm độ chắc chắn nhưng đôi khi vẫn đoán được. |
| `strong` | Nhiều câu đóng vai trò thiết yếu; thiếu một ý thì không trả lời chắc chắn. | Bỏ câu quan trọng khiến answer không còn được chứng minh đầy đủ. |

**Test thực hành:** tách context thành từng câu, lần lượt che từng câu, rồi kiểm tra liệu người đọc còn trả lời chắc chắn không.

---

## 8. Hướng dẫn cho LLM-as-a-Judge

LLM-as-a-Judge phải đánh giá theo đúng bucket đã định nghĩa, không phát minh thêm nhãn và không suy diễn vượt quá context. Với trường hợp biên, mô hình chọn bucket thấp hơn nếu evidence chưa đủ rõ.

### Output schema đề xuất cho extraction

```json
{
  "quality_band": "weak | usable | strong",
  "difficulty_band": "easy | medium | hard"
}
```

### Output schema đề xuất cho multi-sentence

```json
{
  "quality_band": "weak | usable | strong",
  "difficulty_band": "easy | medium | hard",
  "inferential_validity_band": "weak | usable | strong"
}
```

Nguyên tắc chấm: không nâng bucket chỉ vì chủ đề nghe học thuật; luôn hỏi liệu answer có được chứng minh bởi context không; với multi-sentence, luôn kiểm tra câu hỏi có thật sự cần hơn một câu hay không.

---

## 9. Hướng dẫn cho human verification

Human verification dùng để kiểm tra tính hợp lý của nhãn do LLM gán, hiệu chỉnh rubric khi phát hiện LLM quá dễ dãi hoặc quá khắt khe, và tạo bằng chứng đánh giá thủ công cho báo cáo.

| Bước | Thao tác | Câu hỏi tự kiểm tra |
|---|---|---|
| 1 | Đọc toàn bộ context, question, answer. | Không chấm khi chỉ nhìn question hoặc answer riêng lẻ. |
| 2 | Gán quality_band. | Question có tự nhiên, rõ nghĩa và khớp answer không? |
| 3 | Gán difficulty_band. | Answer có lộ không? Có distractor hoặc cần đọc kỹ không? |
| 4 | Nếu là multi-sentence, gán inferential_validity_band. | Dùng sentence removal test để kiểm tra từng câu chứng cứ. |

---

## 10. Inter-Annotator Agreement và calibration

Trước khi human verification diện rộng, cần calibration trên một tập nhỏ gồm cả extraction và multi-sentence. Nhiều annotator chấm độc lập, sau đó so sánh các trường hợp bất đồng để thống nhất cách xử lý case biên.

Độ tin cậy giữa annotator được đo bằng Cohen’s Kappa hoặc một chỉ số tương đương. Nếu một dimension có agreement thấp, đó là tín hiệu guideline chưa đủ rõ hoặc nhãn đó là vùng khó cần thảo luận trong phần limitations.

| Giai đoạn | Mục tiêu |
|---|---|
| Calibration set | Chọn 15-20 mẫu đại diện, gồm cả dễ, khó và case biên. |
| Independent annotation | Annotator chấm độc lập, không thảo luận trong lúc chấm. |
| Disagreement review | Ghi lại trường hợp lệch nhãn và lý do. |
| Guideline update | Bổ sung quy tắc phân xử nếu phát hiện định nghĩa chưa rõ. |

---

## 11. Khuyến nghị sử dụng bucket trong pipeline

- Dùng `quality_band` làm tín hiệu chính để giữ hoặc ưu tiên mẫu.
- Dùng `difficulty_band` làm metadata cho EDA và phân tích độ khó, không nên dùng làm tiêu chí loại mặc định.
- Dùng `inferential_validity_band` để rà lại nhóm multi-sentence và tách các mẫu suy luận yếu sang vùng bridge nếu cần.
- Giữ các mẫu có `quality_band` là `usable` hoặc `strong`; cân nhắc loại hoặc hạ cấp các mẫu `weak`.
- Với multi-sentence, ưu tiên các mẫu có `inferential_validity_band` là `usable` hoặc `strong`.

---

## Phụ lục A. Prompt rút gọn cho LLM-as-a-Judge

### A.1. Extraction judge

```text
Bạn là chuyên gia đánh giá QA tiếng Việt.
Hãy gán bucket cho mẫu extraction theo hai tiêu chí:
- quality_band: weak | usable | strong
- difficulty_band: easy | medium | hard

Chỉ trả về JSON đúng schema:
{"quality_band":"...","difficulty_band":"..."}
```

### A.2. Multi-sentence judge

```text
Bạn là chuyên gia đánh giá QA tiếng Việt.
Hãy gán bucket cho mẫu multi-sentence theo ba tiêu chí:
- quality_band: weak | usable | strong
- difficulty_band: easy | medium | hard
- inferential_validity_band: weak | usable | strong

Luôn tự hỏi: câu hỏi có thật sự cần hơn một câu để trả lời không?
Chỉ trả về JSON đúng schema:
{"quality_band":"...","difficulty_band":"...","inferential_validity_band":"..."}
```
