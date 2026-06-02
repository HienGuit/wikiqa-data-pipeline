# Data Dictionary

Tài liệu này mô tả chi tiết schema (cấu trúc dữ liệu) của bộ dữ liệu Vietnamese WikiQA. Bộ dữ liệu bao gồm hai định dạng chính: **Tập dữ liệu hỏi đáp (QA Pairs)** và **Ma trận đặc trưng (Feature Matrix)**.

## 1. QA Pairs (`train.jsonl`, `val.jsonl`, `test.jsonl`)

Mỗi dòng trong các file `.jsonl` là một object JSON đại diện cho một cặp Hỏi-Đáp kèm theo ngữ cảnh từ Wikipedia Tiếng Việt.

| Tên trường | Kiểu dữ liệu | Ý nghĩa | Miền giá trị (Domain) |
|---|---|---|---|
| `chunk_id` | String | ID định danh duy nhất cho đoạn văn (chunk). | Ví dụ: `19794_0` |
| `domain` | String | Lĩnh vực/Chủ đề của bài viết Wikipedia. | `Kinh te`, `Van hoa`, `Lich su`, `Khoa hoc`, `Dia ly`... |
| `title` | String | Tựa đề bài viết Wikipedia. | Bất kỳ chuỗi văn bản nào |
| `section` | String | Tiêu đề của mục con (heading) chứa ngữ cảnh. | Bất kỳ chuỗi văn bản nào |
| `context` | String | Ngữ cảnh văn bản (Chunk) dùng để trả lời câu hỏi. | Văn bản tiếng Việt, max 500 từ. |
| `reasoning_type` | String | Loại câu hỏi do mô hình tự quyết định khi sinh. | `extraction`, `bridge`, `multi-sentence` |
| `question` | String | Câu hỏi tiếng Việt được sinh tự động. | Câu kết thúc bằng dấu `?`. |
| `answer` | String | Câu trả lời (hoặc trích xuất). Không dài dòng. | Bất kỳ chuỗi văn bản nào. |
| `is_valid` | Boolean | Trạng thái hợp lệ của cặp QA. | `true`, `false` |
| `error` | String/Null | Chuỗi mô tả lỗi (nếu sinh bị thất bại hoặc bị lọc). | `null`, `too_short`, `hallucination`,... |
| `quality_band` | String | Nhãn chất lượng sinh ra từ bước Dual-Judge. | `strong`, `usable`, `weak` |
| `difficulty_band` | String | Nhãn độ khó của cặp QA sinh ra từ Dual-Judge. | `easy`, `medium`, `hard` |
| `inferential_validity_band` | String | Mức độ suy luận (liên kết nhiều câu) sinh từ Dual-Judge. | `strong`, `usable`, `weak` |
| `final_reasoning_bucket` | String | Nhãn phân loại cuối cùng (đã hiệu chỉnh sau bước Judge). | `extraction`, `bridge`, `multi-sentence` |

---

## 2. Feature Matrix (`feature_matrix_final.csv`)

Đây là ma trận chứa các đặc trưng số học (Numerical & Categorical Features) được trích xuất từ dữ liệu QA, dùng để phân tích EDA và huấn luyện các mô hình Machine Learning phụ (nếu có).

| Tên trường | Kiểu dữ liệu | Ý nghĩa | Miền giá trị (Domain) |
|---|---|---|---|
| `row_id` | String | ID định danh duy nhất (tương tự `chunk_id`). | `19794_0` |
| `chunk_id` | String | ID của đoạn văn Wikipedia. | - |
| `title` | String | Tựa đề bài viết. | - |
| `domain` | String | Lĩnh vực bài viết. | - |
| `section` | String | Mục chứa đoạn văn. | - |
| `reasoning_type` | String | Loại câu hỏi ban đầu. | `extraction`, `bridge`, `multi-sentence` |
| `final_reasoning_bucket` | String | Phân loại câu hỏi cuối cùng. | `extraction`, `bridge`, `multi-sentence` |
| `quality_band` | String | Chất lượng cặp QA. | `strong`, `usable`, `weak` |
| `difficulty_band` | String | Độ khó cặp QA. | `easy`, `medium`, `hard` |
| `inferential_validity_band` | String | Mức độ suy luận. | `strong`, `usable`, `weak` |
| `q_length` | Integer | Độ dài câu hỏi (số lượng từ/token). | $\ge 1$ |
| `a_length` | Integer | Độ dài câu trả lời (số lượng từ/token). | $\ge 1$ |
| `ctx_length` | Integer | Độ dài ngữ cảnh (số lượng từ/token). | $\ge 1$ |
| `ctx_sentence_count` | Integer | Số lượng câu trong ngữ cảnh. | $\ge 1$ |
| `answer_position_ratio` | Float | Tỷ lệ vị trí của câu trả lời trong ngữ cảnh (0.0 - 1.0). | $[0.0, 1.0]$ |
| `lexical_overlap_ratio` | Float | Tỷ lệ trùng lặp từ vựng giữa câu hỏi và câu trả lời. | $[0.0, 1.0]$ |
| `ttr_question` | Float | Type-Token Ratio của câu hỏi (độ đa dạng từ vựng). | $[0.0, 1.0]$ |
| `question_type` | String | Từ để hỏi phổ biến (`Cái gì`, `Ở đâu`, `Khi nào`...). | Categorical |
| `answer_type` | String | Phân loại câu trả lời (định danh, số lượng, thời gian). | Categorical |
| `popularity_source` | String | Nguồn phổ biến (nếu có). | - |
| `knowledge_difficulty` | Float | Độ khó kiến thức ước tính. | Giá trị thực |
| `page_views_rank` | Integer | Thứ hạng lượt xem bài viết Wikipedia. | $> 0$ |
| `wiki_count_rank` | Integer | Thứ hạng mức độ xuất hiện/link nội bộ. | $> 0$ |
| `statements_rank` | Integer | Thứ hạng số lượng phát biểu (statement) trên Wikidata. | $> 0$ |

> **Lưu ý:** Ma trận đặc trưng này đã được qua bước loại bỏ các đặc trưng đa cộng tuyến (Multicollinearity pruning threshold = 0.80) để đảm bảo chất lượng và tính độc lập của các đặc trưng.
