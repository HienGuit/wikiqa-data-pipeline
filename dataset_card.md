---
language:
- vi
task_categories:
- question-answering
- text-generation
pretty_name: Vietnamese WikiQA
size_categories:
- 1K<n<10K
---

# Dataset Card for Vietnamese WikiQA

## Dataset Description

- **Repository:** Vietnamese WikiQA Pipeline
- **Language(s) (NLP):** Vietnamese (`vi`)
- **License:** CC BY-SA 4.0

### Dataset Summary

**Vietnamese WikiQA** là một bộ dữ liệu tiếng Việt chất lượng cao dành cho bài toán Máy Đọc Hiểu (Machine Reading Comprehension) và Truy xuất - Sinh văn bản (Retrieval-Augmented Generation - RAG). 

Bộ dữ liệu chứa **7,592 cặp Hỏi-Đáp (QA pairs)** được tổng hợp tự động và tinh chỉnh cực kỳ khắt khe từ Wikipedia Tiếng Việt. Mỗi cặp QA đều đi kèm với một ngữ cảnh (context) hoàn chỉnh, độc lập và được phân loại theo ba mức độ suy luận (reasoning types): `extraction` (trích xuất thông tin trực tiếp), `bridge` (bắc cầu qua nhiều câu), và `multi-sentence` (tổng hợp từ nhiều câu).

### Supported Tasks and Leaderboards

- `extractive-qa`, `open-domain-qa`: Đào tạo các mô hình RAG hoặc extractive QA tiếng Việt.
- Có thể dùng làm tập đánh giá (Benchmark) cho khả năng đọc hiểu tiếng Việt của các LLMs nhờ vào phân loại độ khó (Difficulty Band) rõ ràng.

## Dataset Structure

Bộ dữ liệu được chia theo tỷ lệ 80/10/10 với việc đảm bảo **không có sự rò rỉ dữ liệu (Zero Leakage)**: Tất cả các đoạn văn (chunk) thuộc cùng một bài viết (Title) đều được gom chung vào một tập (split) duy nhất.

| Split | Cặp QA (Rows) |
|---|---|
| `train` | 6092 |
| `val` | 761 |
| `test` | 739 |
| **Total** | **7592** |

### Phân phối Reasoning Bucket

Bộ dữ liệu duy trì tỷ lệ gần như đồng đều giữa các loại suy luận trên tất cả các tập:
- `extraction`: ~40%
- `bridge`: ~30%
- `multi-sentence`: ~30%

## Dataset Creation

### Curation Rationale

Việc tạo ra bộ dữ liệu QA tiếng Việt thường đối mặt với vấn đề sinh câu hỏi bị "hallucination" hoặc câu trả lời không thực sự nằm trong đoạn văn. Bộ dữ liệu này được thiết kế để giải quyết triệt để vấn đề đó bằng quy trình **Dual-Judge Verification**.

### Source Data

- **Wikipedia Tiếng Việt:** Dữ liệu gốc được tải xuống từ các bài viết Wikipedia thuộc các chủ đề cốt lõi (Core Domains): Kinh tế, Lịch sử, Khoa học, Địa lý, Văn hóa, Thể thao, v.v.
- Các bài viết được làm sạch HTML và cắt thành các khối ngữ nghĩa (Semantic Chunking) dài tối đa 500 từ.

### Annotations

Quy trình gán nhãn (Annotation) diễn ra hoàn toàn tự động bằng các mô hình Ngôn ngữ Lớn (LLMs) mạnh nhất nhưng được kiểm duyệt qua nhiều bước (Multi-step verification):
1. **Sinh QA ban đầu:** Sử dụng `Gemini 3.1 Flash Lite` và `DeepSeek V4 Flash`.
2. **Dual-Judge (Kiểm duyệt chéo):** Mỗi cặp QA sinh ra sẽ được chấm điểm độc lập bởi 2 giám khảo LLM khác nhau. Chỉ những cặp đạt điểm "strong" hoặc "usable" mới được giữ lại.
3. **Succinct Repair:** Các câu trả lời bị dính đại từ nhân xưng, hoặc dài dòng lặp lại câu hỏi sẽ được cắt tỉa thành dạng ngắn gọn (Succinct) nhất có thể nhưng vẫn giữ nguyên độ chính xác.

## Considerations for Using the Data

- **Giới hạn (Limitations):** Dữ liệu được lấy từ Wikipedia nên mang văn phong bách khoa toàn thư, có thể không phản ánh đúng ngữ pháp tiếng nói hàng ngày.
- **Biases:** Phụ thuộc vào độ chính xác và trung lập của các bài viết Wikipedia Tiếng Việt tính đến thời điểm crawl dữ liệu.

## Citation

```bibtex
@dataset{vietnamese_wikiqa,
  author = {Your Name / Organization},
  title = {Vietnamese WikiQA: High-Quality MRC and RAG Dataset},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/your-repo/vietnamese-wikiqa}
}
```
