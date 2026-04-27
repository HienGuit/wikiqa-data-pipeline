# 🚀 Wikipedia Data Engineering Pipeline

Hệ thống tự động hóa thu thập, làm sạch và tiền xử lý dữ liệu từ Wikipedia tiếng Việt. Dự án được thiết kế chuyên biệt để tạo ra bộ Core Corpus (Tập dữ liệu lõi) chất lượng cao, phục vụ cho việc huấn luyện Mô hình Ngôn ngữ Lớn (LLM) và các hệ thống RAG (Retrieval-Augmented Generation).

## ✨ Tính năng Nổi bật

* **Tự động Khám phá & Cân bằng (Auto-Discovery & Quota Rollover):** Sử dụng thuật toán Duyệt theo chiều rộng (BFS) kết hợp Quota động. Tự động chuyển chỉ tiêu bài viết bị thiếu sang các chuyên mục khác, đảm bảo độ cân bằng cho 8 lĩnh vực tri thức.
* **Xử lý Đa luồng (Multithreading):** Tối ưu hóa I/O bound khi gọi API, giảm thời gian thu thập hàng vạn bài viết từ nhiều giờ xuống chỉ còn tính bằng phút.
* **Kiến trúc Chống lỗi (Resilience Architecture):**
    * Tích hợp Checkpoint tự động: Resume lại chính xác tiến độ nếu mất mạng/sập nguồn.
    * **Atomic Write:** Ghi dữ liệu vào file tạm (`.tmp`) và chỉ đổi tên khi hoàn tất 100%, đảm bảo tuyệt đối không rác/mất dữ liệu.
* **Plain Text Tinh khiết:** Khai thác API Extracts thay vì Wikitext, loại bỏ sạch sẽ thẻ HTML, template, và cấu trúc nhiễu, tích hợp bộ lọc độ dài (Truncate) thông minh.

## 📂 Cấu trúc Dự án

```text
wiki-data-pipeline/
├── data/                       # Chứa dữ liệu đầu ra 
│   └── taxonomy.json           # File cấu hình hạt giống (Seed Categories)
├── src/                        # Chứa Source Code lõi
│   ├── __init__.py
│   ├── config.py               # Quản lý cấu hình toàn cục
│   ├── crawler.py              # Xử lý Metadata & Thuật toán BFS
│   ├── pipeline.py             # Fetcher đa luồng & Clean Data
│   └── utils.py                # Các hàm tiện ích (Load taxonomy)
├── main.py                     # Entry point khởi chạy toàn bộ luồng
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu dự án