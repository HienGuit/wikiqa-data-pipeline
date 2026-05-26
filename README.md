# WikiQA Data Pipeline

Pipeline thu thập, làm sạch, chunking, phân tích EDA và lọc mẫu cho dữ liệu Wikipedia tiếng Việt phục vụ QA / RAG / huấn luyện LLM.

Repo hiện được tổ chức theo 3 lớp dữ liệu:
- `data/raw`: dữ liệu nguồn và metadata đã làm sạch ở mức bài viết
- `data/interim`: chunk pool, filtered pool, sampled pool
- `data/processed`: các bước QA dataset về sau

## Kiến trúc chính

Luồng xử lý hiện tại:
1. `src.crawler` thu metadata bài viết từ taxonomy
2. `src.content_cleaner` tải plain-text extract từ Wikipedia
3. `src.chunker` tạo `wiki_chunks.jsonl`
4. `src.chunk_filter` lọc kỹ thuật + lọc chất lượng + lấy mẫu cân bằng theo domain
5. `eda/notebooks/01_chunk_pool_eda.ipynb` dùng để phân tích chunk pool

Các đường dẫn dùng chung đều đi qua `src.config`.

## Cấu trúc repo

```text
wikiqa-data-pipeline/
├── configs/
│   ├── filter_config.yaml
│   └── qa_gen_config.yaml
├── data/
│   ├── raw/
│   │   ├── taxonomy.json
│   │   ├── wiki_pages_raw.jsonl
│   │   └── wiki_pages_content.jsonl
│   ├── interim/
│   │   ├── wiki_chunks.jsonl
│   │   ├── chunks_filtered.jsonl
│   │   └── chunks_sampled.jsonl
│   └── processed/
├── eda/
│   ├── figures/
│   │   └── 01_chunk_pool_eda/
│   │       ├── chunk_distribution/
│   │       ├── text_quality/
│   │       └── sampling_strategy/
│   ├── notebooks/
│   │   ├── 01_chunk_pool_eda.ipynb
│   │   └── 02_qa_dataset_eda.ipynb
│   └── utils/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── crawler.py
│   ├── content_cleaner.py
│   ├── chunker.py
│   ├── chunk_filter.py
│   └── utils.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Chạy pipeline

Chạy toàn bộ flow crawler -> content cleaner -> chunker:

```bash
python main.py
```

Chạy riêng bước chunk filtering và sampling:

```bash
python -m src.chunk_filter
```

## Chạy EDA

Notebook chính:
- `eda/notebooks/01_chunk_pool_eda.ipynb`

Notebook này sẽ lưu figure vào:
- `eda/figures/01_chunk_pool_eda/chunk_distribution`
- `eda/figures/01_chunk_pool_eda/text_quality`
- `eda/figures/01_chunk_pool_eda/sampling_strategy`

## QA test script

Script QA test hiện dùng `google.genai` và mặc định đọc từ:
- `data/interim/chunks_sampled.jsonl`

Output mặc định:
- `tests/chunk_for_tests/qa_test_output.jsonl`

## Kiểm thử

```bash
pytest -q
```

## Ghi chú

- `src.config.ensure_dirs()` sẽ tự tạo các thư mục cần thiết.
- `configs/filter_config.yaml` chứa ngưỡng lọc/sampling lấy từ EDA.
- `data/processed/` đang là placeholder cho các bước QA dataset downstream.
