# WikiQA Data Pipeline

Pipeline thu thập, làm sạch, chunking, lọc mẫu, sinh QA, judge, repair, và finalization cho dữ liệu Wikipedia tiếng Việt.

## Cau truc repo

```text
wikiqa-data-pipeline/
|-- configs/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|       |-- datasets/
|       |-- runs/
|       |   `-- qa/
|       |-- reports/
|       |   `-- qa/
|       `-- archive/
|           `-- qa/
|-- eda/
|-- scripts/
|   `-- qa/
|-- src/
|   `-- qa/
|-- main.py
`-- requirements.txt
```

## Data layout

- `data/raw`: taxonomy và article-level source data
  Path chuẩn hiện tại của taxonomy là `data/raw/taxonomy.json`.
- `data/interim`: chunk pool, filtered pool, sampled pool, top-up chunk pools
- `data/processed/datasets`: dataset JSONL đang sử dụng
- `data/processed/runs/qa`: shard outputs, judge runs, repair runs
- `data/processed/reports/qa`: summary, manifest, merge reports
- `data/processed/archive/qa`: backup, judge exports cũ, exploratory artifacts

Bo artifact QA hien tai:

- `qa_pairs_canonical.jsonl`
- `qa_pairs_canonical_judged.jsonl`
- `qa_pairs_canonical_context_cleaned.jsonl`
- `qa_pairs_canonical_judged_context_cleaned.jsonl`
- `qa_pairs_split_ready.jsonl`
- `qa_inferential_usable_only.jsonl`

## QA subsystem

`src/qa/` duoc tach thanh:

- `prompts.py`: prompt generation + judge
- `provider.py`: model providers
- `generator.py`: orchestration sinh QA
- `validators.py`: validation rules
- `batch.py`: smoke, full, topup, judge, repair runners
- `dataset.py`: selection + merge utilities

Helper context cleaning dùng chung:

- `src/text_cleaning.py`

Mọi path QA quan trọng đều đi qua `src.config`.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy pipeline cơ bản

Crawler -> cleaner -> chunker:

```bash
python main.py
```

Lọc chunk / lấy mẫu:

```bash
python -m src.chunk_filter
```

## Chạy QA pipeline

Smoke test QA generation:

```bash
python -m src.qa.batch smoke
```

Full QA generation:

```bash
python -m src.qa.batch full --shard-index 0 --shard-size 800
```

Inferential top-up:

```bash
python -m src.qa.batch topup --shard-index 0 --shard-size 100
```

Judge:

```bash
python -m src.qa.batch judge --provider openrouter --reasoning-type all
```

Repair succinct contextual prefix:

```bash
python -m src.qa.batch repair-succinct --reasoning-type all
```

## Merge / dataset utilities

Chọn top-up chunks:

```bash
python -m src.qa.dataset select-topup
```

Merge main shards:

```bash
python -m src.qa.dataset merge-main
```

Merge top-up:

```bash
python -m src.qa.dataset merge-topup
```

Merge judge shards:

```bash
python -m src.qa.dataset merge-judge
```

Refresh judged / filtered downstream artifacts after repair:

```bash
python -m src.qa.dataset refresh-derived
```

## Finalization workflow

Context clean + downstream sync:

```bash
python scripts/qa/retro_clean_context.py
python scripts/qa/finalize_qa_dataset.py
```

Hoặc dùng launcher:

```powershell
.\scripts\qa\run_finalize_qa_pipeline.ps1
```

Workflow final dataset:

1. `qa_pairs_canonical.jsonl`
2. `qa_pairs_canonical_judged.jsonl`
3. `qa_pairs_canonical_context_cleaned.jsonl`
4. `qa_pairs_canonical_judged_context_cleaned.jsonl`
5. `qa_pairs_split_ready.jsonl`
6. `qa_inferential_usable_only.jsonl`

## Automation scripts

- `scripts/qa/run_full_judge_openrouter_flex.ps1`
- `scripts/qa/run_repair_succinct_deepseek.ps1`
- `scripts/qa/run_refresh_pipeline.ps1`
- `scripts/qa/run_finalize_qa_pipeline.ps1`
- `scripts/qa/build_human_verification_sets.py`
- `scripts/qa/clean_annotation_pool.py`
- `scripts/qa/retro_clean_context.py`
- `scripts/qa/finalize_qa_dataset.py`

`clean_annotation_pool.py` is now a compatibility sync:
- it copies the canonical cleaned judged pool into the historical
  `qa_pairs_canonical_judged_cleaned.jsonl` filename for older notebooks
- it does not run a separate cleaning policy anymore

## Ghi chú

- `src.config.ensure_dirs()` tự tạo cấu trúc thư mục cần thiết.
- Không hardcode path mới trong script mới; nếu cần thêm artifact, thêm constant vào `src.config` trước.
- Artifact exploratory nên đưa vào `data/processed/archive/qa/experiments` thay vì để ở root `data/processed`.
- Human verification sets hiện tại được giữ nguyên; finalization không regenerate lại các file này.
- `clean_annotation_pool.py` chỉ còn là lớp tương thích cho notebook/workflow cũ.
