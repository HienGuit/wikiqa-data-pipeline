param()

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
Set-Location $repoRoot

python scripts\qa\retro_clean_context.py
if ($LASTEXITCODE -ne 0) {
    throw "retro_clean_context failed."
}

python scripts\qa\finalize_qa_dataset.py
if ($LASTEXITCODE -ne 0) {
    throw "finalize_qa_dataset failed."
}

python scripts\qa\clean_annotation_pool.py
if ($LASTEXITCODE -ne 0) {
    throw "clean_annotation_pool failed."
}

Write-Host "QA dataset finalization completed."
