param(
    [string]$CanonicalInput = "data/processed/datasets/qa_pairs_canonical.jsonl",
    [string]$JudgedPath = "data/processed/datasets/qa_pairs_canonical_judged.jsonl",
    [string]$FilteredPath = "data/processed/datasets/qa_pairs_split_ready.jsonl",
    [string]$InferentialPath = "data/processed/datasets/qa_inferential_usable_only.jsonl",
    [string]$ReportOutput = "data/processed/reports/qa/qa_refresh_derived_report.json"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
Set-Location $repoRoot

python -m src.qa.dataset refresh-derived `
    --canonical-input $CanonicalInput `
    --judged-path $JudgedPath `
    --filtered-path $FilteredPath `
    --inferential-path $InferentialPath `
    --report-output $ReportOutput

if ($LASTEXITCODE -ne 0) {
    throw "Refresh-derived failed."
}

Write-Host "Refresh completed."
