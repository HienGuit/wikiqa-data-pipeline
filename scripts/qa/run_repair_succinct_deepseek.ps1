param(
    [int]$Workers = 4,
    [int]$ShardSize = 300,
    [int]$RpmLimit = 120,
    [string]$Model = "deepseek-v4-flash",
    [string]$InputPath = "data/processed/datasets/qa_pairs_canonical.jsonl",
    [string]$OutputDir = "data/processed/runs/qa/repair_succinct"
)

$ErrorActionPreference = "Stop"

if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is not set."
}

$repoRoot = (Resolve-Path ".").Path
$outputAbs = Join-Path $repoRoot $OutputDir
$logsDir = Join-Path $outputAbs "logs"
New-Item -ItemType Directory -Force -Path $outputAbs | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$manifest = @{
    started_at_utc = [DateTime]::UtcNow.ToString("s") + "Z"
    input_path = (Resolve-Path $InputPath).Path
    output_dir = $outputAbs
    workers = $Workers
    shard_size = $ShardSize
    rpm_limit = $RpmLimit
    model = $Model
    command = "python -m src.qa.batch repair-succinct"
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $outputAbs "run_manifest.json")

$processes = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $stdout = Join-Path $logsDir ("repair_worker_{0:00}.out.log" -f ($i + 1))
    $stderr = Join-Path $logsDir ("repair_worker_{0:00}.err.log" -f ($i + 1))
    $argList = @(
        "-m", "src.qa.batch", "repair-succinct",
        "--input", $InputPath,
        "--model", $Model,
        "--rpm-limit", "$RpmLimit",
        "--reasoning-type", "all",
        "--shard-index", "$i",
        "--shard-size", "$ShardSize",
        "--output-dir", $OutputDir
    )
    $proc = Start-Process -FilePath "python" -ArgumentList $argList -WorkingDirectory $repoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $processes += $proc
}

Write-Host "Started $($processes.Count) worker(s)."
foreach ($proc in $processes) {
    Write-Host ("PID={0} worker started" -f $proc.Id)
}
Write-Host "Output dir: $outputAbs"
Write-Host "Logs dir  : $logsDir"
