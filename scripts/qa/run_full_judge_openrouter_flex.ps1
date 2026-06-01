param(
    [string]$InputPath = "data/processed/datasets/qa_pairs_canonical.jsonl",
    [string]$OutputDir = "data/processed/runs/qa/judge_openrouter_gemini3_flash_flex",
    [string]$MergedOutput = "data/processed/archive/qa/judge_exports/qa_judge_openrouter_gemini3_flash_flex.jsonl",
    [string]$RejectOutput = "data/processed/archive/qa/judge_exports/qa_judge_openrouter_gemini3_flash_flex_rejects.jsonl",
    [string]$SummaryOutput = "data/processed/reports/qa/qa_judge_openrouter_gemini3_flash_flex_summary.json",
    [string]$Model = "google/gemini-3-flash-preview",
    [int]$Workers = 4,
    [int]$ShardSize = 400,
    [int]$RpmLimitPerWorker = 30,
    [int]$TimeoutSeconds = 180,
    [int]$FlushEvery = 20,
    [switch]$SkipMerge
)

$ErrorActionPreference = "Stop"

if (-not $env:OPENROUTER_API_KEY) {
    throw "Missing OPENROUTER_API_KEY. Set it in your shell before running this script."
}
if ($Workers -lt 1) {
    throw "Workers must be >= 1."
}
if ($ShardSize -lt 1) {
    throw "ShardSize must be >= 1."
}
if (-not (Test-Path $InputPath)) {
    throw "Input file not found: $InputPath"
}

$repoRoot = (Resolve-Path ".").Path
$inputFullPath = (Resolve-Path $InputPath).Path
$outputFullPath = Join-Path $repoRoot $OutputDir
$logDir = Join-Path $outputFullPath "logs"
New-Item -ItemType Directory -Force -Path $outputFullPath | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sampleCount = (Get-Content -LiteralPath $inputFullPath | Where-Object { $_.Trim() }).Count
$shardCount = [math]::Ceiling($sampleCount / $ShardSize)

$manifest = [ordered]@{
    mode = "full_judge_openrouter_flex"
    input_path = $inputFullPath
    output_dir = (Resolve-Path $outputFullPath).Path
    model = $Model
    service_tier = "flex"
    provider = "openrouter"
    sample_count = $sampleCount
    shard_size = $ShardSize
    shard_count = $shardCount
    workers = $Workers
    rpm_limit_per_worker = $RpmLimitPerWorker
    timeout_seconds = $TimeoutSeconds
    started_at_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
}
$manifestPath = Join-Path $outputFullPath "run_manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $manifestPath

Write-Host "Input samples : $sampleCount"
Write-Host "Shard count   : $shardCount"
Write-Host "Workers       : $Workers"
Write-Host "Output dir    : $outputFullPath"
Write-Host "Manifest      : $manifestPath"

$jobs = @()
for ($worker = 0; $worker -lt $Workers; $worker++) {
    $workerId = $worker
    $job = Start-Job -Name "judge-worker-$workerId" -ArgumentList @(
        $repoRoot,
        $inputFullPath,
        $outputFullPath,
        $logDir,
        $Model,
        $workerId,
        $Workers,
        $shardCount,
        $ShardSize,
        $RpmLimitPerWorker,
        $TimeoutSeconds,
        $FlushEvery
    ) -ScriptBlock {
        param(
            $RepoRoot,
            $InputFullPath,
            $OutputFullPath,
            $LogDir,
            $Model,
            $WorkerId,
            $Workers,
            $ShardCount,
            $ShardSize,
            $RpmLimitPerWorker,
            $TimeoutSeconds,
            $FlushEvery
        )

        Set-Location $RepoRoot
        $env:PYTHONIOENCODING = "utf-8"
        $workerLog = Join-Path $LogDir ("worker_{0:D2}.log" -f $WorkerId)
        $workerErr = Join-Path $LogDir ("worker_{0:D2}.err.log" -f $WorkerId)

        for ($shardIndex = $WorkerId; $shardIndex -lt $ShardCount; $shardIndex += $Workers) {
            $started = (Get-Date).ToUniversalTime().ToString("s") + "Z"
            "[$started] worker=$WorkerId shard=$shardIndex start" | Add-Content -Encoding UTF8 -Path $workerLog

            python -m src.qa.batch judge `
                --provider openrouter `
                --model $Model `
                --service-tier flex `
                --reasoning-type all `
                --input $InputFullPath `
                --shard-index $shardIndex `
                --shard-size $ShardSize `
                --output-dir $OutputFullPath `
                --flush-every $FlushEvery `
                --rpm-limit $RpmLimitPerWorker `
                --timeout $TimeoutSeconds `
                1>> $workerLog 2>> $workerErr

            if ($LASTEXITCODE -ne 0) {
                throw "Worker $WorkerId failed on shard $shardIndex. See $workerErr"
            }

            $finished = (Get-Date).ToUniversalTime().ToString("s") + "Z"
            "[$finished] worker=$WorkerId shard=$shardIndex done" | Add-Content -Encoding UTF8 -Path $workerLog
        }
    }
    $jobs += $job
}

Write-Host "Started $($jobs.Count) worker job(s). Waiting..."
Wait-Job $jobs | Out-Null

$failed = $jobs | Where-Object { $_.State -ne "Completed" }
foreach ($job in $jobs) {
    Receive-Job $job
}
Remove-Job $jobs

if ($failed) {
    throw "At least one worker failed. Check logs under $logDir"
}

Write-Host "All workers completed."

if (-not $SkipMerge) {
    python -m src.qa.dataset merge-judge `
        --input-dirs $OutputFullPath `
        --merged-output $MergedOutput `
        --reject-output $RejectOutput `
        --summary-output $SummaryOutput

    if ($LASTEXITCODE -ne 0) {
        throw "Merge failed."
    }
}

$manifest["finished_at_utc"] = (Get-Date).ToUniversalTime().ToString("s") + "Z"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $manifestPath
Write-Host "Done."
