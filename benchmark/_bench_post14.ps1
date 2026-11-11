# BENCH MATRIX post-#14 - TKV (benchmark/bseq_post14.exe, compiler moi) vs PyTorch,
# cung phien, cung may. seq 8/256/1024. mode 0 (fwd+bwd) / mode 1 (full step,
# best-of-3) / mode 2 (pha 5 buoc, 1 lan) + torch tier d / e.
$ErrorActionPreference = "Continue"
$root = "D:\TokenVector.Inference"
Set-Location $root

$cfg = "tv/tests/_seq_cfg.txt"
$exe = "benchmark/bseq_post14.exe"
$origCfg = Get-Content $cfg -Raw
$log = "out/_bench_matrix_post14_$(Get-Date -Format yyyyMMdd_HHmmss).log"
function Say($s) { $s | Tee-Object -FilePath $log -Append }

Say "=== HOST ==="
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Say ("CPU: " + $cpu.Name.Trim() + " | cores=" + $cpu.NumberOfCores + " logical=" + $cpu.NumberOfLogicalProcessors)
Say ("Affinity (usable) logical CPUs: " + [System.Environment]::ProcessorCount)
Say ("OS: " + [System.Environment]::OSVersion.VersionString)
Say ("dotnet: " + (dotnet --version))
$tv = (& python -c "import torch;print(torch.__version__)" 2>$null | Select-Object -Last 1)
$tt = (& python -c "import torch;print(torch.get_num_threads())" 2>$null | Select-Object -Last 1)
Say ("torch: " + $tv + " threads=" + $tt)
Say ("TKV binary: $exe  " + (Get-Item $exe).LastWriteTime)
Say ("compiler: D:\TokenVector\Release\dist\tkvc-tv.exe  " + (Get-Item "D:\TokenVector\Release\dist\tkvc-tv.exe").LastWriteTime + "  sha256=" + (Get-FileHash "D:\TokenVector\Release\dist\tkvc-tv.exe" -Algorithm SHA256).Hash.Substring(0,16))
Say ("native dll: " + (Get-Item "D:\TokenVector\Release\dist\tkv_simd_native.dll" -ErrorAction SilentlyContinue).LastWriteTime)
Say ("git HEAD: " + (git rev-parse --short HEAD))
Say ""

Say "=== TOKENVECTOR mode 0 (fwd+bwd) best-of-3 ==="
foreach ($seq in 8, 256, 1024) {
  $best = $null; $lines = @()
  for ($r = 1; $r -le 3; $r++) {
    Set-Content -Path $cfg -Value "$seq`n0`n1`n1"
    $out = & dotnet $exe 2>$null
    $ln = $out | Where-Object { $_ -match "BSEQ" }
    $lines += ($ln -join " ")
    foreach ($l in $ln) {
      if ($l -match "total=([0-9.]+)") { $ms = [double]$Matches[1]
        if ($null -eq $best -or $ms -lt $best) { $best = $ms } }
    }
  }
  Say ("TKV seq=$seq mode=0 best=$best ms  | runs: " + ($lines -join " || "))
}
Say ""

Say "=== TOKENVECTOR mode 1 (full step) best-of-3 ==="
foreach ($seq in 8, 256, 1024) {
  $best = $null; $lines = @()
  for ($r = 1; $r -le 3; $r++) {
    Set-Content -Path $cfg -Value "$seq`n1`n1`n1"
    $out = & dotnet $exe 2>$null
    $ln = $out | Where-Object { $_ -match "BSEQ full-step" }
    $lines += ($ln -join " ")
    if ($ln -match "ms=([0-9.]+)") { $ms = [double]$Matches[1]
      if ($null -eq $best -or $ms -lt $best) { $best = $ms } }
  }
  Say ("TKV seq=$seq mode=1 best=$best ms  | runs: " + ($lines -join " || "))
}
Say ""

Say "=== TOKENVECTOR mode 2 (5 pha, 1 lan) ==="
foreach ($seq in 8, 256, 1024) {
  Set-Content -Path $cfg -Value "$seq`n2`n1`n1"
  $out = & dotnet $exe 2>$null
  $ln = $out | Where-Object { $_ -match "BPHASE|BSEQ" }
  Say ("TKV seq=$seq mode=2: " + ($ln -join " | "))
}
Set-Content -Path $cfg -Value $origCfg
Say ""

Say "=== PYTORCH (bench_torch d: fwd+bwd and full step) ==="
foreach ($seq in 8, 256, 1024) {
  $out = & python tools/bench_torch.py d $seq 2>$null
  $out | Where-Object { $_ -match "^BENCH" } | ForEach-Object { Say $_ }
}
Say ""

Say "=== PYTORCH PHASE (bench_torch e, T=1 and T=14) ==="
foreach ($seq in 8, 256, 1024) {
  $out = & python tools/bench_torch.py e $seq 2>$null
  $out | Where-Object { $_ -match "^BENCH" } | ForEach-Object { Say $_ }
}
Say ""
Say "LOG: $log"
