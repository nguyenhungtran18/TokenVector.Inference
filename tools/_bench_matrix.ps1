# BENCH MATRIX - TKV (best-of-3) vs PyTorch, sequential, same machine, same session.
# seq 8/256/1024 x mode 0 (fwd+bwd) and mode 1 (full step).
$ErrorActionPreference = "Continue"
$root = "D:\TokenVector.Inference"
Set-Location $root

$cfg = "tv/tests/_seq_cfg.txt"
$log = "out/_bench_matrix_$(Get-Date -Format yyyyMMdd_HHmmss).log"
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
Say ("TKV binary: out/bseq.exe  " + (Get-Item out/bseq.exe).LastWriteTime)
Say ("git HEAD: " + (git rev-parse --short HEAD))
Say ""

Say "=== TOKENVECTOR (best-of-3, min) ==="
foreach ($seq in 8, 256, 1024) {
  foreach ($mode in 0, 1) {
    $best = $null
    $lines = @()
    for ($r = 1; $r -le 3; $r++) {
      Set-Content -Path $cfg -Value "$seq`n$mode`n1`n1"
      $out = & dotnet out/bseq.exe 2>$null
      $ln = $out | Where-Object { $_ -match "BSEQ (full-step|fwd=)" }
      $lines += $ln
      $ms = $null
      if ($ln -match "ms=([0-9.]+)") { $ms = [double]$Matches[1] }
      elseif ($ln -match "total=([0-9.]+)") { $ms = [double]$Matches[1] }
      if ($null -ne $ms -and ($null -eq $best -or $ms -lt $best)) { $best = $ms }
    }
    Say ("TKV seq=$seq mode=$mode best=$best ms  | runs: " + ($lines -join " || "))
  }
}
Say ""

Say "=== PYTORCH (bench_torch d: fwd+bwd and full step) ==="
foreach ($seq in 8, 256, 1024) {
  $out = & python tools/bench_torch.py d $seq 2>$null
  $out | Where-Object { $_ -match "^BENCH" } | ForEach-Object { Say $_ }
}
Say ""

Say "=== PYTORCH PHASE (bench_torch e, T=1 and T=14) ==="
foreach ($seq in 8, 1024) {
  $out = & python tools/bench_torch.py e $seq 2>$null
  $out | Where-Object { $_ -match "^BENCH" } | ForEach-Object { Say $_ }
}
Say ""
Say "LOG: $log"
