# TokenVector.Inference Competitor Benchmark & Stress Test Report

## 1. Executive Summary
Benchmark and stress suites target the pure **TokenVector (`.tkv`)** engine only. Results
must be measured with the `tkvc` toolchain on the hardware under test — **do not quote
numbers from any previous backend**.

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
& $tkvc build tv/benchmarks/compare_rivals.tkv --out compare_rivals.exe
.\compare_rivals.exe
# or, with the whole pipeline:
powershell -File build_inference.ps1 -BenchmarkTimeoutSeconds 300
```

**Read this before quoting any number.** As of 2026-09-28 the timing helpers in
`compare_rivals.tkv` (`now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`) are still
stubs returning `0`, so the suite prints `0.0 ms`, `0 B` and `req/s` values derived from
a division by zero. **No timing number in this suite is currently meaningful.** The
`QUICK-RUN SCALE` banner in the output says so too. Wire up real timing helpers before
recording anything.

Hardware note: there is **no SIMD**. `tv/optimization/kernels.tkv` uses a hand-rolled
8-wide blocked loop (`VEC_W = 8`) that is still scalar, so a machine with AVX2/FMA gains
nothing today. See `SIMD_NOTES.md` for why and what the compiler would need.

---

## 2. Suite Structure (compare_rivals.tkv)

The suite has six scenarios. The first four are benchmarks, the last two are stress.

| Scenario | What it measures |
| :--- | :--- |
| Cold-start & arena alloc | Graph build + arena setup, via `session_from_graph` |
| In-process single-sample latency | Fused linear / elementwise kernels |
| Heap per N inferences | Allocation pressure across repeated `run_single` |
| FP32 vs INT8 throughput | Quantized GEMM vs float reference |
| Hot-path endurance | Continuous inference loop, reports GC counters |
| Concurrent burst | `DynamicBatcher.enqueue` in a loop, reports ok/err |

Iteration counts are deliberately small (latency 500, heap 500, throughput 200,
endurance 2000, burst 500) so the suite finishes in CI. These are smoke/regression
gates, not a performance comparison against the C# SIMD baseline.

`stress_concurrent` reports `ok=` and `err=`: an `err` count above zero means requests
failed rather than merely ran slowly.

---

## 3. Recording Results

When you run the suite, append a dated table here with:

- Host CPU model and flags
- `tkvc` build id / commit hash
- Scenario, metric, median ± spread
- Rival baseline command lines for the same machine (ONNX Runtime, llama.cpp, PyTorch) if compared

Until a fresh run is recorded, leave result cells empty rather than copying historical
figures. Same rule for the hardcoded ORT/Torch constants in the source: they are
literals in the file, not measurements from this machine.
