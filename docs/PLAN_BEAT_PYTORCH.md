# PLAN: TokenVector.Inference beat PyTorch (GPT-2 124M full train step)

Ngay: 2026-11-04
Muc tieu: full step <= PyTorch 14T (1042.7 / 1572.7 / 4148.4 ms) o ca 3 seq, gate 135/135 + trainer 15/15 + loss EXACT.

## 0. Baseline hien tai (do moi)
- TV SIMD:
  - seq8 fwd+bwd 1750, full step 3063
  - seq256 fwd+bwd 4375, full step 5141
  - seq1024 fwd+bwd 13078, full step 15297
- PyTorch:
  - seq8 full step 1072
  - seq1024 fwd+bwd 3344.7, full step 4138.4
- Gap: backward 73-79% toan bo gap; copy round-trip + interpreted loops + threading.

## 1. Nguyen nhan (do duoc)
1. Backward nang gap ~5x forward (GEMM volume ~3x).
2. Copy round-trip ~40% (r1..r5, copy-back, zero_grad).
3. list[f32] interpreted trong BWD (bounds-check + indexing).
4. Threading chi 1.37 thread (torch scale 1.33/2.77/3.64x).
5. Chi phi co dinh 38.6M (m6 + op_embed_bwd) = 578ms/step.

## 2. Phases (theo thu tu)
### P0 - Do va ghi so (hoan thanh)
- Da do TV va PyTorch, ghi bang so.
### P1 - H4a/H4b: giu scratch m6 + bo copy 154MB
- File: tv/training/gpt2_ref.tkv, gpt2_ref124.tkv
- Muc tieu: -380~-430ms/step.
- Gate: 135/135 + loss exact.
### P2 - H2: codegen list[f32] nhanh
- Yeu cau compiler team: bo 48 TkvInt:: trong gpt2_block_bwd, hoist mu, LICM.
- Fallback: builtin softmax_bwd/layernorm_bwd/gelu_bwd/bias_grad.
- Muc tieu: block_bwd 1617 -> ~45ms/goid.
### P3 - H3: threading 14C
- Sweep TKV_GEMM_THREADS 4/8/14.
- Thread native cho elementwise/norm/optimizer.
- BWD parallel op-level trong DLL.
- Muc tieu: seq8 ~1.5-1.7s, seq1024 ~7s.
### P4 - H4c: native scatter op_embed_bwd
- Muc tieu: 234ms -> ~50ms.
### P5 - Peel cuoi theo budget table
- Moi bucket <5% bo qua; chi lam lever co so do.

## 2b. Trang thai do lai 2026-11-05 (HANDOFF §0.64) — doi thu tu P3/P5

Do lai nen tang trong cung phien, 3 ket luan lat nguoc gia dinh cu:

1. **May KHONG co AVX-512** (CPUID `AVX512F=0`; `/arch:AVX512` -> illegal instruction).
   Duong "dung ISA nhu torch" da dong.
2. **GEMM khong phai nut that.** tkv GEMM 1 luong 34.5 GFLOP/s, dinh may 75.5 (torch 76.2,
   numpy 75.7 — torch chi chay dung dinh, khong co phep thuong). Nhung GEMM chi ~31%
   thoi gian 1 luong va ~17% o 14 luong. Da thu 4 bien the micro-kernel (5x16, 6x16,
   tang con tro) + sweep KC: **khong cai nao nhanh hon** -> **hoan P2/P3 muc kernel**.
3. **Nut that la duong non-GEMM:** 26.5s non-GEMM (1 luong) so voi 9.8s cua torch = **2.7×**,
   chiem 69% thoi gian. Thread scale 2.31× vs 3.70× cua torch la nut that thu hai.

Thu tu moi: **A non-GEMM (.tkv elementwise + TkvInt + copy) > B thread scaling >
C determinism theo so luong > D kernel GEMM (hoan)**.

Canh bao: loss doi theo so luong (13.442917 / 13.442922 / 13.442913 o 1T/7T/14T) — phai
chot so luong hoac sua diem cong khong tat dinh truoc khi so voi PyTorch tren may khac.

## 2c. Trang thai 2026-11-05 (HANDOFF §0.72) — da lam xong duong "native cho k nho"

Ha nguong `_smm_core` tu `k >= 256` xuong **`k >= 64`** (3 tep simd_matmul). Moi GEMM cua
GPT-2 124M co `k ∈ {64, 768, 3072}` nen **khong con GEMM nao di duong tiled 256 tile**.

| seq | truoc | sau | torch T=14 | ti le |
|---:|---:|---:|---:|---:|
| 8 | 3829 | **2875** | 1009.4 | 2.85x |
| 256 | 5672 | **5218** | 1538.4 | 3.39x |
| 1024 | 16375 | **14516** | 4077.1 | 3.56x |

Gate 9/10 (gap_tests hong `InvalidProgramException` tren .NET 10 — co tu truoc, da chung minh
bang stash), trainer 15/15, loss seq8/seq1024 EXACT, seq256 lech 4e-6 (ULP native vs tiled).

**Bang so moi (seq1024, 14 luong):** fwd 3344 (3.08x) · **bwd 10593 (4.51x)** · opt 1188 (1.80x)
· copyback 141 · tong 15266 vs torch 4134.

### Thu tu lever moi (thay the P2/P3 o tren)
1. **BWD non-GEMM** — 8.2 s trong 11.2 s chenh lech. `gpt2_block_bwd` 1 luong: mlp_bwd 50.5%
   · attention_bwd 36.6% · layernorm 3.0%. Can native `layernorm_bwd`/`gelu_bwd`/`softmax_bwd`/
   `bias_grad` (compiler team, REQUEST #9/#11) hoac cat vong `.tkv` bang cac op native da co.
2. **Thread scale 2.3x vs 3.7x** — song song cap op trong bwd (hien chi GEMM song song).
3. **opt.step 1.8x** — 1.19 s khong scale (r1 copy 498 MB + adamw + r3 copy-back).
4. GEMM kernel: **hoan** (da dat 46% dinh 1 luong, 4 biên the micro-kernel khong cai nao hon).

## 3. Win condition
- seq8 < 1042.7 ms
- seq256 < 1572.7 ms
- seq1024 < 4148.4 ms
- Loss EXACT, gate xanh, RAM seq1024 khong tang.

## 4. Quy tac
- Do truoc, sua sau.
- Gate 135/135 sau moi thay doi.
- Ghi so vao HANDOFF/report sau moi phase.
}