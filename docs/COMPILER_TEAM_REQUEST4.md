# YÊU CẦU COMPILER TEAM #4 — ĐIỀU TRA `adamw_step_glm` (n=124M) + ĐÍNH CHỈNH SỐ COPY-BACK

**Ngày:** 2026-10-27 · **Nguồn:** HANDOFF §0.53 (Phase B) · **Repro:** `tv/tests/_gl1.tkv`
**Ghép với:** đơn #3 §2.2 (số copy-back) — đơn Phase C tách thành **#5**

Hai mục: **(A)** `adamw_step_glm` chậm trong production bench nhưng KHÔNG chậm khi đo
gọi trực tiếp — cần tìm ra cơ chế. **(B)** Đính chính số «copy-back 844 ms → 30 ms»
trong đơn #3 — số thật lệch cả hai phía và A/B «0%» của repo là **đo sai chỗ**.

---

## Mục A — `adamw_step_glm` @ n=124,439,808: production +1.9 s, probe trực tiếp ≈ parity

### A1. Số đo production (bench full-step, seq1024, best-of-3 xen kẽ, cùng tkvc 21:48 + DLL 20:55)

| exe | cấu hình step | kết quả vs pb0 cùng phiên |
|---|---|---|
| `bseq_pb1` (glm + list-add + unflatten) | nhánh `coef==1.0` → `adamw_step_glm(gl list)` | **+1.9 s** (pb1 24156/24719/24297 vs pb0 22250/22625/23141 — 3/3 rep, chênh đều) |
| `bseq_pb2` (glm + list-add) | như trên | +1.5 s (1 rep: 24172 vs 22625) |
| `bseq_pb3` (không glm — `r1 copy` + `adamw_step` nguyên bản) | original | **−1.7 s** (21062 best vs 22719) |

Bisect: pb2 vs pb3 **chỉ khác duy nhất khối step()** → tín hiệu quy cho glm branch
là **+1.5…+3.2 s/bước @seq1024** (số pb2/pb3 chênh ~3.2 s nhưng pb3 spread 2.1 s —
mời team tự chạy lại 3 rep để lấy median chắc hơn).

### A2. Số đo gọi TRỰC TIẾP — probe `_gl1` (không lặp lại được)

```text
GL1 [1b] bitdiff adamw_step vs glm: 0/1024          # numerics đúng hợp đồng
GL1 [1a] n=1048576     adamw_step=0.0 ms | glm=0.0 ms   # dưới phân giải 15.6 ms
GL1 [1c] n=124439808   adamw_step=297.0 ms | glm=282.0 ms
```

Cùng input, xen kẽ 3 vòng best-of, warmup trước — `adamw_step_glm` **nhanh hơn**
(được −78 ms copy r1 như cam kết) trong process probe đứng máy idle.

→ **Nghịch lý cần giải:** call trực tiếp parity/nhanh hơn, nhưng trong bench full-step
(seq1024, heap 8–9 GB, sau FWD+BWD) glm branch tốn thêm ~1.9 s.

### A3. Giả thuyết (chưa chốt — cần team kiểm)

1. **Pin managed array 496 MB dưới áp lực GC**: wrapper glm phải pin
   `List<Single>._items` (496 MB) mỗi call; `adamw_step` truyền buffer native (không
   pin). Ở heap đầy/gen2 churn của bench, pin+unpin + GC sync có thể đắt; ở probe
   (heap sạch, 2.5 GB) thì không.
2. Hidden alloc trong `AdamwGlmM` khi n lớn (mặc dù `assert_il` đã đếm `ToArray = 0`
   — cần đếm cả alloc/GCHandle/box khác trong wrapper).
3. Tác động gián tiếp: bỏ `f32_copy_to_buf` r1 làm đổi **thời điểm GC** trong step
   (memcpy 496 MB là điểm giữ nhịp) → gen2 collection dồn vào phase nặng hơn.

### A4. Repro cho team

- **Call trực tiếp:** `tkvc-tv.exe tv/tests/_gl1.tkv out/gl1.exe main && ./out/gl1.exe`
  (máy này — source đầy đủ trong `_gl1.tkv`, tự chứa, 0 thay đổi code train).
- **Production bench:** `powershell out/ab_bench_pb.ps1 -Seq 1024 -Reps 3
  -ExeA out/bseq_pb0.exe -ExeB out/bseq_pb1.exe` (pb1 = có glm).
- Đề nghị chạy `_gl1` ở **2 điều kiện**: máy idle (chúng tôi đo được parity) và sau khi
 _alloc/churn ~8 GB trước đó_ (mô phỏng bench); nếu slowdown chỉ hiện ở điều kiện 2
  → khẳng định giả thuyết GC/pin.

### A5. Ask

Rà IL wrapper `AdamwGlmM` vs `AdamwStepM` line-by-line (alloc/pin/box), chạy repro 2
điều kiện, trả lời: **cơ chế là gì + cách sửa**. Nếu không sửa được trong ngắn hạn,
xác nhận chính thức để repo ghi nhận «glm chỉ dùng khi RAM dư dả» (hiện tại **đã
drop theo bench**, giữ nguyên `r1 copy` — xem comment `step()` trong `trainer.tkv`).

---

## Mục B — Đính chính «copy-back 844 ms → 30 ms» (đơn #3 §2.2)

### B1. Số cũ và số mới

| nguồn | interpreted | native | ghi chú |
|---|---|---|---|
| đơn #3 §2.2 (ước của repo) | **844 ms** | ~30 ms | ước từ HANDOFF A3 cũ |
| `bench_dp` của team (reply #3 §4.1) | 1000 ms (38.6M contiguous) | 15 ms | 1 call Copy toàn bộ |
| CJ8 production-shape (inline `d[j]=opt.p[off+j]`) | 875 ms | — | |
| **probe `_gl1` [2]** — 148 tensor thật, 124,439,808 ptu | **4453 ms** | **94 ms (47×, sample_bad=0)** | slice `f32buf_wrap(buf+off*4)` + `f32_copy_from_buf`, 148 call |

**Đính chính:** số 844 ms **thấp hơn thực tế** (875–4453 ms tùy JIT-shape — probe
trong `main()` to bằng ~35 ns/ptu, cj8 ~7 ns/ptu), và phía native **94 ms cho đúng
hình dạng 148-slice** (không phải 15 ms contiguous-call như bench_dp). **Hướng kết
luận đảo ngược đơn #3 §5 của repo trước đó** (repo từng chốt «A/B = 0% → drop»).

### B2. Vì sao A/B bench của repo ra 0% — lỗi đo của repo (thừa nhận tại đây)

Bench `simd_bench_124m_seq` dùng **bản copy inline** `_step124` (`d[j] = opt.p[off+j]`
ngay trong bench) — **không gọi `gpt2_unflatten`** → A/B pb3 vs pb4 (wire
`gpt2_unflatten` buffer-slice) gần như 2 binary giống nhau → «0%» là kết quả tất
nhiên, không nói lên gì. HANDOFF §0.53 sẽ được đính chính theo. Repo sẽ sửa bench
gọi đúng đường production rồi đo lại trước khi chốt giữ/drop wire unflatten.

### B3. Ask

1. Ghi nhận số thật cho shape production: **148-slice → 94 ms native** (probe `_gl1`,
   source kèm repro); nếu team muốn, thêm case 148-slice vào `bench_dp` để đối chiếu.
2. Số «844 ms» trong #3 §2.2 thay bằng khoảng **875–4453 ms** (JIT-shape nhạy) —
   ghi chú này chủ yếu để hai bên cùng dùng một con số trong budget Phase C-F.

---

## Ghi chú quy trình

- Repro probe `_gl1.tkv`: 0 thay đổi code train, compile được ngay bằng tkvc hiện tại
  (3 lần sửa lỗi type-inference nhỏ trong lúc viết: seed `list[Tensor]`, bind
  `ts[i].data` ra local, thứ tự `f32_copy_to_buf(src, p, n)` — đáng ghi REFERENCE).
- Gate repo không đổi ở lần này (probe-only + viết đơn) — chưa cần chạy lại 135/135.
- Số liệu đầy đủ: `out/gl1.log`, `out/ab_pb_s1024.log`, HANDOFF §0.53.