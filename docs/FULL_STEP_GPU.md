# GPU-resident full training step — design (Nguyen Hung Tran, 2026-11-17/18)

Mục tiêu (user yêu cầu triển khai thẳng): `TKV_FULL_GPU=1` → 1 native call/step làm
Toàn bộ train step trên CUDA (fwd + loss + bwd + AdamW), thay cho hàng trăm op qua interpreter.
Fallback vẫn nguyên vẹn khi flag off hoặc bridge rc != 0.

## 1. Đo đạc hiện trạng (почему per-op là nút cổ chai)

GPU bridge (Colab T4, (28)): TONG 3 849 ms (fwd 796 / bwd 1 582) vs torch CUDA ~270 ms.
Op-time GPU ~2 712 ms; mỗi op đi qua .NET pinvoke → shim → ATen → H2D/D2H pageable
→ không giữ gì trên GPU giữa các op. Băng thông host+interpreter là nút cổ chai.

## 2. Thiết kế: một entry point `TkvFullStep124m` (bridge ATen, fp32)

```
TkvFullBegin(int64_t n_tensors, int64_t NLayers)      ->  int64_t rc   (reset registry, CUDA ctx warm)
TkvFullSetTensor(int64_t idx, const float* p, int64_t numel, int64_t role) -> int64_t
     role: 0 = weight (leaf), 1 = opt param-slot (ps order == g layout == p layout)
TkvFullSetIds(const int* ids, int64_t T, int64_t nvec, int64_t pos_mode)
     pos_mode: 0 = pos = range(T); 1 = truyền pos riêng
TkvFullSetPos(const int* pos, int64_t T)              (optional)
TkvFullSetOpt(const float* p, const float* m, const float* v,
              float lr, float b1, float b2, float eps, float wd,
              int64_t total_steps, int64_t cur_step)  -> int64_t
TkvFullRun(int64_t mode)  /* 0 = fwd+bwd+adam (full step); 1 = fwd only (spike verify) */
   -> float loss   /* fwd/loss in [0]; backward inside [0]; p/m/v cập nhật tại chỗ */
```

- Registry lưu CUDA Tensor (at::empty + copy_ 1 lần/step); gọi xong `p` pointer vào CUDA buffer —
  không cần pageable giữa chừng.
- AdamW thay .tkv `opt.step()`: flat grads trong p-order (ps order)
  = [wte, wpe, lnf_g, lnf_b] + 12×[ln1_g, ln1_b, wqkv, bqkv, wp, bp, ln2_g, ln2_b, w1, b1, w2, b2].
  Clip/nhóm chia chỉ dùng 1 group (beta1/beta2/eps/wd từ _mk_opt).
- Numerics: waiver như mọi bridge (loss_val ~17.111x, native EXACT không đổi).

## 3. Các pha trong TkvFullRun (all ATen on CUDA, fp32)

```
fwd:
  x = wte[ids] + wpe[pos]                 /* index_select + add */
  for li in 0..11:
      a  = LN(x, ln1g, ln1b, 1e-6)         /* manual LN fwd */
      qkv = a @ wqkv + bqkv                /* addmm [T,h]x[h,3h] */
      q,k,v = split ; softmax(q·k^T·inv)causal; ctx = P@v  /* SDPA flash fp32 */
      x  = x + ctx @ wp + bp               /* residual */
      m   = LN(x, ln2g, ln2b)
      u   = m @ w1 + b1                    /* [T,4h] */
      g1  = erf_gelu(u)
      h2  = g1 @ w2 + b2
      x   = x + h2                         /* residual */
  xf = LN(x, lnfg, lnfb)
  logits = xf @ wte^T                      /* [T,V] — tied */
loss = mean_t( logsumexp(logits) - logits[t, y_t] )
```

```
bwd (tắt autograd, công thức tay như mọi bridge path khác):
  dlogits = softmax(logits);  dlogits[ids] -= 1/T            /* mean-CE */
  dxf  = dlogits @ wte          /* [T,h] */
  dwte = dlogits^T @ xf         /* [V,h] */
  dx   = LN_bwd(dxf, x, lnfg, lnfb)
  for li in 11..0:
      dh2 = dx ; dh_ink2 = dh2
      dg1 = dh2 @ w2^T                       /* dx @ w2^T? careful dims */
      dw2 = g1^T @ dh2 ; db2 = sum(dh2, 0)
      du  = dg1 * erf_gelu'(u)               /* u saved */
      dm  = du @ w1^T ; dw1 = m^T @ du ; db1 = sum(du, 0)
      dxm = LN_bwd(dm, x_res1, ln2g, ln2b)
      dx  = dxm + dh_ink2
      d_ctx  = dx @ wp^T    /* attention proj layer */   /* naming a = attn proc */
      dwproj = ctx^T @ dx   /* ctx was saved */
      dbp    = sum(dx, 0)
      dP = d_ctx @ v^T ; dS = inv * P * (dP - sum(dP*P, -1))
      dq = dS @ k ; dk = dS^T @ q ; dv = P^T @ d_ctx
      dqkv = cat(dq, dk, dv, 1)             /* [T,3h] */
      d_a = dqkv @ wqkv^T
      dwqkv = a^T @ dqkv ; dbqkv = sum(dqkv, 0)
      dxa = LN_bwd(d_a, x_res0, ln1g, ln1b)
      dx  = dxa + dx                        /* residual */
  dx  = d(wte embedding gather)
  d_wte = scatter_add(d_emb, ids) + dwte    /* tied */
  d_wpe = scatter_add(d_emb, pos)
grads_flat[i] = concat(dwte, dwpe, dlnfg, dlnfb, per-layer 12)
```

## 4. Milestones (từng phiên 1 verification)

- **M1 (this)**: skeleton + fwd + loss; ctypes test cell in notebook → loss ≈ ~17.11x.
- **M2**: bwd inside Run; compare grads to native fallback (spot-check wqkv/wp/+ emb wte)
  via scalar dump into a small .tkv or python side tensor print.
- **M3**: AdamW step inside Run; .tkv branch `if TKV_FULL_GPU==1: TkvFullStep(...)` replacing
  the whole `_step124` body; fallback `else` keeps current path.
- **M4**: full bench verify TONG; record `BENCHMARKS.md`.

## 5. Rủi ro / quyết định

- `.tkv`→native does NOT take list[Tensor] — registry Begin/SetTensor thay list.
- Numerics waiver (loss ≠ native EXACT) — user chấp nhận (LN/LM-head/attention đều waiver).
- Giữ native EXACT path untouched: `TKV_FULL_GPU=1` gate; fallback khi rc != 0.
- AdamW clip/beta chuẩn: single group (như `_mk_opt`).
- `pos == ids` only if pos = range(T) (bench hiện tại); hỗ trợ pos_mode riêng nếu cần.
