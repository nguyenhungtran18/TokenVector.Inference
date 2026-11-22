# Ke hoach viet lai GPU bridge (torch_fused.cpp) bang TokenVector

## 1. Nguyen ly chot (da doc tu torch_fused.cpp 818-1175)

### Du lieu
- `g_p_all`: master **fp32** [N] tren CUDA.
- `g_p[i]`: **fp16** view cua `g_p_all.narrow(off_i, numel_i)` â€” de fwd/bwd nhanh.
- `g_m_dev`, `g_v_dev`: state AdamW fp32 [N]. `g_grad_dev`: grad flat fp32 [N].
- params order: `[wte, wpe, ln_f_g, ln_f_b] + L * [ln1g, ln1b, wqkv, bqkv, wp, bp, ln2g, ln2b, w1, b1, w2, b2]`.

### Forward (fp16, eps=1e-6, inv = 1/sqrt(hd))
```
x = index_select(wte, ids) + index_select(wpe, pos)          # fp16
for li in 0..L-1:
    a   = LayerNorm(x, ln1g, ln1b, eps) -> fp16
    qkv = addmm(bqkv, a, wqkv)                               # fp16
    q,k,v = view(qkv, [T,3,nh,hd]) -> [1,nh,T,hd]
    attn = SDPA(q,k,v, causal=true)                          # fp16
    ctx  = attn.transpose(1,2).reshape(T,h)
    x    = x + addmm(bp, ctx, wp)
    m    = LayerNorm(x, ln2g, ln2b, eps) -> fp16
    u    = addmm(b1, m, w1)
    g1   = gelu(u)
    x    = x + addmm(b2, g1, w2)
xf    = LayerNorm(x, ln_f_g, ln_f_b, eps) -> fp16
logits= mm(xf, wte.T) -> fp32
loss  = -mean(log_softmax(logits).gather(target))
```

### Backward
- `loss.backward()` (autograd) -> grad trong `g_p[i].grad`.
- `put(idx, grad)`: copy grad fp32 vao `g_grad_dev.narrow(off_idx, numel)` theo layout flat.
- AdamW kernel **da viet bang .tkv** (`adamw.tkv`, goi qua dlsym):
  `m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2; p -= lr*(m*bc2s)/(sqrt(v)+eps*bc2s) - lr*wd*p`
- copy master fp32 -> fp16 views `g_p[i]` (no_grad) de step sau dung weights moi.

### Dieu kien hieu nang (2.24x hien tai)
- Khong D2H: p/m/v/grad de tren VRAM.
- Giust 148 param copy_ tai moi step (da thu gop bang cat/foreach -> THAT BAI, xem note M4/M4b).
- SDPA dung ATen kernel (same as HF).

## 2. Anh xa sang TokenVector

| Thanh phan | Cach lam TokenVector |
|---|---|
| AdamW | DA CO: adamw.tkv -> cuda_feature.tkv -> adamw.cuda.cu |
| LN fwd/bwd | Da co dang CPU; can ban CUDA (at::layer_norm / cong thuc mean-var-rstd nhu cpu_feature) |
| GEMM/addmm fp16 | CUDA backend `half_mm2`/`half_mat2` -> builtin `torch_mm` / `torch_addmm` |
| SDPA | builtin `torch_sdpa` (goi at::scaled_dot_product_attention) |
| gelu / gelu_backward | builtin `torch_gelu` / `torch_gelu_bwd` |
| log_softmax / gather / CE | builtin `torch_ce` |
| index_select embedding | builtin `torch_embed` |
| view / transpose / narrow | builtin `torch_view` / `torch_transpose` / `torch_narrow` |
| orchestration forward+backward+put+adam | file `.tkv` moi, backend `torch_feature.tkv` sinh C++ |

## 3. Cac phase

1. `torch_feature.tkv`: backend sinh C++/ATen (include torch, EXPORT, body tu DSL). Marker `# @tkv.torch_kernel`.
2. Port cac export don gian: `TkvFullPing`, `TkvFullAdamwPath`, `Begin/SetTensor/SetIds/SetTargets/SetGrad/SetOpt/GetParams`.
3. Port `TkvFullStep124m`: forward (phase 2) -> backward (autograd) -> put -> AdamW -> copy fp16.
4. Sinh `torch_fused_generated.cpp`, build bang nvcc, thay the `torch_fused.cpp`.
5. Verify: loss 3.8906 -> 3.7419 (400 steps) va speedup >= 2.2x.

## 4. Nguyen tac
- Khong viet hand C++: mo moi thanh phan la `.tkv`, compiler sinh C++.
- Giust nguyen so do, thu tu va dtype nhu tren (fp16 activations, fp32 master).
- Khong thay SDPA bang scalar (se mat toc do).